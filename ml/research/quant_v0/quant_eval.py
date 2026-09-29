"""Quantization sweep for the merged LFM2.5-1.2B refiner.

Backends: mlx (quantize with mlx_lm.convert, generate + teacher-forced NLL in MLX) and openai (an
OpenAI-compatible server such as llama-server, generation only). Metrics per config and eval set:
fair/strict WER, worse-than-input fraction, exact-match and word agreement against the bf16 reference
outputs (HF bf16 predictions from eval_corrector, and the backend's own bf16/unquantized run),
mean NLL per target token (mlx only), decode tokens/s, size on disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "corrector_v0"))
from common import chat_messages, fair_norm, pair_wer, score_lists  # noqa: E402
from eval_corrector import SET_ORDER, eval_rows  # noqa: E402

DATA = Path(os.environ.get("PHONON_DATA_ROOT", "/data"))
PRED_DIR = DATA / "phonon_corrector_v0/eval_predictions"


def corpus_wer(refs, hyps, norm):
    import jiwer
    r = [norm(x) or "<empty>" for x in refs]
    h = [norm(x) for x in hyps]
    return float(jiwer.wer(r, h))


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# ---------------------------------------------------------------- configs
def parse_config(name: str) -> dict:
    """bf16 | q{bits}_g{group} | q{bits}_g{group}_{mode} | mixed_4_6 | mxfp4 | nvfp4"""
    if name == "bf16":
        return {"quantize": False}
    if name.startswith("mixed_"):
        return {"quantize": True, "quant_predicate": name, "q_bits": 4, "q_group_size": 64}
    if name.startswith("custom_"):
        # custom_emb8      : 4-bit g64 everywhere, embeddings (tied lm_head) 8-bit
        # custom_attn6emb8 : as above plus 6-bit on the attention projections
        # custom_edge6emb8 : as emb8 plus 6-bit on the first and last two decoder layers
        kind = name[len("custom_"):]

        def pred(path: str, module, config: dict | None = None):
            if path.endswith("embed_tokens") or path.endswith("lm_head"):
                return {"bits": 8, "group_size": 64}
            if kind == "attn6emb8" and ".self_attn." in path:
                return {"bits": 6, "group_size": 64}
            if kind == "edge6emb8":
                import re
                m = re.search(r"layers\.(\d+)\.", path)
                if m and int(m.group(1)) in (0, 1, 14, 15):
                    return {"bits": 6, "group_size": 64}
            return {"bits": 4, "group_size": 64}

        return {"quantize": True, "quant_predicate": pred, "q_bits": 4, "q_group_size": 64}
    if name in ("mxfp4", "nvfp4", "mxfp8"):
        return {"quantize": True, "q_mode": name, "q_bits": 4 if name != "mxfp8" else 8, "q_group_size": 32 if name != "nvfp4" else 16}
    parts = name.split("_")
    cfg = {"quantize": True, "q_bits": int(parts[0][1:]), "q_group_size": int(parts[1][1:])}
    if len(parts) > 2:
        cfg["q_mode"] = parts[2]
    return cfg


# ---------------------------------------------------------------- mlx backend
class MlxBackend:
    def __init__(self, merged: Path, out_dir: Path):
        self.merged, self.out_dir = merged, out_dir

    def prepare(self, name: str) -> Path:
        path = self.out_dir / name
        if not (path / "config.json").exists():
            from mlx_lm import convert
            cfg = parse_config(name)
            t0 = time.time()
            convert(str(self.merged), str(path), **cfg)
            print(f"converted {name} in {time.time()-t0:.0f}s", flush=True)
        return path

    def load(self, path: Path):
        from mlx_lm import load
        self.model, self.tok = load(str(path))

    def generate(self, texts: list[str], max_tokens: int, batch_size: int) -> tuple[list[str], float]:
        from mlx_lm.generate import batch_generate
        outs, gen_tokens, gen_time = [], 0, 0.0
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            prompts = [self.tok.apply_chat_template(chat_messages(t), add_generation_prompt=True, tokenize=True) for t in chunk]
            t0 = time.time()
            resp = batch_generate(self.model, self.tok, prompts, max_tokens=max_tokens, verbose=False)
            gen_time += time.time() - t0
            gen_tokens += getattr(resp.stats, "generation_tokens", 0) if hasattr(resp, "stats") else 0
            outs.extend(resp.texts)
        return outs, (gen_tokens / gen_time if gen_time else 0.0)

    def nll(self, rows: list[dict]) -> tuple[float, int]:
        import mlx.core as mx
        total, count = 0.0, 0
        for row in rows:
            prompt = self.tok.apply_chat_template(chat_messages(row["input"]), add_generation_prompt=True, tokenize=True)
            full = self.tok.apply_chat_template(chat_messages(row["input"], row["reference"]), add_generation_prompt=False, tokenize=True)
            if full[: len(prompt)] != prompt:
                continue
            ids = mx.array(full)[None]
            logits = self.model(ids)[0, :-1].astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tgt = mx.array(full[1:])
            tok_lp = mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0]
            span = tok_lp[len(prompt) - 1:]
            mx.eval(span)
            total -= float(span.sum().item())
            count += int(span.shape[0])
        return (total / count if count else float("nan")), count


# ---------------------------------------------------------------- openai backend
class OpenAIBackend:
    def __init__(self, base_url: str, model: str = "default", size_path: Path | None = None):
        self.base_url, self.model_name, self.size_path = base_url.rstrip("/"), model, size_path

    def prepare(self, name: str) -> Path:
        return self.size_path or Path(".")

    def load(self, path: Path):
        pass

    def generate(self, texts: list[str], max_tokens: int, batch_size: int) -> tuple[list[str], float]:
        import concurrent.futures as cf
        import urllib.request

        def one(text: str) -> tuple[str, int]:
            body = json.dumps({"model": self.model_name, "messages": chat_messages(text), "temperature": 0.0,
                               "max_tokens": max_tokens, "seed": 0}).encode()
            req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"], d.get("usage", {}).get("completion_tokens", 0)

        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=batch_size) as ex:
            res = list(ex.map(one, texts))
        dt = time.time() - t0
        return [r[0] for r in res], (sum(r[1] for r in res) / dt if dt else 0.0)

    def nll(self, rows):
        return float("nan"), 0


# ---------------------------------------------------------------- main
def load_ref_preds(label: str, set_name: str) -> dict[str, str]:
    path = PRED_DIR / f"{label}_{set_name}.jsonl"
    if not path.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l)["output"] for l in path.read_text().splitlines() if l.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx", "openai"], default="mlx")
    ap.add_argument("--merged", default=str(DATA / "phonon_quant_v0/merged_lfm2.5-1.2b_real"))
    ap.add_argument("--out-dir", default=str(DATA / "phonon_quant_v0/mlx"))
    ap.add_argument("--configs", default="bf16,q8_g64,q4_g32,q4_g64,q4_g128,q6_g64,q5_g64,q3_g64,mixed_4_6,mixed_3_6,mxfp4")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--config-name", default="openai", help="label for the openai backend run")
    ap.add_argument("--size-path", default=None, help="file/dir whose size is reported for the openai backend")
    ap.add_argument("--sets", default=",".join(SET_ORDER))
    ap.add_argument("--ref-label", default="lfm2.5-1.2b_real", help="HF bf16 predictions label in eval_predictions")
    ap.add_argument("--own-ref", default="bf16", help="config whose outputs are the backend-side reference")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--pred-dir", default=str(DATA / "phonon_quant_v0/predictions"))
    args = ap.parse_args()

    sets = {k: v for k, v in eval_rows().items() if k in args.sets.split(",")}
    if args.limit:
        sets = {k: v[: args.limit] for k, v in sets.items()}
    pred_dir = Path(args.pred_dir); pred_dir.mkdir(parents=True, exist_ok=True)

    out_json = Path(args.out_json)
    payload = json.loads(out_json.read_text()) if out_json.exists() else {"runs": {}}
    if args.backend == "mlx":
        backend = MlxBackend(Path(args.merged), Path(args.out_dir))
        configs = args.configs.split(",")
    else:
        backend = OpenAIBackend(args.base_url, size_path=Path(args.size_path) if args.size_path else None)
        configs = [args.config_name]

    own_ref = {}
    for set_name in sets:
        p = pred_dir / f"{args.own_ref}_{set_name}.jsonl"
        if p.exists():
            own_ref[set_name] = {json.loads(l)["id"]: json.loads(l)["output"] for l in p.read_text().splitlines() if l.strip()}

    for name in configs:
        if name in payload["runs"] and all(s in payload["runs"][name] for s in sets):
            print(f"skip {name} (done)", flush=True); continue
        path = backend.prepare(name)
        backend.load(path)
        size = dir_size(path) if path.is_dir() else path.stat().st_size
        run = payload["runs"].setdefault(name, {})
        run["size_bytes"] = size
        for set_name, rows in sets.items():
            outs, tps = backend.generate([r["input"] for r in rows], args.max_tokens, args.batch_size)
            outs = [o.strip() for o in outs]
            refs = [r["reference"] for r in rows]
            sc = score_lists(refs, outs)
            worse = sum(1 for r, o in zip(rows, outs) if pair_wer(r["reference"], o, fair_norm) - pair_wer(r["reference"], r["input"], fair_norm) > 1e-9) / len(rows)
            hf_ref = load_ref_preds(args.ref_label, set_name)
            m = {"n": len(rows), "fair_wer": sc["fair_wer"], "strict_lc_wer": sc["strict_lc_wer"], "worse_frac": worse, "tok_s": tps}
            if hf_ref:
                pairs = [(hf_ref[r["id"]], o) for r, o in zip(rows, outs) if r["id"] in hf_ref]
                m["exact_vs_hf_bf16"] = sum(1 for a, b in pairs if fair_norm(a) == fair_norm(b)) / len(pairs)
                m["word_agree_vs_hf_bf16"] = 1.0 - corpus_wer([a for a, _ in pairs], [b for _, b in pairs], fair_norm)
            if set_name in own_ref and name != args.own_ref:
                pairs = [(own_ref[set_name][r["id"]], o) for r, o in zip(rows, outs) if r["id"] in own_ref[set_name]]
                m["exact_vs_own_bf16"] = sum(1 for a, b in pairs if fair_norm(a) == fair_norm(b)) / len(pairs)
                m["word_agree_vs_own_bf16"] = 1.0 - corpus_wer([a for a, _ in pairs], [b for _, b in pairs], fair_norm)
            nll, ntok = backend.nll(rows)
            m["nll_per_tok"] = nll; m["nll_tokens"] = ntok
            run[set_name] = m
            with (pred_dir / f"{name}_{set_name}.jsonl").open("w") as f:
                for r, o in zip(rows, outs):
                    f.write(json.dumps({"id": r["id"], "input": r["input"], "output": o, "reference": r["reference"]}) + "\n")
            if name == args.own_ref:
                own_ref[set_name] = {r["id"]: o for r, o in zip(rows, outs)}
            print(f"{name} {set_name} fair={m['fair_wer']:.4f} worse={worse:.3f} nll={nll:.4f} tok/s={tps:.0f}", flush=True)
        out_json.write_text(json.dumps(payload, indent=1))
        Path(args.out_md).write_text(build_md(payload, list(sets)))
    Path(args.out_md).write_text(build_md(payload, list(sets)))
    print("wrote", args.out_md)
    return 0


def f(x, d=4):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def build_md(payload: dict, sets: list[str]) -> str:
    lines = ["# Quantization sweep: merged LFM2.5-1.2B refiner (lfm2.5-1.2b_real)", "",
             "Fair WER = whisper_normalizer EnglishTextNormalizer then jiwer. Agreement columns compare the config's outputs with the bf16 outputs (HF transformers bf16 from eval_corrector, and the same backend unquantized). NLL is mean teacher-forced negative log-likelihood per target token (mlx backend only).", ""]
    for s in sets:
        lines += [f"## {s}", "", "| config | size MB | fair WER | strict lc WER | worse than input | exact vs HF bf16 | word agree vs HF bf16 | exact vs own bf16 | word agree vs own bf16 | NLL/tok | tok/s |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, run in payload["runs"].items():
            m = run.get(s)
            if not m:
                continue
            lines.append(f"| {name} | {run['size_bytes']/1e6:.0f} | {f(m['fair_wer'])} | {f(m['strict_lc_wer'])} | {f(m['worse_frac'],3)} | {f(m.get('exact_vs_hf_bf16'),3)} | {f(m.get('word_agree_vs_hf_bf16'))} | {f(m.get('exact_vs_own_bf16'),3)} | {f(m.get('word_agree_vs_own_bf16'))} | {f(m.get('nll_per_tok'))} | {f(m.get('tok_s'),0)} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
