"""Run each refiner adapter over the term-eval ASR hypotheses. GPU 1."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf")
os.environ.setdefault("HF_HUB_CACHE", "/data/hf/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf/hub")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/home/user/phonon/research/corrector_v0")

from eval_corrector import generate_batch, load_stack  # noqa: E402

ADAPTERS = Path("/data/phonon_corrector_v0/adapters")
SYSTEMS = [
    ("lfm2.5-1.2b", "LiquidAI/LFM2.5-1.2B-Instruct", "lfm2.5-1.2b"),
    ("ref1_lfm2.5-1.2b_synthreal", "LiquidAI/LFM2.5-1.2B-Instruct", "ref1_lfm2.5-1.2b_synthreal"),
    ("lfm2.5-350m", "LiquidAI/LFM2.5-350M", "lfm2.5-350m"),
    ("ref_lfm2.5-350m_synthreal", "LiquidAI/LFM2.5-350M", "ref_lfm2.5-350m_synthreal"),
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B", "qwen3-0.6b"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", type=Path, default=Path("/data/phonon_term_eval_v0/asr.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/data/phonon_term_eval_v0/refined"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.asr.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"rows={len(rows)}", flush=True)

    import torch
    only = {s for s in args.only.split(",") if s}
    for label, base, adapter_name in SYSTEMS:
        if only and label not in only:
            continue
        out_path = args.out_dir / f"{label}.jsonl"
        if out_path.exists() and sum(1 for l in out_path.read_text().splitlines() if l.strip()) == len(rows):
            print(f"skip {label} (already complete)", flush=True)
            continue
        adapter = ADAPTERS / adapter_name
        print(f"=== {label} base={base} adapter={adapter}", flush=True)
        t0 = time.perf_counter()
        model, processor, tokenizer = load_stack(base, adapter)
        try:
            order = sorted(range(len(rows)), key=lambda i: len(rows[i]["hyp"]))
            outs = [""] * len(rows)
            for start in range(0, len(order), args.batch_size):
                idx = order[start : start + args.batch_size]
                texts = [rows[i]["hyp"] for i in idx]
                res = generate_batch(model, processor, tokenizer, texts, args.max_new_tokens)
                for i, h in zip(idx, res):
                    outs[i] = h
                if (start // args.batch_size) % 5 == 0 or start + args.batch_size >= len(order):
                    el = time.perf_counter() - t0
                    done = min(start + args.batch_size, len(order))
                    print(f"{label} {done}/{len(order)} {done/max(el,1e-9):.2f} rows/s "
                          f"elapsed_min={el/60:.1f}", flush=True)
        finally:
            del model
            torch.cuda.empty_cache()
        with out_path.open("w", encoding="utf-8") as h:
            for row, out in zip(rows, outs):
                h.write(json.dumps({"id": row["id"], "term": row["term"], "voice": row["voice"],
                                    "reference": row["reference"], "hyp": row["hyp"],
                                    "output": out}, ensure_ascii=False) + "\n")
        print(f"{label} done wall_s={time.perf_counter()-t0:.1f} -> {out_path}", flush=True)
    print("REFINE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
