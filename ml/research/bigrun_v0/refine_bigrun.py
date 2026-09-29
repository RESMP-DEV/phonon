"""bigrun_v0 eval step 2: one adapter over the three term-clip sets and the real-audio holdout.

Term sets: heldout_old seen voices, heldout_old unseen voices, heldout_new (new terms AND new
voices), each in none / oracle / retrieved. Real audio: the 580-row date-split holdout in
none / retrieved. Per-row outputs land in /data/phonon_bigrun_v0/refined/.

Adapters are named on the command line as `label:base:dir` (base and dir default to
LFM2.5-1.2B-Instruct and the label), so nothing is hardcoded and the metrics are untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf")
os.environ.setdefault("HF_HUB_CACHE", "/data/hf/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf/hub")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/home/user/phonon/research/corrector_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")

from vocab_common import read_jsonl  # noqa: E402

ADAPTERS = Path("/data/phonon_corrector_v0/adapters")
OUT = Path("/data/phonon_bigrun_v0/refined")
LFM12 = "LiquidAI/LFM2.5-1.2B-Instruct"
LFM350 = "LiquidAI/LFM2.5-350M"
BASE_ALIAS = {"lfm12": LFM12, "lfm350": LFM350, "": LFM12}

TERM_SETS = {
    "seen": Path("/data/phonon_vocab_v1/eval_conditions.jsonl"),
    "unseen": Path("/data/phonon_term_eval_v1/eval_conditions.jsonl"),
    "new": Path("/data/phonon_bigrun_v0/eval_conditions_new.jsonl"),
}
REAL = Path("/data/phonon_term_eval_v1/real/eval_holdout_date_audio.jsonl")
TERM_CONDS = ("none", "oracle", "retrieved")
REAL_CONDS = ("none", "retrieved")


# ---- decode-time LENGTH GUARD ------------------------------------------------
# cap_i = 1.5 x input tokens + 32, never above the set's max_new_tokens. Greedy decoding is
# independent per row, so generating the batch at max(cap) and slicing row i at cap_i is
# identical to having generated row i with max_new_tokens=cap_i. A word-count post-check then
# falls back to the raw input when the output runs away or collapses.
GUARD_TOK_MULT = 1.5
GUARD_TOK_ADD = 32
GUARD_MAX_RATIO = 2.0
GUARD_MIN_RATIO = 0.4
GUARD_MIN_WORDS = 8


def guard_cap(tokenizer, text: str, hard_cap: int) -> int:
    n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return max(8, min(hard_cap, int(GUARD_TOK_MULT * n) + GUARD_TOK_ADD))


def guard_post(inp: str, out: str) -> tuple[str, bool]:
    """Fall back to the raw input on a runaway or a collapse."""
    iw = len(inp.split())
    ow = len(out.split())
    if iw == 0:
        return out, False
    if ow > GUARD_MAX_RATIO * iw:
        return inp, True
    if iw > GUARD_MIN_WORDS and ow < GUARD_MIN_RATIO * iw:
        return inp, True
    return out, False


def generate_capped(model, tokenizer, prompts, caps):
    import torch

    from eval_corrector import strip_thinking

    prev = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = prev
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        if hasattr(model.generation_config, "temperature"):
            model.generation_config.temperature = None
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max(caps), do_sample=False,
                             pad_token_id=pad_id, eos_token_id=eos_id)
    texts, capped = [], []
    for i, c in enumerate(caps):
        seq = out[i, input_len:input_len + c]
        texts.append(strip_thinking(tokenizer.decode(seq, skip_special_tokens=True)))
        if eos_id is None:
            capped.append(int(seq.shape[0] >= c))
        else:
            capped.append(int(not bool((seq == eos_id).any().item())))
    return texts, capped


def run_jobs_guard(model, processor, tokenizer, jobs, batch_size, max_new_tokens, label):
    """run_jobs with the length guard; returns (outs, fired, capped)."""
    import time as _time

    from refine_vocab1 import apply_prompt_vocab

    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i]["text"]) + 4 * len(jobs[i]["vocab"]))
    outs = [""] * len(jobs)
    fired = [0] * len(jobs)
    capped = [0] * len(jobs)
    t0 = _time.perf_counter()
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        prompts = [apply_prompt_vocab(processor, jobs[i]["text"], jobs[i]["vocab"]) for i in idx]
        caps = [guard_cap(tokenizer, jobs[i]["text"], max_new_tokens) for i in idx]
        res, cap_hit = generate_capped(model, tokenizer, prompts, caps)
        for i, h, ch in zip(idx, res, cap_hit):
            txt, fb = guard_post(jobs[i]["text"], h)
            outs[i] = txt
            fired[i] = int(fb)
            capped[i] = int(ch)
        done = min(start + batch_size, len(order))
        if (start // batch_size) % 10 == 0 or done == len(order):
            el = _time.perf_counter() - t0
            print(f"  {label} {done}/{len(order)} {done/max(el,1e-9):.1f} rows/s "
                  f"fired={sum(fired)} capped={sum(capped)}", flush=True)
    return outs, fired, capped


def parse_adapter(spec: str) -> tuple[str, str, Path]:
    parts = spec.split(":")
    label = parts[0]
    base = parts[1] if len(parts) > 1 and parts[1] else ""
    d = parts[2] if len(parts) > 2 and parts[2] else label
    return label, BASE_ALIAS.get(base, base or LFM12), ADAPTERS / d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", default=[],
                    help="label[:base[:dir]]; repeatable")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--real-batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--real-max-new-tokens", type=int, default=2048)
    ap.add_argument("--sets", default="seen,unseen,new,real")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--guard", action="store_true",
                    help="decode-time length guard: per-row max_new_tokens and a word post-check")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    import torch

    from eval_corrector import load_stack
    from refine_vocab1 import run_jobs as run_jobs_plain

    out_root = args.out_dir

    def run_jobs(model, processor, tokenizer, jobs, bs, mnt, label):
        if args.guard:
            return run_jobs_guard(model, processor, tokenizer, jobs, bs, mnt, label)
        return run_jobs_plain(model, processor, tokenizer, jobs, bs, mnt, label), None, None

    want = {s for s in args.sets.split(",") if s}
    term_rows = {k: read_jsonl(p) for k, p in TERM_SETS.items() if k in want}
    real_rows = read_jsonl(REAL) if "real" in want else []
    print(f"sets={ {k: len(v) for k, v in term_rows.items()} } real={len(real_rows)}", flush=True)
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for spec in args.adapter:
        label, base, adapter = parse_adapter(spec)
        if not (adapter / "adapter_config.json").exists():
            print(f"MISSING adapter {adapter}; skipping {label}", flush=True)
            continue
        paths = {k: out_root / f"{label}__{k}.jsonl" for k in list(term_rows) + (
            ["real"] if real_rows else [])}
        if paths and all(p.exists() for p in paths.values()) and not args.force:
            print(f"skip {label} (complete)", flush=True)
            continue
        print(f"=== {label} base={base} adapter={adapter}", flush=True)
        t0 = time.perf_counter()
        model, processor, tokenizer = load_stack(base, adapter)
        try:
            for key, rows in term_rows.items():
                if paths[key].exists() and not args.force:
                    print(f"  skip {label}/{key}", flush=True)
                    continue
                res, gfired, gcapped = {}, {}, {}
                for cond in TERM_CONDS:
                    vocab_key = {"none": None, "oracle": "vocab_oracle",
                                 "retrieved": "vocab_retrieved"}[cond]
                    jobs = [{"text": r["hyp"],
                             "vocab": [] if vocab_key is None else r[vocab_key]} for r in rows]
                    res[cond], gfired[cond], gcapped[cond] = run_jobs(
                        model, processor, tokenizer, jobs, args.batch_size,
                        args.max_new_tokens, f"{label}/{key}/{cond}")
                tmp = paths[key].with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as h:
                    for i, r in enumerate(rows):
                        rec = {"id": r["id"], "term": r["term"], "kind": r["kind"],
                               "voice": r["voice"], "reference": r["reference"], "hyp": r["hyp"],
                               "retrieved_has_term": r["retrieved_has_term"],
                               "vocab_oracle": r["vocab_oracle"],
                               "vocab_retrieved": r["vocab_retrieved"]}
                        for cond in TERM_CONDS:
                            rec[f"out_{cond}"] = res[cond][i]
                            if gfired[cond] is not None:
                                rec[f"fired_{cond}"] = gfired[cond][i]
                                rec[f"capped_{cond}"] = gcapped[cond][i]
                        h.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tmp.rename(paths[key])
                print(f"  wrote {paths[key].name} [{time.perf_counter()-t0:.0f}s]", flush=True)
            if real_rows and (not paths["real"].exists() or args.force):
                res, gfired, gcapped = {}, {}, {}
                for cond in REAL_CONDS:
                    jobs = [{"text": r["input"],
                             "vocab": r["vocab_retrieved"] if cond == "retrieved" else []}
                            for r in real_rows]
                    res[cond], gfired[cond], gcapped[cond] = run_jobs(
                        model, processor, tokenizer, jobs,
                        args.real_batch_size, args.real_max_new_tokens,
                        f"{label}/real/{cond}")
                tmp = paths["real"].with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as h:
                    for i, r in enumerate(real_rows):
                        rec = {"id": r["id"], "input": r["input"], "reference": r["reference"],
                               "true_terms": r["true_terms"],
                               "vocab_retrieved": r["vocab_retrieved"]}
                        for cond in REAL_CONDS:
                            rec[f"out_{cond}"] = res[cond][i]
                            if gfired[cond] is not None:
                                rec[f"fired_{cond}"] = gfired[cond][i]
                                rec[f"capped_{cond}"] = gcapped[cond][i]
                        h.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tmp.rename(paths["real"])
                print(f"  wrote {paths['real'].name} [{time.perf_counter()-t0:.0f}s]", flush=True)
        finally:
            del model
            torch.cuda.empty_cache()
        print(f"{label} done wall_s={time.perf_counter()-t0:.0f}", flush=True)
    print("REFINE BIGRUN DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
