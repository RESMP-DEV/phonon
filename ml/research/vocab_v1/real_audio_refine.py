"""Step 3: vocab_v1 refiner on the honest date-split real-audio holdout.

Builds the retrieved vocabulary list from each raw Parakeet input, runs the requested
adapters in none/retrieved, writes per-row outputs to /data/phonon_term_eval_v1/real/.
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

from vocab_common import Lexicon, read_jsonl, seeded  # noqa: E402

HOLDOUT = Path("/data/phonon_corrector_v0/datesplit/holdout_date_audio.jsonl")
OUT = Path("/data/phonon_term_eval_v1/real")
ADAPTERS = Path("/data/phonon_corrector_v0/adapters")
LFM12 = "LiquidAI/LFM2.5-1.2B-Instruct"
TOPK = 30

# label : (base, adapter dir, conditions)
SYSTEMS = {
    "vocab1_real_acoustic": (LFM12, "vocab1_real_acoustic", ("none", "retrieved")),
    "date_lfm12_full_e2": (LFM12, "date_lfm12_full_e2", ("none",)),
    "vocab1_date": (LFM12, "vocab1_date", ("none", "retrieved")),
}


def build_lists() -> list[dict]:
    p = OUT / "eval_holdout_date_audio.jsonl"
    if p.exists():
        return read_jsonl(p)
    from retrieve import load_pool
    rows = read_jsonl(HOLDOUT)
    lex = Lexicon()
    R = load_pool()
    print(f"rows={len(rows)} pool={len(R.terms)}", flush=True)
    lists = R.topk([r["input"] for r in rows], k=TOPK, log=lambda s: print(s, flush=True))
    out = []
    for r, L in zip(rows, lists):
        terms = [t for t, _ in L]
        rng = seeded("v1realeval", "holdout_date_audio", r["id"])
        rng.shuffle(terms)
        out.append({"id": r["id"], "input": r["input"], "reference": r["reference"],
                    "true_terms": lex.find_terms(r["reference"]), "vocab_retrieved": terms})
    OUT.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as h:
        for r in out:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    cov = sum(sum(1 for t in r["true_terms"] if t in r["vocab_retrieved"]) for r in out)
    tot = sum(len(r["true_terms"]) for r in out)
    print(f"ref-term recall@30 = {cov/max(tot,1):.3f} ({cov}/{tot})", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--lists-only", action="store_true")
    args = ap.parse_args()

    rows = build_lists()
    if args.lists_only:
        return 0

    import torch
    from refine_vocab1 import run_jobs
    from eval_corrector import load_stack

    only = {s for s in args.only.split(",") if s}
    OUT.mkdir(parents=True, exist_ok=True)
    for label, (base, adapter_name, conds) in SYSTEMS.items():
        if only and label not in only:
            continue
        adapter = ADAPTERS / adapter_name
        if not (adapter / "adapter_config.json").exists():
            print(f"MISSING adapter {adapter}; skipping {label}", flush=True)
            continue
        dst = OUT / f"{label}.jsonl"
        if dst.exists():
            print(f"skip {label} (done)", flush=True)
            continue
        print(f"=== {label} adapter={adapter}", flush=True)
        t0 = time.perf_counter()
        model, processor, tokenizer = load_stack(base, adapter)
        try:
            res = {}
            for cond in conds:
                jobs = [{"text": r["input"],
                         "vocab": r["vocab_retrieved"] if cond == "retrieved" else []}
                        for r in rows]
                res[cond] = run_jobs(model, processor, tokenizer, jobs, args.batch_size,
                                     args.max_new_tokens, f"{label}/{cond}")
            with dst.open("w", encoding="utf-8") as h:
                for i, r in enumerate(rows):
                    rec = {"id": r["id"], "input": r["input"], "reference": r["reference"],
                           "true_terms": r["true_terms"], "vocab_retrieved": r["vocab_retrieved"]}
                    for cond in conds:
                        rec[f"out_{cond}"] = res[cond][i]
                    h.write(json.dumps(rec, ensure_ascii=False) + "\n")
        finally:
            del model
            torch.cuda.empty_cache()
        print(f"{label} done wall_s={time.perf_counter()-t0:.0f}", flush=True)
    print("REAL REFINE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
