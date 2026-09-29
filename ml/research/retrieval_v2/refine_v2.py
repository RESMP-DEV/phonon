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
OUT = Path("/data/phonon_retrieval_v2/refined")
LFM12 = "LiquidAI/LFM2.5-1.2B-Instruct"
LFM350 = "LiquidAI/LFM2.5-350M"
BASE_ALIAS = {"lfm12": LFM12, "lfm350": LFM350, "": LFM12}

TERM_SETS = {

    "unseen": Path("/data/phonon_retrieval_v2/cond_unseen_v2.jsonl"),
    "new": Path("/data/phonon_retrieval_v2/cond_new_v2.jsonl"),
}
REAL = Path("/data/phonon_term_eval_v1/real/eval_holdout_date_audio.jsonl")
TERM_CONDS = ("retrieved",)
REAL_CONDS = ("none", "retrieved")


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
    ap.add_argument("--sets", default="unseen,new")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch

    from eval_corrector import load_stack
    from refine_vocab1 import run_jobs

    want = {s for s in args.sets.split(",") if s}
    term_rows = {k: read_jsonl(p) for k, p in TERM_SETS.items() if k in want}
    real_rows = read_jsonl(REAL) if "real" in want else []
    print(f"sets={ {k: len(v) for k, v in term_rows.items()} } real={len(real_rows)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for spec in args.adapter:
        label, base, adapter = parse_adapter(spec)
        if not (adapter / "adapter_config.json").exists():
            print(f"MISSING adapter {adapter}; skipping {label}", flush=True)
            continue
        paths = {k: OUT / f"{label}__{k}.jsonl" for k in list(term_rows) + (
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
                res = {}
                for cond in TERM_CONDS:
                    vocab_key = {"none": None, "oracle": "vocab_oracle",
                                 "retrieved": "vocab_retrieved"}[cond]
                    jobs = [{"text": r["hyp"],
                             "vocab": [] if vocab_key is None else r[vocab_key]} for r in rows]
                    res[cond] = run_jobs(model, processor, tokenizer, jobs, args.batch_size,
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
                        h.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tmp.rename(paths[key])
                print(f"  wrote {paths[key].name} [{time.perf_counter()-t0:.0f}s]", flush=True)
            if real_rows and (not paths["real"].exists() or args.force):
                res = {}
                for cond in REAL_CONDS:
                    jobs = [{"text": r["input"],
                             "vocab": r["vocab_retrieved"] if cond == "retrieved" else []}
                            for r in real_rows]
                    res[cond] = run_jobs(model, processor, tokenizer, jobs,
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
