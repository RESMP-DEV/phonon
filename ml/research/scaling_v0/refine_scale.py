"""scaling_v0 step 4: one adapter over seen-voice clips, unseen-voice clips and real audio.

Conditions: no vocabulary line and the retrieved list on both clip sets, retrieved list only on
the real-audio date holdout. Outputs land in /data/phonon_scaling_v0/refined/.
"""
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
sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")

from vocab_common import read_jsonl  # noqa: E402

ADAPTERS = Path("/data/phonon_corrector_v0/adapters")
OUT = Path("/data/phonon_scaling_v0/refined")
SEEN = Path("/data/phonon_vocab_v1/eval_conditions.jsonl")
UNSEEN = Path("/data/phonon_term_eval_v1/eval_conditions.jsonl")
REAL = Path("/data/phonon_term_eval_v1/real/eval_holdout_date_audio.jsonl")
LFM12 = "LiquidAI/LFM2.5-1.2B-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True, help="comma list of adapter dir names")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--real-batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--real-max-new-tokens", type=int, default=2048)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    from eval_corrector import load_stack
    from refine_vocab1 import run_jobs

    seen, unseen, real = read_jsonl(SEEN), read_jsonl(UNSEEN), read_jsonl(REAL)
    print(f"seen={len(seen)} unseen={len(unseen)} real={len(real)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for name in [a for a in args.adapters.split(",") if a]:
        adapter = ADAPTERS / name
        if not (adapter / "adapter_config.json").exists():
            print(f"MISSING adapter {adapter}; skipping", flush=True)
            continue
        paths = {k: OUT / f"{name}__{k}.jsonl" for k in ("seen", "unseen", "real")}
        if all(p.exists() for p in paths.values()) and not args.force:
            print(f"skip {name} (complete)", flush=True)
            continue
        print(f"=== {name}", flush=True)
        t0 = time.perf_counter()
        model, processor, tokenizer = load_stack(LFM12, adapter)
        try:
            for key, rows in (("seen", seen), ("unseen", unseen)):
                if paths[key].exists() and not args.force:
                    continue
                jn = [{"text": r["hyp"], "vocab": []} for r in rows]
                jr = [{"text": r["hyp"], "vocab": r["vocab_retrieved"]} for r in rows]
                o_none = run_jobs(model, processor, tokenizer, jn, args.batch_size,
                                  args.max_new_tokens, f"{name}/{key}/none")
                o_ret = run_jobs(model, processor, tokenizer, jr, args.batch_size,
                                 args.max_new_tokens, f"{name}/{key}/retrieved")
                with paths[key].open("w", encoding="utf-8") as h:
                    for i, r in enumerate(rows):
                        h.write(json.dumps({
                            "id": r["id"], "term": r["term"], "kind": r["kind"],
                            "voice": r["voice"], "reference": r["reference"], "hyp": r["hyp"],
                            "retrieved_has_term": r["retrieved_has_term"],
                            "out_none": o_none[i], "out_retrieved": o_ret[i],
                        }, ensure_ascii=False) + "\n")
            if not paths["real"].exists() or args.force:
                jr = [{"text": r["input"], "vocab": r["vocab_retrieved"]} for r in real]
                o_ret = run_jobs(model, processor, tokenizer, jr, args.real_batch_size,
                                 args.real_max_new_tokens, f"{name}/real/retrieved")
                with paths["real"].open("w", encoding="utf-8") as h:
                    for i, r in enumerate(real):
                        h.write(json.dumps({
                            "id": r["id"], "input": r["input"], "reference": r["reference"],
                            "true_terms": r["true_terms"],
                            "out_retrieved": o_ret[i],
                        }, ensure_ascii=False) + "\n")
        finally:
            del model
            torch.cuda.empty_cache()
        print(f"{name} done wall_s={time.perf_counter()-t0:.0f}", flush=True)
    print("REFINE SCALE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
