"""Step 3: run each adapter over the term-eval clips in 3 vocabulary conditions, plus the
real-dictation holdouts in 2 conditions. GPU 1."""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vocab_common import read_jsonl, system_with_vocab  # noqa: E402

DATA = Path("/data/phonon_vocab_v0")
ADAPTERS = Path("/data/phonon_corrector_v0/adapters")
LFM12 = "LiquidAI/LFM2.5-1.2B-Instruct"
LFM350 = "LiquidAI/LFM2.5-350M"

SYSTEMS = [
    ("vocab_real", LFM12, "vocab_real"),
    ("vocab_real_acoustic", LFM12, "vocab_real_acoustic"),
    ("novocab_real_acoustic", LFM12, "vocab_novocab_real_acoustic"),
    ("vocab_real_acoustic_350m", LFM350, "vocab_real_acoustic_350m"),
    # control: the existing personal adapter, never trained with a vocabulary line
    ("lfm2.5-1.2b_personal", LFM12, "lfm2.5-1.2b"),
]


def apply_prompt_vocab(processor, user_text: str, vocab: list[str] | None) -> str:
    messages = [
        {"role": "system", "content": system_with_vocab(vocab)},
        {"role": "user", "content": user_text},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    for extra in ({"enable_thinking": False}, {}):
        try:
            text = processor.apply_chat_template(messages, **kwargs, **extra)
            if isinstance(text, list):
                text = text[0]
            return str(text)
        except TypeError:
            continue
    raise RuntimeError("chat template failed")


def generate(model, processor, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
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
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False,
                  "pad_token_id": pad_id, "eos_token_id": getattr(tokenizer, "eos_token_id", None)}
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        if hasattr(model.generation_config, "temperature"):
            model.generation_config.temperature = None
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    return [strip_thinking(tokenizer.decode(out[i, input_len:], skip_special_tokens=True))
            for i in range(out.shape[0])]


def run_jobs(model, processor, tokenizer, jobs: list[dict], batch_size: int,
             max_new_tokens: int, label: str) -> list[str]:
    """jobs: [{'text':..., 'vocab': [...]}] -> outputs in the same order."""
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i]["text"]) + 4 * len(jobs[i]["vocab"]))
    outs = [""] * len(jobs)
    t0 = time.perf_counter()
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        prompts = [apply_prompt_vocab(processor, jobs[i]["text"], jobs[i]["vocab"]) for i in idx]
        res = generate(model, processor, tokenizer, prompts, max_new_tokens)
        for i, h in zip(idx, res):
            outs[i] = h
        done = min(start + batch_size, len(order))
        if (start // batch_size) % 10 == 0 or done == len(order):
            el = time.perf_counter() - t0
            print(f"  {label} {done}/{len(order)} {done/max(el,1e-9):.1f} rows/s", flush=True)
    return outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--out-dir", type=Path, default=DATA / "refined")
    ap.add_argument("--real-max-new-tokens", type=int, default=2048,
                    help="Real dictation can run past 600 words; 320 silently truncated them.")
    ap.add_argument("--skip-terms", action="store_true")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch

    from eval_corrector import load_stack

    term_rows = read_jsonl(DATA / "eval_conditions.jsonl")
    if args.limit:
        term_rows = term_rows[: args.limit]
    real_sets = {name: read_jsonl(DATA / f"eval_{name}.jsonl")
                 for name in ("wispr_holdout120", "wispr_text_holdout")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s}

    for label, base, adapter_name in SYSTEMS:
        if only and label not in only:
            continue
        adapter = ADAPTERS / adapter_name
        if not (adapter / "adapter_config.json").exists():
            print(f"MISSING adapter {adapter}; skipping {label}", flush=True)
            continue
        done_paths = ([] if args.skip_terms else [args.out_dir / f"{label}__terms.jsonl"]) + (
            [] if args.skip_real else [args.out_dir / f"{label}__{n}.jsonl" for n in real_sets])
        if done_paths and all(p.exists() for p in done_paths) and not args.force:
            print(f"skip {label} (complete)", flush=True)
            continue
        print(f"=== {label} base={base} adapter={adapter}", flush=True)
        t0 = time.perf_counter()
        model, processor, tokenizer = load_stack(base, adapter)
        try:
            conds = {} if args.skip_terms else {
                "none": [{"text": r["hyp"], "vocab": []} for r in term_rows],
                "oracle": [{"text": r["hyp"], "vocab": r["vocab_oracle"]} for r in term_rows],
                "realistic": [{"text": r["hyp"], "vocab": r["vocab_realistic"]} for r in term_rows],
            }
            res = {c: run_jobs(model, processor, tokenizer, jobs, args.batch_size,
                               args.max_new_tokens, f"{label}/{c}")
                   for c, jobs in conds.items()}
            if not args.skip_terms:
              with (args.out_dir / f"{label}__terms.jsonl").open("w", encoding="utf-8") as h:
                for i, r in enumerate(term_rows):
                    h.write(json.dumps({
                        "id": r["id"], "term": r["term"], "kind": r["kind"], "voice": r["voice"],
                        "reference": r["reference"], "hyp": r["hyp"],
                        "realistic_has_term": r["realistic_has_term"],
                        "vocab_oracle": r["vocab_oracle"], "vocab_realistic": r["vocab_realistic"],
                        "out_none": res["none"][i], "out_oracle": res["oracle"][i],
                        "out_realistic": res["realistic"][i],
                    }, ensure_ascii=False) + "\n")
            for name, rows in ({} if args.skip_real else real_sets).items():
                jobs_none = [{"text": r["input"], "vocab": []} for r in rows]
                jobs_ref = [{"text": r["input"], "vocab": r["vocab_ref"]} for r in rows]
                o_none = run_jobs(model, processor, tokenizer, jobs_none, args.batch_size,
                                  args.real_max_new_tokens, f"{label}/{name}/none")
                o_ref = run_jobs(model, processor, tokenizer, jobs_ref, args.batch_size,
                                 args.real_max_new_tokens, f"{label}/{name}/ref")
                with (args.out_dir / f"{label}__{name}.jsonl").open("w", encoding="utf-8") as h:
                    for i, r in enumerate(rows):
                        h.write(json.dumps({
                            "id": r["id"], "input": r["input"], "reference": r["reference"],
                            "true_terms": r["true_terms"], "vocab_ref": r["vocab_ref"],
                            "out_none": o_none[i], "out_vocabref": o_ref[i],
                        }, ensure_ascii=False) + "\n")
        finally:
            del model
            torch.cuda.empty_cache()
        print(f"{label} done wall_s={time.perf_counter()-t0:.0f}", flush=True)
    print("REFINE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
