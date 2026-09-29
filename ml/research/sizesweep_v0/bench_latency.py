"""sizesweep_v0: batch-1 decode latency and throughput for each adapter, on real holdout rows.

Same prompts, same greedy decode and same length guard as the eval. Reports per model:
tokens/s decoded at batch 1, median and p90 ms per row, and batch-16 rows/s.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
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
sys.path.insert(0, "/home/user/phonon/research/bigrun_v0")

from refine_bigrun import REAL, guard_cap, parse_adapter  # noqa: E402
from vocab_common import read_jsonl  # noqa: E402

OUT = Path("/data/phonon_sizesweep_v0/latency.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", default=[])
    ap.add_argument("--rows", type=int, default=36)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    import torch

    from eval_corrector import load_stack
    from refine_vocab1 import apply_prompt_vocab

    rows = read_jsonl(REAL)
    rows = sorted(rows, key=lambda r: len(r["input"].split()))
    # a deterministic spread over the length distribution
    step = max(1, len(rows) // args.rows)
    sample = rows[::step][: args.rows]
    results = {}
    if args.out.exists():
        try:
            results = json.loads(args.out.read_text())
        except Exception:
            results = {}

    for spec in args.adapter:
        label, base, adapter = parse_adapter(spec)
        if not (adapter / "adapter_config.json").exists():
            print(f"MISSING adapter {adapter}; skipping {label}", flush=True)
            continue
        print(f"=== {label} base={base}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t_load = time.perf_counter()
        model, processor, tokenizer = load_stack(base, adapter)
        load_s = time.perf_counter() - t_load
        weights_gb = torch.cuda.max_memory_allocated() / 1e9
        prompts = [apply_prompt_vocab(processor, r["input"], r["vocab_retrieved"])
                   for r in sample]
        caps = [guard_cap(tokenizer, r["input"], 2048) for r in sample]
        pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.do_sample = False
            if hasattr(model.generation_config, "temperature"):
                model.generation_config.temperature = None
        per_row, new_tok, in_tok = [], [], []
        prev = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        try:
            for i, (p, c) in enumerate(zip(prompts, caps)):
                enc = tokenizer(p, return_tensors="pt")
                enc = {k: v.to(model.device) for k, v in enc.items()}
                nin = enc["input_ids"].shape[-1]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=c, do_sample=False,
                                         pad_token_id=pad_id, eos_token_id=eos_id)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                n_new = int(out.shape[-1] - nin)
                if i >= args.warmup:
                    per_row.append(dt)
                    new_tok.append(n_new)
                    in_tok.append(nin)
            # batched throughput, same rows
            torch.cuda.synchronize()
            tb0 = time.perf_counter()
            nb_tok = 0
            for s in range(0, len(prompts), args.batch):
                chunk = prompts[s:s + args.batch]
                cc = caps[s:s + args.batch]
                enc = tokenizer(chunk, return_tensors="pt", padding=True)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                nin = enc["input_ids"].shape[-1]
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=max(cc), do_sample=False,
                                         pad_token_id=pad_id, eos_token_id=eos_id)
                nb_tok += int((out[:, nin:] != pad_id).sum().item())
            torch.cuda.synchronize()
            batch_s = time.perf_counter() - tb0
        finally:
            tokenizer.padding_side = prev
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        tot_t = sum(per_row)
        ms = sorted(x * 1000 for x in per_row)
        results[label] = {
            "base": base,
            "rows_timed": len(per_row),
            "load_s": round(load_s, 1),
            "weights_gb": round(weights_gb, 2),
            "peak_gb_bs1": round(peak_gb, 2),
            "decode_tokens_per_s_bs1": round(sum(new_tok) / tot_t, 1),
            "rows_per_s_bs1": round(len(per_row) / tot_t, 2),
            "ms_per_row_median": round(statistics.median(ms), 1),
            "ms_per_row_p90": round(ms[int(0.9 * (len(ms) - 1))], 1),
            "ms_per_row_mean": round(sum(ms) / len(ms), 1),
            "mean_input_tokens": round(sum(in_tok) / len(in_tok), 1),
            "mean_new_tokens": round(sum(new_tok) / len(new_tok), 1),
            f"rows_per_s_bs{args.batch}": round(len(prompts) / batch_s, 2),
            f"decode_tokens_per_s_bs{args.batch}": round(nb_tok / batch_s, 1),
        }
        print(json.dumps({label: results[label]}), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1) + "\n")
        del model
        torch.cuda.empty_cache()
    print("BENCH DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
