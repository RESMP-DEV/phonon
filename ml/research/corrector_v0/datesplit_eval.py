"""Evaluate raw + adapters on the date-split holdouts.

Reuses eval_corrector.load_stack / generate_batch and common.score_lists; does not
touch eval_corrector.py itself. Sets are plain jsonl with id/input/reference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_ROOT,
    fair_norm,
    heartbeat,
    pair_wer,
    read_jsonl,
    score_lists,
    set_hf_env,
)
from eval_corrector import generate_batch, load_stack  # noqa: E402

SPLIT_DIR = DATA_ROOT / "datesplit"
SETS = {
    "holdout_date_text": SPLIT_DIR / "holdout_date_text.jsonl",
    "holdout_date_audio": SPLIT_DIR / "holdout_date_audio.jsonl",
}


def score_block(rows: list[dict], hyps: list[str]) -> dict:
    refs = [r["reference"] for r in rows]
    ins = [r["input"] for r in rows]
    per_row = []
    worse = 0
    win = tie = loss = 0
    for r, h in zip(rows, hyps, strict=True):
        in_w = pair_wer(r["reference"], r["input"], fair_norm)
        out_w = pair_wer(r["reference"], h, fair_norm)
        d = out_w - in_w
        per_row.append({"id": r["id"], "in": in_w, "out": out_w, "delta": d})
        if d > 1e-9:
            worse += 1
            loss += 1
        elif d < -1e-9:
            win += 1
        else:
            tie += 1
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "corrector": score_lists(refs, hyps),
        "baseline": score_lists(refs, ins),
        "damage_rate": worse / n,
        "win": win, "tie": tie, "loss": loss,
        "win_rate": win / n, "tie_rate": tie / n, "loss_rate": loss / n,
        "mean_delta_fair_wer": sum(p["delta"] for p in per_row) / n,
        "worst": sorted(per_row, key=lambda p: -p["delta"])[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="label:model_id:adapter")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-json", type=Path, default=SPLIT_DIR / "eval_datesplit.json")
    args = ap.parse_args()
    set_hf_env()

    sets = {name: read_jsonl(path) for name, path in SETS.items()}
    for name, rows in sets.items():
        print(f"set {name}: n={len(rows)}", flush=True)

    out = {"sets": {n: len(r) for n, r in sets.items()}, "models": []}
    if args.out_json.exists():
        prev = json.loads(args.out_json.read_text())
        out["models"] = prev.get("models", [])
    done = {m["label"] for m in out["models"]}

    # raw baseline
    if "raw" not in done:
        raw = {"label": "raw", "model_id": None, "adapter": None, "sets": {}}
        for name, rows in sets.items():
            raw["sets"][name] = score_block(rows, [r["input"] for r in rows])
        out["models"].append(raw)
        args.out_json.write_text(json.dumps(out, indent=2) + "\n")
        print("raw done", flush=True)

    import torch

    for spec in args.model:
        label, model_id, adapter = spec.split(":", 2)
        if label in done:
            print(f"skip {label}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            model, processor, tokenizer = load_stack(model_id, Path(adapter) if adapter else None)
        except Exception as exc:
            print(f"LOAD_FAIL {label}: {type(exc).__name__}: {exc}", flush=True)
            out["models"].append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
            args.out_json.write_text(json.dumps(out, indent=2) + "\n")
            continue
        entry = {"label": label, "model_id": model_id, "adapter": adapter, "sets": {}}
        try:
            for name, rows in sets.items():
                order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input"]))
                hyps = [""] * len(rows)
                for start in range(0, len(order), args.batch_size):
                    idx = order[start:start + args.batch_size]
                    outs = generate_batch(model, processor, tokenizer,
                                          [rows[i]["input"] for i in idx], args.max_new_tokens)
                    for i, h in zip(idx, outs, strict=True):
                        hyps[i] = h
                    if (start // args.batch_size) % 4 == 0:
                        print(f"{label} {name} {start + len(idx)}/{len(rows)}", flush=True)
                        heartbeat()
                entry["sets"][name] = score_block(rows, hyps)
                (SPLIT_DIR / "preds").mkdir(parents=True, exist_ok=True)
                (SPLIT_DIR / "preds" / f"{label}_{name}.jsonl").write_text(
                    "".join(json.dumps({"id": r["id"], "input": r["input"], "output": h,
                                        "reference": r["reference"]}, ensure_ascii=False) + "\n"
                            for r, h in zip(rows, hyps, strict=True)), encoding="utf-8")
        finally:
            del model
            torch.cuda.empty_cache()
        entry["seconds"] = time.perf_counter() - t0
        out["models"].append(entry)
        args.out_json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"{label} done in {entry['seconds']:.0f}s "
              + " ".join(f"{n}={b['corrector']['fair_wer']:.4f}" for n, b in entry["sets"].items()),
              flush=True)

    print("EVAL done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
