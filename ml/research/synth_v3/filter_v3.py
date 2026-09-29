"""Step 2b: register filter over the generated candidates -> clean_v3.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register import (  # noqa: E402
    FEATURES_TEXT,
    in_band_count,
    read_jsonl,
    structural_reject,
    text_features,
)

CAND = Path("/data/phonon_synth_v3/candidates_v3.jsonl")
BAND = Path("/data/phonon_synth_v3/register_band.json")
OUT = Path("/data/phonon_synth_v3/clean_v3.jsonl")
STATS = Path("/data/phonon_synth_v3/filter_stats_v3.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", type=Path, default=CAND)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--min-in-band", type=int, default=5)
    ap.add_argument("--require", default="func_frac",
                    help="comma-separated features that must be in band")
    ap.add_argument("--target", type=int, default=10000)
    args = ap.parse_args()

    band = {k: tuple(v) for k, v in json.loads(BAND.read_text()).items() if k in FEATURES_TEXT}
    required = [f for f in args.require.split(",") if f]
    reasons = Counter()
    miss_counter = Counter()
    kept = []
    n = 0
    for row in read_jsonl(args.cand):
        n += 1
        text = row.get("text") or ""
        if not row.get("pre_ok"):
            reasons["pre_" + (row.get("pre_reason") or "?")] += 1
            continue
        sr = structural_reject(text)
        if sr:
            reasons["struct_" + sr] += 1
            continue
        feats = text_features(text)
        if feats is None:
            reasons["no_features"] += 1
            continue
        nb, misses = in_band_count(feats, band)
        for m in misses:
            miss_counter[m] += 1
        if any(f in misses for f in required):
            reasons["out_of_band_required"] += 1
            continue
        if nb < args.min_in_band:
            reasons[f"in_band_{nb}_of_{len(band)}"] += 1
            continue
        reasons["kept"] += 1
        out = dict(row)
        out["id"] = f"utt_v3_{len(kept) + 1:06d}"
        out["register"] = {k: round(v, 5) for k, v in feats.items()}
        out["in_band"] = nb
        kept.append(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "candidates": n,
        "kept": len(kept),
        "keep_rate": len(kept) / max(n, 1),
        "pre_ok": sum(1 for k, v in reasons.items() if k == "kept") + sum(
            v for k, v in reasons.items() if not k.startswith("pre_")),
        "reasons": dict(reasons.most_common()),
        "band_misses": dict(miss_counter.most_common()),
        "min_in_band": args.min_in_band,
        "required": required,
        "band": {k: list(v) for k, v in band.items()},
        "out": str(args.out),
    }
    STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: stats[k] for k in
                      ("candidates", "kept", "keep_rate", "pre_ok", "reasons", "band_misses")},
                     indent=1))
    print(f"wrote {args.out} ({len(kept)} rows); target {args.target}", flush=True)
    return 0 if len(kept) >= args.target else 2


if __name__ == "__main__":
    raise SystemExit(main())
