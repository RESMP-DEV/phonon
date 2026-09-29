"""Keep (parakeet_raw, target) TTS pairs with non-empty hyp and fair WER <= 0.6."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TTS_PAIRS, pair_wer, fair_norm, read_jsonl, record_failure, write_jsonl  # noqa: E402


def filter_rows(rows: list[dict], max_wer: float = 0.6) -> tuple[list[dict], dict]:
    kept = []
    dropped_empty = 0
    dropped_wer = 0
    for row in rows:
        if row.get("id") == "_meta":
            continue
        raw = (row.get("parakeet_raw") or "").strip()
        target = row.get("target") or ""
        if not raw:
            dropped_empty += 1
            continue
        wer = pair_wer(target, raw, fair_norm)
        if wer > max_wer:
            dropped_wer += 1
            continue
        out = {
            "id": row["id"],
            "source_id": row.get("source_id"),
            "voice": row.get("voice"),
            "degraded": bool(row.get("degraded")),
            "parakeet_raw": raw,
            "target": target,
            "wer": wer,
            "wav": row.get("wav"),
            "split": "tts_synth",
        }
        if row.get("wav_seconds") is not None:
            out["wav_seconds"] = row["wav_seconds"]
        kept.append(out)
    wers = [row["wer"] for row in kept]
    seconds = [float(row["wav_seconds"]) for row in kept if row.get("wav_seconds") is not None]
    stats = {
        "input": len(rows),
        "kept": len(kept),
        "dropped_empty": dropped_empty,
        "dropped_wer": dropped_wer,
        "max_wer": max_wer,
        "mean_fair_wer": (sum(wers) / len(wers)) if wers else 0.0,
        "wav_seconds": sum(seconds),
        "wav_hours": (sum(seconds) / 3600.0) if seconds else 0.0,
    }
    return kept, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=TTS_PAIRS)
    parser.add_argument("--max-wer", type=float, default=0.6)
    args = parser.parse_args()
    rows = read_jsonl(args.inp)
    kept, stats = filter_rows(rows, args.max_wer)
    write_jsonl(args.out, kept)
    print(stats, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        record_failure("filter_tts_pairs", f"{type(exc).__name__}: {exc}", 1)
        raise
