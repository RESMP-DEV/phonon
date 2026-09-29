"""Union train pairs, drop holdout, dedupe (input, target), write chat SFT JSONL."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEV_JSONL,
    HOLDOUT_SPLITS,
    PARAKEET_JSONL,
    TRAIN_JSONL,
    TRAIN_SPLITS,
    TTS_PAIRS,
    chat_messages,
    load_wispr_pairs,
    read_jsonl,
    record_failure,
    write_jsonl,
)


def add_pair(
    bucket: dict[tuple[str, str], dict],
    raw: str,
    target: str,
    source: str,
    row_id: str,
) -> None:
    raw = (raw or "").strip()
    target = (target or "").strip()
    if not raw or not target:
        return
    key = (raw, target)
    if key in bucket:
        return
    bucket[key] = {
        "id": row_id,
        "source": source,
        "input": raw,
        "target": target,
        "messages": chat_messages(raw, target),
    }


def collect(
    include_parakeet: bool = True,
    include_tts: bool = True,
    include_wispr: bool = True,
) -> tuple[list[dict], dict]:
    bucket: dict[tuple[str, str], dict] = {}
    stats = {
        "wispr_asr": 0,
        "parakeet_train": 0,
        "tts": 0,
        "skipped_holdout": 0,
        "missing_parakeet": False,
        "missing_tts": False,
        "include_wispr": include_wispr,
        "include_parakeet": include_parakeet,
        "include_tts": include_tts,
    }
    for row in load_wispr_pairs():
        split = row.get("split")
        if split in HOLDOUT_SPLITS:
            stats["skipped_holdout"] += 1
            continue
        if split not in TRAIN_SPLITS:
            continue
        if not include_wispr:
            continue
        add_pair(bucket, row.get("asr") or "", row.get("target") or "", f"wispr_asr:{split}", row["id"])
        stats["wispr_asr"] += 1

    if include_parakeet:
        if not PARAKEET_JSONL.exists():
            stats["missing_parakeet"] = True
        else:
            for row in read_jsonl(PARAKEET_JSONL):
                if row.get("split") != "wispr_train":
                    continue
                add_pair(
                    bucket,
                    row.get("parakeet_raw") or "",
                    row.get("target") or "",
                    "parakeet:wispr_train",
                    f"pk_{row['id']}",
                )
                stats["parakeet_train"] += 1
    holdout_ids = {row["id"] for row in load_wispr_pairs() if row.get("split") in HOLDOUT_SPLITS}
    if include_tts:
        if not TTS_PAIRS.exists():
            stats["missing_tts"] = True
        else:
            for row in read_jsonl(TTS_PAIRS):
                source_id = str(row.get("source_id") or "")
                if source_id in holdout_ids or row.get("split") in HOLDOUT_SPLITS:
                    stats["skipped_holdout"] += 1
                    continue
                add_pair(
                    bucket,
                    row.get("parakeet_raw") or "",
                    row.get("target") or "",
                    f"tts:{row.get('voice')}",
                    row.get("id") or "",
                )
                stats["tts"] += 1
    rows = list(bucket.values())
    stats["unique"] = len(rows)
    return rows, stats


def split_dev(rows: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_dev = max(1, int(round(len(shuffled) * frac))) if shuffled else 0
    n_dev = min(n_dev, max(0, len(shuffled) - 1)) if len(shuffled) > 1 else n_dev
    return shuffled[n_dev:], shuffled[:n_dev]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", type=Path, default=TRAIN_JSONL)
    parser.add_argument("--dev-out", type=Path, default=DEV_JSONL)
    parser.add_argument("--dev-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-parakeet", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument(
        "--no-wispr",
        action="store_true",
        help="Drop Wispr-ASR inputs; keep Parakeet/TTS pairs only.",
    )
    args = parser.parse_args()
    rows, stats = collect(
        include_parakeet=not args.no_parakeet,
        include_tts=not args.no_tts,
        include_wispr=not args.no_wispr,
    )
    train_rows, dev_rows = split_dev(rows, args.dev_frac, args.seed)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.dev_out, dev_rows)
    print(
        {
            **stats,
            "train": len(train_rows),
            "dev": len(dev_rows),
            "train_out": str(args.train_out),
            "dev_out": str(args.dev_out),
        },
        flush=True,
    )
    holdout_ids = {
        row["id"] for row in load_wispr_pairs() if row.get("split") in HOLDOUT_SPLITS
    }
    leaked = [row for row in rows if row["id"] in holdout_ids]
    if leaked:
        raise RuntimeError(f"holdout leak: {len(leaked)} rows")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        record_failure("build_train", f"{type(exc).__name__}: {exc}", 1)
        raise
