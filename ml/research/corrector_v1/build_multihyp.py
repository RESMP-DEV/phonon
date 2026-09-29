"""Build S1 / M3 / M4 chat SFT JSONL. Holdout ids never enter train or dev."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    DATA_ROOT,
    HYP_PATHS,
    LETTERS,
    LETTER_MODEL,
    MISSING,
    PARAKEET_JSONL,
    SEED,
    V0_DEV_JSONL,
    V0_TRAIN_JSONL,
    chat_messages,
    ensure_data_dirs,
    format_user,
    load_hyp_map,
    load_v0_common,
    read_jsonl,
    set_hf_env,
    write_jsonl,
)


def add_pair(
    bucket: dict[tuple[str, str], dict],
    user_text: str,
    target: str,
    source: str,
    row_id: str,
    letters: tuple[str, ...],
) -> None:
    user_text = (user_text or "").strip()
    target = (target or "").strip()
    if not user_text or not target:
        return
    key = (user_text, target)
    if key in bucket:
        return
    bucket[key] = {
        "id": row_id,
        "source": source,
        "input": user_text,
        "target": target,
        "letters": list(letters),
        "messages": chat_messages(user_text, target, letters),
    }


def split_dev(rows: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_dev = max(1, int(round(len(shuffled) * frac))) if shuffled else 0
    n_dev = min(n_dev, max(0, len(shuffled) - 1)) if len(shuffled) > 1 else n_dev
    return shuffled[n_dev:], shuffled[:n_dev]


def dropout_copy(
    hyps: dict[str, str],
    letters: tuple[str, ...],
    row_id: str,
    seed: int,
) -> dict[str, str]:
    v0 = load_v0_common()
    rng = random.Random((v0.stable_hash(row_id) ^ seed) & 0xFFFFFFFF)
    n_drop = rng.choice((1, 2))
    n_drop = min(n_drop, len(letters))
    dropped = set(rng.sample(list(letters), n_drop))
    out = dict(hyps)
    for letter in dropped:
        out[letter] = MISSING
    return out


def load_required_hyps(model_keys: tuple[str, ...]) -> dict[str, dict[str, str]]:
    maps = {key: load_hyp_map(HYP_PATHS[key]) for key in model_keys}
    missing_files = [key for key, path in ((k, HYP_PATHS[k]) for k in model_keys) if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"missing hyp files: {missing_files}")
    by_id: dict[str, dict[str, str]] = {}
    ids = set.intersection(*(set(maps[key]) for key in model_keys))
    for row_id in ids:
        by_id[row_id] = {
            letter: (maps[LETTER_MODEL[letter]].get(row_id) or {}).get("hypothesis") or ""
            for letter in LETTERS
            if LETTER_MODEL[letter] in model_keys
        }
    return by_id


def collect_multihyp(letters: tuple[str, ...], seed: int) -> tuple[list[dict], dict]:
    v0 = load_v0_common()
    model_keys = tuple(LETTER_MODEL[letter] for letter in letters)
    hyps_by_id = load_required_hyps(model_keys)
    bucket: dict[tuple[str, str], dict] = {}
    stats = {
        "multihyp": 0,
        "dropout": 0,
        "wispr_text": 0,
        "skipped_holdout": 0,
        "skipped_no_target": 0,
        "skipped_missing_hyp": 0,
        "letters": list(letters),
    }
    holdout_ids = {row["id"] for row in v0.load_wispr_pairs() if row.get("split") in v0.HOLDOUT_SPLITS}
    for row in read_jsonl(PARAKEET_JSONL):
        if row.get("split") != "wispr_train":
            continue
        row_id = row["id"]
        if row_id in holdout_ids:
            stats["skipped_holdout"] += 1
            continue
        target = (row.get("target") or "").strip()
        if not target:
            stats["skipped_no_target"] += 1
            continue
        if row_id not in hyps_by_id:
            stats["skipped_missing_hyp"] += 1
            continue
        hyps = hyps_by_id[row_id]
        user = format_user(hyps, letters)
        add_pair(bucket, user, target, "multihyp:wispr_train", row_id, letters)
        stats["multihyp"] += 1
        dropped = dropout_copy(hyps, letters, row_id, seed)
        add_pair(
            bucket,
            format_user(dropped, letters),
            target,
            "dropout:wispr_train",
            f"{row_id}:drop",
            letters,
        )
        stats["dropout"] += 1
    for row in v0.load_wispr_pairs():
        split = row.get("split")
        if split in v0.HOLDOUT_SPLITS:
            stats["skipped_holdout"] += 1
            continue
        if split != "wispr_text_train":
            continue
        row_id = row["id"]
        if row_id in holdout_ids:
            stats["skipped_holdout"] += 1
            continue
        hyps = {letter: MISSING for letter in letters}
        hyps["A"] = (row.get("asr") or "").strip()
        add_pair(
            bucket,
            format_user(hyps, letters),
            (row.get("target") or "").strip(),
            f"wispr_asr:{split}",
            row_id,
            letters,
        )
        stats["wispr_text"] += 1
    rows = list(bucket.values())
    leaked = [
        row
        for row in rows
        if row["id"] in holdout_ids or str(row["id"]).split(":")[0] in holdout_ids
    ]
    if leaked:
        raise RuntimeError(f"holdout leak: {len(leaked)} rows")
    stats["unique"] = len(rows)
    return rows, stats


def copy_s1() -> dict[str, Any]:
    if not V0_TRAIN_JSONL.exists() or not V0_DEV_JSONL.exists():
        raise FileNotFoundError("v0 train/dev jsonl missing")
    train_out = DATA_ROOT / "s1_train.jsonl"
    dev_out = DATA_ROOT / "s1_dev.jsonl"
    shutil.copy2(V0_TRAIN_JSONL, train_out)
    shutil.copy2(V0_DEV_JSONL, dev_out)
    return {
        "arm": "S1",
        "train": sum(1 for _ in read_jsonl(train_out)),
        "dev": sum(1 for _ in read_jsonl(dev_out)),
        "train_out": str(train_out),
        "dev_out": str(dev_out),
        "source": "exact copy of v0 train.jsonl / dev.jsonl",
    }


def write_arm(arm: str, letters: tuple[str, ...], frac: float, seed: int) -> dict[str, Any]:
    rows, stats = collect_multihyp(letters, seed)
    train_rows, dev_rows = split_dev(rows, frac, seed)
    train_out = DATA_ROOT / f"{arm.lower()}_train.jsonl"
    dev_out = DATA_ROOT / f"{arm.lower()}_dev.jsonl"
    write_jsonl(train_out, train_rows)
    write_jsonl(dev_out, dev_rows)
    return {
        **stats,
        "arm": arm,
        "train": len(train_rows),
        "dev": len(dev_rows),
        "train_out": str(train_out),
        "dev_out": str(dev_out),
        "seed": seed,
        "dev_frac": frac,
    }


def score_train_recognizers() -> dict[str, Any]:
    v0 = load_v0_common()
    targets = {
        row["id"]: row.get("target") or ""
        for row in read_jsonl(PARAKEET_JSONL)
        if row.get("split") == "wispr_train" and (row.get("target") or "").strip()
    }
    report = {}
    for key, path in HYP_PATHS.items():
        if not path.exists():
            report[key] = {"missing": True, "path": str(path)}
            continue
        hyps = load_hyp_map(path)
        ids = sorted(set(targets) & set(hyps))
        refs = [targets[i] for i in ids]
        preds = [(hyps[i].get("hypothesis") or "") for i in ids]
        audio = sum(float(hyps[i].get("audio_seconds") or 0.0) for i in ids)
        elapsed = sum(float(hyps[i].get("elapsed_seconds") or 0.0) for i in ids)
        meta_path = Path(str(path) + ".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        scores = v0.score_lists(refs, preds)
        report[key] = {
            "n": len(ids),
            "fair_wer": scores["fair_wer"],
            "strict_lc_wer": scores["strict_lc_wer"],
            "audio_seconds": audio,
            "elapsed_seconds": elapsed,
            "realtime_factor": (elapsed / audio) if audio else None,
            "meta_realtime_factor": meta.get("realtime_factor"),
            "path": str(path),
        }
    out = DATA_ROOT / "train_asr_scores.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def count_tokens(paths: list[Path], model_id: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from config import QWEN_ID, apply_chat_template, hf_token

    tok = AutoTokenizer.from_pretrained(model_id or QWEN_ID, token=hf_token(), trust_remote_code=True)
    counts: list[int] = []
    max_row = None
    for path in paths:
        for row in read_jsonl(path):
            text = apply_chat_template(tok, row["messages"], add_generation_prompt=False)
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            counts.append(n)
            if max_row is None or n > max_row[0]:
                max_row = (n, row.get("id"), str(path))
    if not counts:
        return {"n": 0, "max_tokens": 0, "mean_tokens": 0.0, "p95_tokens": 0, "over_1024": 0, "over_2048": 0}
    counts_sorted = sorted(counts)
    p95 = counts_sorted[min(len(counts_sorted) - 1, int(round(0.95 * (len(counts_sorted) - 1))))]
    return {
        "n": len(counts),
        "max_tokens": max(counts),
        "mean_tokens": sum(counts) / len(counts),
        "p95_tokens": p95,
        "over_1024": sum(c > 1024 for c in counts),
        "over_2048": sum(c > 2048 for c in counts),
        "max_id": None if max_row is None else max_row[1],
        "max_path": None if max_row is None else max_row[2],
        "seq_len": 1024 if max(counts) <= 1024 else 2048,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-tokens", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_hf_env()
    ensure_data_dirs()
    payload = {
        "seed": args.seed,
        "s1": copy_s1(),
        "m3": write_arm("M3", ("A", "B", "C"), args.dev_frac, args.seed),
        "m4": write_arm("M4", LETTERS, args.dev_frac, args.seed),
        "train_asr": score_train_recognizers(),
    }
    if not args.skip_tokens:
        from config import QWEN_ID

        token_paths = [
            DATA_ROOT / "m4_train.jsonl",
            DATA_ROOT / "m4_dev.jsonl",
            DATA_ROOT / "m3_train.jsonl",
            DATA_ROOT / "s1_train.jsonl",
        ]
        payload["tokens"] = count_tokens([p for p in token_paths if p.exists()], QWEN_ID)
        max_tok = int(payload["tokens"]["max_tokens"])
        payload["tokens"]["seq_len"] = 1024 if max_tok <= 1024 else 2048
    (DATA_ROOT / "build_stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
