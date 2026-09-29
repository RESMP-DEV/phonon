"""Build a date-honest split of the Wispr corrector pairs.

Cutoff = first of the last 15 percent of dictation days (2026-09-04, 11 of 72 days).
Everything strictly before the cutoff may train; everything on or after is holdout.

Unlike train.jsonl (id-hash split) this filters the Parakeet-of-real-audio rows by
date too: they carry the same timestamps as their source dictations, so they are NOT
date-free synthetic data.

Writes /data/phonon_corrector_v0/datesplit/{train_date,holdout_date_text,
holdout_date_audio}.jsonl plus curve/train_date_first{250,500,1000}.jsonl and
datesplit_counts.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

sys.path.insert(0, "/home/user/phonon/research/corrector_v0")
from common import (  # noqa: E402
    DATA_ROOT,
    PARAKEET_JSONL,
    TRAIN_JSONL,
    chat_messages,
    read_jsonl,
    load_wispr_pairs,
    write_jsonl,
)

OUT = DATA_ROOT / "datesplit"
CURVE = OUT / "curve"


def day_of(row: dict) -> str | None:
    ts = row.get("timestamp") or ""
    return ts[:10] if len(ts) >= 10 else None


def add_pair(bucket: dict, raw: str, target: str, source: str, row_id: str, ts: str) -> bool:
    raw = (raw or "").strip()
    target = (target or "").strip()
    if not raw or not target:
        return False
    key = (raw, target)
    if key in bucket:
        return False
    bucket[key] = {
        "id": row_id,
        "source": source,
        "input": raw,
        "target": target,
        "ts": ts,
        "messages": chat_messages(raw, target),
    }
    return True


def main() -> int:
    pairs = load_wispr_pairs()
    for r in pairs:
        r["day"] = day_of(r)
    all_days = sorted({r["day"] for r in pairs if r["day"]})
    k = max(1, int(round(0.15 * len(all_days))))
    cut_days = set(all_days[-k:])
    cutoff = all_days[-k]
    print(f"days={len(all_days)} holdout_days={k} cutoff={cutoff} "
          f"range={all_days[-k]}..{all_days[-1]}", flush=True)

    parakeet = {r["id"]: r for r in read_jsonl(PARAKEET_JSONL)}

    before = [r for r in pairs if r["day"] and r["day"] not in cut_days]
    after = [r for r in pairs if r["day"] in cut_days]

    # ---- train_date -------------------------------------------------------
    bucket: dict = {}
    n_wispr = n_pk = 0
    for r in sorted(before, key=lambda r: (r["timestamp"], r["id"])):
        if add_pair(bucket, r.get("asr"), r.get("target"),
                    f"wispr_asr:{r.get('split')}", r["id"], r["timestamp"]):
            n_wispr += 1
    # Parakeet-raw form of the same pre-cutoff audio clips (train.jsonl has 519 of these).
    for r in sorted(before, key=lambda r: (r["timestamp"], r["id"])):
        if not r.get("has_audio"):
            continue
        pk = parakeet.get(r["id"])
        if not pk:
            continue
        if add_pair(bucket, pk.get("parakeet_raw"), r.get("target"),
                    f"parakeet:{r.get('split')}", f"pk_{r['id']}", r["timestamp"]):
            n_pk += 1
    train_rows = list(bucket.values())
    # time order; parakeet duplicates of the same clip sort after all wispr rows so the
    # first-N subsets are the earliest N real dictations (same convention as make_curve.py)
    train_rows.sort(key=lambda r: (r["source"].startswith("parakeet"), r["ts"], r["id"]))

    # ---- holdouts ---------------------------------------------------------
    # Every post-cutoff row has audio (Wispr keeps wavs only for recent dictations),
    # so there is no text-only tail. The two holdouts are the same rows in the two
    # input modes: Parakeet raw (audio path, analogue of wispr_holdout120) and
    # Wispr ASR text (text path, analogue of wispr_text_holdout).
    def base_id(x: str) -> str:
        return x[3:] if x.startswith("pk_") else x

    train_inputs = {r["input"].strip() for r in train_rows}
    train_targets = {r["target"].strip() for r in train_rows}
    train_ids = {base_id(r["id"]) for r in train_rows}

    h_text, h_audio, h_audio_missing, dropped_dup = [], [], 0, 0
    for r in sorted(after, key=lambda r: (r["timestamp"], r["id"])):
        raw_asr = (r.get("asr") or "").strip()
        target = (r.get("target") or "").strip()
        if not target or not raw_asr:
            continue
        pk = parakeet.get(r["id"])
        pk_raw = ((pk or {}).get("parakeet_raw") or "").strip()
        if not pk_raw:
            h_audio_missing += 1
            continue
        # a verbatim repeat of a pre-cutoff utterance is not a fair holdout row
        if (raw_asr in train_inputs or pk_raw in train_inputs
                or target in train_targets or r["id"] in train_ids):
            dropped_dup += 1
            continue
        base = {"id": r["id"], "reference": target, "wispr_asr": raw_asr,
                "ts": r["timestamp"], "old_split": r.get("split")}
        h_audio.append({**base, "input": pk_raw})
        h_text.append({**base, "input": raw_asr})

    # ---- overlap checks ---------------------------------------------------
    checks = {}
    for name, rows in (("holdout_date_text", h_text), ("holdout_date_audio", h_audio)):
        ids = {r["id"] for r in rows}
        checks[name] = {
            "n": len(rows),
            "id_overlap_with_train": len(ids & train_ids),
            "input_text_overlap_with_train_inputs": len({r["input"].strip() for r in rows} & train_inputs),
            "reference_overlap_with_train_targets": len({r["reference"].strip() for r in rows} & train_targets),
            "date_range": [rows[0]["ts"][:19], rows[-1]["ts"][:19]] if rows else None,
            "days": len({r["ts"][:10] for r in rows}),
            "from_old_holdout120": sum(1 for r in rows if r["old_split"] == "wispr_holdout120"),
            "from_old_train_splits": sum(1 for r in rows if r["old_split"] in ("wispr_train", "wispr_text_train")),
            "from_old_text_holdout": sum(1 for r in rows if r["old_split"] == "wispr_text_holdout"),
        }

    # what changed vs the id-split train.jsonl
    old_train = read_jsonl(TRAIN_JSONL)
    old_ids = {base_id(r["id"]) for r in old_train}
    after_ids = {r["id"] for r in after}
    old_leak = len(old_ids & after_ids)

    OUT.mkdir(parents=True, exist_ok=True)
    CURVE.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "train_date.jsonl", train_rows)
    write_jsonl(OUT / "holdout_date_text.jsonl", h_text)
    write_jsonl(OUT / "holdout_date_audio.jsonl", h_audio)
    for n in (250, 500, 1000):
        sub = train_rows[:n]
        write_jsonl(CURVE / f"train_date_first{n}.jsonl", sub)
        print(f"train_date_first{n}: n={len(sub)} span {sub[0]['ts'][:10]} -> {sub[-1]['ts'][:10]} "
              f"parakeet_rows={sum(1 for r in sub if r['source'].startswith('parakeet'))}", flush=True)

    counts = {
        "cutoff": cutoff,
        "total_days": len(all_days),
        "holdout_days": k,
        "holdout_day_range": [all_days[-k], all_days[-1]],
        "export_pairs": len(pairs),
        "export_pairs_before_cutoff": len(before),
        "export_pairs_on_or_after_cutoff": len(after),
        "train_date": {
            "n": len(train_rows),
            "wispr_asr_rows": n_wispr,
            "parakeet_rows": n_pk,
            "date_range": [min(r["ts"] for r in train_rows)[:19], max(r["ts"] for r in train_rows)[:19]],
            "days": len({r["ts"][:10] for r in train_rows}),
            "sources": dict(Counter(r["source"] for r in train_rows)),
        },
        "curve_subsets": {
            f"n{n}": {
                "n": len(train_rows[:n]),
                "span": [train_rows[0]["ts"][:10], train_rows[min(n, len(train_rows)) - 1]["ts"][:10]],
            }
            for n in (250, 500, 1000)
        },
        "holdouts": checks,
        "holdout_note": ("every post-cutoff export row has audio, so holdout_date_text and "
                         "holdout_date_audio are the same rows in two input modes: "
                         "Wispr ASR text and Parakeet raw"),
        "holdout_audio_rows_dropped_no_parakeet": h_audio_missing,
        "holdout_rows_dropped_as_verbatim_repeats_of_train": dropped_dup,
        "id_split_train_jsonl": {
            "n": len(old_train),
            "sources": dict(Counter(r["source"] for r in old_train)),
            "rows_on_or_after_cutoff (leak under a date split)": old_leak,
        },
    }
    (OUT / "datesplit_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2), flush=True)
    bad = sum(c["id_overlap_with_train"] + c["input_text_overlap_with_train_inputs"]
              for c in checks.values())
    print(f"OVERLAP_TOTAL={bad}", flush=True)
    print("STEP 1 done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
