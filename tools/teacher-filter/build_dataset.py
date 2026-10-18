#!/usr/bin/env python3
"""Build a tiered Phonon training manifest from three raw-audio ASR teachers."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import statistics
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_TEACHER_ORDER = ("cohere", "voxtral", "qwen")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    value = (text or "").lower()
    return " ".join(re.sub(r"[^a-z0-9']+", " ", value).split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    previous = list(range(len(hyp) + 1))
    for index, expected in enumerate(ref, 1):
        current = [index] + [0] * len(hyp)
        last = previous[0]
        current[0] = index
        for position, actual in enumerate(hyp, 1):
            before_delete = previous[position]
            before_insert = current[position - 1]
            before_substitute = last
            current[position] = min(
                before_delete + 1,
                before_insert + 1,
                before_substitute + (expected != actual),
            )
            last = before_delete
        previous = current
    return previous[-1] / len(ref)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audio_stem(value: object) -> str:
    return Path(str(value or "")).stem


def derived_segment_stem(value: object) -> str:
    stem = audio_stem(value)
    parts = stem.split("-")
    return parts[-1] if len(parts) > 1 and parts[-1].isdigit() else stem


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohere", type=Path, required=True)
    parser.add_argument("--voxtral", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--auto-max-mean-wer", type=float, default=0.10)
    parser.add_argument("--review-max-mean-wer", type=float, default=0.40)
    parser.add_argument("--dataset-id", default="parallel-isolation-v2")
    parser.add_argument(
        "--derived-arm",
        action="append",
        default=[],
        choices=("rnnoise-mp", "rnnoise-bd", "dfn-pf-atten6"),
        help="add eligible rows to a separate derived-audio augmentation manifest",
    )
    parser.add_argument(
        "--derived-audio-root",
        type=Path,
        default=None,
        help="root containing <arm>/audio/<arm>-NNNN.wav",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.auto_max_mean_wer <= args.review_max_mean_wer:
        parser.error("require 0 <= auto threshold <= review threshold")
    if args.derived_arm and not args.derived_audio_root:
        parser.error("--derived-arm requires --derived-audio-root")
    return args


def validate_teacher_rows(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    teachers = {
        "cohere": read_jsonl(args.cohere),
        "voxtral": read_jsonl(args.voxtral),
        "qwen": read_jsonl(args.qwen),
    }
    slice_rows = read_jsonl(args.slice)
    expected_count = len(slice_rows)
    if not expected_count:
        raise ValueError("empty audio slice")
    for name, rows in teachers.items():
        if len(rows) != expected_count:
            raise ValueError(
                f"{name} row count {len(rows)} does not match slice count {expected_count}"
            )
        for index, (teacher_row, slice_row) in enumerate(
            zip(rows, slice_rows, strict=True)
        ):
            if audio_stem(teacher_row.get("audio")) != audio_stem(
                slice_row.get("audio")
            ):
                raise ValueError(f"{name} row {index} audio key mismatch")
    return teachers


def choose_label(texts: dict[str, str]) -> tuple[str, str, float]:
    groups: dict[str, list[str]] = {}
    for teacher, text in texts.items():
        groups.setdefault(normalize(text), []).append(teacher)
    majority = max(groups.values(), key=lambda members: len(members))
    if len(majority) >= 2:
        preferred = next(
            (name for name in DEFAULT_TEACHER_ORDER if name in majority), majority[0]
        )
        return texts[preferred], f"majority:{preferred}", len(majority) / len(texts)

    scores: dict[str, float] = {}
    for teacher, text in texts.items():
        others = [other for name, other in texts.items() if name != teacher]
        scores[teacher] = statistics.mean(
            word_error_rate(text, other) for other in others
        )
    preferred = min(DEFAULT_TEACHER_ORDER, key=lambda name: scores[name])
    return texts[preferred], f"medoid:{preferred}", 1 / len(texts)


def build_rows(
    args: argparse.Namespace,
    teachers: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    slice_rows = read_jsonl(args.slice)
    rows: list[dict[str, Any]] = []
    for index, slice_row in enumerate(slice_rows):
        texts = {name: str(teachers[name][index].get("hyp", "")) for name in teachers}
        pairwise = [
            word_error_rate(texts[left], texts[right])
            for left, right in itertools.combinations(sorted(texts), 2)
        ]
        mean_wer = statistics.mean(pairwise)
        max_wer = max(pairwise)
        empty_count = sum(not normalize(text) for text in texts.values())
        if empty_count >= 2 or mean_wer <= args.auto_max_mean_wer:
            tier = "auto"
        elif mean_wer <= args.review_max_mean_wer:
            tier = "review"
        else:
            tier = "exclude"
        label, label_source, agreement = (
            choose_label(texts) if tier == "auto" else ("", "", 0.0)
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": args.dataset_id,
                "id": f"{args.dataset_id}-{index:04d}",
                "audio": str(slice_row["audio"]),
                "duration": float(
                    slice_row.get("dur") or slice_row.get("duration") or 0
                ),
                "tier": tier,
                "label": label,
                "label_source": label_source,
                "teacher_agreement": round(agreement, 4),
                "mean_pairwise_wer": round(mean_wer, 6),
                "max_pairwise_wer": round(max_wer, 6),
                "empty_teacher_count": empty_count,
                "teacher_transcripts": texts,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    teachers = validate_teacher_rows(args)
    rows = build_rows(args, teachers)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "dataset-manifest.jsonl", rows)
    write_jsonl(
        args.out / "pack-manifest.jsonl",
        [
            {
                "audio": row["audio"],
                "corrected": row["label"],
                "duration": row["duration"],
                "label_source": row["label_source"],
            }
            for row in rows
            if row["tier"] == "auto"
        ],
    )
    write_jsonl(
        args.out / "review-manifest.jsonl",
        [row for row in rows if row["tier"] == "review"],
    )
    write_jsonl(
        args.out / "exclude-manifest.jsonl",
        [row for row in rows if row["tier"] == "exclude"],
    )
    derived_rows: list[dict[str, Any]] = []
    if args.derived_arm:
        for row in rows:
            if row["tier"] != "auto":
                continue
            source_stem = derived_segment_stem(row["audio"])
            for arm in args.derived_arm:
                derived_rows.append(
                    {
                        "audio": f"{arm}/audio/{arm}-{source_stem}.wav",
                        "corrected": row["label"],
                        "duration": row["duration"],
                        "label_source": row["label_source"],
                        "source_audio": row["audio"],
                        "arm": arm,
                    }
                )
        write_jsonl(args.out / "derived-augmentation-manifest.jsonl", derived_rows)

    def summarize(tier: str) -> dict[str, float]:
        selected = [row for row in rows if row["tier"] == tier]
        return {
            "rows": len(selected),
            "seconds": round(sum(row["duration"] for row in selected), 3),
            "percent_rows": round(100 * len(selected) / len(rows), 2),
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": args.dataset_id,
        "audio_root": str(args.audio_root.resolve()),
        "thresholds": {
            "auto_max_mean_wer": args.auto_max_mean_wer,
            "review_max_mean_wer": args.review_max_mean_wer,
        },
        "teachers": {
            name: {
                "path": str(getattr(args, name).resolve()),
                "sha256": sha256(getattr(args, name)),
            }
            for name in teachers
        },
        "tiers": {tier: summarize(tier) for tier in ("auto", "review", "exclude")},
        "derived_augmentation_rows": len(derived_rows),
        "derived_arms": args.derived_arm,
        "non_claims": [
            "Auto labels are three-teacher consensus labels, not human ground truth.",
            "Derived denoising arms are not included in the default pack manifest.",
            "This manifest does not by itself qualify a model or preprocessing variant.",
        ],
    }
    (args.out / "filter-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out": str(args.out), "tiers": report["tiers"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
