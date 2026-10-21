#!/usr/bin/env python3
"""Manifest normalization and group-disjoint train/eval split construction.

Source records are JSONL objects carrying ``row_id``, ``audio_sha256``,
``duration_seconds``, ``leakage_group``, ``source_kind``, and ``source_id``.
Audio bytes arrive either as an extracted ``audio_path`` or inline as
``audio_bytes_base64``; extraction itself is an upstream responsibility.

The builder assigns whole leakage groups to one side only, so no group can
appear in both train and eval, and it reaches exact millisecond duration
targets by choosing whole groups.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

REQUIRED_RECORD_FIELDS = (
    "row_id",
    "audio_sha256",
    "duration_seconds",
    "leakage_group",
    "source_kind",
    "source_id",
)

# Audio location alternatives. ``audio_path`` points at bytes extracted
# upstream; ``audio_bytes_base64`` carries the same bytes inline.
AUDIO_LOCATION_FIELDS = ("audio_path", "audio_bytes_base64")

DECLARED_SPLITS = ("train", "eval", "test")

# Guard against a pathological subset-sum search. Reachable millisecond
# sums are unique, so the state count is bounded by the target; 24 hours
# of audio is 86.4 million milliseconds.
MAX_SEARCH_STATES = 100_000_000


class ManifestError(ValueError):
    """Raised for invalid source records or unsatisfiable split targets."""


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _require_string(record: dict[str, Any], field: str, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} requires a nonempty string {field}")
    return value.strip()


def _require_number(record: dict[str, Any], field: str, label: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} requires a numeric {field}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ManifestError(f"{label} requires a positive finite {field}")
    return number


def normalize_record(record: dict[str, Any], label: str) -> dict[str, Any]:
    """Validate one source record and return its normalized manifest form."""
    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            raise ManifestError(f"{label} is missing {field}")
    row_id = _require_string(record, "row_id", label)
    label = f"row {row_id}"
    audio_sha256 = _require_string(record, "audio_sha256", label).lower()
    if len(audio_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in audio_sha256
    ):
        raise ManifestError(f"{label} requires a 64-hex audio_sha256")
    duration_seconds = _require_number(record, "duration_seconds", label)
    leakage_group = _require_string(record, "leakage_group", label)
    source_kind = _require_string(record, "source_kind", label)
    source_id = _require_string(record, "source_id", label)

    locations = [field for field in AUDIO_LOCATION_FIELDS if record.get(field)]
    if not locations:
        raise ManifestError(
            f"{label} requires audio_path or audio_bytes_base64; audio "
            "extraction is handled upstream"
        )

    normalized: dict[str, Any] = {
        "row_id": row_id,
        "audio_sha256": audio_sha256,
        "duration_seconds": duration_seconds,
        "duration_milliseconds": round(duration_seconds * 1000.0),
        "leakage_group": leakage_group,
        "source_kind": source_kind,
        "source_id": source_id,
    }
    if record.get("audio_path"):
        normalized["audio_path"] = str(record["audio_path"])
    if record.get("audio_bytes_base64"):
        normalized["audio_bytes_base64"] = str(record["audio_bytes_base64"])
    if record.get("split"):
        normalized["declared_split"] = str(record["split"]).strip().lower()
    if record.get("domain"):
        normalized["domain"] = str(record["domain"])
    return normalized


def load_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load and normalize every record from the given manifest files."""
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for path in paths:
        for index, record in enumerate(load_jsonl(path), start=1):
            item = normalize_record(record, f"{path.name} record {index}")
            row_id = item["row_id"]
            if row_id in seen_ids:
                raise ManifestError(f"duplicate row_id: {row_id}")
            seen_ids.add(row_id)
            previous = seen_hashes.get(item["audio_sha256"])
            if previous is not None:
                raise ManifestError(
                    f"row {row_id} repeats the audio hash of row {previous}"
                )
            seen_hashes[item["audio_sha256"]] = row_id
            normalized.append(item)
    if not normalized:
        raise ManifestError("no manifest records were found")
    return normalized


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records, reporting the line number of any malformed row."""
    records: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ManifestError(f"cannot read manifest {path}: {error}") from error
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ManifestError(
                f"{path}:{number} is not valid JSON: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ManifestError(f"{path}:{number} is not a JSON object")
        records.append(record)
    return records


def group_rows(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group records by leakage group in a deterministic order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["leakage_group"], []).append(record)
    for rows in groups.values():
        rows.sort(key=lambda item: item["row_id"])
    return dict(sorted(groups.items()))


def declared_split_of(rows: Sequence[dict[str, Any]]) -> str | None:
    """Return the split declared by every row of one group, if any."""
    declared = {row.get("declared_split") for row in rows if row.get("declared_split")}
    if not declared:
        return None
    if len(declared) > 1:
        raise ManifestError(
            "one leakage group declares conflicting splits: "
            + ", ".join(sorted(declared))
        )
    split = declared.pop()
    if split not in DECLARED_SPLITS:
        raise ManifestError(
            f"unsupported declared split {split!r}; expected one of "
            + ", ".join(DECLARED_SPLITS)
        )
    return split


def solve_subset(items: Sequence[tuple[str, int]], target: int) -> list[str] | None:
    """Return item names summing exactly to ``target``, or None.

    The search is a bounded subset-sum dynamic program with parent tracking.
    Each item is used at most once because the reachable sums are snapshotted
    before a new item contributes.
    """
    if target == 0:
        return []
    if target < 0:
        return None
    parents: dict[int, tuple[int, int]] = {0: (-1, -1)}
    for index, (_, value) in enumerate(items):
        for current in list(parents):
            total = current + value
            if total <= target and total not in parents:
                parents[total] = (current, index)
                if total == target:
                    break
        if target in parents:
            break
        if len(parents) > MAX_SEARCH_STATES:
            raise ManifestError(
                "the split search exceeded its state budget; declare explicit "
                "splits or use fewer leakage groups"
            )
    if target not in parents:
        return None
    chosen: list[str] = []
    node = target
    while node != 0:
        previous, index = parents[node]
        chosen.append(items[index][0])
        node = previous
    chosen.reverse()
    return chosen


def assign_splits(
    groups: dict[str, list[dict[str, Any]]],
    *,
    train_target_ms: int,
    eval_target_ms: int,
) -> dict[str, str]:
    """Assign whole leakage groups to train/eval at exact duration targets.

    Groups that declare a split are pinned to it. Groups without a declared
    split are assigned so that the train side reaches its exact millisecond
    target; the eval side then holds the remaining non-test duration and must
    match its own exact target.
    """
    if train_target_ms < 0 or eval_target_ms < 0:
        raise ManifestError("split duration targets must not be negative")

    pinned: dict[str, str] = {}
    free: list[tuple[str, int]] = []
    for name, rows in groups.items():
        split = declared_split_of(rows)
        if split is None:
            free.append((name, sum(row["duration_milliseconds"] for row in rows)))
        else:
            pinned[name] = split

    pinned_train_ms = sum(
        sum(row["duration_milliseconds"] for row in groups[name])
        for name, split in pinned.items()
        if split == "train"
    )
    pinned_eval_ms = sum(
        sum(row["duration_milliseconds"] for row in groups[name])
        for name, split in pinned.items()
        if split == "eval"
    )
    free_ms = sum(value for _, value in free)
    if pinned_train_ms + pinned_eval_ms + free_ms != train_target_ms + eval_target_ms:
        raise ManifestError(
            "total non-test duration does not equal the train plus eval target"
        )

    remaining = train_target_ms - pinned_train_ms
    if remaining < 0:
        raise ManifestError(
            "pinned train groups already exceed the train duration target"
        )
    chosen = solve_subset(free, remaining)
    if chosen is None:
        raise ManifestError(
            "no assignment of whole leakage groups reaches the train duration "
            "target exactly"
        )
    chosen_set = set(chosen)
    assignment = dict(pinned)
    for name, _ in free:
        assignment[name] = "train" if name in chosen_set else "eval"

    train_ms = sum(
        sum(row["duration_milliseconds"] for row in groups[name])
        for name, split in assignment.items()
        if split == "train"
    )
    eval_ms = sum(
        sum(row["duration_milliseconds"] for row in groups[name])
        for name, split in assignment.items()
        if split == "eval"
    )
    if train_ms != train_target_ms or eval_ms != eval_target_ms:
        raise ManifestError(
            "split durations did not reach the exact train and eval targets"
        )
    return assignment


def build_manifest(
    records: Sequence[dict[str, Any]],
    *,
    train_target_ms: int,
    eval_target_ms: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the group-disjoint manifest and its aggregate-only report."""
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        normalized.append(normalize_record(record, f"record {index}"))
    groups = group_rows(normalized)
    assignment = assign_splits(
        groups, train_target_ms=train_target_ms, eval_target_ms=eval_target_ms
    )
    manifest: list[dict[str, Any]] = []
    for name, rows in groups.items():
        split = assignment[name]
        for row in rows:
            entry = dict(row)
            entry["split"] = split
            manifest.append(entry)
    manifest.sort(key=lambda item: (item["split"], item["row_id"]))

    train_groups = {name for name, split in assignment.items() if split == "train"}
    eval_groups = {name for name, split in assignment.items() if split == "eval"}
    if train_groups & eval_groups:
        raise ManifestError("leakage groups overlap between train and eval")

    by_group: dict[str, set[str]] = {}
    for entry in manifest:
        by_group.setdefault(entry["leakage_group"], set()).add(entry["split"])
    for name, splits in by_group.items():
        if len(splits) != 1:
            raise ManifestError(f"leakage group {name} spans multiple splits")

    train_ms = sum(
        row["duration_milliseconds"] for row in manifest if row["split"] == "train"
    )
    eval_ms = sum(
        row["duration_milliseconds"] for row in manifest if row["split"] == "eval"
    )
    if train_ms != train_target_ms or eval_ms != eval_target_ms:
        raise ManifestError(
            "split durations did not reach the exact train and eval targets"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": utc_now(),
        "rows": len(manifest),
        "leakage_groups": len(groups),
        "train_rows": sum(1 for row in manifest if row["split"] == "train"),
        "eval_rows": sum(1 for row in manifest if row["split"] == "eval"),
        "test_rows": sum(1 for row in manifest if row["split"] == "test"),
        "train_groups": len(train_groups),
        "eval_groups": len(eval_groups),
        "train_duration_milliseconds": train_ms,
        "eval_duration_milliseconds": eval_ms,
        "test_duration_milliseconds": sum(
            row["duration_milliseconds"] for row in manifest if row["split"] == "test"
        ),
        "train_target_milliseconds": train_target_ms,
        "eval_target_milliseconds": eval_target_ms,
        "source_kind_counts": dict(
            sorted(Counter(row["source_kind"] for row in manifest).items())
        ),
        "leakage_groups_disjoint": True,
        "exact_duration_targets_met": True,
    }
    return manifest, report


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write records to a JSONL file in a stable key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
