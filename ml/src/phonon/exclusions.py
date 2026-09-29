from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXCLUSION_DIR = Path("labels") / "exclusions"
ROW_ID_FILES = ("excluded_ids.txt", "non_english_ids.txt")
SOURCE_ID_FILES = ("excluded_source_ids.txt", "non_english_source_ids.txt")


@dataclass(frozen=True)
class Exclusions:
    row_ids: frozenset[str]
    source_ids: frozenset[str]


def _read_id_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        ids.add(value)
    return ids


def load_exclusions(dataset_root: Path) -> Exclusions:
    exclusion_dir = dataset_root / EXCLUSION_DIR
    row_ids: set[str] = set()
    source_ids: set[str] = set()
    for filename in ROW_ID_FILES:
        row_ids.update(_read_id_file(exclusion_dir / filename))
    for filename in SOURCE_ID_FILES:
        source_ids.update(_read_id_file(exclusion_dir / filename))
    return Exclusions(frozenset(row_ids), frozenset(source_ids))


def is_excluded_row(row: dict[str, Any], exclusions: Exclusions) -> bool:
    row_id = str(row.get("id") or "").strip()
    source_id = str(row.get("source_id") or "").strip()
    return bool((row_id and row_id in exclusions.row_ids) or (source_id and source_id in exclusions.source_ids))


def filter_excluded_rows(rows: list[dict[str, Any]], dataset_root: Path) -> tuple[list[dict[str, Any]], int]:
    exclusions = load_exclusions(dataset_root)
    filtered = [row for row in rows if not is_excluded_row(row, exclusions)]
    return filtered, len(rows) - len(filtered)
