#!/usr/bin/env python3
"""Convert pinned public embedding seed files into one balanced triplet set."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import triplets

REPO_ROOT = Path(__file__).resolve().parents[2]
STSB_MIN_POSITIVE_SCORE = 0.8
SPECS = {
    "all_nli_train": (
        "sentence-transformers/all-nli",
        "d482672c8e74ce18da116f430137434ba2e52fab",
        "all-nli/triplet/train-00000-of-00001.parquet",
        "triplet",
    ),
    "all_nli_dev": (
        "sentence-transformers/all-nli",
        "d482672c8e74ce18da116f430137434ba2e52fab",
        "all-nli/triplet/dev-00000-of-00001.parquet",
        "triplet",
    ),
    "all_nli_test": (
        "sentence-transformers/all-nli",
        "d482672c8e74ce18da116f430137434ba2e52fab",
        "all-nli/triplet/test-00000-of-00001.parquet",
        "triplet",
    ),
    "quora": (
        "sentence-transformers/quora-duplicates",
        "41f699770310302022a4dd75d4cf903bfef9ea46",
        "quora-duplicates/triplet/train-00000-of-00001.parquet",
        "triplet",
    ),
    "stsb": (
        "sentence-transformers/stsb",
        "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308",
        "stsb/data/train-00000-of-00001.parquet",
        "stsb",
    ),
    "sentence_compression": (
        "embedding-data/sentence-compression",
        "57eee282a374a0a4e5993f4a76d5b1d2da184ae8",
        "sentence-compression/sentence-compression_compressed.jsonl.gz",
        "set_pair",
    ),
    "simple_wiki": (
        "embedding-data/simple-wiki",
        "ce219b5f4b856ee0ab2868baccfc94917164d683",
        "simple-wiki/SimpleWiki_processed.jsonl.gz",
        "set_pair",
    ),
    "squad_train": (
        "rajpurkar/squad",
        "7b6d24c440a36b6815f21b70d25016731768db1f",
        "squad/plain_text/train-00000-of-00001.parquet",
        "squad",
    ),
    "squad_validation": (
        "rajpurkar/squad",
        "7b6d24c440a36b6815f21b70d25016731768db1f",
        "squad/plain_text/validation-00000-of-00001.parquet",
        "squad",
    ),
    "code_search_net_python": (
        "code-search-net/code_search_net",
        "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
        "code-search-net-python/python/validation-00000-of-00001.parquet",
        "code",
    ),
}


class SeedError(ValueError):
    """Raised for missing or malformed public seed files."""


def safe_output(path: Path, root: Path) -> Path:
    """Keep public outputs beside their pinned seed inventory."""
    if path.is_symlink():
        raise SeedError("output path must not be a symbolic link")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise SeedError("public output must stay under the seed root") from error
    return resolved


def reservoir(
    items: list[tuple[int, dict[str, Any]]], size: int, seed: str
) -> list[tuple[int, dict[str, Any]]]:
    """Select a deterministic bounded sample without loading text twice."""
    rng = random.Random(seed)
    selected: list[tuple[int, dict[str, Any]]] = []
    for seen, item in enumerate(items):
        if len(selected) < size:
            selected.append(item)
            continue
        index = rng.randrange(seen + 1)
        if index < size:
            selected[index] = item
    return selected


def read_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read a bounded public source into row-index records."""
    rows: list[tuple[int, dict[str, Any]]] = []
    if path.suffix == ".parquet":
        from pyarrow import parquet

        for batch in parquet.ParquetFile(path).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                rows.append((len(rows), row))
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    rows.append((len(rows), json.loads(line)))
    else:
        raise SeedError(f"unsupported seed format: {path}")
    return rows


def eligible_rows(
    rows: list[tuple[int, dict[str, Any]]], kind: str
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Apply source-specific positive-pair quality filters before sampling."""
    if kind != "stsb":
        return rows, 0
    eligible: list[tuple[int, dict[str, Any]]] = []
    for item in rows:
        score = item[1].get("score")
        if (
            isinstance(score, int | float)
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            and float(score) >= STSB_MIN_POSITIVE_SCORE
        ):
            eligible.append(item)
    return eligible, len(rows) - len(eligible)


def cyclic_different_candidate(
    sampled: list[tuple[int, dict[str, Any]]], index: int, field: str
) -> dict[str, Any] | None:
    """Choose a rotated negative while preserving the source grouping rule."""
    for offset in range(1, len(sampled)):
        candidate = sampled[(index + offset) % len(sampled)][1]
        if candidate.get(field) != sampled[index][1].get(field):
            return candidate
    return None


def base_record(
    name: str, source: str, revision: str, relative: str, row_index: int
) -> dict[str, Any]:
    """Build non-text triplet provenance."""
    return {
        "schema_version": triplets.SCHEMA_VERSION,
        "source": source,
        "source_revision": revision,
        "seed_name": name,
        "seed_relative": relative,
        "seed_row_index": row_index,
    }


def convert(
    name: str,
    source: str,
    revision: str,
    relative: str,
    kind: str,
    sampled: list[tuple[int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int]:
    """Convert one sampled source to validated triplets."""
    output: list[dict[str, Any]] = []
    rejected = 0
    for index, (row_index, row) in enumerate(sampled):
        base = base_record(name, source, revision, relative, row_index)
        triplet: dict[str, Any] | None = None
        if kind == "triplet":
            triplet = {
                **base,
                "kind": "public_text",
                "anchor": str(row.get("anchor") or ""),
                "positive": str(row.get("positive") or ""),
                "negative": str(row.get("negative") or ""),
            }
        elif kind == "stsb":
            score = row.get("score")
            if (
                not isinstance(score, int | float)
                or isinstance(score, bool)
                or not math.isfinite(float(score))
                or float(score) < STSB_MIN_POSITIVE_SCORE
            ):
                rejected += 1
                continue
            other = sampled[(index + 1) % len(sampled)][1]
            triplet = {
                **base,
                "kind": "public_text",
                "anchor": str(row.get("sentence1") or ""),
                "positive": str(row.get("sentence2") or ""),
                "negative": str(other.get("sentence2") or ""),
            }
        elif kind == "set_pair":
            values = row.get("set") or []
            other = sampled[(index + 1) % len(sampled)][1].get("set") or []
            if len(values) >= 2 and len(other) >= 2:
                triplet = {
                    **base,
                    "kind": "public_text",
                    "anchor": str(values[0]),
                    "positive": str(values[1]),
                    "negative": str(other[1]),
                }
        elif kind == "squad":
            other = cyclic_different_candidate(sampled, index, "title")
            if other is not None:
                triplet = {
                    **base,
                    "kind": "public_qa",
                    "anchor": str(row.get("question") or ""),
                    "positive": str(row.get("context") or ""),
                    "negative": str(other.get("context") or ""),
                }
        elif kind == "code":
            other = cyclic_different_candidate(sampled, index, "repository_name")
            if other is not None:
                triplet = {
                    **base,
                    "kind": "public_code",
                    "anchor": str(row.get("func_documentation_string") or ""),
                    "positive": str(row.get("func_code_string") or ""),
                    "negative": str(other.get("func_code_string") or ""),
                }
        if triplet is None:
            rejected += 1
            continue
        try:
            triplets.validate_triplet(triplet)
        except triplets.TripletError:
            rejected += 1
            continue
        output.append(triplet)
    return output, rejected


def build_public_seed_set(
    seed_root: Path, output: Path, per_source_limit: int
) -> dict[str, Any]:
    """Build one deduplicated public triplet set and aggregate-only report."""
    if per_source_limit < 1:
        raise SeedError("per-source limit must be positive")
    safe_output(output, seed_root)
    records: list[dict[str, Any]] = []
    rejected = Counter()
    quality_filtered = Counter()
    file_hashes: dict[str, str] = {}
    for name, (source, revision, relative, kind) in SPECS.items():
        path = seed_root / relative
        if not path.is_file():
            raise SeedError(f"missing pinned seed: {path}")
        file_hashes[name] = triplets.sha256_file(path)
        eligible, filtered = eligible_rows(read_rows(path), kind)
        quality_filtered[name] = filtered
        sampled = reservoir(eligible, per_source_limit, f"{name}:{revision}")
        converted, invalid = convert(name, source, revision, relative, kind, sampled)
        records.extend(converted)
        rejected[name] = invalid
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in records:
        key = (row["anchor"], row["positive"], row["negative"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    triplets.write_jsonl(output, unique, private=False)
    return {
        "schema_version": triplets.SCHEMA_VERSION,
        "seed_root": str(seed_root),
        "output": str(output),
        "per_source_limit": per_source_limit,
        "source_count": len(SPECS),
        "source_counts": dict(sorted(Counter(r["seed_name"] for r in unique).items())),
        "kind_counts": dict(sorted(Counter(r["kind"] for r in unique).items())),
        "written": len(unique),
        "invalid": dict(sorted(rejected.items())),
        "quality_filtered": dict(sorted(quality_filtered.items())),
        "duplicate_triplets_removed": len(records) - len(unique),
        "file_sha256": dict(sorted(file_hashes.items())),
        "emits_text": True,
        "emits_private_data": False,
    }


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-source-limit", type=int, default=1000)
    return parser.parse_args(argv)


def validate_output_aliases(
    seed_root: Path, output: Path, report: Path
) -> tuple[Path, Path]:
    """Reject output collisions before a pinned seed or result can be replaced."""
    resolved_output = safe_output(output, seed_root)
    resolved_report = safe_output(report, seed_root)
    source_paths = {
        (seed_root / relative).resolve() for _, (_, _, relative, _) in SPECS.items()
    }
    if (
        resolved_output == resolved_report
        or resolved_output in source_paths
        or resolved_report in source_paths
    ):
        raise SeedError(
            "public outputs must be distinct and must not overwrite pinned seeds"
        )
    return resolved_output, resolved_report


def main(argv: list[str] | None = None) -> int:
    """Convert pinned public seeds and print an aggregate-only report."""
    try:
        arguments = parse_arguments(argv if argv is not None else sys.argv[1:])
        output, report_path = validate_output_aliases(
            arguments.seed_root, arguments.out, arguments.report
        )
        report = build_public_seed_set(
            arguments.seed_root, output, arguments.per_source_limit
        )
        triplets.write_json(report_path, report, private=False)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - public-data CLI failure boundary
        print(f"FAIL public-seeds: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
