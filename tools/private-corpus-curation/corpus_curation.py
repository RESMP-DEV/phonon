#!/usr/bin/env python3
"""Build privacy-preserving fingerprints, deduplication, and grouped splits.

The tool may read private Aqua dictionary terms and private canonical Phonon
Parquets, but tracked/public output must be aggregate-only. Row-level split
manifests and full lexicon files remain in the local research share.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import time_ns
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
LABEL_FIELDS = ("insert_text", "normalized_text", "verbatim_text")
DEFAULT_SPLIT_FRACTIONS = {"train": 0.8, "dev": 0.1, "test": 0.1}
SUPPORTED_DIMENSIONS = (128, 256, 512, 768)
EMBEDDING_DIMENSION = 768
RESEARCH_ROOT = Path.home() / ".local" / "share" / "phonon-research"


class CurationError(ValueError):
    """Raised for invalid curation inputs or unsafe output paths."""


def normalize_phrase(value: Any) -> str:
    """Normalize terms while preserving common technical-token punctuation."""
    return " ".join(
        token
        for token in re.findall(
            r"[a-z0-9]+(?:[+#._/-]+[a-z0-9]+)*[+#]*",
            str(value or "").lower(),
        )
        if token
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_aqua_lexicon(settings_path: Path) -> dict[str, Any]:
    """Read Aqua dictionary/replacement values without printing them."""
    try:
        settings_text = settings_path.read_text(encoding="utf-8")
        settings = json.loads(settings_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CurationError(f"cannot read Aqua settings: {error}") from error
    dictionary = settings.get("dictionary")
    replacements = settings.get("replacements")
    if not isinstance(dictionary, list) or not isinstance(replacements, list):
        raise CurationError("Aqua dictionary and replacements must be arrays")

    by_phrase: dict[str, dict[str, Any]] = {}
    for value in dictionary:
        phrase = str(value or "").strip()
        normalized = normalize_phrase(phrase)
        if not normalized:
            continue
        entry = by_phrase.setdefault(
            normalized, {"phrase": phrase, "normalized": normalized, "sources": []}
        )
        entry["sources"].append("dictionary")
    for item in replacements:
        if not isinstance(item, dict) or not isinstance(item.get("to"), str):
            raise CurationError("each Aqua replacement must contain string from/to")
        phrase = item["to"].strip()
        normalized = normalize_phrase(phrase)
        if not normalized:
            continue
        entry = by_phrase.setdefault(
            normalized, {"phrase": phrase, "normalized": normalized, "sources": []}
        )
        entry["sources"].append("replacement")
        entry["spoken_source"] = item.get("from")

    terms = sorted(by_phrase.values(), key=lambda entry: entry["normalized"])
    for entry in terms:
        entry["sources"] = sorted(set(entry["sources"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "aqua_voice_settings",
        "settings_path": str(settings_path),
        "settings_sha256": sha256_text(settings_text),
        "terms": terms,
    }


def lexicon_aggregate(lexicon: dict[str, Any]) -> dict[str, Any]:
    """Return a public-safe lexicon summary with no phrase text."""
    terms = lexicon.get("terms", [])
    sources = Counter(source for term in terms for source in term.get("sources", []))
    normalized = [term["normalized"] for term in terms]
    return {
        "term_count": len(terms),
        "source_counts": dict(sorted(sources.items())),
        "normalized_count": len(normalized),
        "emits_phrases": False,
    }


def write_private_json(path: Path, value: Any) -> None:
    """Write private JSON locally and constrain permissions to this user."""
    safe_output_path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)
    path.chmod(0o600)


def safe_output_path(path: Path) -> Path:
    """Permit only local research-share or ignored repository build outputs."""
    if path.is_symlink():
        raise CurationError("output path must not be a symbolic link")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        if relative.parts and relative.parts[0] == "build":
            reject_symlink_ancestors(path.absolute(), REPO_ROOT / "build")
            return resolved
        raise CurationError("repository output must stay under build/")
    try:
        resolved.relative_to(RESEARCH_ROOT.resolve())
    except ValueError as error:
        raise CurationError(
            "output must be under the local Phonon research share or build/"
        ) from error
    reject_symlink_ancestors(path.absolute(), RESEARCH_ROOT)
    return resolved


def reject_symlink_ancestors(path: Path, root: Path) -> None:
    """Reject symlinked output components inside a private output root."""
    current = path
    while current != root:
        if current.is_symlink():
            raise CurationError(f"symlinked output component: {current}")
        if root not in current.parents:
            raise CurationError("output path is outside its declared root")
        current = current.parent
    if root.is_symlink():
        raise CurationError("private output root must not be a symbolic link")


def validate_output_aliases(output: Path, protected: Iterable[Path]) -> None:
    """Prevent an output from replacing a private input path."""
    resolved_output = output.resolve()
    for candidate in protected:
        resolved_candidate = candidate.resolve()
        if (
            resolved_output == resolved_candidate
            or resolved_candidate in resolved_output.parents
            or resolved_output in resolved_candidate.parents
        ):
            raise CurationError("output path must not overwrite an input")


def iter_parquet_rows(root: Path) -> Iterator[dict[str, Any]]:
    """Yield metadata rows from one canonical dataset mirror."""
    try:
        from pyarrow import parquet
    except ImportError as error:
        raise CurationError(
            "pyarrow is required; run with `uv run --with pyarrow`"
        ) from error
    paths = sorted((root / "data").glob("*.parquet"))
    if not paths:
        raise CurationError(f"no Parquet files under {root / 'data'}")
    columns = [
        "id",
        "audio",
        "duration_seconds",
        "sha256_audio",
        "domain_tags",
        "difficulty_tags",
        "leakage_group",
        "split",
        "train_allowed",
        "eval_allowed",
        "label_status",
        "teacher_transcripts",
        *LABEL_FIELDS,
    ]
    for path in paths:
        source = parquet.ParquetFile(path)
        names = set(source.schema_arrow.names)
        missing = [name for name in columns if name not in names]
        if missing:
            raise CurationError(f"{path.name} lacks columns: {', '.join(missing)}")
        for batch in source.iter_batches(columns=columns, batch_size=256):
            for row in batch.to_pylist():
                row["_dataset"] = root.name
                yield row


def reference_text(row: dict[str, Any]) -> str:
    for field in LABEL_FIELDS:
        if isinstance(row.get(field), str) and row[field].strip():
            return " ".join(row[field].split())
    return ""


def term_counts(text: str, terms: Sequence[dict[str, Any]]) -> list[float]:
    """Count normalized dictionary phrases without emitting those phrases."""
    normalized = normalize_phrase(text)
    if not normalized:
        return [0.0] * len(terms)
    tokens = normalized.split()
    token_ngrams: dict[int, Counter[str]] = {}
    counts: list[float] = []
    for term in terms:
        needle = term["normalized"]
        if not needle:
            counts.append(0.0)
            continue
        width = len(needle.split())
        if width == 1:
            counts.append(float(tokens.count(needle)))
        else:
            ngrams = token_ngrams.setdefault(width, Counter())
            if not ngrams:
                ngrams.update(
                    " ".join(tokens[index : index + width])
                    for index in range(len(tokens) - width + 1)
                )
            counts.append(float(ngrams[needle]))
    return counts


def load_embeddings(path: Path | None) -> dict[str, dict[str, list[float]]]:
    """Load row embeddings keyed by row ID and modality."""
    result: dict[str, dict[str, list[float]]] = {}
    if path is None:
        return result
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CurationError(f"cannot read embeddings: {error}") from error
    dimensions: dict[str, int] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CurationError(f"embedding line {line_number} is invalid") from error
        if not isinstance(row, dict):
            raise CurationError(f"embedding line {line_number} is not an object")
        vector = row.get("vector")
        modalities: list[tuple[str, list[Any]]] = []
        if isinstance(vector, list):
            modalities.append((str(row.get("modality")), vector))
        for modality in ("text", "audio"):
            nested = row.get(modality)
            if isinstance(nested, list):
                modalities.append((modality, nested))
        if not modalities:
            raise CurationError(f"embedding line {line_number} lacks a vector")
        if not isinstance(row.get("id"), str) or any(
            modality not in {"text", "audio"}
            or not values
            or not all(
                isinstance(value, int | float) and math.isfinite(float(value))
                for value in values
            )
            for modality, values in modalities
        ):
            raise CurationError(f"embedding line {line_number} is malformed")
        for modality, values in modalities:
            dimension = len(values)
            previous = dimensions.setdefault(modality, dimension)
            if previous != dimension:
                raise CurationError(
                    f"embedding line {line_number} mixes {modality} dimensions"
                )
            result.setdefault(row["id"], {})[modality] = [
                float(value) for value in values
            ]
    return result


def load_embedding_files(
    paths: Sequence[Path] | None,
) -> dict[str, dict[str, list[float]]]:
    """Load repeatable embedding files and reject cross-file conflicts."""
    combined: dict[str, dict[str, list[float]]] = {}
    for path in paths or []:
        for row_id, modalities in load_embeddings(path).items():
            existing = combined.setdefault(row_id, {})
            for modality, vector in modalities.items():
                if modality in existing and existing[modality] != vector:
                    raise CurationError(f"conflicting {modality} vectors for {row_id}")
                existing[modality] = vector
    return combined


def normalize_vector(vector: Sequence[float]) -> list[float]:
    if not vector or not all(math.isfinite(value) for value in vector):
        raise CurationError("embedding vectors must be nonempty and finite")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        raise CurationError("embedding vectors must have nonzero norm")
    return [value / norm for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise CurationError("embedding dimensions differ")
    return sum(a * b for a, b in zip(left, right, strict=True))


def vector_similarity_pairs(
    vectors: dict[str, list[float]], threshold: float
) -> list[tuple[str, str, float]]:
    """Find all cosine pairs above threshold, vectorized when NumPy exists."""
    if not 0.0 < threshold <= 1.0:
        raise CurationError("similarity threshold must be in (0, 1]")
    identifiers = sorted(vectors)
    normalized = {row_id: normalize_vector(vectors[row_id]) for row_id in identifiers}
    pairs: list[tuple[str, str, float]] = []
    try:
        import numpy as np

        matrix = np.asarray(
            [normalized[row_id] for row_id in identifiers], dtype=np.float64
        )
        for index, row_id in enumerate(identifiers):
            if index + 1 >= len(identifiers):
                break
            similarities = matrix[index + 1 :] @ matrix[index]
            matches = np.flatnonzero(similarities >= threshold)
            for offset in matches:
                pairs.append(
                    (
                        row_id,
                        identifiers[index + 1 + int(offset)],
                        float(similarities[offset]),
                    )
                )
    except ImportError:
        warnings.warn(
            "NumPy unavailable; embedding similarity uses slow O(n^2) Python loops",
            RuntimeWarning,
            stacklevel=2,
        )
        for index, left in enumerate(identifiers):
            for right in identifiers[index + 1 :]:
                similarity = cosine(normalized[left], normalized[right])
                if similarity >= threshold:
                    pairs.append((left, right, similarity))
    return pairs


class UnionFind:
    """Small deterministic union-find used to block duplicate leakage."""

    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        left_root, right_root = min((left_root, right_root), (right_root, left_root))
        self.parent[right_root] = left_root


def build_plan(
    rows: list[dict[str, Any]],
    lexicon: dict[str, Any],
    embeddings: dict[str, dict[str, list[float]]],
    *,
    text_threshold: float = 0.98,
    audio_threshold: float = 0.995,
    fractions: dict[str, float] | None = None,
    require_complete_embeddings: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate rows and assign leakage groups to content-balanced splits."""
    fractions = fractions or DEFAULT_SPLIT_FRACTIONS
    if set(fractions) != {"train", "dev", "test"} or not math.isclose(
        sum(fractions.values()), 1.0, abs_tol=1e-9
    ):
        raise CurationError("split fractions must contain train/dev/test summing to 1")
    embedding_requested = bool(embeddings)

    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    exact_links: list[tuple[str, str]] = []
    kept: list[dict[str, Any]] = []
    duplicate_ids = duplicate_hashes = 0
    for row in rows:
        row_id = row.get("id")
        audio_hash = row.get("sha256_audio")
        if not isinstance(row_id, str) or not row_id:
            raise CurationError("every dataset row needs a nonempty string id")
        if not isinstance(audio_hash, str) or not audio_hash:
            raise CurationError(f"row {row_id} lacks sha256_audio")
        group = str(row.get("leakage_group") or row.get("_dataset") or row_id)
        if row_id in seen_ids:
            duplicate_ids += 1
            exact_links.append((seen_ids[row_id], group))
            continue
        seen_ids[row_id] = group
        if audio_hash in seen_hashes:
            duplicate_hashes += 1
            exact_links.append((seen_hashes[audio_hash], group))
            continue
        seen_hashes[audio_hash] = group
        kept.append(
            {
                "id": row_id,
                "dataset": row["_dataset"],
                "group": group,
                "duration": max(0.0, float(row.get("duration_seconds") or 0.0)),
                "text": reference_text(row),
                "audio_hash": audio_hash,
                "train_allowed": row.get("train_allowed") is True,
                "eval_allowed": row.get("eval_allowed") is True,
                "domain_tags": list(row.get("domain_tags") or []),
                "difficulty_tags": list(row.get("difficulty_tags") or []),
                "lexical": term_counts(reference_text(row), lexicon.get("terms", [])),
                "text_vector": embeddings.get(row_id, {}).get("text"),
                "audio_vector": embeddings.get(row_id, {}).get("audio"),
            }
        )

    text_pairs = vector_similarity_pairs(
        {row["id"]: row["text_vector"] for row in kept if row["text_vector"]},
        text_threshold,
    )
    audio_pairs = vector_similarity_pairs(
        {row["id"]: row["audio_vector"] for row in kept if row["audio_vector"]},
        audio_threshold,
    )
    text_coverage = sum(row["text_vector"] is not None for row in kept) / len(kept)
    audio_coverage = sum(row["audio_vector"] is not None for row in kept) / len(kept)
    embedding_coverage_complete = text_coverage == 1.0 and audio_coverage == 1.0
    if require_complete_embeddings and not embedding_coverage_complete:
        raise CurationError(
            "complete text and audio embeddings are required to freeze this split"
        )
    id_to_group = {row["id"]: row["group"] for row in kept}
    cross_text = [
        pair for pair in text_pairs if id_to_group[pair[0]] != id_to_group[pair[1]]
    ]
    cross_audio = [
        pair for pair in audio_pairs if id_to_group[pair[0]] != id_to_group[pair[1]]
    ]

    exact_groups = {group for link in exact_links for group in link}
    union_find = UnionFind(sorted({*id_to_group.values(), *exact_groups}))
    for left_group, right_group in exact_links:
        union_find.union(left_group, right_group)
    for left, right, _ in (*cross_text, *cross_audio):
        union_find.union(id_to_group[left], id_to_group[right])
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        components[union_find.find(row["group"])].append(row)

    embedding_dimensions = {
        field: len(next(row[field] for row in kept if row[field]))
        for field in ("text_vector", "audio_vector")
        if any(row[field] for row in kept)
    }
    global_centroid = aggregate_feature(
        kept, len(lexicon.get("terms", [])), embedding_dimensions
    )
    targets = {name: fraction for name, fraction in fractions.items()}
    totals = {name: 0.0 for name in targets}
    assignments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ordered = sorted(
        components.values(),
        key=lambda component: (
            -rarity_score(component, len(lexicon.get("terms", []))),
            -sum(row["duration"] for row in component),
            component[0]["id"],
        ),
    )
    total_duration = max(sum(row["duration"] for row in kept), 1.0)
    if len(ordered) >= len(targets):
        for split, component in zip(("train", "dev", "test"), ordered, strict=False):
            assignments[split].extend(component)
            totals[split] += sum(row["duration"] for row in component)
        remaining = ordered[len(targets) :]
    else:
        remaining = ordered
    for component in remaining:
        component_duration = sum(row["duration"] for row in component)
        choices = {}
        for split, target in targets.items():
            candidate_rows = [*assignments[split], *component]
            centroid = aggregate_feature(
                candidate_rows,
                len(lexicon.get("terms", [])),
                embedding_dimensions,
            )
            distance = 1.0 - cosine(global_centroid, centroid)
            overshoot = max(
                0.0, totals[split] + component_duration - target * total_duration * 1.25
            )
            choices[split] = distance + 10.0 * overshoot / total_duration
        selected = min(choices, key=lambda split: (choices[split], split))
        assignments[selected].extend(component)
        totals[selected] += component_duration

    component_by_root = {
        union_find.find(component[0]["group"]): component
        for component in components.values()
    }
    component_splits = {
        root: next(item["split"] for item in manifest_rows(component, assignments))
        for root, component in component_by_root.items()
    }
    if len(component_splits) >= len(targets):
        component_splits = optimize_component_splits(
            component_by_root,
            component_splits,
            len(lexicon.get("terms", [])),
            total_duration,
            targets,
            embedding_dimensions,
        )
    assignments = defaultdict(list)
    for root, split in component_splits.items():
        assignments[split].extend(component_by_root[root])

    manifest: list[dict[str, Any]] = []
    for split in sorted(assignments):
        for row in sorted(assignments[split], key=lambda item: item["id"]):
            manifest.append(
                {
                    "id": row["id"],
                    "split": split,
                    "group": row["group"],
                    "dataset": row["dataset"],
                    "audio_sha256": row["audio_hash"],
                    "duration_seconds": row["duration"],
                    "train_allowed": row["train_allowed"],
                    "eval_allowed": row["eval_allowed"],
                }
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": datetime.now(UTC).isoformat(),
        "privacy": {
            "emits_transcripts": False,
            "emits_dictionary_phrases": False,
            "emits_row_ids_in_report": False,
            "row_ids_only_in_local_manifest": True,
        },
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "duplicate_ids": duplicate_ids,
        "duplicate_audio_hashes": duplicate_hashes,
        "embedding_rows": {
            "text": sum(row["text_vector"] is not None for row in kept),
            "audio": sum(row["audio_vector"] is not None for row in kept),
        },
        "embedding_coverage": {
            "requested": embedding_requested,
            "text": text_coverage,
            "audio": audio_coverage,
            "complete": embedding_coverage_complete,
        },
        "near_duplicate_pairs": {
            "text_threshold": text_threshold,
            "audio_threshold": audio_threshold,
            "text": len(text_pairs),
            "audio": len(audio_pairs),
            "cross_group_text": len(cross_text),
            "cross_group_audio": len(cross_audio),
        },
        "connected_group_components": len(components),
        "splits": {},
        "promotion": {
            "train_permission_rows": sum(row["train_allowed"] for row in kept),
            "eval_permission_rows": sum(row["eval_allowed"] for row in kept),
            "permission_gate_remains": any(not row["train_allowed"] for row in kept),
            "freeze_eligible": embedding_coverage_complete
            and all(row["train_allowed"] for row in kept),
        },
    }
    for split, split_rows in sorted(assignments.items()):
        centroid = aggregate_feature(
            split_rows, len(lexicon.get("terms", [])), embedding_dimensions
        )
        report["splits"][split] = {
            "rows": len(split_rows),
            "groups": len({row["group"] for row in split_rows}),
            "duration_seconds": sum(row["duration"] for row in split_rows),
            "content_centroid_distance": 1.0 - cosine(global_centroid, centroid),
            "rare_term_coverage": rare_term_coverage(
                split_rows, len(lexicon.get("terms", []))
            ),
        }
    return manifest, report


def manifest_rows(
    component: list[dict[str, Any]], assignments: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Find existing manifest-shaped rows for one component during optimization."""
    identifiers = {row["id"] for row in component}
    for split, rows in assignments.items():
        if any(row["id"] in identifiers for row in rows):
            return [
                {"id": row["id"], "split": split}
                for row in rows
                if row["id"] in identifiers
            ]
    raise CurationError("component was not assigned to a split")


def split_cost(
    component_by_root: dict[str, list[dict[str, Any]]],
    component_splits: dict[str, str],
    term_count: int,
    total_duration: float,
    targets: dict[str, float],
    dimensions: dict[str, int],
) -> float:
    """Score duration, semantic, and rare-term imbalance for one allocation."""
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for root, split in component_splits.items():
        by_split[split].extend(component_by_root[root])
    if any(not by_split.get(split) for split in targets):
        return float("inf")
    durations = {
        split: sum(row["duration"] for row in rows) for split, rows in by_split.items()
    }
    duration_error = sum(
        abs(durations[split] - target * total_duration) / total_duration
        for split, target in targets.items()
    )
    global_centroid = aggregate_feature(
        [row for rows in by_split.values() for row in rows],
        term_count,
        dimensions,
    )
    semantic_error = sum(
        1.0
        - cosine(
            global_centroid,
            aggregate_feature(by_split[split], term_count, dimensions),
        )
        for split in targets
    )
    coverages = [rare_term_coverage(by_split[split], term_count) for split in targets]
    rare_error = (max(coverages) - min(coverages)) / max(1, max(coverages))
    return duration_error + semantic_error + 0.25 * rare_error


def optimize_component_splits(
    component_by_root: dict[str, list[dict[str, Any]]],
    initial: dict[str, str],
    term_count: int,
    total_duration: float,
    targets: dict[str, float],
    dimensions: dict[str, int],
) -> dict[str, str]:
    """Improve a grouped allocation with deterministic whole-component moves."""
    current = dict(initial)
    current_cost = split_cost(
        component_by_root,
        current,
        term_count,
        total_duration,
        targets,
        dimensions,
    )
    improved = True
    iteration = 0
    max_iterations = max(100, len(component_by_root) * len(targets) * 100)
    while improved and iteration < max_iterations:
        improved = False
        for root in sorted(component_by_root):
            source = current[root]
            for destination in sorted(targets):
                if destination == source:
                    continue
                candidate = dict(current)
                candidate[root] = destination
                cost = split_cost(
                    component_by_root,
                    candidate,
                    term_count,
                    total_duration,
                    targets,
                    dimensions,
                )
                if cost + 1e-12 < current_cost:
                    current, current_cost = candidate, cost
                    improved = True
                    break
            if improved:
                break
        iteration += 1
    if improved:
        warnings.warn(
            "grouped split optimization reached its iteration bound",
            RuntimeWarning,
            stacklevel=2,
        )
    return current


def aggregate_feature(
    rows: Sequence[dict[str, Any]],
    term_count: int,
    dimensions: dict[str, int] | None = None,
) -> list[float]:
    """Combine lexical, text, and audio features into one unit-normalized vector."""
    parts: list[list[float]] = []
    lexical = [0.0] * term_count
    for row in rows:
        for index, value in enumerate(row.get("lexical", [])):
            lexical[index] += float(value)
    lexical_total = sum(lexical)
    if lexical_total:
        lexical = [value / lexical_total for value in lexical]
    else:
        lexical = [1.0 / math.sqrt(term_count)] * term_count if term_count else [1.0]
    parts.append(lexical)
    dimensions = dimensions or {}
    for field, dimension in dimensions.items():
        vectors = [row[field] for row in rows if row.get(field)]
        if vectors:
            mean = [
                sum(vector[index] for vector in vectors) / len(vectors)
                for index in range(dimension)
            ]
        else:
            mean = [1.0 if index == 0 else 0.0 for index in range(dimension)]
        parts.append(mean)
    combined: list[float] = []
    for part in parts:
        normalized = normalize_vector(part)
        combined.extend(normalized)
    if not combined:
        return [1.0]
    return normalize_vector(combined)


def rarity_score(rows: Sequence[dict[str, Any]], term_count: int) -> float:
    if not term_count:
        return 0.0
    totals = [0.0] * term_count
    for row in rows:
        for index, value in enumerate(row.get("lexical", [])):
            totals[index] += float(value)
    return sum(1.0 / (count + 1.0) for count in totals) / term_count


def rare_term_coverage(rows: Sequence[dict[str, Any]], term_count: int) -> int:
    if not term_count:
        return 0
    present = [False] * term_count
    for row in rows:
        for index, value in enumerate(row.get("lexical", [])):
            if value > 0:
                present[index] = True
    return sum(present)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    lexicon = subparsers.add_parser("lexicon", help="import Aqua terms locally")
    lexicon.add_argument("--settings", type=Path, required=True)
    lexicon.add_argument("--out", type=Path, required=True)

    plan = subparsers.add_parser("plan", help="deduplicate and plan grouped splits")
    plan.add_argument("--dataset", type=Path, action="append", required=True)
    plan.add_argument("--lexicon", type=Path, required=True)
    plan.add_argument("--embeddings", type=Path, action="append")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--report", type=Path, required=True)
    plan.add_argument("--text-threshold", type=float, default=0.98)
    plan.add_argument("--audio-threshold", type=float, default=0.995)
    plan.add_argument("--require-complete-embeddings", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one curation verb and return a conventional CLI status."""
    try:
        arguments = parse_arguments(argv if argv is not None else sys.argv[1:])
        if arguments.command == "lexicon":
            validate_output_aliases(
                safe_output_path(arguments.out), [arguments.settings]
            )
            lexicon = load_aqua_lexicon(arguments.settings)
            write_private_json(arguments.out, lexicon)
            print(json.dumps(lexicon_aggregate(lexicon), indent=2, sort_keys=True))
            return 0

        protected = [*arguments.dataset, arguments.lexicon]
        protected.extend(arguments.embeddings)
        for output in (arguments.manifest, arguments.report):
            validate_output_aliases(safe_output_path(output), protected)
        lexicon = json.loads(arguments.lexicon.read_text(encoding="utf-8"))
        if not isinstance(lexicon, dict) or not isinstance(lexicon.get("terms"), list):
            raise CurationError("lexicon file is malformed")
        rows = [
            row for dataset in arguments.dataset for row in iter_parquet_rows(dataset)
        ]
        embeddings = load_embedding_files(arguments.embeddings)
        manifest, report = build_plan(
            rows,
            lexicon,
            embeddings,
            text_threshold=arguments.text_threshold,
            audio_threshold=arguments.audio_threshold,
            require_complete_embeddings=arguments.require_complete_embeddings,
        )
        write_private_json(
            arguments.manifest, {"schema_version": SCHEMA_VERSION, "rows": manifest}
        )
        write_private_json(arguments.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - private-input CLI failure boundary
        print(f"FAIL private-corpus-curation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
