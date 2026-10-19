#!/usr/bin/env python3
"""Scrub private Phonon consensus rows into local embedding triplets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import time_ns
from typing import Any

import numpy as np

PRIVATE_TOOL_ROOT = Path(__file__).resolve().parents[1] / "private-corpus-curation"
if str(PRIVATE_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(PRIVATE_TOOL_ROOT))

import triplets
from corpus_curation import (
    iter_parquet_rows,
    normalize_phrase,
    reference_text,
    safe_output_path,
    term_counts,
    validate_output_aliases,
)
from glm_reconcile import parse_teachers

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIMENSION = 768


@dataclass(frozen=True)
class PrivateRow:
    """One deduplicated private row with local fingerprints and audio."""

    row_id: str
    group: str
    split: str
    duration: float
    reference: str
    audio_hash: str
    audio_bytes: bytes
    teachers: dict[str, str]
    text_vector: np.ndarray
    audio_vector: np.ndarray


class ScrubError(ValueError):
    """Raised for malformed private scrub inputs or unsafe outputs."""


def load_split_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Load the local whole-group split manifest by row ID."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScrubError(f"cannot read split manifest: {error}") from error
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ScrubError("split manifest must contain a rows array")
    result: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ScrubError(f"split row {index} lacks id")
        if row["id"] in result:
            raise ScrubError(f"duplicate split row id at index {index}")
        for field in ("split", "group", "audio_sha256"):
            if not isinstance(row.get(field), str):
                raise ScrubError(f"split row {row['id']} lacks {field}")
        result[row["id"]] = row
    return result


def load_lexicon(path: Path) -> list[dict[str, Any]]:
    """Load the local Aqua lexicon without emitting its phrases."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScrubError(f"cannot read lexicon: {error}") from error
    terms = payload.get("terms") if isinstance(payload, dict) else None
    if not isinstance(terms, list):
        raise ScrubError("lexicon lacks terms array")
    for index, term in enumerate(terms):
        if not isinstance(term, dict) or not isinstance(term.get("normalized"), str):
            raise ScrubError(f"lexicon term {index} is malformed")
    return terms


def load_vectors(path: Path, modality: str) -> dict[str, np.ndarray]:
    """Load one provenance-checked EmbeddingGemma vector file."""
    vectors: dict[str, np.ndarray] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ScrubError(f"cannot read {modality} vectors: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ScrubError(f"vector line {line_number} is invalid") from error
        if (
            not isinstance(row, dict)
            or row.get("model") != MODEL_ID
            or row.get("revision") != MODEL_REVISION
            or row.get("modality") != modality
            or row.get("dimension") != DIMENSION
            or not isinstance(row.get("id"), str)
            or not isinstance(row.get("vector"), list)
            or len(row["vector"]) != DIMENSION
        ):
            raise ScrubError(f"vector line {line_number} has mixed provenance")
        if row["id"] in vectors:
            raise ScrubError(f"duplicate vector id on line {line_number}")
        values = np.asarray(row["vector"], dtype=np.float64)
        norm = np.linalg.norm(values)
        if not np.isfinite(values).all() or norm <= 0:
            raise ScrubError(f"vector line {line_number} is not finite")
        vectors[row["id"]] = values / norm
    return vectors


def load_private_rows(
    dataset: Path,
    split_manifest: dict[str, dict[str, str]],
    text_vectors: dict[str, np.ndarray],
    audio_vectors: dict[str, np.ndarray],
) -> tuple[list[PrivateRow], dict[str, int]]:
    """Deduplicate and validate canonical rows selected by the split manifest."""
    rows: list[PrivateRow] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    stats: Counter = Counter()
    for source in iter_parquet_rows(dataset):
        row_id = source.get("id")
        audio_hash = source.get("sha256_audio")
        if not isinstance(row_id, str) or not row_id:
            raise ScrubError("private row lacks id")
        if not isinstance(audio_hash, str) or not audio_hash:
            raise ScrubError(f"private row {row_id} lacks audio hash")
        if row_id in seen_ids or audio_hash in seen_hashes:
            stats["duplicates"] += 1
            continue
        seen_ids.add(row_id)
        seen_hashes.add(audio_hash)
        assignment = split_manifest.get(row_id)
        if assignment is None:
            stats["unassigned"] += 1
            continue
        if assignment["audio_sha256"] != audio_hash:
            raise ScrubError(f"split manifest audio hash mismatch for {row_id}")
        if row_id not in text_vectors or row_id not in audio_vectors:
            stats["missing_vectors"] += 1
            continue
        audio = source.get("audio")
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if not isinstance(audio_bytes, bytes | bytearray):
            stats["missing_audio"] += 1
            continue
        if hashlib.sha256(audio_bytes).hexdigest() != audio_hash:
            raise ScrubError(f"embedded audio hash mismatch for {row_id}")
        duration = source.get("duration_seconds")
        reference = reference_text(source)
        if (
            not isinstance(duration, int | float)
            or not math.isfinite(float(duration))
            or not 1.0 <= float(duration) <= 40.0
            or not reference
        ):
            stats["quality_filtered"] += 1
            continue
        rows.append(
            PrivateRow(
                row_id=row_id,
                group=assignment["group"],
                split=assignment["split"],
                duration=float(duration),
                reference=reference,
                audio_hash=audio_hash,
                audio_bytes=bytes(audio_bytes),
                teachers=parse_teachers(source.get("teacher_transcripts")),
                text_vector=text_vectors[row_id],
                audio_vector=audio_vectors[row_id],
            )
        )
    return rows, dict(stats)


def materialize_audio(row: PrivateRow, directory: Path) -> Path:
    """Write verified private audio to a hash-addressed user-only cache."""
    path = directory / f"{row.audio_hash}.wav"
    if path.exists():
        if path.stat().st_mode & 0o777 != 0o600:
            path.chmod(0o600)
        return path
    temporary = directory / f".{row.audio_hash}.{os.getpid()}.{time_ns()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(row.audio_bytes)
    temporary.replace(path)
    path.chmod(0o600)
    return path


def token_set(value: str) -> set[str]:
    """Return normalized lexical tokens for conservative overlap scoring."""
    return set(normalize_phrase(value).split())


def teacher_overlap(reference: str, hypothesis: str) -> float:
    """Score a teacher by token recall plus balanced Jaccard overlap."""
    reference_tokens = token_set(reference)
    hypothesis_tokens = token_set(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return -1.0
    intersection = len(reference_tokens & hypothesis_tokens)
    union = len(reference_tokens | hypothesis_tokens)
    recall = intersection / len(reference_tokens)
    jaccard = intersection / union
    return 0.7 * recall + 0.3 * jaccard


def choose_teacher(row: PrivateRow) -> str | None:
    """Choose the most reference-consistent teacher hypothesis."""
    candidates = {
        name: hypothesis
        for name, hypothesis in row.teachers.items()
        if hypothesis.strip() != row.reference.strip()
    }
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda name: (
            teacher_overlap(row.reference, candidates[name]),
            len(candidates[name]),
            name,
        ),
    )


def negative_candidates(
    row: PrivateRow, rows_by_split: dict[str, list[PrivateRow]], field: str
) -> list[tuple[float, PrivateRow]]:
    """Return same-split hard-negative candidates from different groups."""
    candidates: list[tuple[float, PrivateRow]] = []
    vector = getattr(row, field)
    for candidate in rows_by_split[row.split]:
        if candidate.group == row.group or candidate.row_id == row.row_id:
            continue
        similarity = float(np.dot(vector, getattr(candidate, field)))
        candidates.append((similarity, candidate))
    candidates.sort(key=lambda item: (-item[0], item[1].row_id))
    return candidates


def choose_negative(
    candidates: list[tuple[float, PrivateRow]],
    *,
    minimum: float,
    maximum: float,
    forbidden_term: str | None = None,
    used: set[str],
) -> tuple[PrivateRow, float] | None:
    """Choose a deterministic hard negative in the desired similarity band."""
    forbidden_tokens = set(forbidden_term.split()) if forbidden_term else set()
    for similarity, candidate in candidates:
        if candidate.row_id in used or similarity < minimum or similarity > maximum:
            continue
        if forbidden_tokens and forbidden_tokens <= token_set(candidate.reference):
            continue
        return candidate, similarity
    return None


def source_record(row: PrivateRow, source: str, revision: str) -> dict[str, Any]:
    """Build local-only provenance fields for a private triplet."""
    return {
        "source": source,
        "source_revision": revision,
        "split": row.split,
        "row_id": row.row_id,
        "leakage_group": row.group,
        "audio_sha256": row.audio_hash,
    }


def build_private_triplets(
    rows: list[PrivateRow],
    terms: list[dict[str, Any]],
    output_dir: Path,
    *,
    source: str,
    source_revision: str,
    minimum_negative: float,
    maximum_negative: float,
    max_term_examples: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Generate teacher, cross-modal, term, and replacement triplets."""
    rows_by_split: dict[str, list[PrivateRow]] = defaultdict(list)
    for row in rows:
        rows_by_split[row.split].append(row)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    outputs: dict[str, list[dict[str, Any]]] = {
        split: [] for split in ("train", "dev", "test")
    }
    stats: Counter = Counter()
    negative_similarities: list[float] = []

    for row in rows:
        teacher_name = choose_teacher(row)
        if teacher_name is None:
            stats["missing_teacher"] += 1
        else:
            negative = choose_negative(
                negative_candidates(row, rows_by_split, "text_vector"),
                minimum=minimum_negative,
                maximum=maximum_negative,
                used=set(),
            )
            if negative is None:
                stats["missing_text_negative"] += 1
            else:
                negative_row, similarity = negative
                negative_similarities.append(similarity)
                outputs[row.split].append(
                    {
                        "schema_version": triplets.SCHEMA_VERSION,
                        "kind": "teacher_consistency",
                        **source_record(row, source, source_revision),
                        "teacher": teacher_name,
                        "anchor": row.teachers[teacher_name],
                        "positive": row.reference,
                        "negative": negative_row.reference,
                        "negative_row_id": negative_row.row_id,
                        "negative_similarity": similarity,
                    }
                )
                stats["teacher_consistency"] += 1

        audio_negative = choose_negative(
            negative_candidates(row, rows_by_split, "audio_vector"),
            minimum=minimum_negative,
            maximum=maximum_negative,
            used=set(),
        )
        if audio_negative is None:
            stats["missing_audio_negative"] += 1
        else:
            negative_row, similarity = audio_negative
            negative_similarities.append(similarity)
            positive_path = materialize_audio(row, audio_dir)
            negative_path = materialize_audio(negative_row, audio_dir)
            outputs[row.split].append(
                {
                    "schema_version": triplets.SCHEMA_VERSION,
                    "kind": "text_audio",
                    **source_record(row, source, source_revision),
                    "anchor": row.reference,
                    "positive": str(positive_path),
                    "negative": str(negative_path),
                    "negative_row_id": negative_row.row_id,
                    "negative_similarity": similarity,
                }
            )
            stats["text_audio"] += 1

    term_use: Counter = Counter()
    for row in rows:
        counts = term_counts(row.reference, terms)
        present = [
            terms[index]["normalized"]
            for index, count in enumerate(counts)
            if count > 0
        ]
        for term in sorted(present):
            if term_use[term] >= max_term_examples:
                break
            negative = choose_negative(
                negative_candidates(row, rows_by_split, "text_vector"),
                minimum=minimum_negative,
                maximum=maximum_negative,
                forbidden_term=term,
                used=set(),
            )
            if negative is None:
                continue
            negative_row, similarity = negative
            negative_similarities.append(similarity)
            term_use[term] += 1
            outputs[row.split].append(
                {
                    "schema_version": triplets.SCHEMA_VERSION,
                    "kind": "term_context",
                    **source_record(row, source, source_revision),
                    "term": term,
                    "anchor": term,
                    "positive": row.reference,
                    "negative": negative_row.reference,
                    "negative_row_id": negative_row.row_id,
                    "negative_similarity": similarity,
                }
            )
            stats["term_context"] += 1

    replacement_terms = [
        term
        for term in terms
        if isinstance(term.get("spoken_source"), str) and term.get("phrase")
    ]
    for term in replacement_terms:
        anchor_tokens = token_set(term["spoken_source"])
        ranked = sorted(
            (
                candidate
                for candidate in replacement_terms
                if candidate["normalized"] != term["normalized"]
            ),
            key=lambda candidate: (
                -len(anchor_tokens & token_set(candidate.get("spoken_source") or "")),
                candidate["normalized"],
            ),
        )
        if not ranked:
            continue
        negative = ranked[0]
        if normalize_phrase(term["spoken_source"]) == term[
            "normalized"
        ] or normalize_phrase(term["phrase"]) == normalize_phrase(negative["phrase"]):
            stats["replacement_filtered"] += 1
            continue
        outputs["train"].append(
            {
                "schema_version": triplets.SCHEMA_VERSION,
                "kind": "replacement",
                "source": source,
                "source_revision": source_revision,
                "split": "train",
                "term": term["normalized"],
                "anchor": term["spoken_source"],
                "positive": term["phrase"],
                "negative": negative["phrase"],
            }
        )
        stats["replacement"] += 1

    for split, records in outputs.items():
        seen: set[tuple[str, str, str]] = set()
        unique: list[dict[str, Any]] = []
        for record in records:
            key = (record["anchor"], record["positive"], record["negative"])
            if key in seen:
                stats["duplicate_triplets"] += 1
                continue
            seen.add(key)
            unique.append(record)
        outputs[split] = unique

    report = {
        "written_by_split": {split: len(items) for split, items in outputs.items()},
        "kind_counts": dict(sorted(stats.items())),
        "terms_used": len(term_use),
        "replacement_terms": len(replacement_terms),
        "hard_negative_similarity": {
            "count": len(negative_similarities),
            "mean": statistics.fmean(negative_similarities)
            if negative_similarities
            else None,
            "min": min(negative_similarities) if negative_similarities else None,
            "max": max(negative_similarities) if negative_similarities else None,
        },
    }
    return outputs, report


def write_private_outputs(
    output_dir: Path, records_by_split: dict[str, list[dict[str, Any]]]
) -> None:
    """Write split JSONL files with user-only permissions."""
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for split, records in records_by_split.items():
        triplets.write_jsonl(output_dir / f"{split}.jsonl", records, private=True)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse private scrub arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--audio-embeddings", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source", default="private/youtube_technical_consensus_extended_v0"
    )
    parser.add_argument(
        "--source-revision",
        default="412fdeec4bf119297a66e66ce2775b5af8b18b6f",
    )
    parser.add_argument("--minimum-negative-similarity", type=float, default=0.45)
    parser.add_argument("--maximum-negative-similarity", type=float, default=0.97)
    parser.add_argument("--max-term-examples", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Scrub private rows and write local-only triplet files."""
    try:
        arguments = parse_arguments(argv if argv is not None else sys.argv[1:])
        output_dir = safe_output_path(arguments.out_dir)
        protected = [
            arguments.dataset,
            arguments.lexicon,
            arguments.split_manifest,
            arguments.text_embeddings,
            arguments.audio_embeddings,
        ]
        validate_output_aliases(output_dir, protected)
        split_manifest = load_split_manifest(arguments.split_manifest)
        terms = load_lexicon(arguments.lexicon)
        text_vectors = load_vectors(arguments.text_embeddings, "text")
        audio_vectors = load_vectors(arguments.audio_embeddings, "audio")
        rows, load_stats = load_private_rows(
            arguments.dataset, split_manifest, text_vectors, audio_vectors
        )
        records, scrub_report = build_private_triplets(
            rows,
            terms,
            output_dir,
            source=arguments.source,
            source_revision=arguments.source_revision,
            minimum_negative=arguments.minimum_negative_similarity,
            maximum_negative=arguments.maximum_negative_similarity,
            max_term_examples=arguments.max_term_examples,
        )
        write_private_outputs(output_dir, records)
        report = {
            "schema_version": triplets.SCHEMA_VERSION,
            "output_dir": str(output_dir),
            "input_rows": len(rows),
            "load_stats": load_stats,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dimension": DIMENSION,
            **scrub_report,
            "permission_gate": {
                "source_rows_promoted": 0,
                "all_source_permissions_remain_false": True,
                "fine_tune_eligible": False,
                "reason": "frozen-evaluation contamination and permission promotion remain incomplete",
            },
            "privacy": {
                "emits_transcripts": False,
                "emits_dictionary_phrases": False,
                "emits_row_ids": False,
                "row_ids_and_text_only_in_local_triplets": True,
                "output_permissions": "0600",
            },
        }
        triplets.write_json(output_dir / "report.json", report, private=True)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - private-input CLI failure boundary
        print(f"FAIL private-scrub: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
