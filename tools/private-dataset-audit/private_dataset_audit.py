#!/usr/bin/env python3
"""Audit private Phonon Parquet datasets without emitting transcript-shaped data.

The CLI validates embedded audio, permissions, labels, teacher coverage, and
duplicate identities. Its JSON output contains counts and file hashes only;
source URLs, source paths, speaker identities, row IDs, transcripts, teacher
transcripts, terms, and notes are never copied to the report.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import wave
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA_VERSION = 1
SELECTED_COLUMNS = (
    "id",
    "audio",
    "sample_rate",
    "duration_seconds",
    "sha256_audio",
    "source_kind",
    "source_id",
    "clip_start",
    "clip_end",
    "speaker_id",
    "domain_tags",
    "leakage_group",
    "split",
    "train_allowed",
    "eval_allowed",
    "label_status",
    "verbatim_text",
    "insert_text",
    "normalized_text",
    "teacher_transcripts",
    "term_spans",
    "notes",
)
LABEL_FIELDS = ("verbatim_text", "insert_text", "normalized_text")
PROVENANCE_FIELDS = frozenset({"repo_id", "revision"})
NEVER_REPORT_COLUMNS = (
    "id",
    "source_id",
    "source_url",
    "original_audio",
    "speaker_id",
    *LABEL_FIELDS,
    "teacher_transcripts",
    "term_spans",
    "notes",
)


class AuditError(ValueError):
    """Raised for an invalid audit invocation or malformed aggregate input."""


def parse_teachers(value: Any) -> dict[str, str]:
    """Parse teacher transcripts without returning them in public aggregates."""
    parsed = value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = {"raw": value}
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, str] = {}
    for name, transcript in parsed.items():
        text = transcript.get("text") if isinstance(transcript, dict) else transcript
        if isinstance(text, str) and text.strip():
            result[str(name)] = text
    return result


def _as_bool(value: Any) -> bool:
    """Accept only a literal Parquet boolean True."""
    return value is True


def _text_present(value: Any) -> bool:
    """Return whether a string field has non-whitespace content."""
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: Any) -> bool:
    """Reject NaN and infinities in addition to nonnumeric metadata."""
    return isinstance(value, int | float) and math.isfinite(float(value))


def _decode_wav(data: bytes) -> tuple[str, float] | None:
    """Decode only the WAV properties needed for aggregate validation."""
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            channels = audio.getnchannels()
            bits = audio.getsampwidth() * 8
            rate = audio.getframerate()
            frames = audio.getnframes()
            duration = frames / rate if rate else 0.0
            return f"{channels}ch_{bits}bit_{rate}hz", duration
    except (wave.Error, EOFError, OSError):
        return None


def summarize_rows(rows: Iterable[dict[str, Any]], dataset: str) -> dict[str, Any]:
    """Return a sanitized aggregate for one canonical Phonon dataset."""
    report: dict[str, Any] = {
        "dataset": dataset,
        "rows": 0,
        "unique_row_ids": 0,
        "duplicate_rows": 0,
        "unique_audio_hashes": 0,
        "duplicate_audio_rows": 0,
        "audio_format_counts": Counter(),
        "sample_rate_column_counts": Counter(),
        "source_kind_counts": Counter(),
        "split_counts": Counter(),
        "label_status_counts": Counter(),
        "permission_counts": Counter(),
        "label_nonempty_counts": Counter(),
        "teacher_name_counts": Counter(),
        "teacher_count_distribution": Counter(),
        "missing_required_counts": Counter(),
        "hash_mismatches": 0,
        "sample_rate_mismatches": 0,
        "invalid_wav_rows": 0,
        "duration_mismatches": 0,
        "clip_interval_mismatches": 0,
        "noncanonical_audio_rows": 0,
        "rows_with_terms": 0,
        "rows_with_notes": 0,
        "audio_bytes": 0,
        "audio_seconds": 0.0,
        "unique_source_ids": 0,
        "unique_leakage_groups": 0,
        "unique_speaker_ids": 0,
        "unique_domain_tags": 0,
        "train_permission_rows": 0,
        "eval_permission_rows": 0,
        "labeled_train_permission_rows": 0,
    }
    row_ids: set[str] = set()
    audio_hashes: set[str] = set()
    source_ids: set[str] = set()
    leakage_groups: set[str] = set()
    speaker_ids: set[str] = set()
    domain_tags: set[str] = set()
    usable_rows: set[str] = set()
    duplicate_ids: set[str] = set()
    duplicate_hashes: set[str] = set()
    first_row_id_by_hash: dict[str, str] = {}

    for row in rows:
        report["rows"] += 1
        row_id = row.get("id")
        valid_id = isinstance(row_id, str) and bool(row_id)
        if valid_id:
            assert isinstance(row_id, str)
            if row_id in row_ids:
                report["duplicate_rows"] += 1
                usable_rows.discard(row_id)
                duplicate_ids.add(row_id)
            else:
                row_ids.add(row_id)
                usable_rows.add(row_id)
        else:
            report["missing_required_counts"]["id"] += 1

        audio = row.get("audio")
        data = audio.get("bytes") if isinstance(audio, dict) else None
        expected_hash = row.get("sha256_audio")
        row_usable = valid_id and row_id not in duplicate_ids
        decoded: tuple[str, float] | None = None
        if not isinstance(data, (bytes, bytearray)) or not data:
            report["missing_required_counts"]["audio_bytes"] += 1
            row_usable = False
        else:
            assert isinstance(data, bytes | bytearray)
            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash in duplicate_hashes:
                row_usable = False
            if not isinstance(expected_hash, str) or not expected_hash:
                report["missing_required_counts"]["sha256_audio"] += 1
                row_usable = False
            elif actual_hash != expected_hash:
                report["hash_mismatches"] += 1
                row_usable = False
            if actual_hash in audio_hashes:
                report["duplicate_audio_rows"] += 1
                duplicate_hashes.add(actual_hash)
                previous_id = first_row_id_by_hash.get(actual_hash)
                if previous_id is not None:
                    usable_rows.discard(previous_id)
            if actual_hash in duplicate_hashes:
                row_usable = False
            audio_hashes.add(actual_hash)
            if valid_id and actual_hash not in duplicate_hashes:
                assert isinstance(row_id, str)
                first_row_id_by_hash.setdefault(actual_hash, row_id)
            report["audio_bytes"] += len(data)
            decoded = _decode_wav(bytes(data))
            if decoded is None:
                report["invalid_wav_rows"] += 1
                row_usable = False
            else:
                audio_format, observed_duration = decoded
                report["audio_format_counts"][audio_format] += 1
                if audio_format != "1ch_16bit_16000hz":
                    report["noncanonical_audio_rows"] += 1
                    row_usable = False
                duration = row.get("duration_seconds")
                if (
                    not _finite_number(duration)
                    or abs(float(duration) - observed_duration) > 0.051
                ):
                    report["duration_mismatches"] += 1
                    row_usable = False
                else:
                    report["audio_seconds"] += float(duration)

        sample_rate = row.get("sample_rate")
        report["sample_rate_column_counts"][str(sample_rate)] += 1
        decoded_rate = (
            None
            if decoded is None
            else int(decoded[0].removesuffix("hz").rsplit("_", 1)[-1])
        )
        if sample_rate != 16000 or sample_rate != decoded_rate:
            report["sample_rate_mismatches"] += 1
            row_usable = False

        start = row.get("clip_start")
        end = row.get("clip_end")
        duration = row.get("duration_seconds")
        if (start is not None or end is not None) and (
            not _finite_number(start)
            or not _finite_number(end)
            or not _finite_number(duration)
            or float(end) - float(start) <= 0
            or abs((float(end) - float(start)) - float(duration)) > 0.101
        ):
            report["clip_interval_mismatches"] += 1
            row_usable = False

        train_allowed = _as_bool(row.get("train_allowed"))
        eval_allowed = _as_bool(row.get("eval_allowed"))
        report["permission_counts"][f"train={train_allowed},eval={eval_allowed}"] += 1
        if train_allowed:
            report["train_permission_rows"] += 1
        if eval_allowed:
            report["eval_permission_rows"] += 1
        if train_allowed and eval_allowed:
            row_usable = False

        labels_present = False
        for field in LABEL_FIELDS:
            if _text_present(row.get(field)):
                report["label_nonempty_counts"][field] += 1
                labels_present = True
        if train_allowed and not eval_allowed and labels_present:
            report["labeled_train_permission_rows"] += 1
        else:
            row_usable = False

        teachers = parse_teachers(row.get("teacher_transcripts"))
        if teachers:
            report["teacher_count_distribution"][str(len(teachers))] += 1
            for teacher in teachers:
                report["teacher_name_counts"][teacher] += 1
        if not row_usable and valid_id:
            assert isinstance(row_id, str)
            usable_rows.discard(row_id)

        for field, counts in (
            ("source_kind", report["source_kind_counts"]),
            ("split", report["split_counts"]),
            ("label_status", report["label_status_counts"]),
        ):
            counts[str(row.get(field))] += 1
        if isinstance(row.get("term_spans"), list) and row["term_spans"]:
            report["rows_with_terms"] += 1
        if _text_present(row.get("notes")):
            report["rows_with_notes"] += 1
        for field, identities in (
            ("source_id", source_ids),
            ("leakage_group", leakage_groups),
            ("speaker_id", speaker_ids),
        ):
            value = row.get(field)
            if isinstance(value, str) and value:
                identities.add(value)
        tags = row.get("domain_tags")
        if isinstance(tags, list):
            domain_tags.update(str(tag) for tag in tags if tag)

    if report["rows"] == 0:
        raise AuditError(f"dataset {dataset!r} contains no rows")
    report["unique_row_ids"] = len(row_ids)
    report["unique_audio_hashes"] = len(audio_hashes)
    report["unique_source_ids"] = len(source_ids)
    report["unique_leakage_groups"] = len(leakage_groups)
    report["unique_speaker_ids"] = len(speaker_ids)
    report["unique_domain_tags"] = len(domain_tags)
    report["usable_train_rows"] = len(usable_rows)
    integrity_checks = (
        report["missing_required_counts"],
        report["hash_mismatches"],
        report["sample_rate_mismatches"],
        report["invalid_wav_rows"],
        report["duration_mismatches"],
        report["clip_interval_mismatches"],
        report["noncanonical_audio_rows"],
    )
    report["integrity_passed"] = all(
        not value if isinstance(value, int) else not sum(value.values())
        for value in integrity_checks
    )
    for key in (
        "audio_format_counts",
        "sample_rate_column_counts",
        "source_kind_counts",
        "split_counts",
        "label_status_counts",
        "permission_counts",
        "label_nonempty_counts",
        "teacher_name_counts",
        "teacher_count_distribution",
        "missing_required_counts",
    ):
        report[key] = dict(sorted(report[key].items()))
    report["usage_verdict"] = (
        "train_ready_subset"
        if report["usable_train_rows"]
        else "blocked_by_permissions_or_labels"
    )
    return report


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_rows(
    rows: Iterable[dict[str, Any]], row_ids: set[str], audio_hashes: set[str]
) -> Iterator[dict[str, Any]]:
    """Stream rows while retaining only identity hashes for overlap checks."""
    for row in rows:
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id:
            row_ids.add(row_id)
        digest = row.get("sha256_audio")
        if isinstance(digest, str) and digest:
            audio_hashes.add(digest)
        yield row


def iter_parquet_rows(root: Path) -> Iterator[dict[str, Any]]:
    """Yield canonical rows from a local private Parquet mirror."""
    try:
        from pyarrow import parquet
    except ImportError as error:
        raise AuditError(
            "pyarrow is required to read private dataset Parquets; "
            "run with `uv run --with pyarrow`"
        ) from error
    paths = sorted((root / "data").glob("*.parquet"))
    if not paths:
        raise AuditError(f"no Parquet files found under {root / 'data'}")
    for path in paths:
        source = parquet.ParquetFile(path)
        names = set(source.schema_arrow.names)
        missing = [name for name in SELECTED_COLUMNS if name not in names]
        if missing:
            raise AuditError(
                f"{path.name} is missing canonical columns: {', '.join(missing)}"
            )
        for batch in source.iter_batches(
            columns=list(SELECTED_COLUMNS), batch_size=256
        ):
            yield from batch.to_pylist()


def audit_datasets(
    roots: list[Path], provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Audit local mirrors while tracking only aggregate cross-dataset overlap."""
    provenance = provenance or {}
    datasets: list[dict[str, Any]] = []
    identities: dict[str, tuple[set[str], set[str]]] = {}
    for dataset_index, root in enumerate(roots, start=1):
        if not root.is_dir():
            raise AuditError(f"dataset directory does not exist: {root}")
        local_ids: set[str] = set()
        local_hashes: set[str] = set()

        report = summarize_rows(
            observed_rows(iter_parquet_rows(root), local_ids, local_hashes),
            f"dataset_{dataset_index}",
        )
        files = []
        for file_index, path in enumerate(
            sorted((root / "data").glob("*.parquet")), start=1
        ):
            files.append(
                {
                    "file": f"file_{file_index}",
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "row_groups": parquet_row_groups(path),
                }
            )
        report["files"] = files
        report["provenance"] = provenance.get(root.name)
        datasets.append(report)
        identities[report["dataset"]] = (local_ids, local_hashes)
    for report in datasets:
        dataset = report["dataset"]
        local_ids, local_hashes = identities[dataset]
        other_ids: set[str] = set()
        other_hashes: set[str] = set()
        for other_name, (row_ids, audio_hashes) in identities.items():
            if other_name == dataset:
                continue
            other_ids.update(row_ids)
            other_hashes.update(audio_hashes)
        report["cross_dataset_duplicate_ids"] = len(local_ids & other_ids)
        report["cross_dataset_duplicate_audio_hashes"] = len(
            local_hashes & other_hashes
        )
        if (
            report["cross_dataset_duplicate_ids"]
            or report["cross_dataset_duplicate_audio_hashes"]
        ):
            report["usable_train_rows"] = 0
            report["usage_verdict"] = "blocked_by_cross_dataset_duplicate_identity"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "recorded_at": datetime.now(UTC).isoformat(),
        "privacy": {
            "emits_transcripts": False,
            "emits_row_ids": False,
            "emits_dataset_or_file_names": False,
            "emits_source_urls_or_original_paths": False,
            "never_reported_columns": list(NEVER_REPORT_COLUMNS),
        },
        "rows_audited": sum(report["rows"] for report in datasets),
        "usable_train_rows": sum(report["usable_train_rows"] for report in datasets),
        "datasets": datasets,
    }


def parquet_row_groups(path: Path) -> int:
    from pyarrow import parquet

    return parquet.ParquetFile(path).num_row_groups


def safe_output_path(path: Path) -> Path:
    """Keep aggregate outputs outside Git except for ignored build reports."""
    if path.is_symlink():
        raise AuditError("report output must not be a symbolic link")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved
    if relative.parts and relative.parts[0] == "build":
        return resolved
    raise AuditError("report output must be outside the repository or under build/")


def validate_output_aliases(
    output: Path, dataset_roots: Iterable[Path], provenance: Path | None
) -> None:
    """Prevent report invalidation from deleting a private input path."""
    resolved_output = output.resolve()
    protected = [*dataset_roots]
    if provenance is not None:
        protected.append(provenance)
    for candidate in protected:
        resolved_candidate = candidate.resolve()
        if (
            resolved_output == resolved_candidate
            or resolved_candidate in resolved_output.parents
            or resolved_output in resolved_candidate.parents
        ):
            raise AuditError("report output must not overwrite an input path")


def invalidate_output(path: Path) -> None:
    """Remove an earlier report before a rerun can leave a stale PASS."""
    path.unlink(missing_ok=True)


def load_provenance(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read provenance {path}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError("provenance must be a JSON object keyed by directory name")
    normalized: dict[str, dict[str, str]] = {}
    for dataset, entry in value.items():
        if not isinstance(dataset, str) or not dataset:
            raise AuditError("provenance dataset keys must be nonempty strings")
        if not isinstance(entry, dict) or set(entry) != PROVENANCE_FIELDS:
            raise AuditError(
                "each provenance entry must contain exactly repo_id and revision"
            )
        if any(
            not isinstance(entry[field], str) or not entry[field].strip()
            for field in PROVENANCE_FIELDS
        ):
            raise AuditError("provenance repo_id and revision must be nonempty strings")
        revision = entry["revision"]
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision.lower()
        ):
            raise AuditError("provenance revision must be a 40-hex commit SHA")
        normalized[dataset] = {
            "repo_id": entry["repo_id"],
            "revision": revision.lower(),
        }
    return normalized


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the local private-dataset audit invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    return parser.parse_args(argv)


def write_failure_report(path: Path, failure: str) -> None:
    """Publish an exception-class-only failure marker after invalidation."""
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "recorded_at": datetime.now(UTC).isoformat(),
        "passed": False,
        "failure": failure,
        "non_claims": [
            "This failure marker does not identify the failing private row or path.",
            "No integrity or training-eligibility claim is made.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Write one sanitized aggregate report and enforce audio integrity."""
    output: Path | None = None
    try:
        arguments = parse_arguments(argv if argv is not None else sys.argv[1:])
        candidate_output = safe_output_path(arguments.out)
        validate_output_aliases(
            candidate_output, arguments.dataset, arguments.provenance
        )
        output = candidate_output
        invalidate_output(output)
        report = audit_datasets(
            arguments.dataset, load_provenance(arguments.provenance)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    except Exception as error:  # noqa: BLE001 - private-input CLI failure boundary
        if output is not None:
            try:
                write_failure_report(output, type(error).__name__)
            except Exception as report_error:  # noqa: BLE001 - report boundary
                print(
                    f"FAIL private-dataset-audit report: {report_error}",
                    file=sys.stderr,
                )
        print(f"FAIL private-dataset-audit: {error}", file=sys.stderr)
        return 2
    integrity_passed = all(
        dataset["integrity_passed"] for dataset in report["datasets"]
    )
    print(f"{'PASS' if integrity_passed else 'FAIL'} {output}")
    return 0 if integrity_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
