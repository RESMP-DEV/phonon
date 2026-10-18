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
import sys
import wave
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
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
        if isinstance(transcript, str) and transcript.strip():
            result[str(name)] = transcript
    return result


def _as_bool(value: Any) -> bool:
    return value is True


def _text_present(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _decode_wav(data: bytes) -> tuple[str, float] | None:
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
        if not isinstance(data, (bytes, bytearray)) or not data:
            report["missing_required_counts"]["audio_bytes"] += 1
            row_usable = False
        else:
            assert isinstance(data, bytes | bytearray)
            actual_hash = hashlib.sha256(data).hexdigest()
            if not isinstance(expected_hash, str) or not expected_hash:
                report["missing_required_counts"]["sha256_audio"] += 1
                row_usable = False
            elif actual_hash != expected_hash:
                report["hash_mismatches"] += 1
                row_usable = False
            if actual_hash in audio_hashes:
                report["duplicate_audio_rows"] += 1
            audio_hashes.add(actual_hash)
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
                    not isinstance(duration, int | float)
                    or abs(float(duration) - observed_duration) > 0.051
                ):
                    report["duration_mismatches"] += 1
                    row_usable = False
                else:
                    report["audio_seconds"] += float(duration)

        sample_rate = row.get("sample_rate")
        report["sample_rate_column_counts"][str(sample_rate)] += 1
        if sample_rate != 16000:
            row_usable = False

        start = row.get("clip_start")
        end = row.get("clip_end")
        duration = row.get("duration_seconds")
        if (
            not isinstance(start, int | float)
            or not isinstance(end, int | float)
            or not isinstance(duration, int | float)
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
    global_ids: set[str] = set()
    global_hashes: set[str] = set()
    datasets: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            raise AuditError(f"dataset directory does not exist: {root}")
        local_ids: set[str] = set()
        local_hashes: set[str] = set()

        report = summarize_rows(
            observed_rows(iter_parquet_rows(root), local_ids, local_hashes),
            root.name,
        )
        report["cross_dataset_duplicate_ids"] = len(local_ids & global_ids)
        report["cross_dataset_duplicate_audio_hashes"] = len(
            local_hashes & global_hashes
        )
        files = []
        for path in sorted((root / "data").glob("*.parquet")):
            files.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "row_groups": parquet_row_groups(path),
                }
            )
        report["files"] = files
        report["provenance"] = provenance.get(root.name)
        datasets.append(report)
        global_ids.update(local_ids)
        global_hashes.update(local_hashes)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "recorded_at": datetime.now(UTC).isoformat(),
        "privacy": {
            "emits_transcripts": False,
            "emits_row_ids": False,
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
    return value


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write one sanitized aggregate report and enforce audio integrity."""
    try:
        arguments = parse_arguments(argv if argv is not None else sys.argv[1:])
        output = safe_output_path(arguments.out)
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
    except (AuditError, OSError) as error:
        print(f"FAIL private-dataset-audit: {error}", file=sys.stderr)
        return 2
    integrity_passed = all(
        dataset["integrity_passed"] for dataset in report["datasets"]
    )
    print(f"{'PASS' if integrity_passed else 'FAIL'} {output}")
    return 0 if integrity_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
