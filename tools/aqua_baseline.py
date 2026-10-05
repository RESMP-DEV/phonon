#!/usr/bin/env python3
"""No-model scoring harness for the frozen Aqua parity protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TOOL_ROOT = Path(__file__).resolve().parent
_DATA_ROOT = _TOOL_ROOT / "aqua_baseline_data"
sys.path.insert(0, str(_DATA_ROOT))

from whisper_english import EnglishTextNormalizer

SCHEMA_VERSION = 1
DEFAULT_METADATA_FIELDS = (
    "model",
    "model_revision",
    "runtime",
    "runtime_revision",
    "kernel_mode",
    "dataset_revision",
    "manifest_revision",
)
DEFAULT_SCHEMA_CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "rows": {"id": "id", "reference": "ref"},
    "candidate": {
        "id": "id",
        "hypothesis": "hyp",
        "generation_seconds": "generation_seconds",
        "end_to_end_seconds": "end_to_end_seconds",
    },
    "baseline": {"id": "id", "hypothesis": "raw_aqua"},
    "manifest": {"id": "id"},
    "metadata": {"allowed_fields": list(DEFAULT_METADATA_FIELDS)},
}


class BaselineError(ValueError):
    """A deterministic protocol violation that must fail the run."""


_FAIR_NORMALIZER = EnglishTextNormalizer()


def fair_norm(text: str) -> str:
    """Apply the registered Whisper-fair English normalization."""
    return _FAIR_NORMALIZER((text or "").strip())


def strict_norm(text: str) -> str:
    """Apply the registered lowercase, whitespace-trimmed strict rule."""
    return (text or "").strip().lower()


def _require_str_mapping(
    value: Any, context: str, fields: tuple[str, ...]
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise BaselineError(f"{context} must be an object")
    if set(value) != set(fields):
        expected = ", ".join(fields)
        raise BaselineError(f"{context} must contain exactly: {expected}")
    result: dict[str, str] = {}
    for field in fields:
        mapped = value[field]
        if not isinstance(mapped, str) or not mapped.strip():
            raise BaselineError(f"{context}.{field} must be a non-empty field name")
        result[field] = mapped
    if len(set(result.values())) != len(result):
        raise BaselineError(f"{context} field names must be unique")
    return result


def load_schema_config(path: Path | None) -> tuple[dict[str, Any], Path | None]:
    if path is None:
        return DEFAULT_SCHEMA_CONFIG, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot read schema config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineError("schema config must be a JSON object")
    required = {
        "schema_version",
        "rows",
        "candidate",
        "baseline",
        "manifest",
        "metadata",
    }
    if set(value) != required:
        raise BaselineError(
            "schema config must contain exactly: schema_version, rows, candidate, "
            "baseline, manifest, metadata"
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise BaselineError(f"schema config version must be {SCHEMA_VERSION}")
    value["rows"] = _require_str_mapping(value["rows"], "rows", ("id", "reference"))
    candidate_fields = (
        "id",
        "hypothesis",
        "generation_seconds",
        "end_to_end_seconds",
    )
    candidate_value = value["candidate"]
    if not isinstance(candidate_value, dict) or set(candidate_value) != set(
        candidate_fields
    ):
        raise BaselineError(
            "candidate must contain exactly: id, hypothesis, generation_seconds, "
            "end_to_end_seconds"
        )
    candidate = {field: candidate_value.get(field) for field in candidate_fields}
    for field in ("id", "hypothesis"):
        if not isinstance(candidate[field], str) or not candidate[field].strip():
            raise BaselineError(f"candidate.{field} must be a non-empty field name")
    for field in ("generation_seconds", "end_to_end_seconds"):
        mapped = candidate[field]
        if mapped is not None and (not isinstance(mapped, str) or not mapped.strip()):
            raise BaselineError(
                f"candidate.{field} must be null or a non-empty field name"
            )
    mapped_names = [name for name in candidate.values() if name is not None]
    if len(mapped_names) != len(set(mapped_names)):
        raise BaselineError("candidate field names must be unique")
    value["candidate"] = candidate
    row_reference_field = value["rows"]["reference"]
    if row_reference_field in set(candidate.values()):
        raise BaselineError("row reference and candidate field names must not overlap")
    if value["baseline"] is not None:
        value["baseline"] = _require_str_mapping(
            value["baseline"], "baseline", ("id", "hypothesis")
        )
        if row_reference_field in set(value["baseline"].values()):
            raise BaselineError("row reference and baseline field names must not overlap")
    if value["manifest"] is not None:
        value["manifest"] = _require_str_mapping(value["manifest"], "manifest", ("id",))
    if value["metadata"] is not None:
        metadata = value["metadata"]
        if not isinstance(metadata, dict) or set(metadata) != {"allowed_fields"}:
            raise BaselineError("metadata must contain exactly: allowed_fields")
        fields = metadata["allowed_fields"]
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(field, str) or not field.strip() for field in fields)
            or len(set(fields)) != len(fields)
        ):
            raise BaselineError(
                "metadata.allowed_fields must be unique non-empty strings"
            )
    return value, path


@dataclass(frozen=True)
class LoadedRows:
    label: str
    path: Path
    sha256: str
    fields: dict[str, str]
    records: dict[str, dict[str, Any]]

    @property
    def row_count(self) -> int:
        return len(self.records)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, UnicodeError) as exc:
        raise BaselineError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _field(record: dict[str, Any], field: str, label: str, line_number: int) -> Any:
    if field not in record:
        raise BaselineError(f"{label} line {line_number} is missing field {field!r}")
    return record[field]


def _stable_id(value: Any, label: str, line_number: int, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BaselineError(
            f"{label} line {line_number}: {field!r} must be a non-empty string "
            "without leading or trailing whitespace"
        )
    return value


def _jsonl_object(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise BaselineError(
            f"{path} line {line_number} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BaselineError(f"{path} line {line_number} must be a JSON object")
    return value


def load_jsonl(path: Path, label: str, fields: dict[str, str]) -> LoadedRows:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise BaselineError(f"cannot read {label} JSONL {path}: {exc}") from exc
    records: dict[str, dict[str, Any]] = {}
    id_field = fields["id"]
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record = _jsonl_object(line, path, line_number)
        row_id = _stable_id(
            _field(record, id_field, label, line_number), label, line_number, id_field
        )
        if row_id in records:
            raise BaselineError(f"{label} has duplicate stable row ID {row_id!r}")
        records[row_id] = record
    if not records:
        raise BaselineError(f"{label} JSONL {path} contains no rows")
    return LoadedRows(label, path, sha256_file(path), fields, records)


def validate_references(rows: LoadedRows) -> None:
    field = rows.fields["reference"]
    for row_id, record in rows.records.items():
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise BaselineError(
                f"{rows.label} row {row_id!r}: {field!r} must be non-empty text"
            )


def validate_hypotheses(rows: LoadedRows) -> None:
    field = rows.fields["hypothesis"]
    for row_id, record in rows.records.items():
        if field not in record:
            raise BaselineError(
                f"{rows.label} row {row_id!r} is missing field {field!r}"
            )
        if not isinstance(record[field], str):
            raise BaselineError(
                f"{rows.label} row {row_id!r}: {field!r} must be a string"
            )


def require_same_ids(source: LoadedRows, target: LoadedRows) -> None:
    source_ids = set(source.records)
    target_ids = set(target.records)
    missing = sorted(source_ids - target_ids)
    extra = sorted(target_ids - source_ids)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing IDs: {missing}")
        if extra:
            details.append(f"extra IDs: {extra}")
        raise BaselineError(
            f"{target.label} does not match {source.label}; " + "; ".join(details)
        )


def load_manifest(
    path: Path | None, fields: dict[str, str] | None, expected_ids: set[str]
) -> tuple[str, str, int | None]:
    if path is None:
        return "(not supplied)", "", None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise BaselineError(f"cannot read manifest {path}: {exc}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    records: list[Any]
    if isinstance(parsed, list):
        records = parsed
    elif isinstance(parsed, dict):
        if set(parsed) != {"rows"} or not isinstance(parsed["rows"], list):
            raise BaselineError("manifest object must contain exactly a rows array")
        records = parsed["rows"]
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BaselineError(
                    f"manifest {path} line {line_number} is invalid: {exc}"
                ) from exc
            records.append(value)

    id_field = fields["id"] if fields else "id"
    manifest_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or id_field not in record:
            raise BaselineError(
                f"manifest record {index} is missing field {id_field!r}"
            )
        row_id = _stable_id(record[id_field], "manifest", index, id_field)
        if row_id in manifest_ids:
            raise BaselineError(f"manifest has duplicate stable row ID {row_id!r}")
        manifest_ids.add(row_id)
    if not manifest_ids:
        raise BaselineError(f"manifest {path} contains no rows")
    if records:
        missing = sorted(expected_ids - manifest_ids)
        extra = sorted(manifest_ids - expected_ids)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing IDs: {missing}")
            if extra:
                details.append(f"extra IDs: {extra}")
            raise BaselineError("manifest does not match rows; " + "; ".join(details))
    return str(path), sha256_file(path), len(manifest_ids) if records else None


def load_metadata(
    path: Path | None, allowed_fields: list[str]
) -> tuple[str, str, dict[str, Any]]:
    if path is None:
        return "(not supplied)", "", {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot read metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineError("metadata must be a JSON object")
    disallowed = sorted(set(value) - set(allowed_fields))
    if disallowed:
        raise BaselineError(
            f"metadata fields are outside the schema allowlist: {disallowed}; "
            "this prevents transcript-like content from entering a receipt"
        )
    safe: dict[str, Any] = {}
    for field, contents in value.items():
        if contents is None or isinstance(contents, (bool, int, float)):
            if isinstance(contents, float) and not math.isfinite(contents):
                raise BaselineError(
                    f"metadata.{field} must be finite, not NaN or infinity"
                )
            safe[field] = contents
        elif isinstance(contents, str):
            if (
                not contents
                or "\n" in contents
                or "\r" in contents
                or len(contents) > 256
            ):
                raise BaselineError(
                    f"metadata.{field} must be a one-line string of at most 256 chars"
                )
            safe[field] = contents
        else:
            raise BaselineError(
                f"metadata.{field} must be null, boolean, number, or short string"
            )
    return str(path), sha256_file(path), safe


def _sentinel(text: str, normalized: str) -> str:
    return normalized or "<empty>"


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row, ref_word in enumerate(reference, start=1):
        current = [row]
        for column, hyp_word in enumerate(hypothesis, start=1):
            substitution = previous[column - 1] + (ref_word != hyp_word)
            deletion = previous[column] + 1
            insertion = current[column - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


def _row_wer(reference: str, hypothesis: str, normalizer: Any) -> float:
    normalized_reference = normalizer(reference)
    normalized_hypothesis = normalizer(hypothesis)
    ref_words = _sentinel(reference, normalized_reference).split()
    hyp_words = _sentinel(hypothesis, normalized_hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def score_texts(
    references: list[str], hypotheses: list[str], normalizer: Any
) -> dict[str, float]:
    fair_exact_denominator = len(references)
    fair_exact = sum(
        fair_norm(reference) == fair_norm(hypothesis)
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    )
    error_count = 0
    reference_word_count = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        normalized_reference = normalizer(reference)
        normalized_hypothesis = normalizer(hypothesis)
        ref_words = _sentinel(reference, normalized_reference).split()
        hyp_words = _sentinel(hypothesis, normalized_hypothesis).split()
        reference_word_count += len(ref_words)
        error_count += edit_distance(ref_words, hyp_words)
    return {
        "wer": error_count / reference_word_count if reference_word_count else 0.0,
        "exact_rate": fair_exact / fair_exact_denominator
        if fair_exact_denominator
        else 0.0,
    }


def score_hypothesis(
    records: dict[str, dict[str, Any]],
    fields: dict[str, str],
    reference_field: str,
    normalizer: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    hypothesis_field = fields["hypothesis"]
    references: list[str] = []
    hypotheses: list[str] = []
    row_wers: dict[str, float] = {}
    for row_id, record in records.items():
        reference = record.get(reference_field)
        hypothesis = record.get(hypothesis_field)
        if not isinstance(reference, str) or not reference.strip():
            raise BaselineError(
                f"scored row {row_id!r}: {reference_field!r} must be non-empty text"
            )
        if not isinstance(hypothesis, str):
            raise BaselineError(
                f"scored row {row_id!r}: {hypothesis_field!r} must be a string"
            )
        references.append(reference)
        hypotheses.append(hypothesis)
    fair = score_texts(references, hypotheses, fair_norm)
    strict = score_texts(references, hypotheses, strict_norm)
    for row_id, reference, hypothesis in zip(
        records, references, hypotheses, strict=True
    ):
        row_wers[row_id] = _row_wer(reference, hypothesis, fair_norm)
    aggregate = {
        "fair_wer": fair["wer"],
        "strict_wer": strict["wer"],
        "exact_rate": fair["exact_rate"],
    }
    return aggregate, row_wers


def compare_to_baseline(
    candidate_wers: dict[str, float], baseline_wers: dict[str, float] | None
) -> dict[str, int | float | None]:
    if baseline_wers is None:
        return {
            "wins": None,
            "ties": None,
            "losses": None,
            "worse_than_input_rate": None,
        }
    wins = ties = losses = 0
    for row_id, candidate in candidate_wers.items():
        baseline = baseline_wers[row_id]
        if candidate < baseline:
            wins += 1
        elif candidate == baseline:
            ties += 1
        else:
            losses += 1
    row_count = len(candidate_wers)
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "worse_than_input_rate": losses / row_count if row_count else 0.0,
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def extract_timings(
    records: dict[str, dict[str, Any]], fields: dict[str, str | None]
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for metric, field in fields.items():
        values: list[float] = []
        if field is not None:
            for row_id, record in records.items():
                if field not in record or record[field] is None:
                    continue
                value = record[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise BaselineError(
                        f"candidate row {row_id!r}: {field!r} must be a number or null"
                    )
                numeric = float(value)
                if not math.isfinite(numeric) or numeric < 0:
                    raise BaselineError(
                        f"candidate row {row_id!r}: {field!r} must be a finite non-negative number"
                    )
                values.append(numeric)
        result[metric] = {
            "count": len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
        }
    return result


def _input_summary(
    path: Path, digest: str, row_count: int | None = None
) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "sha256": digest}
    if row_count is not None:
        summary["row_count"] = row_count
    return summary


def build_receipt(
    *,
    rows: LoadedRows,
    candidate: LoadedRows,
    baseline: LoadedRows | None,
    schema: dict[str, Any],
    schema_path: Path | None,
    schema_sha256: str,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    candidate_metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
    comparison: dict[str, Any],
    latency: dict[str, Any],
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "rows": _input_summary(rows.path, rows.sha256, rows.row_count),
        "candidate": _input_summary(
            candidate.path, candidate.sha256, candidate.row_count
        ),
    }
    if baseline is not None:
        inputs["baseline"] = _input_summary(
            baseline.path, baseline.sha256, baseline.row_count
        )
    inputs["schema_config"] = (
        _input_summary(schema_path, schema_sha256) if schema_path is not None else None
    )
    inputs["manifest"] = manifest
    inputs["metadata"] = metadata
    non_claims = [
        "Aggregate scoring only; this run executed no model, decoder, or audio path.",
        "The receipt intentionally contains no references, hypotheses, per-row timings, audio, or other transcripts.",
        "A matching row count and manifest do not by themselves prove that these inputs are the registered frozen 500-row slice.",
        "Wins, ties, losses, and worse-than-input rate use fair-normalized row WER and are unavailable without a baseline.",
    ]
    if baseline is None:
        non_claims.append(
            "No baseline hypothesis was supplied, so no candidate-versus-baseline claim is made."
        )
    return {
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "schema_version": SCHEMA_VERSION,
        "schema": schema,
        "inputs": inputs,
        "row_count": rows.row_count,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": comparison,
        "latency": latency,
        "non_claims": non_claims,
    }


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        temporary.replace(path)
    except (OSError, UnicodeError) as exc:
        try:
            temporary.unlink()
        except (OSError, UnboundLocalError):
            pass
        raise BaselineError(f"cannot write receipt {path}: {exc}") from exc


def run(args: argparse.Namespace) -> dict[str, Any]:
    schema, schema_path = load_schema_config(args.schema_config)
    schema_sha256 = sha256_file(schema_path) if schema_path is not None else ""
    output = Path(args.output)
    protected = [args.rows, args.candidate, schema_path, args.manifest, args.metadata]
    if args.baseline is not None:
        protected.append(args.baseline)
    if any(
        _same_path(output, Path(source)) for source in protected if source is not None
    ):
        raise BaselineError("output must not overwrite an input file")

    rows = load_jsonl(Path(args.rows), "frozen rows", schema["rows"])
    validate_references(rows)
    candidate = load_jsonl(Path(args.candidate), "candidate", schema["candidate"])
    require_same_ids(rows, candidate)
    validate_hypotheses(candidate)

    baseline: LoadedRows | None = None
    if args.baseline is not None:
        if schema["baseline"] is None:
            raise BaselineError(
                "--baseline was supplied, but schema config disables baseline"
            )
        baseline = load_jsonl(Path(args.baseline), "baseline", schema["baseline"])
        require_same_ids(rows, baseline)
        validate_hypotheses(baseline)

    expected_ids = set(rows.records)
    manifest_path, manifest_sha256, manifest_count = load_manifest(
        args.manifest, schema["manifest"], expected_ids
    )
    allowed_metadata = (
        schema["metadata"]["allowed_fields"] if schema["metadata"] is not None else []
    )
    if args.metadata is not None and not allowed_metadata:
        raise BaselineError(
            "--metadata was supplied, but schema config allows no metadata fields"
        )
    metadata_path, metadata_sha256, metadata = load_metadata(
        args.metadata, allowed_metadata
    )

    candidate_scored_records = {
        row_id: {**candidate.records[row_id], **record}
        for row_id, record in rows.records.items()
    }
    candidate_metrics, candidate_row_wers = score_hypothesis(
        candidate_scored_records,
        schema["candidate"],
        rows.fields["reference"],
        fair_norm,
    )
    baseline_metrics: dict[str, float] | None = None
    baseline_row_wers: dict[str, float] | None = None
    if baseline is not None:
        baseline_scored_records = {
            row_id: {**baseline.records[row_id], **record}
            for row_id, record in rows.records.items()
        }
        baseline_metrics, baseline_row_wers = score_hypothesis(
            baseline_scored_records,
            schema["baseline"],
            rows.fields["reference"],
            fair_norm,
        )
    comparison = compare_to_baseline(candidate_row_wers, baseline_row_wers)
    latency = extract_timings(
        candidate.records,
        {
            key: schema["candidate"][key]
            for key in ("generation_seconds", "end_to_end_seconds")
        },
    )
    receipt = build_receipt(
        rows=rows,
        candidate=candidate,
        baseline=baseline,
        schema=schema,
        schema_path=schema_path,
        schema_sha256=schema_sha256,
        manifest={
            "path": manifest_path,
            "sha256": manifest_sha256,
            "row_count": manifest_count,
        },
        metadata={"path": metadata_path, "sha256": metadata_sha256, "fields": metadata},
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        comparison=comparison,
        latency=latency,
    )
    _atomic_write_json(output, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score explicit Aqua JSONL paths without model or network execution."
    )
    parser.add_argument(
        "--print-schema-config",
        action="store_true",
        help="print the default schema contract and exit without scoring",
    )
    parser.add_argument(
        "--rows", type=Path, help="JSONL with stable IDs and references"
    )
    parser.add_argument(
        "--candidate", type=Path, help="JSONL with stable IDs and hypotheses"
    )
    parser.add_argument(
        "--baseline", type=Path, help="optional baseline hypothesis JSONL"
    )
    parser.add_argument(
        "--schema-config", type=Path, help="optional JSON field-mapping contract"
    )
    parser.add_argument(
        "--manifest", type=Path, help="optional manifest keyed by stable row ID"
    )
    parser.add_argument(
        "--metadata", type=Path, help="optional allowlisted run metadata JSON object"
    )
    parser.add_argument("--output", type=Path, help="receipt JSON destination")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_schema_config:
        print(json.dumps(DEFAULT_SCHEMA_CONFIG, indent=2))
        return 0
    missing = [
        name for name in ("rows", "candidate", "output") if getattr(args, name) is None
    ]
    if missing:
        parser.error("--rows, --candidate, and --output are required for scoring")
    try:
        receipt = run(args)
    except BaselineError as exc:
        print(f"aqua_baseline: error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {args.output}: {receipt['row_count']} rows, "
        f"fair_wer={receipt['candidate_metrics']['fair_wer']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
