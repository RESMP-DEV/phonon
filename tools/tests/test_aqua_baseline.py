from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import aqua_baseline


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def score(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    baseline: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    manifest: list[dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any], Path]:
    paths = {
        "rows": write_jsonl(tmp_path / "rows.jsonl", rows),
        "candidate": write_jsonl(tmp_path / "candidate.jsonl", candidate),
        "output": tmp_path / "receipt.json",
    }
    argv = [
        "--rows",
        str(paths["rows"]),
        "--candidate",
        str(paths["candidate"]),
        "--output",
        str(paths["output"]),
    ]
    if baseline is not None:
        paths["baseline"] = write_jsonl(tmp_path / "baseline.jsonl", baseline)
        argv.extend(["--baseline", str(paths["baseline"])])
    if config is not None:
        paths["config"] = write_json(tmp_path / "schema.json", config)
        argv.extend(["--schema-config", str(paths["config"])])
    if metadata is not None:
        paths["metadata"] = write_json(tmp_path / "metadata.json", metadata)
        argv.extend(["--metadata", str(paths["metadata"])])
    if manifest is not None:
        paths["manifest"] = write_jsonl(tmp_path / "manifest.jsonl", manifest)
        argv.extend(["--manifest", str(paths["manifest"])])
    exit_code = aqua_baseline.main(argv)
    if exit_code != 0:
        return exit_code, {}, paths["output"]
    receipt = json.loads(paths["output"].read_text(encoding="utf-8"))
    return exit_code, receipt, paths


def custom_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rows": {"id": "text_id", "reference": "target"},
        "candidate": {
            "id": "text_id",
            "hypothesis": "prediction",
            "generation_seconds": "seconds",
            "end_to_end_seconds": None,
        },
        "baseline": {"id": "text_id", "hypothesis": "input"},
        "manifest": {"id": "text_id"},
        "metadata": {"allowed_fields": ["model_revision"]},
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fair_normalization_and_strict_wer(tmp_path: Path) -> None:
    rows = [{"id": "synthetic:001", "ref": "hello two cats"}]
    candidate = [{"id": "synthetic:001", "hyp": "hello  2 cats"}]
    code, receipt, _ = score(tmp_path, rows, candidate)
    assert code == 0
    assert receipt["candidate_metrics"] == {
        "fair_wer": 0.0,
        "strict_wer": 1 / 3,
        "exact_rate": 1.0,
    }


def test_custom_field_mappings_and_input_hashes(tmp_path: Path) -> None:
    rows = [{"text_id": "row-a", "target": "alpha beta"}]
    candidate = [
        {
            "text_id": "row-a",
            "prediction": "alpha gamma",
            "seconds": 0.25,
        }
    ]
    baseline = [{"text_id": "row-a", "input": "alpha delta"}]
    metadata = {"model_revision": "synthetic-revision-1"}
    code, receipt, paths = score(
        tmp_path,
        rows,
        candidate,
        baseline=baseline,
        config=custom_config(),
        metadata=metadata,
    )
    assert code == 0
    assert receipt["inputs"]["rows"]["sha256"] == digest(paths["rows"])
    assert receipt["inputs"]["candidate"]["sha256"] == digest(paths["candidate"])
    assert receipt["inputs"]["baseline"]["sha256"] == digest(paths["baseline"])
    assert receipt["inputs"]["metadata"]["sha256"] == digest(paths["metadata"])
    assert receipt["inputs"]["metadata"]["fields"] == metadata
    assert receipt["latency"]["generation_seconds"] == {
        "count": 1,
        "p50": 0.25,
        "p95": 0.25,
    }
    assert receipt["latency"]["end_to_end_seconds"] == {
        "count": 0,
        "p50": None,
        "p95": None,
    }


def test_rejects_duplicate_and_missing_stable_ids(tmp_path: Path) -> None:
    rows = [
        {"id": "row-a", "ref": "alpha"},
        {"id": "row-a", "ref": "beta"},
    ]
    candidate = [{"id": "row-a", "hyp": "alpha"}]
    code, _, output = score(tmp_path, rows, candidate)
    assert code == 2
    assert not output.exists()

    rows = [{"id": "row-a", "ref": "alpha"}]
    candidate = [{"id": "row-b", "hyp": "alpha"}]
    code, _, output = score(tmp_path, rows, candidate)
    assert code == 2
    assert not output.exists()


def test_empty_reference_is_rejected(tmp_path: Path) -> None:
    code, _, output = score(
        tmp_path,
        [{"id": "row-a", "ref": "   "}],
        [{"id": "row-a", "hyp": "alpha"}],
    )
    assert code == 2
    assert not output.exists()


def test_wins_ties_losses_and_worse_than_input_rate(tmp_path: Path) -> None:
    rows = [
        {"id": "row-a", "ref": "one two three"},
        {"id": "row-b", "ref": "four five six"},
        {"id": "row-c", "ref": "seven eight"},
    ]
    candidate = [
        {"id": "row-a", "hyp": "one two three"},
        {"id": "row-b", "hyp": "four five x"},
        {"id": "row-c", "hyp": "seven eight x"},
    ]
    baseline = [
        {"id": "row-a", "raw_aqua": "one two x"},
        {"id": "row-b", "raw_aqua": "four five x"},
        {"id": "row-c", "raw_aqua": "seven eight"},
    ]
    code, receipt, _ = score(tmp_path, rows, candidate, baseline=baseline)
    assert code == 0
    assert receipt["comparison"] == {
        "wins": 1,
        "ties": 1,
        "losses": 1,
        "worse_than_input_rate": 1 / 3,
    }


def test_missing_timing_fields_yield_null_aggregates(tmp_path: Path) -> None:
    rows = [{"id": "row-a", "ref": "alpha beta"}]
    candidate = [{"id": "row-a", "hyp": "alpha beta"}]
    code, receipt, _ = score(tmp_path, rows, candidate)
    assert code == 0
    for timing in receipt["latency"].values():
        assert timing == {"count": 0, "p50": None, "p95": None}


def test_manifest_ids_are_validated(tmp_path: Path) -> None:
    rows = [{"id": "row-a", "ref": "alpha"}]
    candidate = [{"id": "row-a", "hyp": "alpha"}]
    manifest = [{"id": "row-a"}, {"id": "row-b"}]
    code, _, output = score(tmp_path, rows, candidate, manifest=manifest)
    assert code == 2
    assert not output.exists()


def test_empty_or_malformed_manifest_object_is_rejected(tmp_path: Path) -> None:
    malformed = write_json(tmp_path / "manifest.json", {"values": []})
    try:
        aqua_baseline.load_manifest(malformed, {"id": "id"}, {"row-a"})
    except aqua_baseline.BaselineError as exc:
        assert "exactly a rows array" in str(exc)
    else:
        raise AssertionError("malformed manifest object was accepted")

    empty = write_json(tmp_path / "empty-manifest.json", {"rows": []})
    try:
        aqua_baseline.load_manifest(empty, {"id": "id"}, {"row-a"})
    except aqua_baseline.BaselineError as exc:
        assert "contains no rows" in str(exc)
    else:
        raise AssertionError("empty manifest was accepted")


def test_metadata_and_receipt_are_transcript_redacted(tmp_path: Path) -> None:
    reference = "private-alpha-reference"
    hypothesis = "private-beta-hypothesis"
    rows = [{"id": "row-a", "ref": reference}]
    candidate = [
        {
            "id": "row-a",
            "hyp": hypothesis,
            "generation_seconds": 1.0,
            "end_to_end_seconds": 2.0,
        }
    ]
    metadata = {"runtime_revision": "synthetic-runtime"}
    code, receipt, _ = score(tmp_path, rows, candidate, metadata=metadata)
    assert code == 0
    receipt_text = json.dumps(receipt)
    assert reference not in receipt_text
    assert hypothesis not in receipt_text
    assert receipt["recorded_at"].endswith("Z")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T.+Z", receipt["recorded_at"])


def test_metadata_transcript_key_is_rejected(tmp_path: Path) -> None:
    path = write_json(tmp_path / "metadata.json", {"reference": "secret"})
    try:
        aqua_baseline.load_metadata(path, ["model"])
    except aqua_baseline.BaselineError as exc:
        assert "outside the schema allowlist" in str(exc)
    else:
        raise AssertionError("transcript-shaped metadata key was accepted")


def test_nonfinite_metadata_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "metadata.json"
    source.write_text('{"runtime": NaN}', encoding="utf-8")
    try:
        aqua_baseline.load_metadata(source, ["runtime"])
    except aqua_baseline.BaselineError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("NaN metadata was accepted")


def test_overlapping_field_mappings_are_rejected(tmp_path: Path) -> None:
    config = custom_config()
    config["candidate"]["hypothesis"] = "target"
    path = write_json(tmp_path / "schema.json", config)
    try:
        aqua_baseline.load_schema_config(path)
    except aqua_baseline.BaselineError as exc:
        assert "row reference and candidate field names must not overlap" in str(exc)
    else:
        raise AssertionError("overlapping field mapping was accepted")


def test_percentile_interpolation() -> None:
    assert aqua_baseline.percentile([], 0.5) is None
    assert aqua_baseline.percentile([4.0], 0.95) == 4.0
    values = [10.0, 20.0, 30.0, 40.0]
    assert aqua_baseline.percentile(values, 0.5) == 25.0
    assert aqua_baseline.percentile(values, 0.95) == 38.5
