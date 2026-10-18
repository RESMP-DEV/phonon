from __future__ import annotations

import hashlib
import io
import json
import math
import wave
from pathlib import Path

import private_dataset_audit as audit


def wav_bytes(samples: int = 1_600, rate: int = 16_000) -> bytes:
    """Create a short canonical synthetic WAV fixture."""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * samples)
    return output.getvalue()


def row(
    row_id: str, *, train: bool = True, text: str = "private reference"
) -> dict[str, object]:
    """Create a canonical synthetic row containing deliberately private markers."""
    audio = wav_bytes()
    return {
        "id": row_id,
        "audio": {"bytes": audio, "path": f"/private/{row_id}.wav"},
        "sample_rate": 16_000,
        "duration_seconds": 0.1,
        "sha256_audio": hashlib.sha256(audio).hexdigest(),
        "source_kind": "synthetic",
        "source_id": "private-source",
        "source_url": "https://private.invalid/video",
        "clip_start": 0.0,
        "clip_end": 0.1,
        "speaker_id": "private-speaker",
        "domain_tags": ["synthetic"],
        "leakage_group": "private-leakage",
        "split": "train",
        "train_allowed": train,
        "eval_allowed": False,
        "label_status": "synthetic",
        "verbatim_text": text,
        "insert_text": text,
        "normalized_text": text,
        "teacher_transcripts": json.dumps({"teacher-a": "private teacher text"}),
        "term_spans": ["private-term"],
        "notes": "private note",
        "original_audio": "/private/original.wav",
    }


def row_with_audio(
    row_id: str,
    audio: bytes,
    *,
    sample_rate: int = 16_000,
) -> dict[str, object]:
    """Create a synthetic row with caller-selected audio identity and rate."""
    result = row(row_id)
    result["audio"] = {"bytes": audio, "path": f"/private/{row_id}.wav"}
    result["sample_rate"] = sample_rate
    result["sha256_audio"] = hashlib.sha256(audio).hexdigest()
    return result


def test_sanitized_aggregate_accepts_canonical_trainable_row() -> None:
    """A canonical train-permitted row is counted without retaining its text."""
    report = audit.summarize_rows([row("synthetic:a")], "synthetic")
    serialized = json.dumps(report)

    assert report["rows"] == 1
    assert report["usable_train_rows"] == 1
    assert report["integrity_passed"] is True
    assert report["audio_format_counts"] == {"1ch_16bit_16000hz": 1}
    assert report["permission_counts"] == {"train=True,eval=False": 1}
    assert report["teacher_name_counts"] == {"teacher-a": 1}
    for private_value in (
        "synthetic:a",
        "private reference",
        "private teacher text",
        "private-source",
        "private-speaker",
        "private-leakage",
        "https://private.invalid/video",
        "/private/original.wav",
        "private note",
    ):
        assert private_value not in serialized


def test_duplicate_id_is_not_counted_usable() -> None:
    """A repeated row ID cannot contribute two training observations."""
    report = audit.summarize_rows([row("synthetic:a"), row("synthetic:a")], "synthetic")

    assert report["rows"] == 2
    assert report["unique_row_ids"] == 1
    assert report["duplicate_rows"] == 1
    assert report["usable_train_rows"] == 0


def test_duplicate_audio_under_distinct_ids_is_not_usable() -> None:
    """Identical audio under two IDs cannot double-count training audio."""
    audio = wav_bytes()
    report = audit.summarize_rows(
        [
            row_with_audio("synthetic:a", audio),
            row_with_audio("synthetic:b", audio),
        ],
        "synthetic",
    )

    assert report["duplicate_audio_rows"] == 1
    assert report["unique_row_ids"] == 2
    assert report["usable_train_rows"] == 0


def test_metadata_sample_rate_must_match_wav_container() -> None:
    """Metadata and WAV-container sample rates must agree."""
    report = audit.summarize_rows(
        [row_with_audio("synthetic:a", wav_bytes(), sample_rate=8_000)],
        "synthetic",
    )

    assert report["sample_rate_mismatches"] == 1
    assert report["integrity_passed"] is False
    assert report["usable_train_rows"] == 0


def test_permission_or_label_gap_blocks_usability_without_failing_integrity() -> None:
    """Permission gaps block use without being reported as media corruption."""
    blocked = row("synthetic:a", train=False)
    report = audit.summarize_rows([blocked], "synthetic")

    assert report["integrity_passed"] is True
    assert report["usage_verdict"] == "blocked_by_permissions_or_labels"
    assert report["usable_train_rows"] == 0


def test_teacher_parser_accepts_json_and_raw_string() -> None:
    """Teacher hypotheses are parsed only for aggregate coverage counts."""
    assert audit.parse_teachers('{"a":"one"}') == {"a": "one"}
    assert audit.parse_teachers({"a": {"text": "one"}}) == {"a": "one"}
    assert audit.parse_teachers("raw transcript") == {"raw": "raw transcript"}
    assert audit.parse_teachers({"a": "one", "b": "  "}) == {"a": "one"}


def test_nullable_clip_interval_is_valid_only_when_both_bounds_are_null() -> None:
    """Canonical whole-file rows may omit both clip bounds."""
    whole_file = row("synthetic:a")
    whole_file["clip_start"] = None
    whole_file["clip_end"] = None
    report = audit.summarize_rows([whole_file], "synthetic")

    assert report["clip_interval_mismatches"] == 0
    assert report["integrity_passed"] is True
    assert report["usable_train_rows"] == 1

    partial = row("synthetic:b")
    partial["clip_start"] = 0.0
    partial["clip_end"] = None
    partial_report = audit.summarize_rows([partial], "synthetic")
    assert partial_report["clip_interval_mismatches"] == 1


def test_output_path_must_stay_outside_git_or_under_build(tmp_path: Path) -> None:
    """Aggregate output cannot enter tracked repository state."""
    assert audit.REPO_ROOT == Path(audit.__file__).resolve().parents[2]
    assert (audit.REPO_ROOT / "scripts" / "check_quality.py").is_file()
    build_report = audit.REPO_ROOT / "build" / "reports" / "private" / "report.json"
    assert audit.safe_output_path(build_report) == build_report.resolve()

    tracked_report = audit.REPO_ROOT / "evidence" / "report.json"
    try:
        audit.safe_output_path(tracked_report)
    except audit.AuditError as error:
        assert "outside the repository" in str(error)
    else:
        raise AssertionError("tracked aggregate output was accepted")


def test_nonfinite_duration_is_integrity_failure() -> None:
    """NaN duration metadata cannot silently pass or poison JSON."""
    invalid = row("synthetic:a")
    invalid["duration_seconds"] = math.nan

    report = audit.summarize_rows([invalid], "synthetic")

    assert report["duration_mismatches"] == 1
    assert report["integrity_passed"] is False
    assert report["usable_train_rows"] == 0
    assert "NaN" not in json.dumps(report)


def test_output_cannot_alias_private_inputs(tmp_path: Path) -> None:
    """A report rerun cannot delete a dataset root or provenance file."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}", encoding="utf-8")

    audit.validate_output_aliases(tmp_path / "report.json", [dataset], None)
    for output in (
        dataset,
        dataset / "data" / "part-000.parquet",
        tmp_path,
        provenance,
    ):
        try:
            audit.validate_output_aliases(output, [dataset], provenance)
        except audit.AuditError:
            continue
        raise AssertionError(f"protected output alias was accepted: {output}")


def test_provenance_schema_is_allowlisted(tmp_path: Path) -> None:
    """Only private HF repo IDs and pinned commit SHAs reach a report."""
    revision = "b" * 40
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps({"dataset": {"repo_id": "org/dataset", "revision": revision}}),
        encoding="utf-8",
    )
    assert audit.load_provenance(valid) == {
        "dataset": {"repo_id": "org/dataset", "revision": revision}
    }

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            {
                "dataset": {
                    "repo_id": "org/dataset",
                    "revision": revision,
                    "source_url": "https://private.invalid",
                }
            }
        ),
        encoding="utf-8",
    )
    try:
        audit.load_provenance(invalid)
    except audit.AuditError as error:
        assert "exactly repo_id and revision" in str(error)
    else:
        raise AssertionError("extra provenance field was accepted")
