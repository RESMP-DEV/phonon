from __future__ import annotations

import hashlib
import io
import json
import wave
from pathlib import Path

import private_dataset_audit as audit


def wav_bytes(samples: int = 1_600, rate: int = 16_000) -> bytes:
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


def test_sanitized_aggregate_accepts_canonical_trainable_row() -> None:
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
    report = audit.summarize_rows([row("synthetic:a"), row("synthetic:a")], "synthetic")

    assert report["rows"] == 2
    assert report["unique_row_ids"] == 1
    assert report["duplicate_rows"] == 1
    assert report["usable_train_rows"] == 0


def test_permission_or_label_gap_blocks_usability_without_failing_integrity() -> None:
    blocked = row("synthetic:a", train=False)
    report = audit.summarize_rows([blocked], "synthetic")

    assert report["integrity_passed"] is True
    assert report["usage_verdict"] == "blocked_by_permissions_or_labels"
    assert report["usable_train_rows"] == 0


def test_teacher_parser_accepts_json_and_raw_string() -> None:
    assert audit.parse_teachers('{"a":"one"}') == {"a": "one"}
    assert audit.parse_teachers("raw transcript") == {"raw": "raw transcript"}
    assert audit.parse_teachers({"a": "one", "b": "  "}) == {"a": "one"}


def test_output_path_must_stay_outside_git_or_under_build(tmp_path: Path) -> None:
    build_report = audit.REPO_ROOT / "build" / "reports" / "private" / "report.json"
    assert audit.safe_output_path(build_report) == build_report.resolve()

    tracked_report = audit.REPO_ROOT / "evidence" / "report.json"
    try:
        audit.safe_output_path(tracked_report)
    except audit.AuditError as error:
        assert "outside the repository" in str(error)
    else:
        raise AssertionError("tracked aggregate output was accepted")
