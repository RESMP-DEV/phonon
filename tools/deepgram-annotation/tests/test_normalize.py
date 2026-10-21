from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from normalize import (
    NormalizationError,
    normalize_response,
    sha256_bytes,
)
from validate import validate_row

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "listen_response.json"

SOURCE_ROW = {
    "row_id": "row-1",
    "audio_sha256": "a" * 64,
    "duration_seconds": 2.4,
    "leakage_group": "g1",
    "source_kind": "owned_public_youtube",
    "source_id": "src",
    "split": "train",
}


@pytest.fixture()
def fixture_payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def normalize_helper(
    fixture_payload: dict[str, Any],
    arm: str,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    from deepgram_client import arm_parameters

    raw = json.dumps(fixture_payload, sort_keys=True).encode("utf-8")
    parameters = arm_parameters(
        arm, keyterms=["CUDA"] if arm.startswith("keyterm") else None
    )
    entry = normalize_response(
        fixture_payload,
        raw_bytes=raw,
        row=SOURCE_ROW,
        arm=arm,
        parameters=parameters,
        elapsed_ms=123.0,
        attempts=1,
        request_tag="phonon-dg-v1-test",
    )
    return entry, parameters


def test_fixture_normalized_row_has_required_provenance(
    fixture_payload: dict[str, Any],
) -> None:
    entry, _parameters = normalize_helper(fixture_payload, "base_smart")
    assert entry["provider_request_id"] == "6a1f9c31-5f9e-4b93-8f8c-51b45cd9f001"
    assert entry["model"]["returned_uuid"] == "2187e11a-3532-4498-b076-81fa530bdd49"
    assert entry["model"]["requested_version"] == "2025-07-31.0"
    assert entry["mip_opt_out"] is True
    assert entry["request_tag"] == "phonon-dg-v1-test"
    expected = hashlib.sha256(
        json.dumps(fixture_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert entry["provider_payload_sha256"] == expected
    assert entry["provider_payload_canonical_sha256"] is not None
    assert entry["timings"]["attempts"] == 1
    assert entry["timings"]["provider_elapsed_ms"] == 123.0


def test_fixture_words_turns_and_utterances(fixture_payload: dict[str, Any]) -> None:
    entry, _ = normalize_helper(fixture_payload, "base_smart")
    assert len(entry["words"]) == 8
    assert entry["words"][0]["word"] == "hello"
    assert entry["words"][0]["punctuated_word"] == "Hello"
    assert entry["words"][0]["start"] == 0.02
    assert entry["words"][0]["end"] == 0.36
    assert entry["words"][0]["confidence"] == 0.99
    assert entry["turns"] == [
        {
            "speaker": 0,
            "start": 0.02,
            "end": 1.55,
            "word_indices": [0, 1, 2, 3, 4],
            "word_count": 5,
        },
        {
            "speaker": 1,
            "start": 1.72,
            "end": 2.38,
            "word_indices": [5, 6, 7],
            "word_count": 3,
        },
    ]
    assert len(entry["utterances"]) == 2
    assert entry["utterances"][0]["transcript"] == "Hello world, CUDA is fast."
    assert entry["utterances"][1]["speaker"] == 1


def test_fixture_paragraphs_entities_and_intelligence(
    fixture_payload: dict[str, Any],
) -> None:
    entry, _ = normalize_helper(fixture_payload, "base_smart")
    assert len(entry["paragraphs"]) == 2
    assert (
        entry["paragraphs"][0]["sentences"][0]["text"] == "Hello world, CUDA is fast."
    )
    assert entry["entities"][0]["label"] == "TECHNOLOGY"
    assert entry["entities"][0]["value"] == "CUDA"
    assert entry["entities"][0]["start_word"] == 2.0
    assert entry["topics"][0]["topics"][0]["value"] == "GPUs"
    assert entry["intents"][0]["intents"][0]["value"] == "state_fact"
    assert entry["sentiment"]["average"]["sentiment"] == "positive"
    assert entry["summary"]["short"] == "GPU discussion."
    assert entry["summary"]["paragraphs"][0]["summary"].startswith("Two speakers")


def test_smart_and_raw_arms_validate_against_their_own_features(
    fixture_payload: dict[str, Any],
) -> None:
    entry, parameters = normalize_helper(fixture_payload, "base_smart")
    failures = validate_row(
        entry,
        source_row=SOURCE_ROW,
        arm="base_smart",
        parameters=parameters,
        expected_audio_sha256=SOURCE_ROW["audio_sha256"],
    )
    assert failures == []

    raw_entry, raw_parameters = normalize_helper(fixture_payload, "base_raw")
    raw_failures = validate_row(
        raw_entry,
        source_row=SOURCE_ROW,
        arm="base_raw",
        parameters=raw_parameters,
        expected_audio_sha256=SOURCE_ROW["audio_sha256"],
    )
    assert raw_failures == []


def test_validation_preserves_out_of_bounds_and_rejects_privacy_drift(
    fixture_payload: dict[str, Any],
) -> None:
    fixture_payload["results"]["channels"][0]["alternatives"][0]["words"][0]["end"] = (
        99.0
    )
    entry, parameters = normalize_helper(fixture_payload, "base_smart")
    failures = validate_row(
        entry,
        source_row=SOURCE_ROW,
        arm="base_smart",
        parameters=parameters,
        expected_audio_sha256=SOURCE_ROW["audio_sha256"],
    )
    assert failures == []
    assert "last_word_extends_beyond_provider_duration" in entry["quality"]["flags"]

    privacy_broken = dict(parameters)
    privacy_broken["mip_opt_out"] = "false"
    failures = validate_row(
        entry,
        source_row=SOURCE_ROW,
        arm="base_smart",
        parameters=list(privacy_broken.items()),
        expected_audio_sha256=SOURCE_ROW["audio_sha256"],
    )
    assert any("mip_opt_out" in item for item in failures)


def test_validation_rejects_model_drift(fixture_payload: dict[str, Any]) -> None:
    entry, parameters = normalize_helper(fixture_payload, "base_smart")
    entry["model"]["requested_model"] = "nova-2"
    failures = validate_row(
        entry,
        source_row=SOURCE_ROW,
        arm="base_smart",
        parameters=parameters,
        expected_audio_sha256=SOURCE_ROW["audio_sha256"],
    )
    assert any("nova-3" in item for item in failures)


def test_normalization_rejects_unknown_arm(fixture_payload: dict[str, Any]) -> None:
    with pytest.raises(NormalizationError, match="unknown arm"):
        normalize_response(
            fixture_payload,
            raw_bytes=b"{}",
            row=SOURCE_ROW,
            arm="bogus_arm",
            parameters=[],
            elapsed_ms=1.0,
            attempts=1,
            request_tag="t",
        )


def test_sha256_bytes_is_stable() -> None:
    assert sha256_bytes(b"phonon") == hashlib.sha256(b"phonon").hexdigest()
