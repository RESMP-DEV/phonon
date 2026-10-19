from __future__ import annotations

import json
from pathlib import Path

import glm_reconcile as reconciler


def test_teacher_parser_accepts_strings_and_nested_records() -> None:
    """Teacher text is normalized without exposing it in a receipt."""
    assert reconciler.parse_teachers('{"a":"one"}') == {"a": "one"}
    assert reconciler.parse_teachers({"a": {"text": " one "}}) == {"a": "one"}
    assert reconciler.parse_teachers("raw") == {"raw": "raw"}


def test_json_payload_accepts_strict_schema() -> None:
    """Fenced strict JSON is accepted and confidence is normalized."""
    valid = {
        "decision": "revise",
        "transcript": "CUTLASS kernel",
        "confidence": 0.8,
        "reason": "teacher consensus",
    }
    fenced = f"```json\n{json.dumps(valid)}\n```"
    assert reconciler.json_payload(fenced) == valid


def test_json_payload_rejects_invalid_confidence() -> None:
    """Out-of-range confidence cannot enter the curation record."""
    invalid = {
        "decision": "keep",
        "transcript": "text",
        "confidence": 1.2,
        "reason": "invalid",
    }
    try:
        reconciler.json_payload(json.dumps(invalid))
    except reconciler.CurationError:
        pass
    else:
        raise AssertionError("invalid confidence accepted")


def test_existing_ids_rejects_duplicate_and_mixed_rows(tmp_path: Path) -> None:
    """Resume rows must carry the exact prompt/model provenance."""
    path = tmp_path / "reconciliation.jsonl"
    base = {
        "id": "row-a",
        "schema_version": reconciler.PROMPT_SCHEMA_VERSION,
        "model": "zai,glm-5.3",
        "prompt_schema_sha256": reconciler.PROMPT_SCHEMA_SHA256,
        "finish_reason": "stop",
        "decision": "keep",
        "transcript": "text",
        "confidence": 0.9,
        "reason": "test",
    }
    path.write_text(
        json.dumps(base) + "\n" + json.dumps(dict(base, model="other")) + "\n",
        encoding="utf-8",
    )

    try:
        reconciler.existing_ids(path, model="zai,glm-5.3")
    except reconciler.CurationError as error:
        assert "mixed provenance" in str(error)
    else:
        raise AssertionError("mixed reconciliation provenance accepted")
