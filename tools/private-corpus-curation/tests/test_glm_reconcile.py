from __future__ import annotations

import json

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
