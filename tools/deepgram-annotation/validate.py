#!/usr/bin/env python3
"""Validate normalized annotation rows before they enter the output stream.

Each gate reports one concrete failure so an operator can repair the input
without re-running the whole pipeline. A failed gate marks the row as failed
and leaves the append-only output unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from deepgram_client import ARMS, EXPECTED_MODEL_UUID, SMART_FEATURES


class ValidationError(ValueError):
    """Raised when a normalized annotation row fails a validation gate."""


def validate_row(
    entry: dict[str, Any],
    *,
    source_row: dict[str, Any],
    arm: str,
    parameters: Sequence[tuple[str, str]],
    expected_audio_sha256: str,
) -> list[str]:
    """Validate one normalized annotation row, returning failure reasons."""
    failures: list[str] = []
    if entry["row_id"] != source_row["row_id"]:
        failures.append("row_id does not match the source manifest")
    if entry["audio_sha256"] != expected_audio_sha256:
        failures.append("audio_sha256 does not match the source manifest")
    if arm not in ARMS:
        failures.append(f"unknown arm {arm!r}")
    parameter_map = dict(parameters)
    expected_features = SMART_FEATURES if arm.endswith("_smart") else set()
    expected_values = {"diarize_model": "v2", "summarize": "v2"}
    for name in expected_features:
        wanted = expected_values.get(name, "true")
        if parameter_map.get(name) != wanted:
            failures.append(f"smart arm {arm!r} must request {name}={wanted}")
    if arm.endswith("_raw"):
        for name in SMART_FEATURES:
            if name in parameter_map:
                failures.append(f"raw arm {arm!r} must not request {name}")
    if parameter_map.get("model") != "nova-3":
        failures.append("the request must pin model=nova-3")
    if parameter_map.get("version") != "2025-07-31.0":
        failures.append("the request must pin version=2025-07-31.0")
    if parameter_map.get("mip_opt_out") != "true":
        failures.append("the request must pin mip_opt_out=true")
    if entry.get("mip_opt_out") is not True:
        failures.append("mip_opt_out must be true in the normalized row")
    if not entry.get("provider_request_id"):
        failures.append("the provider response has no request id")
    model = entry.get("model") or {}
    if model.get("requested_model") != "nova-3":
        failures.append("the response must identify nova-3")
    if model.get("returned_uuid") != EXPECTED_MODEL_UUID:
        failures.append(f"the response model UUID must be {EXPECTED_MODEL_UUID}")
    if model.get("returned_architecture") != "nova-3":
        failures.append("the response must report architecture nova-3")
    if not entry.get("provider_payload_sha256"):
        failures.append("the normalized row has no provider payload hash")
    if entry.get("transcript") is None:
        failures.append("the response has no transcript")

    for word in entry.get("words") or []:
        start = word.get("start")
        end = word.get("end")
        if start is None or end is None:
            failures.append(f"word {word.get('index')} has no start or end")
            continue
        if start < -0.001 or end < -0.001:
            failures.append(f"word {word['index']} has a negative timestamp")
        if start > end + 0.001:
            failures.append(f"word {word['index']} starts after it ends")
        confidence = word.get("confidence")
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            failures.append(
                f"word {word['index']} confidence {confidence} is out of range"
            )

    for utterance in entry.get("utterances") or []:
        start = utterance.get("start")
        end = utterance.get("end")
        if start is None or end is None:
            failures.append("an utterance has no start or end")
            continue
        if start < -0.001 or end < -0.001:
            failures.append("an utterance has a negative timestamp")
        if start > end + 0.001:
            failures.append("an utterance starts after it ends")
    return failures


def validate_provider_identity(
    payload: dict[str, Any],
    *,
    expected_model: str = "nova-3",
) -> None:
    """Check the provider payload reports the pinned model family."""
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValidationError("the provider response has no metadata object")
    models = metadata.get("models")
    if not isinstance(models, list) or not models:
        raise ValidationError("the provider response has no models list")
    names = [str(name).lower() for name in models if isinstance(name, str)]
    if expected_model not in names:
        raise ValidationError(
            f"the provider response does not identify the {expected_model} model"
        )
