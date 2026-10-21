#!/usr/bin/env python3
"""Normalize Deepgram listen responses into append-only annotation rows.

Each normalized row keeps the full provider payload hash alongside the
surfaces used downstream: word text with start, end, confidence, speaker, and
the punctuated surface; speaker turns; utterances; paragraphs and sentences;
entities; topics, intents, and sentiment; summary text; model and request
provenance; timings; and the arm metadata that produced the request.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

SCHEMA_VERSION = 1

MODEL = "nova-3"
MODEL_VERSION = "2025-07-31.0"

ARM_NAMES: tuple[str, ...] = (
    "base_raw",
    "base_smart",
    "keyterm_raw",
    "keyterm_smart",
)

# Rich features that every smart arm must request.
SMART_FEATURES: frozenset[str] = frozenset(
    {
        "diarize_model",
        "paragraphs",
        "utterances",
        "filler_words",
        "detect_entities",
        "topics",
        "intents",
        "sentiment",
        "summarize",
    }
)

# Features that every raw arm must not request.
RAW_ONLY_FORBIDDEN: frozenset[str] = SMART_FEATURES

TIMESTAMP_TOLERANCE = 0.001


class NormalizationError(ValueError):
    """Raised when a provider payload cannot be normalized."""


class ValidationError(ValueError):
    """Raised when a normalized annotation row fails a validation gate."""


def canonical_json(value: Any) -> str:
    """Return a stable JSON rendering used for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def _as_float(value: Any) -> float | None:
    """Return a float for a numeric provider value, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _first_channel(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the first channel of a listen response."""
    channels = payload.get("results", {}).get("channels") or []
    if not channels:
        raise NormalizationError("the response has no result channels")
    first = channels[0]
    if not isinstance(first, dict):
        raise NormalizationError("the response channel is not an object")
    return first


def _first_alternative(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the first alternative of the first channel."""
    alternatives = _first_channel(payload).get("alternatives") or []
    if not alternatives:
        raise NormalizationError("the response channel has no alternatives")
    first = alternatives[0]
    if not isinstance(first, dict):
        raise NormalizationError("the response alternative is not an object")
    return first


def normalize_words(alternative: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize word records with timestamps, confidence, and surfaces."""
    words: list[dict[str, Any]] = []
    for index, word in enumerate(alternative.get("words") or []):
        if not isinstance(word, dict):
            raise NormalizationError(f"word {index} is not an object")
        start = _as_float(word.get("start"))
        end = _as_float(word.get("end"))
        entry: dict[str, Any] = {
            "index": index,
            "word": word.get("word"),
            "start": start,
            "end": end,
            "confidence": _as_float(word.get("confidence")),
        }
        punctuated = word.get("punctuated_word")
        if punctuated is not None:
            entry["punctuated_word"] = punctuated
        if word.get("speaker") is not None:
            entry["speaker"] = word.get("speaker")
        if word.get("speaker_confidence") is not None:
            entry["speaker_confidence"] = _as_float(word.get("speaker_confidence"))
        words.append(entry)
    return words


def normalize_turns(words: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group words into contiguous speaker turns."""
    turns: list[dict[str, Any]] = []
    for word in words:
        speaker = word.get("speaker")
        if turns and turns[-1]["speaker"] == speaker:
            turn = turns[-1]
            turn["end"] = word.get("end", turn["end"])
            turn["word_indices"].append(word["index"])
            turn["word_count"] += 1
            continue
        turns.append(
            {
                "speaker": speaker,
                "start": word.get("start"),
                "end": word.get("end"),
                "word_indices": [word["index"]],
                "word_count": 1,
            }
        )
    return turns


def normalize_utterances(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize utterance records with their word surfaces."""
    utterances: list[dict[str, Any]] = []
    container = payload.get("results", {}).get("utterances") or []
    for index, utterance in enumerate(container):
        if not isinstance(utterance, dict):
            raise NormalizationError(f"utterance {index} is not an object")
        entry: dict[str, Any] = {
            "index": index,
            "start": _as_float(utterance.get("start")),
            "end": _as_float(utterance.get("end")),
            "confidence": _as_float(utterance.get("confidence")),
            "speaker": utterance.get("speaker"),
            "transcript": utterance.get("transcript"),
        }
        if utterance.get("id") is not None:
            entry["utterance_id"] = utterance.get("id")
        word_entries: list[dict[str, Any]] = []
        for word in utterance.get("words") or []:
            if not isinstance(word, dict):
                continue
            word_entry: dict[str, Any] = {
                "word": word.get("word"),
                "start": _as_float(word.get("start")),
                "end": _as_float(word.get("end")),
                "confidence": _as_float(word.get("confidence")),
            }
            if word.get("punctuated_word") is not None:
                word_entry["punctuated_word"] = word.get("punctuated_word")
            if word.get("speaker") is not None:
                word_entry["speaker"] = word.get("speaker")
            word_entries.append(word_entry)
        entry["words"] = word_entries
        utterances.append(entry)
    return utterances


def normalize_paragraphs(alternative: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize paragraph and sentence spans."""
    paragraphs: list[dict[str, Any]] = []
    container = alternative.get("paragraphs") or {}
    if not isinstance(container, dict):
        return paragraphs
    for index, paragraph in enumerate(container.get("paragraphs") or []):
        if not isinstance(paragraph, dict):
            raise NormalizationError(f"paragraph {index} is not an object")
        sentences: list[dict[str, Any]] = []
        for sentence in paragraph.get("sentences") or []:
            if not isinstance(sentence, dict):
                continue
            sentences.append(
                {
                    "text": sentence.get("text"),
                    "start": _as_float(sentence.get("start")),
                    "end": _as_float(sentence.get("end")),
                }
            )
        paragraphs.append(
            {
                "index": index,
                "speaker": paragraph.get("speaker"),
                "start": _as_float(paragraph.get("start")),
                "end": _as_float(paragraph.get("end")),
                "num_words": paragraph.get("num_words"),
                "sentences": sentences,
            }
        )
    return paragraphs


def normalize_entities(alternative: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize entity spans with their word-index bounds."""
    entities: list[dict[str, Any]] = []
    for index, entity in enumerate(alternative.get("entities") or []):
        if not isinstance(entity, dict):
            raise NormalizationError(f"entity {index} is not an object")
        entities.append(
            {
                "index": index,
                "label": entity.get("label"),
                "value": entity.get("value"),
                "raw_value": entity.get("raw_value"),
                "confidence": _as_float(entity.get("confidence")),
                "start_word": _as_float(entity.get("start_word")),
                "end_word": _as_float(entity.get("end_word")),
            }
        )
    return entities


def _normalize_segments(container: Any, label: str) -> list[dict[str, Any]]:
    """Normalize topic or intent segments with their child candidates."""
    segments: list[dict[str, Any]] = []
    if not isinstance(container, dict):
        return segments
    for index, segment in enumerate(container.get("segments") or []):
        if not isinstance(segment, dict):
            raise NormalizationError(f"{label} segment {index} is not an object")
        candidates: list[dict[str, Any]] = []
        for candidate in segment.get(label) or []:
            if not isinstance(candidate, dict):
                continue
            candidates.append(
                {
                    "value": candidate.get("topic")
                    if label == "topics"
                    else candidate.get("intent"),
                    "confidence": _as_float(candidate.get("confidence_score")),
                }
            )
        segments.append(
            {
                "index": index,
                "text": segment.get("text"),
                "start_word": _as_float(segment.get("start_word")),
                "end_word": _as_float(segment.get("end_word")),
                label: candidates,
            }
        )
    return segments


def normalize_sentiments(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize the document-level sentiment average and its segments."""
    container = payload.get("results", {}).get("sentiments") or {}
    if not isinstance(container, dict):
        return {"average": None, "segments": []}
    average = container.get("average")
    average_entry: dict[str, Any] | None = None
    if isinstance(average, dict):
        average_entry = {
            "sentiment": average.get("sentiment"),
            "sentiment_score": _as_float(average.get("sentiment_score")),
        }
    segments: list[dict[str, Any]] = []
    for index, segment in enumerate(container.get("segments") or []):
        if not isinstance(segment, dict):
            continue
        segments.append(
            {
                "index": index,
                "text": segment.get("text"),
                "start_word": _as_float(segment.get("start_word")),
                "end_word": _as_float(segment.get("end_word")),
                "sentiment": segment.get("sentiment"),
                "sentiment_score": _as_float(segment.get("sentiment_score")),
            }
        )
    return {"average": average_entry, "segments": segments}


def normalize_summary(
    payload: dict[str, Any], alternative: dict[str, Any]
) -> dict[str, Any]:
    """Normalize the summary result and any per-paragraph summaries."""
    container = payload.get("results", {}).get("summary") or {}
    entry: dict[str, Any] = {"result": None, "short": None}
    if isinstance(container, dict):
        entry["result"] = container.get("result")
        entry["short"] = container.get("short")
    paragraphs: list[dict[str, Any]] = []
    for index, summary in enumerate(alternative.get("summaries") or []):
        if not isinstance(summary, dict):
            continue
        paragraphs.append(
            {
                "index": index,
                "summary": summary.get("summary"),
                "start_word": _as_float(summary.get("start_word")),
                "end_word": _as_float(summary.get("end_word")),
            }
        )
    entry["paragraphs"] = paragraphs
    return entry


def model_identity(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the model identity the provider reported."""
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    model_info = metadata.get("model_info") or {}
    if not isinstance(model_info, dict):
        model_info = {}
    names = metadata.get("models")
    first_model = names[0] if isinstance(names, list) and names else None
    # Deepgram's live response is keyed by model UUID. Accept the fixture's
    # pre-flattened form as well so tests exercise the same normalization path.
    if model_info and "arch" not in model_info and "uuid" not in model_info:
        model_info = next(iter(model_info.values()), {})
        if not isinstance(model_info, dict):
            model_info = {}
    returned_uuid = model_info.get("uuid") or first_model
    return {
        "requested_model": MODEL,
        "requested_version": MODEL_VERSION,
        "returned_name": model_info.get("name"),
        "returned_uuid": returned_uuid,
        "returned_architecture": model_info.get("arch"),
    }


def normalize_response(
    payload: dict[str, Any],
    *,
    raw_bytes: bytes,
    row: dict[str, Any],
    arm: str,
    parameters: Sequence[tuple[str, str]],
    elapsed_ms: float,
    attempts: int,
    request_tag: str,
    keyterm_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one provider response into an append-only annotation row."""
    if arm not in ARM_NAMES:
        raise NormalizationError(f"unknown arm: {arm}")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    alternative = _first_alternative(payload)
    words = normalize_words(alternative)
    turns = normalize_turns(words)
    word_ends = [word["end"] for word in words if word.get("end") is not None]
    provider_duration = _as_float(metadata.get("duration"))
    quality_flags: list[str] = []
    if (
        word_ends
        and provider_duration is not None
        and max(word_ends) > provider_duration + 0.25
    ):
        quality_flags.append("last_word_extends_beyond_provider_duration")
    if provider_duration is not None and abs(
        provider_duration - float(row["duration_seconds"])
    ) > max(0.05, float(row["duration_seconds"]) * 0.005):
        quality_flags.append("provider_duration_differs_from_manifest")

    entry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "row_id": row["row_id"],
        "audio_sha256": row["audio_sha256"],
        "duration_seconds": row["duration_seconds"],
        "leakage_group": row["leakage_group"],
        "source_kind": row["source_kind"],
        "source_id": row["source_id"],
        "split": row.get("split"),
        "arm": arm,
        "provider": "deepgram",
        "endpoint": "/v1/listen",
        "request_tag": request_tag,
        "request_parameters": [
            {"name": name, "value": value} for name, value in parameters
        ],
        "mip_opt_out": True,
        "provider_payload_sha256": sha256_bytes(raw_bytes),
        "provider_payload_canonical_sha256": sha256_bytes(
            canonical_json(payload).encode("utf-8")
        ),
        "provider_request_id": metadata.get("request_id"),
        "provider_created": metadata.get("created"),
        "provider_duration_seconds": _as_float(metadata.get("duration")),
        "provider_channels": metadata.get("channels"),
        "provider_tags": metadata.get("tags"),
        "model": model_identity(payload),
        "transcript": alternative.get("transcript"),
        "transcript_confidence": _as_float(alternative.get("confidence")),
        "words": words,
        "turns": turns,
        "quality": {
            "flags": quality_flags,
            "last_word_end_seconds": max(word_ends) if word_ends else None,
            "manifest_duration_seconds": float(row["duration_seconds"]),
            "provider_duration_seconds": provider_duration,
        },
        "utterances": normalize_utterances(payload),
        "paragraphs": normalize_paragraphs(alternative),
        "entities": normalize_entities(alternative),
        "topics": _normalize_segments(
            payload.get("results", {}).get("topics"), "topics"
        ),
        "intents": _normalize_segments(
            payload.get("results", {}).get("intents"), "intents"
        ),
        "sentiment": normalize_sentiments(payload),
        "summary": normalize_summary(payload, alternative),
        "timings": {
            "provider_elapsed_ms": round(elapsed_ms, 3),
            "attempts": attempts,
        },
    }
    if keyterm_metadata is not None:
        entry["keyterm_profile"] = keyterm_metadata
    return entry
