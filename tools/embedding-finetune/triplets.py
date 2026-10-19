#!/usr/bin/env python3
"""Shared schema and privacy-safe output helpers for embedding triplets."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from time import time_ns
from typing import Any

SCHEMA_VERSION = 1
VALID_KINDS = frozenset(
    {
        "public_text",
        "public_qa",
        "public_code",
        "teacher_consistency",
        "text_audio",
        "term_context",
        "replacement",
    }
)
AUDIO_URI = re.compile(r"audio://[0-9a-f]{64}\.wav")


class TripletError(ValueError):
    """Raised when a triplet is malformed or an output path is unsafe."""


def validate_triplet(row: dict[str, Any]) -> None:
    """Validate one record before it can enter a local training file."""
    required = {
        "schema_version",
        "kind",
        "source",
        "source_revision",
        "anchor",
        "positive",
        "negative",
    }
    missing = required - set(row)
    if missing:
        raise TripletError(f"triplet lacks fields: {sorted(missing)}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise TripletError("unsupported triplet schema version")
    if row["kind"] not in VALID_KINDS:
        raise TripletError(f"unsupported triplet kind: {row['kind']!r}")
    for field in ("anchor", "positive", "negative"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise TripletError(f"triplet field {field} must be nonempty text/path")
    if row["kind"] == "text_audio":
        for field in ("positive", "negative"):
            if AUDIO_URI.fullmatch(row[field]) is None:
                raise TripletError(
                    "text_audio fields must be portable audio://<sha256>.wav URIs"
                )
    if len({row["anchor"], row["positive"], row["negative"]}) != 3:
        raise TripletError("anchor, positive, and negative must be distinct")


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, private: bool) -> None:
    """Atomically write validated JSONL with user-only permissions when private."""
    path.parent.mkdir(mode=0o700 if private else 0o755, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time_ns()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode=0o600 if private else 0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for row in rows:
                validate_triplet(row)
                output.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any, *, private: bool) -> None:
    """Atomically write JSON with restrictive permissions when private."""
    path.parent.mkdir(mode=0o700 if private else 0o755, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time_ns()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode=0o600 if private else 0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    """Return a durable SHA-256 for a downloaded seed or model artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
