#!/usr/bin/env python3
"""Resumable concurrent annotation runner with aggregate-only receipts."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from deepgram_client import (
    ARMS,
    MAX_ATTEMPTS,
    ClientError,
    DeepgramClient,
    keyterm_profile_metadata,
    read_keyring_credential,
)
from normalize import NormalizationError, normalize_response
from validate import ValidationError, validate_row


class RunnerError(RuntimeError):
    """Raised when the runner cannot complete its work safely."""


@dataclass(frozen=True)
class RunResult:
    """The aggregate outcome of one runner invocation."""

    completed: int
    failed: int
    skipped: int
    failures: list[str]


def load_keyterms(path: Path) -> list[str]:
    """Read the keyterm profile and return its terms in file order."""
    from deepgram_client import load_keyterms as _load

    return _load(path)


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(UTC).isoformat()


def audio_row_identity(row_id: str, audio_sha256: str, arm: str) -> str:
    """Return the stable identity used for resume and idempotency."""
    return f"{row_id}|{audio_sha256}|{arm}"


def load_existing_identities(output_path: Path) -> set[str]:
    """Read the identities already present in an append-only output file."""
    identities: set[str] = set()
    if not output_path.exists():
        return identities
    try:
        text = output_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RunnerError(
            f"cannot read the output file {output_path}: {error}"
        ) from error
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RunnerError(
                f"{output_path}:{number} is not valid JSON; the append-only "
                "output must stay readable"
            ) from error
        identity = audio_row_identity(
            str(record.get("row_id") or ""),
            str(record.get("audio_sha256") or ""),
            str(record.get("arm") or ""),
        )
        identities.add(identity)
    return identities


def load_audio_bytes(
    row: dict[str, Any],
    *,
    expected_sha256: str | None = None,
) -> bytes:
    """Load audio bytes from a path or inline base64 and verify the hash."""
    import base64
    import hashlib

    if row.get("audio_bytes_base64"):
        payload = base64.b64decode(row["audio_bytes_base64"], validate=True)
    elif row.get("audio_path"):
        path = Path(row["audio_path"])
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise RunnerError(
                f"cannot read the audio file for row {row['row_id']}: {error}"
            ) from error
    else:
        raise RunnerError(
            f"row {row['row_id']} has neither audio_path nor audio_bytes_base64"
        )
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RunnerError(
            f"row {row['row_id']} audio hash {digest} does not match the manifest "
            f"hash {expected_sha256}"
        )
    return payload


def archive_raw_response(
    archive_dir: Path,
    row: dict[str, Any],
    arm: str,
    raw_bytes: bytes,
) -> Path:
    """Archive the raw provider bytes under a content-addressed path."""
    import hashlib

    digest = hashlib.sha256(raw_bytes).hexdigest()
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / row["audio_sha256"] / f"{row['row_id']}_{arm}_{digest}.json"
    if target.exists():
        return target
    payload_dir = target.parent
    payload_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(raw_bytes)
    temporary.replace(target)
    return target


def request_tag(row: dict[str, Any]) -> str:
    """Return the opaque request tag for a row and arm."""
    import hashlib

    digest = hashlib.sha256(
        f"{row['row_id']}|{row['audio_sha256']}".encode()
    ).hexdigest()
    return digest[:16]


def process_row_arm(
    client: DeepgramClient,
    row: dict[str, Any],
    arm: str,
    *,
    keyterms: Sequence[str] | None,
    keyterm_metadata: dict[str, Any] | None,
    output_path: Path,
    archive_dir: Path,
    request_tag_value: str,
    output_lock: Lock,
) -> str:
    """Run one row/arm unit and append its normalized row to the output."""
    from deepgram_client import arm_parameters as _arm_parameters

    parameters = _arm_parameters(arm, keyterms=keyterms)
    parameters.append(("tag", request_tag_value))
    audio = load_audio_bytes(row, expected_sha256=row["audio_sha256"])
    result = client.listen(audio, parameters=parameters)
    raw_path = archive_raw_response(archive_dir, row, arm, result.raw_bytes)
    try:
        entry = normalize_response(
            result.payload,
            raw_bytes=result.raw_bytes,
            row=row,
            arm=arm,
            parameters=parameters,
            elapsed_ms=result.elapsed_ms,
            attempts=result.attempts,
            request_tag=request_tag_value,
            keyterm_metadata=keyterm_metadata,
        )
    except NormalizationError as error:
        raise RunnerError(
            f"row {row['row_id']} arm {arm} could not be normalized: {error}"
        ) from error

    entry["raw_response_path"] = str(raw_path)
    failures = validate_row(
        entry,
        source_row=row,
        arm=arm,
        parameters=parameters,
        expected_audio_sha256=row["audio_sha256"],
    )
    if failures:
        raise RunnerError(
            f"row {row['row_id']} arm {arm} failed validation: {'; '.join(failures)}"
        )
    entry["raw_response_path"] = str(raw_path)
    from manifest import append_jsonl

    with output_lock:
        append_jsonl(output_path, entry)
    return "completed"


def run(
    manifest_path: Path,
    output_path: Path,
    archive_dir: Path,
    receipt_path: Path,
    *,
    arms: Sequence[str],
    keyterm_path: Path | None = None,
    api_key: str | None = None,
    base_url: str = "https://api.deepgram.com",
    concurrency: int = 5,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Any = time.sleep,
    opener: Any = None,
    request_tag_prefix: str = "phonon-dg-v1",
) -> RunResult:
    """Run the annotation plan over a manifest with bounded concurrency."""
    from manifest import load_jsonl

    if concurrency < 1 or concurrency > 5:
        raise RunnerError("concurrency must be between 1 and 5")
    if not arms or any(arm not in ARMS for arm in arms):
        raise RunnerError("arms must be a nonempty subset of the four supported arms")

    rows = [
        row
        for row in load_jsonl(manifest_path)
        if row.get("split") in ("train", "eval", "test")
    ]
    if not rows:
        raise RunnerError(f"no train/eval/test rows in the manifest {manifest_path}")

    keyterms: list[str] | None = None
    keyterm_metadata: dict[str, Any] | None = None
    if any(arm.startswith("keyterm_") for arm in arms):
        if keyterm_path is None:
            raise RunnerError("keyterm arms require a keyterm profile path")
        keyterms = load_keyterms(keyterm_path)
        keyterm_metadata = keyterm_profile_metadata(keyterm_path)

    if api_key is None:
        api_key = read_keyring_credential(profile="default")

    existing = load_existing_identities(output_path)
    client = DeepgramClient(
        api_key,
        base_url=base_url,
        max_attempts=max_attempts,
        sleep=sleep,
        opener=opener,
    )

    pending: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        for arm in arms:
            if (
                audio_row_identity(row["row_id"], row["audio_sha256"], arm)
                not in existing
            ):
                pending.append((row, arm))
    completed = 0
    failed = 0
    failures: list[str] = []
    failed_by_arm: dict[str, int] = {}
    output_lock = Lock()
    if not pending:
        return RunResult(
            completed=0, failed=0, skipped=len(rows) * len(arms), failures=[]
        )

    def task(item: tuple[dict[str, Any], str]) -> str:
        row, arm = item
        return process_row_arm(
            client,
            row,
            arm,
            keyterms=keyterms,
            keyterm_metadata=keyterm_metadata,
            output_path=output_path,
            archive_dir=archive_dir,
            request_tag_value=request_tag_prefix + "-" + request_tag(row),
            output_lock=output_lock,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(task, item) for item in pending]
        for future, item in zip(futures, pending, strict=False):
            row, arm = item
            try:
                future.result()
                completed += 1
            except (
                ClientError,
                NormalizationError,
                ValidationError,
                RunnerError,
                OSError,
                ValueError,
            ) as error:
                failed += 1
                failures.append(f"arm {arm}: {type(error).__name__}")
                failed_by_arm[arm] = failed_by_arm.get(arm, 0) + 1

    skipped = len(rows) * len(arms) - len(pending)
    receipt = {
        "schema_version": 1,
        "recorded_at": utc_now(),
        "manifest": str(manifest_path),
        "output": str(output_path),
        "archive": str(archive_dir),
        "arms": list(arms),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "failed_by_arm": dict(sorted(failed_by_arm.items())),
        "failures": failures[:100],
        "concurrency": concurrency,
        "split_rows": {
            split: sum(1 for row in rows if row.get("split") == split)
            for split in ("train", "eval", "test")
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return RunResult(
        completed=completed,
        failed=failed,
        skipped=skipped,
        failures=failures,
    )
