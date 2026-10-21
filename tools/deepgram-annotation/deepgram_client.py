#!/usr/bin/env python3
"""Deepgram ``/v1/listen`` client with keyring-only credentials and retries.

The API key is read from the macOS keyring using the deepctl credential
service and account. It is never printed, logged, archived, or written into
a receipt. Only the request parameters, timings, and response hashes are
recorded.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL = "nova-3"
MODEL_VERSION = "2025-07-31.0"
EXPECTED_MODEL_UUID = "2187e11a-3532-4498-b076-81fa530bdd49"
DEFAULT_BASE_URL = "https://api.deepgram.com"
LISTEN_PATH = "/v1/listen"

# The deepctl credential service and account used by the official CLI.
KEYRING_SERVICE = "com.deepgram.dx.deepctl"
KEYRING_ACCOUNT_TEMPLATE = "api-key.{profile}"

DEFAULT_CONCURRENCY = 5
MAX_CONCURRENCY = 5
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT_SECONDS = 300.0
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 30.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

ARM_NAMES: tuple[str, ...] = (
    "base_raw",
    "base_smart",
    "keyterm_raw",
    "keyterm_smart",
)

# Rich features requested for the smart arms only.
SMART_FEATURES: dict[str, str] = {
    "diarize_model": "v2",
    "paragraphs": "true",
    "utterances": "true",
    "filler_words": "true",
    "detect_entities": "true",
    "topics": "true",
    "intents": "true",
    "sentiment": "true",
    "summarize": "v2",
}


class ClientError(RuntimeError):
    """Raised when a request cannot be completed or a response is unusable."""


class CredentialError(ClientError):
    """Raised when the macOS keyring holds no usable Deepgram credential."""


@dataclass(frozen=True)
class Arm:
    """One frozen transcription factor of the annotation design."""

    name: str
    smart: bool
    keyterms: bool


ARMS: dict[str, Arm] = {
    "base_raw": Arm(name="base_raw", smart=False, keyterms=False),
    "base_smart": Arm(name="base_smart", smart=True, keyterms=False),
    "keyterm_raw": Arm(name="keyterm_raw", smart=False, keyterms=True),
    "keyterm_smart": Arm(name="keyterm_smart", smart=True, keyterms=True),
}


def arm_parameters(
    arm: str, keyterms: Sequence[str] | None = None
) -> list[tuple[str, str]]:
    """Return the ordered query parameters for one annotation arm.

    ``keyterm`` repeats once per term because the provider expects repeated
    query parameters rather than a delimited list.
    """
    if arm not in ARMS:
        raise ClientError(f"unknown arm: {arm}")
    spec = ARMS[arm]
    parameters: list[tuple[str, str]] = [
        ("model", MODEL),
        ("version", MODEL_VERSION),
        ("mip_opt_out", "true"),
    ]
    if spec.smart:
        for name, value in SMART_FEATURES.items():
            parameters.append((name, value))
        parameters.append(("smart_format", "true"))
        parameters.append(("punctuate", "true"))
    else:
        parameters.append(("smart_format", "false"))
        parameters.append(("punctuate", "false"))
    if spec.keyterms:
        terms = list(keyterms or [])
        if not terms:
            raise ClientError(f"arm {arm} requires at least one keyterm")
        parameters.extend(("keyterm", term) for term in terms)
    return parameters


def encode_query(parameters: Sequence[tuple[str, str]]) -> str:
    """Encode query parameters, repeating keys where the provider expects it."""
    return urllib.parse.urlencode(parameters, quote_via=urllib.parse.quote)


def load_keyterms(path: Path) -> list[str]:
    """Read the keyterm profile and return its terms in file order."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ClientError(f"cannot read keyterm profile {path}: {error}") from error
    terms = payload.get("terms")
    if not isinstance(terms, list) or not terms:
        raise ClientError(f"keyterm profile {path} has no terms")
    cleaned: list[str] = []
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            raise ClientError(f"keyterm profile {path} has an empty term")
        cleaned.append(term.strip())
    if len(set(cleaned)) != len(cleaned):
        raise ClientError(f"keyterm profile {path} has duplicate terms")
    return cleaned


def keyterm_profile_metadata(path: Path) -> dict[str, Any]:
    """Return non-sensitive keyterm profile metadata for provenance."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ClientError(f"cannot read keyterm profile {path}: {error}") from error
    metadata: dict[str, Any] = {}
    for field_name in ("profile_id", "sha256", "schema", "created_at"):
        if isinstance(payload.get(field_name), str):
            metadata[field_name] = payload[field_name]
    metadata["term_count"] = len(payload.get("terms") or [])
    return metadata


def read_keyring_credential(profile: str = "default") -> str:
    """Read the Deepgram API key from the macOS keyring.

    The key is returned to the caller for use as an authorization header and
    is never printed or persisted by this module.
    """
    account = KEYRING_ACCOUNT_TEMPLATE.format(profile=profile)
    command = [
        "/usr/bin/security",
        "find-generic-password",
        "-s",
        KEYRING_SERVICE,
        "-a",
        account,
        "-w",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CredentialError(
            f"cannot read the Deepgram credential from the macOS keyring: {error}"
        ) from error
    if completed.returncode != 0:
        raise CredentialError(
            "no Deepgram credential in the macOS keyring for service "
            f"{KEYRING_SERVICE!r} and account {account!r}; run "
            "'deepctl login' to create one"
        )
    secret = completed.stdout.strip()
    if not secret:
        raise CredentialError("the macOS keyring returned an empty credential")
    return secret


@dataclass
class ListenResult:
    """The outcome of one provider request."""

    status_code: int
    payload: dict[str, Any]
    raw_bytes: bytes
    elapsed_ms: float
    attempts: int
    request_headers: dict[str, str] = field(default_factory=dict)

    @property
    def payload_sha256(self) -> str:
        """Return the SHA-256 of the raw response bytes."""
        import hashlib

        return hashlib.sha256(self.raw_bytes).hexdigest()


def _retry_delay(attempt: int, retry_after: float | None) -> float:
    """Return the bounded backoff delay before the next attempt."""
    if retry_after is not None:
        return min(max(retry_after, 0.0), RETRY_MAX_DELAY)
    return min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)


def _parse_retry_after(headers: Any) -> float | None:
    """Read the Retry-After header as seconds when it is a plain number."""
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


class DeepgramClient:
    """A small ``/v1/listen`` client with bounded retries and no key output."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        max_attempts: int = MAX_ATTEMPTS,
        sleep: Any = time.sleep,
        opener: Any = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._opener = opener or urllib.request.urlopen

    def listen(
        self,
        audio: bytes,
        *,
        parameters: Sequence[tuple[str, str]],
        content_type: str = "audio/wav",
    ) -> ListenResult:
        """POST audio to ``/v1/listen`` and return the parsed response."""
        url = f"{self.base_url}{LISTEN_PATH}?{encode_query(parameters)}"
        body = bytes(audio)
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": content_type,
            "Accept": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(
                url, data=body, headers=headers, method="POST"
            )
            started = time.monotonic()
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    raw = response.read()
                    status = int(getattr(response, "status", 200))
                    response_headers = dict(getattr(response, "headers", {}) or {})
                elapsed_ms = (time.monotonic() - started) * 1000.0
            except urllib.error.HTTPError as error:
                elapsed_ms = (time.monotonic() - started) * 1000.0
                status = int(error.code)
                response_headers = dict(error.headers or {})
                raw = error.read() if hasattr(error, "read") else b""
                if status in RETRYABLE_STATUS and attempt < self.max_attempts:
                    last_error = error
                    self._sleep(
                        _retry_delay(attempt, _parse_retry_after(response_headers))
                    )
                    continue
                raise ClientError(
                    f"Deepgram returned HTTP {status} for the listen request"
                ) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                elapsed_ms = (time.monotonic() - started) * 1000.0
                last_error = error
                if attempt < self.max_attempts:
                    self._sleep(_retry_delay(attempt, None))
                    continue
                raise ClientError(
                    "the Deepgram listen request failed after "
                    f"{self.max_attempts} attempts: {error}"
                ) from error

            if status in RETRYABLE_STATUS and attempt < self.max_attempts:
                last_error = ClientError(f"Deepgram returned HTTP {status}")
                self._sleep(_retry_delay(attempt, _parse_retry_after(response_headers)))
                continue
            if status >= 400:
                raise ClientError(
                    f"Deepgram returned HTTP {status} for the listen request"
                )
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ClientError(
                    "the Deepgram listen response was not valid JSON"
                ) from error
            if not isinstance(payload, dict):
                raise ClientError("the Deepgram listen response was not an object")
            return ListenResult(
                status_code=status,
                payload=payload,
                raw_bytes=raw,
                elapsed_ms=elapsed_ms,
                attempts=attempt,
                request_headers={"content_type": content_type},
            )
        raise ClientError(f"the Deepgram listen request failed: {last_error}")
