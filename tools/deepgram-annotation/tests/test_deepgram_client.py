from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import deepgram_client
import pytest
from deepgram_client import (
    DeepgramClient,
    arm_parameters,
    encode_query,
    load_keyterms,
    read_keyring_credential,
)


class FakeResponse:
    def __init__(
        self, payload: bytes, status: int = 200, headers: dict | None = None
    ) -> None:
        self._io = io.BytesIO(payload)
        self.status = status
        self.headers = headers or {}

    def read(self, amount: int = -1) -> bytes:
        return self._io.read(amount)

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_arm_parameters_pin_model_version_and_privacy() -> None:
    parameters = dict(arm_parameters("base_raw"))
    assert parameters["model"] == "nova-3"
    assert parameters["version"] == "2025-07-31.0"
    assert parameters["mip_opt_out"] == "true"
    assert parameters["smart_format"] == "false"
    assert parameters["punctuate"] == "false"

    smart = dict(arm_parameters("base_smart"))
    assert smart["smart_format"] == "true"
    assert smart["punctuate"] == "true"
    assert smart["diarize_model"] == "v2"
    assert smart["summarize"] == "v2"
    assert smart["paragraphs"] == "true"
    assert smart["utterances"] == "true"
    assert smart["filler_words"] == "true"
    assert smart["detect_entities"] == "true"
    assert smart["topics"] == "true"
    assert smart["intents"] == "true"
    assert smart["sentiment"] == "true"


def test_keyterm_arm_requires_terms() -> None:
    with pytest.raises(deepgram_client.ClientError, match="keyterm"):
        arm_parameters("keyterm_raw")
    terms = ["CUDA", "GPU"]
    parameters = arm_parameters("keyterm_raw", keyterms=terms)
    values = [value for name, value in parameters if name == "keyterm"]
    assert values == terms


def test_encode_query_repeats_keyterm_parameters() -> None:
    query = encode_query(arm_parameters("keyterm_raw", keyterms=["A B", "C"]))
    assert query.count("keyterm=") == 2
    assert "A%20B" in query


def test_keyterm_profile_reader(tmp_path: Path) -> None:
    profile = tmp_path / "keyterms.json"
    profile.write_text(
        json.dumps({"profile_id": "p", "sha256": "h", "terms": ["A", "B"]}),
        encoding="utf-8",
    )
    assert load_keyterms(profile) == ["A", "B"]
    meta = deepgram_client.keyterm_profile_metadata(profile)
    assert meta["term_count"] == 2
    assert meta["profile_id"] == "p"
    duplicate = tmp_path / "dupe.json"
    duplicate.write_text(json.dumps({"terms": ["A", "A"]}), encoding="utf-8")
    with pytest.raises(deepgram_client.ClientError, match="duplicate"):
        load_keyterms(duplicate)


def test_keyring_credential_uses_deepctl_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs

        class Completed:
            returncode = 0
            stdout = "dg_secret"
            stderr = ""

        return Completed()

    monkeypatch.setattr(deepgram_client.subprocess, "run", fake_run)
    assert read_keyring_credential("default") == "dg_secret"
    command = seen["command"]
    assert isinstance(command, list)
    assert deepgram_client.KEYRING_SERVICE in command
    assert "api-key.default" in command


def test_keyring_missing_credential_raises() -> None:
    class Failed:
        returncode = 44
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        return Failed()

    deepgram_client.subprocess.run = fake_run
    with pytest.raises(deepgram_client.CredentialError, match="no Deepgram credential"):
        read_keyring_credential("missing")


class RetrySchedule:
    def __init__(self) -> None:
        self.calls: list[float] = []


def test_client_retries_on_429_with_retry_after() -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    def opener(request, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                {"Retry-After": "7"},
                io.BytesIO(b"{}"),
            )
        return FakeResponse(
            json.dumps({"metadata": {"request_id": "r1"}}).encode(), 200
        )

    client = DeepgramClient("key", opener=opener, sleep=sleeps.append, max_attempts=5)
    result = client.listen(b"audio", parameters=arm_parameters("base_raw"))
    assert result.status_code == 200
    assert result.attempts == 3
    assert sleeps == [7.0, 7.0]


def test_client_backoff_on_500_without_retry_after() -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    def opener(request, timeout=None):
        attempts.append(1)
        if len(attempts) < 2:
            raise urllib.error.HTTPError(
                request.full_url, 500, "boom", {}, io.BytesIO(b"")
            )
        return FakeResponse(
            json.dumps({"metadata": {"request_id": "r1"}}).encode(), 200
        )

    client = DeepgramClient("key", opener=opener, sleep=sleeps.append, max_attempts=5)
    result = client.listen(b"audio", parameters=arm_parameters("base_raw"))
    assert result.attempts == 2
    assert sleeps and sleeps[0] == 1.0


def test_client_gives_up_after_max_attempts() -> None:
    sleeps: list[float] = []

    def opener(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 503, "unavailable", {}, io.BytesIO(b"")
        )

    client = DeepgramClient("key", opener=opener, sleep=sleeps.append, max_attempts=3)
    with pytest.raises(deepgram_client.ClientError, match="503"):
        client.listen(b"audio", parameters=arm_parameters("base_raw"))
    assert len(sleeps) == 2
