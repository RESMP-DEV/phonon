from __future__ import annotations

import base64
import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest
import runner
from runner import (
    audio_row_identity,
    load_audio_bytes,
    load_existing_identities,
    request_tag,
    run,
)


class FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._io = io.BytesIO(payload)
        self.status = status
        self.headers = {}

    def read(self, amount: int = -1) -> bytes:
        return self._io.read(amount)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "listen_response.json"


def hashlib_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def source_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "row_id": "row-1",
        "audio_path": "/tmp/audio.wav",
        "audio_sha256": hashlib_sha256(b"audio-bytes"),
        "duration_seconds": 2.4,
        "leakage_group": "g1",
        "source_kind": "owned_public_youtube",
        "source_id": "src",
        "split": "train",
    }
    row.update(overrides)
    return row


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def make_opener(respond_with: bytes, statuses: list[int] | None = None):
    calls: list[str] = []

    def opener(request, timeout=None):
        index = min(len(calls), len(statuses or []) - 1)
        status = (statuses or [200])[index]
        calls.append(request.full_url)
        if status >= 400:
            raise urllib.error.HTTPError(
                request.full_url, status, "fail", {}, io.BytesIO(b"")
            )
        return FakeResponse(respond_with, status)

    return opener, calls


def test_row_identity_is_stable() -> None:
    assert audio_row_identity("r", "a", "arm") == "r|a|arm"


def test_existing_identities_are_loaded(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    output.write_text(
        json.dumps(
            {
                "row_id": "row-1",
                "audio_sha256": "a" * 64,
                "arm": "base_raw",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_existing_identities(output) == {
        audio_row_identity("row-1", "a" * 64, "base_raw")
    }


def test_audio_bytes_from_path_and_base64(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio-bytes")
    row = source_row(audio_path=str(audio))
    assert load_audio_bytes(row) == b"audio-bytes"

    inline = source_row(
        audio_path=None,
        audio_bytes_base64=base64.b64encode(b"inline-bytes").decode("ascii"),
    )
    assert load_audio_bytes(inline) == b"inline-bytes"

    wrong_hash = source_row(audio_path=str(audio), audio_sha256="b" * 64)
    with pytest.raises(runner.RunnerError, match="does not match"):
        load_audio_bytes(wrong_hash, expected_sha256=wrong_hash["audio_sha256"])


def test_raw_response_archive_is_content_addressed(tmp_path: Path) -> None:
    row = source_row()
    target = runner.archive_raw_response(
        tmp_path / "archive", row, "base_raw", b"payload"
    )
    assert target.read_bytes() == b"payload"
    again = runner.archive_raw_response(
        tmp_path / "archive", row, "base_raw", b"payload"
    )
    assert again == target


def test_request_tag_is_opaque_and_stable() -> None:
    row = source_row()
    first = request_tag(row)
    second = request_tag(row)
    assert first == second
    assert len(first) == 16


def test_run_completes_mocked_round_trip(tmp_path, monkeypatch):
    fixture_bytes = FIXTURE.read_bytes()
    manifest_path = tmp_path / "manifest.jsonl"
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio-bytes")
    second_audio = tmp_path / "audio2.wav"
    second_audio.write_bytes(b"audio-bytes-2")
    rows = [
        source_row(audio_path=str(audio_path)),
        source_row(
            row_id="row-2",
            audio_path=str(second_audio),
            audio_sha256=hashlib_sha256(b"audio-bytes-2"),
            leakage_group="g2",
        ),
    ]
    write_manifest(manifest_path, rows)
    output = tmp_path / "annotations.jsonl"
    archive = tmp_path / "archive"
    receipt = tmp_path / "receipt.json"

    opener, calls = make_opener(fixture_bytes)
    result = run(
        manifest_path,
        output,
        archive,
        receipt,
        arms=["base_raw", "base_smart"],
        api_key="test-key",
        concurrency=2,
        opener=opener,
        sleep=lambda seconds: None,
    )
    assert result.completed == 4
    assert result.failed == 0
    assert len(calls) == 4
    assert all("tag=phonon-dg-v1-" in url for url in calls)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    entries = [json.loads(line) for line in lines]
    assert all(entry["mip_opt_out"] is True for entry in entries)
    assert all(entry["model"]["requested_model"] == "nova-3" for entry in entries)
    assert {entry["arm"] for entry in entries} == {"base_raw", "base_smart"}
    assert {entry["row_id"] for entry in entries} == {"row-1", "row-2"}
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_data["completed"] == 4
    assert receipt_data["failed"] == 0


def test_run_is_idempotent_on_resume(tmp_path, monkeypatch):
    fixture_bytes = FIXTURE.read_bytes()
    manifest_path = tmp_path / "manifest.jsonl"
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio-bytes")
    write_manifest(manifest_path, [source_row(audio_path=str(audio_path))])
    output = tmp_path / "annotations.jsonl"
    opener, calls = make_opener(fixture_bytes)

    run(
        manifest_path,
        output,
        tmp_path / "archive",
        tmp_path / "receipt.json",
        arms=["base_raw"],
        api_key="test-key",
        opener=opener,
        sleep=lambda seconds: None,
    )
    assert len(calls) == 1
    result = run(
        manifest_path,
        output,
        tmp_path / "archive",
        tmp_path / "receipt.json",
        arms=["base_raw"],
        api_key="test-key",
        opener=opener,
        sleep=lambda seconds: None,
    )
    assert result.completed == 0
    assert result.skipped == 1
    assert len(calls) == 1


def test_run_reports_failures_without_stopping(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.jsonl"
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio-bytes")
    second_audio = tmp_path / "audio2.wav"
    second_audio.write_bytes(b"other-bytes")
    rows = [
        source_row(audio_path=str(audio_path)),
        source_row(
            row_id="row-2",
            audio_path=str(second_audio),
            audio_sha256=hashlib_sha256(b"other-bytes"),
            leakage_group="g2",
        ),
    ]
    write_manifest(manifest_path, rows)
    output = tmp_path / "annotations.jsonl"
    fixture_bytes = FIXTURE.read_bytes()

    class FlakyOpener:
        def __init__(self):
            self.count = 0

        def __call__(self, request, timeout=None):
            self.count += 1
            if self.count <= 2 and request.get_method() == "POST":
                payload = request.data or b""
                if payload == b"other-bytes":
                    raise urllib.error.HTTPError(
                        request.full_url, 401, "denied", {}, io.BytesIO(b"")
                    )
            return FakeResponse(fixture_bytes, 200)

    result = run(
        manifest_path,
        output,
        tmp_path / "archive",
        tmp_path / "receipt.json",
        arms=["base_raw"],
        api_key="test-key",
        opener=FlakyOpener(),
        sleep=lambda seconds: None,
    )
    assert result.completed == 1
    assert result.failed == 1
    assert result.failures == ["arm base_raw: ClientError"]
    receipt_data = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt_data["failed"] == 1
    assert receipt_data["failed_by_arm"] == {"base_raw": 1}


def test_run_rejects_bad_concurrency(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(manifest_path, [source_row()])
    with pytest.raises(runner.RunnerError, match="concurrency"):
        run(
            manifest_path,
            tmp_path / "out.jsonl",
            tmp_path / "archive",
            tmp_path / "receipt.json",
            arms=["base_raw"],
            api_key="test-key",
            concurrency=6,
        )


def test_run_requires_keyterms_for_keyterm_arm(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.jsonl"
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio-bytes")
    write_manifest(manifest_path, [source_row(audio_path=str(audio_path))])
    with pytest.raises(runner.RunnerError, match="keyterm"):
        run(
            manifest_path,
            tmp_path / "out.jsonl",
            tmp_path / "archive",
            tmp_path / "receipt.json",
            arms=["keyterm_raw"],
            api_key="test-key",
        )
