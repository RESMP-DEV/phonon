from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "transcribe_voxtral_vllm.py"
spec = importlib.util.spec_from_file_location("transcribe_voxtral_vllm", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
import sys

sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_resume_ignores_rows_without_hypothesis(tmp_path: Path) -> None:
    output = tmp_path / "hyps.jsonl"
    output.write_text(
        json.dumps({"audio": "one.flac", "hyp": "complete"}) + "\n"
        + json.dumps({"audio": "two.flac", "hyp": ""}) + "\n"
    )
    assert module.completed_keys(output) == {"one.flac"}


def test_only_loopback_http_is_allowed() -> None:
    module.validate_local_url("http://127.0.0.1:8302/v1/audio/transcriptions")
    module.validate_local_url("http://localhost:8302/v1/audio/transcriptions")
    for url in (
        "https://127.0.0.1/v1/audio/transcriptions",
        "http://example.com/v1/audio/transcriptions",
    ):
        try:
            module.validate_local_url(url)
        except ValueError as error:
            assert "loopback" in str(error)
        else:
            raise AssertionError("expected nonlocal transcription URL to fail")


def test_empty_slice_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    try:
        module.read_slice(empty)
    except RuntimeError as error:
        assert "empty audio slice" in str(error)
    else:
        raise AssertionError("expected empty slice to fail")
