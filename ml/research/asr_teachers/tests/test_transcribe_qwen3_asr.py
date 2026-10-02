from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "transcribe_qwen3_asr.py"
spec = importlib.util.spec_from_file_location("transcribe_qwen3_asr", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_resume_ignores_rows_without_hypothesis(tmp_path: Path) -> None:
    output = tmp_path / "hyps.jsonl"
    output.write_text(
        json.dumps({"audio": "one.flac", "hyp": "complete"}) + "\n"
        + json.dumps({"audio": "two.flac", "hyp": ""}) + "\n"
    )
    assert module.completed_keys(output) == {"one.flac"}


def test_decode_accepts_transformers_batch_list() -> None:
    assert module.decode_transcription([" transcript "]) == "transcript"
    assert module.decode_transcription(" transcript ") == "transcript"


def test_metadata_is_json_serializable(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    module.write_metadata(path, {"completed_rows": {"one.flac", "two.flac"}})
    assert '"completed_rows": [' in path.read_text()


def test_empty_slice_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    try:
        module.read_slice(empty)
    except RuntimeError as error:
        assert "empty audio slice" in str(error)
    else:
        raise AssertionError("expected empty slice to fail")
