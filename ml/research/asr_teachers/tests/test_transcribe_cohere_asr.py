from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "transcribe_cohere_asr.py"
spec = importlib.util.spec_from_file_location("transcribe_cohere_asr", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_resume_ignores_empty_hypotheses(tmp_path: Path) -> None:
    output = tmp_path / "hyps.jsonl"
    output.write_text(
        json.dumps({"audio": "one.flac", "hyp": "one"}) + "\n"
        + json.dumps({"audio": "two.flac", "hyp": ""}) + "\n"
    )
    assert module.completed_keys(output) == {"one.flac"}


def test_decode_accepts_batch_list_and_metadata_serializes_sets(tmp_path: Path) -> None:
    assert module.decode_transcription([" transcript "]) == "transcript"
    path = tmp_path / "meta.json"
    module.write_metadata(path, {"completed_rows": {"one.flac"}})
    assert '"completed_rows": [' in path.read_text()
