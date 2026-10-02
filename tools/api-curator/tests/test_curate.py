from __future__ import annotations

import json
from pathlib import Path

import pytest
from api_curator.__main__ import read_rows, read_teacher_map
from api_curator.curate import (
    CuratedRow,
    curate_one,
    curate_rows,
    format_prompt,
    parse_json_block,
    validate_judgment,
)


def test_parse_fenced_json_only() -> None:
    text = 'prefix ```json\n{"decision":"keep"}\n``` suffix'
    assert parse_json_block(text) == {"decision": "keep"}


def test_validate_judgment_rules() -> None:
    keep = validate_judgment(
        {"decision": "Keep", "corrected": "hello", "reason_code": "exact", "confidence": 1},
        "teacher",
        1,
    )
    assert keep.decision == "keep"
    discard = validate_judgment(
        {"decision": "discard", "corrected": "", "reason_code": "unsafe", "confidence": 1},
        "teacher",
        1,
    )
    assert discard.decision == "discard"
    with pytest.raises(ValueError):
        validate_judgment({"decision": "discard", "corrected": "unsafe text", "reason_code": "unsafe", "confidence": 1}, "teacher", 1)
    with pytest.raises(ValueError):
        validate_judgment({"decision": "revise", "corrected": "", "reason_code": "rewrite", "confidence": 1}, "teacher", 1)
    with pytest.raises(ValueError):
        validate_judgment({"decision": "keep", "corrected": "x", "reason_code": "exact", "confidence": 1.1}, "teacher", 1)


def test_request_reaches_validated_response(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            completion = '```json\n{"decision":"revise","corrected":"Deploy Phonon v2.","reason_code":"terminology","confidence":0.94}\n```'
            return json.dumps(
                {"choices": [{"message": {"content": completion}}]}
            ).encode()

    def fake_urlopen(request, timeout=None):
        captured["path"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("api_curator.curate.urllib.request.urlopen", fake_urlopen)
    judgment, completion = curate_one(
        CuratedRow(
            "42",
            "deploy phone on v2",
            "Deploy Phonon v2.",
            {"qwen3_asr_1p7b": "deploy Phonon v2"},
        ),
        endpoint="https://teacher.test/v1",
        api_key="secret",
        model="teacher",
        timeout=7,
    )
    assert captured["path"] == "https://teacher.test/v1/chat/completions"
    assert captured["body"]["messages"][0]["role"] == "system"
    user_content = captured["body"]["messages"][1]["content"]
    assert "<raw>deploy phone on v2</raw>" in user_content
    assert '<hypothesis source="qwen3_asr_1p7b">deploy Phonon v2</hypothesis>' in user_content
    assert "<final>Deploy Phonon v2.</final>" in user_content
    assert judgment.decision == "revise"
    assert judgment.model == "teacher"
    assert "Phonon v2" in completion


def test_resume_and_append(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_curate_one(row, **kwargs):
        calls.append(row.id)
        return validate_judgment(
            {"decision": "keep", "corrected": row.final, "reason_code": "exact", "confidence": 1},
            "teacher",
            0,
        ), "completion"

    monkeypatch.setattr("api_curator.curate.curate_one", fake_curate_one)
    rows = [CuratedRow(str(i), f"raw {i}", f"final {i}") for i in range(3)]
    output = tmp_path / "curated.jsonl"
    first = curate_rows(rows, output, endpoint="x", api_key="y", model="teacher")
    assert len(first.read_text().splitlines()) == 3
    curate_rows(rows[:2], output, endpoint="x", api_key="y", model="teacher")
    assert calls == ['0', '1', '2']


def test_read_rows_preserves_supplied_ids(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"custom":"row-a","raw":"a","corrected":"A"}\n')
    rows = read_rows(source, "raw", "corrected", "custom", 0)
    assert rows == [CuratedRow("row-a", "a", "A", {})]


def test_named_teacher_maps_are_joined_by_id_and_escaped(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        '{"audio":"one.flac","raw":"a & b","corrected":"A & B"}\n'
        '{"audio":"two.flac","raw":"c","corrected":"C"}\n'
    )
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(
        '{"audio":"one.flac","hyp":"A <and> B"}\n'
        '{"audio":"missing.flac","hyp":"unused"}\n'
    )
    teachers = read_teacher_map(teacher, "audio", "hyp")
    rows = read_rows(source, "raw", "corrected", "audio", 0, {"qwen": teachers})
    assert rows[0].teachers == {"qwen": "A <and> B"}
    assert rows[1].teachers == {}
    prompt = format_prompt(rows[0].raw, rows[0].final, rows[0].teachers)
    assert '<hypothesis source="qwen">A &lt;and&gt; B</hypothesis>' in prompt


def test_omitting_final_removes_the_accepted_text_block() -> None:
    prompt = format_prompt("raw", "", {"teacher": "candidate"})
    assert "<asr_hypotheses>" in prompt
    assert "<hypothesis source=\"teacher\">candidate</hypothesis>" in prompt
    assert "<final>" not in prompt


def test_curate_image_contract(monkeypatch) -> None:
    from api_curator.curate import curate_image

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            completion = '```json\n{"decision":"keep","corrected":"PHONON_API_VISION_42","reason_code":"exact","confidence":1}\n```'
            return json.dumps({"choices": [{"message": {"content": completion}}]}).encode()

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr("api_curator.curate.urllib.request.urlopen", fake_urlopen)
    judgment, _ = curate_image(
        "data:image/png;base64,AAA",
        endpoint="https://teacher.test/v1",
        api_key="secret",
        model="vision-teacher",
        timeout=9,
    )
    assert judgment.corrected == "PHONON_API_VISION_42"
    assert captured["body"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png")
