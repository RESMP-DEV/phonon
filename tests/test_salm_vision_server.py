import json

import pytest

import sidecar.salm_vision_server as server


def capture_events(monkeypatch, lines, *, transcribe, caption):
    events = []
    monkeypatch.setattr(server, "emit", events.append)
    server.handle_requests(iter(lines), transcribe, caption)
    return events


def test_protocol_transcribes_audio_with_bounded_local_history(tmp_path, monkeypatch):
    wav = tmp_path / "utterance.wav"
    wav.write_bytes(b"audio")
    history = [{"heard": "graph ql", "intended": "GraphQL"}]
    lines = [
        json.dumps(
            {
                "cmd": "transcribe",
                "path": str(wav),
                "id": "audio",
                "history": history,
            }
        )
    ]
    calls = []

    def transcribe(path, received_history):
        calls.append((path, received_history))
        return "hello"

    events = capture_events(monkeypatch, lines, transcribe=transcribe, caption=None)

    assert calls == [(str(wav), history)]
    assert events[0]["type"] == "result"
    assert events[0]["id"] == "audio"
    assert events[0]["text"] == "hello"
    assert events[0]["partial"] is False


def test_protocol_image_request_requires_explicit_capability_and_consent(
    tmp_path, monkeypatch
):
    image = tmp_path / "screenshot.png"
    image.write_bytes(b"image")
    lines = [
        json.dumps({"cmd": "caption", "path": str(image), "id": "no-consent"}),
        json.dumps(
            {
                "cmd": "caption",
                "path": str(image),
                "id": "wrong-capability",
                "capability": "ocr",
                "consent": True,
            }
        ),
        json.dumps(
            {
                "cmd": "caption",
                "path": str(image),
                "id": "image",
                "capability": "screen_image_model",
                "consent": True,
            }
        ),
    ]
    calls = []

    def caption(path):
        calls.append(path)
        return "visible text", 17

    events = capture_events(monkeypatch, lines, transcribe=None, caption=caption)

    assert calls == [str(image)]
    assert events[:2] == [
        {
            "type": "error",
            "id": "no-consent",
            "msg": "image inference requires capability=screen_image_model and consent=true",
        },
        {
            "type": "error",
            "id": "wrong-capability",
            "msg": "image inference requires capability=screen_image_model and consent=true",
        },
    ]
    assert events[2]["type"] == "result"
    assert events[2]["kind"] == "image"
    assert events[2]["image_feature_count"] == 17


def test_protocol_rejects_streaming_and_reports_failures(tmp_path, monkeypatch):
    wav = tmp_path / "utterance.wav"
    wav.write_bytes(b"audio")
    lines = [
        json.dumps({"cmd": "stream_start", "id": "stream"}),
        json.dumps({"cmd": "transcribe", "path": str(wav), "id": "failure"}),
    ]

    def transcribe(_path, _history):
        raise RuntimeError("boom")

    events = capture_events(
        monkeypatch, lines, transcribe=transcribe, caption=lambda _path: ("", 0)
    )

    assert [event["type"] for event in events] == ["error", "error"]
    assert "unsupported" in events[0]["msg"]
    assert events[1]["msg"] == "transcribe failed: boom"


def test_history_examples_match_the_trained_prompt_shape_and_escape_text():
    rendered = server.render_history_examples(
        [{"heard": "<raw>", "intended": "GraphQL & Phonon"}]
    )

    assert rendered.startswith(server.AUDIO_SYSTEM)
    assert "<example><heard>&lt;raw&gt;</heard>" in rendered
    assert "<intended>GraphQL &amp; Phonon</intended></example>" in rendered
    assert "not phrases to copy" in rendered


def test_history_examples_reject_unbounded_or_malformed_input():
    with pytest.raises(ValueError, match="one or two"):
        server.render_history_examples([])
    with pytest.raises(ValueError, match="bounded"):
        server.render_history_examples([{"heard": "x" * 401, "intended": "y"}])
