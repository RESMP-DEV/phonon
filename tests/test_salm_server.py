import json
from pathlib import Path

from sidecar.salm_server import handle_requests


def run_requests(tmp_path: Path, lines: list[str], transcript="hello"):
    wav = tmp_path / "utterance.wav"
    wav.write_bytes(b"fake audio")
    output = []

    def capture(value):
        output.append(value)

    def transcribe(path):
        assert path == str(wav)
        return transcript

    original = __import__("sidecar.salm_server", fromlist=["emit"]).emit
    module = __import__("sidecar.salm_server", fromlist=["emit"])
    module.emit = capture
    try:
        handle_requests(iter(lines), transcribe, model_id="fake-model")
    finally:
        module.emit = original
    return output


def test_protocol_ready_is_outside_request_loop(tmp_path):
    # `ready` is emitted by main after load; request handling starts afterward.
    events = run_requests(tmp_path, ['{"cmd":"ping"}'])
    assert events == [{"type": "pong"}]


def test_protocol_transcribe_and_shutdown(tmp_path):
    events = run_requests(
        tmp_path,
        ['{"cmd":"ping"}', json.dumps({"cmd": "transcribe", "path": str(tmp_path / "utterance.wav"), "id": "1"}), '{"cmd":"shutdown"}'],
    )
    assert events[0] == {"type": "pong"}
    assert events[1]["type"] == "result"
    assert events[1]["id"] == "1"
    assert events[1]["path"] == str(tmp_path / "utterance.wav")
    assert events[1]["text"] == "hello"
    assert events[1]["partial"] is False
    assert events[1]["seconds"] >= 0
    assert events[2]["type"] == "status"
    assert events[2]["phase"] == "shutdown"


def test_protocol_rejects_streaming_and_bad_paths(tmp_path):
    missing = tmp_path / "missing.wav"
    events = run_requests(
        tmp_path,
        [
            json.dumps({"cmd": "stream_start", "id": "s"}),
            json.dumps({"cmd": "stream_chunk", "id": "s"}),
            json.dumps({"cmd": "stream_stop", "id": "s"}),
            json.dumps({"cmd": "transcribe", "path": str(missing), "id": "m"}),
        ],
    )
    assert all(event["type"] == "error" for event in events)
    assert [event["id"] for event in events] == ["s", "s", "s", "m"]
    assert "unsupported" in events[0]["msg"]
    assert "missing file" in events[-1]["msg"]


def test_protocol_reports_bad_json_and_unknown_commands(tmp_path):
    events = run_requests(tmp_path, ["{bad json", '{"cmd":"wat"}'])
    assert events[0]["type"] == "error"
    assert events[0]["msg"].startswith("bad json:")
    assert events[1]["type"] == "error"
    assert events[1]["msg"] == "unknown cmd: wat"


def test_protocol_reports_transcription_failure(tmp_path):
    events = run_requests(
        tmp_path,
        [json.dumps({"cmd": "warmup_stream", "path": str(tmp_path / "utterance.wav"), "id": "e"})],
    )
    # The helper's transcriber succeeds; exercise failure through a direct call.
    module = __import__("sidecar.salm_server", fromlist=["handle_requests"])
    output = []
    module.emit = output.append
    wav = tmp_path / "utterance.wav"
    wav.write_bytes(b"audio")
    module.handle_requests(
        iter([json.dumps({"cmd": "transcribe", "path": str(wav), "id": "e"})]),
        lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
        model_id="fake",
    )
    assert len(output) == 1
    assert output[0]["type"] == "error"
    assert output[0]["id"] == "e"
    assert output[0]["msg"] == "transcribe failed: boom"
