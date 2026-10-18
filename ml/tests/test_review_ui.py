from __future__ import annotations

import json
from pathlib import Path

from phonon.review_ui import build_review_ui, render_review_ui


def _queue(path: Path) -> Path:
    rows = [
        {"id": "one", "draft_text": "one", "audio_path": "/missing/one.wav"},
        {"id": "two", "draft_text": "two", "audio_path": "/missing/two.wav"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_approving_advances_to_the_next_clip_in_rendered_html() -> None:
    html = render_review_ui(
        {
            "title": "test",
            "created_at": "now",
            "review_id": "test-review",
            "export_name": "test.jsonl",
            "row_count": 0,
            "dedupe_removed": 0,
            "rows": [],
        }
    )

    assert "if (verdict === 'approved' && current < rows.length - 1)" in html
    assert "setVerdict('approved')" in html


def test_regeneration_reuses_the_previous_review_identity(tmp_path: Path) -> None:
    queue = _queue(tmp_path / "queue.jsonl")
    output = tmp_path / "review" / "index.html"
    json_output = tmp_path / "review" / "index.json"
    first = build_review_ui(
        queue_file=queue,
        out=output,
        dataset_root=tmp_path,
        dataset="unused",
        split="unused",
        json_out=json_output,
        copy_audio=False,
    )
    first_payload = json.loads(json_output.read_text())
    second = build_review_ui(
        queue_file=queue,
        out=output,
        dataset_root=tmp_path,
        dataset="unused",
        split="unused",
        json_out=json_output,
        copy_audio=False,
    )
    second_payload = json.loads(json_output.read_text())

    assert first["rows"] == 2
    assert second["rows"] == 2
    assert first_payload["review_id"] == second_payload["review_id"]
