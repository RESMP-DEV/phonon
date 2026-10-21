from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cli import build_command, parse_arguments


def record(row_id: str, group: str, duration: float) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "audio_path": f"/tmp/{row_id}.wav",
        "audio_sha256": hashlib.sha256(row_id.encode()).hexdigest(),
        "duration_seconds": duration,
        "leakage_group": group,
        "source_kind": "owned_public_youtube",
        "source_id": "src",
    }


def test_build_writes_manifest_and_report(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                record("a1", "g1", 1.0),
                record("a2", "g1", 1.5),
                record("b1", "g2", 2.0),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"
    report = tmp_path / "report.json"
    arguments = parse_arguments(
        [
            "build",
            "--input",
            str(source),
            "--output",
            str(output),
            "--report",
            str(report),
            "--train-target-ms",
            "2500",
            "--eval-target-ms",
            "2000",
        ]
    )
    assert build_command(arguments) == 0
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3
    splits = {row["row_id"]: row["split"] for row in rows}
    assert splits["a1"] == splits["a2"]
    assert splits["b1"] != splits["a1"]
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["leakage_groups_disjoint"] is True
    assert payload["train_duration_milliseconds"] == 2500
    assert payload["eval_duration_milliseconds"] == 2000
