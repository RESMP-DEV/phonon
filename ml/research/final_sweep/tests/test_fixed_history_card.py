from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "build_fixed_history_card.py"
spec = importlib.util.spec_from_file_location("build_fixed_history_card", MODULE_PATH)
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def row(index: int, *, corrected: bool = True, duration: float = 8.0) -> dict[str, object]:
    raw = f"deploy the service instance number {index} now"
    fixed = f"deploy the service instance number {index} now" if not corrected else (
        f"deploy the Kubernetes service instance number {index} now"
    )
    return {
        "audio": f"audio/AQ_{index:04d}.flac",
        "raw": raw,
        "corrected": fixed,
        "duration": duration,
        "has_correction": corrected,
        "timestamp": f"2026-06-{index + 1:02d}T00:00:00.000Z",
        "session_id": index,
    }


def test_card_is_deterministic_and_excludes_eval_audio(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    rows = [row(index) for index in range(12)]
    write_manifest(manifest, rows)
    write_manifest(excluded, [rows[0], rows[1]])
    excluded_ids = module.load_ids(excluded)
    candidates = module.card_candidates(manifest, excluded_ids, 2.0, 20.0)
    first = module.select_card(candidates, 4)
    second = module.select_card(candidates, 4)
    assert [item["audio"] for item in first] == [item["audio"] for item in second]
    assert not ({item["audio"] for item in first} & excluded_ids)
    assert len({item["audio"] for item in first}) == 4
    timestamps = [str(item["timestamp"]) for item in first]
    assert timestamps == sorted(timestamps)


def test_card_requires_a_real_correction_and_token_cap(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            row(0, corrected=False),
            row(1, duration=1.0),
            row(2),
            {**row(3), "raw": " ".join(f"token{index}" for index in range(80))},
        ],
    )
    candidates = module.card_candidates(manifest, set(), 2.0, 20.0)
    assert [item["audio"] for item in candidates] == ["audio/AQ_0002.flac"]


def test_select_card_rejects_impossible_count(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [row(0)])
    candidates = module.card_candidates(manifest, set(), 2.0, 20.0)
    try:
        module.select_card(candidates, 5)
    except RuntimeError as error:
        assert "only 1 are usable" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError for an oversized card")
