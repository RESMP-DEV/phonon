from __future__ import annotations

import hashlib
import json
from pathlib import Path

import manifest
import pytest


def record(
    row_id: str,
    group: str,
    *,
    duration: float = 10.0,
    split: str | None = None,
    audio_sha256: str | None = None,
    source_kind: str = "owned_public_youtube",
    source_id: str = "src",
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "audio_path": f"/tmp/{row_id}.wav",
        "audio_sha256": audio_sha256 or hashlib.sha256(row_id.encode()).hexdigest(),
        "duration_seconds": duration,
        "leakage_group": group,
        "source_kind": source_kind,
        "source_id": source_id,
        "split": split,
    }


def test_record_validation_rejects_missing_fields() -> None:
    base = record("a", "g1")
    for missing in manifest.REQUIRED_RECORD_FIELDS:
        broken = dict(base)
        broken.pop(missing)
        with pytest.raises(manifest.ManifestError, match=missing):
            manifest.normalize_record(broken, "test")


def test_record_rejects_bad_hash_duration_and_audio_location() -> None:
    base = record("a", "g1")
    bad_hash = dict(base)
    bad_hash["audio_sha256"] = "not-a-hash"
    with pytest.raises(manifest.ManifestError, match="64-hex"):
        manifest.normalize_record(bad_hash, "test")
    bad_duration = dict(base)
    bad_duration["duration_seconds"] = 0
    with pytest.raises(manifest.ManifestError, match="positive"):
        manifest.normalize_record(bad_duration, "test")
    no_audio = dict(base)
    no_audio.pop("audio_path")
    with pytest.raises(manifest.ManifestError, match="audio_path"):
        manifest.normalize_record(no_audio, "test")


def test_duplicate_row_ids_and_hashes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(json.dumps(item) for item in (record("a", "g1"), record("a", "g2")))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(manifest.ManifestError, match="duplicate row_id"):
        manifest.load_records([path])
    duplicate_hash = hashlib.sha256(b"shared").hexdigest()
    second = tmp_path / "rows2.jsonl"
    second.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                record("a", "g1", audio_sha256=duplicate_hash),
                record("b", "g2", audio_sha256=duplicate_hash),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(manifest.ManifestError, match="repeats the audio hash"):
        manifest.load_records([second])


def test_groups_are_disjoint_and_exact_targets_met() -> None:
    records = [
        record("a1", "g1", duration=1.0),
        record("a2", "g1", duration=1.5),
        record("b1", "g2", duration=2.0),
        record("c1", "g3", duration=0.5),
        record("d1", "g4", duration=3.0),
    ]
    train_target = 6_000
    eval_target = 2_000
    manifest_rows, report = manifest.build_manifest(
        records, train_target_ms=train_target, eval_target_ms=eval_target
    )
    splits = {row["row_id"]: row["split"] for row in manifest_rows}
    assert splits["a1"] == splits["a2"]
    assert (
        len(
            {row["leakage_group"] for row in manifest_rows if row["split"] == "train"}
            & {row["leakage_group"] for row in manifest_rows if row["split"] == "eval"}
        )
        == 0
    )
    train_ms = sum(
        row["duration_milliseconds"] for row in manifest_rows if row["split"] == "train"
    )
    eval_ms = sum(
        row["duration_milliseconds"] for row in manifest_rows if row["split"] == "eval"
    )
    assert train_ms == train_target
    assert eval_ms == eval_target
    assert report["leakage_groups_disjoint"] is True
    assert report["exact_duration_targets_met"] is True


def test_pinned_split_is_respected(tmp_path: Path) -> None:
    records = [
        record("a1", "g1", duration=1.0, split="eval"),
        record("b1", "g2", duration=1.0, split="eval"),
        record("c1", "g3", duration=2.0),
    ]
    manifest_rows, report = manifest.build_manifest(
        records, train_target_ms=2_000, eval_target_ms=2_000
    )
    splits = {row["row_id"]: row["split"] for row in manifest_rows}
    assert splits["a1"] == "eval"
    assert splits["b1"] == "eval"
    assert splits["c1"] == "train"
    assert report["eval_groups"] == 2


def test_unreachable_target_raises() -> None:
    records = [
        record("a1", "g1", duration=1.0),
        record("b1", "g2", duration=2.0),
    ]
    with pytest.raises(manifest.ManifestError, match="exact"):
        manifest.build_manifest(records, train_target_ms=1_500, eval_target_ms=1_500)


def test_split_search_honors_item_order_deterministically() -> None:
    items = [("a", 1000), ("b", 2000), ("c", 3000)]
    assert manifest.solve_subset(items, 3000) == ["a", "b"]
    assert manifest.solve_subset(items, 1000) == ["a"]
    assert manifest.solve_subset(items, 5000) == ["b", "c"]
    assert manifest.solve_subset(items, 700) is None
