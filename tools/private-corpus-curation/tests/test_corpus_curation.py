from __future__ import annotations

import json
from pathlib import Path

import corpus_curation as curation


def settings(tmp_path: Path) -> Path:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "dictionary": ["CUTLASS", "RTX 6000 Pro Blackwell", ""],
                "replacements": [{"from": "mxfp four", "to": "MXFP4"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def row(
    row_id: str,
    group: str,
    *,
    text: str = "CUTLASS kernel MXFP4",
    audio_hash: str | None = None,
) -> dict[str, object]:
    return {
        "id": row_id,
        "_dataset": "dataset-a",
        "sha256_audio": audio_hash or row_id,
        "duration_seconds": 10.0,
        "leakage_group": group,
        "train_allowed": False,
        "eval_allowed": False,
        "domain_tags": ["gpu"],
        "difficulty_tags": ["technical"],
        "insert_text": text,
        "normalized_text": text,
        "verbatim_text": text,
    }


def test_aqua_lexicon_import_is_private_and_aggregate_safe(tmp_path: Path) -> None:
    lexicon = curation.load_aqua_lexicon(settings(tmp_path))
    aggregate = curation.lexicon_aggregate(lexicon)
    serialized = json.dumps(aggregate)

    assert aggregate["term_count"] == 3
    assert aggregate["source_counts"] == {"dictionary": 2, "replacement": 1}
    assert aggregate["emits_phrases"] is False
    for private_phrase in ("CUTLASS", "RTX 6000 Pro Blackwell", "MXFP4"):
        assert private_phrase not in serialized
    assert len(lexicon["terms"]) == 3


def test_exact_id_and_audio_deduplication() -> None:
    lexicon = curation.load_aqua_lexicon(settings(Path("/tmp")))
    rows = [
        row("a", "g1"),
        row("a", "g1"),
        row("b", "g2", audio_hash="same"),
        row("c", "g3", audio_hash="same"),
    ]
    manifest, report = curation.build_plan(rows, lexicon, {})

    assert report["input_rows"] == 4
    assert report["kept_rows"] == 2
    assert report["duplicate_ids"] == 1
    assert report["duplicate_audio_hashes"] == 1
    assert len(manifest) == 2
    assert len({item["id"] for item in manifest}) == 2


def test_near_duplicate_cross_group_edges_do_not_cross_splits() -> None:
    lexicon = curation.load_aqua_lexicon(settings(Path("/tmp")))
    rows = [
        row("a", "g1", text="kernel one"),
        row("b", "g2", text="kernel two"),
        row("c", "g3", text="compiler pass"),
        row("d", "g4", text="distributed training"),
    ]
    embeddings = {
        "a": {"text": [1.0, 0.0]},
        "b": {"text": [0.999999, 0.001]},
        "c": {"text": [0.0, 1.0]},
        "d": {"text": [1.0, 1.0]},
    }
    manifest, report = curation.build_plan(
        rows,
        lexicon,
        embeddings,
        text_threshold=0.99,
        fractions={"train": 0.5, "dev": 0.25, "test": 0.25},
    )
    split_of_group = {item["group"]: item["split"] for item in manifest}

    assert report["near_duplicate_pairs"]["cross_group_text"] >= 1
    assert split_of_group["g1"] == split_of_group["g2"]
    for split in ("train", "dev", "test"):
        assert any(item["split"] == split for item in manifest)
    assert report["privacy"]["emits_dictionary_phrases"] is False
    assert "kernel one" not in json.dumps(report)


def test_manifest_keeps_leakage_groups_intact() -> None:
    lexicon = curation.load_aqua_lexicon(settings(Path("/tmp")))
    rows = [row(f"row-{index}", f"g{index % 4}") for index in range(12)]
    manifest, _ = curation.build_plan(
        rows,
        lexicon,
        {},
        fractions={"train": 0.5, "dev": 0.25, "test": 0.25},
    )
    split_by_group: dict[str, str] = {}
    for item in manifest:
        if item["group"] in split_by_group:
            assert split_by_group[item["group"]] == item["split"]
        split_by_group[item["group"]] = item["split"]


def test_output_paths_reject_tracked_repo_and_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "report.json"
    assert curation.safe_output_path(outside) == outside.resolve()

    tracked = curation.REPO_ROOT / "evidence" / "report.json"
    try:
        curation.safe_output_path(tracked)
    except curation.CurationError:
        pass
    else:
        raise AssertionError("tracked output accepted")

    linked = tmp_path / "linked.json"
    linked.symlink_to(outside)
    try:
        curation.safe_output_path(linked)
    except curation.CurationError:
        pass
    else:
        raise AssertionError("symlink output accepted")
