from __future__ import annotations

import json
from pathlib import Path

import corpus_curation as curation
from pytest import MonkeyPatch


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


def test_technical_token_punctuation_is_preserved() -> None:
    """Technical spelling distinctions survive lexical normalization."""
    assert curation.normalize_phrase("C++") == "c++"
    assert curation.normalize_phrase("GPU-accelerated") == "gpu-accelerated"
    assert curation.normalize_phrase("RTX 6000") == "rtx 6000"


def test_exact_id_and_audio_deduplication(tmp_path: Path) -> None:
    lexicon = curation.load_aqua_lexicon(settings(tmp_path))
    rows = [
        row("a", "g1"),
        row("a", "g1"),
        row("b", "g2", audio_hash="same"),
        row("c", "g3", audio_hash="same"),
        row("d", "g3"),
    ]
    manifest, report = curation.build_plan(rows, lexicon, {})

    assert report["input_rows"] == 5
    assert report["kept_rows"] == 3
    assert report["duplicate_ids"] == 1
    assert report["duplicate_audio_hashes"] == 1
    assert len(manifest) == 3
    assert len({item["id"] for item in manifest}) == 3
    split_of_group = {item["group"]: item["split"] for item in manifest}
    assert split_of_group["g2"] == split_of_group["g3"]


def test_near_duplicate_cross_group_edges_do_not_cross_splits(
    tmp_path: Path,
) -> None:
    lexicon = curation.load_aqua_lexicon(settings(tmp_path))
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


def test_manifest_keeps_leakage_groups_intact(tmp_path: Path) -> None:
    lexicon = curation.load_aqua_lexicon(settings(tmp_path))
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


def test_output_paths_reject_tracked_repo_and_symlinks(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(curation, "RESEARCH_ROOT", tmp_path)
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


def test_output_paths_reject_locations_outside_research_root(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(curation, "RESEARCH_ROOT", tmp_path)
    try:
        curation.safe_output_path(Path("/tmp") / "phonon-report.json")
    except curation.CurationError:
        pass
    else:
        raise AssertionError("arbitrary filesystem output accepted")


def test_multiword_terms_use_token_boundaries(tmp_path: Path) -> None:
    """A longer number does not falsely match a multiword term."""
    lexicon = curation.load_aqua_lexicon(settings(tmp_path))
    terms = lexicon["terms"]
    counts = curation.term_counts("rtx 60000 pro blackwell", terms)

    assert sum(counts) == 0


def test_load_embeddings_rejects_mixed_dimensions(tmp_path: Path) -> None:
    """One modality must use one vector dimension throughout the file."""
    path = tmp_path / "embeddings.jsonl"
    path.write_text(
        json.dumps({"id": "a", "text": [1.0]})
        + "\n"
        + json.dumps({"id": "b", "text": [1.0, 0.0]})
        + "\n",
        encoding="utf-8",
    )

    try:
        curation.load_embeddings(path)
    except curation.CurationError as error:
        assert "mixes text dimensions" in str(error)
    else:
        raise AssertionError("mixed dimensions accepted")
