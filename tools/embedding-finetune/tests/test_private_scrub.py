from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import private_scrub as scrub


def private_row(
    row_id: str,
    group: str,
    split: str,
    *,
    reference: str,
    vector: list[float],
) -> scrub.PrivateRow:
    audio = f"audio-{row_id}".encode()
    digest = hashlib.sha256(audio).hexdigest()
    array = np.asarray(vector, dtype=np.float64)
    array /= np.linalg.norm(array)
    return scrub.PrivateRow(
        row_id=row_id,
        group=group,
        split=split,
        duration=2.0,
        reference=reference,
        audio_hash=digest,
        audio_bytes=audio,
        teachers={"teacher": f"{reference} hypothesis"},
        text_vector=array,
        audio_vector=array,
    )


def test_load_vectors_rejects_mixed_provenance(tmp_path: Path) -> None:
    path = tmp_path / "vectors.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "row",
                "model": "other",
                "revision": scrub.MODEL_REVISION,
                "modality": "text",
                "dimension": scrub.DIMENSION,
                "vector": [1.0] * scrub.DIMENSION,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        scrub.load_vectors(path, "text")
    except scrub.ScrubError as error:
        assert "mixed provenance" in str(error)
    else:
        raise AssertionError("mixed provenance accepted")


def test_split_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    row = {"id": "a", "split": "train", "group": "g", "audio_sha256": "hash"}
    path.write_text(json.dumps({"rows": [row, row]}), encoding="utf-8")

    try:
        scrub.load_split_manifest(path)
    except scrub.ScrubError as error:
        assert "duplicate split row id" in str(error)
    else:
        raise AssertionError("duplicate split row accepted")


def test_private_builder_emits_core_kinds(tmp_path: Path) -> None:
    rows = [
        private_row(
            "a", "g1", "train", reference="CUTLASS kernel", vector=[1.0] + [0.0] * 767
        ),
        private_row(
            "b",
            "g2",
            "train",
            reference="CUTLASS matmul",
            vector=[0.8, 0.6] + [0.0] * 766,
        ),
        private_row(
            "c",
            "g3",
            "train",
            reference="compiler pass",
            vector=[0.0, 1.0] + [0.0] * 766,
        ),
        private_row(
            "d",
            "g4",
            "train",
            reference="runtime scheduler",
            vector=[0.0, 0.0, 1.0] + [0.0] * 765,
        ),
    ]
    terms = [
        {
            "phrase": "CUTLASS",
            "normalized": "cutlass",
            "sources": ["dictionary"],
            "spoken_source": "cut lass",
        },
        {
            "phrase": "MXFP4",
            "normalized": "mxfp4",
            "sources": ["replacement"],
            "spoken_source": "mxfp four",
        },
    ]
    records, report = scrub.build_private_triplets(
        rows,
        terms,
        tmp_path,
        source="synthetic",
        source_revision="0" * 40,
        minimum_negative=0.0,
        maximum_negative=1.0,
        max_term_examples=4,
    )
    kinds = {record["kind"] for record in records["train"]}

    assert {"teacher_consistency", "text_audio", "replacement"} <= kinds
    assert report["written_by_split"]["train"] >= 3
    assert (tmp_path / "audio").is_dir()
