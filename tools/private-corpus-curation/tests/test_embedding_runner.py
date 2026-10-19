from __future__ import annotations

import hashlib
import json
from pathlib import Path

import embedding_runner as runner


def test_existing_ids_and_append_record_are_resumable(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.jsonl"
    runner.append_record(
        path,
        row_id="row-a",
        vector=[2.0, 0.0],
        model="synthetic",
        revision="0" * 40,
        modality="text",
        elapsed_ms=1.25,
        artifact_sha256=None,
    )

    assert runner.existing_ids(
        path,
        model="synthetic",
        revision="0" * 40,
        modality="text",
        dimension=2,
    ) == {"row-a"}
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["vector"] == [1.0, 0.0]
    assert row["dimension"] == 2
    assert "private" not in json.dumps(row)


def test_nonfinite_and_zero_vectors_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.jsonl"
    for vector in ([0.0, 0.0], [1.0, float("inf")]):
        try:
            runner.append_record(
                path,
                row_id="bad",
                vector=vector,
                model="synthetic",
                revision="0" * 40,
                modality="text",
                elapsed_ms=0.0,
                artifact_sha256=None,
            )
        except runner.CurationError:
            continue
        raise AssertionError("invalid embedding vector accepted")
    assert not path.exists()


def test_audio_materialization_is_hash_addressed(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    row = {
        "id": "row-a",
        "sha256_audio": hashlib.sha256(b"RIFF-private-audio").hexdigest(),
        "audio": {"bytes": b"RIFF-private-audio", "path": "/private/audio.wav"},
    }

    first = runner.materialize_audio(row, cache)
    second = runner.materialize_audio(row, cache)

    assert first == second
    assert first.name == f"{row['sha256_audio']}.wav"
    assert first.read_bytes() == b"RIFF-private-audio"


def test_openai_provider_exposes_common_text_interface() -> None:
    provider = runner.OpenAICompatibleEmbedder(
        "http://embedding.invalid", "embedding-model"
    )

    assert callable(provider.encode_text)


def test_existing_ids_rejects_duplicate_resume_rows(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.jsonl"
    record = {
        "id": "row-a",
        "model": "synthetic",
        "revision": "0" * 40,
        "modality": "text",
        "dimension": 1,
        "vector": [1.0],
    }
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    try:
        runner.existing_ids(
            path,
            model="synthetic",
            revision="0" * 40,
            modality="text",
            dimension=1,
        )
    except runner.CurationError as error:
        assert "duplicate embedding output id" in str(error)
    else:
        raise AssertionError("duplicate resume row accepted")


def test_existing_ids_rejects_mixed_model_provenance(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "row-a",
                "model": "other-model",
                "revision": "0" * 40,
                "modality": "text",
                "dimension": 1,
                "vector": [1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        runner.existing_ids(
            path,
            model="synthetic",
            revision="0" * 40,
            modality="text",
            dimension=1,
        )
    except runner.CurationError as error:
        assert "mixed model provenance" in str(error)
    else:
        raise AssertionError("mixed provenance accepted")
