#!/usr/bin/env python3
"""Generate resumable text/audio embeddings for private Phonon curation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from corpus_curation import (
    CurationError,
    iter_parquet_rows,
    normalize_vector,
    reference_text,
    safe_output_path,
    validate_output_aliases,
)

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MODEL_DIMENSION = 768
MODEL_WEIGHTS = "model.safetensors"
MODEL_CONTRACT = "artifact-contract.json"


def parquet_rows(
    datasets: Sequence[Path], modality: str, limit: int | None = None
) -> list[dict[str, Any]]:
    """Load one unique row ID/audio hash with only the needed private fields."""
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for dataset in datasets:
        for source in iter_parquet_rows(dataset):
            row_id = source.get("id")
            audio_hash = source.get("sha256_audio")
            if not isinstance(row_id, str) or not row_id:
                raise CurationError("dataset row lacks a nonempty string id")
            if not isinstance(audio_hash, str) or not audio_hash:
                raise CurationError(f"row {row_id} lacks sha256_audio")
            if row_id in seen_ids or audio_hash in seen_hashes:
                continue
            seen_ids.add(row_id)
            seen_hashes.add(audio_hash)
            if modality == "text" and not reference_text(source):
                continue
            row = {
                "id": row_id,
                "text": reference_text(source),
                "audio": source.get("audio"),
                "sha256_audio": source.get("sha256_audio"),
            }
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def file_sha256(path: Path) -> str:
    """Hash a local artifact before attributing a pinned upstream identity."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_descriptor(model_path: str) -> dict[str, Any]:
    """Validate a local EmbeddingGemma artifact against its local contract."""
    if model_path == MODEL_ID:
        return {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "artifact": None,
            "artifact_sha256": None,
        }
    root = Path(model_path).resolve()
    weights = root / MODEL_WEIGHTS
    contract_path = root / MODEL_CONTRACT
    if not weights.is_file() or not contract_path.is_file():
        raise CurationError(
            "local EmbeddingGemma model path lacks weights or artifact contract"
        )
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CurationError(
            f"invalid EmbeddingGemma artifact contract: {error}"
        ) from error
    expected = {"model", "revision", "weights_sha256", "files_sha256"}
    if (
        not isinstance(contract, dict)
        or set(contract) != expected
        or contract["model"] != MODEL_ID
        or contract["revision"] != MODEL_REVISION
        or not isinstance(contract["files_sha256"], dict)
    ):
        raise CurationError("EmbeddingGemma artifact contract does not match the pin")
    observed = file_sha256(weights)
    if observed != contract["weights_sha256"]:
        raise CurationError("EmbeddingGemma artifact weights do not match contract")
    observed_files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.parts and path.name != MODEL_CONTRACT
    }
    if observed_files != contract["files_sha256"]:
        raise CurationError("EmbeddingGemma artifact files do not match contract")
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "artifact": str(root),
        "artifact_sha256": observed,
    }


def existing_ids(
    path: Path,
    *,
    model: str,
    revision: str,
    modality: str,
    dimension: int = MODEL_DIMENSION,
) -> set[str]:
    """Validate resumable provenance and return completed row IDs."""
    completed: set[str] = set()
    if not path.exists():
        return completed
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CurationError(
                f"embedding output line {line_number} is invalid"
            ) from error
        if not isinstance(row.get("id"), str):
            raise CurationError(f"embedding output line {line_number} lacks id")
        if (
            row.get("model") != model
            or row.get("revision") != revision
            or row.get("modality") != modality
            or row.get("dimension") != dimension
        ):
            raise CurationError(
                f"embedding output line {line_number} has mixed model provenance"
            )
        if row["id"] in completed:
            raise CurationError(f"duplicate embedding output id on line {line_number}")
        completed.add(row["id"])
    return completed


def append_record(
    path: Path,
    *,
    row_id: str,
    vector: Sequence[float],
    model: str,
    revision: str,
    modality: str,
    elapsed_ms: float,
    artifact_sha256: str | None,
) -> None:
    """Append one validated normalized-vector record."""
    normalized = normalize_vector([float(value) for value in vector])
    record = {
        "id": row_id,
        "model": model,
        "revision": revision,
        "modality": modality,
        "dimension": len(normalized),
        "elapsed_ms": round(elapsed_ms, 3),
        "artifact_sha256": artifact_sha256,
        "vector": normalized,
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")
    path.chmod(0o600)


class OpenAICompatibleEmbedder:
    """Small native client for LM Studio's local embeddings endpoint."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def encode_text(self, texts: Sequence[str]) -> list[list[float]]:
        return self.encode(texts)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        request = json.dumps({"model": self.model, "input": list(texts)}).encode(
            "utf-8"
        )
        req = urllib.request.Request(
            f"{self.base_url}/v1/embeddings",
            data=request,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        vectors = [item.get("embedding") for item in payload.get("data", [])]
        if len(vectors) != len(texts) or not all(
            isinstance(vector, list) and vector for vector in vectors
        ):
            raise CurationError(
                "embedding endpoint returned malformed or incomplete batch"
            )
        return [list(map(float, vector)) for vector in vectors]


class EmbeddingGemma2:
    """Pinned native text/audio EmbeddingGemma 2 provider."""

    def __init__(
        self,
        device: str = "auto",
        precision: str = "auto",
        model_path: str = MODEL_ID,
    ) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise CurationError(
                "EmbeddingGemma 2 needs sentence-transformers >= 6.1 and transformers "
                "with EmbeddingGemma2 support"
            ) from error
        if precision == "bfloat16":
            dtype = torch.bfloat16
        elif precision == "float32":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.torch = torch
        self.model = SentenceTransformer(
            model_path,
            revision=MODEL_REVISION,
            device=None if device == "auto" else device,
            config_kwargs={"vision_config": None},
            model_kwargs={"torch_dtype": dtype},
        )

    def encode_text(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            prompt_name="SentenceSimilarity",
            normalize_embeddings=True,
            truncate_dim=MODEL_DIMENSION,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.astype(float).tolist() for vector in vectors]

    def encode_audio(self, paths: Sequence[Path]) -> list[list[float]]:
        """Encode canonical mono-16k WAV files with the audio-only placeholder."""
        vectors = self.model.encode(
            [{"text": "<|audio|>", "audio": str(path)} for path in paths],
            normalize_embeddings=True,
            truncate_dim=MODEL_DIMENSION,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.astype(float).tolist() for vector in vectors]


def materialize_audio(row: dict[str, Any], cache: Path) -> Path:
    """Write one embedded canonical WAV to a private hash-addressed cache."""
    audio = row.get("audio")
    data = audio.get("bytes") if isinstance(audio, dict) else None
    digest = row.get("sha256_audio")
    if not isinstance(data, bytes | bytearray) or not isinstance(digest, str):
        raise CurationError(f"row {row.get('id')} lacks embedded audio bytes")
    if hashlib.sha256(data).hexdigest() != digest:
        raise CurationError(f"row {row.get('id')} has an audio hash mismatch")
    path = cache / f"{digest}.wav"
    if not path.exists():
        temporary = cache / f".{digest}.{time.time_ns()}.tmp"
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o600
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        temporary.replace(path)
        path.chmod(0o600)
    return path


def run_embeddings(
    provider_name: str,
    rows: Sequence[dict[str, Any]],
    output: Path,
    *,
    modality: str,
    batch_size: int,
    base_url: str,
    nomic_model: str,
    device: str,
    precision: str,
    model_path: str,
) -> dict[str, Any]:
    """Generate or resume embeddings and return an aggregate-only receipt."""
    if batch_size < 1:
        raise CurationError("batch size must be positive")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.touch(mode=0o600, exist_ok=True)
    provider: OpenAICompatibleEmbedder | EmbeddingGemma2
    if provider_name == "nomic":
        model_name = nomic_model
        revision = "live-lm-studio"
        artifact = {
            "model": model_name,
            "revision": revision,
            "artifact": None,
            "artifact_sha256": None,
        }
        provider = OpenAICompatibleEmbedder(base_url, nomic_model)
    elif provider_name == "embeddinggemma2":
        artifact = artifact_descriptor(model_path)
        model_name = artifact["model"]
        revision = artifact["revision"]
        provider = EmbeddingGemma2(
            device=device, precision=precision, model_path=model_path
        )
        model_name = MODEL_ID
        revision = MODEL_REVISION
    else:
        raise CurationError(f"unknown embedding provider: {provider_name}")
    output.chmod(0o600)
    completed = existing_ids(
        output, model=model_name, revision=revision, modality=modality
    )
    pending = [row for row in rows if row["id"] not in completed]
    started = time.monotonic()

    audio_cache = output.parent / ".audio-cache"
    audio_cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    batches = 0
    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            batch_started = time.monotonic()
            if modality == "text":
                vectors = provider.encode_text([row["text"] for row in batch])
            elif provider_name == "embeddinggemma2":
                paths = [materialize_audio(row, audio_cache) for row in batch]
                vectors = provider.encode_audio(paths)
            else:
                raise CurationError("Nomic provider supports text only")
            if len(vectors) != len(batch):
                raise CurationError("provider returned a short embedding batch")
            for row, vector in zip(batch, vectors, strict=True):
                append_record(
                    output,
                    row_id=row["id"],
                    vector=vector,
                    model=model_name,
                    revision=revision,
                    modality=modality,
                    elapsed_ms=(time.monotonic() - batch_started) * 1000,
                    artifact_sha256=artifact["artifact_sha256"],
                )
            batches += 1
    finally:
        if hasattr(provider, "model"):
            provider.model = None  # type: ignore[assignment]
    return {
        "provider": provider_name,
        "model": model_name,
        "revision": revision,
        "artifact_sha256": artifact["artifact_sha256"],
        "modality": modality,
        "input_rows": len(rows),
        "already_present": len(rows) - len(pending),
        "generated": len(pending),
        "batches": batches,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "output": str(output),
        "emits_transcripts": False,
        "emits_audio": False,
    }


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument(
        "--provider", choices=("nomic", "embeddinggemma2"), required=True
    )
    parser.add_argument("--modality", choices=("text", "audio"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--nomic-model", default="text-embedding-nomic-embed-text-v1.5")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("auto", "bfloat16", "float32"), default="auto"
    )
    parser.add_argument("--model-path", default=MODEL_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate embeddings and print one aggregate receipt."""
    try:
        arguments = parse_arguments(argv if argv is not None else sys.argv[1:])
        output = safe_output_path(arguments.out)
        validate_output_aliases(output, arguments.dataset)
        rows = parquet_rows(arguments.dataset, arguments.modality, arguments.limit)
        receipt = run_embeddings(
            arguments.provider,
            rows,
            output,
            modality=arguments.modality,
            batch_size=arguments.batch_size,
            base_url=arguments.base_url,
            nomic_model=arguments.nomic_model,
            device=arguments.device,
            precision=arguments.precision,
            model_path=arguments.model_path,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - private-input CLI failure boundary
        print(f"FAIL embedding-runner: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
