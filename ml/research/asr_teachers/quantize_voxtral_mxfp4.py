#!/usr/bin/env python3
"""Create and audit a Voxtral Small MXFP4 W4A16 teacher checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "mistralai/Voxtral-Small-24B-2507"
DEFAULT_REVISION = "da5b42409f279fdd92febee0511a6c32828569c1"
DEFAULT_DEVICE_MAP = "cpu"
DEFAULT_PIPELINE = "sequential"
DEFAULT_SEQUENTIAL_OFFLOAD_DEVICE = "cuda:1"
EXPECTED_SHARDS = 11
IGNORED_LAYERS = (
    "lm_head",
    "re:.*audio_tower.*",
    "re:.*multi_modal_projector.*",
)
AUDIO_CONFIG = {
    "downsample_factor": 4,
    "d_model": 1280,
    "sampling_rate": 16000,
    "hop_length": 160,
    "window_size": 400,
}
DOWNLOAD_ALLOW_PATTERNS = (
    "README.md",
    "*.json",
    "*.txt",
    "*.model",
    "*.model.v3",
    "*.jinja",
    "*.yaml",
    "model-000*.safetensors",
)
DOWNLOAD_BLOCKED_NAMES = ("consolidated.safetensors", "original")
REQUIRED_BASE_FILES = (
    "config.json",
    "generation_config.json",
    "params.json",
    "preprocessor_config.json",
    "tekken.json",
    "model.safetensors.index.json",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _audio_name(row: dict[str, Any]) -> str:
    value = str(row.get("audio") or row.get("audio_path") or "").strip()
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe or empty audio reference: {value!r}")
    return value


def _duration(row: dict[str, Any]) -> float:
    return float(row.get("duration") or row.get("dur") or 0.0)


def _split(row: dict[str, Any]) -> str | None:
    value = row.get("split", row.get("partition"))
    return None if value is None else str(value)


def _text(row: dict[str, Any]) -> str:
    return str(row.get("corrected") or row.get("ref") or row.get("text") or "").strip()


@dataclass(frozen=True)
class CalibrationSelection:
    rows: list[dict[str, Any]]
    stats: dict[str, int | str | list[Any]]


def load_eval_audio_ids(path: Path) -> set[str]:
    return {_audio_name(row) for row in read_jsonl(path)}


def select_calibration_rows(
    manifest: Path,
    eval_slice: Path,
    audio_root: Path,
    *,
    limit: int,
    seed: str,
    min_duration: float,
    max_duration: float,
    allow_splits: Sequence[str],
) -> CalibrationSelection:
    """Select stable, de-duplicated training audio while excluding evaluation IDs."""
    if limit < 1:
        raise ValueError("calibration limit must be positive")
    excluded = load_eval_audio_ids(eval_slice)
    candidates: list[tuple[str, str, dict[str, Any], Path]] = []
    seen: set[str] = set()
    stats: dict[str, int | str | list[Any]] = {
        "manifest_rows": 0,
        "eval_excluded": 0,
        "duplicate_audio": 0,
        "missing_text": 0,
        "invalid_duration": 0,
        "missing_audio": 0,
        "disallowed_split": 0,
        "eligible": 0,
        "selected": 0,
    }
    for row in read_jsonl(manifest):
        stats["manifest_rows"] += 1
        try:
            audio_name = _audio_name(row)
        except ValueError as error:
            raise ValueError(f"{manifest}: {error}") from error
        if audio_name in excluded:
            stats["eval_excluded"] += 1
            continue
        if audio_name in seen:
            stats["duplicate_audio"] += 1
            continue
        seen.add(audio_name)
        if not _text(row):
            stats["missing_text"] += 1
            continue
        duration = _duration(row)
        if not min_duration <= duration <= max_duration:
            stats["invalid_duration"] += 1
            continue
        split = _split(row)
        if split is not None and split not in allow_splits:
            stats["disallowed_split"] += 1
            continue
        audio_path = audio_root / audio_name
        if not audio_path.is_file():
            stats["missing_audio"] += 1
            continue
        stats["eligible"] += 1
        rank = sha256_bytes(f"{seed}\0{audio_name}".encode())
        candidates.append((rank, audio_name, row, audio_path))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected_rows: list[dict[str, Any]] = []
    for rank, audio_name, row, audio_path in candidates[:limit]:
        selected_rows.append(
            {
                "audio": audio_name,
                "audio_path": str(audio_path),
                "audio_sha256": sha256_file(audio_path),
                "duration": _duration(row),
                "split": _split(row),
                "text_sha256": sha256_bytes(_text(row).encode("utf-8")),
                "selection_rank": rank,
            }
        )
    stats["selected"] = len(selected_rows)
    if not selected_rows:
        raise RuntimeError(f"no eligible calibration rows in {manifest}")
    metadata = {
        **stats,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "eval_slice": str(eval_slice),
        "eval_slice_sha256": sha256_file(eval_slice),
        "audio_root": str(audio_root),
        "seed": seed,
        "duration_range": [min_duration, max_duration],
        "allow_splits": list(allow_splits),
    }
    return CalibrationSelection(selected_rows, metadata)


def write_calibration_manifest(path: Path, selection: CalibrationSelection) -> str:
    payload = {
        "schema_version": 1,
        **selection.stats,
        "rows": selection.rows,
    }
    write_json(path, payload)
    return sha256_file(path)


def download_base_snapshot(destination: Path) -> Path:
    """Download the pinned official snapshot without its duplicate single file."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=DEFAULT_MODEL,
            revision=DEFAULT_REVISION,
            local_dir=destination,
            allow_patterns=list(DOWNLOAD_ALLOW_PATTERNS),
            max_workers=8,
        )
    )


def verify_base_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    for name in REQUIRED_BASE_FILES:
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    blocked = [
        str(item.relative_to(path))
        for item in (path, *path.rglob("*"))
        if item != path
        and any(part in DOWNLOAD_BLOCKED_NAMES for part in item.relative_to(path).parts)
    ]
    if blocked:
        raise RuntimeError(f"base snapshot contains blocked duplicate files: {blocked}")
    shards = sorted(path.glob("model-000*-of-*.safetensors"))
    if len(shards) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"expected {EXPECTED_SHARDS} sharded weights, found {len(shards)} in {path}"
        )
    index = json.loads((path / "model.safetensors.index.json").read_text())
    referenced = set(index.get("weight_map", {}).values())
    shard_names = {item.name for item in shards}
    missing = sorted(referenced - shard_names)
    if missing:
        raise RuntimeError(f"weight index references missing shards: {missing[:5]}")
    return {
        "path": str(path),
        "revision": DEFAULT_REVISION,
        "shards": [item.name for item in shards],
        "referenced_weights": len(referenced),
        "total_shard_bytes": sum(item.stat().st_size for item in shards),
    }


def _set_config_value(config: Any, key: str, value: Any) -> None:
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def patch_audio_config(config: Any) -> None:
    """Bake the official Mistral audio fields into both config levels."""
    for key, value in AUDIO_CONFIG.items():
        _set_config_value(config, key, value)
    audio_config = config.get("audio_config") if isinstance(config, dict) else getattr(config, "audio_config", None)
    if audio_config is not None:
        for key, value in AUDIO_CONFIG.items():
            _set_config_value(audio_config, key, value)


def validate_audio_config(config: dict[str, Any]) -> None:
    wrong = {key: config.get(key) for key, expected in AUDIO_CONFIG.items() if config.get(key) != expected}
    if wrong:
        raise ValueError(f"output config has incorrect official audio fields: {wrong}")


def normalize_processor_inputs(inputs: Any) -> dict[str, list[int]]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected one-row Voxtral input IDs, got {tuple(input_ids.shape)}")
    ids = input_ids[0].tolist()
    if attention_mask is None:
        mask = [1] * len(ids)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Voxtral attention mask and input IDs disagree")
        mask = attention_mask[0].tolist()
    return {"input_ids": ids, "attention_mask": mask}


def prepare_calibration_dataset(
    selection: CalibrationSelection,
    processor: Any,
) -> Any:
    """Pin token IDs in Arrow; regenerate audio features in the collator."""
    from datasets import Dataset

    records: list[dict[str, Any]] = []
    for row in selection.rows:
        inputs = processor.apply_transcription_request(
            audio=row["audio_path"],
            model_id=DEFAULT_MODEL,
            language="en",
        )
        normalized = normalize_processor_inputs(inputs)
        if "input_features" not in inputs:
            raise ValueError("Voxtral processor did not return input_features")
        records.append(
            {
                **normalized,
                "audio": row["audio"],
                "audio_path": row["audio_path"],
                "duration": row["duration"],
            }
        )
    return Dataset.from_list(records)


class VoxtralAudioCollator:
    """Regenerate official transcription features for batch-size-one GPTQ."""

    def __init__(self, processor: Any, dtype: Any = None) -> None:
        self.processor = processor
        self.dtype = dtype

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("Voxtral MXFP4 calibration requires batch size one")
        row = features[0]
        inputs = self.processor.apply_transcription_request(
            audio=row["audio_path"],
            model_id=DEFAULT_MODEL,
            language="en",
        )
        normalized = normalize_processor_inputs(inputs)
        for key in ("input_ids", "attention_mask"):
            if normalized[key] != row[key]:
            # The stable token cache and live processor must agree before any
            # calibration activation is trusted.
                raise ValueError(f"Voxtral processor token drift for {row['audio']}: {key}")
        if self.dtype is not None:
            for key in tuple(inputs):
                if "input_features" in key:
                    inputs[key] = inputs[key].to(dtype=self.dtype)
        return dict(inputs)


def installed_versions() -> dict[str, str]:
    packages = (
        "compressed-tensors",
        "datasets",
        "llmcompressor",
        "safetensors",
        "torch",
        "transformers",
    )
    return {name: importlib.metadata.version(name) for name in packages}


def build_recipe(dampening_frac: float) -> Any:
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier

    return GPTQModifier(
        targets="Linear",
        scheme="MXFP4A16",
        ignore=list(IGNORED_LAYERS),
        dampening_frac=dampening_frac,
    )


def _walk_values(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)
    else:
        yield value


def _matching_dicts(value: Any, key: str, expected: Any) -> bool:
    if isinstance(value, dict):
        if value.get(key) == expected:
            return True
        return any(_matching_dicts(child, key, expected) for child in value.values())
    if isinstance(value, list):
        return any(_matching_dicts(child, key, expected) for child in value)
    return False


def _has_mxfp4_weights(value: Any) -> bool:
    if isinstance(value, dict):
        scale_dtype = value.get("scale_dtype")
        matches = (
            value.get("num_bits") == 4
            and value.get("group_size") == 32
            and value.get("type") == "float"
            and str(scale_dtype).removeprefix("torch.").lower() == "uint8"
        )
        if matches:
            return True
        return any(_has_mxfp4_weights(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_mxfp4_weights(child) for child in value)
    return False


def validate_output_config(config: dict[str, Any]) -> dict[str, Any]:
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise TypeError("output config lacks quantization_config")
    if quantization.get("quant_method") != "compressed-tensors":
        raise ValueError("output quant_method is not compressed_tensors")
    if not _matching_dicts(quantization, "format", "mxfp4-pack-quantized"):
        raise ValueError("output format is not mxfp4-pack-quantized")
    if not _matching_dicts(quantization, "num_bits", 4):
        raise ValueError("output does not record four-bit weights")
    if not _matching_dicts(quantization, "group_size", 32):
        raise ValueError("output does not record 32-element MXFP4 groups")
    if not _has_mxfp4_weights(quantization):
        raise ValueError("output does not record float MXFP4 with uint8 E8M0 scales")
    validate_audio_config(config)
    return {
        "quant_method": quantization.get("quant_method"),
        "format": "mxfp4-pack-quantized",
        "audio_config": AUDIO_CONFIG,
    }


def _is_uint8(dtype: Any) -> bool:
    return str(dtype).removeprefix("torch.").removeprefix("str.").lower() in {"uint8", "u8"}


def validate_safetensors_header(
    packed: dict[str, tuple[Any, list[int]]],
    scales: dict[str, tuple[Any, list[int]]],
) -> dict[str, int | list[str]]:
    if len(packed) < 1:
        raise ValueError("no packed MXFP4 tensors found")
    if len(scales) < 1:
        raise ValueError("no MXFP4 E8M0 scale tensors found")
    bad_dtypes = {
        **{name: str(dtype) for name, (dtype, _) in packed.items() if not _is_uint8(dtype)},
        **{name: str(dtype) for name, (dtype, _) in scales.items() if not _is_uint8(dtype)},
    }
    if bad_dtypes:
        raise ValueError(f"non-uint8 MXFP4 packed tensors: {bad_dtypes}")
    ignored = [
        name
        for name in packed
        if "audio_tower" in name
        or "multi_modal_projector" in name
        or name.startswith("lm_head.")
        or ".lm_head." in name
    ]
    if ignored:
        raise ValueError(f"ignored model component was quantized: {ignored[:10]}")
    bad_shapes = []
    for name, (_, shape) in packed.items():
        if len(shape) != 2:
            bad_shapes.append((name, shape))
    if bad_shapes:
        raise ValueError(f"unexpected packed tensor ranks: {bad_shapes[:10]}")
    return {
        "packed_tensors": len(packed),
        "scale_tensors": len(scales),
        "quantized_ignored_tensors": ignored,
    }


def audit_output(output: Path) -> dict[str, Any]:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    config_audit = validate_output_config(config)
    from safetensors import safe_open

    packed: dict[str, tuple[Any, list[int]]] = {}
    scales: dict[str, tuple[Any, list[int]]] = {}
    shard_names: list[str] = []
    for shard in sorted(output.glob("*.safetensors")):
        shard_names.append(shard.name)
        with safe_open(shard, framework="numpy") as archive:
            for name in list(archive.keys()):
                tensor = archive.get_slice(name)
                dtype = tensor.get_dtype()
                shape = tensor.get_shape()
                if name.endswith("weight_packed"):
                    packed[name] = (dtype, shape)
                elif name.endswith("weight_scale"):
                    scales[name] = (dtype, shape)
    tensor_audit = validate_safetensors_header(packed, scales)
    return {
        **config_audit,
        **tensor_audit,
        "shards": shard_names,
    }


def hash_output(output: Path) -> dict[str, str]:
    files = sorted(item for item in output.rglob("*") if item.is_file())
    return {str(item.relative_to(output)): sha256_file(item) for item in files}


def prepare_output_dir(path: Path, *, allow_existing: bool) -> None:
    if path.exists() and any(path.iterdir()) and not allow_existing:
        raise RuntimeError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-slice", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest-out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", default="voxtral-mxfp4-audio-v1")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--allow-splits", nargs="+", default=["train"])
    parser.add_argument("--overwrite-calibration-manifest", action="store_true")
    parser.add_argument("--base", default=DEFAULT_MODEL)
    parser.add_argument("--download-base-to", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--dampening-frac", type=float, default=0.01)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--device-map", default=DEFAULT_DEVICE_MAP)
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument(
        "--sequential-offload-device",
        default=DEFAULT_SEQUENTIAL_OFFLOAD_DEVICE,
    )
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _write_calibration_artifact(args: argparse.Namespace, selection: CalibrationSelection) -> str:
    if args.calibration_manifest_out.exists() and not args.overwrite_calibration_manifest:
        raise RuntimeError(
            f"calibration manifest already exists: {args.calibration_manifest_out}"
        )
    return write_calibration_manifest(args.calibration_manifest_out, selection)


def _base_receipt(args: argparse.Namespace, selection: CalibrationSelection) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base_model": DEFAULT_MODEL,
        "base_revision": DEFAULT_REVISION,
        "recipe": {
            "method": "GPTQ",
            "scheme": "MXFP4A16",
            "format": "mxfp4-pack-quantized",
            "targets": "Linear",
            "ignore": list(IGNORED_LAYERS),
            "dampening_frac": args.dampening_frac,
        },
        "calibration": selection.stats,
        "runtime": {
            "device_map": args.device_map,
            "pipeline": args.pipeline,
            "sequential_offload_device": args.sequential_offload_device,
        },
        "output": str(args.output),
        "privacy": {
            "eval_slice_excluded": True,
            "audio_text_pairs": True,
            "transcripts_in_receipt": False,
            "provider": "local-B550-only",
        },
    }


def run(args: argparse.Namespace) -> None:
    selection = select_calibration_rows(
        args.manifest,
        args.eval_slice,
        args.audio_root,
        limit=args.limit,
        seed=args.seed,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        allow_splits=args.allow_splits,
    )
    base_payload = _base_receipt(args, selection)

    if args.dry_run:
        plan = {
            **base_payload,
            "status": "dry-run",
            "download_allow_patterns": list(DOWNLOAD_ALLOW_PATTERNS),
            "download_blocked_names": list(DOWNLOAD_BLOCKED_NAMES),
            "expected_base_shards": EXPECTED_SHARDS,
            "calibration_manifest": str(args.calibration_manifest_out),
            "non_claims": [
                "does not load the 24B model",
                "does not prove GPTQ quality",
                "does not prove vLLM execution",
            ],
        }
        print(json.dumps(plan, indent=1, sort_keys=True), flush=True)
        return

    receipt_path = args.receipt or args.output.parent / f"{args.output.name}-receipt.json"
    if receipt_path.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("receipt must live outside the hashed output directory")
    calibration_sha256 = _write_calibration_artifact(args, selection)

    started = time.time()
    running_receipt = {
        **base_payload,
        "status": "running",
        "started_at_unix": started,
        "calibration_manifest_sha256": calibration_sha256,
    }
    write_json(receipt_path, running_receipt)
    try:
        if args.download_base_to is not None:
            args.base = str(download_base_snapshot(args.download_base_to))
        base_audit = verify_base_snapshot(Path(args.base))
        versions = installed_versions()
        prepare_output_dir(args.output, allow_existing=args.allow_existing_output)

        import torch
        import transformers
        from llmcompressor import oneshot

        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("Voxtral MXFP4 GPTQ requires CUDA on B550")
        write_json(
            receipt_path,
            {
                **running_receipt,
                "base_audit": base_audit,
                "runtime_versions": versions,
                "cuda_device_count": torch.cuda.device_count(),
            },
        )

        revision = DEFAULT_REVISION if args.base == DEFAULT_MODEL else None
        processor = transformers.AutoProcessor.from_pretrained(
            args.base, revision=revision
        )
        dataset = prepare_calibration_dataset(selection, processor)
        model = transformers.AutoModelForMultimodalLM.from_pretrained(
            args.base,
            revision=revision,
            dtype=torch.bfloat16,
            device_map=args.device_map,
        )
        patch_audio_config(model.config)
        recipe = build_recipe(args.dampening_frac)
        collator = VoxtralAudioCollator(processor, dtype=model.dtype)
        oneshot(
            model=model,
            processor=processor,
            dataset=dataset,
            recipe=recipe,
            output_dir=str(args.output),
            save_compressed=True,
            batch_size=1,
            data_collator=collator,
            num_calibration_samples=len(dataset),
            shuffle_calibration_samples=False,
            max_seq_length=args.max_seq_length,
            pad_to_max_length=False,
            pipeline=args.pipeline,
            sequential_offload_device=args.sequential_offload_device,
        )

        output_config_path = args.output / "config.json"
        output_config = json.loads(output_config_path.read_text(encoding="utf-8"))
        patch_audio_config(output_config)
        write_json(output_config_path, output_config)
        audit = audit_output(args.output)
        output_hashes = hash_output(args.output)
        completed = {
            **base_payload,
            "status": "complete",
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "wall_s": round(time.time() - started, 3),
            "calibration_manifest_sha256": calibration_sha256,
            "base_audit": base_audit,
            "runtime_versions": versions,
            "output_audit": audit,
            "output_file_count": len(output_hashes),
            "output_sha256": output_hashes,
            "non_claims": [
                "serialization audit does not prove transcription quality",
                "success does not prove vLLM kernel compatibility",
                "quality requires the frozen 500-row Phonon slice",
            ],
        }
        write_json(receipt_path, completed)
        print("DONE", args.output, flush=True)
        print("RECEIPT", receipt_path, flush=True)
    except Exception as error:
        write_json(
            receipt_path,
            {
                **running_receipt,
                "status": "failed",
                "finished_at_unix": time.time(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
