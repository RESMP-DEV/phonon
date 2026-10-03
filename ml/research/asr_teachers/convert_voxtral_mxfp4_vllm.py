#!/usr/bin/env python3
"""Convert Phonon's HF Voxtral MXFP4 artifact to vLLM Mistral names."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from quantize_voxtral_mxfp4 import (
    audit_output,
    hash_output,
    patch_audio_config,
    validate_output_config,
    write_json,
)

VLLM_IGNORE_LAYERS = (
    "re:.*whisper_encoder.*",
    "re:.*audio_language_adapter.*",
)


def rename_weight(name: str) -> str | None:
    """Map official HF Voxtral names to vLLM 0.30 Mistral names."""
    if name.startswith("language_model."):
        return name.removeprefix("language_model.")
    if name == "audio_tower.embed_positions.weight":
        return None
    if name == "audio_tower.layer_norm.weight":
        return "mm_whisper_embeddings.whisper_encoder.transformer.norm.weight"
    if name == "audio_tower.layer_norm.bias":
        return "mm_whisper_embeddings.whisper_encoder.transformer.norm.bias"
    if name in {
        "audio_tower.conv1.weight",
        "audio_tower.conv1.bias",
        "audio_tower.conv2.weight",
        "audio_tower.conv2.bias",
    }:
        part, suffix = name.removeprefix("audio_tower.").split(".", 1)
        index = "0" if part == "conv1" else "1"
        return f"mm_whisper_embeddings.whisper_encoder.conv_layers.{index}.{suffix}"
    if name == "multi_modal_projector.linear_1.weight":
        return "mm_whisper_embeddings.audio_language_projection.0.weight"
    if name == "multi_modal_projector.linear_2.weight":
        return "mm_whisper_embeddings.audio_language_projection.2.weight"

    layer = re.fullmatch(r"audio_tower\.layers\.(\d+)\.(.+)", name)
    if layer is None:
        return name
    index, suffix = layer.groups()
    replacements = {
        "self_attn.q_proj.": "attention.wq.",
        "self_attn.k_proj.": "attention.wk.",
        "self_attn.v_proj.": "attention.wv.",
        "self_attn.out_proj.": "attention.wo.",
        "self_attn_layer_norm.": "attention_norm.",
        "fc1.": "feed_forward.w1.",
        "fc2.": "feed_forward.w2.",
        "final_layer_norm.": "ffn_norm.",
    }
    for source, target in replacements.items():
        if suffix.startswith(source):
            suffix = target + suffix.removeprefix(source)
            break
    else:
        raise ValueError(f"unmapped Voxtral audio tensor: {name}")
    return (
        "mm_whisper_embeddings.whisper_encoder.transformer."
        f"layers.{index}.{suffix}"
    )


def update_vllm_config(config: dict[str, Any]) -> dict[str, Any]:
    validate_output_config(config)
    patch_audio_config(config)
    quantization = config["quantization_config"]
    ignore = [
        "output" if entry == "lm_head" else entry
        for entry in quantization.get("ignore", [])
    ]
    ignore = list(dict.fromkeys([*ignore, "output", *VLLM_IGNORE_LAYERS]))
    quantization["ignore"] = ignore
    return config


def prepare_output(source: Path, output: Path, *, allow_existing: bool) -> None:
    if output.exists() and any(output.iterdir()) and not allow_existing:
        raise RuntimeError(f"vLLM output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.suffix != ".safetensors":
            shutil.copy2(item, output / item.name)


def convert_shards(source: Path, output: Path) -> dict[str, Any]:
    from safetensors.torch import load_file, save_file

    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        new_weight_map: dict[str, str] = {}
        total_size = 0
        for shard_name in sorted(set(index.get("weight_map", {}).values())):
            source_shard = source / shard_name
            tensors = load_file(str(source_shard), device="cpu")
            renamed: dict[str, Any] = {}
            dropped: list[str] = []
            for name, tensor in tensors.items():
                new_name = rename_weight(name)
                if new_name is None:
                    dropped.append(name)
                    continue
                if new_name in renamed:
                    raise ValueError(f"duplicate renamed tensor: {new_name}")
                renamed[new_name] = tensor
                total_size += tensor.numel() * tensor.element_size()
            save_file(renamed, str(output / shard_name), metadata={"format": "pt"})
            print(
                f"converted {shard_name}: {len(renamed)} tensors, "
                f"dropped {len(dropped)} fixed-position tensors",
                flush=True,
            )
        for tensor_name, shard_name in index.get("weight_map", {}).items():
            new_name = rename_weight(tensor_name)
            if new_name is not None:
                new_weight_map[new_name] = shard_name
        new_index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
        (output / "model.safetensors.index.json").write_text(
            json.dumps(new_index, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        return new_index

    source_shard = source / "model.safetensors"
    tensors = load_file(str(source_shard), device="cpu")
    renamed = {
        new_name: tensor
        for name, tensor in tensors.items()
        if (new_name := rename_weight(name)) is not None
    }
    save_file(renamed, str(output / "model.safetensors"), metadata={"format": "pt"})
    return {"weight_map": {name: "model.safetensors" for name in renamed}}


def validate_vllm_header(output: Path) -> dict[str, int]:
    from safetensors import safe_open

    forbidden_prefixes = ("audio_tower.", "multi_modal_projector.")
    required_prefix = "mm_whisper_embeddings.whisper_encoder."
    required_projectors = {
        "mm_whisper_embeddings.audio_language_projection.0.weight",
        "mm_whisper_embeddings.audio_language_projection.2.weight",
    }
    names: list[str] = []
    quantized_audio = 0
    for shard in sorted(output.glob("*.safetensors")):
        with safe_open(shard, framework="numpy") as archive:
            for name in list(archive.keys()):
                names.append(name)
                if ("whisper_encoder" in name or "audio_language_projection" in name) and (
                    name.endswith(("weight_packed", "weight_scale"))
                ):
                    quantized_audio += 1
    forbidden = [name for name in names if name.startswith(forbidden_prefixes)]
    if forbidden:
        raise ValueError(f"unconverted HF audio names remain: {forbidden[:5]}")
    if not any(name.startswith(required_prefix) for name in names):
        raise ValueError("vLLM output lacks Mistral whisper encoder names")
    missing_projectors = sorted(required_projectors - set(names))
    if missing_projectors:
        raise ValueError(f"vLLM output lacks Mistral projector names: {missing_projectors}")
    if quantized_audio:
        raise ValueError(f"vLLM audio path was quantized: {quantized_audio} tensors")
    return {
        "tensors": len(names),
        "quantized_audio_tensors": quantized_audio,
        "mistral_encoder_tensors": sum(name.startswith(required_prefix) for name in names),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    source_audit = audit_output(args.model)
    write_json(
        args.receipt,
        {
            "schema_version": 1,
            "status": "running",
            "source": str(args.model),
            "source_audit": source_audit,
            "output": str(args.output),
            "started_at_unix": started,
        },
    )
    try:
        prepare_output(args.model, args.output, allow_existing=args.allow_existing_output)
        config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
        config = update_vllm_config(config)
        write_json(args.output / "config.json", config)
        index = convert_shards(args.model, args.output)
        header_audit = validate_vllm_header(args.output)
        compressed_audit = audit_output(args.output)
        hashes = hash_output(args.output)
        write_json(
            args.receipt,
            {
                "schema_version": 1,
                "status": "complete",
                "source": str(args.model),
                "source_audit": source_audit,
                "output": str(args.output),
                "started_at_unix": started,
                "finished_at_unix": time.time(),
                "wall_s": round(time.time() - started, 3),
                "index_tensors": len(index.get("weight_map", {})),
                "vllm_header_audit": header_audit,
                "compressed_audit": compressed_audit,
                "output_file_count": len(hashes),
                "output_sha256": hashes,
                "non_claims": [
                    "name conversion does not prove vLLM startup",
                    "name conversion does not change quantization quality",
                ],
            },
        )
        print("DONE", args.output, flush=True)
    except Exception as error:
        write_json(
            args.receipt,
            {
                "schema_version": 1,
                "status": "failed",
                "source": str(args.model),
                "output": str(args.output),
                "started_at_unix": started,
                "finished_at_unix": time.time(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


if __name__ == "__main__":
    main()
