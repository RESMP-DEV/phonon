#!/usr/bin/env python3
"""Evaluate a fused audio-native SALM checkpoint on real and blank screenshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer
import torch
from liquid_audio import LFM2AudioModel
from PIL import Image
from safetensors.torch import load_file
from salm_vision import VisionDataLoader, generate_text, install_vision, vision_collator
from transformers import AutoImageProcessor, AutoTokenizer


def load_merged_language(model: LFM2AudioModel, path: Path) -> int:
    target = model.state_dict()
    merged = load_file(str(path))
    adapted = {}
    for key, tensor in merged.items():
        if "lora_" in key:
            continue
        candidate = f"lfm.{key}".replace(".base_layer.", ".")
        if candidate in target and target[candidate].shape == tensor.shape:
            adapted[candidate] = tensor
    if not adapted:
        raise RuntimeError("no merged language tensors matched the base model")
    missing, unexpected = model.load_state_dict(adapted, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected merged keys: {unexpected[:5]}")
    print(
        f"merged language weights applied: {len(adapted)}/{len(merged)} keys; "
        f"base-model keys left unchanged={len(missing)}",
        flush=True,
    )
    return len(adapted)


def metrics(refs: list[str], hyps: list[str]) -> dict[str, float | int]:
    return {
        "n": len(refs),
        "wer": float(jiwer.wer(refs or ["<empty>"], hyps or ["<empty>"])),
        "exact": sum(ref == hyp for ref, hyp in zip(refs, hyps)) / max(1, len(refs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--pack", default="/home/kearm/salm-lora/aqua-vision-dataset-v1-eval")
    parser.add_argument("--transplant-dir", default="/home/kearm/salm-lora/vision-transplant")
    parser.add_argument("--init-variant", default="omp")
    parser.add_argument("--tokenizer", default="/home/kearm/salm-lora/vision-transplant/tokenizer")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = LFM2AudioModel.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B", device=device, dtype=torch.bfloat16
    )
    load_merged_language(model, args.merged)
    install_vision(
        model,
        rows_path=Path(args.transplant_dir) / f"rows_{args.init_variant}.safetensors",
        init_variant=args.init_variant,
        verbose=False,
    )
    model.eval()

    processor = AutoImageProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B")
    real_processor = processor
    dataset = VisionDataLoader(args.pack, processor, context_length=args.context_length)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = min(args.limit, len(dataset))
    results: dict[str, list[dict[str, object]]] = {"real": [], "blank": []}

    def references(index: int) -> str:
        row = dataset.dataset[index]
        ids = [
            token
            for token, supervised in zip(row["input_ids"], row["supervision_mask"], strict=True)
            if supervised
        ]
        return tokenizer.decode(ids).replace("<|im_end|>", "").strip()

    for condition in ("real", "blank"):
        if condition == "blank":
            dataset.ip = lambda image, **kwargs: real_processor(
                Image.new("RGB", image.size, (255, 255, 255)), **kwargs
            )
        for index in range(rows):
            batch = vision_collator([dataset[index]]).to(device)
            output_ids = generate_text(model, batch, max_new_tokens=args.max_new_tokens)
            hypothesis = tokenizer.decode(output_ids).strip()
            results[condition].append(
                {"idx": index, "ref": references(index), "hyp": hypothesis}
            )
            if (index + 1) % 25 == 0 or index + 1 == rows:
                print(f"{condition} {index + 1}/{rows}", flush=True)
        dataset.ip = real_processor

    summary = {
        condition: metrics(
            [str(row["ref"]) for row in values], [str(row["hyp"]) for row in values]
        )
        for condition, values in results.items()
    }
    payload = {
        "summary": summary,
        "rows": results,
        "merged": str(args.merged),
        "notes": "blank uses the same token layout with a white image",
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
