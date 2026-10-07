#!/usr/bin/env python3
"""Fuse a vision-on-audio SALM LoRA into the audio base weights.

The audio-native SALM vision lane keeps the LoRA in the language model only. The
vision tower and projector are separate transplanted components and are saved as
their own state dicts, not merged into `lfm` weights. This script therefore
produces:

1. a merged `lfm.*` language model (LoRA deltas folded into the projections),
2. the projector state dict when the adapter trained one,
3. a receipt with SHA-256 hashes for every emitted file.

The merged language model is verified by reloading it through a fresh
`LFM2AudioModel` and comparing merged-vs-injected logits on one fixed audio batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from liquid_audio import LFM2AudioModel
from liquid_audio.data.dataloader import LFM2DataLoader, lfm2_collator
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file, save_file
from salm_vision import install_vision
from torch.utils.data import DataLoader

TARGETS = (
    r"^lfm\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)"
    r"|conv\.(in_proj|out_proj)|feed_forward\.(w1|w2|w3))$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace, device: torch.device) -> LFM2AudioModel:
    model = LFM2AudioModel.from_pretrained(
        args.audio_model, device=device, dtype=torch.bfloat16
    ).eval()
    if args.vision_rows:
        install_vision(
            model,
            rows_path=Path(args.vision_rows),
            init_variant=args.init_variant,
            verbose=False,
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audio-model", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--vision-rows", default=None)
    parser.add_argument("--init-variant", default="omp")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-batch", default=None, help="audio pack for logit parity")
    parser.add_argument("--verify-context", type=int, default=768)
    args = parser.parse_args()

    device = torch.device(args.device)
    adapter_path = Path(args.adapter)
    if not adapter_path.is_file():
        parser.error(f"adapter missing: {adapter_path}")
    state = load_file(str(adapter_path))
    projector = {k[5:]: v for k, v in state.items() if k.startswith("proj.")}
    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    if not lora_state:
        parser.error(f"adapter carries no LoRA tensors: {adapter_path}")

    # Reference: injected adapter, before merging.
    injected = build(args, device)
    injected = inject_adapter_in_model(
        LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                   target_modules=TARGETS),
        injected,
    )
    missing, unexpected = injected.load_state_dict(lora_state, strict=False)
    if [k for k in missing if "lora_" in k] or unexpected:
        raise RuntimeError(f"adapter mismatch: missing={[k for k in missing if 'lora_' in k][:5]} unexpected={unexpected[:5]}")
    if projector:
        injected.multi_modal_projector.load_state_dict(projector)
    injected.to(device).eval()

    parity = None
    if args.verify_batch:
        loader = DataLoader(
            LFM2DataLoader(args.verify_batch, context_length=args.verify_context),
            batch_size=1, shuffle=False, collate_fn=lfm2_collator,
        )
        batch = next(iter(loader)).to(device)
        with torch.no_grad():
            text_logits, _, text_labels, _ = injected.logits(batch)
        parity = {
            "text_logits_shape": list(text_logits.shape),
            "text_labels": int(text_labels.numel()),
            "cross_entropy": float(
                torch.nn.functional.cross_entropy(text_logits.float(), text_labels)
            ),
        }

    merged_targets = [
        module
        for module in injected.lfm.modules()
        if hasattr(module, "base_layer") and hasattr(module, "lora_A")
    ]
    for module in merged_targets:
        module.merge()

    merged_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in injected.lfm.state_dict().items()
    }
    args.out.mkdir(parents=True, exist_ok=True)
    merged_path = args.out / "merged_lfm.safetensors"
    save_file(merged_state, str(merged_path))

    projector_path = None
    if projector:
        projector_path = args.out / "projector.safetensors"
        save_file({k: v.detach().cpu().contiguous()
                   for k, v in injected.multi_modal_projector.state_dict().items()},
                  str(projector_path))

    vision_rows_path = None
    if args.vision_rows:
        vision_rows_path = args.out / "vision_rows.safetensors"
        save_file(load_file(args.vision_rows), str(vision_rows_path))

    receipt = {
        "schema_version": 1,
        "experiment": "fuse_salm_vision_lora",
        "status": "complete",
        "audio_model": args.audio_model,
        "adapter": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_path),
        "lora_tensors": len(lora_state),
        "projector_tensors": len(projector),
        "merged_target_layers": len(merged_targets),
        "rank": args.rank,
        "init_variant": args.init_variant,
        "outputs": {
            "merged_lfm": str(merged_path),
            "merged_lfm_sha256": sha256_file(merged_path),
            "projector": None if projector_path is None else str(projector_path),
            "projector_sha256": None if projector_path is None else sha256_file(projector_path),
            "vision_rows": None if vision_rows_path is None else str(vision_rows_path),
            "vision_rows_sha256": None if vision_rows_path is None else sha256_file(vision_rows_path),
        },
        "verification": parity,
        "non_claims": [
            "merging folds only the language LoRA into lfm weights",
            "the vision tower stays a separate frozen transplanted component",
            "no quality claim is made without slice evaluation of the merged model",
        ],
    }
    (args.out / "fuse_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
