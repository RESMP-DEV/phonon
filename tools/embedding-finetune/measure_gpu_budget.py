#!/usr/bin/env python3
"""Measure EmbeddingGemma 2 fine-tune VRAM and throughput on real CUDA silicon.

This script exists to turn GPU selection into a measured decision instead of a
spec-sheet guess. Every input is synthetic random token ids: no private corpus,
transcript, audio, or dictionary value is read, sent, or written, so the probe is
safe to run on a rented third-party provider.

Two training shapes are measured because they have very different memory floors:

- ``lora`` freezes the backbone and trains adapters on the text tower, so
  activation memory (batch x 3 triplet columns x seq) dominates.
- ``full`` trains every backbone parameter, so fp32 optimizer state dominates.

``--grad-checkpoint`` is the decisive lever: measured peak allocated memory drops
from 21.39 GiB to 4.66 GiB at batch 8 / seq 512, at roughly 1.4x the step time.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"

# The text tower uses plain nn.Linear at these paths. The audio tower instead
# wraps projections in Gemma4ClippableLinear whose leaf name is "linear", so a
# bare "linear" target silently matches only the audio tower: text-only forward
# then produces zero gradient. Target the text tower explicitly.
TEXT_TOWER_TARGETS = (
    r"language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)


class MeasureError(RuntimeError):
    """Raised when the probe cannot establish a usable CUDA baseline."""


def load_torch() -> Any:
    """Import torch lazily so pure-logic use needs no CUDA stack installed."""
    try:
        import torch
    except ImportError as error:
        raise MeasureError(
            "this probe needs torch; run it in the pinned training environment"
        ) from error
    return torch


def infonce(anchor: Any, positive: Any, negative: Any) -> Any:
    """MultipleNegativesRankingLoss objective on already-pooled embeddings."""
    torch = load_torch()
    sim_pos = torch.nn.functional.cosine_similarity(
        anchor.unsqueeze(1), positive.unsqueeze(0), dim=2
    )
    sim_neg = torch.nn.functional.cosine_similarity(
        anchor.unsqueeze(1), negative.unsqueeze(0), dim=2
    )
    scores = torch.cat([sim_pos, sim_neg], dim=1)
    labels = torch.arange(scores.size(0), device=scores.device)
    return torch.nn.functional.cross_entropy(scores, labels)


def build_model(device: str, seq: int, mode: str, rank: int) -> tuple[Any, Any]:
    """Load the pinned model and attach LoRA adapters for the ``lora`` mode."""
    from sentence_transformers import SentenceTransformer

    torch = load_torch()

    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device=device,
        model_kwargs={"torch_dtype": torch.bfloat16},
        config_kwargs={"vision_config": None},
    )
    model.max_seq_length = seq

    if mode == "full":
        for param in model.parameters():
            param.requires_grad_(True)
        return model, None

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        task_type="FEATURE_EXTRACTION",
        target_modules=TEXT_TOWER_TARGETS,
    )
    adapted = get_peft_model(model[0].auto_model, config)
    model[0].auto_model = adapted
    return model, adapted


def run_case(
    *,
    device: str,
    batch: int,
    seq: int,
    mode: str,
    rank: int,
    grad_checkpoint: bool,
    seconds: float,
    max_steps: int,
) -> dict[str, Any]:
    """Measure one configuration and return aggregate VRAM/throughput facts."""
    torch = load_torch()
    torch.cuda.set_device(device)
    torch.zeros(1, device=device)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(torch.device(device))

    model, adapted = build_model(device, seq, mode, rank)
    model.train()
    if adapted is not None:
        adapted.train()
    if grad_checkpoint and adapted is not None:
        adapted.gradient_checkpointing_enable()
    elif grad_checkpoint:
        raise MeasureError("grad checkpointing requires lora mode in this probe")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5
    )

    # Synthetic random tokens only. Range avoids special ids.
    ids = torch.randint(1000, 200000, (batch * 3, seq), device=device)
    attention = torch.ones(batch * 3, seq, dtype=torch.long, device=device)
    features = {"input_ids": ids, "attention_mask": attention}

    elapsed: list[float] = []
    steps = 0
    first_loss: float | None = None
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and steps < max_steps:
        torch.cuda.synchronize()
        started = time.monotonic()
        embeddings = model(features)["sentence_embedding"]
        loss = infonce(
            embeddings[:batch], embeddings[batch : 2 * batch], embeddings[2 * batch :]
        )
        if first_loss is None:
            first_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed.append(time.monotonic() - started)
        steps += 1

    if steps == 0:
        raise MeasureError("no training step completed before the time budget")

    result = {
        "mode": mode,
        "batch": batch,
        "seq": seq,
        "lora_r": rank if mode == "lora" else None,
        "grad_ckpt": grad_checkpoint,
        "trainable_params": trainable,
        "total_params": total,
        "steps": steps,
        "first_loss": round(first_loss, 4) if first_loss is not None else None,
        "sec_per_step": round(sum(elapsed) / len(elapsed), 4),
        "triplets_per_sec": round(steps * batch / sum(elapsed), 2),
        "peak_alloc_gib": round(
            torch.cuda.max_memory_allocated(torch.device(device)) / 2**30, 3
        ),
        "peak_reserved_gib": round(
            torch.cuda.max_memory_reserved(torch.device(device)) / 2**30, 3
        ),
        "gpu": torch.cuda.get_device_name(device),
        "vram_gib": round(
            torch.cuda.get_device_properties(device).total_memory / 2**30, 2
        ),
    }
    del model, adapted, features, ids, attention
    return result


def parse_cases(raw: str) -> list[tuple[int, int, str, bool]]:
    """Parse ``batch:seq:mode[:ckpt]`` case specs."""
    cases: list[tuple[int, int, str, bool]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in (3, 4) or parts[2] not in ("lora", "full"):
            raise MeasureError(f"bad case spec: {item}")
        grad_checkpoint = len(parts) == 4 and parts[3] == "ckpt"
        cases.append((int(parts[0]), int(parts[1]), parts[2], grad_checkpoint))
    if not cases:
        raise MeasureError("no cases requested")
    return cases


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Parse GPU-budget probe arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cases",
        default="8:512:lora:ckpt,16:512:lora:ckpt,8:512:lora,8:512:full",
        help="comma-separated batch:seq:mode[:ckpt] configurations",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--seconds-per-case", type=float, default=40.0)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--out", type=Path, help="optional JSON report path")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Measure each configuration and print one JSON record per line."""
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    torch = load_torch()
    if not torch.cuda.is_available():
        print("FAIL gpu-budget: CUDA is unavailable on this host", file=sys.stderr)
        return 2
    try:
        cases = parse_cases(arguments.cases)
    except MeasureError as error:
        print(f"FAIL gpu-budget: {error}", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    for batch, seq, mode, grad_checkpoint in cases:
        try:
            record = run_case(
                device=arguments.device,
                batch=batch,
                seq=seq,
                mode=mode,
                rank=arguments.lora_rank,
                grad_checkpoint=grad_checkpoint,
                seconds=arguments.seconds_per_case,
                max_steps=arguments.max_steps,
            )
        except torch.cuda.OutOfMemoryError:
            record = {
                "mode": mode,
                "batch": batch,
                "seq": seq,
                "lora_r": arguments.lora_rank if mode == "lora" else None,
                "grad_ckpt": grad_checkpoint,
                "oom": True,
                "gpu": torch.cuda.get_device_name(arguments.device),
            }
        except Exception as error:  # noqa: BLE001 - probe failure boundary
            record = {
                "mode": mode,
                "batch": batch,
                "seq": seq,
                "lora_r": arguments.lora_rank if mode == "lora" else None,
                "grad_ckpt": grad_checkpoint,
                "error": f"{type(error).__name__}: {error}"[:300],
            }
        results.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        torch.cuda.empty_cache()
        gc.collect()

    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
