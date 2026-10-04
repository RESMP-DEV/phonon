#!/usr/bin/env python3
"""CPT for the reverse LFM2.5 graft: Audio head -> VL language model."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from liquid_audio.data.dataloader import LFM2DataLoader, lfm2_collator
from peft import LoraConfig, inject_adapter_in_model
from reverse_audio_vl import LORA_TARGETS, ReverseAudioVL, value
from safetensors.torch import save_file
from torch.utils.data import DataLoader


def trainable_state(model: ReverseAudioVL) -> dict[str, torch.Tensor]:
    payload = {}
    for name, parameter in model.vl.named_parameters():
        if "lora_" in name:
            payload["vl." + name] = parameter.detach().cpu().contiguous()
    for name, parameter in model.audio.audio_adapter.named_parameters():
        payload["audio_adapter." + name] = parameter.detach().cpu().contiguous()
    return payload


def load_checkpoint(
    path: Path, model: ReverseAudioVL, optimizer: torch.optim.Optimizer
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vl_state = {k.removeprefix("vl."): v for k, v in payload["sd"].items() if k.startswith("vl.")}
    adapter_state = {k.removeprefix("audio_adapter."): v for k, v in payload["sd"].items() if k.startswith("audio_adapter.")}
    result_vl = model.vl.load_state_dict(vl_state, strict=False)
    model.audio.audio_adapter.load_state_dict(adapter_state, strict=True)
    if result_vl.unexpected_keys:
        raise RuntimeError(f"unexpected LoRA keys: {result_vl.unexpected_keys[:5]}")
    optimizer.load_state_dict(payload["opt"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if torch.cuda.is_available() and "cuda_rng" in payload:
        torch.cuda.set_rng_state(payload["cuda_rng"].cpu(), torch.cuda.current_device())
    return int(payload["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aqua-hq-seq-v1")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--context-length", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--keep-ckpts", type=int, default=3)
    parser.add_argument("--out", default="reverse-audio-vl-v1")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="profile-only Torch Inductor/Triton compilation of the logits path",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = ReverseAudioVL.from_pretrained(device=device)
    for parameter in model.vl.parameters():
        parameter.requires_grad_(False)
    for parameter in model.audio.parameters():
        parameter.requires_grad_(False)
    for parameter in model.audio.audio_adapter.parameters():
        parameter.requires_grad_(True)

    inject_adapter_in_model(
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGETS,
        ),
        model.vl,
    )
    model.vl.to(device)
    model.audio.to(device)
    logits_function = model.logits
    if args.torch_compile:
        logits_function = torch.compile(model.logits, dynamic=True)

    lora_parameters = [p for n, p in model.vl.named_parameters() if p.requires_grad]
    adapter_parameters = list(model.audio.audio_adapter.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.lr},
            {"params": adapter_parameters, "lr": args.adapter_lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    print(
        f"trainable: lora={len(lora_parameters)} ({sum(p.numel() for p in lora_parameters):,}), "
        f"adapter={len(adapter_parameters)} ({sum(p.numel() for p in adapter_parameters):,})",
        flush=True,
    )
    if not lora_parameters:
        raise RuntimeError("LoRA target regex matched no VL tensors")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    if args.resume:
        path = Path(args.resume)
        if args.resume == "auto":
            candidates = sorted(out_dir.glob("ckpt_step*.pt"))
            if not candidates:
                raise RuntimeError(f"no checkpoints in {out_dir}")
            path = candidates[-1]
        step = load_checkpoint(path, model, optimizer)
        print(f"resumed step {step} from {path}", flush=True)

    dataset = LFM2DataLoader(args.dataset, context_length=args.context_length)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lfm2_collator,
        num_workers=0,
    )
    iterator = iter(loader)
    losses: list[float] = []
    started = time.time()

    def lr_at(current: int) -> float:
        if current < args.warmup:
            return args.lr * (current + 1) / args.warmup
        progress = (current - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    def save_checkpoint(current: int) -> None:
        payload = {
            "step": current,
            "sd": trainable_state(model),
            "opt": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "losses": losses[-50:],
        }
        if device.type == "cuda":
            payload["cuda_rng"] = torch.cuda.get_rng_state(device)
        temporary = out_dir / ".ckpt_tmp.pt"
        torch.save(payload, temporary)
        temporary.replace(out_dir / f"ckpt_step{current:05d}.pt")
        if args.keep_ckpts > 0:
            old = sorted(out_dir.glob("ckpt_step*.pt"))
            for path in old[: -args.keep_ckpts]:
                path.unlink()

    while step < args.steps:
        torch.cuda.nvtx.range_push(f"step_{step:06d}")
        torch.cuda.nvtx.range_push("load_batch")
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("batch_to_device")
        batch = batch.to(device)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("forward")
        logits = logits_function(batch)
        ids = model.full_ids(batch)
        supervision = value(batch, "supervision_mask")
        labels = ids[:, 1:].clone()
        labels[~supervision[:, 1:]] = -100
        flat_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
        flat_labels = labels.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("optimizer")
        optimizer.param_groups[0]["lr"] = lr_at(step)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.vl.parameters() if p.requires_grad]
            + list(model.audio.audio_adapter.parameters()),
            1.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.nvtx.range_pop()
        losses.append(float(loss.detach()))
        step += 1
        torch.cuda.nvtx.range_pop()
        if step == 1 or step % 10 == 0:
            mean = sum(losses[-10:]) / min(10, len(losses))
            print(f"step {step}/{args.steps} loss {mean:.4f} ({time.time()-started:.0f}s)", flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_checkpoint(step)

    save_file(trainable_state(model), str(out_dir / "reverse_audio_vl_adapter.safetensors"))
    (out_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "dataset": args.dataset,
                "context_length": args.context_length,
                "lr": args.lr,
                "adapter_lr": args.adapter_lr,
                "rank": args.rank,
                "torch_compile": args.torch_compile,
                "mean_last10": sum(losses[-10:]) / min(10, len(losses)),
                "wall_s": round(time.time() - started, 1),
            },
            indent=1,
        )
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
