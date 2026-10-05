#!/usr/bin/env python3
"""CPT for the reverse LFM2.5 graft: Audio head -> VL language model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from liquid_audio.data.dataloader import LFM2DataLoader, lfm2_collator
from peft import LoraConfig, inject_adapter_in_model
from reverse_audio_vl import LORA_TARGETS, ReverseAudioVL, value
from safetensors.torch import save_file
from torch.utils.data import DataLoader


def trainable_state(
    model: ReverseAudioVL, *, conformer_blocks: int = 0
) -> dict[str, torch.Tensor]:
    payload = {}
    for name, parameter in model.vl.named_parameters():
        if "lora_" in name:
            payload["vl." + name] = parameter.detach().cpu().contiguous()
    for name, parameter in model.audio.audio_adapter.named_parameters():
        payload["audio_adapter." + name] = parameter.detach().cpu().contiguous()
    if conformer_blocks:
        for name, parameter in conformer_parameters(model).items():
            payload["conformer." + name] = parameter.detach().cpu().contiguous()
    return payload


def conformer_layers(model: ReverseAudioVL) -> torch.nn.ModuleList:
    layers = getattr(model.audio.conformer, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        raise TypeError("conformer has no nn.ModuleList attribute named 'layers'")
    return layers


def conformer_parameters(model: ReverseAudioVL) -> dict[str, torch.nn.Parameter]:
    """Parameters of unfrozen conformer layers, keyed by layer index."""

    payload: dict[str, torch.nn.Parameter] = {}
    for index, layer in enumerate(conformer_layers(model)):
        for name, parameter in layer.named_parameters():
            if parameter.requires_grad:
                payload[f"{index}.{name}"] = parameter
    return payload


def load_conformer_state(
    model: ReverseAudioVL, state: dict[str, torch.Tensor]
) -> None:
    """Restore selectively fine-tuned conformer layers from trainable state."""

    layers = conformer_layers(model)
    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        if "." not in key:
            raise RuntimeError(f"malformed conformer tensor key: {key}")
        index_text, parameter_name = key.split(".", 1)
        index = int(index_text)
        if not 0 <= index < len(layers):
            raise RuntimeError(f"conformer layer index out of range: {index}")
        by_layer.setdefault(index, {})[parameter_name] = tensor
    for index, layer_state in by_layer.items():
        layers[index].load_state_dict(layer_state, strict=True)


def pack_contract(dataset: Path) -> dict[str, Any]:
    """Read only non-textual pack identity fields for telemetry."""

    metadata_path = dataset / "pack_meta.json"
    metadata: dict[str, Any] = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    safe_keys = {
        "source",
        "revision",
        "manifest_sha256",
        "prompt_id",
        "prompt_sha256",
        "rows",
        "scanned",
        "skipped",
        "duration_range",
        "context_length",
    }
    safe_metadata = {key: metadata[key] for key in safe_keys if key in metadata}
    encoded = json.dumps(safe_metadata, sort_keys=True, separators=(",", ":")).encode()
    return {
        "metadata": safe_metadata,
        "metadata_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def optional_module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except (ImportError, AttributeError, RuntimeError):
        return None
    return str(getattr(module, "__version__", "present"))


def wandb_config(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    metadata = contract["metadata"]
    return {
        "stage": args.wandb_stage,
        "dataset": {
            "prompt_id": metadata.get("prompt_id"),
            "prompt_sha256": metadata.get("prompt_sha256"),
            "metadata_sha256": contract["metadata_sha256"],
            "rows": metadata.get("rows"),
            "source": metadata.get("source"),
            "revision": metadata.get("revision"),
            "manifest_sha256": metadata.get("manifest_sha256"),
        },
        "training": {
            "steps": args.steps,
            "context_length": args.context_length,
            "lr": args.lr,
            "adapter_lr": args.adapter_lr,
            "rank": args.rank,
            "lora_dropout": args.lora_dropout,
            "warmup": args.warmup,
            "torch_compile": args.torch_compile,
            "liger_cross_entropy": args.liger_cross_entropy,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "flash_attn": optional_module_version("flash_attn"),
            "causal_conv1d": optional_module_version("causal_conv1d"),
            "liger_kernel": optional_module_version("liger_kernel"),
        },
    }


def start_wandb(
    args: argparse.Namespace,
    out_dir: Path,
    contract: dict[str, Any],
):
    """Start optional telemetry; no audio, transcript, or path payload leaves here."""

    if not args.wandb_project or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "--wandb-project requires the wandb package in the research environment"
        ) from error

    # W&B writes its local transaction log beside the public/Aqua stage outputs.
    # Keeping it outside the stage directory prevents checkpoint retention from
    # disturbing a resumed run.
    # Offline mode intentionally ignores resume and emits one transaction per
    # process; sync_wandb_offline.py merges those transactions by run ID.
    wandb_dir = out_dir.parent / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    return wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        id=args.wandb_run_id,
        name=args.wandb_name,
        resume="allow" if args.wandb_run_id else "never",
        config=wandb_config(args, contract),
        tags=["phonon", "reverse-audio-vl", args.wandb_stage],
        allow_val_change=True,
    )


def load_checkpoint(
    path: Path, model: ReverseAudioVL, optimizer: torch.optim.Optimizer
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vl_state = {k.removeprefix("vl."): v for k, v in payload["sd"].items() if k.startswith("vl.")}
    adapter_state = {k.removeprefix("audio_adapter."): v for k, v in payload["sd"].items() if k.startswith("audio_adapter.")}
    conformer_state = {
        key.removeprefix("conformer."): value
        for key, value in payload["sd"].items()
        if key.startswith("conformer.")
    }
    result_vl = model.vl.load_state_dict(vl_state, strict=False)
    model.audio.audio_adapter.load_state_dict(adapter_state, strict=True)
    if conformer_state:
        load_conformer_state(model, conformer_state)
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
    parser.add_argument("--conformer-lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--keep-ckpts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument(
        "--conformer-blocks",
        type=int,
        default=0,
        help="Unfreeze the last N conformer layers and include them in the checkpoint",
    )
    parser.add_argument("--out", default="reverse-audio-vl-v1")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="profile-only Torch Inductor/Triton compilation of the selected forward path",
    )
    parser.add_argument(
        "--liger-cross-entropy",
        action="store_true",
        help="fuse lm_head and cross entropy without materializing logits",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="offline",
    )
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--wandb-stage",
        choices=("public", "aqua", "profile"),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.gradient_accumulation < 1:
        parser.error("--gradient-accumulation must be at least 1")
    if args.conformer_blocks < 0:
        parser.error("--conformer-blocks cannot be negative")
    if args.wandb_project and args.wandb_mode != "disabled" and not args.wandb_stage:
        parser.error("--wandb-stage is required when W&B telemetry is enabled")

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
    conformer_layers = getattr(model.audio.conformer, "layers", None)
    if args.conformer_blocks:
        if conformer_layers is None or not isinstance(conformer_layers, torch.nn.ModuleList):
            parser.error("conformer has no nn.ModuleList attribute named 'layers'")
        for layer in conformer_layers[-args.conformer_blocks :]:
            for parameter in layer.parameters():
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
    liger_loss = None
    if args.liger_cross_entropy:
        try:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        except ImportError as error:
            raise RuntimeError(
                "--liger-cross-entropy requires liger-kernel in the research environment"
            ) from error
        liger_loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

    forward_function = model.hidden_states if args.liger_cross_entropy else model.logits
    if args.torch_compile:
        forward_function = torch.compile(forward_function, dynamic=True)

    lora_parameters = [p for n, p in model.vl.named_parameters() if p.requires_grad]
    adapter_parameters = list(model.audio.audio_adapter.parameters())
    conformer_parameters = []
    if args.conformer_blocks and conformer_layers is not None:
        conformer_parameters = [
            p
            for layer in conformer_layers[-args.conformer_blocks :]
            for p in layer.parameters()
            if p.requires_grad
        ]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.lr},
            {"params": adapter_parameters, "lr": args.adapter_lr},
            {"params": conformer_parameters, "lr": args.conformer_lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    if not lora_parameters:
        raise RuntimeError("LoRA target regex matched no VL tensors")
    effective_batch = args.batch_size * args.gradient_accumulation
    print(
        f"trainable: lora={len(lora_parameters)} ({sum(p.numel() for p in lora_parameters):,}), "
        f"adapter={len(adapter_parameters)} ({sum(p.numel() for p in adapter_parameters):,}), "
        f"conformer={len(conformer_parameters)} ({sum(p.numel() for p in conformer_parameters):,}), "
        f"conformer_lr={args.conformer_lr:.2e}, "
        f"batch={args.batch_size}, grad_acc={args.gradient_accumulation} "
        f"(effective_batch={effective_batch})",
        flush=True,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = pack_contract(Path(args.dataset))
    wandb_run = start_wandb(args, out_dir, contract)
    step = 0
    stage_start_step = 0
    if args.resume:
        path = Path(args.resume)
        if args.resume == "auto":
            candidates = sorted(out_dir.glob("ckpt_step*.pt"))
            if not candidates:
                raise RuntimeError(f"no checkpoints in {out_dir}")
            path = candidates[-1]
        step = load_checkpoint(path, model, optimizer)
        stage_start_step = step
        print(f"resumed step {step} from {path}", flush=True)

    dataset = LFM2DataLoader(args.dataset, context_length=args.context_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
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
        "sd": trainable_state(model, conformer_blocks=args.conformer_blocks),
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
        hidden_or_logits = forward_function(batch)
        ids = model.full_ids(batch)
        supervision = value(batch, "supervision_mask")
        labels = ids[:, 1:].clone()
        labels[~supervision[:, 1:]] = -100
        flat_labels = labels.reshape(-1)
        if liger_loss is None:
            flat_logits = hidden_or_logits[:, :-1, :].reshape(
                -1, hidden_or_logits.shape[-1]
            ).float()
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        else:
            flat_hidden = hidden_or_logits[:, :-1, :].reshape(
                -1, hidden_or_logits.shape[-1]
            )
            loss = liger_loss(
                model.vl.lm_head.weight, flat_hidden, flat_labels
            )
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("backward")
        (loss / args.gradient_accumulation).backward()
        losses.append(float(loss.detach()))
        step += 1
        torch.cuda.nvtx.range_pop()
        if step % args.gradient_accumulation == 0 or step == args.steps:
            torch.cuda.nvtx.range_push("optimizer")
            optimizer.param_groups[0]["lr"] = lr_at(step - 1)
            scale = lr_at(step - 1) / max(args.lr, 1e-12)
            optimizer.param_groups[1]["lr"] = args.adapter_lr * scale
            optimizer.param_groups[2]["lr"] = args.conformer_lr * scale
            all_trainable = (
                lora_parameters + adapter_parameters + conformer_parameters
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.nvtx.range_pop()
        if step == 1 or step % 10 == 0:
            mean = sum(losses[-10:]) / min(10, len(losses))
            print(f"step {step}/{args.steps} loss {mean:.4f} ({time.time()-started:.0f}s)", flush=True)
            if wandb_run is not None:
                elapsed = time.time() - started
                wandb_run.log(
                    {
                        "train/loss": mean,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/adapter_learning_rate": optimizer.param_groups[1]["lr"],
                        "train/gradient_norm": float(grad_norm.detach()),
                        "train/steps_per_second": (step - stage_start_step) / elapsed,
                        "train/wall_s": elapsed,
                    },
                    step=step,
                )
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_checkpoint(step)

    adapter_path = out_dir / "reverse_audio_vl_adapter.safetensors"
    save_file(
        trainable_state(model, conformer_blocks=args.conformer_blocks),
        str(adapter_path),
    )
    adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    final_loss = sum(losses[-10:]) / min(10, len(losses))
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
                "liger_cross_entropy": args.liger_cross_entropy,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "conformer_blocks": args.conformer_blocks,
                "mean_last10": final_loss,
                "wall_s": round(time.time() - started, 1),
                "adapter_sha256": adapter_sha256,
                "wandb": {
                    "project": args.wandb_project,
                    "mode": args.wandb_mode,
                    "run_id": args.wandb_run_id,
                    "stage": args.wandb_stage,
                },
            },
            indent=1,
        )
    )
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "train/final_loss": final_loss,
                "train/final_wall_s": time.time() - started,
                "artifact/adapter_sha256": adapter_sha256,
            }
        )
        wandb_run.finish()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
