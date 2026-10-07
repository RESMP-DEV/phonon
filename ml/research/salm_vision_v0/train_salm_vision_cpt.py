#!/usr/bin/env python3
"""Continuous pre-training of the vision-transplanted SALM on B550.

Three lanes interleaved by ratio (audio dictation / text corrector / vision),
because pack composition is load-bearing (measured 2026-09-30: pure-audio
plateaus 0.03 fair-WER worse than mixed). LoRA into the LM as in the SFT
trainer; the SigLIP2 tower stays frozen; the projector is optionally
trainable at a low LR (--unfreeze-projector, stage B).

--probe N runs the init race instead of training: mean text CE over N vision
rows with the LoRA untouched, for each rows_*.safetensors you pass. Zero
training, directly measures the 2506.06607 claim on this model pair.

Long-running by design: --ckpt-every + --resume auto keep it alive across
restarts; judge only by slice metrics, never this train loss.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from liquid_audio import LFM2AudioModel
from liquid_audio.data.dataloader import LFM2DataLoader, lfm2_collator
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file
from salm_vision import VisionDataLoader, install_vision, vision_collator
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

TARGETS = (
    r"^lfm\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)"
    r"|conv\.(in_proj|out_proj)|feed_forward\.(w1|w2|w3))$"
)


def parse_ratio(s: str) -> tuple[int, int, int]:
    a, c, v = (int(x) for x in s.split(":"))
    return a, c, v


def build_lanes(args, device):
    lanes = []
    if args.audio_pack:
        ld = DataLoader(LFM2DataLoader(args.audio_pack, context_length=args.audio_context),
                        batch_size=1, shuffle=True, collate_fn=lfm2_collator)
        lanes.append(("audio", ld, iter(ld)))
    if args.corrector_pack:
        ld = DataLoader(LFM2DataLoader(args.corrector_pack, context_length=args.text_context),
                        batch_size=1, shuffle=True, collate_fn=lfm2_collator)
        lanes.append(("corrector", ld, iter(ld)))
    if args.vision_pack:
        ip = AutoImageProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B")
        ld = DataLoader(VisionDataLoader(args.vision_pack, ip, context_length=args.vision_context),
                        batch_size=1, shuffle=True, collate_fn=vision_collator)
        lanes.append(("vision", ld, iter(ld)))
    return lanes


def probe(args, model, device) -> None:
    """Init race: text CE per variant over the first N vision rows, plus one
    audio row to prove the audio path is untouched by the splice."""
    ip = AutoImageProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B")
    vds = VisionDataLoader(args.vision_pack, ip, context_length=args.vision_context)
    aud = next(iter(DataLoader(LFM2DataLoader(args.audio_pack, context_length=args.audio_context),
                               batch_size=1, shuffle=False, collate_fn=lfm2_collator)))
    variants = args.probe_variants.split(",")
    for v in variants:
        rows_path = Path(args.transplant_dir) / f"rows_{v}.safetensors"
        install_vision(model, rows_path=rows_path, init_variant=v, verbose=False)
        model.eval()
        with torch.no_grad():
            ces = []
            for i in range(args.probe):
                batch = vds[i % len(vds)]
                batch_t = vision_collator([batch]).to(device)
                text_logits, _, text_labels, _ = model.logits(batch_t)
                ce = torch.nn.functional.cross_entropy(text_logits.float(), text_labels)
                ces.append(float(ce))
            a = aud.to(device)
            t_logits, _, t_labels, _ = model.logits(a)
            audio_ce = float(torch.nn.functional.cross_entropy(t_logits.float(), t_labels))
        print(f"PROBE variant={v}: vision CE mean {sum(ces)/len(ces):.4f} "
              f"(first {ces[0]:.4f} last {ces[-1]:.4f}) | audio CE {audio_ce:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-pack", default="aqua-sft-dataset-v3")
    ap.add_argument("--corrector-pack", default="aqua-correction-v4")
    ap.add_argument("--vision-pack", default="aqua-vision-dataset-v1")
    ap.add_argument("--ratio", default="2:1:2", help="audio:corrector:vision step mix")
    ap.add_argument("--transplant-dir", default="vision-transplant")
    ap.add_argument("--init-variant", default="omp", choices=("omp", "copy", "reserved"))
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="salm-vision-cpt-v1")
    ap.add_argument("--ckpt-every", type=int, default=400)
    ap.add_argument("--keep-ckpts", type=int, default=3)
    ap.add_argument("--resume", default=None, help="ckpt path or 'auto'")
    ap.add_argument(
        "--init-adapter",
        default=None,
        help=(
            "start task tuning from an existing adapter instead of a fresh LoRA; "
            "use this to continue post-CPT ASR/vision SFT from a prior stage"
        ),
    )
    ap.add_argument("--audio-context", type=int, default=768)
    ap.add_argument("--text-context", type=int, default=768)
    ap.add_argument("--vision-context", type=int, default=1024)
    ap.add_argument("--unfreeze-projector", action="store_true")
    ap.add_argument("--projector-lr", type=float, default=1e-5)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--probe", type=int, default=0, help="N>0: run the init race and exit")
    ap.add_argument("--probe-variants", default="reserved,copy,omp")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = LFM2AudioModel.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B", device=device, dtype=torch.bfloat16)

    if args.probe:
        probe(args, model, device)
        return

    install_vision(model, rows_path=Path(args.transplant_dir) / f"rows_{args.init_variant}.safetensors",
                   init_variant=args.init_variant, unfreeze_projector=args.unfreeze_projector)

    n_frozen = 0
    for n, p in model.named_parameters():
        if not n.startswith("lfm."):
            p.requires_grad_(False)
            n_frozen += 1
    if args.unfreeze_projector:
        for p in model.multi_modal_projector.parameters():
            p.requires_grad_(True)

    model = inject_adapter_in_model(
        LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=args.lora_dropout,
                   target_modules=TARGETS), model)
    if args.init_adapter:
        from safetensors.torch import load_file

        init_path = Path(args.init_adapter)
        if not init_path.is_file():
            ap.error(f"--init-adapter missing: {init_path}")
        init_state = load_file(str(init_path))
        proj_state = {k[5:]: v for k, v in init_state.items() if k.startswith("proj.")}
        if proj_state:
            if not args.unfreeze_projector:
                ap.error("--init-adapter carries projector weights; pass --unfreeze-projector")
            model.multi_modal_projector.load_state_dict(proj_state)
            print(f"projector restored from {init_path} ({len(proj_state)} tensors)", flush=True)
        lora_state = {k: v for k, v in init_state.items() if "lora_" in k}
        missing, unexpected = model.load_state_dict(lora_state, strict=False)
        missing_lora = [k for k in missing if "lora_" in k]
        if missing_lora or unexpected:
            raise RuntimeError(
                f"--init-adapter mismatch: missing_lora={missing_lora[:5]} unexpected={unexpected[:5]}"
            )
        print(f"init-adapter loaded: {init_path} ({len(lora_state)} LoRA tensors)", flush=True)
    model.to(device)
    if args.grad_checkpoint:
        model.lfm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"frozen tensors: {n_frozen}, trainable: {len(train_params)} "
          f"({sum(p.numel() for p in train_params):,} params)", flush=True)

    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n],
               "lr": args.lr}]
    if args.unfreeze_projector:
        groups.append({"params": list(model.multi_modal_projector.parameters()), "lr": args.projector_lr})
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
                            fused=(device.type == "cuda"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def trainable_sd() -> dict:
        sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
              if "lora_" in k}
        if args.unfreeze_projector:
            sd.update({f"proj.{k}": v.detach().cpu().contiguous()
                       for k, v in model.multi_modal_projector.state_dict().items()})
        return sd

    def save_ckpt(step_: int) -> None:
        payload = {"step": step_, "sd": trainable_sd(), "opt": opt.state_dict(),
                   "torch_rng": torch.get_rng_state(), "losses": losses[-50:],
                   "lane_losses": lane_losses}
        if device.type == "cuda":
            payload["cuda_rng"] = torch.cuda.get_rng_state(device)
        tmp = out_dir / ".ckpt_tmp.pt"
        torch.save(payload, tmp)
        tmp.replace(out_dir / f"ckpt_step{step_:05d}.pt")
        if args.keep_ckpts > 0:
            for old in sorted(out_dir.glob("ckpt_step*.pt"))[: -args.keep_ckpts]:
                old.unlink()

    step = 0
    losses: list[float] = []
    lane_losses: dict[str, list[float]] = {}
    if args.resume:
        rpath = Path(args.resume)
        if args.resume == "auto":
            cks = sorted(out_dir.glob("ckpt_step*.pt"))
            if not cks:
                ap.error(f"--resume auto: no ckpts in {out_dir}")
            rpath = cks[-1]
        ck = torch.load(rpath, map_location=device, weights_only=True)
        model.load_state_dict(ck["sd"], strict=False)
        if args.unfreeze_projector:
            proj = {k[5:]: v for k, v in ck["sd"].items() if k.startswith("proj.")}
            if proj:
                model.multi_modal_projector.load_state_dict(proj)
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        losses = list(ck.get("losses") or [])
        lane_losses = dict(ck.get("lane_losses") or {})
        torch.set_rng_state(ck["torch_rng"].cpu())
        if device.type == "cuda" and "cuda_rng" in ck:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device)
        print(f"resumed step {step} from {rpath}", flush=True)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    ra, rc, rv = parse_ratio(args.ratio)
    pattern = ["audio"] * ra + ["corrector"] * rc + ["vision"] * rv
    by_lane = {name: (ld, it) for name, ld, it in build_lanes(args, device)}

    t0 = time.time()
    pi = 0
    while step < args.steps:
        lane = pattern[pi % len(pattern)]
        pi += 1
        if lane not in by_lane:
            continue
        loader, it = by_lane[lane]
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            by_lane[lane] = (loader, it)
            batch = next(it)
        batch = batch.to(device)
        out = model(batch)
        out.loss.backward()
        opt.param_groups[0]["lr"] = lr_at(step)  # projector group keeps its fixed low lr
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss))
        lane_losses.setdefault(lane, []).append(float(out.loss))
        step += 1
        if step % 10 == 0 or step == 1:
            tag = " ".join(f"{k}={sum(v[-10:]) / min(10, len(v)):.3f}" for k, v in lane_losses.items())
            print(f"step {step}/{args.steps} loss {sum(losses[-10:]) / min(10, len(losses)):.4f} "
                  f"[{tag}] ({time.time() - t0:.0f}s)", flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_ckpt(step)

    save_file(trainable_sd(), str(out_dir / "cpt_adapter.safetensors"))
    (out_dir / "train_meta.json").write_text(json.dumps({
        "steps": args.steps, "ratio": args.ratio, "lr": args.lr, "rank": args.rank,
        "init_variant": args.init_variant, "unfreeze_projector": args.unfreeze_projector,
        "packs": {"audio": args.audio_pack, "corrector": args.corrector_pack,
                  "vision": args.vision_pack},
        "final_loss": losses[-1], "mean_last10": sum(losses[-10:]) / min(10, len(losses)),
        "lane_mean_last50": {k: sum(v[-50:]) / min(50, len(v)) for k, v in lane_losses.items()},
        "wall_s": round(time.time() - t0, 1),
    }, indent=1))
    print(f"DONE mean_last10={sum(losses[-10:]) / min(10, len(losses)):.4f}", flush=True)


if __name__ == "__main__":
    main()
