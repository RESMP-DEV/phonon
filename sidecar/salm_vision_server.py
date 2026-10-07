#!/usr/bin/env python3
"""Private vision-on-audio SALM prototype JSONL sidecar.

This is an explicit opt-in research server. It does not alter the default
dictation engine and does not persist images.

Protocol additions over the SALM sidecar:
  {"cmd":"caption","id":"...","path":"/path/to/image.png","prompt":"optional"}
returns:
  {"type":"result","id":"...","path":"...","text":"...","seconds":0.1}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "ml/research/salm_vision_v0"
for path in (RESEARCH_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from sidecar.phonon_prompts import get_prompt
except ModuleNotFoundError:
    from phonon_prompts import get_prompt

from salm_vision import VisionBatch, generate_text, install_vision


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


DEFAULT_ADAPTER = os.environ.get(
    "PHONON_SALM_VISION_ADAPTER",
    str(Path.home() / ".local/share/phonon/salm-vision/cpt_adapter.safetensors"),
)
DEFAULT_ROWS = os.environ.get(
    "PHONON_SALM_VISION_ROWS",
    str(Path.home() / ".local/share/phonon/salm-vision/rows_omp.safetensors"),
)
DEFAULT_TOKENIZER = os.environ.get(
    "PHONON_SALM_VISION_TOKENIZER", "LiquidAI/LFM2.5-VL-1.6B"
)
DEFAULT_SYSTEM = get_prompt("prose_dictation_v1")
IMAGE_TOKEN_ID = 396
IMAGE_START_ID = 498
IMAGE_END_ID = 499


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_image_prompt(torch, tokenizer, system: str, feature_count: int):
    template = (
        "<|startoftext|>"
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|image_start|><|image_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    ids = tokenizer.encode(template, add_special_tokens=False)
    positions = [index for index, value in enumerate(ids) if value == IMAGE_TOKEN_ID]
    if len(positions) != 1:
        raise RuntimeError(f"image prompt expected one <image>, got {len(positions)}")
    position = positions[0]
    ids = ids[:position] + [IMAGE_TOKEN_ID] * feature_count + ids[position + 1:]
    modality = [4 if value == IMAGE_TOKEN_ID else 1 for value in ids]
    text = [value for value, flag in zip(ids, modality) if flag != 4]
    return (
        torch.tensor([text], dtype=torch.long),
        torch.tensor([modality], dtype=torch.long),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-model", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--vl-model", default="LiquidAI/LFM2.5-VL-1.6B")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--vision-rows", default=DEFAULT_ROWS)
    parser.add_argument("--init-variant", default="omp")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    if not Path(args.adapter).is_file():
        parser.error(f"adapter missing: {args.adapter}")
    if not Path(args.vision_rows).is_file():
        parser.error(f"vision rows missing: {args.vision_rows}")
    return args


def main() -> None:
    args = parse_args()
    device_name = args.device or torch_device()
    if device_name == "cpu":
        print("--device cpu is not supported", file=sys.stderr)
        raise SystemExit(2)

    import torch
    from liquid_audio import LFM2AudioModel

    device = torch.device(device_name)
    from peft import LoraConfig, inject_adapter_in_model
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoImageProcessor, AutoTokenizer

    emit({"type": "status", "phase": "loading", "pct": 0.1, "msg": "audio base"})
    model = LFM2AudioModel.from_pretrained(
        args.audio_model, device=device, dtype=torch.bfloat16
    ).eval()
    emit({"type": "status", "phase": "loading", "pct": 0.4, "msg": "vision graft"})
    install_vision(
        model,
        rows_path=Path(args.vision_rows),
        init_variant=args.init_variant,
        vl_repo=args.vl_model,
        verbose=False,
    )
    if args.adapter:
        emit({"type": "status", "phase": "loading", "pct": 0.7, "msg": "adapter"})
        model = inject_adapter_in_model(
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=2 * args.lora_rank,
                lora_dropout=0.0,
                target_modules=(
                    r"^lfm\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)"
                    r"|conv\.(in_proj|out_proj)|feed_forward\.(w1|w2|w3))$"
                ),
            ),
            model,
        )
        state = load_file(args.adapter)
        missing, _unexpected = model.load_state_dict(state, strict=False)
        missing_lora = [key for key in missing if "lora" in key]
        if missing_lora:
            raise RuntimeError(f"adapter keys missing: {missing_lora[:5]}")
        model.to(device).eval()

    image_processor = AutoImageProcessor.from_pretrained(args.vl_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    emit(
        {
            "type": "ready",
            "model": "salm-vision",
            "audio_model": args.audio_model,
            "vl_model": args.vl_model,
            "adapter": args.adapter,
            "adapter_sha256": sha256_file(Path(args.adapter)),
            "vision_rows_sha256": sha256_file(Path(args.vision_rows)),
            "prompt_id": "prose_dictation_v1",
            "prompt_sha256": __import__("hashlib").sha256(
                DEFAULT_SYSTEM.encode()
            ).hexdigest(),
            "vision_enabled": True,
        }
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            emit({"type": "error", "msg": f"bad json: {error}"})
            continue
        command = request.get("cmd")
        request_id = request.get("id")
        if command == "shutdown":
            emit({"type": "status", "phase": "shutdown", "pct": 1.0, "msg": "bye"})
            break
        if command == "ping":
            emit({"type": "pong"})
            continue
        if command != "caption":
            emit(
                {
                    "type": "error",
                    "id": request_id,
                    "msg": "supported commands: caption, ping, shutdown",
                }
            )
            continue
        image_path = request.get("path") or ""
        if not image_path or not Path(image_path).is_file():
            emit({"type": "error", "id": request_id, "msg": f"missing image: {image_path}"})
            continue
        started = time.perf_counter()
        try:
            image = Image.open(image_path).convert("RGB")
            processed = image_processor(image, return_row_col_info=True)
            pixel_values = torch.as_tensor(processed["pixel_values"])
            spatial_shapes = torch.as_tensor(
                processed["spatial_shapes"], dtype=torch.long
            )
            attention_mask = torch.as_tensor(
                processed["pixel_attention_mask"], dtype=torch.long
            )
            feature_count = int(attention_mask.sum()) // 4
            text, modality = encode_image_prompt(
                tokenizer, request.get("prompt") or DEFAULT_SYSTEM, feature_count
            )
            batch = VisionBatch(
                text=text.to(device),
                audio_in=torch.empty((128, 0), dtype=torch.float32, device=device),
                audio_in_lens=torch.empty((0,), dtype=torch.long, device=device),
                audio_out=torch.empty((8, 0), dtype=torch.long, device=device),
                modality_flag=modality.to(device),
                supervision_mask=torch.zeros_like(modality, dtype=torch.bool, device=device),
                pixel_values=pixel_values.to(device),
                spatial_shapes=spatial_shapes.to(device),
                pixel_attention_mask=attention_mask.to(device),
            )
            output_ids = generate_text(
                model, batch, max_new_tokens=args.max_new_tokens
            )
            text_out = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            emit(
                {
                    "type": "result",
                    "id": request_id,
                    "path": image_path,
                    "text": text_out,
                    "seconds": time.perf_counter() - started,
                    "partial": False,
                    "image_feature_count": feature_count,
                }
            )
        except Exception as error:  # noqa: BLE001
            emit({"type": "error", "id": request_id, "msg": str(error)})
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
