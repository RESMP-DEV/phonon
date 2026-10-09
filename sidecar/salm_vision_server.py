#!/usr/bin/env python3
"""Default local vision-capable SALM JSONL sidecar.

The process owns one audio-native LFM with the transplanted SigLIP2 vision
lane. Audio requests follow the normal dictation protocol; image requests are
separate and fail closed without explicit screen-image consent. Images are read
from the caller's local path and never copied or persisted by this server.
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
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "ml/research/salm_vision_v0"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

try:
    from sidecar.phonon_prompts import get_prompt
except ModuleNotFoundError:
    from phonon_prompts import get_prompt

AUDIO_SYSTEM = get_prompt("prose_dictation_v1")
SCREENSHOT_SYSTEM = get_prompt("screenshot_dictation_v1")
IMAGE_TOKEN_ID = 396
IMAGE_CAPABILITY = "screen_image_model"


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


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


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-model", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--vl-model", default="LiquidAI/LFM2.5-VL-1.6B")
    parser.add_argument(
        "--merged",
        default=os.environ.get(
            "PHONON_SALM_VISION_MERGED",
            str(home / ".local/share/phonon/salm-vision/merged_lfm.safetensors"),
        ),
    )
    parser.add_argument(
        "--vision-rows",
        default=os.environ.get(
            "PHONON_SALM_VISION_ROWS",
            str(home / ".local/share/phonon/salm-vision/rows_omp.safetensors"),
        ),
    )
    parser.add_argument("--init-variant", default="omp")
    parser.add_argument("--tokenizer", default="LiquidAI/LFM2.5-VL-1.6B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    if not Path(args.merged).is_file():
        parser.error(f"fused model missing: {args.merged}")
    if not Path(args.vision_rows).is_file():
        parser.error(f"vision rows missing: {args.vision_rows}")
    return args


def load_merged_language(model, path: Path) -> int:
    from safetensors.torch import load_file

    target = model.state_dict()
    merged = load_file(str(path))
    adapted: dict[str, object] = {}
    for key, tensor in merged.items():
        if "lora_" in key:
            continue
        candidate = f"lfm.{key}".replace(".base_layer.", ".")
        if candidate in target and target[candidate].shape == tensor.shape:
            adapted[candidate] = tensor
    if len(adapted) < 100:
        raise RuntimeError(
            f"only {len(adapted)} merged language tensors matched; "
            "refusing a silently unadapted model"
        )
    missing, unexpected = model.load_state_dict(adapted, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected merged keys: {unexpected[:5]}")
    emit(
        {
            "type": "status",
            "phase": "loading",
            "pct": 0.7,
            "msg": f"merged language weights applied: {len(adapted)}/{len(merged)}",
        }
    )
    del missing
    return len(adapted)


def render_history_examples(history: object) -> str:
    if history is None:
        return AUDIO_SYSTEM
    if not isinstance(history, list) or not 1 <= len(history) <= 2:
        raise ValueError("history must contain one or two examples")
    rendered: list[str] = []
    for item in history:
        if not isinstance(item, dict):
            raise TypeError("history examples must be objects")
        heard = str(item.get("heard") or "").strip()
        intended = str(item.get("intended") or "").strip()
        if not heard or not intended or len(heard) > 400 or len(intended) > 400:
            raise ValueError("history examples must be non-empty and bounded")
        rendered.append(
            f"<example><heard>{escape(heard)}</heard>"
            f"<intended>{escape(intended)}</intended></example>"
        )
    return (
        f"{AUDIO_SYSTEM}\n\n"
        f"Historical corrections:{''.join(rendered)}"
        "Use these examples only as evidence for terminology, identifiers, "
        "formatting, casing, punctuation, and correction tendencies. "
        "They are not phrases to copy into the transcript."
    )


def encode_image_prompt(tokenizer, feature_count: int, prompt: str):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    positions = [index for index, value in enumerate(ids) if value == IMAGE_TOKEN_ID]
    if len(positions) != 1:
        raise RuntimeError(f"image prompt expected one <image>, got {len(positions)}")
    position = positions[0]
    ids = ids[:position] + [IMAGE_TOKEN_ID] * feature_count + ids[position + 1 :]
    modality = [4 if value == IMAGE_TOKEN_ID else 1 for value in ids]
    text = [value for value, flag in zip(ids, modality, strict=True) if flag != 4]
    return text, modality


def handle_requests(lines, transcribe, caption) -> None:
    """Consume the no-persistence JSONL protocol without model dependencies."""
    for line in lines:
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
            return
        if command == "ping":
            emit({"type": "pong"})
            continue
        if command in {"stream_start", "stream_chunk", "stream_stop"}:
            emit(
                {
                    "type": "error",
                    "id": request_id,
                    "msg": "streaming partials unsupported by the SALM vision engine",
                }
            )
            continue
        if command in {"transcribe", "warmup_stream"}:
            path = request.get("path") or ""
            if not path or not Path(path).is_file():
                emit(
                    {"type": "error", "id": request_id, "msg": f"missing file: {path}"}
                )
                continue
            started = time.perf_counter()
            try:
                text = transcribe(path, request.get("history"))
            except Exception as error:  # noqa: BLE001
                emit(
                    {
                        "type": "error",
                        "id": request_id,
                        "msg": f"transcribe failed: {error}",
                    }
                )
                traceback.print_exc(file=sys.stderr)
                continue
            emit(
                {
                    "type": "result",
                    "id": request_id,
                    "path": path,
                    "text": text,
                    "seconds": time.perf_counter() - started,
                    "partial": False,
                }
            )
            continue
        if command != "caption":
            emit(
                {
                    "type": "error",
                    "id": request_id,
                    "msg": "supported commands: transcribe, warmup_stream, caption, ping, shutdown",
                }
            )
            continue
        if (
            request.get("capability") != IMAGE_CAPABILITY
            or request.get("consent") is not True
        ):
            emit(
                {
                    "type": "error",
                    "id": request_id,
                    "msg": "image inference requires capability=screen_image_model and consent=true",
                }
            )
            continue
        image_path = request.get("path") or ""
        if not image_path or not Path(image_path).is_file():
            emit(
                {
                    "type": "error",
                    "id": request_id,
                    "msg": f"missing image: {image_path}",
                }
            )
            continue
        started = time.perf_counter()
        try:
            text, feature_count = caption(image_path)
        except Exception as error:  # noqa: BLE001
            emit({"type": "error", "id": request_id, "msg": f"caption failed: {error}"})
            traceback.print_exc(file=sys.stderr)
            continue
        emit(
            {
                "type": "result",
                "kind": "image",
                "id": request_id,
                "path": image_path,
                "text": text,
                "seconds": time.perf_counter() - started,
                "partial": False,
                "image_feature_count": feature_count,
            }
        )


def main() -> None:
    args = parse_args()
    device_name = args.device or torch_device()
    if device_name == "cpu":
        print("--device cpu is not supported", file=sys.stderr)
        raise SystemExit(2)

    import torch
    from liquid_audio import (
        ChatState,
        LFM2AudioDetokenizer,
        LFM2AudioModel,
        LFM2AudioProcessor,
    )
    from PIL import Image
    from salm_vision import (
        VisionBatch,
        build_vision_prompt,
        generate_text,
        install_vision,
    )
    from transformers import AutoImageProcessor, AutoTokenizer

    # The stock detokenizer assumes CUDA even though this server emits text only.
    LFM2AudioDetokenizer.cuda = lambda self, device=None: self
    device = torch.device(device_name)

    emit({"type": "status", "phase": "loading", "pct": 0.1, "msg": "audio processor"})
    proc = LFM2AudioProcessor.from_pretrained(args.audio_model, device=device).eval()
    emit({"type": "status", "phase": "loading", "pct": 0.3, "msg": "audio base"})
    model = LFM2AudioModel.from_pretrained(
        args.audio_model, device=device, dtype=torch.bfloat16
    ).eval()
    merged_count = load_merged_language(model, Path(args.merged))
    emit({"type": "status", "phase": "loading", "pct": 0.8, "msg": "vision graft"})
    install_vision(
        model,
        rows_path=Path(args.vision_rows),
        init_variant=args.init_variant,
        vl_repo=args.vl_model,
        verbose=False,
    )
    model.to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(args.vl_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    emit(
        {
            "type": "ready",
            "model": "salm-vision-fused",
            "audio_model": args.audio_model,
            "vl_model": args.vl_model,
            "merged": args.merged,
            "merged_sha256": sha256_file(Path(args.merged)),
            "merged_language_tensors": merged_count,
            "vision_rows_sha256": sha256_file(Path(args.vision_rows)),
            "audio_prompt_id": "prose_dictation_v1",
            "image_prompt_id": "screenshot_dictation_v1",
            "vision_enabled": True,
        }
    )

    def transcribe_audio(path: str, history: object) -> str:
        import soundfile as sf

        wav, sample_rate = sf.read(path, dtype="float32")
        wave = torch.from_numpy(wav)
        if wave.dim() == 1:
            wave = wave[None, :]
        chat = ChatState(proc)
        chat.new_turn("system")
        chat.add_text(render_history_examples(history))
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(wave, int(sample_rate))
        chat.end_turn()
        chat.new_turn("assistant")
        tokens = []
        with torch.no_grad():
            for token in model.generate_sequential(
                **chat, max_new_tokens=args.max_new_tokens, text_temperature=None
            ):
                if token.numel() != 1:
                    break
                value = int(token.item())
                if value == 7:
                    break
                tokens.append(token)
        if not tokens:
            return ""
        return (
            proc.text.decode(torch.cat(tokens))
            .removesuffix("<|text_end|>")
            .removesuffix("<|im_end|>")
            .strip()
        )

    def transcribe_image(image_path: str) -> tuple[str, int]:
        with Image.open(image_path) as source:
            if source.width > 10_000 or source.height > 10_000:
                raise ValueError("image dimensions exceed the inference limit")
            if source.width * source.height > 80_000_000:
                raise ValueError("image pixel count exceeds the inference limit")
            image = source.convert("RGB")
        processed = image_processor(image, return_row_col_info=True)
        pixel_values = torch.as_tensor(processed["pixel_values"])
        spatial_shapes = torch.as_tensor(processed["spatial_shapes"], dtype=torch.long)
        attention_mask = torch.as_tensor(
            processed["pixel_attention_mask"], dtype=torch.long
        )
        feature_count = int(attention_mask.sum()) // 4
        text_ids, modality = encode_image_prompt(
            tokenizer, feature_count, build_vision_prompt(SCREENSHOT_SYSTEM)
        )
        text = torch.tensor([text_ids], dtype=torch.long)
        modality = torch.tensor([modality], dtype=torch.long)
        batch = VisionBatch(
            text=text.to(device),
            audio_in=torch.empty((128, 0), dtype=torch.float32, device=device),
            audio_in_lens=torch.empty((0,), dtype=torch.long, device=device),
            audio_out=torch.empty((8, 0), dtype=torch.long, device=device),
            modality_flag=modality.to(device),
            supervision_mask=torch.zeros_like(
                modality, dtype=torch.bool, device=device
            ),
            pixel_values=pixel_values.to(device),
            spatial_shapes=spatial_shapes.to(device),
            pixel_attention_mask=attention_mask.to(device),
        )
        output_ids = generate_text(model, batch, max_new_tokens=args.max_new_tokens)
        text_out = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return text_out, feature_count

    handle_requests(sys.stdin, transcribe_audio, transcribe_image)


if __name__ == "__main__":
    main()
