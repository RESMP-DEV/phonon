#!/usr/bin/env python3
"""JSONL SALM sidecar: LFM2.5-Audio-1.5B + LoRA, single-stage dictation.

Speaks the same stdio protocol as asr_server.py so the Rust side can swap
engines without protocol changes: `transcribe`/`warmup_stream`/`ping`/
`shutdown` in, `status`/`ready`/`result`/`error` out. Streaming partials
(`stream_*`) are Parakeet-specific and report an error here; the audio
language model transcribes whole utterances.

Unlike the Parakeet sidecar this is a full dictation engine: the LoRA
adapter was trained on dictation audio -> intended text, so the output is
already corrected (technical terms, identifiers, punctuation) and the
dictionary-retrieval + polish stages downstream are unnecessary when this
engine is selected.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


SYSTEM = (
    "You are a personal dictation engine. Transcribe the user's audio into the "
    "text they intended, including technical terms, identifiers and punctuation. "
    "Text only."
)

LORA_TARGETS = (
    r"^lfm\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)"
    r"|conv\.(in_proj|out_proj)|feed_forward\.(w1|w2|w3))$"
)

DEFAULT_ADAPTER = os.environ.get(
    "PHONON_SALM_ADAPTER",
    str(Path.home() / ".local/share/phonon/salm/lora_adapter.safetensors"),
)


def load(model_id: str, adapter: str | None, rank: int, device_name: str):
    import torch

    device = torch.device(device_name)
    from liquid_audio import (
        ChatState,
        LFM2AudioDetokenizer,
        LFM2AudioModel,
        LFM2AudioProcessor,
    )

    # processor.py hardcodes .cuda() on the audio-out detokenizer; this
    # sidecar never generates audio, so neuter it for MPS/CPU hosts.
    LFM2AudioDetokenizer.cuda = lambda self, device=None: self

    emit({"type": "status", "phase": "loading", "pct": 0.1,
          "msg": f"processor ({model_id})"})
    proc = LFM2AudioProcessor.from_pretrained(model_id, device=device).eval()

    emit({"type": "status", "phase": "loading", "pct": 0.3,
          "msg": "model weights"})
    model = LFM2AudioModel.from_pretrained(
        model_id, dtype=torch.bfloat16, device=device
    ).eval()

    if adapter:
        emit({"type": "status", "phase": "loading", "pct": 0.7,
              "msg": f"adapter {Path(adapter).name}"})
        from peft import LoraConfig, inject_adapter_in_model
        from safetensors.torch import load_file

        model = inject_adapter_in_model(
            LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                       target_modules=LORA_TARGETS),
            model,
        )
        missing, unexpected = model.load_state_dict(
            load_file(adapter), strict=False
        )
        missing_lora = [k for k in missing if "lora" in k]
        if missing_lora:
            raise RuntimeError(f"adapter keys missing: {missing_lora[:5]}")
        model.to(device).eval()
    return model, proc, ChatState


def handle_requests(
    lines,
    transcribe: callable,
    *,
    model_id: str,
    stopped: callable = lambda: False,
) -> None:
    """Consume JSONL requests without knowing how transcription is implemented."""
    for line in lines:
        if stopped():
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"type": "error", "msg": f"bad json: {e}"})
            continue
        cmd = req.get("cmd")
        rid = req.get("id")
        if cmd == "shutdown":
            emit({"type": "status", "phase": "shutdown", "pct": 1.0, "msg": "bye"})
            break
        if cmd == "ping":
            emit({"type": "pong"})
            continue
        if cmd in ("stream_start", "stream_chunk", "stream_stop"):
            emit({"type": "error", "id": rid,
                  "msg": "streaming partials unsupported by the SALM engine"})
            continue
        if cmd in ("transcribe", "warmup_stream"):
            path = req.get("path") or ""
            if not path or not Path(path).is_file():
                emit({"type": "error", "id": rid, "msg": f"missing file: {path}"})
                continue
            t1 = time.perf_counter()
            try:
                text = transcribe(path)
                emit({"type": "result", "id": rid, "path": path, "text": text,
                      "seconds": time.perf_counter() - t1,
                      "partial": False})
            except Exception as e:  # noqa: BLE001
                emit({"type": "error", "id": rid, "msg": f"transcribe failed: {e}"})
            continue
        emit({"type": "error", "msg": f"unknown cmd: {cmd}"})


def main() -> None:
    model_id = "LiquidAI/LFM2.5-Audio-1.5B"
    adapter = DEFAULT_ADAPTER
    rank = 16
    max_new_tokens = 256
    if torch_device() == "cpu":
        print("--device cpu is not supported (audio LM decode)", file=sys.stderr)
        sys.exit(2)
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--model" and i + 1 < len(args):
            model_id = args[i + 1]
        elif a == "--adapter" and i + 1 < len(args):
            adapter = args[i + 1]
        elif a == "--no-adapter":
            adapter = None
        elif a == "--lora-rank" and i + 1 < len(args):
            rank = int(args[i + 1])
        elif a == "--max-new-tokens" and i + 1 < len(args):
            max_new_tokens = int(args[i + 1])

    if adapter and not Path(adapter).is_file():
        emit({"type": "error",
              "msg": f"adapter missing: {adapter} "
                     f"(set PHONON_SALM_ADAPTER or pass --no-adapter)"})
        return

    try:
        model, proc, ChatState = load(model_id, adapter, rank, torch_device())
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "msg": f"load failed: {e}"})
        traceback.print_exc(file=sys.stderr)
        return
    emit({"type": "ready", "model": model_id})

    import soundfile as sf
    import torch

    def transcribe_file(path: str) -> str:
        wav, sr = sf.read(path, dtype="float32")
        wave = torch.from_numpy(wav)
        if wave.dim() == 1:
            wave = wave[None, :]
        chat = ChatState(proc)
        chat.new_turn("system")
        chat.add_text(SYSTEM)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(wave, int(sr))
        chat.end_turn()
        chat.new_turn("assistant")
        toks = []
        with torch.no_grad():
            for t in model.generate_sequential(
                **chat, max_new_tokens=max_new_tokens, text_temperature=None
            ):
                if t.numel() != 1:
                    break  # <|audio_start|> drift: text is done
                v = int(t.item())
                if v == 7:  # <|im_end|>
                    break
                toks.append(t)
        if not toks:
            return ""
        return (
            proc.text.decode(torch.cat(toks))
            .removesuffix("<|text_end|>")
            .removesuffix("<|im_end|>")
            .strip()
        )

    handle_requests(sys.stdin, transcribe_file, model_id=model_id)

def torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()
