#!/usr/bin/env python3
"""JSONL reverse Audio-to-VL SALM sidecar for whole-utterance dictation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "ml/research"
for path in (REPO_ROOT, RESEARCH_ROOT / "final_sweep", RESEARCH_ROOT / "reverse_vl_v0"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reverse_audio_vl import AUDIO_REPO, VL_REPO, ReverseAudioVL
from salm_server import handle_requests
from transcribe_reverse_audio_vl import load_adapter

from prompts import get_prompt, prompt_sha256

DEFAULT_ADAPTER = os.environ.get(
    "PHONON_REVERSE_SALM_ADAPTER",
    str(Path.home() / ".local/share/phonon/reverse-salm/lora_adapter.safetensors"),
)
DEFAULT_PROMPT_ID = os.environ.get("PHONON_REVERSE_SALM_PROMPT_ID", "prose_dictation_v1")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-model", default=AUDIO_REPO)
    parser.add_argument("--vl-model", default=VL_REPO)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--prompt-id", default=DEFAULT_PROMPT_ID)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    get_prompt(args.prompt_id)
    return args


def main() -> None:
    args = parse_args()
    device_name = args.device or torch_device()
    if device_name == "cpu":
        print("--device cpu is not supported (audio LM decode)", file=sys.stderr)
        raise SystemExit(2)
    adapter = None if args.no_adapter else Path(args.adapter)
    if adapter is not None and not adapter.is_file():
        emit(
            {
                "type": "error",
                "msg": (
                    f"adapter missing: {adapter} "
                    "(set PHONON_REVERSE_SALM_ADAPTER or pass --no-adapter)"
                ),
            }
        )
        return

    import soundfile as sf
    import torch

    model = ReverseAudioVL.from_pretrained(
        device=device_name,
        audio_repo=args.audio_model,
        vl_repo=args.vl_model,
    )
    if adapter is not None:
        load_adapter(model, adapter, rank=args.lora_rank)

    def transcribe(path: str) -> str:
        wav, sampling_rate = sf.read(path, dtype="float32")
        wave = torch.from_numpy(wav)
        if wave.dim() == 1:
            wave = wave[None, :]
        return model.generate(
            wave,
            int(sampling_rate),
            max_new_tokens=args.max_new_tokens,
            prompt_id=args.prompt_id,
        )

    emit(
        {
            "type": "ready",
            "model": "reverse-audio-vl",
            "audio_model": args.audio_model,
            "vl_model": args.vl_model,
            "adapter": None if adapter is None else str(adapter),
            "adapter_sha256": None if adapter is None else sha256_file(adapter),
            "prompt_id": args.prompt_id,
            "prompt_sha256": prompt_sha256(args.prompt_id),
        }
    )
    handle_requests(sys.stdin, transcribe, model_id="reverse-audio-vl")


if __name__ == "__main__":
    main()
