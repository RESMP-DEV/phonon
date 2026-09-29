"""Canary-Qwen-2.5B encoder loader for L4 feature-KD.

The Canary-Qwen encoder is a standard nemo.collections.asr.modules.ConformerEncoder
(d_model=1024, subsampling_factor=8, rel_pos, 32 layers, 811M params) -- identical
architecture/dim/frame-rate to the Parakeet encoder. We instantiate it from the
HF config.json and load the `perception.encoder.*` weights from model.safetensors
(speechlm2 is not in this NeMo build, but the encoder itself is plain NeMo).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from hydra.utils import instantiate


def _find_canary_files() -> tuple[Path, Path]:
    base = Path(
        "/home/user/.cache/huggingface/hub/models--nvidia--canary-qwen-2.5b/snapshots"
    )
    snap = next(iter(base.glob("*")))
    return snap / "config.json", snap / "model.safetensors"


def load_teacher_encoder(device: str = "cuda") -> tuple[Any, list[str], list[str]]:
    """Return (encoder, missing_keys, unexpected_keys) loaded + on `device`."""
    cfg_path, st_path = _find_canary_files()
    cfg = json.loads(cfg_path.read_text())
    enc_cfg = OmegaConf.create(cfg["perception"]["encoder"])
    encoder = instantiate(enc_cfg)
    from safetensors.torch import load_file

    sd = load_file(str(st_path))
    enc_sd = {
        k[len("perception.encoder.") :]: v
        for k, v in sd.items()
        if k.startswith("perception.encoder.")
    }
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, missing, unexpected
