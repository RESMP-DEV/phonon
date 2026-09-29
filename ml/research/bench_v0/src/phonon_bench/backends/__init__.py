"""Backend registry."""
from __future__ import annotations

from .base import Backend

NAMES = ("hf", "mlx", "llama", "vllm")


def make(name: str, args) -> Backend:
    if name == "hf":
        from .hf import HFBackend
        return HFBackend(args.base, args.adapter, dtype=args.dtype, attn=args.attn)
    if name == "mlx":
        from .mlx import MLXBackend
        return MLXBackend(args.model_dir or args.base)
    if name == "llama":
        from .llama import LlamaBackend
        return LlamaBackend(args.base_url, concurrency=args.max_batch)
    if name == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(args.base, args.adapter, max_model_len=args.max_model_len,
                           gpu_frac=args.gpu_frac, max_lora_rank=args.max_lora_rank)
    raise SystemExit(f"unknown backend {name!r}; one of {NAMES}")
