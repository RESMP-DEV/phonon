"""Convert a merged LFM2.5-1.2B refiner to the shipping MLX mix: 4-bit g64 everywhere, 6-bit on
the attention projections, 8-bit on the (tied) embeddings - `custom_attn6emb8` from
research/quant_v0/quant_eval.py, which is the int4 pick at 746 MB."""
import sys
from pathlib import Path

src, dst = sys.argv[1], sys.argv[2]


def pred(path: str, module, config: dict | None = None):
    if path.endswith("embed_tokens") or path.endswith("lm_head"):
        return {"bits": 8, "group_size": 64}
    if ".self_attn." in path:
        return {"bits": 6, "group_size": 64}
    return {"bits": 4, "group_size": 64}


from mlx_lm import convert  # noqa: E402

convert(src, dst, quantize=True, quant_predicate=pred, q_bits=4, q_group_size=64)
mb = sum(p.stat().st_size for p in Path(dst).rglob("*") if p.is_file()) / 1e6
print(f"MLX_OUT {dst} {mb:.1f} MB")
