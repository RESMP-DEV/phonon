"""Switch-Conformer MoE: drop-in sparse MoE for NeMo's ConformerFeedForward.

Replaces the second FFN (feed_forward2) in each ConformerLayer with a top-k
mixture-of-experts. Built for from-scratch ASR training on a single Blackwell.

Design choices:
  - top-2 routing, NO token dropping (ASR frames must not be silently dropped;
    dropping frames corrupts the alignment the TDT loss relies on).
  - load-balancing aux loss (Switch Transformer) + router z-loss (ST-MoE) for
    from-scratch routing stability. Router math is always BF16/FP32.
  - per-module precision policy: "bf16" (default, correct grads) or "fp8"
    (transient fp8 matmul via torch._scaled_mm). FP8 is a speed optimization
    and is gated behind an explicit flag; BF16 is the safe default for the
    first from-scratch run.
  - aux losses are stored per-module (last_lb, last_z) and collected by the
    trainer via collect_moe_aux_loss(), added to the main loss before backward
    so gradients reach the router weights.
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from nemo.collections.asr.parts.submodules.conformer_modules import (
        ConformerFeedForward,
    )
except Exception:  # pragma: no cover - allow import without nemo for unit-ish tests
    ConformerFeedForward = None

_E4M3_MAX = 240.0  # max representable value for float8_e4m3fn


def fp8_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Linear (x @ weight.T [+ bias]) with a transient FP8 e4m3 matmul.

    Weights are kept in BF16; we quantize both x and weight to fp8 e4m3 with a
    per-tensor scale, run torch._scaled_mm (fast_accum), and return BF16.
    Gradients flow back through the cast (PyTorch fp8 autograd) to the BF16
    master weights. Falls back to BF16 if _scaled_mm is unavailable.
    """
    if not hasattr(torch, "_scaled_mm") or x.is_cpu:
        return F.linear(x, weight, bias)
    # x: (N, In) ; weight: (Out, In)
    sx = (x.detach().abs().amax().clamp(min=1e-4) / _E4M3_MAX).to(torch.float32)
    sw = (weight.detach().abs().amax().clamp(min=1e-4) / _E4M3_MAX).to(torch.float32)
    xf = (x.float() / sx).to(torch.float8_e4m3fn)
    wf = (weight.float() / sw).to(torch.float8_e4m3fn)
    try:
        out = torch._scaled_mm(
            xf, wf.t(), sx, sw, out_dtype=torch.bfloat16, use_fast_accum=True
        )
    except Exception:
        return F.linear(x, weight, bias)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


class _ExpertFFN(nn.Module):
    """A single expert: Linear(d->d_ff) -> Swish -> Dropout -> Linear(d_ff->d).

    Identical structure to ConformerFeedForward so weight init / capacity is
    directly comparable to the dense baseline.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        use_bias: bool = True,
        precision: str = "bf16",
    ):
        super().__init__()
        self.precision = precision
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout)

    def _lin(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if self.precision == "fp8" and x.is_cuda:
            return fp8_linear(x, layer.weight, layer.bias)
        return F.linear(x, layer.weight, layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._lin(self.linear1, x)
        x = F.silu(x)  # Swish
        x = self.dropout(x)
        x = self._lin(self.linear2, x)
        return x


class SwitchMoEFeedForward(nn.Module):
    """Drop-in replacement for ConformerFeedForward (second FFN of a Conformer block).

    Args:
        d_model, d_ff, dropout, use_bias: same as ConformerFeedForward.
        num_experts: E (number of experts).
        top_k: experts activated per token (2 recommended for stability).
        precision: "bf16" or "fp8" for the expert FFN matmuls.
        router_fp32: keep router weights/logits in fp32 (recommended True).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
        use_bias: bool = True,
        precision: str = "bf16",
        router_fp32: bool = True,
    ):
        super().__init__()
        assert top_k <= num_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.precision = precision
        self.router_fp32 = router_fp32
        self._is_switch_moe = True  # tag for collect_moe_aux_loss

        router_dtype = torch.float32 if router_fp32 else torch.bfloat16
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.router.weight.data = self.router.weight.data.to(router_dtype)

        self.experts = nn.ModuleList(
            [
                _ExpertFFN(
                    d_model, d_ff, dropout, use_bias=use_bias, precision=precision
                )
                for _ in range(num_experts)
            ]
        )
        # last forward's aux terms (graph-attached); trainer collects them
        self.last_lb: torch.Tensor | None = None
        self.last_z: torch.Tensor | None = None

    @torch.no_grad()
    def active_param_count(self) -> int:
        """Params computed per token (one expert path + router), for reporting."""
        # router + k experts
        router_p = sum(p.numel() for p in self.router.parameters())
        one_expert = sum(p.numel() for p in self.experts[0].parameters())
        return router_p + self.top_k * one_expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        B, T, D = x.shape
        N = B * T
        flat = x.reshape(N, D)

        # Router (fp32 for numerical stability, then cast probs to compute dtype)
        rt = flat.to(self.router.weight.dtype)
        logits = self.router(rt)  # (N, E)
        probs = F.softmax(logits, dim=-1)  # (N, E) in router dtype

        # top-k routing
        topk_g, topk_idx = probs.topk(self.top_k, dim=-1)  # (N, k)
        topk_g = topk_g / topk_g.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        gate_dtype = x.dtype
        topk_g = topk_g.to(gate_dtype)
        topk_idx = topk_idx.to(torch.long)

        # Aux losses (computed in fp32 for stability)
        with torch.amp.autocast("cuda", enabled=False) if flat.is_cuda else _no_op():
            probs_f32 = probs.float()
            # load balance: f_i = frac of tokens routed to expert i; P_i = mean prob
            one_hot = F.one_hot(topk_idx, self.num_experts).float()  # (N, k, E)
            routed = one_hot.any(dim=1).float()  # (N, E)
            f = routed.mean(dim=0)  # (E,)
            P = probs_f32.mean(dim=0)  # (E,)
            lb = self.num_experts * (f * P).sum()
            # router z-loss: mean(logsumexp(logits)^2) (ST-MoE form)
            log_z = torch.logsumexp(logits.float(), dim=-1)  # (N,)
            z = (log_z * log_z).mean()
        self.last_lb = lb
        self.last_z = z

        # Dispatch (E expert FFN calls; gather/scatter). Correct, not maximally fast.
        out = torch.zeros_like(flat)
        expert_arange = torch.arange(self.num_experts, device=flat.device)
        for e in range(self.num_experts):
            # tokens that route to expert e in any of their k slots
            mask = topk_idx == e  # (N, k) bool
            tok_idx = mask.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
            if tok_idx.numel() == 0:
                continue
            # gate weight for expert e per chosen token (pick the slot that is e)
            slot_mask = mask[tok_idx].float()  # (n, k)
            gate = (topk_g[tok_idx] * slot_mask).sum(dim=-1)  # (n,)
            xe = flat[tok_idx]
            ye = self.experts[e](xe)
            out.index_add_(0, tok_idx, ye * gate.unsqueeze(-1))

        return out.reshape(B, T, D)


class _no_op:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def collect_moe_aux_loss(
    model: nn.Module, lb_coef: float = 0.01, z_coef: float = 0.01
) -> torch.Tensor:
    """Sum load-balance + z-loss across all SwitchMoEFeedForward modules.

    Returns a scalar tensor (graph-attached) to add to the main loss before
    backward so router weights receive gradients.
    """
    total = None
    device = next(model.parameters(), torch.tensor(0.0)).device
    acc = torch.zeros((), device=device, dtype=torch.float32)
    found = 0
    for m in model.modules():
        if getattr(m, "_is_switch_moe", False) and m.last_lb is not None:
            acc = acc + lb_coef * m.last_lb.float() + z_coef * m.last_z.float()
            found += 1
    if found == 0:
        return torch.zeros((), device=device, dtype=torch.float32)
    return acc


def reset_moe_aux_loss(model: nn.Module) -> None:
    for m in model.modules():
        if getattr(m, "_is_switch_moe", False):
            m.last_lb = None
            m.last_z = None


def convert_encoder_to_moe(
    encoder: nn.Module,
    num_experts: int,
    top_k: int = 2,
    precision_layer_map: dict[int, str] | None = None,
    default_precision: str = "bf16",
    dropout: float = 0.1,
    use_bias: bool = True,
    copy_ff1_init: bool = True,
) -> nn.Module:
    """Swap encoder.layers[i].feed_forward2 with SwitchMoEFeedForward in-place.

    Args:
        encoder: a NeMo ConformerEncoder with .layers (nn.ModuleList).
        num_experts, top_k: MoE config.
        precision_layer_map: {layer_idx: "bf16"|"fp8"} per-layer precision. Layers
            not in the map use default_precision.
        default_precision: precision for unmapped layers.
        copy_ff1_init: if True, expert linear weights are initialized from
            feed_forward1's weights (so the MoE starts as ~the dense model with
            expert multiplicity, stabilizing early training).
    """
    precision_layer_map = precision_layer_map or {}
    layers = encoder.layers
    for i, layer in enumerate(layers):
        d_model = layer.feed_forward1.d_model
        d_ff = layer.feed_forward1.d_ff
        prec = precision_layer_map.get(i, default_precision)
        moe = SwitchMoEFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            use_bias=use_bias,
            precision=prec,
        )
        if copy_ff1_init:
            ff1 = layer.feed_forward1
            with torch.no_grad():
                for ex in moe.experts:
                    ex.linear1.weight.copy_(ff1.linear1.weight)
                    ex.linear2.weight.copy_(ff1.linear2.weight)
                    if ff1.linear1.bias is not None:
                        ex.linear1.bias.copy_(ff1.linear1.bias)
                        ex.linear2.bias.copy_(ff1.linear2.bias)
        layer.feed_forward2 = moe
    return encoder


def moe_param_counts(model: nn.Module) -> dict[str, int]:
    """Report total, active (per-token), and dense params for a model with MoE layers."""
    total = 0
    active = 0
    moe_modules = []
    for p in model.parameters():
        total += p.numel()
    for m in model.modules():
        if getattr(m, "_is_switch_moe", False):
            moe_modules.append(m)
            # subtract expert params from active (they're sparse); add k-active
            expert_total = sum(
                sum(p.numel() for p in ex.parameters()) for ex in m.experts
            )
            one_expert = sum(p.numel() for p in m.experts[0].parameters())
            router = sum(p.numel() for p in m.router.parameters())
            # active counts one expert path per token: router + k experts
            active_contribution = router + m.top_k * one_expert
            # we'll reconcile below
    # active = total - (all expert params) + (k * one_expert per moe + router already in total)
    all_expert = sum(
        sum(p.numel() for p in m.experts.parameters()) for m in moe_modules
    )
    one_expert_per = [
        sum(p.numel() for p in m.experts[0].parameters()) for m in moe_modules
    ]
    # total includes all experts + routers. active = total - all_expert + sum(k*one_expert)
    active = (
        total
        - all_expert
        + sum(m.top_k * oe for m, oe in zip(moe_modules, one_expert_per))
    )
    return {"total": total, "active": active, "num_moe_layers": len(moe_modules)}
