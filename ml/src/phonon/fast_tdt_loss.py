from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import torch
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import load_inline

_RAW_MODULE: Any | None = None
_LOGITS_DIRECT_FALLBACK_COUNT = 0


def _require_power_of_two(name: str, value: int) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def _load_benchmark_module() -> Any:
    benchmark_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "benchmark_tdt_raw_cuda_alpha_beta.py"
    )
    spec = importlib.util.spec_from_file_location("phonon_tdt_raw_benchmark", benchmark_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load raw TDT benchmark source from {benchmark_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_raw_tdt_module() -> Any:
    global _RAW_MODULE
    if _RAW_MODULE is None:
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if not cuda_home:
            for candidate in ("/usr/local/cuda-13.2", "/usr/local/cuda-13", "/usr/local/cuda"):
                if (Path(candidate) / "include" / "cuda_runtime.h").exists():
                    cuda_home = candidate
                    break
        if cuda_home:
            os.environ.setdefault("CUDA_HOME", cuda_home)
            os.environ.setdefault("CUDA_PATH", cuda_home)
            cpp_extension.CUDA_HOME = cuda_home
        benchmark = _load_benchmark_module()
        dense_threads = int(os.environ.get("TDT_DENSE_THREADS", "1024"))
        compact_threads = int(os.environ.get("TDT_COMPACT_THREADS", str(dense_threads)))
        _require_power_of_two("TDT_DENSE_THREADS", dense_threads)
        _require_power_of_two("TDT_COMPACT_THREADS", compact_threads)
        combined_alpha_beta = int(os.environ.get("TDT_COMBINED_ALPHA_BETA", "0"))
        logits_direct = int(os.environ.get("TDT_LOGITS_DIRECT", "0"))
        _RAW_MODULE = load_inline(
            name=(
                f"phonon_tdt_alpha_beta_raw_cuda_dt{dense_threads}"
                f"_ct{compact_threads}_ab{combined_alpha_beta}_ld{logits_direct}"
            ),
            cpp_sources=benchmark.CPP_SRC,
            cuda_sources=benchmark.CUDA_SRC,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                f"-DTDT_DENSE_THREADS={dense_threads}",
                f"-DTDT_COMPACT_THREADS={compact_threads}",
                f"-DTDT_COMBINED_ALPHA_BETA={combined_alpha_beta}",
            ],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _RAW_MODULE


class RawTDTLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        token_logits: torch.Tensor,
        duration_logits: torch.Tensor,
        labels: torch.Tensor,
        durations_t: torch.Tensor,
        blank: int,
        sigma: float,
        module: Any,
    ) -> torch.Tensor:
        token_logp = torch.log_softmax(token_logits.float(), dim=-1).contiguous()
        duration_logp = torch.log_softmax(duration_logits.float(), dim=-1).contiguous()
        _alpha, _beta, ll_forward, _ll_backward, token_grad, duration_grad = (
            module.tdt_alpha_beta_grad_cuda(
                token_logp, duration_logp, labels, durations_t, blank, sigma
            )
        )
        ctx.save_for_backward(token_grad, duration_grad)
        return -ll_forward.sum()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        token_grad, duration_grad = ctx.saved_tensors
        scale = grad_output.to(token_grad.dtype)
        return token_grad * scale, duration_grad * scale, None, None, None, None, None


class LogitsDirectTDTLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        token_logits: torch.Tensor,
        duration_logits: torch.Tensor,
        labels: torch.Tensor,
        durations_t: torch.Tensor,
        blank: int,
        sigma: float,
        module: Any,
    ) -> torch.Tensor:
        _alpha, _beta, ll_forward, _ll_backward, token_grad, duration_grad = (
            module.tdt_logits_direct_grad_cuda(
                token_logits.float().contiguous(),
                duration_logits.float().contiguous(),
                labels,
                durations_t,
                blank,
                sigma,
            )
        )
        if _should_fallback_from_logits_direct(ll_forward, token_grad, duration_grad):
            global _LOGITS_DIRECT_FALLBACK_COUNT
            _LOGITS_DIRECT_FALLBACK_COUNT += 1
            token_logp = torch.log_softmax(token_logits.float(), dim=-1).contiguous()
            duration_logp = torch.log_softmax(duration_logits.float(), dim=-1).contiguous()
            _alpha, _beta, ll_forward, _ll_backward, token_grad, duration_grad = (
                module.tdt_alpha_beta_grad_cuda(
                    token_logp, duration_logp, labels, durations_t, blank, sigma
                )
            )
        ctx.save_for_backward(token_grad, duration_grad)
        return -ll_forward.sum()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        token_grad, duration_grad = ctx.saved_tensors
        scale = grad_output.to(token_grad.dtype)
        return token_grad * scale, duration_grad * scale, None, None, None, None, None


def _should_fallback_from_logits_direct(
    ll_forward: torch.Tensor, token_grad: torch.Tensor, duration_grad: torch.Tensor
) -> bool:
    """Return true when experimental logits-direct gradients look unsafe.

    This path is intentionally opt-in because the checks synchronize. It is for
    debugging the faster logits-direct kernel on real batches without allowing a
    single bad sample to poison model gradients.
    """

    if os.environ.get("TDT_LOGITS_DIRECT_GUARD", "0") != "1":
        return False
    max_abs = float(os.environ.get("TDT_LOGITS_DIRECT_MAX_ABS_GRAD", "16.0"))
    for tensor in (ll_forward, token_grad, duration_grad):
        detached = tensor.detach()
        if not bool(torch.isfinite(detached).all().item()):
            return True
        if tensor is not ll_forward and bool((detached.abs() > max_abs).any().item()):
            return True
    return False


def logits_direct_fallback_count() -> int:
    return _LOGITS_DIRECT_FALLBACK_COUNT


def reset_logits_direct_fallback_count() -> None:
    global _LOGITS_DIRECT_FALLBACK_COUNT
    _LOGITS_DIRECT_FALLBACK_COUNT = 0


class RawTDTVariableLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        token_logits: torch.Tensor,
        duration_logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        durations_t: torch.Tensor,
        blank: int,
        sigma: float,
        module: Any,
    ) -> torch.Tensor:
        token_logp = torch.log_softmax(token_logits.float(), dim=-1).contiguous()
        duration_logp = torch.log_softmax(duration_logits.float(), dim=-1).contiguous()
        _alpha, _beta, ll_forward, _ll_backward, token_grad, duration_grad = (
            module.tdt_alpha_beta_grad_var_cuda(
                token_logp,
                duration_logp,
                labels.int().contiguous(),
                input_lengths.int().contiguous(),
                target_lengths.int().contiguous(),
                durations_t,
                blank,
                sigma,
            )
        )
        ctx.save_for_backward(token_grad, duration_grad)
        return -ll_forward

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        token_grad, duration_grad = ctx.saved_tensors
        scale = grad_output.to(token_grad.dtype).reshape(-1, 1, 1, 1)
        return token_grad * scale, duration_grad * scale, None, None, None, None, None, None, None


def reduce_losses(
    losses: torch.Tensor, target_lengths: torch.Tensor, reduction: str | None
) -> torch.Tensor:
    if reduction == "mean_batch":
        return losses.mean()
    if reduction == "mean":
        return torch.div(losses, target_lengths.float()).mean()
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean_volume":
        return losses.sum() / target_lengths.float().sum()
    return losses


def reduce_loss_chunks(
    losses: list[torch.Tensor], target_lengths: torch.Tensor, reduction: str | None
) -> torch.Tensor:
    if not losses:
        raise RuntimeError("cannot reduce an empty TDT loss list")
    if os.environ.get("TDT_NO_CAT_REDUCE", "0") != "1":
        return reduce_losses(torch.cat(losses, dim=0), target_lengths, reduction)
    if reduction in {"sum", "mean_volume"}:
        total = losses[0].sum()
        for chunk in losses[1:]:
            total = total + chunk.sum()
        if reduction == "sum":
            return total
        return total / target_lengths.float().sum()
    if reduction == "mean_batch":
        total = losses[0].sum()
        count = losses[0].numel()
        for chunk in losses[1:]:
            total = total + chunk.sum()
            count += chunk.numel()
        return total / count
    return reduce_losses(torch.cat(losses, dim=0), target_lengths, reduction)


class VariableLengthRawTDTLoss(torch.nn.Module):
    """Deterministic TDT loss for NeMo fused Parakeet training.

    This intentionally does not implement NeMo's stochastic RNNT/TDT omega
    mixing. Use it as a speed experiment behind an explicit training flag.
    """

    def __init__(
        self,
        raw_module: Any | None = None,
        *,
        blank: int,
        durations: list[int],
        sigma: float,
        reduction: str | None,
        variable_batch: bool = False,
        hybrid_pad_ratio: float | None = None,
        hybrid_max_group: int | None = None,
    ) -> None:
        super().__init__()
        self.raw_module = raw_module if raw_module is not None else load_raw_tdt_module()
        self.blank = int(blank)
        self.durations = [int(x) for x in durations]
        self.sigma = float(sigma)
        self.reduction = reduction
        self.variable_batch = bool(variable_batch)
        if hybrid_pad_ratio is None:
            hybrid_pad_ratio_env = os.environ.get("TDT_HYBRID_PAD_RATIO")
            hybrid_pad_ratio = float(hybrid_pad_ratio_env) if hybrid_pad_ratio_env else None
        if hybrid_max_group is None:
            hybrid_max_group_env = os.environ.get("TDT_HYBRID_MAX_GROUP")
            hybrid_max_group = int(hybrid_max_group_env) if hybrid_max_group_env else 8
        self.hybrid_pad_ratio = (
            float(hybrid_pad_ratio) if hybrid_pad_ratio is not None and hybrid_pad_ratio > 0 else None
        )
        self.hybrid_max_group = int(hybrid_max_group)
        logits_direct_requested = os.environ.get("TDT_LOGITS_DIRECT", "0") == "1"
        allow_unstable_logits_direct = os.environ.get("TDT_ALLOW_UNSTABLE_LOGITS_DIRECT", "0") == "1"
        if logits_direct_requested and not allow_unstable_logits_direct:
            raise RuntimeError(
                "TDT_LOGITS_DIRECT=1 is disabled for training because real-batch "
                "diagnostics found non-finite decoder gradients. Use the default "
                "TDT_LOGITS_DIRECT=0 raw-CUDA path, or set "
                "TDT_ALLOW_UNSTABLE_LOGITS_DIRECT=1 only for isolated kernel debugging."
            )
        self.logits_direct = logits_direct_requested
        self._durations_t: torch.Tensor | None = None

    def _durations_tensor(self, device: torch.device) -> torch.Tensor:
        if self._durations_t is None or self._durations_t.device != device:
            self._durations_t = torch.tensor(self.durations, device=device, dtype=torch.int32)
        return self._durations_t

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        token_dim = log_probs.shape[-1] - len(self.durations)
        token_logits, duration_logits = torch.split(
            log_probs, [token_dim, len(self.durations)], dim=-1
        )
        durations_t = self._durations_tensor(log_probs.device)
        if not self.variable_batch:
            losses = self._sample_or_hybrid_losses(
                token_logits,
                duration_logits,
                targets,
                input_lengths,
                target_lengths,
                durations_t,
            )
            return reduce_loss_chunks(losses, target_lengths, self.reduction)

        max_frames = int(input_lengths.max().item())
        max_labels = int(target_lengths.max().item())
        losses_t = RawTDTVariableLossFunction.apply(
            token_logits[:, :max_frames, : max_labels + 1, :].contiguous(),
            duration_logits[:, :max_frames, : max_labels + 1, :].contiguous(),
            targets[:, :max_labels].int().contiguous(),
            input_lengths.int().contiguous(),
            target_lengths.int().contiguous(),
            durations_t,
            self.blank,
            self.sigma,
            self.raw_module,
        )
        return reduce_losses(losses_t, target_lengths, self.reduction)

    def _sample_loss(
        self,
        token_logits: torch.Tensor,
        duration_logits: torch.Tensor,
        targets: torch.Tensor,
        durations_t: torch.Tensor,
        sample_idx: int,
        frames: int,
        labels_n: int,
    ) -> torch.Tensor:
        sample_token = token_logits[sample_idx : sample_idx + 1, :frames, : labels_n + 1, :]
        sample_duration = duration_logits[sample_idx : sample_idx + 1, :frames, : labels_n + 1, :]
        if os.environ.get("TDT_SAMPLE_PRECONTIGUOUS", "0") == "1":
            sample_token = sample_token.contiguous()
            sample_duration = sample_duration.contiguous()
        sample_targets = targets[sample_idx : sample_idx + 1, :labels_n].int().contiguous()
        loss_fn = LogitsDirectTDTLossFunction if self.logits_direct else RawTDTLossFunction
        loss = loss_fn.apply(
            sample_token,
            sample_duration,
            sample_targets,
            durations_t,
            self.blank,
            self.sigma,
            self.raw_module,
        )
        return loss.reshape(1)

    def _hybrid_groups(
        self, input_lengths: torch.Tensor, target_lengths: torch.Tensor
    ) -> list[list[int]]:
        lengths = [
            (idx, int(input_lengths[idx].item()), int(target_lengths[idx].item()) + 1)
            for idx in range(input_lengths.shape[0])
        ]
        if self.hybrid_pad_ratio is None or len(lengths) < 2:
            return [[idx] for idx, _frames, _labels in lengths]

        groups: list[list[int]] = []
        pending: list[tuple[int, int, int]] = []
        pending_volume = 0
        for item in sorted(lengths, key=lambda row: (row[1], row[2])):
            next_pending = [*pending, item]
            max_t = max(row[1] for row in next_pending)
            max_u = max(row[2] for row in next_pending)
            next_volume = pending_volume + item[1] * item[2]
            padded_volume = max_t * max_u * len(next_pending)
            too_padded = padded_volume > self.hybrid_pad_ratio * next_volume
            too_large = len(next_pending) > self.hybrid_max_group
            if pending and (too_padded or too_large):
                groups.append([row[0] for row in pending])
                pending = [item]
                pending_volume = item[1] * item[2]
            else:
                pending = next_pending
                pending_volume = next_volume
        if pending:
            groups.append([row[0] for row in pending])
        return groups

    def _sample_or_hybrid_losses(
        self,
        token_logits: torch.Tensor,
        duration_logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        durations_t: torch.Tensor,
    ) -> list[torch.Tensor]:
        losses: list[torch.Tensor] = []
        for group in self._hybrid_groups(input_lengths, target_lengths):
            if len(group) == 1:
                i = group[0]
                frames = int(input_lengths[i].item())
                labels_n = int(target_lengths[i].item())
                losses.append(
                    self._sample_loss(
                        token_logits,
                        duration_logits,
                        targets,
                        durations_t,
                        i,
                        frames,
                        labels_n,
                    )
                )
                continue

            group_idx = torch.tensor(group, device=token_logits.device, dtype=torch.long)
            group_input_lengths = input_lengths.index_select(0, group_idx).int().contiguous()
            group_target_lengths = target_lengths.index_select(0, group_idx).int().contiguous()
            max_frames = int(group_input_lengths.max().item())
            max_labels = int(group_target_lengths.max().item())
            group_token = token_logits.index_select(0, group_idx)[
                :, :max_frames, : max_labels + 1, :
            ].contiguous()
            group_duration = duration_logits.index_select(0, group_idx)[
                :, :max_frames, : max_labels + 1, :
            ].contiguous()
            group_targets = targets.index_select(0, group_idx)[:, :max_labels].int().contiguous()
            group_losses = RawTDTVariableLossFunction.apply(
                group_token,
                group_duration,
                group_targets,
                group_input_lengths,
                group_target_lengths,
                durations_t,
                self.blank,
                self.sigma,
                self.raw_module,
            )
            losses.append(group_losses.reshape(-1))
        return losses

    def reduce(
        self,
        losses: torch.Tensor | list[torch.Tensor],
        target_lengths: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(losses, list):
            if isinstance(target_lengths, list):
                target_lengths = torch.cat(target_lengths, dim=0)
            return reduce_loss_chunks(losses, target_lengths, self.reduction)
        if isinstance(target_lengths, list):
            target_lengths = torch.cat(target_lengths, dim=0)
        return reduce_losses(losses, target_lengths, self.reduction)


def install_fast_tdt_loss(model: Any) -> dict[str, Any]:
    joint = getattr(model, "joint", None)
    original_loss = getattr(joint, "_loss", None)
    inner_loss = getattr(original_loss, "_loss", original_loss)
    if joint is None or original_loss is None or inner_loss is None:
        raise RuntimeError("Model does not expose a NeMo-style joint._loss")

    durations = [int(x) for x in getattr(inner_loss, "durations")]
    sigma = float(getattr(inner_loss, "sigma"))
    blank = int(getattr(original_loss, "_blank", getattr(inner_loss, "blank", -1)))
    reduction = getattr(original_loss, "reduction", getattr(inner_loss, "reduction", None))
    variable_batch = os.environ.get("TDT_VARIABLE_BATCH", "0") == "1"
    custom_loss = VariableLengthRawTDTLoss(
        blank=blank,
        durations=durations,
        sigma=sigma,
        reduction=reduction,
        variable_batch=variable_batch,
    )
    joint.set_fuse_loss_wer(True, loss=custom_loss, metric=model.wer)
    return {
        "enabled": True,
        "kind": "raw_cuda_deterministic_tdt",
        "blank": blank,
        "durations": durations,
        "sigma": sigma,
        "reduction": reduction,
        "variable_batch": custom_loss.variable_batch,
        "hybrid_pad_ratio": custom_loss.hybrid_pad_ratio,
        "hybrid_max_group": custom_loss.hybrid_max_group,
        "combined_alpha_beta": os.environ.get("TDT_COMBINED_ALPHA_BETA", "0"),
        "compact_threads": os.environ.get("TDT_COMPACT_THREADS", os.environ.get("TDT_DENSE_THREADS", "1024")),
        "logits_direct": os.environ.get("TDT_LOGITS_DIRECT", "0"),
        "no_cat_reduce": os.environ.get("TDT_NO_CAT_REDUCE", "0"),
        "caveat": "does not implement NeMo omega stochastic RNNT/TDT mixing",
    }
