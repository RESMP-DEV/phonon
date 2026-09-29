"""LoRA SFT of a dictation corrector with peft + transformers + trl SFTTrainer."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_ROOT,
    DEV_JSONL,
    QWEN_ID,
    TRAIN_JSONL,
    ensure_data_dirs,
    heartbeat,
    hf_token,
    read_jsonl,
    record_failure,
    set_hf_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=QWEN_ID)
    parser.add_argument("--name", default="", help="Adapter directory name under adapters/.")
    parser.add_argument("--train", "--train-file", type=Path, default=TRAIN_JSONL)
    parser.add_argument("--dev", "--dev-file", type=Path, default=DEV_JSONL)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout. 0.0 removes an fp32 dropout kernel per adapter per step.",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto from model size.")
    parser.add_argument("--grad-accum", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop training after this many seconds (0 = no cap).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help=(
            "Continue training from an existing PEFT adapter directory: load it with "
            "is_trainable=True instead of creating a fresh LoRA. Everything else "
            "(data, schedule, saving) is unchanged."
        ),
    )
    parser.add_argument(
        "--keep-weight",
        type=float,
        default=0.0,
        help=(
            "research/restraint_v0: per-token loss weight on target tokens copied from the "
            "input (tokens inside an unchanged span of the input/target alignment); edited "
            "tokens keep weight 1.0. 0 = off (default). Needs --fused-ce off."
        ),
    )
    perf = parser.add_argument_group("sm_120 perf (see research/kernels_sm120/results.md)")
    perf.add_argument(
        "--grad-checkpointing",
        choices=("on", "off"),
        default="off",
        help=(
            "Activation checkpointing. Default off: 1.30x (Qwen3-0.6B) / 1.40x "
            "(LFM2.5-1.2B) faster, peak 22-43 GB on a 96 GB card."
        ),
    )
    perf.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="HF optim name. Default fused AdamW (same math, one kernel).",
    )
    perf.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help="SFTTrainer dataloader_num_workers.",
    )
    perf.add_argument(
        "--attn-impl",
        default="sdpa",
        help="transformers attn_implementation (sdpa, eager, flash_attention_2).",
    )
    perf.add_argument(
        "--sdpa-backend",
        choices=("auto", "flash", "cudnn", "efficient", "math"),
        default="auto",
        help=(
            "Pin the SDPA backend. Leave auto for padded batches: flash cannot take the "
            "padding mask and silently falls back to the math path (measured 1.33x slower, "
            "77 GB peak). Pin flash only with --packing on."
        ),
    )
    perf.add_argument(
        "--tf32",
        choices=("on", "off"),
        default="on",
        help="Allow TF32 for the fp32 ops (LoRA glue, norms); bf16 matmuls are unaffected.",
    )
    perf.add_argument(
        "--bf16-reduced-reduction",
        choices=("on", "off"),
        default="on",
        help="cuBLAS bf16 split-k reduction in bf16. on is torch's default and faster.",
    )
    perf.add_argument(
        "--group-by-length",
        choices=("on", "off"),
        default="on",
        help=(
            "Length-grouped batch sampler (HF LengthGroupedSampler). Default on: the "
            "2k mid slice is 56%% useful tokens with random batches at bs16 and 98%% "
            "grouped. Same math, different batch composition."
        ),
    )
    perf.add_argument(
        "--fused-ce",
        choices=("on", "off"),
        default="off",
        help=(
            "Fused linear cross-entropy (liger LigerFusedLinearCrossEntropyLoss, else a "
            "plain torch chunked CE): lm_head + CE together on the non-ignored tokens "
            "only. Loss-neutral but MEASURED SLOWER (0.81x LFM2.5-1.2B, 0.79x Qwen3-1.7B) "
            "and +6 to +16 GB, because TRL's own loss_type='chunked_nll' default already "
            "skips the lm_head matmul on ignored tokens. See research/perf_v1/"
            "results_fusedce.md. Default off."
        ),
    )
    perf.add_argument(
        "--fused-ce-chunk",
        type=int,
        default=2048,
        help="Token chunk for the plain-torch fallback CE (ignored when liger is used).",
    )
    perf.add_argument(
        "--legacy",
        action="store_true",
        help="Restore the pre-2026-09-18 defaults (no length grouping, lora dropout 0.05).",
    )
    perf.add_argument(
        "--packing",
        choices=("on", "off"),
        default="off",
        help="TRL example packing. Removes pad tokens; changes which tokens share a batch.",
    )
    perf.add_argument(
        "--padding-free",
        choices=("on", "off"),
        default="off",
        help="TRL padding-free batching (needs flash_attention_2); same math, no pad tokens.",
    )
    perf.add_argument(
        "--compile",
        choices=("off", "default", "max-autotune"),
        default="off",
        help="torch.compile the model before the LoRA wrap.",
    )
    perf.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="Profile this many steps (after --profile-wait) with torch.profiler, then dump.",
    )
    perf.add_argument(
        "--profile-wait",
        type=int,
        default=5,
        help="Steps to skip before profiling.",
    )
    perf.add_argument(
        "--profile-out",
        type=Path,
        default=None,
        help="Where to write the kernel table json (default DATA_ROOT/logs/profile_<name>.json).",
    )
    parsed = parser.parse_args()
    if parsed.legacy:
        parsed.group_by_length = "off"
        parsed.fused_ce = "off"
        if parsed.lora_dropout == 0.0:
            parsed.lora_dropout = 0.05
    return parsed


_SDPA_BACKENDS = {
    "flash": "FLASH_ATTENTION",
    "cudnn": "CUDNN_ATTENTION",
    "efficient": "EFFICIENT_ATTENTION",
    "math": "MATH",
}


def apply_backend_flags(args: argparse.Namespace) -> None:
    """Global torch knobs measured in research/kernels_sm120/results.md."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = args.tf32 == "on"
    torch.backends.cudnn.allow_tf32 = args.tf32 == "on"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        args.bf16_reduced_reduction == "on"
    )
    if args.sdpa_backend != "auto":
        from torch.nn.attention import SDPBackend

        backend = getattr(SDPBackend, _SDPA_BACKENDS[args.sdpa_backend])
        try:
            torch.backends.cuda.enable_flash_sdp(backend == SDPBackend.FLASH_ATTENTION)
            torch.backends.cuda.enable_cudnn_sdp(backend == SDPBackend.CUDNN_ATTENTION)
            torch.backends.cuda.enable_mem_efficient_sdp(
                backend == SDPBackend.EFFICIENT_ATTENTION
            )
            torch.backends.cuda.enable_math_sdp(
                backend in (SDPBackend.MATH, SDPBackend.FLASH_ATTENTION)
            )
        except Exception as exc:  # pragma: no cover - torch version drift
            print(f"sdpa backend pin failed ({exc}); leaving auto", flush=True)
    print(
        f"perf: gc={args.grad_checkpointing} optim={args.optim} "
        f"workers={args.dataloader_workers} attn={args.attn_impl} "
        f"sdpa={args.sdpa_backend} tf32={args.tf32} compile={args.compile} "
        f"group_by_length={args.group_by_length} lora_dropout={args.lora_dropout} "
        f"fused_ce={args.fused_ce}",
        flush=True,
    )


def profile_callback(args: argparse.Namespace, out_path: Path):
    """TrainerCallback that profiles a window of steps and writes a kernel table."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import TrainerCallback

    class ProfileCallback(TrainerCallback):
        def __init__(self) -> None:
            self.prof = None
            self.started = 0
            self.done = False
            self.t0 = 0.0

        def on_step_begin(self, targs, state, control, **kwargs):
            if self.done or state.global_step < args.profile_wait:
                return control
            if self.prof is None:
                torch.cuda.synchronize()
                self.prof = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                )
                self.prof.__enter__()
                self.t0 = time.perf_counter()
            return control

        def on_step_end(self, targs, state, control, **kwargs):
            if self.prof is None or self.done:
                return control
            self.started += 1
            if self.started >= args.profile_steps:
                torch.cuda.synchronize()
                wall = time.perf_counter() - self.t0
                self.prof.__exit__(None, None, None)
                self.dump(wall)
                self.done = True
            return control

        def dump(self, wall: float) -> None:
            from torch.autograd import DeviceType

            rows = []
            for ev in self.prof.key_averages():
                dev = getattr(ev, "self_device_time_total", 0) or 0
                if dev > 0 and getattr(ev, "device_type", None) == DeviceType.CUDA:
                    rows.append({"kernel": ev.key, "ms": dev / args.profile_steps / 1000.0,
                                 "calls": ev.count / args.profile_steps})
            rows.sort(key=lambda r: -r["ms"])
            total = sum(r["ms"] for r in rows)
            payload = {
                "model": args.model,
                "profile_steps": args.profile_steps,
                "wall_ms_per_step": wall / args.profile_steps * 1000.0,
                "cuda_ms_per_step": total,
                "kernels": rows[:40],
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote profile {out_path} "
                  f"(wall {payload['wall_ms_per_step']:.1f} ms/step, "
                  f"cuda {total:.1f} ms/step)", flush=True)

    return ProfileCallback()


def adapter_name(model_id: str, explicit: str) -> str:
    if explicit:
        return explicit
    tail = model_id.split("/")[-1].lower().replace("_", "-")
    return tail


def auto_batch(model_id: str) -> tuple[int, int]:
    lower = model_id.lower()
    if "0.6b" in lower or "0.6-b" in lower:
        return 8, 2
    if "1.7b" in lower:
        return 4, 4
    # 96 GB Blackwell: a 0.8B model peaks under 5 GB at batch 16, so do not accumulate.
    return 16, 1


def apply_chat_template(processor, messages: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    for extra in ({"enable_thinking": False}, {}):
        try:
            text = processor.apply_chat_template(messages, **kwargs, **extra)
            if isinstance(text, list):
                text = text[0]
            return str(text)
        except TypeError:
            continue
        except Exception:
            mm = []
            for msg in messages:
                content = msg["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                mm.append({"role": msg["role"], "content": content})
            try:
                text = processor.apply_chat_template(mm, tokenize=False, add_generation_prompt=False)
                if isinstance(text, list):
                    text = text[0]
                return str(text)
            except Exception:
                pass
    parts = []
    for msg in messages:
        parts.append(f"{msg['role']}: {msg['content']}")
    return "\n".join(parts)


def ensure_training_chat_template(processor) -> None:
    """TRL SFT requires `{% generation %}` markers; some bases ship unpatchable templates."""
    import re

    holders = [processor]
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and tok is not processor:
        holders.append(tok)
    template = None
    for obj in holders:
        template = getattr(obj, "chat_template", None)
        if template:
            break
    if not template or re.search(r"\{%-?\s*generation\s*-?%\}", str(template)):
        return
    simple = (
        "{%- for message in messages -%}"
        "{%- if message['role'] == 'system' -%}"
        "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{%- elif message['role'] == 'user' -%}"
        "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{%- elif message['role'] == 'assistant' -%}"
        "{{- '<|im_start|>assistant\\n' }}{% generation %}{{- message['content'] + '<|im_end|>\\n' }}{% endgeneration %}"
        "{%- endif -%}"
        "{%- endfor -%}"
    )
    for obj in holders:
        if hasattr(obj, "chat_template"):
            obj.chat_template = simple
    print("patched chat_template with generation markers for TRL SFT", flush=True)


def load_processor(model_id: str, token: str | None):
    from transformers import AutoProcessor, AutoTokenizer

    last_err = None
    for loader in (AutoProcessor, AutoTokenizer):
        try:
            proc = loader.from_pretrained(model_id, token=token, trust_remote_code=True)
            if hasattr(proc, "padding_side"):
                proc.padding_side = "right"
            tok = getattr(proc, "tokenizer", None)
            if tok is not None and hasattr(tok, "padding_side"):
                tok.padding_side = "right"
            ensure_training_chat_template(proc)
            return proc
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"failed to load tokenizer/processor for {model_id}: {last_err}")


def load_model(model_id: str, token: str | None, attn_impl: str = "sdpa"):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "token": token,
        "trust_remote_code": True,
        "attn_implementation": attn_impl,
    }
    errors = []
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs), "causal"
    except Exception as exc:
        errors.append(f"AutoModelForCausalLM: {type(exc).__name__}: {exc}")
    try:
        from transformers import AutoModelForMultimodalLM

        model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
        return model, "multimodal"
    except Exception as exc:
        errors.append(f"AutoModelForMultimodalLM: {type(exc).__name__}: {exc}")
    raise RuntimeError("model load failed: " + " | ".join(errors))


def render_dataset(processor, rows: list[dict], limit: int):
    from datasets import Dataset

    records = []
    for row in rows:
        if limit and len(records) >= limit:
            break
        records.append(
            {
                "messages": row["messages"],
                "text": apply_chat_template(processor, row["messages"]),
            }
        )
    return Dataset.from_list(records)


def lora_config(r: int, dropout: float = 0.05):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=r,
        lora_alpha=r * 2,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )



# --------------------------------------------------------------------------- #
# Fused linear cross-entropy
# --------------------------------------------------------------------------- #
# The default TRL/HF loss materialises [batch, seq, vocab] logits and upcasts them
# to fp32 (plus a second fp32 copy for the softmax gradient and a third for the
# entropy/accuracy metrics). On LFM2.5-1.2B (vocab 65,536) that is 41.6 GB at
# batch 16 and an OOM at batch 48. The fused path keeps hidden states only: it
# drops every position whose label is -100 (assistant-only mask included), then
# runs lm_head + CE together in token chunks so no full logit tensor ever exists.


_LIGER_STATE = {"cls": None, "tried": False}


def _liger_loss_cls():
    """LigerFusedLinearCrossEntropyLoss if importable, else None (cached)."""
    if not _LIGER_STATE["tried"]:
        _LIGER_STATE["tried"] = True
        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )

            _LIGER_STATE["cls"] = LigerFusedLinearCrossEntropyLoss
        except Exception as exc:  # pragma: no cover - env drift
            print(f"fused-ce: liger unavailable ({type(exc).__name__}: {exc}); "
                  "using the plain-torch chunked CE", flush=True)
    return _LIGER_STATE["cls"]


def torch_chunked_ce(weight, hidden, target, reduction: str, chunk: int = 2048):
    """Plain-torch fallback: lm_head + CE in token chunks, logits recomputed in backward."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def _one(h, t):
        return F.cross_entropy(F.linear(h, weight).float(), t, reduction="sum")

    total = None
    for i in range(0, hidden.shape[0], chunk):
        h, t = hidden[i : i + chunk], target[i : i + chunk]
        part = checkpoint(_one, h, t, use_reentrant=False) if h.requires_grad else _one(h, t)
        total = part if total is None else total + part
    if total is None:
        total = hidden.new_zeros((), dtype=torch.float32)
    if reduction == "mean":
        total = total / max(1, target.numel())
    return total


def resolve_head_and_decoder(model):
    """(causal_lm, decoder, lm_head) for a PEFT / compiled / plain causal LM."""
    causal = model
    for _ in range(4):
        inner = getattr(causal, "_orig_mod", None)
        if inner is not None:
            causal = inner
            continue
        if hasattr(causal, "get_base_model"):
            base = causal.get_base_model()
            if base is not causal:
                causal = base
                continue
        break
    lm_head = None
    if hasattr(causal, "get_output_embeddings"):
        lm_head = causal.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(causal, "lm_head", None)
    decoder = None
    for getter in ("get_decoder",):
        fn = getattr(causal, getter, None)
        if callable(fn):
            try:
                decoder = fn()
            except Exception:
                decoder = None
        if decoder is not None:
            break
    if decoder is None:
        decoder = getattr(causal, "model", None)
    if decoder is None or lm_head is None:
        raise RuntimeError(
            f"--fused-ce cannot find lm_head/decoder on {type(causal).__name__}; use --fused-ce off"
        )
    return causal, decoder, lm_head


def fused_ce_trainer_cls(base_cls, chunk: int):
    """SFTTrainer subclass whose compute_loss never materialises full logits."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    class FusedCESFTTrainer(base_cls):
        _fce_parts = None
        _fce_loss_fn = None
        _fce_announced = False

        def _fce_setup(self, model):
            if self._fce_parts is None:
                causal, decoder, lm_head = resolve_head_and_decoder(model)
                if isinstance(lm_head, BaseTunerLayer):
                    raise RuntimeError(
                        "--fused-ce needs an untrained lm_head, but it is wrapped by a PEFT "
                        "adapter. Drop lm_head from target_modules or pass --fused-ce off."
                    )
                cfg = getattr(causal, "config", None)
                softcap = getattr(cfg, "final_logit_softcapping", None)
                if softcap is None and hasattr(cfg, "text_config"):
                    softcap = getattr(cfg.text_config, "final_logit_softcapping", None)
                self._fce_parts = (causal, decoder, lm_head, softcap)
                self._fce_sig = set(inspect.signature(decoder.forward).parameters)
                if not self._fce_announced:
                    kernel = "liger" if _liger_loss_cls() is not None else f"torch/chunk{chunk}"
                    print(
                        f"fused-ce: {kernel}, head={type(lm_head).__name__} "
                        f"[{tuple(lm_head.weight.shape)}, trainable={lm_head.weight.requires_grad}], "
                        f"decoder={type(decoder).__name__}, softcap={softcap}",
                        flush=True,
                    )
                    FusedCESFTTrainer._fce_announced = True
            return self._fce_parts

        def _fce_loss(self, hidden, target, weight, softcap, num_items_in_batch):
            reduction = "sum" if num_items_in_batch is not None else "mean"
            cls = _liger_loss_cls()
            if cls is not None:
                key = (reduction, softcap)
                if self._fce_loss_fn is None or self._fce_loss_fn[0] != key:
                    self._fce_loss_fn = (
                        key,
                        cls(ignore_index=-100, reduction=reduction, softcap=softcap),
                    )
                loss = self._fce_loss_fn[1](weight, hidden, target)
                if isinstance(loss, tuple):
                    loss = loss[0]
            else:
                if softcap:
                    raise RuntimeError("fused-ce fallback does not implement logit softcapping")
                loss = torch_chunked_ce(weight, hidden, target, reduction, chunk)
            if reduction == "sum":
                loss = loss / num_items_in_batch
            return loss

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            causal, decoder, lm_head, softcap = self._fce_setup(model)
            inputs = dict(inputs)
            labels = inputs.pop("labels", None)
            shift_labels = inputs.pop("shift_labels", None)
            inputs.pop("num_items_in_batch", None)
            fwd = {k: v for k, v in inputs.items() if k in self._fce_sig}
            fwd["use_cache"] = False
            out = decoder(**fwd)
            hidden = out[0] if not hasattr(out, "last_hidden_state") else out.last_hidden_state
            if shift_labels is None:
                if labels is None:
                    raise RuntimeError("--fused-ce needs labels in the batch")
                hidden = hidden[..., :-1, :]
                shift_labels = labels[..., 1:]
            shift_labels = shift_labels.to(hidden.device)
            keep = shift_labels != -100
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])[keep.reshape(-1)]
            flat_labels = shift_labels.reshape(-1)[keep.reshape(-1)]
            if flat_labels.numel() == 0:
                loss = (hidden.sum() * 0.0).float()
            else:
                loss = self._fce_loss(
                    flat_hidden, flat_labels, lm_head.weight, softcap, num_items_in_batch
                )
            try:
                mode = "train" if self.model.training else "eval"
                if "attention_mask" in inputs:
                    tok = int(inputs["attention_mask"].sum().item())
                else:
                    tok = int(keep.numel())
                if mode == "train":
                    self._total_train_tokens += tok
                self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
            except Exception:
                pass
            if return_outputs:
                return loss, out
            return loss

    return FusedCESFTTrainer


def sft_config(output_dir: Path, args: argparse.Namespace, batch: int, accum: int):  # noqa: C901
    from trl import SFTConfig

    params = inspect.signature(SFTConfig.__init__).parameters
    kwargs = {
        "output_dir": str(output_dir / "trainer"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": batch,
        "per_device_eval_batch_size": max(1, batch // 2),
        "gradient_accumulation_steps": accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "eval_strategy": "no",
        "report_to": "none",
        "gradient_checkpointing": args.grad_checkpointing == "on",
        "max_grad_norm": 1.0,
        "optim": args.optim,
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_workers,
        "packing": args.packing == "on",
    }
    if "group_by_length" in params and args.group_by_length == "on" and args.packing != "on":
        kwargs["group_by_length"] = True
    if args.padding_free == "on" and "padding_free" in params:
        kwargs["padding_free"] = True
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs.pop("eval_strategy")
        kwargs["evaluation_strategy"] = "no"
    if "max_length" in params:
        kwargs["max_length"] = args.max_seq_len
    elif "max_seq_length" in params:
        kwargs["max_seq_length"] = args.max_seq_len
    if "gradient_checkpointing_kwargs" in params and args.grad_checkpointing == "on":
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if args.dataloader_workers and "dataloader_persistent_workers" in params:
        kwargs["dataloader_persistent_workers"] = True
    if args.dataloader_workers and "dataloader_pin_memory" in params:
        kwargs["dataloader_pin_memory"] = True
    if getattr(args, "fused_ce", "off") == "on" and "loss_type" in params:
        # TRL's own chunked_nll patches the model forward; the fused path replaces
        # compute_loss instead, so keep the stock nll wiring underneath it.
        kwargs["loss_type"] = "nll"
    if "completion_only_loss" in params:
        kwargs["completion_only_loss"] = True
    if "assistant_only_loss" in params:
        kwargs["assistant_only_loss"] = True
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return SFTConfig(**kwargs)
    return SFTConfig(**{k: v for k, v in kwargs.items() if k in params})


def make_trainer(model, processor, config, train_ds, peft_cfg, callbacks=None, cls=None):
    from trl import SFTTrainer

    cls = cls or SFTTrainer
    print(f"trl loss_type={getattr(config, 'loss_type', None)!r} trainer={cls.__name__}", flush=True)
    params = inspect.signature(SFTTrainer.__init__).parameters
    kwargs = {"model": model, "args": config, "train_dataset": train_ds, "peft_config": peft_cfg}
    if "processing_class" in params:
        kwargs["processing_class"] = processor
    elif "tokenizer" in params:
        kwargs["tokenizer"] = processor
    if callbacks is not None and "callbacks" in params:
        kwargs["callbacks"] = callbacks
    return cls(**kwargs)


def train_callbacks(max_seconds: float):
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class LeaseHeartbeatCallback(TrainerCallback):
        def __init__(self) -> None:
            self.last = time.perf_counter()

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ) -> TrainerControl:
            now = time.perf_counter()
            if now - self.last >= 600:
                heartbeat()
                self.last = now
            return control

    class TimeLimitCallback(TrainerCallback):
        def __init__(self, limit: float) -> None:
            self.limit = limit
            self.t0 = time.perf_counter()

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ) -> TrainerControl:
            if self.limit > 0 and time.perf_counter() - self.t0 >= self.limit:
                print(f"hit max-seconds={self.limit:.0f}; stopping training", flush=True)
                control.should_training_stop = True
            return control

    class StepTimerCallback(TrainerCallback):
        """Median wall time per optimizer step, warmup steps dropped."""

        def __init__(self) -> None:
            self.last: float | None = None
            self.times: list[float] = []
            self.tokens = 0.0

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ) -> TrainerControl:
            now = time.perf_counter()
            if self.last is not None:
                self.times.append(now - self.last)
            self.last = now
            return control

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kwargs,
        ) -> TrainerControl:
            if logs and "num_tokens" in logs:
                try:
                    self.tokens = float(logs["num_tokens"])
                except (TypeError, ValueError):
                    pass
            return control

        def summary(self, skip: int = 3) -> dict:
            warm = self.times[skip:]
            if not warm:
                return {}
            warm_sorted = sorted(warm)
            med = warm_sorted[len(warm_sorted) // 2]
            out = {"median_step_s": med, "timed_steps": len(warm)}
            if self.tokens and self.times:
                out["real_tokens_logged"] = self.tokens
                out["real_tokens_per_s"] = self.tokens / sum(self.times) * (
                    len(self.times) / max(1, len(self.times))
                )
            return out

    callbacks = [LeaseHeartbeatCallback(), StepTimerCallback()]
    if max_seconds and max_seconds > 0:
        callbacks.append(TimeLimitCallback(max_seconds))
    return callbacks


def save_processor(processor, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for method in ("save_pretrained",):
        if hasattr(processor, method):
            processor.save_pretrained(path)
            return
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)


def already_done(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and (
        (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
    )


def train_one(args: argparse.Namespace) -> Path:
    import torch

    set_hf_env()
    ensure_data_dirs()
    apply_backend_flags(args)
    name = adapter_name(args.model, args.name)
    out = DATA_ROOT / "adapters" / name
    if already_done(out) and not args.force:
        print(f"skip existing adapter {out}", flush=True)
        return out
    token = hf_token()
    rows = read_jsonl(args.train)
    if not rows:
        raise RuntimeError(f"empty train set {args.train}")
    processor = load_processor(args.model, token)
    if getattr(processor, "pad_token", None) is None and hasattr(processor, "eos_token"):
        processor.pad_token = processor.eos_token
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    train_ds = render_dataset(processor, rows, args.limit)
    print(f"train rows={len(train_ds)} model={args.model} out={out}", flush=True)
    model, kind = load_model(args.model, token, args.attn_impl)
    print(f"loaded kind={kind}", flush=True)
    if hasattr(model, "config") and hasattr(tokenizer, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    if args.grad_checkpointing == "on" and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if args.compile != "off":
        mode = None if args.compile == "default" else args.compile
        model = torch.compile(model, mode=mode)
        print(f"torch.compile mode={args.compile}", flush=True)
    batch, accum = auto_batch(args.model)
    if args.batch_size:
        batch = args.batch_size
    if args.grad_accum:
        accum = args.grad_accum
    if args.init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(args.init_adapter), is_trainable=True
        )
        peft_cfg = None
        print(
            f"continuing from adapter {args.init_adapter} (is_trainable=True); "
            "lora_r/lora_dropout come from that adapter",
            flush=True,
        )
    else:
        peft_cfg = lora_config(args.lora_r, args.lora_dropout)
    trainer_cls = None
    if args.fused_ce == "on":
        from trl import SFTTrainer

        trainer_cls = fused_ce_trainer_cls(SFTTrainer, args.fused_ce_chunk)
    if getattr(args, "keep_weight", 0.0):
        if args.fused_ce == "on":
            raise RuntimeError(
                "--keep-weight needs per-token logits; rerun with --fused-ce off"
            )
        from trl import SFTTrainer

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "restraint_v0"))
        from keep_weight import keep_weight_trainer_cls

        trainer_cls = keep_weight_trainer_cls(args.keep_weight, SFTTrainer)
        print(f"keep-weight loss: copied target tokens x{args.keep_weight}", flush=True)
    last_err = None
    callbacks = train_callbacks(args.max_seconds)
    prof_cb = None
    if args.profile_steps > 0:
        prof_out = args.profile_out or (DATA_ROOT / "logs" / f"profile_{name}.json")
        prof_cb = profile_callback(args, Path(prof_out))
        callbacks.append(prof_cb)
    for attempt_batch, attempt_accum in ((batch, accum), (max(1, batch // 2), accum * 2), (1, 16)):
        try:
            torch.cuda.empty_cache()
            config = sft_config(out, args, attempt_batch, attempt_accum)
            trainer = make_trainer(
                model, processor, config, train_ds, peft_cfg, callbacks, trainer_cls
            )
            t0 = time.perf_counter()
            trainer.train()
            out.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(out))
            save_processor(processor, out)
            elapsed = time.perf_counter() - t0
            meta = {
                "model": args.model,
                "kind": kind,
                "batch_size": attempt_batch,
                "grad_accum": attempt_accum,
                "epochs": args.epochs,
                "lr": args.lr,
                "max_seq_len": args.max_seq_len,
                "lora_r": args.lora_r,
                "lora_dropout": args.lora_dropout,
                "init_adapter": str(args.init_adapter) if args.init_adapter else None,
                "max_seconds": args.max_seconds,
                "train_rows": len(train_ds),
                "seconds": elapsed,
                "hit_time_cap": bool(args.max_seconds and elapsed >= args.max_seconds),
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "grad_checkpointing": args.grad_checkpointing,
                "optim": args.optim,
                "dataloader_workers": args.dataloader_workers,
                "attn_impl": args.attn_impl,
                "sdpa_backend": args.sdpa_backend,
                "compile": args.compile,
                "packing": args.packing,
                "padding_free": args.padding_free,
                "group_by_length": args.group_by_length,
                "fused_ce": args.fused_ce,
                "keep_weight": args.keep_weight,
                "fused_ce_kernel": (
                    None
                    if args.fused_ce != "on"
                    else ("liger" if _liger_loss_cls() is not None else "torch_chunked")
                ),
            }
            steps = getattr(trainer.state, "global_step", 0) or 0
            if steps:
                seq = args.max_seq_len
                meta["steps"] = steps
                meta["sec_per_step"] = elapsed / steps
                meta["tokens_per_s"] = (
                    steps * attempt_batch * attempt_accum * seq / elapsed
                )
            for cb in callbacks:
                if hasattr(cb, "summary"):
                    meta.update(cb.summary())
            (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(json.dumps(meta), flush=True)
            return out
        except torch.cuda.OutOfMemoryError as exc:
            last_err = exc
            print(f"OOM batch={attempt_batch}: {exc}; retrying smaller", flush=True)
            torch.cuda.empty_cache()
    raise RuntimeError(f"training OOM after retries: {last_err}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        train_one(args)
        return 0
    except Exception as exc:
        record_failure(f"train_lora:{args.model}", f"{type(exc).__name__}: {exc}", 1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
