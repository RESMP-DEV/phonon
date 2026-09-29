"""LoRA SFT for corrector v1 arms. Hyperparameters match v0; writes v1 adapter dirs."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    DATA_ROOT,
    QWEN_ID,
    SEED,
    apply_chat_template,
    ensure_data_dirs,
    heartbeat,
    hf_token,
    read_jsonl,
    set_hf_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=QWEN_ID)
    parser.add_argument("--name", required=True, help="Adapter directory name under adapters/.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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
            return proc
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"failed to load tokenizer/processor for {model_id}: {last_err}")


def load_model(model_id: str, token: str | None):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "token": token,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def render_dataset(processor, rows: list[dict], limit: int):
    from datasets import Dataset

    records = []
    for row in rows:
        if limit and len(records) >= limit:
            break
        records.append(
            {
                "messages": row["messages"],
                "text": apply_chat_template(processor, row["messages"], add_generation_prompt=False),
            }
        )
    return Dataset.from_list(records)


def lora_config(r: int):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=r,
        lora_alpha=r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )


def sft_config(output_dir: Path, args: argparse.Namespace, batch: int, accum: int):
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
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "packing": False,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs.pop("eval_strategy")
        kwargs["evaluation_strategy"] = "no"
    if "max_length" in params:
        kwargs["max_length"] = args.max_seq_len
    elif "max_seq_length" in params:
        kwargs["max_seq_length"] = args.max_seq_len
    if "gradient_checkpointing_kwargs" in params:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "completion_only_loss" in params:
        kwargs["completion_only_loss"] = True
    if "assistant_only_loss" in params:
        kwargs["assistant_only_loss"] = True
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return SFTConfig(**kwargs)
    return SFTConfig(**{k: v for k, v in kwargs.items() if k in params})


def make_trainer(model, processor, config, train_ds, peft_cfg, callbacks=None):
    from trl import SFTTrainer

    params = inspect.signature(SFTTrainer.__init__).parameters
    kwargs = {"model": model, "args": config, "train_dataset": train_ds, "peft_config": peft_cfg}
    if "processing_class" in params:
        kwargs["processing_class"] = processor
    elif "tokenizer" in params:
        kwargs["tokenizer"] = processor
    if callbacks is not None and "callbacks" in params:
        kwargs["callbacks"] = callbacks
    return SFTTrainer(**kwargs)


def train_callbacks():
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

    return [LeaseHeartbeatCallback()]


def save_processor(processor, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(processor, "save_pretrained"):
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
    out = DATA_ROOT / "adapters" / args.name
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
    print(f"train rows={len(train_ds)} model={args.model} out={out} seed={args.seed}", flush=True)
    model = load_model(args.model, token)
    if hasattr(model, "config") and hasattr(tokenizer, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    batch, accum = args.batch_size, args.grad_accum
    peft_cfg = lora_config(args.lora_r)
    last_err = None
    callbacks = train_callbacks()
    heartbeat()
    for attempt_batch, attempt_accum in ((batch, accum), (max(1, batch // 2), accum * 2), (1, 16)):
        try:
            torch.cuda.empty_cache()
            config = sft_config(out, args, attempt_batch, attempt_accum)
            trainer = make_trainer(model, processor, config, train_ds, peft_cfg, callbacks)
            t0 = time.perf_counter()
            trainer.train()
            out.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(out))
            save_processor(processor, out)
            elapsed = time.perf_counter() - t0
            meta = {
                "model": args.model,
                "batch_size": attempt_batch,
                "grad_accum": attempt_accum,
                "epochs": args.epochs,
                "lr": args.lr,
                "max_seq_len": args.max_seq_len,
                "lora_r": args.lora_r,
                "seed": args.seed,
                "train_rows": len(train_ds),
                "train_file": str(args.train),
                "seconds": elapsed,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            }
            (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(json.dumps(meta), flush=True)
            heartbeat()
            return out
        except torch.cuda.OutOfMemoryError as exc:
            last_err = exc
            print(f"OOM batch={attempt_batch}: {exc}; retrying smaller", flush=True)
            torch.cuda.empty_cache()
    raise RuntimeError(f"training OOM after retries: {last_err}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    train_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
