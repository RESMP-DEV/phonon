"""LoRA SFT of Qwen3.5-4B on persona-gym trajectories (assistant-only loss).

Usage: python train_lora.py --data sft/ --model Qwen/Qwen3.5-4B --out out/
Labels are built by rendering the chat template incrementally so only assistant
turns (text and tool calls) contribute to the loss.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)


def render(tok, messages, tools):
    return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=False)


def encode(tok, sample: dict, max_len: int):
    """Token ids + labels (-100 outside assistant spans)."""
    msgs, tools = sample["messages"], sample.get("tools")
    ids: list[int] = []
    labels: list[int] = []
    prev_text = ""
    for k in range(1, len(msgs) + 1):
        text = render(tok, msgs[:k], tools)
        if not text.startswith(prev_text):
            # template re-wrote earlier text (should not happen); fall back to full mask
            return None
        delta = text[len(prev_text):]
        new_ids = tok(delta, add_special_tokens=False)["input_ids"]
        ids += new_ids
        labels += new_ids if msgs[k - 1]["role"] == "assistant" else [-100] * len(new_ids)
        prev_text = text
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": labels}


class Collate:
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(feats), n), -100, dtype=torch.long)
        att = torch.zeros((len(feats), n), dtype=torch.long)
        for i, f in enumerate(feats):
            L = len(f["input_ids"])
            ids[i, :L] = torch.tensor(f["input_ids"])
            lab[i, :L] = torch.tensor(f["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=40960)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    samples = []
    for name in ("train_trajectories.jsonl", "train_aux.jsonl", "train_generic.jsonl"):
        p = Path(a.data) / name
        if p.exists():
            rows = [json.loads(l) for l in open(p)]
            print(f"[data] {name}: {len(rows)}", file=sys.stderr)
            samples += rows
    enc, skipped, lens = [], 0, []
    for s in samples:
        e = encode(tok, s, a.max_len)
        if e is None:
            skipped += 1
            continue
        enc.append(e)
        lens.append(len(e["input_ids"]))
    lens.sort()
    print(f"[data] encoded {len(enc)}, skipped {skipped} (> {a.max_len} tok or template drift); "
          f"len p50={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} max={lens[-1]} "
          f"total={sum(lens)/1e6:.1f}M tok", file=sys.stderr)
    random.Random(a.seed).shuffle(enc)

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(enc) / a.grad_accum)
    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=1,
        gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=5, save_strategy="epoch", save_total_limit=2,
        report_to=[], seed=a.seed, dataloader_num_workers=2,
        group_by_length=True, remove_unused_columns=False,
    )
    print(f"[train] {len(enc)} samples, {steps_per_epoch} optimizer steps/epoch", file=sys.stderr)
    tr = Trainer(model=model, args=args, train_dataset=enc, data_collator=Collate(tok.pad_token_id))
    tr.train()
    model.save_pretrained(Path(a.out) / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(Path(a.out) / "merged", safe_serialization=True)
    tok.save_pretrained(Path(a.out) / "merged")
    print("[train] done: adapter + merged saved", file=sys.stderr)


if __name__ == "__main__":
    main()
