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
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)


def render(tok, messages, tools):
    return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=False)


ASST = "<|im_start|>assistant\n"
END = "<|im_end|>\n"


def normalize(msgs):
    """Coerce tool_call arguments to dicts (template iterates them as a mapping)."""
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = json.loads(json.dumps(c))
                fn = c.get("function", c)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        fn["arguments"] = {"input": args}
                elif args is None:
                    fn["arguments"] = {}
                calls.append(c)
            m["tool_calls"] = calls
        out.append(m)
    return out


def encode(tok, sample: dict, max_len: int):
    """Token ids + labels (-100 outside assistant spans).

    Renders the full conversation once and labels every
    `<|im_start|>assistant\n ... <|im_end|>\n` span (header masked, end token
    supervised). Chunk-wise tokenization keeps span boundaries exact; the
    boundaries sit next to special tokens so no merges cross them.
    """
    try:
        text = render(tok, normalize(sample["messages"]), sample.get("tools"))
    except Exception as e:  # noqa: BLE001
        print(f"[data] render failed: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
        return None
    ids: list[int] = []
    labels: list[int] = []

    def add(chunk: str, supervised: bool):
        if not chunk:
            return
        t = tok(chunk, add_special_tokens=False)["input_ids"]
        ids.extend(t)
        labels.extend(t if supervised else [-100] * len(t))

    pos = 0
    while True:
        s = text.find(ASST, pos)
        if s < 0:
            add(text[pos:], False)
            break
        e = text.find(END, s)
        e = len(text) if e < 0 else e + len(END)
        add(text[pos:s + len(ASST)], False)
        add(text[s + len(ASST):e], True)
        pos = e
    if len(ids) > max_len or not any(l != -100 for l in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def _chunk_loss(head, hc, tc):
    logits = head(hc).float()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), tc.reshape(-1),
                           ignore_index=-100, reduction="sum")


class ChunkedTrainer(Trainer):
    """Token-summed CE computed in 4096-position chunks so full-vocab logits for a
    40k-token sample never materialize at once (27 GiB in fp32 otherwise)."""
    CHUNK = 4096

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        causal = model.base_model.model if hasattr(model, "base_model") else model
        dec = causal.get_decoder()
        head = causal.get_output_embeddings()
        h = dec(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        h, tgt = h[:, :-1], labels[:, 1:]
        total = torch.zeros((), device=h.device, dtype=torch.float32)
        for i in range(0, h.shape[1], self.CHUNK):
            hc, tc = h[:, i:i + self.CHUNK], tgt[:, i:i + self.CHUNK]
            if (tc != -100).any():
                total = total + checkpoint(_chunk_loss, head, hc, tc, use_reentrant=False)
        if num_items_in_batch is not None and getattr(self, "model_accepts_loss_kwargs", False):
            n = num_items_in_batch
        else:
            n = (tgt != -100).sum().clamp(min=1)
        loss = total / n
        return (loss, None) if return_outputs else loss


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
    ap.add_argument("--dry-run", action="store_true")
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
    if a.dry_run:
        e = next(x for x in enc if len(x["input_ids"]) < 6000)
        sup = sum(1 for l in e["labels"] if l != -100)
        print(f"[dry] sample len={len(e['input_ids'])} supervised={sup}", file=sys.stderr)
        print(tok.decode([t for t, l in zip(e["input_ids"], e["labels"]) if l != -100])[:1500], file=sys.stderr)
        full = tok(tok.decode(e["input_ids"]), add_special_tokens=False)["input_ids"]
        print(f"[dry] retokenize len={len(full)} (chunked={len(e['input_ids'])})", file=sys.stderr)
        return

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
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=10,
        bf16=True, logging_steps=5, save_strategy="epoch", save_total_limit=2,
        report_to=[], seed=a.seed, dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    print(f"[train] {len(enc)} samples, {steps_per_epoch} optimizer steps/epoch", file=sys.stderr)
    tr = ChunkedTrainer(model=model, args=args, train_dataset=enc, data_collator=Collate(tok.pad_token_id))
    longest = max(enc, key=lambda e: len(e["input_ids"]))
    batch = {k: v.cuda() for k, v in Collate(tok.pad_token_id)([longest]).items()}
    model.train()
    loss = tr.compute_loss(model, batch)
    loss.backward()
    model.zero_grad(set_to_none=True)
    print(f"[smoke] longest={len(longest['input_ids'])} loss={loss.item():.3f} "
          f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", file=sys.stderr)
    torch.cuda.reset_peak_memory_stats()
    tr.train()
    model.save_pretrained(Path(a.out) / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(Path(a.out) / "merged", safe_serialization=True)
    tok.save_pretrained(Path(a.out) / "merged")
    print("[train] done: adapter + merged saved", file=sys.stderr)


if __name__ == "__main__":
    main()
