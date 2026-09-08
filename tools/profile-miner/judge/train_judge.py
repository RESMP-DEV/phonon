"""Per-term keep/drop classifier: LoRA on Qwen3.5-4B, trained on Opus labels, gated on a held-out split.

Scores a candidate by the next-token logit margin keep vs drop after the assistant header (thinking off), so
the same harness gives the base model's zero-shot per-term score and the adapter's score.
usage: python train_judge.py --input judge_final.json --labels labels_opus.json --heldout heldout_ids.json
       --gold gold_terms.json --out out/
"""
import argparse
import json
import math
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import PER_TERM, fmt_single

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--labels", required=True)
ap.add_argument("--heldout", required=True); ap.add_argument("--gold", default=None)
ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--out", default="out")
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--max-len", type=int, default=1024); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--target-regex", default=None,
                help="PEFT target_modules regex, e.g. r'.*language_model.*\\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$' "
                     "to skip the audio and vision towers of a multimodal model")
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)

cands = json.load(open(a.input))
labels = {int(k): v for k, v in json.load(open(a.labels)).items()}
held = set(json.load(open(a.heldout)))
gold = set(json.load(open(a.gold))) if a.gold else set()
cands = [c for c in cands if c["id"] in labels]
train = [c for c in cands if c["id"] not in held]
test = [c for c in cands if c["id"] in held]
print(f"[judge] {len(cands)} labelled, train {len(train)} (keep {sum(labels[c['id']] for c in train)}), "
      f"held-out {len(test)} (keep {sum(labels[c['id']] for c in test)})", file=sys.stderr)

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
KEEP = tok.encode("keep", add_special_tokens=False)[0]
DROP = tok.encode("drop", add_special_tokens=False)[0]
END = tok.encode("<|im_end|>", add_special_tokens=False)


def prompt(c):
    msgs = [{"role": "system", "content": PER_TERM}, {"role": "user", "content": fmt_single(c)}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                             device_map="cuda")
model.config.use_cache = False


@torch.no_grad()
def score(items, bs=16):
    """keep-minus-drop logit margin per item."""
    model.eval()
    out = {}
    for i in range(0, len(items), bs):
        batch = items[i:i + bs]
        enc = tok([prompt(c) for c in batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=a.max_len).to("cuda")
        logits = model(**enc).logits[:, -1, :].float()
        m = (logits[:, KEEP] - logits[:, DROP]).tolist()
        top = logits.argmax(-1).tolist()
        for c, mm, t in zip(batch, m, top):
            out[c["id"]] = {"margin": mm, "top_is_answer": t in (KEEP, DROP)}
    return out


def metrics(scores, items, thr=0.0):
    y = [labels[c["id"]] for c in items]
    p = [scores[c["id"]]["margin"] > thr for c in items]
    tp = sum(a_ and b for a_, b in zip(y, p)); fp = sum((not a_) and b for a_, b in zip(y, p))
    fn = sum(a_ and (not b) for a_, b in zip(y, p))
    agree = sum(a_ == b for a_, b in zip(y, p)) / max(len(y), 1)
    g = [c for c in items if c["term"].strip().lower() in gold]
    return {"n": len(items), "agreement": round(agree, 3), "keep_rate": round(sum(p) / max(len(p), 1), 3),
            "keep_precision": round(tp / max(tp + fp, 1), 3), "keep_recall": round(tp / max(tp + fn, 1), 3),
            "gold_in_set": len(g), "gold_kept": sum(scores[c["id"]]["margin"] > thr for c in g),
            "answer_is_top_token": round(sum(scores[c["id"]]["top_is_answer"] for c in items) / max(len(items), 1), 3)}


def best_threshold(scores, items):
    ms = sorted(scores[c["id"]]["margin"] for c in items)
    best = (0.0, -1)
    for t in [ms[0] - 1] + ms:
        acc = metrics(scores, items, t)["agreement"]
        if acc > best[1]:
            best = (t, acc)
    return best


summary = {"train": len(train), "heldout": len(test)}
t0 = time.time()
zs = score(test)
summary["zero_shot_per_term"] = metrics(zs, test)
thr, acc = best_threshold(zs, test)
summary["zero_shot_per_term_best_threshold"] = {"threshold": round(thr, 3), "agreement": acc}
print(f"[judge] zero-shot per-term {summary['zero_shot_per_term']} best-thr {summary['zero_shot_per_term_best_threshold']} "
      f"({time.time() - t0:.0f}s)", file=sys.stderr)

# ---- LoRA training on the answer tokens only
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=a.target_regex or TARGETS)
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()


def encode(c):
    p = tok.encode(prompt(c), add_special_tokens=False)[-(a.max_len - 4):]
    ans = [KEEP if labels[c["id"]] else DROP] + END
    return p + ans, [-100] * len(p) + ans


def collate(items):
    encs = [encode(c) for c in items]
    n = max(len(x) for x, _ in encs)
    ids = torch.full((len(encs), n), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(encs), n), -100, dtype=torch.long)
    att = torch.zeros((len(encs), n), dtype=torch.long)
    for i, (x, y) in enumerate(encs):
        ids[i, n - len(x):] = torch.tensor(x); lab[i, n - len(x):] = torch.tensor(y); att[i, n - len(x):] = 1
    return ids.cuda(), lab.cuda(), att.cuda()


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(train) / a.bs)
total = steps_per_epoch * a.epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / total))
model.train()
step = 0
t0 = time.time()
for ep in range(a.epochs):
    order = train[:]
    random.shuffle(order)
    run = 0.0
    for i in range(0, len(order), a.bs):
        ids, lab, att = collate(order[i:i + a.bs])
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run += loss.item(); step += 1
        if step % 25 == 0:
            print(f"[train] ep {ep} step {step}/{total} loss {run / 25:.4f} {time.time() - t0:.0f}s", file=sys.stderr)
            run = 0.0
    model.eval()
    ev = metrics(score(test), test)
    summary[f"epoch_{ep + 1}"] = ev
    print(f"[judge] epoch {ep + 1} held-out {ev}", file=sys.stderr)
    model.train()

model.eval()
model.save_pretrained(f"{a.out}/adapter")
ft = score(test)
summary["lora_per_term"] = metrics(ft, test)
thr, acc = best_threshold(ft, test)
summary["lora_per_term_best_threshold"] = {"threshold": round(thr, 3), "agreement": acc}
tr = score(train)
summary["lora_train_fit"] = metrics(tr, train)
allp = score(cands)
json.dump({str(k): v for k, v in allp.items()}, open(f"{a.out}/preds_all.json", "w"))
kept_all = [c["term"] for c in cands if allp[c["id"]]["margin"] > 0]
summary["lora_all"] = {"n": len(cands), "kept": len(kept_all),
                       "gold_kept": sum(t.strip().lower() in gold for t in kept_all),
                       "gold_in_set": sum(c["term"].strip().lower() in gold for c in cands)}
# disagreements on held-out for the DEVLOG
dis = [{"term": c["term"], "opus": labels[c["id"]], "margin": round(ft[c["id"]]["margin"], 2)}
       for c in test if (ft[c["id"]]["margin"] > 0) != labels[c["id"]]]
summary["heldout_disagreements"] = dis[:60]
summary["train_seconds"] = round(time.time() - t0)
json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
print(f"[judge] LoRA held-out {summary['lora_per_term']} best-thr {summary['lora_per_term_best_threshold']}; "
      f"train fit {summary['lora_train_fit']['agreement']}; all-2000 kept {len(kept_all)}", file=sys.stderr)
print("TRAIN_DONE")
