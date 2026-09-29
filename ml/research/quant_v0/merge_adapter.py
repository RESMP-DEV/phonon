"""Merge a PEFT LoRA adapter into its base and save a plain HF checkpoint (bf16) for quantization."""
import argparse, json, shutil
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base", required=True); p.add_argument("--adapter", required=True); p.add_argument("--out", required=True)
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
out = Path(a.out); shutil.rmtree(out, ignore_errors=True)
model.save_pretrained(out, safe_serialization=True); tok.save_pretrained(out)
ct = Path(a.adapter) / "chat_template.jinja"
if ct.exists(): shutil.copy(ct, out / "chat_template.jinja")
n = sum(p.numel() for p in model.parameters())
(out / "merge_meta.json").write_text(json.dumps({"base": a.base, "adapter": a.adapter, "params": n}, indent=1))
print("MERGED", out, n)
