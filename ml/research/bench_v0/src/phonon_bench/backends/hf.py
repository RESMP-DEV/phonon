"""transformers + PEFT backend: bf16, sdpa, left padding, greedy, per-row token caps."""
from __future__ import annotations

import time

from ..prompts import apply_chat_template, chat_messages, strip_thinking
from .base import Backend


class HFBackend(Backend):
    name = "hf"
    supports_numerics = True
    supports_hidden = True

    def __init__(self, base: str, adapter: str | None = None, dtype: str = "bfloat16",
                 attn: str = "sdpa", device: str = "cuda:0"):
        self.base, self.adapter, self.dtype_name, self.attn, self.device = (
            base, adapter, dtype, attn, device)
        self.model = self.processor = self.tokenizer = None

    def load(self) -> dict:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        src = self.base
        if self.adapter:
            from pathlib import Path
            if (Path(self.adapter) / "tokenizer_config.json").exists():
                src = self.adapter
        processor, last = None, None
        for loader in (AutoProcessor, AutoTokenizer):
            try:
                processor = loader.from_pretrained(src, trust_remote_code=True)
                break
            except Exception as exc:
                last = exc
        if processor is None:
            raise RuntimeError(f"processor load failed: {last}")
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        if getattr(self.tokenizer, "pad_token", None) is None and \
                getattr(self.tokenizer, "eos_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"dtype": getattr(torch, self.dtype_name), "device_map": {"": 0},
                  "trust_remote_code": True, "attn_implementation": self.attn}
        self.model = AutoModelForCausalLM.from_pretrained(self.base, **kwargs)
        if self.adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(self.adapter))
        self.model.eval()
        gc = getattr(self.model, "generation_config", None)
        if gc is not None:
            gc.do_sample = False
            if hasattr(gc, "temperature"):
                gc.temperature = None
        self.pad_id = getattr(self.tokenizer, "pad_token_id", None) or \
            getattr(self.tokenizer, "eos_token_id", None)
        self.eos_id = getattr(self.tokenizer, "eos_token_id", None)
        import transformers
        return {"base": self.base, "adapter": self.adapter, "dtype": self.dtype_name,
                "attn": self.attn, "transformers": transformers.__version__,
                "torch": torch.__version__,
                "n_params": sum(p.numel() for p in self.model.parameters())}

    def warmup(self) -> None:
        import torch

        p = self.build_prompt("warm up the kernels please", [])
        enc = self.tokenizer([p], return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.inference_mode():
            self.model.generate(**enc, max_new_tokens=8, do_sample=False,
                                pad_token_id=self.pad_id, eos_token_id=self.eos_id)
        torch.cuda.synchronize()

    def build_prompt(self, text: str, vocab: list[str]) -> str:
        return str(apply_chat_template(self.processor, chat_messages(text, vocab)))

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(x) for x in self.tokenizer(texts, add_special_tokens=False)["input_ids"]]

    def _encode_left(self, prompts):
        prev = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            self.tokenizer.padding_side = prev
        return {k: v.to(self.model.device) for k, v in enc.items()}

    def generate_batch(self, prompts, caps):
        import torch

        inputs = self._encode_left(prompts)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max(caps), do_sample=False,
                                      pad_token_id=self.pad_id, eos_token_id=self.eos_id)
        texts, capped, ntok = [], [], 0
        for i, c in enumerate(caps):
            seq = out[i, input_len:input_len + c]
            texts.append(strip_thinking(self.tokenizer.decode(seq, skip_special_tokens=True)))
            if self.eos_id is None:
                hit = bool(seq.shape[0] >= c)
                ntok += int(seq.shape[0])
            else:
                eq = (seq == self.eos_id)
                hit = not bool(eq.any().item())
                ntok += int(seq.shape[0]) if hit else int(eq.float().argmax().item()) + 1
            capped.append(int(hit))
        return texts, capped, ntok

    def generate_timed(self, prompt: str, cap: int):
        import torch

        inputs = self._encode_left([prompt])
        input_len = inputs["input_ids"].shape[-1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            self.model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                pad_token_id=self.pad_id, eos_token_id=self.eos_id)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0
        t1 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=cap, do_sample=False,
                                      pad_token_id=self.pad_id, eos_token_id=self.eos_id)
        torch.cuda.synchronize()
        total = time.perf_counter() - t1
        seq = out[0, input_len:]
        n = int(seq.shape[0])
        if self.eos_id is not None:
            eq = (seq == self.eos_id)
            if bool(eq.any().item()):
                n = int(eq.float().argmax().item()) + 1
        txt = strip_thinking(self.tokenizer.decode(seq, skip_special_tokens=True))
        return txt, ttft, total, n

    # ---- numerics -----------------------------------------------------------
    def _tf_ids(self, prompt: str, target: str):
        pid = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = self.tokenizer(prompt + target, add_special_tokens=False)["input_ids"]
        if full[:len(pid)] != pid:
            # retokenisation at the seam: fall back to encoding the target alone
            tid = self.tokenizer(target, add_special_tokens=False)["input_ids"]
            full = pid + tid
        return pid, full

    def _logits(self, full_ids, need_hidden=False):
        import torch

        ids = torch.tensor([full_ids], device=self.model.device)
        with torch.inference_mode():
            out = self.model(ids, output_hidden_states=need_hidden)
        return out

    def teacher_forced(self, prompt: str, target: str, topk: int):
        import torch

        pid, full = self._tf_ids(prompt, target)
        if len(full) <= len(pid):
            return None
        out = self._logits(full)
        logits = out.logits[0, len(pid) - 1:len(full) - 1].float()
        lp = torch.log_softmax(logits, dim=-1)
        k = min(topk, lp.shape[-1])
        vals, idx = torch.topk(lp, k, dim=-1)
        tgt = torch.tensor(full[len(pid):], device=lp.device)
        tlp = lp.gather(-1, tgt[:, None])[:, 0]
        return {"top_ids": idx.cpu().numpy().astype("int32"),
                "top_logprobs": vals.cpu().numpy().astype("float32"),
                "top1": idx[:, 0].cpu().numpy().astype("int32"),
                "target_ids": tgt.cpu().numpy().astype("int32"),
                "target_logprob": tlp.cpu().numpy().astype("float32")}

    def logprobs_at(self, prompt: str, target: str, ids):
        import numpy as np
        import torch

        pid, full = self._tf_ids(prompt, target)
        if len(full) <= len(pid):
            return None
        out = self._logits(full)
        logits = out.logits[0, len(pid) - 1:len(full) - 1].float()
        lp = torch.log_softmax(logits, dim=-1)
        t = min(lp.shape[0], ids.shape[0])
        want = torch.tensor(np.asarray(ids[:t], dtype="int64"), device=lp.device)
        got = lp[:t].gather(-1, want)
        return got.cpu().numpy().astype("float32"), lp[:t].argmax(-1).cpu().numpy().astype("int32")

    def hidden_last_prompt(self, prompt: str):
        import torch

        pid = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([pid], device=self.model.device)
        with torch.inference_mode():
            out = self.model(ids, output_hidden_states=True)
        hs = out.hidden_states
        return torch.stack([h[0, -1] for h in hs]).float().cpu().numpy()

    def peak_vram_gb(self):
        import torch

        return torch.cuda.max_memory_allocated() / 1e9

    def close(self):
        import torch

        self.model = None
        torch.cuda.empty_cache()
