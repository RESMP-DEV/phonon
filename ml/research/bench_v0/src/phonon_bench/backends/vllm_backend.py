"""vLLM offline engine: the whole set in one `generate` call.

LoRA is attached with a LoRARequest when --adapter is given; otherwise the base or merged
directory is served directly. vLLM's own scheduler batches, so --max-batch only bounds how many
prompts the bench hands it at a time (default: the whole set).
"""
from __future__ import annotations

import time

from ..prompts import apply_chat_template, chat_messages, strip_thinking
from .base import Backend


class VLLMBackend(Backend):
    name = "vllm"
    supports_numerics = False
    supports_hidden = False

    def __init__(self, base: str, adapter: str | None = None, max_model_len: int = 4096,
                 gpu_frac: float = 0.85, max_lora_rank: int = 64):
        self.base, self.adapter = base, adapter
        self.max_model_len, self.gpu_frac, self.max_lora_rank = (
            max_model_len, gpu_frac, max_lora_rank)
        self.llm = self.tok = self.lora = None

    def load(self) -> dict:
        from vllm import LLM
        from transformers import AutoTokenizer

        src = self.adapter if self.adapter else self.base
        try:
            self.tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(self.base, trust_remote_code=True)
        kwargs = dict(model=self.base, dtype="bfloat16", max_model_len=self.max_model_len,
                      gpu_memory_utilization=self.gpu_frac, trust_remote_code=True,
                      enforce_eager=False, disable_log_stats=True)
        if self.adapter:
            kwargs.update(enable_lora=True, max_lora_rank=self.max_lora_rank, max_loras=1)
        self.llm = LLM(**kwargs)
        if self.adapter:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("bench", 1, str(self.adapter))
        import vllm
        return {"base": self.base, "adapter": self.adapter, "vllm": vllm.__version__}

    def warmup(self) -> None:
        self.generate_batch([self.build_prompt("warm up the kernels please", [])], [8])

    def build_prompt(self, text: str, vocab: list[str]) -> str:
        return str(apply_chat_template(self.tok, chat_messages(text, vocab)))

    def count_tokens(self, texts):
        return [len(x) for x in self.tok(texts, add_special_tokens=False)["input_ids"]]

    def _params(self, caps):
        from vllm import SamplingParams
        return [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(c), seed=0)
                for c in caps]

    def generate_batch(self, prompts, caps):
        kw = {"lora_request": self.lora} if self.lora else {}
        outs = self.llm.generate(prompts, self._params(caps), use_tqdm=False, **kw)
        texts, capped, ntok = [], [], 0
        for o, c in zip(outs, caps):
            g = o.outputs[0]
            texts.append(strip_thinking(g.text))
            capped.append(int(g.finish_reason == "length"))
            ntok += len(g.token_ids)
        return texts, capped, ntok

    def generate_timed(self, prompt: str, cap: int):
        t0 = time.perf_counter()
        txt, capped, n = self.generate_batch([prompt], [cap])
        total = time.perf_counter() - t0
        return txt[0], float("nan"), total, n

    def peak_vram_gb(self):
        try:
            import torch
            return torch.cuda.max_memory_allocated() / 1e9
        except Exception:
            return None
