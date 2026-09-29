"""OpenAI-compatible server (llama-server, or anything else that speaks /v1/chat/completions).

Generation only: no logprobs, no hidden states. The batch is issued as concurrent requests, so
`--max-batch` is the client concurrency.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
import urllib.request

from ..prompts import chat_messages, strip_thinking
from .base import Backend


class LlamaBackend(Backend):
    name = "llama"
    supports_numerics = False
    supports_hidden = False

    def __init__(self, base_url: str, model: str = "default", concurrency: int = 16):
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.concurrency = concurrency

    def load(self) -> dict:
        try:
            with urllib.request.urlopen(self.base_url + "/v1/models", timeout=30) as r:
                models = json.load(r)
        except Exception as exc:
            raise RuntimeError(f"{self.base_url} not reachable: {exc}") from exc
        return {"base_url": self.base_url, "models": models}

    def warmup(self) -> None:
        self._one("warm up the kernels please", [], 8)

    def build_prompt(self, text: str, vocab: list[str]) -> str:
        # the server applies the chat template; carry (text, vocab) as JSON
        return json.dumps({"text": text, "vocab": vocab})

    def count_tokens(self, texts: list[str]) -> list[int]:
        # no tokenizer client-side; 1.3 tokens/word is the measured LFM2 ratio on this corpus
        return [max(1, int(1.3 * len(t.split())) + 1) for t in texts]

    def _one(self, text, vocab, max_tokens):
        body = json.dumps({"model": self.model_name,
                           "messages": chat_messages(text, vocab),
                           "temperature": 0.0, "max_tokens": int(max_tokens), "seed": 0}).encode()
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        ch = d["choices"][0]
        return (strip_thinking(ch["message"]["content"] or ""),
                int(ch.get("finish_reason") == "length"),
                int(d.get("usage", {}).get("completion_tokens", 0)))

    def generate_batch(self, prompts, caps):
        jobs = [json.loads(p) for p in prompts]
        with cf.ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            res = list(ex.map(lambda a: self._one(a[0]["text"], a[0]["vocab"], a[1]),
                              zip(jobs, caps)))
        return [r[0] for r in res], [r[1] for r in res], sum(r[2] for r in res)

    def generate_timed(self, prompt: str, cap: int):
        j = json.loads(prompt)
        t0 = time.perf_counter()
        txt, _, n = self._one(j["text"], j["vocab"], cap)
        total = time.perf_counter() - t0
        return txt, float("nan"), total, n
