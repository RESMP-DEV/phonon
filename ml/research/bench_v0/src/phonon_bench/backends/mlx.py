"""mlx_lm backend: a quantized or bf16 MLX model directory.

Works on gpubox through the mlx[cuda13] wheel and on a Mac through Metal. Per-row token caps are
honoured by grouping the batch's rows by cap (length-sorted batching already makes the caps
nearly uniform inside a batch).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# MLX's CUDA backend keeps fixed-size kernel caches. The bench changes max_tokens every batch
# (per-row guard caps), which thrashes the conv, SDPA and CUDA-graph caches and aborts the run
# with "Cache thrashing is happening". Set before mlx.core is imported.
for _v in ("MLX_CUDA_CONV_CACHE_SIZE", "MLX_CUDA_SDPA_CACHE_SIZE", "MLX_CUDA_GRAPH_CACHE_SIZE"):
    os.environ.setdefault(_v, "8192")

from ..prompts import apply_chat_template, chat_messages, strip_thinking
from .base import Backend


class MLXBackend(Backend):
    name = "mlx"
    supports_numerics = True
    supports_hidden = False

    def __init__(self, model_dir: str):
        self.model_dir = str(model_dir)
        self.model = self.tok = None

    def load(self) -> dict:
        from mlx_lm import load

        self.model, self.tok = load(self.model_dir)
        import mlx.core as mx
        size = sum(p.stat().st_size for p in Path(self.model_dir).rglob("*") if p.is_file())
        try:
            import mlx_lm
            ver = mlx_lm.__version__
        except Exception:
            ver = "?"
        return {"model_dir": self.model_dir, "mlx": mx.__version__, "mlx_lm": ver,
                "dir_bytes": size}

    def warmup(self) -> None:
        self.generate_batch([self.build_prompt("warm up the kernels please", [])], [8])

    def build_prompt(self, text: str, vocab: list[str]) -> str:
        return str(apply_chat_template(self.tok, chat_messages(text, vocab)))

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(self.tok.encode(t)) for t in texts]

    def generate_batch(self, prompts, caps):
        """One batch_generate at max(caps), then each row is truncated to its own cap.

        Greedy decoding is independent per row, so generating at max(cap) and slicing row i at
        cap_i is the same as having generated row i with max_tokens=cap_i - the identity
        refine_bigrun.py relies on. The cap binds on about 1 row in 3,000.
        """
        from mlx_lm.generate import batch_generate

        toks = [self.tok.encode(p) for p in prompts]
        resp = batch_generate(self.model, self.tok, toks, max_tokens=max(caps), verbose=False)
        ntok = int(getattr(getattr(resp, "stats", None), "generation_tokens", 0) or 0)
        texts, capped = [], []
        for t, c in zip(resp.texts, caps):
            ids = self.tok.encode(t, add_special_tokens=False)
            if len(ids) > c:
                t = self.tok.decode(ids[:c])
                capped.append(1)
            else:
                capped.append(int(len(ids) >= c))
            texts.append(strip_thinking(t))
        return texts, capped, ntok

    def generate_timed(self, prompt: str, cap: int):
        import mlx.core as mx
        from mlx_lm.generate import stream_generate

        t0 = time.perf_counter()
        ttft, n, parts = None, 0, []
        for resp in stream_generate(self.model, self.tok, self.tok.encode(prompt),
                                    max_tokens=cap):
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(resp.text)
            n += 1
        mx.synchronize() if hasattr(mx, "synchronize") else None
        total = time.perf_counter() - t0
        return strip_thinking("".join(parts)), (ttft or total), total, n

    # ---- numerics -----------------------------------------------------------
    def _tf_ids(self, prompt: str, target: str):
        pid = self.tok.encode(prompt)
        full = self.tok.encode(prompt + target)
        if full[:len(pid)] != pid:
            full = pid + self.tok.encode(target)
        return pid, full

    def _logprobs(self, full):
        import mlx.core as mx

        ids = mx.array(full)[None]
        logits = self.model(ids)[0].astype(mx.float32)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def teacher_forced(self, prompt: str, target: str, topk: int):
        import mlx.core as mx
        import numpy as np

        pid, full = self._tf_ids(prompt, target)
        if len(full) <= len(pid):
            return None
        lp = self._logprobs(full)[len(pid) - 1:len(full) - 1]
        k = min(topk, lp.shape[-1])
        idx = mx.argpartition(-lp, k - 1, axis=-1)[:, :k]
        vals = mx.take_along_axis(lp, idx, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        idx = mx.take_along_axis(idx, order, axis=-1)
        vals = mx.take_along_axis(vals, order, axis=-1)
        tgt = mx.array(full[len(pid):])
        tlp = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
        mx.eval(idx, vals, tlp)
        return {"top_ids": np.array(idx).astype("int32"),
                "top_logprobs": np.array(vals).astype("float32"),
                "top1": np.array(idx[:, 0]).astype("int32"),
                "target_ids": np.array(tgt).astype("int32"),
                "target_logprob": np.array(tlp).astype("float32")}

    def logprobs_at(self, prompt: str, target: str, ids):
        import mlx.core as mx
        import numpy as np

        pid, full = self._tf_ids(prompt, target)
        if len(full) <= len(pid):
            return None
        lp = self._logprobs(full)[len(pid) - 1:len(full) - 1]
        t = min(lp.shape[0], ids.shape[0])
        want = mx.array(np.asarray(ids[:t], dtype="int32"))
        got = mx.take_along_axis(lp[:t], want, axis=-1)
        top1 = mx.argmax(lp[:t], axis=-1)
        mx.eval(got, top1)
        return np.array(got).astype("float32"), np.array(top1).astype("int32")

    def peak_vram_gb(self):
        try:
            import mlx.core as mx
            return float(mx.get_peak_memory()) / 1e9
        except Exception:
            return None
