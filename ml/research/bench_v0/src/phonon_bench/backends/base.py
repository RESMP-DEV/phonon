"""One interface for every engine: load weights once, generate batched, optionally expose
teacher-forced log-probabilities and hidden states."""
from __future__ import annotations


class Backend:
    name = "base"
    supports_numerics = False
    supports_hidden = False

    # ---- lifecycle ----
    def load(self) -> dict:
        """Load weights + adapter. Returns a small dict of provenance."""
        raise NotImplementedError

    def warmup(self) -> None:
        """One forward/decode so the first timed batch is not paying kernel autotune."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ---- prompting ----
    def build_prompt(self, text: str, vocab: list[str]) -> str:
        raise NotImplementedError

    def count_tokens(self, texts: list[str]) -> list[int]:
        raise NotImplementedError

    # ---- generation ----
    def generate_batch(self, prompts: list[str], caps: list[int]):
        """Greedy. Returns (texts, capped_flags, generated_token_count)."""
        raise NotImplementedError

    def generate_timed(self, prompt: str, cap: int):
        """Batch 1. Returns (text, ttft_s, total_s, n_tokens)."""
        raise NotImplementedError

    # ---- numerics ----
    def teacher_forced(self, prompt: str, target: str, topk: int):
        """Returns dict(top_ids [T,K] int32, top_logprobs [T,K] float32, top1 [T] int32,
        target_ids [T] int32, target_logprob [T] float32) or None."""
        return None

    def logprobs_at(self, prompt: str, target: str, ids):
        """Candidate log-probs at the reference's top-K ids. Returns (lp [T,K], top1 [T])."""
        return None

    def hidden_last_prompt(self, prompt: str):
        """[n_layers+1, hidden] float32 of the last prompt position, or None."""
        return None

    # ---- resources ----
    def peak_vram_gb(self) -> float | None:
        return None
