"""Prompt format and the decode-time length guard.

The system prompt and the vocabulary line are byte-identical to research/vocab_v0/vocab_common.py;
the guard constants are the ones in research/bigrun_v0/refine_bigrun.py (results_guard.md).
"""
from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "Rewrite the raw dictation transcript into the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)
VOCAB_PREFIX = "Vocabulary the speaker may use: "

# decode-time LENGTH GUARD (research/bigrun_v0/results_guard.md)
GUARD_TOK_MULT = 1.5
GUARD_TOK_ADD = 32
GUARD_MAX_RATIO = 2.0
GUARD_MIN_RATIO = 0.4
GUARD_MIN_WORDS = 8


def system_with_vocab(vocab: list[str] | None) -> str:
    if not vocab:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n" + VOCAB_PREFIX + ", ".join(vocab)


def chat_messages(raw: str, vocab: list[str] | None = None,
                  target: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": system_with_vocab(vocab)},
        {"role": "user", "content": raw},
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs


def apply_chat_template(processor, messages, add_generation_prompt=True, tokenize=False):
    kwargs = {"tokenize": tokenize, "add_generation_prompt": add_generation_prompt}
    last = None
    for extra in ({"enable_thinking": False}, {}):
        try:
            out = processor.apply_chat_template(messages, **kwargs, **extra)
            if isinstance(out, list) and out and isinstance(out[0], str):
                out = out[0]
            return out
        except TypeError as exc:
            last = exc
    raise RuntimeError(f"chat template failed: {last}")


def prompt_text(processor, user_text: str, vocab: list[str] | None) -> str:
    return str(apply_chat_template(processor, chat_messages(user_text, vocab)))


def guard_cap(n_input_tokens: int, hard_cap: int) -> int:
    return max(8, min(hard_cap, int(GUARD_TOK_MULT * n_input_tokens) + GUARD_TOK_ADD))


def guard_post(inp: str, out: str) -> tuple[str, bool]:
    """Fall back to the raw input on a runaway or a collapse."""
    iw = len(inp.split())
    ow = len(out.split())
    if iw == 0:
        return out, False
    if ow > GUARD_MAX_RATIO * iw:
        return inp, True
    if iw > GUARD_MIN_WORDS and ow < GUARD_MIN_RATIO * iw:
        return inp, True
    return out, False


_THINK = (
    re.compile(r"<think>.*?</think>", re.S | re.I),
    re.compile(r"<\|think\|>.*?(?:<\|/?think\|>|$)", re.S),
    re.compile(r"<\|channel\|>thought.*?<\|channel\|>", re.S),
)


def strip_thinking(text: str) -> str:
    cleaned = text or ""
    for pat in _THINK:
        cleaned = pat.sub("", cleaned)
    return cleaned.strip()
