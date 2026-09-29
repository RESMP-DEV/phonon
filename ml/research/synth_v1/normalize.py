"""Elliot-style target normalization: ASCII punct, no markdown, no um/uh."""
from __future__ import annotations

import re

EM = "\u2014\u2015\u2e3a\u2e3b"
EN = "\u2013\u2012\u2212"
CURLY_SINGLE = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
}
CURLY_DOUBLE = {
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2033": '"',
}

FILLER_RE = re.compile(r"(?:^|(?<=[\s,.;:!?]))(?:um+|uh+|uhh|umm)(?=[\s,.;:!?]|$)")
MD_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
MD_HEAD = re.compile(r"^#{1,6}\s+", re.M)
MULTI_SPACE = re.compile(r"[ \t]{2,}")
SPACE_COMMA = re.compile(r"\s+,")
SPACE_PERIOD = re.compile(r"\s+\.")


def normalize_target(text: str) -> str:
    t = text or ""
    for ch in EM:
        t = t.replace(ch, ", ")
    for ch in EN:
        t = t.replace(ch, "-")
    for a, b in CURLY_SINGLE.items():
        t = t.replace(a, b)
    for a, b in CURLY_DOUBLE.items():
        t = t.replace(a, b)
    t = t.replace("`", "")
    t = t.replace("\u00a0", " ")
    t = t.replace("\u2026", "...")
    t = MD_BOLD.sub(lambda m: m.group(1) or m.group(2) or "", t)
    t = MD_HEAD.sub("", t)
    t = FILLER_RE.sub(" ", t)
    t = SPACE_COMMA.sub(",", t)
    t = SPACE_PERIOD.sub(".", t)
    t = re.sub(r",\s*,+", ", ", t)
    t = MULTI_SPACE.sub(" ", t)
    t = re.sub(r"\s+\n", "\n", t)
    return t.strip()


def chat_messages(raw: str, target: str, system: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": raw},
        {"role": "assistant", "content": target},
    ]
