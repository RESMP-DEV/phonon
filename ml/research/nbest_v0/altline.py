"""Shared: turn recogniser n-best hypotheses into one `ASR alternatives:` prompt line.

Spans are the word ranges of an alternative that differ from the 1-best under a word-level
difflib alignment. Punctuation-only differences are dropped; casing and word-boundary
differences are kept (they carry the boundary evidence). Capped at MAX_SPANS.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

PREFIX = "ASR alternatives: "
MAX_SPANS = 6
MAX_SPAN_WORDS = 8


def _sq(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s)


def alt_spans(best: str, alts: list[str], max_spans: int = MAX_SPANS) -> list[str]:
    bw = (best or "").split()
    spans: list[str] = []
    seen: set[str] = set()
    for a in alts:
        aw = (a or "").split()
        if not aw:
            continue
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=bw, b=aw, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            piece = " ".join(aw[j1:j2]).strip()
            if not piece:
                continue
            if len(piece.split()) > MAX_SPAN_WORDS:
                piece = " ".join(piece.split()[:MAX_SPAN_WORDS])
            if _sq(piece) == _sq(" ".join(bw[i1:i2])):
                continue
            if piece in seen:
                continue
            seen.add(piece)
            spans.append(piece)
            if len(spans) >= max_spans:
                return spans
    return spans


def alt_line(best: str, alts: list[str], max_spans: int = MAX_SPANS) -> str:
    sp = alt_spans(best, alts, max_spans)
    return (PREFIX + " | ".join(sp)) if sp else ""


def with_alt_line(text: str, line: str) -> str:
    return f"{text}\n{line}" if line else text


def load_nbest(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as h:
        for line in h:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def line_for(rec: dict | None, hyp_1best: str, max_spans: int = MAX_SPANS) -> str:
    """Alternatives line for a clip, aligned against the 1-best the refiner actually sees."""
    if not rec:
        return ""
    cands = [rec.get("best", "")] + list(rec.get("alts") or [])
    cands = [c for c in cands if c and c.strip() != (hyp_1best or "").strip()]
    return alt_line(hyp_1best, cands, max_spans)
