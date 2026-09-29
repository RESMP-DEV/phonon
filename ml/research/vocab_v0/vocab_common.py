"""Shared vocabulary-prompt helpers for research/vocab_v0."""
from __future__ import annotations

import bisect
import json
import random
import re
from pathlib import Path

SYSTEM_PROMPT = (
    "Rewrite the raw dictation transcript into the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)
VOCAB_PREFIX = "Vocabulary the speaker may use: "

LEXICON_PATH = Path("/data/phonon_synth_v1/lexicon_ranked.jsonl")
HELDOUT_PATH = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")


def _common_english() -> set[str]:
    """asr_errors_v0's English word list, read straight off disk (importing that package
    shadows corrector_v0's `common` module)."""
    p = Path("/data/phonon_asr_errors_v0/english_words.txt")
    if not p.exists():
        print("english_words.txt missing; no common-word filter", flush=True)
        return set()
    return {w.strip().lower() for w in p.read_text(encoding="utf-8").splitlines() if w.strip()}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def system_with_vocab(vocab: list[str] | None) -> str:
    if not vocab:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n" + VOCAB_PREFIX + ", ".join(vocab)


def chat_messages(raw: str, vocab: list[str] | None, target: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": system_with_vocab(vocab)},
        {"role": "user", "content": raw},
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs


class Lexicon:
    def __init__(self, exclude: set[str] | None = None) -> None:
        rows = read_jsonl(LEXICON_PATH)
        excl = {t.lower() for t in (exclude or set())}
        self.rows = []
        seen = set()
        for r in rows:
            term = (r.get("term") or "").strip()
            if not term or term.lower() in seen:
                continue
            seen.add(term.lower())
            self.rows.append(
                {"term": term, "kind": r.get("kind") or "other",
                 "rank_score": float(r.get("rank_score") or 0.0)}
            )
        self.allowed = [r for r in self.rows if r["term"].lower() not in excl]
        self.by_kind: dict[str, list[dict]] = {}
        for r in self.allowed:
            self.by_kind.setdefault(r["kind"], []).append(r)
        # sorted-by-rank views for nearest-neighbour selection
        self.kind_sorted = {
            k: sorted(v, key=lambda r: r["rank_score"]) for k, v in self.by_kind.items()
        }
        self.kind_scores = {k: [r["rank_score"] for r in v] for k, v in self.kind_sorted.items()}
        self.all_sorted = sorted(self.allowed, key=lambda r: r["rank_score"])
        self.all_scores = [r["rank_score"] for r in self.all_sorted]
        self._matcher = None
        self._canon = {r["term"].lower(): r["term"] for r in self.rows}

    # ---- matching lexicon terms inside a target text -------------------
    # Case-sensitive, and a match may not sit inside a longer hyphen/dot/slash token:
    # case-insensitive matching turned ordinary words into "terms" (KERNEL <- kernel,
    # bench <- kernel-bench-multi), which would teach the model to uppercase English.
    def matcher(self):
        if self._matcher is None:
            common = _common_english()
            terms = []
            for r in self.rows:
                t = r["term"]
                if t.isalpha() and t.islower() and t.lower() in common:
                    continue  # plain English word in the lexicon
                terms.append(t)
            terms = sorted(set(terms), key=len, reverse=True)
            self._match_terms = set(terms)
            pat = "|".join(re.escape(t) for t in terms)
            self._matcher = re.compile(
                r"(?<![A-Za-z0-9_./-])(?:" + pat + r")(?![A-Za-z0-9_./-])"
            )
        return self._matcher

    def find_terms(self, text: str, limit: int = 8) -> list[str]:
        out: list[str] = []
        for m in self.matcher().finditer(text or ""):
            canon = m.group(0)
            if canon not in out:
                out.append(canon)
            if len(out) >= limit:
                break
        return out

    def kind_of(self, term: str) -> str:
        if not hasattr(self, "_kinds"):
            self._kinds = {r["term"].lower(): r["kind"] for r in self.rows}
        return self._kinds.get((term or "").lower(), "other")

    # ---- distractor sampling -------------------------------------------
    def sample_distractors(self, rng: random.Random, n: int, kinds: list[str],
                           exclude: set[str]) -> list[str]:
        excl = {t.lower() for t in exclude}
        out: list[str] = []
        tries = 0
        while len(out) < n and tries < n * 40:
            tries += 1
            kind = rng.choice(kinds) if kinds and rng.random() < 0.7 else None
            pool = self.by_kind.get(kind) if kind else None
            if not pool:
                pool = self.allowed
            cand = rng.choice(pool)["term"]
            if cand.lower() in excl:
                continue
            excl.add(cand.lower())
            out.append(cand)
        return out

    def nearest(self, kind: str, rank_score: float, n: int, exclude: set[str]) -> list[str]:
        """n lexicon terms of the same kind with the closest rank_score (global fallback)."""
        excl = {t.lower() for t in exclude}
        out: list[str] = []
        for rows, scores in ((self.kind_sorted.get(kind, []), self.kind_scores.get(kind, [])),
                             (self.all_sorted, self.all_scores)):
            if not rows:
                continue
            i = bisect.bisect_left(scores, rank_score)
            lo, hi = i - 1, i
            while len(out) < n and (lo >= 0 or hi < len(rows)):
                pick = None
                if lo < 0:
                    pick = rows[hi]; hi += 1
                elif hi >= len(rows):
                    pick = rows[lo]; lo -= 1
                elif abs(scores[hi] - rank_score) < abs(rank_score - scores[lo]):
                    pick = rows[hi]; hi += 1
                else:
                    pick = rows[lo]; lo -= 1
                if pick["term"].lower() in excl:
                    continue
                excl.add(pick["term"].lower())
                out.append(pick["term"])
            if len(out) >= n:
                break
        return out[:n]


def heldout_terms() -> dict[str, dict]:
    return {r["term"]: r for r in read_jsonl(HELDOUT_PATH)}


def seeded(*parts: str) -> random.Random:
    import hashlib

    key = "vocabv0|" + "|".join(str(p) for p in parts)
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
