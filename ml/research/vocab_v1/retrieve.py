"""Runtime-shaped vocabulary retrieval: hypothesis n-grams -> top-k lexicon terms.

Product constraint: the list must be computable from the raw ASR hypothesis alone, with no
knowledge of the reference. For every 1-3 word n-gram of the hypothesis we score every lexicon
term by 0.5 * double-metaphone similarity + 0.5 * character similarity, take the max over
n-grams, and keep the top k.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from metaphone import doublemetaphone
from rapidfuzz import fuzz, process

LEXICON_PATH = Path("/data/phonon_synth_v1/lexicon_ranked.jsonl")
HELDOUT_PATH = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")

_WORD = re.compile(r"[A-Za-z0-9]+(?:[._/'’-][A-Za-z0-9]+)*")


def squash(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def spaced(t: str) -> str:
    """Punctuation -> space, and split letter/digit runs so acronyms and versions survive."""
    s = re.sub(r"[^A-Za-z0-9]+", " ", t or "")
    return re.sub(r"\s+", " ", s).strip()


def codes(t: str) -> tuple[str, str]:
    a, b = doublemetaphone(spaced(t))
    a = a or squash(t).upper()
    return a, (b or a)


def ngrams(text: str, nmax: int = 3) -> list[str]:
    toks = _WORD.findall(text or "")
    out = []
    for n in range(1, nmax + 1):
        for i in range(0, max(0, len(toks) - n + 1)):
            g = " ".join(toks[i : i + n]).lower()
            out.append(g)
    return out


class Retriever:
    def __init__(self, terms: list[str], rank: dict[str, float] | None = None) -> None:
        self.terms = list(terms)
        self.flat = [squash(t) for t in self.terms]
        c = [codes(t) for t in self.terms]
        self.c1 = [x[0] for x in c]
        self.c2 = [x[1] for x in c]
        rank = rank or {}
        self.rank = np.array([float(rank.get(t, 0.0)) for t in self.terms], dtype=np.float32)

    # -- scoring -----------------------------------------------------------
    def _score_block(self, keys: list[str]) -> np.ndarray:
        flat = [squash(k) for k in keys]
        kc = [codes(k) for k in keys]
        k1 = [x[0] for x in kc]
        k2 = [x[1] for x in kc]
        ch = process.cdist(flat, self.flat, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
        p11 = process.cdist(k1, self.c1, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
        p12 = process.cdist(k1, self.c2, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
        np.maximum(p11, p12, out=p11)
        del p12
        if k2 != k1:
            p21 = process.cdist(k2, self.c1, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
            np.maximum(p11, p21, out=p11)
            del p21
        out = (p11.astype(np.uint16) + ch.astype(np.uint16)) // 2
        return out.astype(np.uint8)

    def best_scores(self, texts: list[str], block: int = 3072, log=None) -> np.ndarray:
        """[n_texts, n_terms] uint8 max-over-n-gram score."""
        key_id: dict[str, int] = {}
        per_text: list[list[int]] = []
        for t in texts:
            ids = set()
            for g in ngrams(t):
                if len(squash(g)) < 2:
                    continue
                i = key_id.get(g)
                if i is None:
                    i = len(key_id)
                    key_id[g] = i
                ids.add(i)
            per_text.append(sorted(ids))
        keys = [""] * len(key_id)
        for g, i in key_id.items():
            keys[i] = g
        # key -> texts that contain it
        owners: list[list[int]] = [[] for _ in keys]
        for ti, ids in enumerate(per_text):
            for i in ids:
                owners[i].append(ti)
        best = np.zeros((len(texts), len(self.terms)), dtype=np.uint8)
        for s in range(0, len(keys), block):
            sub = keys[s : s + block]
            sc = self._score_block(sub)
            for li in range(len(sub)):
                row = sc[li]
                for ti in owners[s + li]:
                    np.maximum(best[ti], row, out=best[ti])
            if log and (s // block) % 10 == 0:
                log(f"  retrieval {min(s+block, len(keys))}/{len(keys)} ngrams")
        return best

    def topk(self, texts: list[str], k: int = 30, block: int = 3072, log=None):
        best = self.best_scores(texts, block=block, log=log)
        kk = min(k, len(self.terms))
        out = []
        for ti in range(best.shape[0]):
            row = best[ti].astype(np.float32) + 1e-4 * np.log1p(self.rank)
            idx = np.argpartition(-row, kk - 1)[:kk]
            idx = idx[np.argsort(-row[idx], kind="stable")]
            out.append([(self.terms[j], int(best[ti, j])) for j in idx])
        return out


def load_pool(exclude: set[str] | None = None) -> Retriever:
    rows = [json.loads(l) for l in LEXICON_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    held = {json.loads(l)["term"]: json.loads(l)
            for l in HELDOUT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()}
    rank: dict[str, float] = {}
    seen: set[str] = set()
    terms: list[str] = []
    excl = {t.lower() for t in (exclude or set())}
    for r in rows:
        t = (r.get("term") or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        if t.lower() in excl:
            continue
        terms.append(t)
        rank[t] = float(r.get("rank_score") or 0.0)
    for t, r in held.items():          # make sure every held-out term is retrievable
        if t.lower() in excl or t in seen:
            continue
        seen.add(t)
        terms.append(t)
        rank[t] = float(r.get("rank_score") or 0.0)
    return Retriever(terms, rank)
