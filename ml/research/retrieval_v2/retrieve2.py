"""retrieval v2: sub-word realisations + phoneme scoring + two-stage rerank.

Every lexicon term is expanded into several *spoken realisations* (joined, spaced sub-words,
spelled-letter, digits-as-words).  Each realisation carries three keys: a squashed character
string, a double-metaphone code pair, and an ARPAbet phoneme string packed one char per phone.
Hypothesis n-grams (1..NMAX words) are scored against every realisation with rapidfuzz.cdist and
max-reduced back to the term.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from metaphone import doublemetaphone
from rapidfuzz import fuzz, process

CACHE_DIR = Path("/data/phonon_retrieval_v2")
G2P_CACHE = CACHE_DIR / "g2p_cache.json"

_WORD = re.compile(r"[A-Za-z0-9]+(?:[._/'’-][A-Za-z0-9]+)*")
_PIECE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[A-Z]+|[a-z]+|[0-9]+")

ARPA = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH "
        "UH UW V W Y Z ZH").split()
_PH2C = {p: chr(65 + i) for i, p in enumerate(ARPA)}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def num_words(s: str) -> str:
    try:
        n = int(s)
    except ValueError:
        return s
    if n < 20:
        return _ONES[n]
    if n < 100:
        return (_TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])).strip()
    if n < 1000:
        return (_ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(str(n % 100)))).strip()
    if n < 10000:
        return (num_words(str(n // 1000)) + " thousand"
                + ("" if n % 1000 == 0 else " " + num_words(str(n % 1000)))).strip()
    return " ".join(_ONES[int(c)] for c in s)


def squash(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def spaced(t: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", " ", t or "")
    return re.sub(r"\s+", " ", s).strip()


def split_words(t: str) -> list[str]:
    """camelCase / snake_case / kebab / dotted / digit boundaries -> sub-words."""
    out: list[str] = []
    for part in re.split(r"[^A-Za-z0-9]+", t or ""):
        if not part:
            continue
        out.extend(m.group(0) for m in _PIECE.finditer(part))
    return out


def realisations(term: str) -> list[str]:
    """Spoken realisations of a term, lowercase space-separated words."""
    ws = split_words(term)
    if not ws:
        return [squash(term)] if squash(term) else []
    out: list[str] = []
    out.append(spaced(term))                       # R1 whole term as one token run
    out.append(" ".join(w.lower() for w in ws))    # R2 sub-words spoken
    # R3 spelled: short all-caps chunks and lone letters become letter sequences
    r3 = []
    for w in ws:
        if w.isdigit():
            r3.append(" ".join(w))
        elif len(w) == 1 or (w.isupper() and len(w) <= 8):
            r3.append(" ".join(w.lower()))
        else:
            r3.append(w.lower())
    out.append(" ".join(r3))
    # R4 digits spoken as words
    r4 = [num_words(w) if w.isdigit() else w.lower() for w in ws]
    out.append(" ".join(r4))
    # R5 everything spelled, for short terms
    sq = squash(term)
    if 0 < len(sq) <= 8:
        out.append(" ".join(sq))
    seen, uniq = set(), []
    for r in out:
        r = re.sub(r"\s+", " ", r).strip()
        if r and r.lower() not in seen:
            seen.add(r.lower())
            uniq.append(r)
    return uniq


@lru_cache(maxsize=1 << 20)
def codes(t: str) -> tuple[str, str]:
    """Memoised (2026-09-18 perf pass): the two-stage path recomputes double-metaphone for
    the same n-gram once in stage 1 and again once per owning text in the rerank loop.
    Identical results; measured 81.6-83.6 -> 86.7-92.0 clips/s on 1k bigrun hypotheses."""
    a, b = doublemetaphone(spaced(t))
    a = a or squash(t).upper()
    return a, (b or a)


def ngrams(text: str, nmax: int = 3) -> list[str]:
    toks = _WORD.findall(text or "")
    out = []
    for n in range(1, nmax + 1):
        for i in range(0, max(0, len(toks) - n + 1)):
            out.append(" ".join(toks[i:i + n]).lower())
    return out


# --------------------------------------------------------------------- phonemes
class Phoneme:
    """Word-level g2p cache; phrase phonemes = concatenation of word phonemes."""

    def __init__(self, cache_path: Path = G2P_CACHE) -> None:
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self._phrase_cache: dict[str, str] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text())
        self._g2p = None

    def _ensure(self):
        if self._g2p is None:
            from g2p_en import G2p
            self._g2p = G2p()
        return self._g2p

    def warm(self, words) -> int:
        todo = sorted({w.lower() for w in words if w and w.lower() not in self.cache})
        if not todo:
            return 0
        g = self._ensure()
        for w in todo:
            try:
                ph = g(w)
            except Exception:
                ph = []
            self.cache[w] = "".join(_PH2C.get(re.sub(r"\d", "", p), "") for p in ph if p.strip())
        return len(todo)

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))

    def phrase(self, s: str) -> str:
        hit = self._phrase_cache.get(s)
        if hit is None:
            hit = "".join(self.cache.get(w.lower(), "") for w in s.split())
            if len(self._phrase_cache) < (1 << 21):
                self._phrase_cache[s] = hit
        return hit


# --------------------------------------------------------------------- index
class Index:
    """Realisation-level index over a term pool."""

    def __init__(self, terms, rank=None, kind=None, use_phonemes=True, ph: Phoneme | None = None,
                 nreal=5):
        self.terms = list(terms)
        rank = rank or {}
        self.rank = np.array([float(rank.get(t, 0.0)) for t in self.terms], dtype=np.float32)
        self.kind = [(kind or {}).get(t, "") for t in self.terms]
        owner, reals = [], []
        for i, t in enumerate(self.terms):
            rs = realisations(t)[:nreal]
            for r in rs:
                owner.append(i)
                reals.append(r)
        self.owner = np.asarray(owner, dtype=np.int32)
        self.reals = reals
        # group boundaries for reduceat (owner is non-decreasing by construction)
        self.bounds = np.searchsorted(self.owner, np.arange(len(self.terms)))
        self.r_flat = [squash(r) for r in reals]
        c = [codes(r) for r in reals]
        self.r_c1 = [x[0] for x in c]
        self.r_c2 = [x[1] for x in c]
        self.ph = ph
        self.use_phonemes = use_phonemes and ph is not None
        self.r_ph = [ph.phrase(r) for r in reals] if self.use_phonemes else None

    def words(self):
        for r in self.reals:
            for w in r.split():
                yield w

    # ---- scoring ----
    def score_keys(self, keys, w_char=0.5, w_meta=0.5, w_ph=0.0, workers=-1):
        """[len(keys), n_terms] uint8 scores."""
        flat = [squash(k) for k in keys]
        acc = None
        if w_char > 0:
            ch = process.cdist(flat, self.r_flat, scorer=fuzz.ratio, dtype=np.uint8,
                               workers=workers).astype(np.float32)
            acc = ch * w_char
        if w_meta > 0:
            kc = [codes(k) for k in keys]
            k1 = [x[0] for x in kc]
            k2 = [x[1] for x in kc]
            p = process.cdist(k1, self.r_c1, scorer=fuzz.ratio, dtype=np.uint8, workers=workers)
            p2 = process.cdist(k1, self.r_c2, scorer=fuzz.ratio, dtype=np.uint8, workers=workers)
            np.maximum(p, p2, out=p)
            del p2
            if k2 != k1:
                p3 = process.cdist(k2, self.r_c1, scorer=fuzz.ratio, dtype=np.uint8, workers=workers)
                np.maximum(p, p3, out=p)
                del p3
            pf = p.astype(np.float32) * w_meta
            acc = pf if acc is None else acc + pf
        if w_ph > 0 and self.use_phonemes:
            kp = [self.ph.phrase(k) for k in keys]
            pp = process.cdist(kp, self.r_ph, scorer=fuzz.ratio, dtype=np.uint8,
                               workers=workers).astype(np.float32) * w_ph
            acc = pp if acc is None else acc + pp
        acc = np.rint(acc).astype(np.uint8)
        # max over realisations of each term
        return np.maximum.reduceat(acc, self.bounds, axis=1)


# --------------------------------------------------------------- multiprocess rerank
_MP_SELF = None          # set in the parent, inherited by fork; never pickled


def _mp_chunk(payload):
    """One chunk of texts in a forked worker: the whole pipeline, top-k lists back."""
    lo, texts, k, workers = payload
    return lo, _MP_SELF._topk_local(texts, k=k, workers=workers)


def auto_procs(procs: int = 0) -> int:
    """0 = auto. A quarter of the threads: the rerank's rapidfuzz calls are single-threaded,
    but every worker copy-on-writes part of the index, and this box has 128 threads / 60 GB."""
    if procs and procs > 0:
        return int(procs)
    return max(1, min(32, (os.cpu_count() or 4) // 4))


def start_pool(retriever, procs: int):
    """A fork pool that inherits the built index instead of pickling it.

    Fork from the MAIN thread, before the GPU backend loads. Forking out of a worker thread
    while another thread initialises CUDA is how a child ends up holding a lock that nobody
    will ever release.
    """
    global _MP_SELF
    procs = auto_procs(procs)
    if procs <= 1:
        return None
    import multiprocessing as mp

    _MP_SELF = retriever
    return mp.get_context("fork").Pool(procs)


class RetrieverV2:
    def __init__(self, index: Index, nmax=6, w_char=0.5, w_meta=0.5, w_ph=0.0,
                 stage1=None, rerank=0, kind_prior=None, rank_eps=1e-4):
        self.ix = index
        self.nmax = nmax
        self.w = (w_char, w_meta, w_ph)
        self.stage1 = stage1          # (w_char, w_meta) cheap weights, or None
        self.rerank = rerank          # candidates kept from stage 1 (0 = single stage)
        self.kind_prior = kind_prior or {}
        self.rank_eps = rank_eps
        self.prior = np.zeros(len(index.terms), dtype=np.float32)
        if kind_prior:
            self.prior = np.array([self.kind_prior.get(k, 0.0) for k in index.kind],
                                  dtype=np.float32)

    def _keymap(self, texts):
        key_id: dict[str, int] = {}
        per_text = []
        for t in texts:
            ids = set()
            for g in ngrams(t, self.nmax):
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
        return keys, per_text

    def best_scores(self, texts, block=2048, workers=-1, log=None):
        keys, per_text = self._keymap(texts)
        owners = [[] for _ in keys]
        for ti, ids in enumerate(per_text):
            for i in ids:
                owners[i].append(ti)
        best = np.zeros((len(texts), len(self.ix.terms)), dtype=np.uint8)
        wc, wm, wp = self.w
        s1 = self.stage1
        for s in range(0, len(keys), block):
            sub = keys[s:s + block]
            if s1 is None or self.rerank <= 0:
                sc = self.ix.score_keys(sub, wc, wm, wp, workers=workers)
            else:
                sc = self.ix.score_keys(sub, s1[0], s1[1], 0.0, workers=workers)
            for li in range(len(sub)):
                row = sc[li]
                for ti in owners[s + li]:
                    np.maximum(best[ti], row, out=best[ti])
            if log and (s // block) % 20 == 0:
                log(f"  stage1 {min(s+block, len(keys))}/{len(keys)} ngrams")
        if s1 is not None and self.rerank > 0:
            best = self._rerank(texts, per_text, keys, best, workers=workers, log=log)
        return best

    def _rerank(self, texts, per_text, keys, best, workers=-1, log=None):
        out = np.zeros_like(best)
        kk = min(self.rerank, best.shape[1])
        for ti, ids in enumerate(per_text):
            sub_keys = [keys[i] for i in ids]
            if not sub_keys:
                continue
            cand = np.argpartition(-best[ti].astype(np.float32), kk - 1)[:kk]
            out[ti, cand] = self._rerank_one(sub_keys, cand)
            if log and ti % 500 == 0:
                log(f"  rerank {ti}/{len(texts)}")
        return out

    def _rerank_one(self, sub_keys, cand):
        wc, wm, wp = self.w
        # realisation slice for the candidate terms
        segs = [(self.ix.bounds[c], self.ix.bounds[c + 1] if c + 1 < len(self.ix.bounds)
                 else len(self.ix.owner)) for c in cand]
        idx = np.concatenate([np.arange(a, b) for a, b in segs])
        sub_bounds = np.cumsum([0] + [b - a for a, b in segs])[:-1]
        acc = None
        if wc > 0:
            ch = process.cdist([squash(k) for k in sub_keys],
                               [self.ix.r_flat[j] for j in idx], scorer=fuzz.ratio,
                               dtype=np.uint8, workers=1).astype(np.float32)
            acc = ch * wc
        if wm > 0:
            kc = [codes(k) for k in sub_keys]
            p = process.cdist([x[0] for x in kc], [self.ix.r_c1[j] for j in idx],
                              scorer=fuzz.ratio, dtype=np.uint8, workers=1)
            p2 = process.cdist([x[0] for x in kc], [self.ix.r_c2[j] for j in idx],
                               scorer=fuzz.ratio, dtype=np.uint8, workers=1)
            np.maximum(p, p2, out=p)
            pf = p.astype(np.float32) * wm
            acc = pf if acc is None else acc + pf
        if wp > 0 and self.ix.use_phonemes:
            pp = process.cdist([self.ix.ph.phrase(k) for k in sub_keys],
                               [self.ix.r_ph[j] for j in idx], scorer=fuzz.ratio,
                               dtype=np.uint8, workers=1).astype(np.float32) * wp
            acc = pp if acc is None else acc + pp
        acc = np.rint(acc).astype(np.uint8)
        return np.maximum.reduceat(acc, sub_bounds, axis=1).max(axis=0)

    def _topk_local(self, texts, k=30, block=2048, workers=-1, log=None):
        best = self.best_scores(texts, block=block, workers=workers, log=log)
        kk = min(k, len(self.ix.terms))
        out = []
        for ti in range(best.shape[0]):
            row = (best[ti].astype(np.float32) + self.rank_eps * np.log1p(self.ix.rank)
                   + self.prior)
            idx = np.argpartition(-row, kk - 1)[:kk]
            idx = idx[np.argsort(-row[idx], kind="stable")]
            out.append([(self.ix.terms[j], float(row[j])) for j in idx])
        return out

    def topk(self, texts, k=30, pool=None, chunk_texts=0, mp_workers=1, log=None, **kw):
        """`pool` is a fork pool from `start_pool`: chunks of texts run the whole pipeline in
        the workers. A text's score row is a max over its own n-grams, so the split changes
        nothing and the output is bit-identical."""
        if pool is None:
            return self._topk_local(texts, k=k, log=log, **kw)
        n = len(texts)
        nproc = getattr(pool, "_processes", 8)
        cs = chunk_texts or max(8, -(-n // (4 * nproc)))
        jobs = [(i, texts[i:i + cs], k, mp_workers) for i in range(0, n, cs)]
        out: list = [None] * n
        done = 0
        for lo, part in pool.imap_unordered(_mp_chunk, jobs):
            out[lo:lo + len(part)] = part
            done += len(part)
            if log and done % 2000 < cs:
                log(f"  retrieval {done}/{n} (mp)")
        return out
