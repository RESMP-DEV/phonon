"""Retrieval v2, two-stage, with the index_v0 adaptive list budget.

Vendored from research/retrieval_v2/retrieve2.py + variants.py and research/index_v0/step2_budget.py.
Pure CPU; the bench runs it in a thread pool while the GPU loads weights.

Defaults are the ones that built the frozen `vocab_retrieved` lists
(research/retrieval_v2/build_cond_v2.py): nmax 8, weights 0.35 char / 0.35 metaphone /
0.30 phoneme, kind log-odds prior at alpha 0.25, rank_eps 1e-4.

Phonemes come from the frozen g2p cache (sets/g2p_cache.json).  g2p_en is optional: if it is
installed, words missing from the cache are computed; if not, they score as empty, which is what
the cached lists already assumed.
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

_WORD = re.compile(r"[A-Za-z0-9]+(?:[._/'’-][A-Za-z0-9]+)*")
_PIECE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[A-Z]+|[a-z]+|[0-9]+")

ARPA = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH "
        "UH UW V W Y Z ZH").split()
_PH2C = {p: chr(65 + i) for i, p in enumerate(ARPA)}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

# adaptive budget (research/index_v0/step2_budget.py), threshold tuned on heldout_old/seen only
BUDGET_MULT, BUDGET_CAP, BUDGET_THRESHOLD = 3, 80, 65


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
        return (_ONES[n // 100] + " hundred"
                + ("" if n % 100 == 0 else " " + num_words(str(n % 100)))).strip()
    if n < 10000:
        return (num_words(str(n // 1000)) + " thousand"
                + ("" if n % 1000 == 0 else " " + num_words(str(n % 1000)))).strip()
    return " ".join(_ONES[int(c)] for c in s)


def squash(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def spaced(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9]+", " ", t or "")).strip()


def split_words(t: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"[^A-Za-z0-9]+", t or ""):
        if part:
            out.extend(m.group(0) for m in _PIECE.finditer(part))
    return out


def realisations(term: str) -> list[str]:
    ws = split_words(term)
    if not ws:
        return [squash(term)] if squash(term) else []
    out = [spaced(term), " ".join(w.lower() for w in ws)]
    r3 = []
    for w in ws:
        if w.isdigit():
            r3.append(" ".join(w))
        elif len(w) == 1 or (w.isupper() and len(w) <= 8):
            r3.append(" ".join(w.lower()))
        else:
            r3.append(w.lower())
    out.append(" ".join(r3))
    out.append(" ".join(num_words(w) if w.isdigit() else w.lower() for w in ws))
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
    a, b = doublemetaphone(spaced(t))
    a = a or squash(t).upper()
    return a, (b or a)


def ngrams(text: str, nmax: int = 8) -> list[str]:
    toks = _WORD.findall(text or "")
    out = []
    for n in range(1, nmax + 1):
        for i in range(0, max(0, len(toks) - n + 1)):
            out.append(" ".join(toks[i:i + n]).lower())
    return out


class Phoneme:
    def __init__(self, cache_path: Path) -> None:
        self.cache: dict[str, str] = {}
        self._phrase_cache: dict[str, str] = {}
        p = Path(cache_path)
        if p.exists():
            self.cache = json.loads(p.read_text())
        self._g2p = None

    def warm(self, words) -> int:
        todo = sorted({w.lower() for w in words if w and w.lower() not in self.cache})
        if not todo:
            return 0
        try:
            from g2p_en import G2p
        except Exception:
            for w in todo:
                self.cache[w] = ""
            return 0
        if self._g2p is None:
            self._g2p = G2p()
        for w in todo:
            try:
                ph = self._g2p(w)
            except Exception:
                ph = []
            self.cache[w] = "".join(_PH2C.get(re.sub(r"\d", "", p), "") for p in ph if p.strip())
        return len(todo)

    def phrase(self, s: str) -> str:
        hit = self._phrase_cache.get(s)
        if hit is None:
            hit = "".join(self.cache.get(w.lower(), "") for w in s.split())
            if len(self._phrase_cache) < (1 << 21):
                self._phrase_cache[s] = hit
        return hit


class Index:
    def __init__(self, terms, rank=None, kind=None, ph: Phoneme | None = None, nreal=5):
        self.terms = list(terms)
        rank = rank or {}
        self.rank = np.array([float(rank.get(t, 0.0)) for t in self.terms], dtype=np.float32)
        self.kind = [(kind or {}).get(t, "") for t in self.terms]
        owner, reals = [], []
        for i, t in enumerate(self.terms):
            for r in realisations(t)[:nreal]:
                owner.append(i)
                reals.append(r)
        self.owner = np.asarray(owner, dtype=np.int32)
        self.reals = reals
        self.bounds = np.searchsorted(self.owner, np.arange(len(self.terms)))
        self.r_flat = [squash(r) for r in reals]
        c = [codes(r) for r in reals]
        self.r_c1 = [x[0] for x in c]
        self.r_c2 = [x[1] for x in c]
        self.ph = ph
        self.use_phonemes = ph is not None
        self.r_ph = [ph.phrase(r) for r in reals] if self.use_phonemes else None

    def words(self):
        for r in self.reals:
            yield from r.split()

    def score_keys(self, keys, w_char, w_meta, w_ph, workers=-1, cols=None):
        r_flat, r_c1, r_c2, r_ph, bounds = self.r_flat, self.r_c1, self.r_c2, self.r_ph, self.bounds
        if cols is not None:
            r_flat = [self.r_flat[j] for j in cols]
            r_c1 = [self.r_c1[j] for j in cols]
            r_c2 = [self.r_c2[j] for j in cols]
            r_ph = [self.r_ph[j] for j in cols] if self.use_phonemes else None
        acc = None
        if w_char > 0:
            acc = process.cdist([squash(k) for k in keys], r_flat, scorer=fuzz.ratio,
                                dtype=np.uint8, workers=workers).astype(np.float32) * w_char
        if w_meta > 0:
            kc = [codes(k) for k in keys]
            k1 = [x[0] for x in kc]
            k2 = [x[1] for x in kc]
            p = process.cdist(k1, r_c1, scorer=fuzz.ratio, dtype=np.uint8, workers=workers)
            np.maximum(p, process.cdist(k1, r_c2, scorer=fuzz.ratio, dtype=np.uint8,
                                        workers=workers), out=p)
            if k2 != k1:
                np.maximum(p, process.cdist(k2, r_c1, scorer=fuzz.ratio, dtype=np.uint8,
                                            workers=workers), out=p)
            pf = p.astype(np.float32) * w_meta
            acc = pf if acc is None else acc + pf
        if w_ph > 0 and self.use_phonemes:
            pp = process.cdist([self.ph.phrase(k) for k in keys], r_ph, scorer=fuzz.ratio,
                               dtype=np.uint8, workers=workers).astype(np.float32) * w_ph
            acc = pp if acc is None else acc + pp
        return np.rint(acc).astype(np.uint8), bounds


class RetrieverV2:
    def __init__(self, index: Index, nmax=8, w_char=0.35, w_meta=0.35, w_ph=0.30,
                 stage1=(0.5, 0.5), rerank=400, kind_prior=None, alpha=0.25, rank_eps=1e-4):
        self.ix = index
        self.nmax = nmax
        self.w = (w_char, w_meta, w_ph)
        self.stage1 = stage1
        self.rerank = rerank
        self.rank_eps = rank_eps
        kp = kind_prior or {}
        self.prior = np.array([alpha * kp.get(k, 0.0) for k in index.kind], dtype=np.float32)

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
        two = self.stage1 is not None and self.rerank > 0
        for s in range(0, len(keys), block):
            sub = keys[s:s + block]
            if two:
                acc, bounds = self.ix.score_keys(sub, self.stage1[0], self.stage1[1], 0.0,
                                                 workers=workers)
            else:
                acc, bounds = self.ix.score_keys(sub, wc, wm, wp, workers=workers)
            red = np.maximum.reduceat(acc, bounds, axis=1)
            for li in range(len(sub)):
                row = red[li]
                for ti in owners[s + li]:
                    np.maximum(best[ti], row, out=best[ti])
            if log and (s // block) % 20 == 0:
                log(f"  retrieval stage1 {min(s + block, len(keys))}/{len(keys)} ngrams")
        if two:
            best = self._rerank(texts, per_text, keys, best, log=log)
        return best

    def _rerank_one(self, sub_keys, cand):
        """The whole per-text body of the rerank: score this text's n-grams against the
        realisations of its candidate terms only. Pure function of (sub_keys, cand)."""
        wc, wm, wp = self.w
        nb = len(self.ix.bounds)
        segs = [(self.ix.bounds[c], self.ix.bounds[c + 1] if c + 1 < nb else
                 len(self.ix.owner)) for c in cand]
        idx = np.concatenate([np.arange(a, b) for a, b in segs])
        sub_bounds = np.cumsum([0] + [b - a for a, b in segs])[:-1]
        acc, _ = self.ix.score_keys(sub_keys, wc, wm, wp, workers=1, cols=idx)
        return np.maximum.reduceat(acc, sub_bounds, axis=1).max(axis=0)

    def _rerank(self, texts, per_text, keys, best, log=None):
        out = np.zeros_like(best)
        kk = min(self.rerank, best.shape[1])
        for ti, ids in enumerate(per_text):
            sub_keys = [keys[i] for i in ids]
            if not sub_keys:
                continue
            cand = np.argpartition(-best[ti].astype(np.float32), kk - 1)[:kk]
            out[ti, cand] = self._rerank_one(sub_keys, cand)
            if log and ti % 500 == 0:
                log(f"  retrieval rerank {ti}/{len(texts)}")
        return out

    def ranked(self, best_row):
        return (best_row.astype(np.float32) + self.rank_eps * np.log1p(self.ix.rank) + self.prior)

    def _topk_local(self, texts, k=100, **kw):
        best = self.best_scores(texts, **kw)
        kk = min(k, len(self.ix.terms))
        out = []
        for ti in range(best.shape[0]):
            row = self.ranked(best[ti])
            idx = np.argpartition(-row, kk - 1)[:kk]
            idx = idx[np.argsort(-row[idx], kind="stable")]
            out.append([(self.ix.terms[j], float(row[j])) for j in idx])
        return out

    def topk(self, texts, k=100, pool=None, chunk_texts=0, mp_workers=1, log=None, **kw):
        """`pool` is a fork pool from `start_pool`; chunks of texts go through the whole
        pipeline in the workers. Bit-identical: a text's score row is a max over its own
        n-grams, so which texts share a batch changes nothing."""
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


def budget(nwords: int, mult=BUDGET_MULT, cap=BUDGET_CAP) -> int:
    return min(cap, max(30, mult * nwords))


def load_lexicon(path):
    terms, rank, kind = [], {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = r["term"]
        terms.append(t)
        rank[t] = float(r.get("rank_score") or 0.0)
        kind[t] = r.get("kind") or ""
    return terms, rank, kind


def gold_terms_in_reference(reference: str, by_squash: dict) -> set:
    toks = _WORD.findall(reference or "")
    found = set()
    for n in (1, 2, 3, 4):
        for i in range(len(toks) - n + 1):
            for t in by_squash.get(squash(" ".join(toks[i:i + n])), []):
                found.add(t)
    return found


def recall_report(rows, tops, set_kind: str, threshold=BUDGET_THRESHOLD, by_squash=None):
    """recall@10 / @30 / @budget over gold (term, row) pairs, plus list-length stats."""
    hits = {10: 0, 30: 0, "budget": 0}
    tot = 0
    lens = []
    n_no_gold = 0
    for r, top in zip(rows, tops):
        if set_kind == "term":
            gold = {r["term"]} if r.get("term") else set()
        else:
            gold = gold_terms_in_reference(r.get("reference", ""), by_squash or {})
        nw = len(str(r.get("input", "")).split())
        b = budget(nw)
        lst_budget = [t for t, s in top[:b] if s >= threshold]
        lens.append(len(lst_budget))
        if not gold:
            n_no_gold += 1
            continue
        s10 = {t for t, _ in top[:10]}
        s30 = {t for t, _ in top[:30]}
        sb = set(lst_budget)
        tot += len(gold)
        hits[10] += len(gold & s10)
        hits[30] += len(gold & s30)
        hits["budget"] += len(gold & sb)
    n = max(1, len(lens))
    return {
        "gold_pairs": tot, "rows_without_gold": n_no_gold,
        "recall@10": hits[10] / max(1, tot),
        "recall@30": hits[30] / max(1, tot),
        "recall@budget": hits["budget"] / max(1, tot),
        "budget_mean_len": sum(lens) / n,
        "budget_max_len": max(lens) if lens else 0,
        "threshold": threshold, "mult": BUDGET_MULT, "cap": BUDGET_CAP,
    }
