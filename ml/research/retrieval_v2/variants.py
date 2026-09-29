"""Shared driver: build indices, score many weight configs over one cdist pass, report recall."""
from __future__ import annotations
import sys, time
import numpy as np
from rapidfuzz import fuzz, process

sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
from retrieve2 import Index, Phoneme, codes, ngrams, squash  # noqa: E402
from common import read_jsonl, ASR_NEW, ASR_OLD_UNSEEN, ASR_OLD_SEEN, REAL_HOLDOUT  # noqa: E402
from common import pool_big, pool_small  # noqa: E402

_WORDRE = __import__("re").compile(r"[A-Za-z0-9]+(?:[._/'’-][A-Za-z0-9]+)*")


def keymap(texts, nmax):
    key_id, per_text = {}, []
    for t in texts:
        ids = set()
        for g in ngrams(t, nmax):
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
    owners = [[] for _ in keys]
    for ti, ids in enumerate(per_text):
        for i in ids:
            owners[i].append(ti)
    return keys, per_text, owners


def components(ix, sub, need_ph, workers=-1):
    flat = [squash(k) for k in sub]
    ch = process.cdist(flat, ix.r_flat, scorer=fuzz.ratio, dtype=np.uint8, workers=workers)
    kc = [codes(k) for k in sub]
    k1 = [x[0] for x in kc]
    k2 = [x[1] for x in kc]
    mt = process.cdist(k1, ix.r_c1, scorer=fuzz.ratio, dtype=np.uint8, workers=workers)
    np.maximum(mt, process.cdist(k1, ix.r_c2, scorer=fuzz.ratio, dtype=np.uint8, workers=workers), out=mt)
    if k2 != k1:
        np.maximum(mt, process.cdist(k2, ix.r_c1, scorer=fuzz.ratio, dtype=np.uint8, workers=workers), out=mt)
    ph = None
    if need_ph:
        kp = [ix.ph.phrase(k) for k in sub]
        ph = process.cdist(kp, ix.r_ph, scorer=fuzz.ratio, dtype=np.uint8, workers=workers)
    return ch, mt, ph


def run_cfgs(ix, texts, cfgs, nmax, block=2048, workers=-1, log=None):
    """cfgs: list of (name, w_char, w_meta, w_ph). Returns {name: best[n_texts, n_terms] uint8}."""
    keys, per_text, owners = keymap(texts, nmax)
    need_ph = any(c[3] > 0 for c in cfgs)
    best = {c[0]: np.zeros((len(texts), len(ix.terms)), dtype=np.uint8) for c in cfgs}
    t0 = time.perf_counter()
    tc = 0.0
    for s in range(0, len(keys), block):
        sub = keys[s:s + block]
        ta = time.perf_counter()
        ch, mt, ph = components(ix, sub, need_ph, workers=workers)
        tc += time.perf_counter() - ta
        for name, wc, wm, wp in cfgs:
            acc = ch.astype(np.float32) * wc + mt.astype(np.float32) * wm
            if wp > 0:
                acc += ph.astype(np.float32) * wp
            red = np.maximum.reduceat(np.rint(acc).astype(np.uint8), ix.bounds, axis=1)
            B = best[name]
            for li in range(len(sub)):
                row = red[li]
                for ti in owners[s + li]:
                    np.maximum(B[ti], row, out=B[ti])
        if log and (s // block) % 20 == 0:
            log(f"    {min(s+block,len(keys))}/{len(keys)} keys [{time.perf_counter()-t0:.0f}s]")
    if log:
        log(f"    keys={len(keys)} cdist={tc:.1f}s total={time.perf_counter()-t0:.1f}s")
    return best, len(keys)


def ranked(best_row, ix, rank_eps=1e-4, prior=None):
    row = best_row.astype(np.float32) + rank_eps * np.log1p(ix.rank)
    if prior is not None:
        row = row + prior
    return row


def recall_stats(best, ix, golds, ks=(10, 30), rank_eps=1e-4, prior=None):
    """golds: list of sets. Returns dict with recall@k and per-clip gold rank."""
    tindex = {t: i for i, t in enumerate(ix.terms)}
    hits = {k: 0 for k in ks}
    tot = 0
    ranks = []
    kmax = max(ks)
    for ti in range(best.shape[0]):
        g = [tindex[t] for t in golds[ti] if t in tindex]
        if not g:
            ranks.append(None)
            continue
        row = ranked(best[ti], ix, rank_eps, prior)
        order = np.argsort(-row, kind="stable")
        pos = {int(j): r for r, j in enumerate(order[:200])}
        for gi in g:
            tot += 1
            r = pos.get(gi)
            if r is None:
                r = int(np.where(order == gi)[0][0])
            for k in ks:
                if r < k:
                    hits[k] += 1
        ranks.append(pos.get(g[0], None) if len(g) == 1 else None)
    out = {f"recall@{k}": hits[k] / max(1, tot) for k in ks}
    out["pairs"] = tot
    return out, ranks


# ---------------------------------------------------------------- eval sets
def load_set(name):
    if name == "new":
        rows = read_jsonl(ASR_NEW)
        return rows, [r["hyp"] for r in rows], [{r["term"]} for r in rows], "big"
    if name == "old_unseen":
        rows = read_jsonl(ASR_OLD_UNSEEN)
        return rows, [r["hyp"] for r in rows], [{r["term"]} for r in rows], "small"
    if name == "old_seen":
        rows = read_jsonl(ASR_OLD_SEEN)
        return rows, [r["hyp"] for r in rows], [{r["term"]} for r in rows], "small"
    raise ValueError(name)


def load_real(terms):
    """Real-audio holdout: gold = lexicon terms whose squashed form appears in the reference."""
    rows = read_jsonl(REAL_HOLDOUT)
    by_sq = {}
    for t in terms:
        s = squash(t)
        if len(s) >= 3:
            by_sq.setdefault(s, []).append(t)
    golds = []
    for r in rows:
        toks = _WORDRE.findall(r["reference"])
        found = set()
        for n in (1, 2, 3, 4):
            for i in range(len(toks) - n + 1):
                s = squash(" ".join(toks[i:i + n]))
                for t in by_sq.get(s, []):
                    found.add(t)
        golds.append(found)
    return rows, [r["input"] for r in rows], golds
