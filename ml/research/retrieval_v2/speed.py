"""Dedicated timing, run with the box otherwise idle.

Throughput = clips/s over 500 heldout_new hypotheses with rapidfuzz workers=-1 (128 cores).
Latency  = one clip, workers=1, index already built (the product keeps it resident).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz, process

sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
from retrieve2 import Index, Phoneme, codes, squash
from common import pool_big
import variants as V

D = Path("/data/phonon_retrieval_v2")
log = lambda s: print(s, flush=True)
PH = Phoneme()
terms, rank, kind = pool_big()
rows, texts, golds, _ = V.load_set("new")
t0 = time.perf_counter()
IX1 = Index(terms, rank, kind, use_phonemes=False, nreal=1)
IX5 = Index(terms, rank, kind, use_phonemes=True, ph=PH, nreal=5)
log(f"indices built pool={len(terms)} reals1={len(IX1.reals)} reals5={len(IX5.reals)} "
    f"[{time.perf_counter()-t0:.1f}s]")


def single(ix, texts, nmax, w, workers, block=4096):
    keys, per_text, owners = V.keymap(texts, nmax)
    need_ph = w[2] > 0
    best = np.zeros((len(texts), len(ix.terms)), dtype=np.uint8)
    for s in range(0, len(keys), block):
        sub = keys[s:s + block]
        ch, mt, ph = V.components(ix, sub, need_ph, workers=workers)
        acc = ch.astype(np.float32) * w[0] + mt.astype(np.float32) * w[1]
        if need_ph:
            acc += ph.astype(np.float32) * w[2]
        red = np.maximum.reduceat(np.rint(acc).astype(np.uint8), ix.bounds, axis=1)
        for li in range(len(sub)):
            for ti in owners[s + li]:
                np.maximum(best[ti], red[li], out=best[ti])
    return best


def twostage(ix1, ix5, texts, nmax, w, workers, cand=200, block=4096):
    s1 = single(ix1, texts, nmax, (0.5, 0.5, 0.0), workers, block)
    out = np.zeros_like(s1)
    for ti in range(len(texts)):
        row = s1[ti].astype(np.float32)
        c = np.argpartition(-row, cand - 1)[:cand]
        segs = [(ix5.bounds[j], ix5.bounds[j + 1] if j + 1 < len(ix5.bounds) else len(ix5.owner))
                for j in c]
        idx = np.concatenate([np.arange(a, b) for a, b in segs])
        sb = np.cumsum([0] + [b - a for a, b in segs])[:-1]
        ks = [g for g in V.ngrams(texts[ti], nmax) if len(squash(g)) >= 2]
        rf = [ix5.r_flat[j] for j in idx]
        rc1 = [ix5.r_c1[j] for j in idx]
        rc2 = [ix5.r_c2[j] for j in idx]
        rp = [ix5.r_ph[j] for j in idx]
        kc = [codes(k) for k in ks]
        acc = process.cdist([squash(k) for k in ks], rf, scorer=fuzz.ratio, dtype=np.uint8,
                            workers=1).astype(np.float32) * w[0]
        m = process.cdist([x[0] for x in kc], rc1, scorer=fuzz.ratio, dtype=np.uint8, workers=1)
        np.maximum(m, process.cdist([x[0] for x in kc], rc2, scorer=fuzz.ratio, dtype=np.uint8,
                                    workers=1), out=m)
        acc += m.astype(np.float32) * w[1]
        acc += process.cdist([ix5.ph.phrase(k) for k in ks], rp, scorer=fuzz.ratio,
                             dtype=np.uint8, workers=1).astype(np.float32) * w[2]
        out[ti, c] = np.maximum.reduceat(np.rint(acc).astype(np.uint8), sb, axis=1).max(axis=0)
    return out


CFG = [("v1  (nmax3, joined only, char+metaphone)", lambda t, wk: single(IX1, t, 3, (.5, .5, 0), wk)),
       ("v1_n8 (nmax8, joined only)", lambda t, wk: single(IX1, t, 8, (.5, .5, 0), wk)),
       ("a8 (nmax8, 5 realisations, no phonemes)", lambda t, wk: single(IX5, t, 8, (.5, .5, 0), wk)),
       ("b8/d (nmax8, 5 realisations, +phonemes)", lambda t, wk: single(IX5, t, 8, (.4, .4, .2), wk)),
       ("c two-stage (v1_n8 -> 200 -> b8)", lambda t, wk: twostage(IX1, IX5, t, 8, (.4, .4, .2), wk))]

N = 500
sub = texts[:N]
out = {}
for name, fn in CFG:
    fn(sub[:8], -1)
    t1 = time.perf_counter()
    fn(sub, -1)
    dt = time.perf_counter() - t1
    lat = []
    for i in range(10):
        t2 = time.perf_counter()
        fn([texts[i]], 1)
        lat.append((time.perf_counter() - t2) * 1000)
    lat.sort()
    out[name] = {"clips_per_s_128w": N / dt, "batch_s": dt,
                 "single_clip_ms_1core_median": lat[len(lat) // 2],
                 "single_clip_ms_1core_max": lat[-1]}
    log(f"{name:45s} {N/dt:8.1f} clips/s (128w)  |  1 clip on 1 core "
        f"{lat[len(lat)//2]:8.1f} ms median, {lat[-1]:8.1f} ms max")
(D / "speed.json").write_text(json.dumps(out, indent=1))
log("SPEED done")
