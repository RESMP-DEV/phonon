"""STEP 1: why the v1 retriever misses at 30, on heldout_new and heldout_old/unseen."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
from retrieve2 import Index, Phoneme, realisations, codes, ngrams, squash, spaced
from common import pool_big, pool_small
import variants as V

OUT = Path("/home/user/phonon/research/retrieval_v2")
D = Path("/data/phonon_retrieval_v2")
log = lambda s: print(s, flush=True)
t0 = time.perf_counter()

ph = Phoneme()
pools = {}
for nm, fn in (("big", pool_big), ("small", pool_small)):
    terms, rank, kind = fn()
    pools[nm] = (terms, rank, kind)
    log(f"pool {nm}={len(terms)}")

IX1, IX5 = {}, {}
for nm, (terms, rank, kind) in pools.items():
    IX1[nm] = Index(terms, rank, kind, use_phonemes=False, ph=None, nreal=1)
    IX5[nm] = Index(terms, rank, kind, use_phonemes=True, ph=ph, nreal=5)
    log(f"index {nm}: reals1={len(IX1[nm].reals)} reals5={len(IX5[nm].reals)} "
        f"[{time.perf_counter()-t0:.0f}s]")


def rtype(term, r):
    rs = realisations(term)
    i = rs.index(r)
    ws = r.split()
    single = sum(1 for w in ws if len(w) == 1)
    if i >= 2 and single >= 2:
        return "spelled"
    if i == 0:
        return "joined"
    return "split"


def diagnose(term, hyp, phn):
    """Per-realisation / per-window best ratios for one gold term against one hypothesis."""
    rs = realisations(term)
    toks = V._WORDRE.findall(hyp or "")
    grams = {n: [" ".join(toks[i:i + n]).lower() for i in range(len(toks) - n + 1)]
             for n in range(1, 7)}
    tc = [(r, squash(r), codes(r), phn.phrase(r), rtype(term, r)) for r in rs]
    best = {}
    for n, gs in grams.items():
        for g in gs:
            gs_sq, gc, gp = squash(g), codes(g), phn.phrase(g)
            if len(gs_sq) < 2:
                continue
            for r, rsq, rc, rph, rt in tc:
                ch = fuzz.ratio(gs_sq, rsq)
                mt = max(fuzz.ratio(gc[0], rc[0]), fuzz.ratio(gc[0], rc[1]),
                         fuzz.ratio(gc[1], rc[0]))
                cm = 0.5 * ch + 0.5 * mt
                phr = fuzz.ratio(gp, rph) if (gp and rph) else 0.0
                key = (n, rt)
                if cm > best.get(key, (-1, None, None, None))[0]:
                    best[key] = (cm, phr, g, r)
    return best


def summarise(best):
    def mx(ns, rts):
        c = [v for (n, rt), v in best.items() if n in ns and rt in rts]
        return max(c, key=lambda x: x[0]) if c else (0.0, 0.0, "", "")
    return {
        "cm_1_3": mx({1, 2, 3}, {"joined"}),
        "cm_4_6": mx({4, 5, 6}, {"joined"}),
        "cm_1_3_any": mx({1, 2, 3}, {"joined", "split", "spelled"}),
        "cm_any": mx(set(range(1, 7)), {"joined", "split", "spelled"}),
        "cm_joined": mx(set(range(1, 7)), {"joined"}),
        "cm_split": mx(set(range(1, 7)), {"split"}),
        "cm_spelled": mx(set(range(1, 7)), {"spelled"}),
        "ph_best": max([v[1] for v in best.values()] or [0.0]),
    }


def categorise(s):
    """What would fix this miss, tested in order of how cheap the fix is."""
    b13, b46 = s["cm_1_3"][0], s["cm_4_6"][0]
    j, sp, spl = s["cm_joined"][0], s["cm_split"][0], s["cm_spelled"][0]
    b13any = s["cm_1_3_any"][0]
    if b46 - b13 >= 10:
        return "wrong_ngram_window"
    if spl - max(j, sp) >= 8 and spl - b13any >= 0:
        return "spelled_letters"
    if sp - j >= 8:
        return "split_into_words"
    if s["ph_best"] >= 75 and s["cm_any"][0] < 62:
        return "phoneme_not_metaphone"
    return "hard_acoustic_loss"


results = {}
for setname, poolname in (("new", "big"), ("old_unseen", "small")):
    rows, texts, golds, _ = V.load_set(setname)
    ix = IX1[poolname]
    log(f"[{setname}] rows={len(rows)} pool={len(ix.terms)}")
    best, nkeys = V.run_cfgs(ix, texts, [("v1", 0.5, 0.5, 0.0)], nmax=3, log=log)
    B = best["v1"]
    stats, _ = V.recall_stats(B, ix, golds)
    log(f"[{setname}] v1 baseline {stats}")
    tindex = {t: i for i, t in enumerate(ix.terms)}
    miss = []
    rank_of = []
    for ti, r in enumerate(rows):
        gi = tindex.get(r["term"])
        row = V.ranked(B[ti], ix)
        order = np.argsort(-row, kind="stable")
        pos = int(np.where(order == gi)[0][0])
        rank_of.append(pos)
        if pos >= 30:
            miss.append((ti, r, pos))
    log(f"[{setname}] misses@30 = {len(miss)} ({len(miss)/len(rows):.3f})")
    cats, examples = {}, {}
    rank_axis = {"31_100": 0, "beyond_100": 0}
    per = []
    for ti, r, pos in miss:
        s = summarise(diagnose(r["term"], r["hyp"], ph))
        c = categorise(s)
        cats[c] = cats.get(c, 0) + 1
        rank_axis["31_100" if pos < 100 else "beyond_100"] += 1
        rec = {"id": r["id"], "term": r["term"], "kind": r["kind"], "rank": pos, "cat": c,
               "hyp": r["hyp"],
               "best_1_3": round(s["cm_1_3"][0], 1), "best_4_6": round(s["cm_4_6"][0], 1),
               "joined": round(s["cm_joined"][0], 1), "split": round(s["cm_split"][0], 1),
               "spelled": round(s["cm_spelled"][0], 1), "ph": round(s["ph_best"], 1),
               "best_any": round(s["cm_any"][0], 1),
               "best_span": s["cm_4_6"][2] if s["cm_4_6"][0] > s["cm_1_3"][0] else s["cm_1_3"][2],
               "best_real": s["cm_4_6"][3] if s["cm_4_6"][0] > s["cm_1_3"][0] else s["cm_1_3"][3]}
        per.append(rec)
        examples.setdefault(c, []).append(rec)
    results[setname] = {"rows": len(rows), "pool": len(ix.terms), "baseline": stats,
                        "misses": len(miss), "cats": cats, "rank_axis": rank_axis,
                        "examples": {k: v[:10] for k, v in examples.items()},
                        "per_miss": per,
                        "cat_by_rank": {}}
    cbr = {}
    for rec in per:
        k = rec["cat"]
        cbr.setdefault(k, {"31_100": 0, "beyond_100": 0})
        cbr[k]["31_100" if rec["rank"] < 100 else "beyond_100"] += 1
    results[setname]["cat_by_rank"] = cbr
    np.save(D / f"v1_rank_{setname}.npy", np.array(rank_of, dtype=np.int32))
    log(f"[{setname}] cats={cats} rank_axis={rank_axis} [{time.perf_counter()-t0:.0f}s]")

(D / "misses.json").write_text(json.dumps(results, indent=1))
log(f"STEP 1 done [{time.perf_counter()-t0:.0f}s]")
