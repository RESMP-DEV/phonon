"""STEP 2c: finish the weight grid at nmax=8, then two-stage (c) and the kind/rank prior (d)."""
from __future__ import annotations
import json, math, pickle, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
from retrieve2 import Index, Phoneme
from common import pool_big, pool_small
import variants as V

D = Path("/data/phonon_retrieval_v2")
log = lambda s: print(s, flush=True)
T0 = time.perf_counter()
PH = Phoneme()
POOLS = {"big": pool_big(), "small": pool_small()}
IX = {pn: Index(t, r, k, use_phonemes=True, ph=PH, nreal=5) for pn, (t, r, k) in POOLS.items()}
SETS = {}
for s in ("new", "old_unseen", "old_seen"):
    rows, texts, golds, pn = V.load_set(s)
    SETS[s] = (rows, texts, golds, pn)

RES = json.loads((D / "step2_recall.json").read_text())
BEST = pickle.load(open(D / "best_matrices.pkl", "rb"))

EXTRA = [("a8_c40m40p20", 0.4, 0.4, 0.2), ("a8_c45m45p10", 0.45, 0.45, 0.1)]
for sname, (rows, texts, golds, pn) in SETS.items():
    ix = IX[pn]
    b, nk = V.run_cfgs(ix, texts, EXTRA, nmax=8)
    for name, M in b.items():
        st, _ = V.recall_stats(M, ix, golds)
        RES[f"{sname}|{name}"] = st
        BEST[(sname, name)] = M
        log(f"[{sname}] {name:14s} r@10={st['recall@10']:.4f} r@30={st['recall@30']:.4f} "
            f"[{time.perf_counter()-T0:.0f}s]")

# ---------------------------------------------------------------- (c) two-stage
def two_stage(s1, s2, ix, golds, cand=200, ks=(10, 30)):
    tindex = {t: i for i, t in enumerate(ix.terms)}
    hits = {k: 0 for k in ks}
    tot = 0
    bonus = 1e-4 * np.log1p(ix.rank)
    for ti in range(s1.shape[0]):
        g = [tindex[t] for t in golds[ti] if t in tindex]
        if not g:
            continue
        r1 = s1[ti].astype(np.float32) + bonus
        c = np.argpartition(-r1, cand - 1)[:cand]
        r2 = s2[ti][c].astype(np.float32) + bonus[c]
        order = c[np.argsort(-r2, kind="stable")]
        pos = {int(j): r for r, j in enumerate(order)}
        for gi in g:
            tot += 1
            r = pos.get(gi, 10 ** 6)
            for k in ks:
                if r < k:
                    hits[k] += 1
    return {f"recall@{k}": hits[k] / max(1, tot) for k in ks} | {"pairs": tot}


for sname in ("old_seen", "old_unseen", "new"):
    rows, texts, golds, pn = SETS[sname]
    ix = IX[pn]
    for s1n, s2n in (("a_c50m50", "a_c40m40p20"), ("a_c50m50", "a_c30m30p40"),
                     ("v1", "a8_c40m40p20")):
        if (sname, s1n) not in BEST or (sname, s2n) not in BEST:
            continue
        st = two_stage(BEST[(sname, s1n)], BEST[(sname, s2n)], ix, golds)
        nm = f"c_2stage200_{s1n}__{s2n}"
        RES[f"{sname}|{nm}"] = st
        log(f"[{sname}] {nm:34s} r@10={st['recall@10']:.4f} r@30={st['recall@30']:.4f}")

# ---------------------------------------------------------------- (d) kind / rank prior
rows_s, texts_s, golds_s, pn_s = SETS["old_seen"]
ix_s = IX[pn_s]
gold_kinds = {}
for g in golds_s:
    for t in g:
        i = ix_s.terms.index(t) if t in ix_s.terms else None
pool_kind_counts = {}
for k in ix_s.kind:
    pool_kind_counts[k] = pool_kind_counts.get(k, 0) + 1
gk = {}
for g in golds_s:
    for t in g:
        kk = POOLS[pn_s][2].get(t, "")
        gk[kk] = gk.get(kk, 0) + 1
ntot = sum(pool_kind_counts.values())
gtot = sum(gk.values())
kind_lo = {k: math.log(((gk.get(k, 0) + 1) / (gtot + len(pool_kind_counts)))
                       / ((v + 1) / (ntot + len(pool_kind_counts))))
           for k, v in pool_kind_counts.items()}
log("kind log-odds (old_seen): " + json.dumps({k: round(v, 2) for k, v in
                                               sorted(kind_lo.items(), key=lambda x: -x[1])}))

BASE = "a8_c40m40p20"
grid = [(a, b) for a in (0.0, 0.25, 0.5, 1.0, 2.0) for b in (1e-4, 0.05, 0.15, 0.3)]
best_ab, best_r = None, -1
for a, b in grid:
    prior_s = np.array([a * kind_lo.get(k, 0.0) for k in ix_s.kind], dtype=np.float32)
    st, _ = V.recall_stats(BEST[("old_seen", BASE)], ix_s, golds_s, rank_eps=b, prior=prior_s)
    if st["recall@30"] > best_r:
        best_r, best_ab = st["recall@30"], (a, b)
    log(f"  prior alpha={a} rank_eps={b} old_seen r@30={st['recall@30']:.4f} "
        f"r@10={st['recall@10']:.4f}")
log(f"prior picked alpha={best_ab[0]} rank_eps={best_ab[1]} old_seen r@30={best_r:.4f}")
a, b = best_ab
for sname in ("old_seen", "old_unseen", "new"):
    rows, texts, golds, pn = SETS[sname]
    ix = IX[pn]
    prior = np.array([a * kind_lo.get(k, 0.0) for k in ix.kind], dtype=np.float32)
    st, _ = V.recall_stats(BEST[(sname, BASE)], ix, golds, rank_eps=b, prior=prior)
    RES[f"{sname}|d_prior_{BASE}"] = st
    log(f"[{sname}] d_prior_{BASE:14s} r@10={st['recall@10']:.4f} r@30={st['recall@30']:.4f}")

(D / "step2_recall.json").write_text(json.dumps(RES, indent=1))
(D / "prior.json").write_text(json.dumps({"alpha": a, "rank_eps": b, "kind_lo": kind_lo}, indent=1))
with open(D / "best_matrices.pkl", "wb") as f:
    pickle.dump(BEST, f, protocol=4)
log(f"STEP 2c done [{time.perf_counter()-T0:.0f}s]")
