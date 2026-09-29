"""STEP 2: retrieval v2 variants -- recall@10/@30 on every gate set, plus speed."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
from retrieve2 import Index, Phoneme, squash
from common import pool_big, pool_small
import variants as V

D = Path("/data/phonon_retrieval_v2")
log = lambda s: print(s, flush=True)
T0 = time.perf_counter()
PH = Phoneme()

POOLS = {"big": pool_big(), "small": pool_small()}
IX = {}
for pn, (terms, rank, kind) in POOLS.items():
    IX[(pn, 1)] = Index(terms, rank, kind, use_phonemes=False, nreal=1)
    IX[(pn, 5)] = Index(terms, rank, kind, use_phonemes=True, ph=PH, nreal=5)
    log(f"index {pn}: terms={len(terms)} reals1={len(IX[(pn,1)].reals)} "
        f"reals5={len(IX[(pn,5)].reals)} [{time.perf_counter()-T0:.0f}s]")

SETS = {}
for s in ("new", "old_unseen", "old_seen"):
    rows, texts, golds, pn = V.load_set(s)
    SETS[s] = (rows, texts, golds, pn)
rows, texts, golds = V.load_real(POOLS["big"][0])
SETS["real"] = (rows, texts, golds, "big")
log("real gold pairs=" + str(sum(len(g) for g in golds)) +
    " rows_with_gold=" + str(sum(1 for g in golds if g)))

# ---- pass definitions: (nreal, nmax) -> list of (cfgname, wc, wm, wp)
WGRID = [("c50m50", 0.5, 0.5, 0.0), ("c40m40p20", 0.4, 0.4, 0.2),
         ("c35m35p30", 0.35, 0.35, 0.3), ("c30m30p40", 0.3, 0.3, 0.4),
         ("c25m25p50", 0.25, 0.25, 0.5), ("c40m20p40", 0.4, 0.2, 0.4),
         ("c50m00p50", 0.5, 0.0, 0.5), ("c33m33p33", 1 / 3, 1 / 3, 1 / 3),
         ("c20m20p60", 0.2, 0.2, 0.6)]

PASSES = [(1, 3, [("v1", 0.5, 0.5, 0.0)]),
          (1, 6, [("v1_n6", 0.5, 0.5, 0.0)]),
          (1, 8, [("v1_n8", 0.5, 0.5, 0.0)]),
          (5, 6, [("a_" + n, a, b, c) for n, a, b, c in WGRID]),
          (5, 8, [("a8_" + n, a, b, c) for n, a, b, c in WGRID if n in
                  ("c50m50", "c35m35p30", "c30m30p40")])]

BEST = {}   # (set, cfg) -> best matrix
RES = {}
for sname, (rows, texts, golds, pn) in SETS.items():
    for nreal, nmax, cfgs in PASSES:
        if sname == "real" and nmax > 6:
            continue
        ix = IX[(pn, nreal)]
        b, nk = V.run_cfgs(ix, texts, cfgs, nmax=nmax, log=None)
        for name, M in b.items():
            st, _ = V.recall_stats(M, ix, golds)
            RES[(sname, name)] = st
            BEST[(sname, name)] = M
            log(f"[{sname}] {name:14s} r@10={st['recall@10']:.4f} r@30={st['recall@30']:.4f} "
                f"pairs={st['pairs']} keys={nk} [{time.perf_counter()-T0:.0f}s]")

json.dump({f"{a}|{b}": v for (a, b), v in RES.items()},
          open(D / "step2_recall.json", "w"), indent=1)
np.save(D / "cache_marker.npy", np.zeros(1))
import pickle
with open(D / "best_matrices.pkl", "wb") as f:
    pickle.dump({k: v for k, v in BEST.items()}, f, protocol=4)
log(f"STEP 2a done [{time.perf_counter()-T0:.0f}s]")
