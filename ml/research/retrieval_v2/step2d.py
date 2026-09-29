"""Two-stage recall with a cheap stage 1, from the cached score matrices."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
from retrieve2 import Index, Phoneme
from common import pool_big, pool_small
import variants as V

D = Path("/data/phonon_retrieval_v2")
log = lambda s: print(s, flush=True)
PH = Phoneme()
POOLS = {"big": pool_big(), "small": pool_small()}
IX = {pn: Index(t, r, k, use_phonemes=True, ph=PH, nreal=5) for pn, (t, r, k) in POOLS.items()}
BEST = pickle.load(open(D / "best_matrices.pkl", "rb"))
RES = json.loads((D / "step2_recall.json").read_text())
PR = json.loads((D / "prior.json").read_text())
KLO, ALPHA = PR["kind_lo"], PR["alpha"]


def two_stage(s1, s2, ix, golds, prior, cand, ks=(10, 30)):
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
        r2 = s2[ti][c].astype(np.float32) + bonus[c] + prior[c]
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
    rows, texts, golds, pn = V.load_set(sname)
    ix = IX[pn]
    prior = np.array([ALPHA * KLO.get(k, 0.0) for k in ix.kind], dtype=np.float32)
    for s1n in ("v1", "v1_n6", "v1_n8", "a_c50m50"):
        for cand in (200, 400):
            if (sname, s1n) not in BEST:
                continue
            st = two_stage(BEST[(sname, s1n)], BEST[(sname, "a8_c40m40p20")], ix, golds, prior, cand)
            nm = f"c_2stage{cand}_{s1n}"
            RES[f"{sname}|{nm}"] = st
            log(f"[{sname}] {nm:22s} r@10={st['recall@10']:.4f} r@30={st['recall@30']:.4f}")
(D / "step2_recall.json").write_text(json.dumps(RES, indent=1))
log("STEP 2d done")
