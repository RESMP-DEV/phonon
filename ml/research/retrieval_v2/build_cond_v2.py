"""STEP 3a: rebuild the `retrieved` condition for heldout_new and heldout_old/unseen with v2.

Only `vocab_retrieved` and `retrieved_has_term` change; `vocab_oracle` is copied verbatim from the
v1 condition files so the oracle arm stays identical.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
from retrieve2 import Index, Phoneme
from common import pool_big, pool_small, read_jsonl
import variants as V
from vocab_common import seeded  # noqa: E402

D = Path("/data/phonon_retrieval_v2")
SRC = {"new": (Path("/data/phonon_bigrun_v0/eval_conditions_new.jsonl"), "big", "bigruneval"),
       "unseen": (Path("/data/phonon_term_eval_v1/eval_conditions.jsonl"), "small", "v1eval")}
log = lambda s: print(s, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--nmax", type=int, default=8)
ap.add_argument("--wc", type=float, default=0.35)
ap.add_argument("--wm", type=float, default=0.35)
ap.add_argument("--wp", type=float, default=0.30)
ap.add_argument("--topk", type=int, default=30)
ap.add_argument("--alpha", type=float, default=0.25)
a = ap.parse_args()

t0 = time.perf_counter()
PH = Phoneme()
KLO = json.loads((D / "prior.json").read_text())["kind_lo"]
POOL = {"big": pool_big(), "small": pool_small()}
stats = {}
for name, (src, pn, tag) in SRC.items():
    terms, rank, kind = POOL[pn]
    ix = Index(terms, rank, kind, use_phonemes=True, ph=PH, nreal=5)
    rows = read_jsonl(src)
    best, nk = V.run_cfgs(ix, [r["hyp"] for r in rows], [("v2", a.wc, a.wm, a.wp)], nmax=a.nmax)
    B = best["v2"]
    prior = np.array([a.alpha * KLO.get(k, 0.0) for k in ix.kind], dtype=np.float32)
    st, _ = V.recall_stats(B, ix, [{r["term"]} for r in rows], ks=(10, 30), prior=prior)
    log(f"[{name}] pool={len(terms)} v2 recall@10={st['recall@10']:.4f} "
        f"recall@30={st['recall@30']:.4f} [{time.perf_counter()-t0:.0f}s]")
    out = D / f"cond_{name}_v2.jsonl"
    with out.open("w", encoding="utf-8") as h:
        for ti, r in enumerate(rows):
            row = V.ranked(B[ti], ix, prior=prior)
            idx = np.argpartition(-row, a.topk - 1)[:a.topk]
            idx = idx[np.argsort(-row[idx], kind="stable")]
            top = [ix.terms[j] for j in idx]
            rec = dict(r)
            rec["retrieved_has_term"] = r["term"] in top
            ret = list(top)
            seeded(tag, r["id"]).shuffle(ret)
            rec["vocab_retrieved"] = ret
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats[name] = {"pool": len(terms), "rows": len(rows), **st,
                   "cfg": {"nmax": a.nmax, "wc": a.wc, "wm": a.wm, "wp": a.wp,
                            "alpha": a.alpha}}
    log(f"[{name}] wrote {out}")
(D / "cond_v2_stats.json").write_text(json.dumps(stats, indent=1))
log(f"STEP 3a done [{time.perf_counter()-t0:.0f}s]")
