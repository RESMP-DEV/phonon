"""STEP 2a: shipped two-stage v2 retriever -> top-100 ranked lists per clip, for every eval set.

Lists are saved once; the adaptive-budget sweep (step2_budget.py) is pure python on top.
Sets: heldout_new (15.7k pool), heldout_pool2 (38.2k union pool), real-audio holdout (15.7k),
heldout_old/seen (4.7k, the only tuning set).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
from retrieve2 import Index, Phoneme, ngrams as _ng  # noqa: E402
from common import (read_jsonl, pool_big, pool_small, pool_union, load_pool2_rows,  # noqa: E402
                    ASR_NEW, ASR_OLD_SEEN)
import variants as V  # noqa: E402

D = Path("/data/phonon_index_v0")
D.mkdir(parents=True, exist_ok=True)
PR = json.loads(Path("/data/phonon_retrieval_v2/prior.json").read_text())
KLO, ALPHA = PR["kind_lo"], PR["alpha"]
CAND = 400
TOPN = 100
log = lambda s: print(s, flush=True)
T0 = time.perf_counter()

WANT = sys.argv[1:] or ["old_seen", "new", "real", "pool2"]
PH = Phoneme()


def build(pool_fn):
    terms, rank, kind = pool_fn()
    ix1 = Index(terms, rank, kind, use_phonemes=False, nreal=1)
    ix5 = Index(terms, rank, kind, use_phonemes=True, ph=PH, nreal=5)
    return terms, ix1, ix5


def run_set(name, pool_fn, rows, texts, golds):
    t0 = time.perf_counter()
    terms, ix1, ix5 = build(pool_fn)
    log(f"[{name}] pool={len(terms)} reals5={len(ix5.reals)} rows={len(rows)} "
        f"[{time.perf_counter()-T0:.0f}s]")
    warm = {w for r in ix5.reals for w in r.split()}
    for t in texts:
        warm.update(_ng(t, 1))
    n_new = PH.warm(warm)
    if n_new:
        PH.save()
        ix5.r_ph = [PH.phrase(r) for r in ix5.reals]
        log(f"[{name}] g2p warmed {n_new}")
    b1, _ = V.run_cfgs(ix1, texts, [("s1", 0.5, 0.5, 0.0)], nmax=6)
    log(f"[{name}] stage1 done [{time.perf_counter()-T0:.0f}s]")
    b5, _ = V.run_cfgs(ix5, texts, [("d", 0.4, 0.4, 0.2)], nmax=8)
    log(f"[{name}] stage2 done [{time.perf_counter()-T0:.0f}s]")
    S1, B = b1["s1"], b5["d"]
    prior = np.array([ALPHA * KLO.get(k, 0.0) for k in ix5.kind], dtype=np.float32)
    bonus = 1e-4 * np.log1p(ix5.rank)
    out = []
    for ti in range(len(texts)):
        r1 = S1[ti].astype(np.float32) + bonus
        c = np.argpartition(-r1, CAND - 1)[:CAND]
        r2 = B[ti][c].astype(np.float32) + bonus[c] + prior[c]
        order = c[np.argsort(-r2, kind="stable")][:TOPN]
        sc = (B[ti][order].astype(np.float32) + bonus[order] + prior[order])
        out.append({
            "id": rows[ti].get("id", f"{name}{ti}"),
            "nwords": len(texts[ti].split()),
            "gold": sorted(golds[ti]),
            "n_gold_in_pool": len(golds[ti]),
            "top": [[terms[int(j)], round(float(s), 3)] for j, s in zip(order, sc)],
        })
    p = D / f"lists_{name}.json"
    p.write_text(json.dumps(out))
    log(f"[{name}] wrote {p} ({len(out)} rows) [{time.perf_counter()-t0:.0f}s]")
    del b1, b5, S1, B, ix1, ix5


if "old_seen" in WANT:
    rows = read_jsonl(ASR_OLD_SEEN)
    run_set("old_seen", pool_small, rows, [r["hyp"] for r in rows], [{r["term"]} for r in rows])
if "new" in WANT:
    rows = read_jsonl(ASR_NEW)
    run_set("new", pool_big, rows, [r["hyp"] for r in rows], [{r["term"]} for r in rows])
if "real" in WANT:
    terms, _, _ = pool_big()
    rows, texts, golds = V.load_real(terms)
    run_set("real", pool_big, rows, texts, golds)
if "pool2" in WANT:
    rows = load_pool2_rows()
    run_set("pool2", pool_union, rows, [r["hyp"] for r in rows], [{r["term"]} for r in rows])
log(f"STEP 2a done [{time.perf_counter()-T0:.0f}s]")
