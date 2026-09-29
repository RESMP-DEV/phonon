"""STEP 4: retrieval v2 over the 38,253-term union pool, on heldout_pool2."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
from retrieve2 import Index, Phoneme
from common import pool_union, load_pool2_rows, read_jsonl
import variants as V
from vocab_common import seeded  # noqa: E402

D = Path("/data/phonon_retrieval_v2")
SRC = Path("/data/phonon_pool2_v0/eval_conditions_pool2.jsonl")
log = lambda s: print(s, flush=True)
T0 = time.perf_counter()

PR = json.loads((D / "prior.json").read_text())
KLO, ALPHA = PR["kind_lo"], PR["alpha"]
PH = Phoneme()
terms, rank, kind = pool_union()
log(f"union pool={len(terms)}")
rows = load_pool2_rows()
texts = [r["hyp"] for r in rows]
golds = [{r["term"]} for r in rows]
log(f"heldout_pool2 rows={len(rows)}")

ix1 = Index(terms, rank, kind, use_phonemes=False, nreal=1)
ix5 = Index(terms, rank, kind, use_phonemes=True, ph=PH, nreal=5)
log(f"reals1={len(ix1.reals)} reals5={len(ix5.reals)} [{time.perf_counter()-T0:.0f}s]")
from retrieve2 import ngrams as _ng
warm_words = {w for r in ix5.reals for w in r.split()}
for _t in texts:
    warm_words.update(_ng(_t, 1))
n_new = PH.warm(warm_words)
if n_new:
    PH.save()
    ix5.r_ph = [PH.phrase(r) for r in ix5.reals]
    log(f"g2p warmed {n_new} new pool-2 words [{time.perf_counter()-T0:.0f}s]")

RES = {}
b1, _ = V.run_cfgs(ix1, texts, [("v1", 0.5, 0.5, 0.0)], nmax=3)
st, _ = V.recall_stats(b1["v1"], ix1, golds)
RES["v1"] = st
log(f"v1        r@10={st['recall@10']:.4f} r@30={st['recall@30']:.4f} "
    f"[{time.perf_counter()-T0:.0f}s]")

b1b, _ = V.run_cfgs(ix1, texts, [("v1_n6", 0.5, 0.5, 0.0)], nmax=6)
st, _ = V.recall_stats(b1b["v1_n6"], ix1, golds)
RES["v1_n6"] = st
log(f"v1_n6     r@10={st['recall@10']:.4f} r@30={st['recall@30']:.4f} "
    f"[{time.perf_counter()-T0:.0f}s]")

b5, _ = V.run_cfgs(ix5, texts, [("d", 0.4, 0.4, 0.2)], nmax=8, log=log)
B = b5["d"]
prior = np.array([ALPHA * KLO.get(k, 0.0) for k in ix5.kind], dtype=np.float32)
st, _ = V.recall_stats(B, ix5, golds, prior=prior)
RES["d_prior_a8_c40m40p20"] = st
log(f"d(full)   r@10={st['recall@10']:.4f} r@30={st['recall@30']:.4f} "
    f"[{time.perf_counter()-T0:.0f}s]")

# two-stage: v1_n6 -> top 400 -> rerank by d
S1 = b1b["v1_n6"]
CAND = 400
bonus = 1e-4 * np.log1p(ix5.rank)
tindex = {t: i for i, t in enumerate(ix5.terms)}
hits = {10: 0, 30: 0}
top30 = []
for ti in range(len(texts)):
    r1 = S1[ti].astype(np.float32) + bonus
    c = np.argpartition(-r1, CAND - 1)[:CAND]
    r2 = B[ti][c].astype(np.float32) + bonus[c] + prior[c]
    order = c[np.argsort(-r2, kind="stable")]
    gi = tindex[rows[ti]["term"]]
    pos = {int(j): r for r, j in enumerate(order[:30])}
    r = pos.get(gi, 10 ** 6)
    for k in (10, 30):
        if r < k:
            hits[k] += 1
    top30.append([ix5.terms[j] for j in order[:30]])
RES["c_2stage400_v1_n6"] = {"recall@10": hits[10] / len(texts),
                            "recall@30": hits[30] / len(texts), "pairs": len(texts)}
log(f"d(2stage) r@10={hits[10]/len(texts):.4f} r@30={hits[30]/len(texts):.4f} "
    f"[{time.perf_counter()-T0:.0f}s]")

# conditions: keep vocab_oracle verbatim, replace vocab_retrieved with the two-stage top 30
src = {r["id"]: r for r in read_jsonl(SRC)}
out = D / "cond_pool2_v2.jsonl"
with out.open("w", encoding="utf-8") as h:
    for ti, r in enumerate(rows):
        base = src[r["id"]]
        rec = dict(base)
        rec["retrieved_has_term"] = r["term"] in top30[ti]
        ret = list(top30[ti])
        seeded("pool2eval", r["id"]).shuffle(ret)
        rec["vocab_retrieved"] = ret
        h.write(json.dumps(rec, ensure_ascii=False) + "\n")
RES["pool"] = len(terms)
RES["rows"] = len(rows)
(D / "pool2_recall.json").write_text(json.dumps(RES, indent=1))
log(f"wrote {out}")
log(f"STEP 4 lists done [{time.perf_counter()-T0:.0f}s]")
