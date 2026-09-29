"""Real-audio holdout: the shipped scoring rule (prior + two-stage) at nmax=6."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
from retrieve2 import Index, Phoneme
from common import pool_big
import variants as V
D = Path("/data/phonon_retrieval_v2")
PH = Phoneme()
terms, rank, kind = pool_big()
ix = Index(terms, rank, kind, use_phonemes=True, ph=PH, nreal=5)
rows, texts, golds = V.load_real(terms)
BEST = pickle.load(open(D / "best_matrices.pkl", "rb"))
RES = json.loads((D / "step2_recall.json").read_text())
PR = json.loads((D / "prior.json").read_text())
prior = np.array([PR["alpha"] * PR["kind_lo"].get(k, 0.0) for k in ix.kind], dtype=np.float32)
st, _ = V.recall_stats(BEST[("real", "a_c40m40p20")], ix, golds, prior=prior)
RES["real|d_prior_a_c40m40p20"] = st
print("real d_prior r@10=%.4f r@30=%.4f" % (st["recall@10"], st["recall@30"]), flush=True)
ng = [len(g) for g in golds]
print("gold terms per row: mean %.2f median %d max %d; rows with >30 gold: %d"
      % (sum(ng) / len(ng), sorted(ng)[len(ng) // 2], max(ng), sum(1 for n in ng if n > 30)),
      flush=True)
# recall ceiling given a 30-slot budget
cap = sum(min(n, 30) for n in ng) / sum(ng)
RES["real|__budget_ceiling@30"] = {"recall@10": sum(min(n, 10) for n in ng) / sum(ng),
                                   "recall@30": cap, "pairs": sum(ng)}
print("budget ceiling recall@30 = %.4f, recall@10 = %.4f" % (cap, sum(min(n, 10) for n in ng) / sum(ng)))
(D / "step2_recall.json").write_text(json.dumps(RES, indent=1))
print("STEP 2e done")
