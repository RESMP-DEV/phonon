"""pool3 eval step 1: retrieval + oracle vocabulary lists for the heldout_pool3 clips.

300 never-trained pool-3 terms x 3 sentences x the 3 unseen Kokoro voices. The retrieval pool is
the UNION lexicon (bigrun lexicon + pool2 + pool3 + every held-out set, ~74k terms) and the
retriever is the shipped two-stage v2 (stage 1: 1 realisation, char+metaphone; stage 2: 5
realisations, char+metaphone+phoneme, kind prior and rank bonus), exactly as
research/index_v0/step2_lists.py runs it.

vocab_retrieved is the top-30, so term hit is comparable with heldout_pool2 and heldout_new.
The adaptive budget min(80, max(30, 3 x hypothesis words)) with the score threshold 65 tuned in
index_v0 is reported as recall@budget and stored per row as vocab_budget.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")

import vocab_common  # noqa: E402
vocab_common.LEXICON_PATH = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
from vocab_common import Lexicon, read_jsonl, seeded  # noqa: E402
from retrieve2 import Index, Phoneme, ngrams as _ng  # noqa: E402
import variants as V  # noqa: E402

D = Path("/data/phonon_pool3_v0")
ASR = [D / "asr_heldout_s0.jsonl"]
LEX_FULL = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
HELD_OLD = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")
HELD_NEW = Path("/data/phonon_bigrun_v0/terms_heldout_new.jsonl")
POOL2 = Path("/data/phonon_pool2_v0/terms_pool2.jsonl")
HELD_P2 = Path("/data/phonon_pool2_v0/terms_heldout_pool2.jsonl")
POOL3 = D / "terms_pool3.jsonl"
HELD_P3 = D / "terms_heldout_pool3.jsonl"
UNION = (LEX_FULL, HELD_OLD, HELD_NEW, POOL2, HELD_P2, POOL3, HELD_P3)
OUT = D / "eval_conditions_pool3.jsonl"
TOPK = 30
CAND = 400
TOPN = 100
MULT, CAP, THR = 3, 80, 65

PR = json.loads(Path("/data/phonon_retrieval_v2/prior.json").read_text())
KLO, ALPHA, RANK_EPS = PR["kind_lo"], PR["alpha"], PR["rank_eps"]
log = lambda s: print(s, flush=True)  # noqa: E731


def union_pool():
    seen, terms, rank, kind = set(), [], {}, {}
    for src in UNION:
        for r in read_jsonl(src):
            t = (r.get("term") or "").strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            terms.append(t)
            rank[t] = float(r.get("rank_score") or 0.0)
            kind[t] = r.get("kind") or ""
    return terms, rank, kind


def budget(nwords: int) -> int:
    return min(CAP, max(30, MULT * nwords))


def main() -> int:
    t0 = time.perf_counter()
    rows = []
    for p in ASR:
        rows += read_jsonl(p)
    rows.sort(key=lambda r: r["id"])
    log(f"asr rows={len(rows)}")

    terms, rank, kind = union_pool()
    log(f"union pool={len(terms)} [{time.perf_counter()-t0:.0f}s]")
    texts = [r["hyp"] for r in rows]

    PH = Phoneme()
    ix1 = Index(terms, rank, kind, use_phonemes=False, nreal=1)
    ix5 = Index(terms, rank, kind, use_phonemes=True, ph=PH, nreal=5)
    log(f"reals1={len(ix1.reals)} reals5={len(ix5.reals)} [{time.perf_counter()-t0:.0f}s]")
    warm = {w for r in ix5.reals for w in r.split()}
    for t in texts:
        warm.update(_ng(t, 1))
    n_new = PH.warm(warm)
    if n_new:
        PH.save()
        ix5.r_ph = [PH.phrase(r) for r in ix5.reals]
    log(f"g2p warmed {n_new} [{time.perf_counter()-t0:.0f}s]")

    b1, _ = V.run_cfgs(ix1, texts, [("s1", 0.5, 0.5, 0.0)], nmax=6, block=1024, log=log)
    log(f"stage1 done [{time.perf_counter()-t0:.0f}s]")
    b5, _ = V.run_cfgs(ix5, texts, [("d", 0.4, 0.4, 0.2)], nmax=8, block=1024, log=log)
    log(f"stage2 done [{time.perf_counter()-t0:.0f}s]")
    S1, B = b1["s1"], b5["d"]
    prior = np.array([ALPHA * KLO.get(k, 0.0) for k in ix5.kind], dtype=np.float32)
    bonus = RANK_EPS * np.log1p(ix5.rank)

    tops = []
    for ti in range(len(texts)):
        r1 = S1[ti].astype(np.float32) + bonus
        c = np.argpartition(-r1, CAND - 1)[:CAND]
        r2 = B[ti][c].astype(np.float32) + bonus[c] + prior[c]
        order = c[np.argsort(-r2, kind="stable")][:TOPN]
        sc = B[ti][order].astype(np.float32) + bonus[order] + prior[order]
        tops.append([(terms[int(j)], float(s)) for j, s in zip(order, sc)])
    log(f"ranked [{time.perf_counter()-t0:.0f}s]")

    held_p3 = {r["term"]: r for r in read_jsonl(HELD_P3)}
    held_all = set(held_p3) | {r["term"] for r in read_jsonl(HELD_OLD)} | \
        {r["term"] for r in read_jsonl(HELD_NEW)} | \
        {r["term"] for r in read_jsonl(HELD_P2)}
    lex_eval = Lexicon(exclude=held_all)

    out_rows, bud_lens = [], []
    for r, L in zip(rows, tops):
        top_terms = [t for t, _ in L]
        ret = top_terms[:TOPK]
        has = r["term"] in ret
        nb = budget(len(r["hyp"].split()))
        vb = [t for t, s in L[:nb] if s >= THR]
        bud_lens.append(len(vb))
        rng = seeded("pool3eval", r["id"])
        ret_sh = list(ret)
        rng.shuffle(ret_sh)
        term, kd = r["term"], r["kind"]
        rk = float(held_p3.get(term, {}).get("rank_score") or 0.0)
        rng2 = seeded("pool3eval_oracle", r["id"])
        od = lex_eval.nearest(kd, rk, 9, {term})
        if len(od) < 9:
            od += lex_eval.sample_distractors(rng2, 9 - len(od), [kd], {term, *od})
        oracle = [term] + list(od)
        rng2.shuffle(oracle)
        out_rows.append({
            "id": r["id"], "term": term, "kind": kd, "voice": r["voice"],
            "degraded": r.get("degraded"), "reference": r["reference"], "hyp": r["hyp"],
            "retrieved_has_term": has, "vocab_oracle": oracle, "vocab_retrieved": ret_sh,
            "vocab_budget": vb, "budget": nb,
        })

    with OUT.open("w", encoding="utf-8") as h:
        for r in out_rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(out_rows)
    stats = {"rows": n, "topk": TOPK, "pool": len(terms), "budget_rule":
             f"min({CAP}, max(30, {MULT} x words)) thr={THR}"}
    for k in (10, 30, 100):
        stats[f"recall@{k}"] = sum(1 for r, L in zip(out_rows, tops)
                                   if r["term"] in [t for t, _ in L[:k]]) / n
    stats["recall@budget"] = sum(1 for r in out_rows if r["term"] in r["vocab_budget"]) / n
    stats["budget_mean_len"] = sum(bud_lens) / n
    stats["missing_term"] = sum(1 for r in out_rows if not r["retrieved_has_term"])
    by_voice: dict[str, list[int]] = {}
    for r in out_rows:
        by_voice.setdefault(r["voice"], []).append(int(r["retrieved_has_term"]))
    stats["recall@30_by_voice"] = {k: sum(v) / len(v) for k, v in sorted(by_voice.items())}
    by_kind: dict[str, list[int]] = {}
    for r in out_rows:
        by_kind.setdefault(r["kind"], []).append(int(r["retrieved_has_term"]))
    stats["recall@30_by_kind"] = {k: sum(v) / len(v) for k, v in sorted(by_kind.items())}
    (D / "eval_pool3_build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, indent=1))
    log(f"BUILD POOL3 DONE [{time.perf_counter()-t0:.0f}s] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
