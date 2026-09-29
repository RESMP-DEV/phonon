"""STEP 3c: v1-retrieval vs v2-retrieval for bigrun_mid_r16 on heldout_new and heldout_old/unseen.

Metric functions are imported unchanged from research/vocab_v1/analyze_vocab1.py, so the v1 rows
here reproduce research/bigrun_v0/results.md exactly.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
sys.path.insert(0, "/home/user/phonon/research/bigrun_v0")
from analyze_vocab1 import (fair, false_inserts, hit_loose, hit_norm, read_jsonl, row_wer,
                            strict_lc, wer)

D2 = Path("/data/phonon_retrieval_v2")
V1REF = Path("/data/phonon_bigrun_v0/refined")
V1COND = {"new": Path("/data/phonon_bigrun_v0/eval_conditions_new.jsonl"),
          "unseen": Path("/data/phonon_term_eval_v1/eval_conditions.jsonl")}
V2COND = {k: D2 / f"cond_{k}_v2.jsonl" for k in V1COND}
LABEL = "bigrun_mid_r16"


def block(rows, outs, vocabs, raws):
    n = len(rows)
    terms = [r["term"] for r in rows]
    refs = [r["reference"] for r in rows]
    hn = [hit_norm(t, h) for t, h in zip(terms, outs)]
    rw = [row_wer(a, b, fair) for a, b in zip(refs, outs)]
    raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, raws)]
    fi = [false_inserts(v, h, r, x, tt)
          for v, h, r, x, tt in zip(vocabs, outs, refs, raws, terms)]
    has = [r["retrieved_has_term"] for r in rows]
    out = {
        "n": n,
        "term_hit_norm": sum(hn) / n,
        "term_hit_loose": sum(hit_loose(t, h) for t, h in zip(terms, outs)) / n,
        "fair_wer": wer(refs, outs, fair),
        "strict_lc_wer": wer(refs, outs, strict_lc),
        "damage_rate": sum(1 for a, b in zip(rw, raw_rw) if a > b + 1e-9) / n,
        "unrelated_insert_rate": sum(1 for b in fi if b[2]) / n,
        "spurious_on_recovered": sum(1 for b, h in zip([x[2] for x in fi], hn) if b and h) / n,
        "recall@30": sum(1 for x in has if x) / n,
    }
    for name, want in (("term_retrieved", True), ("term_missing", False)):
        sel = [h for h, w in zip(hn, has) if w is want]
        out[f"n_{name}"] = len(sel)
        out[f"hit_{name}"] = (sum(sel) / len(sel)) if sel else None
    return out


res = {}
for key in ("new", "unseen"):
    v1 = read_jsonl(V1COND[key])
    v2 = read_jsonl(V2COND[key])
    r1 = read_jsonl(V1REF / f"{LABEL}__{key}.jsonl")
    r2 = read_jsonl(D2 / "refined" / f"{LABEL}__{key}.jsonl")
    m1 = {r["id"]: r for r in r1}
    m2 = {r["id"]: r for r in r2}
    raws = [r["hyp"] for r in v1]
    res[key] = {
        "v1_retrieval": block(v1, [m1[r["id"]]["out_retrieved"] for r in v1],
                              [r["vocab_retrieved"] for r in v1], raws),
        "v2_retrieval": block(v2, [m2[r["id"]]["out_retrieved"] for r in v2],
                              [r["vocab_retrieved"] for r in v2], raws),
        "v1_oracle": block(v1, [m1[r["id"]]["out_oracle"] for r in v1],
                           [r["vocab_oracle"] for r in v1], raws),
    }
    for c in ("v1_retrieval", "v2_retrieval", "v1_oracle"):
        b = res[key][c]
        print(f"[{key}] {c:14s} hit={b['term_hit_norm']:.4f} recall@30={b['recall@30']:.4f} "
              f"fairWER={b['fair_wer']:.4f} damage={b['damage_rate']:.4f} "
              f"unrel={b['unrelated_insert_rate']:.4f} hit|ret={b['hit_term_retrieved']:.4f}",
              flush=True)
(D2 / "reeval.json").write_text(json.dumps(res, indent=1))
print("STEP 3c done", flush=True)
