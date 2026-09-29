"""STEP 4c: heldout_pool2, v1 lists vs v2 lists vs oracle, for bigrun_mid_r16 and bigrun_xl_r16."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
from analyze_vocab1 import (fair, false_inserts, hit_loose, hit_norm, read_jsonl, row_wer,
                            strict_lc, wer)  # noqa: E402


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

D2 = Path("/data/phonon_retrieval_v2")
V1COND = Path("/data/phonon_pool2_v0/eval_conditions_pool2.jsonl")
V2COND = D2 / "cond_pool2_v2.jsonl"
V1REF = Path("/data/phonon_pool2_v0/refined")
LABELS = ["bigrun_mid_r16", "bigrun_xl_r16"]

v1 = read_jsonl(V1COND)
v2 = read_jsonl(V2COND)
raws = [r["hyp"] for r in v1]
res = {}
for lab in LABELS:
    r1 = {r["id"]: r for r in read_jsonl(V1REF / f"{lab}__pool2.jsonl")}
    r2 = {r["id"]: r for r in read_jsonl(D2 / "refined" / f"{lab}__pool2.jsonl")}
    res[lab] = {
        "v1_retrieval": block(v1, [r1[r["id"]]["out_retrieved"] for r in v1],
                              [r["vocab_retrieved"] for r in v1], raws),
        "v2_retrieval": block(v2, [r2[r["id"]]["out_retrieved"] for r in v2],
                              [r["vocab_retrieved"] for r in v2], raws),
        "oracle": block(v1, [r1[r["id"]]["out_oracle"] for r in v1],
                        [r["vocab_oracle"] for r in v1], raws),
    }
    for c in ("v1_retrieval", "v2_retrieval", "oracle"):
        b = res[lab][c]
        print(f"[{lab}] {c:14s} recall@30={b['recall@30']:.4f} hit={b['term_hit_norm']:.4f} "
              f"hit|ret={b['hit_term_retrieved']:.4f} fairWER={b['fair_wer']:.4f} "
              f"damage={b['damage_rate']*100:.1f}% unrel={b['unrelated_insert_rate']*100:.2f}%",
              flush=True)
(D2 / "pool2_reeval.json").write_text(json.dumps(res, indent=1))
print("STEP 4c done", flush=True)
