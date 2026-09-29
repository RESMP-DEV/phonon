"""pool2 eval step 1: retrieval + oracle vocabulary lists for the heldout_pool2 clips.

300 never-trained pool-2 terms x 3 sentences x 3 voices that appear in no training clip. The
retrieval pool is the UNION lexicon (bigrun lexicon + pool2 + every held-out set), exactly the
pool step4b_lists.py used for the training lists.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
sys.path.insert(0, "/home/user/phonon/research/pool2_v0")

import vocab_common  # noqa: E402
vocab_common.LEXICON_PATH = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
from step4b_lists import build_retriever  # noqa: E402
from vocab_common import Lexicon, read_jsonl, seeded  # noqa: E402

D = Path("/data/phonon_pool2_v0")
ASR = [D / f"asr_heldout_half{h}.jsonl" for h in (0, 1)]
HELD_P2 = D / "terms_heldout_pool2.jsonl"
HELD_OLD = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")
HELD_NEW = Path("/data/phonon_bigrun_v0/terms_heldout_new.jsonl")
OUT = D / "eval_conditions_pool2.jsonl"
TOPK = 30


def main() -> int:
    t0 = time.perf_counter()
    log = lambda s: print(s, flush=True)  # noqa: E731
    rows = []
    for p in ASR:
        rows += read_jsonl(p)
    rows.sort(key=lambda r: r["id"])
    log(f"asr rows={len(rows)}")
    held_p2 = {r["term"]: r for r in read_jsonl(HELD_P2)}
    held_all = set(held_p2) | {r["term"] for r in read_jsonl(HELD_OLD)} | \
        {r["term"] for r in read_jsonl(HELD_NEW)}
    lex_eval = Lexicon(exclude=held_all)
    R = build_retriever()
    log(f"pool={len(R.terms)} lexicon_rows={len(lex_eval.rows)}")

    lists = R.topk([r["hyp"] for r in rows], k=TOPK, log=log)
    out_rows = []
    for r, L in zip(rows, lists):
        terms = [t for t, _ in L]
        has = r["term"] in terms
        rng = seeded("pool2eval", r["id"])
        ret = list(terms)
        rng.shuffle(ret)
        term, kind = r["term"], r["kind"]
        rank = float(held_p2.get(term, {}).get("rank_score") or 0.0)
        rng2 = seeded("pool2eval_oracle", r["id"])
        od = lex_eval.nearest(kind, rank, 9, {term})
        if len(od) < 9:
            od += lex_eval.sample_distractors(rng2, 9 - len(od), [kind], {term, *od})
        oracle = [term] + od
        rng2.shuffle(oracle)
        out_rows.append({
            "id": r["id"], "term": term, "kind": kind, "voice": r["voice"],
            "degraded": r.get("degraded"), "reference": r["reference"], "hyp": r["hyp"],
            "retrieved_has_term": has, "vocab_oracle": oracle, "vocab_retrieved": ret,
        })

    with OUT.open("w", encoding="utf-8") as h:
        for r in out_rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {"rows": len(out_rows), "topk": TOPK, "pool": len(R.terms)}
    for k in (10, 30):
        stats[f"recall@{k}"] = sum(
            1 for r, L in zip(out_rows, lists) if r["term"] in [t for t, _ in L[:k]]) / len(out_rows)
    by_voice: dict[str, list[int]] = {}
    for r in out_rows:
        by_voice.setdefault(r["voice"], []).append(int(r["retrieved_has_term"]))
    stats["recall@30_by_voice"] = {k: sum(v) / len(v) for k, v in sorted(by_voice.items())}
    stats["missing_term"] = sum(1 for r in out_rows if not r["retrieved_has_term"])
    by_kind: dict[str, list[int]] = {}
    for r in out_rows:
        by_kind.setdefault(r["kind"], []).append(int(r["retrieved_has_term"]))
    stats["recall@30_by_kind"] = {k: sum(v) / len(v) for k, v in sorted(by_kind.items())}
    (D / "eval_pool2_build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}))
    log(f"BUILD POOL2 DONE [{time.perf_counter()-t0:.0f}s] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
