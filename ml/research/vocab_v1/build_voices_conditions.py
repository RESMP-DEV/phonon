"""vocab_v1 unseen-voice check, step 1b: retrieval + oracle lists for the new-voice clips."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")

from retrieve import load_pool  # noqa: E402
from vocab_common import Lexicon, heldout_terms, read_jsonl, seeded  # noqa: E402

ASR = Path("/data/phonon_term_eval_v1/asr.jsonl")
OUT = Path("/data/phonon_term_eval_v1")
TOPK = 30


def main() -> int:
    t0 = time.perf_counter()
    log = lambda s: print(s, flush=True)  # noqa: E731
    rows = read_jsonl(ASR)
    log(f"asr rows={len(rows)}")
    held = heldout_terms()
    lex_eval = Lexicon()
    R = load_pool()
    log(f"pool={len(R.terms)}")

    lists = R.topk([r["hyp"] for r in rows], k=TOPK, log=log)
    out_rows = []
    for r, L in zip(rows, lists):
        terms = [t for t, _ in L]
        has = r["term"] in terms
        rng = seeded("v1eval", r["id"])
        ret = list(terms)
        rng.shuffle(ret)
        term, kind = r["term"], r["kind"]
        rank = float(held.get(term, {}).get("rank_score") or 0.0)
        rng2 = seeded("v1eval_oracle", r["id"])
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

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "eval_conditions.jsonl").open("w", encoding="utf-8") as h:
        for r in out_rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {"rows": len(out_rows), "topk": TOPK, "pool": len(R.terms)}
    for k in (10, 30):
        stats[f"recall@{k}"] = sum(
            1 for r, L in zip(out_rows, lists) if r["term"] in [t for t, _ in L[:k]]) / len(out_rows)
    by_voice: dict[str, list[int]] = {}
    for r, L in zip(out_rows, lists):
        by_voice.setdefault(r["voice"], []).append(int(r["retrieved_has_term"]))
    stats["recall@30_by_voice"] = {k: sum(v) / len(v) for k, v in sorted(by_voice.items())}
    stats["missing_term_by_voice"] = {k: len(v) - sum(v) for k, v in sorted(by_voice.items())}
    stats["missing_term"] = sum(1 for r in out_rows if not r["retrieved_has_term"])
    by_kind: dict[str, list[int]] = {}
    for r in out_rows:
        by_kind.setdefault(r["kind"], []).append(int(r["retrieved_has_term"]))
    stats["recall@30_by_kind"] = {k: sum(v) / len(v) for k, v in sorted(by_kind.items())}
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}))
    log(f"BUILD DONE [{time.perf_counter()-t0:.0f}s] -> {OUT/'eval_conditions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
