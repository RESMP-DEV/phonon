"""pool3 step 4b: retrieval top-k lists over the UNION lexicon
(bigrun full lexicon + pool2 + pool3 + every held-out set, about 74k terms).

Retriever: the shipped vocab_v1 Retriever, exactly as pool2_v0/step4b_lists.py used it, so the
continue-train sees the same list distribution the xl adapter was trained on. The two-stage v2
retriever is used for the *eval* conditions (build_pool3_conditions.py), where its cost is
affordable; at 74k terms and 145k training rows the dense v2 pass does not fit the night.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
from retrieve import Retriever  # noqa: E402

D = Path("/data/phonon_pool3_v0")
LEX_FULL = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
HELD_OLD = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")
HELD_NEW = Path("/data/phonon_bigrun_v0/terms_heldout_new.jsonl")
POOL2 = Path("/data/phonon_pool2_v0/terms_pool2.jsonl")
HELD_P2 = Path("/data/phonon_pool2_v0/terms_heldout_pool2.jsonl")
POOL3 = D / "terms_pool3.jsonl"
HELD_P3 = D / "terms_heldout_pool3.jsonl"
POOL = D / "acoustic_pool3.jsonl"
BLOCK = 12000

UNION = (LEX_FULL, HELD_OLD, HELD_NEW, POOL2, HELD_P2, POOL3, HELD_P3)


def read_jsonl(p: Path):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_retriever() -> Retriever:
    seen: set[str] = set()
    terms: list[str] = []
    rank: dict[str, float] = {}
    for src in UNION:
        for r in read_jsonl(src):
            t = (r.get("term") or "").strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            terms.append(t)
            rank[t] = float(r.get("rank_score") or 0.0)
    print(f"union retriever terms={len(terms)}", flush=True)
    return Retriever(terms, rank)


def dump(R, rows, key, k, out: Path) -> None:
    t0 = time.perf_counter()
    with out.open("w", encoding="utf-8") as h:
        for s in range(0, len(rows), BLOCK):
            chunk = rows[s:s + BLOCK]
            lists = R.topk([r[key] for r in chunk], k=k)
            for r, L in zip(chunk, lists):
                h.write(json.dumps({"id": r["id"], "terms": [t for t, _ in L]},
                                   ensure_ascii=False) + "\n")
            print(f"  {out.name} {min(s+BLOCK, len(rows))}/{len(rows)} "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)


def main() -> int:
    t0 = time.perf_counter()
    R = build_retriever()
    (D / "union_pool_size.json").write_text(json.dumps({"union_terms": len(R.terms)}) + "\n")
    ac = read_jsonl(POOL)
    print(f"acoustic={len(ac)}", flush=True)
    dump(R, ac, "input", 31, D / "lists_acoustic3.jsonl")
    print(f"STEP 4b done [{time.perf_counter()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
