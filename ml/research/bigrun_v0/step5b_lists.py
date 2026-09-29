"""bigrun_v0 step 5b: retrieval top-k lists over the FULL lexicon plus both held-out sets."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
from retrieve import Retriever  # noqa: E402

D = Path("/data/phonon_bigrun_v0")
LEX_FULL = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
HELD_OLD = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")
HELD_NEW = D / "terms_heldout_new.jsonl"
POOL = D / "acoustic_pool.jsonl"
TRAIN_REAL = Path("/data/phonon_corrector_v0/datesplit/train_date.jsonl")
BLOCK = 12000


def read_jsonl(p: Path):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_retriever() -> Retriever:
    seen: set[str] = set()
    terms: list[str] = []
    rank: dict[str, float] = {}
    for src in (LEX_FULL, HELD_OLD, HELD_NEW):
        for r in read_jsonl(src):
            t = (r.get("term") or "").strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            terms.append(t)
            rank[t] = float(r.get("rank_score") or 0.0)
    print(f"retriever terms={len(terms)}", flush=True)
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
    real = read_jsonl(TRAIN_REAL)
    ac = read_jsonl(POOL)
    print(f"acoustic={len(ac)} real={len(real)}", flush=True)
    dump(R, real, "input", 30, D / "lists_real.jsonl")
    dump(R, ac, "input", 31, D / "lists_acoustic.jsonl")
    print(f"STEP 5b done [{time.perf_counter()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
