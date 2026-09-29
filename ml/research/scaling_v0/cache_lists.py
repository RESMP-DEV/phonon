"""scaling_v0: retrieval top-k lists for every acoustic pool row and every real train row.

Computed once with the training-time retriever (held-out terms excluded) and cached, so each
of the eleven training-mix builds is a lookup instead of a fresh retrieval pass.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
from retrieve import load_pool  # noqa: E402
from vocab_common import heldout_terms, read_jsonl  # noqa: E402

D = Path("/data/phonon_scaling_v0")
POOL = D / "acoustic_pool.jsonl"
TRAIN_REAL = Path("/data/phonon_corrector_v0/datesplit/train_date.jsonl")
TOPK = 31
BLOCK = 16000


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
    held = set(heldout_terms())
    R = load_pool(exclude=held)
    print(f"pool_terms={len(R.terms)} held={len(held)}", flush=True)
    ac = [r for r in read_jsonl(POOL) if (r.get("term") or "") not in held]
    real = read_jsonl(TRAIN_REAL)
    print(f"acoustic={len(ac)} real={len(real)}", flush=True)
    dump(R, real, "input", 30, D / "lists_real.jsonl")
    dump(R, ac, "input", TOPK, D / "lists_acoustic.jsonl")
    print(f"CACHE LISTS DONE [{time.perf_counter()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
