"""bigrun_v0 step 1: widen the term pool to the full 15,881 ranked lexicon, minus the old
300 held-out terms, minus 300 NEW held-out terms drawn stratified by (kind, rank decile)."""
from __future__ import annotations
import json, random
from collections import Counter, defaultdict
from pathlib import Path

LEX_FULL = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
LEX_V1 = Path("/data/phonon_synth_v1/lexicon_ranked.jsonl")
HELD_OLD = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")
ENGLISH = Path("/data/phonon_asr_errors_v0/english_words.txt")
OUT = Path("/data/phonon_bigrun_v0")
N_NEW = 300
SEED = 20260918


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    english = {w.strip().lower() for w in ENGLISH.read_text(encoding="utf-8").splitlines() if w.strip()}
    old_held = {r["term"].lower() for r in read_jsonl(HELD_OLD)}
    v1_terms = {(json.loads(l)["term"] or "").lower() for l in LEX_V1.read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = read_jsonl(LEX_FULL)
    print(f"lexicon_full={len(rows)} v1_subset={len(v1_terms)} old_heldout={len(old_held)}", flush=True)

    drop = Counter()
    seen: set[str] = set()
    pool: list[dict] = []
    for r in rows:
        t = (r.get("term") or "").strip()
        tl = t.lower()
        if not t:
            drop["empty"] += 1; continue
        if tl in seen:
            drop["dup"] += 1; continue
        seen.add(tl)
        if len(t) < 3:
            drop["short"] += 1; continue
        if t.isalpha() and t.islower() and tl in english:
            drop["english"] += 1; continue
        if tl in old_held:
            drop["old_heldout"] += 1; continue
        pool.append({
            "term": t,
            "kind": r.get("kind") or "other",
            "count": r.get("count"),
            "mangle_score": r.get("mangle_score"),
            "rank_score": float(r.get("rank_score") or 0.0),
            "in_v1_subset": tl in v1_terms,
        })
    print(f"after_filter={len(pool)} drops={dict(drop)}", flush=True)

    # rank_score deciles over the surviving pool
    scores = sorted(p["rank_score"] for p in pool)
    n = len(scores)
    cuts = [scores[min(n - 1, int(n * (i + 1) / 10))] for i in range(9)]
    def decile(s: float) -> int:
        d = 0
        for c in cuts:
            if s > c:
                d += 1
        return d
    for p in pool:
        p["decile"] = decile(p["rank_score"])

    # stratified draw of 300 new held-out terms
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for p in pool:
        strata[(p["kind"], p["decile"])].append(p)
    rng = random.Random(SEED)
    keys = sorted(strata)
    # largest-remainder proportional allocation
    quota = {k: N_NEW * len(strata[k]) / len(pool) for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    rem = sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), k))
    i = 0
    while sum(alloc.values()) < N_NEW:
        k = rem[i % len(rem)]
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
        i += 1
        if i > 10000:
            break
    picked: list[dict] = []
    for k in keys:
        cand = sorted(strata[k], key=lambda p: p["term"])
        rng.shuffle(cand)
        # prefer terms outside the 4,738-term v1 subset so >= half are new to it
        cand.sort(key=lambda p: p["in_v1_subset"])
        picked += cand[: alloc[k]]
    # repair: ensure at least half are outside the v1 subset
    n_out = sum(1 for p in picked if not p["in_v1_subset"])
    if n_out < N_NEW // 2:
        chosen = {p["term"] for p in picked}
        spare = [p for p in pool if p["term"] not in chosen and not p["in_v1_subset"]]
        rng.shuffle(spare)
        swap = [p for p in picked if p["in_v1_subset"]]
        rng.shuffle(swap)
        need = N_NEW // 2 - n_out
        for a, b in zip(swap[:need], spare[:need]):
            picked[picked.index(a)] = b
        n_out = sum(1 for p in picked if not p["in_v1_subset"])
    picked.sort(key=lambda p: (-p["rank_score"], p["term"]))
    held_new = set()
    with (OUT / "terms_heldout_new.jsonl").open("w", encoding="utf-8") as h:
        for j, p in enumerate(picked):
            rec = dict(p); rec["term_id"] = f"h{j:04d}"
            held_new.add(p["term"])
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train_pool = [p for p in pool if p["term"] not in held_new]
    train_pool.sort(key=lambda p: (-p["rank_score"], p["term"]))
    with (OUT / "terms_pool.jsonl").open("w", encoding="utf-8") as h:
        for j, p in enumerate(train_pool):
            rec = dict(p); rec["term_id"] = f"b{j:05d}"
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "lexicon_full": len(rows), "after_filter": len(pool), "drops": dict(drop),
        "heldout_new": len(picked),
        "heldout_new_outside_v1_subset": n_out,
        "heldout_new_kinds": dict(Counter(p["kind"] for p in picked)),
        "heldout_new_deciles": dict(sorted(Counter(p["decile"] for p in picked).items())),
        "pool_terms": len(train_pool),
        "pool_kinds": dict(Counter(p["kind"] for p in train_pool).most_common()),
        "pool_in_v1_subset": sum(1 for p in train_pool if p["in_v1_subset"]),
    }
    (OUT / "pool_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    print("STEP 1 done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
