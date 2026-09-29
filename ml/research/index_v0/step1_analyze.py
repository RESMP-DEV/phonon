"""STEP 1b: symbol index vs the shipped 15,881-term lexicon; per-repo inverted index for step 4."""
from __future__ import annotations
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path("/data/phonon_index_v0")
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
IX = D / (f"symbols_local_{TAG}.jsonl" if TAG else "symbols_local.jsonl")
SUF = f"_{TAG}" if TAG else ""
LEX_RANKED = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
LEX_RAW = Path("/data/phonon_synth_v0/lexicon.jsonl")
HELD_NEW = Path("/data/phonon_bigrun_v0/terms_heldout_new.jsonl")
HELD_OLD = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")


def jl(p):
    with Path(p).open(encoding="utf-8") as h:
        for l in h:
            if l.strip():
                yield json.loads(l)


ix_terms, ix_kind, ix_count, ix_repos = {}, {}, {}, {}
for r in jl(IX):
    t = r["term"]
    ix_terms[t.lower()] = t
    ix_kind[t] = r["kind"]
    ix_count[t] = r["count"]
    ix_repos[t] = r["repos"]
print(f"index terms={len(ix_terms)}")

ranked = [r["term"] for r in jl(LEX_RANKED)]
ranked_l = {t.lower() for t in ranked}
raw_l = {r["term"].lower() for r in jl(LEX_RAW)}
held_new = [r["term"] for r in jl(HELD_NEW)]
held_old = [r["term"] for r in jl(HELD_OLD)]

ixl = set(ix_terms)
stats = {
    "index_terms": len(ixl),
    "lexicon_ranked": len(ranked_l),
    "lexicon_raw": len(raw_l),
    "overlap_with_ranked": len(ixl & ranked_l),
    "recall_of_ranked_lexicon": round(len(ixl & ranked_l) / len(ranked_l), 4),
    "overlap_with_raw": len(ixl & raw_l),
    "recall_of_raw_lexicon": round(len(ixl & raw_l) / len(raw_l), 4),
    "new_vs_ranked": len(ixl - ranked_l),
    "new_vs_raw": len(ixl - raw_l),
    "heldout_new_found": sum(1 for t in held_new if t.lower() in ixl),
    "heldout_new_total": len(held_new),
    "heldout_old_found": sum(1 for t in held_old if t.lower() in ixl),
    "heldout_old_total": len(held_old),
}
# where the misses of the ranked lexicon sit
miss = [t for t in ranked if t.lower() not in ixl]
stats["ranked_missed_kinds"] = dict(Counter(
    next((r["kind"] for r in []), "") for t in miss[:0]) )
km = Counter()
for r in jl(LEX_RANKED):
    if r["term"].lower() not in ixl:
        km[r.get("kind", "")] += 1
stats["ranked_missed_by_kind"] = dict(km.most_common())
stats["ranked_missed_examples"] = miss[:25]

# count distribution of new terms
newt = [t for tl, t in ix_terms.items() if tl not in raw_l]
stats["new_vs_raw_count_ge2"] = sum(1 for t in newt if ix_count[t] >= 2)
stats["new_vs_raw_count_ge5"] = sum(1 for t in newt if ix_count[t] >= 5)
stats["new_examples"] = sorted(newt, key=lambda t: -ix_count[t])[:25]
stats["index_kinds"] = dict(Counter(ix_kind.values()).most_common())

# ---- per-repo inverted index (for step 4) ----------------------------------
by_repo = defaultdict(list)
for t, reps in ix_repos.items():
    for rp in reps:
        by_repo[rp].append(t)
stats["repos"] = len(by_repo)
sizes = sorted((len(v) for v in by_repo.values()), reverse=True)
stats["repo_terms_median"] = sizes[len(sizes) // 2] if sizes else 0
stats["repo_terms_p90"] = sizes[int(len(sizes) * 0.1)] if sizes else 0
stats["repo_terms_max"] = sizes[0] if sizes else 0
(D / f"by_repo{SUF}.json").write_text(json.dumps({k: v for k, v in by_repo.items()}))
(D / f"index_overlap{SUF}.json").write_text(json.dumps(stats, indent=1) + "\n")
print(json.dumps({k: v for k, v in stats.items()
                  if k not in ("ranked_missed_examples", "new_examples")}, indent=1))
print("missed examples:", stats["ranked_missed_examples"])
print("new examples:", stats["new_examples"])
print("STEP 1b done")
