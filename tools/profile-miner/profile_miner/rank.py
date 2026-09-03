"""Stage 3 output: mined/candidates.json ranked by count and source breadth."""

import sys

from .common import out_dir, read_json, write_json
from .oracle import load_cache, summarize

DIFF_WEIGHT = {"phonetic": 1.0, "format": 1.0, "case": 1.0, "same": 0.0}  # case kept at 1.0: 43% of the user's gold is case-only (CUDA, vLLM)


def score(c):
    return c["prior"] * DIFF_WEIGHT[c["diff"]] + (1.0 if c["seed"] else 0.0)


def run():
    od = out_dir()
    cands = read_json(od / "candidates" / "candidates_raw.json")
    cache = load_cache(od / "oracle" / "cache.jsonl")
    out = []
    missing = 0
    for c in cands:
        voices = cache.get(c["term"])
        if voices is None:
            missing += 1
            continue
        diff, forms = summarize(c["term"], voices)
        if diff == "same" and not c["seed"]:
            continue
        item = {
            "term": c["term"],
            "count": c["count"],
            "prior": c.get("prior", 0.0),
            "sources": c["sources"],
            "spoken_forms": forms,
            "diff": diff,
            "evidence": c["evidence"],
            "classes": c["classes"],
            "seed": c["seed"],
        }
        item["score"] = round(score(item), 4)
        out.append(item)
    out.sort(key=lambda c: (-c["score"], -c["count"], c["term"].lower()))
    for i, c in enumerate(out):
        c["rank"] = i + 1
    write_json(od / "mined" / "candidates.json", out)
    by = {}
    for c in out:
        by[c["diff"]] = by.get(c["diff"], 0) + 1
    print(f"[rank] {len(out)} terms kept of {len(cands)} ({missing} without oracle result); diff {by}", file=sys.stderr)
    return out
