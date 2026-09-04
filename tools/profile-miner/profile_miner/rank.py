"""Stage 3 output: mined/candidates.json ranked by count and source breadth."""

import sys

import re

from .candidates import English
from .common import out_dir, read_json, write_json
from .oracle import load_cache, summarize

RE_TAG = re.compile(r"^(?:v\d+(?:\.\d+)*|[a-z]\d{1,2})$")  # v1, w2, p1: coordinates and version tags


def live_junk(c, english):
    """Non-lexicon terms no one dictates: version tags, ALLCAPS or Capitalised English words (FILE, WITHOUT,
    Python), three-letter lowercase abbreviations (env, tok, dir). Lexicon terms never pass through here."""
    key = c["key"]
    cls = set(c["classes"])
    if c["seed"]:
        return False
    if RE_TAG.match(key):
        return True
    if cls <= {"cap", "caps", "cap_initial", "camel"} and key in english:
        return True
    if len(key) <= 3 and key.isalpha():
        return True
    parts = re.split(r"[-_]", key)
    if len(parts) > 1 and all(p in english for p in parts if p):
        return True  # read-only, round-trip, open-source
    return False

DIFF_WEIGHT = {"phonetic": 1.0, "format": 1.0, "case": 1.0, "same": 0.0}  # case kept at 1.0: 43% of the user's gold is case-only (CUDA, vLLM)


def score(c):
    return c["prior"] * DIFF_WEIGHT[c["diff"]] + (1.0 if c["seed"] else 0.0)


def run():
    od = out_dir()
    cands = read_json(od / "candidates" / "candidates_raw.json")
    cache = load_cache(od / "oracle" / "cache.jsonl")
    try:
        from .lexicon import load as load_lexicon
        lexicon = load_lexicon()
    except (OSError, ValueError):
        lexicon = {}
    english = English()
    out = []
    missing = junk = 0
    for c in cands:
        voices = cache.get(c["term"])
        if voices is None:
            missing += 1
            continue
        diff, forms = summarize(c["term"], voices)
        if diff == "same" and not c["seed"]:
            continue
        lex = lexicon.get(c["key"])
        if lex is None and live_junk(c, english):
            junk += 1
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
            "lexicon": lex["cats"] if lex else None,
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
    print(f"[rank] {len(out)} terms kept of {len(cands)} ({missing} without oracle result, {junk} live junk); diff {by}",
          file=sys.stderr)
    return out
