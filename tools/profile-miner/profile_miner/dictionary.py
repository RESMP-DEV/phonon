"""Stage 5: the first-run dictionary = lexicon entries present in the user's sources + live mined names.

Tier 1 (prompt and biasing): live non-lexicon terms from rank plus lexicon entries whose spoken form differs
phonetically or in format. Tier 2 (casing map only): lexicon entries the recognizer already spells right or
only miscases (H100, MacBook, MLX); they cost nothing and fix casing deterministically.
Lexicon spoken forms come from the oracle cache; in the product they ship precomputed.
"""
import sys

from .common import out_dir, read_json, write_json
from .lexicon import load as load_lexicon, load_spoken
from .oracle import load_cache, summarize

LOG_SOURCES = ("claude", "codex", "grok")
STRONG_CATS = {"gpu-sku", "gpu-arch", "cuda", "dtype", "model", "company", "devtool", "hf-model", "hf-org", "pypi"}
CURATED_CATS = {"gpu-sku", "gpu-arch", "cuda", "dtype", "model", "devtool", "company"}
MIN_LOG, MIN_CODE, MIN_LEN = 1, 3, 4


def lexicon_selected(key, entry, cand, english):
    """Presence rule tuned on the Wispr gold (2026-09-04): 149 of 154 present gold terms at 1,316 entries
    instead of 2,224. A user-authored mention, or 3+ source files; weak lists (crates, brew) need a mention;
    short names need a curated category."""
    s = cand["sources"]
    logc = sum(s.get(x, 0) for x in LOG_SOURCES)
    cats = set(entry["cats"])
    if len(key) < MIN_LEN and not cats & CURATED_CATS:
        return False
    if cats == {"crate"}:
        return False  # crate names alone: 0 gold, Opus keep rate 0.57 (symlink, addr, hostname, uname)
    if cats == {"pypi"} and logc < 2:
        return False  # PyPI names alone need two mentions: 0 gold among 205, keep rate 0.66
    if not cats & STRONG_CATS and logc == 0:
        return False
    if key in english and cand["term"] != entry["term"] and not (cats & CURATED_CATS and logc >= 2):
        return False  # Cursor, Signal, Linear: the word only counts when the user writes it the way the name is spelled,
        # unless it is a curated name the user mentions twice (codex, llama, triton)
    return logc >= MIN_LOG or s.get("code", 0) >= MIN_CODE


def run(live_top=100, oracle_cache=None):
    od = out_dir()
    cands = {c["key"]: c for c in read_json(od / "candidates" / "candidates_raw.json")}
    lexicon = load_lexicon()
    # shipped forms win: the runtime oracle skips lexicon terms, so they are the
    # only forms a fresh machine has for them
    cache = load_cache(oracle_cache or od / "oracle" / "cache.jsonl")
    cache.update(load_spoken())
    from .candidates import English
    english = English()
    tier1, tier2 = [], []
    stats = {"present": 0, "selected": 0, "no_oracle": 0}
    for key, entry in lexicon.items():
        c = cands.get(key)
        if c is None:
            continue
        stats["present"] += 1
        if not lexicon_selected(key, entry, c, english):
            continue
        stats["selected"] += 1
        voices = cache.get(entry["term"])
        if voices is None:
            stats["no_oracle"] += 1
            diff, forms = "unknown", []
        else:
            diff, forms = summarize(entry["term"], voices)
        item = {"term": entry["term"], "key": key, "origin": "lexicon", "cats": entry["cats"], "diff": diff,
                "spoken_forms": forms, "count": c["count"], "sources": c["sources"], "prior": c.get("prior", 0.0)}
        (tier2 if diff in ("same", "case") else tier1).append(item)
    seen = {i["key"] for i in tier1} | {i["key"] for i in tier2}
    live = 0
    for m in read_json(od / "mined" / "candidates.json"):
        if m.get("lexicon") or m["term"].strip().lower() in seen:
            continue
        tier1.append({"term": m["term"], "key": m["term"].strip().lower(), "origin": "live", "cats": [], "diff": m["diff"],
                      "spoken_forms": m["spoken_forms"], "count": m["count"], "sources": m["sources"], "prior": m["prior"],
                      "evidence": m.get("evidence", [])})
        live += 1
        if live >= live_top:
            break
    tier1.sort(key=lambda i: (-i["prior"], i["term"].lower()))
    tier2.sort(key=lambda i: (-i["prior"], i["term"].lower()))
    out = {"tier1": tier1, "tier2_casing": tier2, "stats": {**stats, "live": live, "tier1": len(tier1), "tier2": len(tier2)}}
    write_json(od / "mined" / "dictionary.json", out)
    print(f"[dictionary] tier1 {len(tier1)} (live {live}), tier2 casing {len(tier2)}; lexicon present {stats['present']}, "
          f"selected {stats['selected']}, without oracle {stats['no_oracle']}", file=sys.stderr)
    return out
