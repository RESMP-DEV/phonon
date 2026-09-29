"""STEP 3b: /data/phonon_pool3_v0/sources.md - where every pool-3 term came from."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("/data/phonon_pool3_v0")
st = json.loads((OUT / "pool3_stats.json").read_text())
pool = [json.loads(l) for l in (OUT / "terms_pool3.jsonl").read_text().splitlines() if l.strip()]
held = [json.loads(l) for l in (OUT / "terms_heldout_pool3.jsonl").read_text().splitlines() if l.strip()]

by_src_pool = Counter(p["source"] for p in pool)
by_src_held = Counter(p["source"] for p in held)
raw_by_repo = {}
for f in sorted((OUT / "per_repo").glob("*.jsonl")):
    lang, rest = f.stem.split("__", 1)
    raw_by_repo[(lang, rest.replace("__", "/"))] = sum(1 for _ in f.open())

lines = ["# Pool 3 sources", "",
         "Term candidates are the symbol index (`research/index_v0/symbol_index.py`, tree-sitter",
         "definitions + doc headings/code spans + config keys + CLI flags) run over 150 shallow",
         "clones of popular public repositories, 25 per language, plus the tree-sitter grammars'",
         "own keyword lists.  Clones were deleted after extraction.  Filters follow",
         "`research/pool2_v0/build_pool2.py`: 3+ chars, speakable, not plain English, deduped",
         "against pools 1-2 and every held-out set; a term must appear at least",
         f"{st['min_count']} times in its repo; each language is capped at {st['per_lang_cap']} terms.",
         "", "## Totals", "",
         f"- per-repo index files: {st['per_repo_files']}",
         f"- raw index terms over all repos: {sum(st['raw_terms_by_lang'].values()):,}",
         f"- candidates after the count>={st['min_count']} cut and cross-repo dedupe: {st['candidates_after_mincount']:,}",
         f"- tree-sitter grammar keywords added: {st['grammar_keywords']}",
         f"- survive the pool-2 filters: {st['after_filter']:,}",
         f"- kept after the per-language cap: {st['pool_terms'] + st['heldout_terms']:,}",
         f"- **terms_pool3.jsonl: {st['pool_terms']:,}**, **terms_heldout_pool3.jsonl: {st['heldout_terms']}**",
         "", "Drops: " + ", ".join(f"{k} {v:,}" for k, v in st["drops"].items()), "",
         "## By language", "",
         "| language | repos | raw index terms | kept in pool 3 | held out |",
         "|---|---:|---:|---:|---:|"]
for lang in sorted(st["repos_by_lang"]):
    lines.append(f"| {lang} | {st['repos_by_lang'][lang]} | {st['raw_terms_by_lang'].get(lang,0):,} | "
                 f"{st['pool_by_lang'].get(lang,0):,} | {st['heldout_by_lang'].get(lang,0)} |")
lines.append(f"| grammar keywords | - | - | {st['pool_by_lang'].get('grammar',0):,} | "
             f"{st['heldout_by_lang'].get('grammar',0)} |")
lines += ["", "## Kinds", "",
          "| kind | pool 3 | held out |", "|---|---:|---:|"]
for k, v in st["pool_kinds"].items():
    lines.append(f"| {k} | {v:,} | {st['heldout_kinds'].get(k,0)} |")

lines += ["", "## Repositories", "",
          "| language | repo | raw index terms | terms kept in pool 3 |", "|---|---|---:|---:|"]
for (lang, repo), n in sorted(raw_by_repo.items()):
    src = f"gh:{lang}:{repo}"
    lines.append(f"| {lang} | {repo} | {n:,} | {by_src_pool.get(src,0) + by_src_held.get(src,0):,} |")
lines += ["", "Held-out terms are a stratified draw by (kind, language), seed "
          f"{20260918}; they are excluded from `terms_pool3.jsonl`.  No audio has been",
          "synthesised: `research/index_v0/run_gen_pool3.sh` is ready for the sibling with the GPU.", ""]
(OUT / "sources.md").write_text("\n".join(lines))
print(f"wrote {OUT/'sources.md'} ({len(lines)} lines)")
