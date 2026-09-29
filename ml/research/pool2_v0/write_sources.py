"""pool2: /data/phonon_pool2_v0/sources.md from pool2_stats.json."""
from __future__ import annotations
import json
from pathlib import Path

D = Path("/data/phonon_pool2_v0")
s = json.loads((D / "pool2_stats.json").read_text())
raw, kept, kbs = s["raw_by_source"], s["pool_sources"], s["kind_by_source"]
hs = s["heldout_sources"]

L = ["# pool 2 term sources", "",
     "Every source is local to gpubox or a public list fetched with plain curl; none of them is",
     "the repo walk that produced `/data/phonon_synth_v0/lexicon_ranked.jsonl`. Terms already in",
     "`terms_pool.jsonl`, `terms_heldout_new.jsonl` or `terms_heldout.jsonl` were dropped",
     "case-insensitively, so pool 2 is disjoint from pool 1 and from every held-out set.", "",
     "| source | what it is | raw | kept | held out | kinds |", "|---|---|---:|---:|---:|---|"]
WHAT = {
    "cuda_headers": "cuda*/cublas*/cudnn* names from /usr/local/cuda/include and cudnn v9 headers",
    "pypi_top4000": "top 4,000 PyPI projects (hugovk top-pypi-packages)",
    "crates_top1000": "top 1,000 crates by downloads (crates.io API, 10 pages)",
    "npm_registry_search": "npm registry search by popularity over 15 generic queries",
    "path_executables": "executables on PATH (/usr/bin, /usr/local/bin, ~/.local/bin, ~/.cargo/bin, ~/.bun/bin)",
    "cli_long_flags": "long flags parsed from `--help` of 63 dev tools",
    "hf_hub": "model and org names under /data/hf/hub",
}
for src in sorted(kept, key=lambda k: -kept[k]):
    what = WHAT.get(src) or f"public names of the installed `{src.split(':', 1)[-1]}` package"
    kinds = ", ".join(f"{k} {v}" for k, v in sorted(kbs.get(src, {}).items(), key=lambda kv: -kv[1]))
    L.append(f"| `{src}` | {what} | {raw.get(src, 0):,} | {kept[src]:,} | {hs.get(src, 0)} | {kinds} |")
L += [f"| **total** | | **{s['raw_candidates']:,}** | **{s['pool_terms']:,}** | "
      f"**{s['heldout_terms']}** | |", ""]
L += ["## Filters", "",
      "Same filters as `research/bigrun_v0/step1_pool.py` plus the lexicon builder's `speakable`:",
      "3+ characters, not a plain lowercase English word, case-insensitive dedupe, no hex or run",
      "ids, at most 40 characters, fewer than 4 underscores, at least 2 letters.", "",
      "| drop | count |", "|---|---:|"]
for k, v in sorted(s["drops"].items(), key=lambda kv: -kv[1]):
    L.append(f"| {k} | {v:,} |")
L += ["", f"Identifiers were capped at {s['identifier_cap']:,} (the cap never bound: "
          f"{s['identifiers_before_cap']:,} survived the filters).", "",
      "## Kinds", "", "| kind | pool | held out |", "|---|---:|---:|"]
hk = s["heldout_kinds"]
for k, v in s["pool_kinds"].items():
    L.append(f"| {k} | {v:,} | {hk.get(k, 0)} |")
L += ["", f"Held-out terms ({s['heldout_terms']}) were drawn stratified by (kind, source) with a",
      "largest-remainder allocation, seed 20260918, and removed from the training pool.", ""]
(D / "sources.md").write_text("\n".join(L))
print(f"wrote {D / 'sources.md'} ({len(L)} lines)")
