"""STEP 5: research/index_v0/results.md + results.json from the step artefacts."""
from __future__ import annotations
import json
from pathlib import Path

D = Path("/data/phonon_index_v0")
P3 = Path("/data/phonon_pool3_v0")
R = Path("/home/user/phonon/research/index_v0")
READ = Path(__file__).with_name("read.md")

st_def = json.loads((D / "index_stats.json").read_text())
st_men = json.loads((D / "index_stats_mentions.json").read_text())
ov_def = json.loads((D / "index_overlap.json").read_text())
ov_men = json.loads((D / "index_overlap_mentions.json").read_text())
bud = json.loads((D / "budget.json").read_text())
p3 = json.loads((P3 / "pool3_stats.json").read_text())
pu = json.loads((D / "peruser.json").read_text())

SETNAME = {"old_seen": "heldout_old/seen (tuning)", "new": "heldout_new",
           "real": "real-audio holdout", "pool2": "heldout_pool2"}
L = []
A = L.append

A("# Symbol index, adaptive list budget, and term pool 3")
A("")
A("`research/retrieval_v2/results.md` left three things open: the product is supposed to build the")
A("30-term list from the user's own repositories offline, a fixed 30-slot list caps term-dense")
A("utterances (13 of the 580 real rows carry more than 30 lexicon terms), and recall keeps falling")
A("as the pool grows.  This run builds the index the product would ship, replaces the fixed list")
A("with a budget, and adds a third term pool.  Everything here is CPU-only on gpubox; no audio was")
A("synthesised.")
A("")
A("## Read")
A("")
A(READ.read_text().strip() if READ.exists() else "_(read pending)_")
A("")

# ---------------------------------------------------------------- step 1
A("## 1. The symbol index")
A("")
A("`research/index_v0/symbol_index.py` walks a repo tree and emits `(term, kind, count, repos,")
A("langs)` with no model in the loop.  Tier A is tree-sitter definitions (Python, Rust, C/C++/CUDA,")
A("JS/TS/TSX, Go via `tree-sitter-language-pack`): functions, classes, structs, enums, traits,")
A("types, macros, constants, modules, imports and includes, plus Markdown headings and code spans,")
A("config keys in toml/yaml/json/ini, and `--flags` in docs and shell.  Tier B (`--mentions`) adds")
A("every technical-looking token anywhere in the file plus a fixed command/host/person seed list,")
A("which is how the original lexicon got `kubectl`, `ffmpeg` and `Infatoshi` - names that never")
A("appear at a definition site.")
A("")
A("Roots: `~/dev`, `~/phonon`, `~/kernels`, `~/research`, `~/sites`, `~/nlanes`, `~/kernelbench.com`.")
A("The lexicon's original roots `~/cuda`, `~/benchmarks`, `~/experiments` and `~/persona-gym` no")
A("longer exist on the current GPU box (they were on an older box before the rebuild), which costs tier A part of its")
A("overlap and costs step 4 a third of its clips.")
A("")
A("| | tier A: definitions | tier A+B: + mentions |")
A("|---|---:|---:|")
A(f"| files walked | {st_def['files_seen']:,} | {st_men['files_seen']:,} |")
A(f"| terms | {ov_def['index_terms']:,} | {ov_men['index_terms']:,} |")
A(f"| walk seconds | {st_def['walk_seconds']} | {st_men['walk_seconds']} |")
A(f"| extract seconds ({st_def['workers']}/{st_men['workers']} workers) | {st_def['parse_seconds']} | {st_men['parse_seconds']} |")
A(f"| **seconds per 1,000 files** | **{st_def['seconds_per_1k_files']}** | **{st_men['seconds_per_1k_files']}** |")
A(f"| overlap with the 15,661-term ranked lexicon | {ov_def['overlap_with_ranked']:,} | {ov_men['overlap_with_ranked']:,} |")
A(f"| **share of that lexicon recovered** | **{ov_def['recall_of_ranked_lexicon']:.3f}** | **{ov_men['recall_of_ranked_lexicon']:.3f}** |")
A(f"| overlap with the 979,675-term raw walk | {ov_def['overlap_with_raw']:,} | {ov_men['overlap_with_raw']:,} |")
A(f"| terms the raw walk never had | {ov_def['new_vs_raw']:,} | {ov_men['new_vs_raw']:,} |")
A(f"| ... of those, seen 5+ times | {ov_def['new_vs_raw_count_ge5']:,} | {ov_men['new_vs_raw_count_ge5']:,} |")
A(f"| `heldout_new` terms found (of 300) | {ov_def['heldout_new_found']} | {ov_men['heldout_new_found']} |")
A(f"| `heldout_old` terms found (of 300) | {ov_def['heldout_old_found']} | {ov_men['heldout_old_found']} |")
A("")
A("Ranked-lexicon terms tier A misses, by kind: "
  + ", ".join(f"{k} {v}" for k, v in ov_def["ranked_missed_by_kind"].items()) + ".")
A("With mentions on, the misses fall to "
  + ", ".join(f"{k} {v}" for k, v in ov_men["ranked_missed_by_kind"].items()) + ".")
A("")
A("Kinds, tier A: " + ", ".join(f"{k} {v:,}" for k, v in list(st_def["kinds"].items())) + ".")
A("")

# ---------------------------------------------------------------- step 2
A("## 2. Adaptive list budget")
A("")
A(f"The list is the top `min({bud['cap']}, max(30, {bud['mult']} x hypothesis words))` terms with")
A("score at or above a threshold, instead of a flat top 30.  Lists come from the shipped v2")
A("two-stage retriever (cheap char/metaphone pass at n<=6, rerank the top 400 with the five spoken")
A("realisations, g2p phonemes at 0.4/0.4/0.2 and the kind log-odds prior).  The threshold was swept")
A(f"on `heldout_old/seen` alone and {bud['threshold']} was chosen (largest threshold within 0.002 of")
A("the best tuning recall); it is applied unchanged everywhere else.")
A("")
A("| set | rows | mean words | mean gold terms | r@10 | r@30 (fixed) | r@budget | mean list | p90 list | 30-slot ceiling | budget ceiling |")
A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for s in ["old_seen", "new", "real", "pool2"]:
    e = bud["sets"][s]
    A(f"| {SETNAME[s]} | {e['rows']} | {e['mean_words']} | {e['mean_gold']} | "
      f"{e['fixed10']['recall']:.4f} | {e['fixed30']['recall']:.4f} | "
      f"**{e['budget_thrT']['recall']:.4f}** | {e['budget_thrT']['mean_len']:.1f} | "
      f"{e['budget_thrT']['p90_len']} | {e['fixed30']['slot_ceiling']:.4f} | "
      f"{e['budget_thrT']['slot_ceiling']:.4f} |")
A("")
A("Real-audio rows by how many lexicon terms the reference carries - this is where the fixed list")
A("was losing:")
A("")
A("| gold terms in the row | rows | gold pairs | r@30 | r@budget | 30-slot ceiling | budget ceiling |")
A("|---|---:|---:|---:|---:|---:|---:|")
for k, v in bud["real_by_gold_count"].items():
    A(f"| {k} | {v['rows']} | {v['gold_pairs']} | {v['fixed30']:.4f} | **{v['budget']:.4f}** | "
      f"{v['fixed30_ceiling']:.4f} | {v['budget_ceiling']:.4f} |")
A("")
A("List-length distribution under the shipped rule (count of clips by list length, buckets of 10):")
A("")
A("| set | " + " | ".join(f"{k}-{k+9}" for k in range(0, 90, 10)) + " |")
A("|---|" + "---:|" * 9)
for s in ["new", "real", "pool2"]:
    h = bud["sets"][s]["len_hist_by10"]
    A(f"| {SETNAME[s]} | " + " | ".join(str(h.get(str(k), h.get(k, 0))) for k in range(0, 90, 10)) + " |")
A("")
A("Threshold is the only free parameter that trades length for recall; the budget itself is nearly")
A("free because the scores above rank 30 are still high:")
A("")
A("| score threshold | " + " | ".join(SETNAME[s] for s in ["old_seen", "new", "real", "pool2"]) + " |")
A("|---|---:|---:|---:|---:|")
for row in bud["threshold_table"]:
    A(f"| {row['thr']} | " + " | ".join(f"{row[s]:.4f} / len {row[f'{s}_len']}"
                                        for s in ["old_seen", "new", "real", "pool2"]) + " |")
A("")
A("Rule sweep (recall / mean list length, threshold fixed at the tuned value):")
A("")
A("| words x | cap | " + " | ".join(SETNAME[s] for s in ["old_seen", "new", "real", "pool2"]) + " |")
A("|---|---:|---:|---:|---:|---:|")
for row in bud["rule_sweep"]:
    A(f"| {row['mult']} | {row['cap']} | " + " | ".join(f"{row[s]:.4f} / {row[f'{s}_len']}"
                                                        for s in ["old_seen", "new", "real", "pool2"]) + " |")
A("")

# ---------------------------------------------------------------- step 3
A("## 3. Term pool 3")
A("")
A(f"150 public repositories, 25 per language across Python, Rust, C/C++, CUDA, JS/TS and Go,")
A("shallow-cloned, indexed with the same extractor (tier A only), clones deleted.  Plus the")
A("tree-sitter grammars' own keyword lists.  Filters are pool 2's: 3+ chars, speakable, not plain")
A("English, deduped against pools 1-2 and every held-out set; a term must occur at least")
A(f"{p3['min_count']} times in its repo; each language capped at {p3['per_lang_cap']} terms.")
A("")
A("| language | repos | raw index terms | in pool 3 | held out |")
A("|---|---:|---:|---:|---:|")
for lang in sorted(p3["repos_by_lang"]):
    A(f"| {lang} | {p3['repos_by_lang'][lang]} | {p3['raw_terms_by_lang'].get(lang,0):,} | "
      f"{p3['pool_by_lang'].get(lang,0):,} | {p3['heldout_by_lang'].get(lang,0)} |")
A(f"| tree-sitter grammars | - | {p3['grammar_keywords']} | {p3['pool_by_lang'].get('grammar',0)} | "
  f"{p3['heldout_by_lang'].get('grammar',0)} |")
A(f"| **total** | **150** | **{sum(p3['raw_terms_by_lang'].values()):,}** | "
  f"**{p3['pool_terms']:,}** | **{p3['heldout_terms']}** |")
A("")
A("Drops: " + ", ".join(f"{k} {v:,}" for k, v in p3["drops"].items())
  + f"; {p3['after_filter']:,} terms survive the filters and the per-language cap keeps the "
    f"highest-ranked {p3['per_lang_cap']} of each.")
A("")
A("Kinds in pool 3: " + ", ".join(f"{k} {v:,}" for k, v in p3["pool_kinds"].items()) + ".")
A(f"Held-out examples: {', '.join('`' + t + '`' for t in p3['heldout_examples'][:12])}.")
A("")
A("Files: `/data/phonon_pool3_v0/terms_pool3.jsonl`, `terms_heldout_pool3.jsonl`, `sources.md`,")
A("`pool3_stats.json`, `per_repo/*.jsonl` (the 150 per-repo indices).  No audio: both GPUs belonged")
A("to siblings tonight, so `research/index_v0/run_gen_pool3.sh` is left ready for the sentence")
A("generator (3 per term, `research/scaling_v0/gen_more.py`), to be followed by the TTS jobs the")
A("way `research/bigrun_v0/make_heldout_jobs.py` builds them.")
A("")

# ---------------------------------------------------------------- step 4
A("## 4. What a per-user index buys")
A("")
A("Condition (a) is the 38,253-term union pool; condition (b) restricts the pool, per clip, to the")
A("symbol index of the repository the gold term actually came from (`heldout_new`) or to the pool-2")
A("source the gold term came from (`heldout_pool2`).  Both conditions use the same scorer - five")
A("realisations, n-grams to 8 words, 0.5 char / 0.5 metaphone, kind prior, no g2p phonemes (the")
A("per-user pools bring hundreds of thousands of uncached words and warming g2p for them would cost")
A("half an hour without changing the comparison), so these numbers are a controlled A/B and are not")
A("the shipped retriever's absolute recall.")
A("")
A("| set | condition | clips | mean pool | r@10 | r@30 | r@budget | mean list |")
A("|---|---|---:|---:|---:|---:|---:|---:|")


def row(set_name, label, key, base=None):
    e = pu[set_name].get(key)
    if not e:
        return
    A(f"| {set_name} | {label} | {e.get('clips', e['pairs'])} | {e['mean_pool']:,.0f} | "
      f"{e['recall@10']:.4f} | {e['recall@30']:.4f} | {e['recall@budget']:.4f} | {e['mean_len']:.1f} |")


row("heldout_new", "(a) full union pool", "full_union")
row("heldout_new", "(b) own repo, definitions tier", "per_user_definitions_all_clips")
row("heldout_new", "(b) definitions tier, repo still on disk", "per_user_definitions_repo_present")
row("heldout_new", "(b) definitions tier, gold term in the index", "per_user_definitions_gold_in_index")
row("heldout_new", "(a) same clips", "full_union_definitions_gold_in_index")
row("heldout_new", "(b) own repo, mentions tier", "per_user_mentions_all_clips")
row("heldout_new", "(b) mentions tier, repo still on disk", "per_user_mentions_repo_present")
row("heldout_new", "(a) same clips", "full_union_mentions_repo_present")
row("heldout_new", "(b) mentions tier, gold term in the index", "per_user_mentions_gold_in_index")
row("heldout_new", "(a) same clips", "full_union_mentions_gold_in_index")
row("heldout_pool2", "(a) full union pool", "full_union")
row("heldout_pool2", "(b) own pool-2 source", "per_user_index")
A("")
A("Three things drive those numbers.  One, coverage: the definitions tier holds the gold term for")
A("only 558 of the 2,700 clips and the mentions tier for 1,674, and 936 clips point at `~/cuda`,")
A("`~/benchmarks` or `~/experiments`, trees that no longer exist on this machine, so their")
A("per-user pool is empty by construction.  Two, size: a per-user pool is not automatically a small")
A("pool - `dev/sites` alone is 181,108 definition terms and 807,903 with mentions, five to twenty")
A("times the whole union pool, and the mean `heldout_new` per-user pool is 225,800 terms, which is")
A("why restricting to it *loses* 0.139 of recall@30 against the union pool on the same clips.")
A("Three, the `heldout_pool2` win is real but flattered: a pool-2 source group always contains its")
A("own held-out term, so that row isolates the false-neighbour effect - at 3,023 terms instead of")
A("38,253 the gold term rises from rank 30 into the top 10 for 0.224 more of the clips and the")
A("budgeted list gets *shorter* (33.2 against 36.9).  The product lesson is that the context model")
A("has to rank and cap the index, not just extract it.")
A("")
A("## Files")
A("")
A("Scripts `research/index_v0/`: `symbol_index.py` (the extractor), `step1_analyze.py`,")
A("`clone_extract.sh` + `pool3_repos.txt` (the 150 public repos), `step3_pool3.py`,")
A("`write_sources.py`, `step2_lists.py`, `step2_budget.py`, `step4_peruser.py`,")
A("`run_gen_pool3.sh` (not run), `render_results.py`, `read.md`.")
A("Data `/data/phonon_index_v0/`: `symbols_local.jsonl` (tier A),")
A("`symbols_local_mentions.jsonl` (tier A+B), `index_stats*.json`, `index_overlap*.json`,")
A("`by_repo*.json`, `heldnew_term2repo.json`, `lists_{old_seen,new,real,pool2}.json` (top-100")
A("ranked lists per clip), `budget.json`, `peruser.json`, `logs/`.  Data")
A("`/data/phonon_pool3_v0/`: `terms_pool3.jsonl`, `terms_heldout_pool3.jsonl`, `sources.md`,")
A("`pool3_stats.json`, `per_repo/` (150 files), `logs/`.")

(R / "results.md").write_text("\n".join(L) + "\n")
res = {"index": {"definitions": {**st_def, **ov_def}, "mentions": {**st_men, **ov_men}},
       "budget": bud, "pool3": p3, "per_user": pu}
for k in ("ranked_missed_examples", "new_examples"):
    res["index"]["definitions"].pop(k, None)
    res["index"]["mentions"].pop(k, None)
(R / "results.json").write_text(json.dumps(res, indent=1) + "\n")
print(f"wrote {R/'results.md'} ({len(L)} lines) and results.json")
