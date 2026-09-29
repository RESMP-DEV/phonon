"""sizesweep_v0: results.json -> results.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("/home/user/phonon/research/sizesweep_v0")
SETS = [("seen", "heldout_old / seen voices"), ("unseen", "heldout_old / unseen voices"),
        ("new", "heldout_new / new terms + new voices")]
CLASSES = ["ENTITY", "FUNCTION", "ORTHOGRAPHY", "NEAR_MISS", "DROP", "INSERT", "OTHER"]


def f(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def pct(x):
    return "-" if x is None else f"{100*x:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", type=Path, default=None)
    ap.add_argument("--order", default="")
    args = ap.parse_args()
    res = json.loads((OUT / "results.json").read_text())
    terms, real, models = res["terms"], res["real"], res["models"]
    lat = res.get("latency", {})
    order = [x for x in args.order.split(",") if x] or ["raw"] + sorted(models)
    order = [x for x in order if x in terms or x in real]

    L = []
    L.append("# Model size sweep: 350M to E2B on the same acoustic corpus\n")
    L.append("`research/bigrun_v0` fixed the corpus and moved term hit with data; this file asks "
             "what model size moves. Every adapter here is LoRA r=16, lr 2e-4, lora_dropout 0.0, "
             "group_by_length on, on `/data/phonon_bigrun_v0/train_mid.jsonl` (179,633 rows, "
             "14,537 distinct acoustic terms), except `bigrun_big_350m`, which is the same "
             "recipe on the 359,345-row `train_big` and is carried over from `bigrun_v0`. "
             "Every number below is re-measured with the `results_guard.md` length guard ON "
             "(per-row token cap plus the word-count post-check), so the rows are directly "
             "comparable to each other and not identical to the unguarded tables in "
             "`research/bigrun_v0/results.md`.\n")
    if args.read and args.read.exists():
        L.append("## Read\n")
        L.append(args.read.read_text().strip() + "\n")

    # ---- systems ----
    L.append("## Systems\n")
    L.append("| adapter | base | params | corpus rows | epochs | batch | grad ckpt | "
             "train min | rows/s | peak VRAM | adapter MB |")
    L.append("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for lab in order:
        if lab == "raw" or lab not in models:
            continue
        m = models[lab]
        L.append(f"| {lab} | {m['pretty']} | {m['params_b']}B | {m['train_rows']:,} | "
                 f"{m['epochs']} | {m['batch_size']} | {m['grad_checkpointing']} | "
                 f"{f(m['train_min'],1)} | {f(m['rows_per_s'],1)} | "
                 f"{f(m['peak_vram_gb'],1)} GB | {m['adapter_mb']} |")
    L.append("")

    # ---- headline ----
    L.append("## Headline: term hit (norm) with the retrieved list\n")
    L.append("| adapter | seen ret | unseen ret | new ret | seen none | new none | seen oracle "
             "| new oracle | real fair WER (ret) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab in order:
        t = terms.get(lab, {})
        def g(s, c, k="term_hit_norm"):
            return t.get(s, {}).get(c, {}).get(k)
        rw = real.get(lab, {}).get("retrieved" if lab != "raw" else "none", {}).get("fair_wer")
        L.append(f"| {lab} | {f(g('seen','retrieved') or g('seen','none'))} | "
                 f"{f(g('unseen','retrieved') or g('unseen','none'))} | "
                 f"{f(g('new','retrieved') or g('new','none'))} | {f(g('seen','none'))} | "
                 f"{f(g('new','none'))} | {f(g('seen','oracle') or g('seen','none'))} | "
                 f"{f(g('new','oracle') or g('new','none'))} | {f(rw,4)} |")
    L.append("")

    # ---- the conditional question ----
    L.append("## The conditional question: hit when the term IS in the retrieved list\n")
    L.append("The retrieved-list number mixes model skill with retrieval recall. This is the "
             "sub-population where the list actually contains the term.\n")
    L.append("| adapter | seen | unseen | new | seen (n) | unseen (n) | new (n) | "
             "oracle seen | oracle new |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab in order:
        t = terms.get(lab, {})
        def gh(s, c="retrieved"):
            return t.get(s, {}).get(c, {}).get("hit_term_retrieved")
        def gn(s):
            return t.get(s, {}).get("retrieved", {}).get("n_term_retrieved") or \
                t.get(s, {}).get("none", {}).get("n_term_retrieved")
        def go(s):
            return t.get(s, {}).get("oracle", {}).get("term_hit_norm") or \
                t.get(s, {}).get("none", {}).get("term_hit_norm")
        L.append(f"| {lab} | {f(gh('seen') or gh('seen','none'))} | "
                 f"{f(gh('unseen') or gh('unseen','none'))} | "
                 f"{f(gh('new') or gh('new','none'))} | {gn('seen')} | {gn('unseen')} | "
                 f"{gn('new')} | {f(go('seen'))} | {f(go('new'))} |")
    L.append("")

    # ---- full term tables ----
    L.append("## Full term-clip tables\n")
    for key, title in SETS:
        L.append(f"### {title}\n")
        L.append("| adapter | cond | n | hit norm | hit loose | fair WER | strict lc WER | "
                 "damage | unrel | spur | hit when retrieved | hit when missing | guard fired |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for lab in order:
            for cond in ("none", "oracle", "retrieved"):
                b = terms.get(lab, {}).get(key, {}).get(cond)
                if not b:
                    continue
                L.append(f"| {lab} | {cond} | {b['n']} | {f(b['term_hit_norm'])} | "
                         f"{f(b['term_hit_loose'])} | {f(b['fair_wer'],4)} | "
                         f"{f(b['strict_lc_wer'],4)} | {pct(b['damage_rate'])} | "
                         f"{pct(b['false_insert_rate_unrelated'])} | "
                         f"{pct(b['false_insert_spurious'])} | {f(b['hit_term_retrieved'])} | "
                         f"{f(b['hit_term_missing'])} | {b.get('guard_fired','-')} |")
        L.append("")

    # ---- real audio ----
    L.append("## Real audio: the 580-row date-split holdout\n")
    L.append("| system | cond | fair WER | worst row dropped | strict lc WER | damage | win | "
             "tie | loss | ENTITY/1k | false insert | runaway | trunc | guard fired |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab in order:
        for cond in ("none", "retrieved"):
            b = real.get(lab, {}).get(cond)
            if not b:
                continue
            ent = b.get("classes", {}).get("ENTITY")
            L.append(f"| {lab} | {cond} | {f(b['fair_wer'],4)} | "
                     f"{f(b['fair_wer_drop_worst_row'],4)} | {f(b['strict_lc_wer'],4)} | "
                     f"{pct(b['damage_rate'])} | {b['win']} | {b['tie']} | {b['loss']} | "
                     f"{f(ent,1)} | {pct(b['false_insert_rate'])} | {b['runaway_rows']} | "
                     f"{b['truncated_rows']} | {b.get('guard_fired','-')} |")
    L.append("")
    L.append("### Error classes on the real-audio holdout (errors per 1000 reference words)\n")
    L.append("| system | cond | " + " | ".join(CLASSES) + " | total |")
    L.append("|---|---|" + "---:|" * (len(CLASSES) + 1))
    for lab in order:
        for cond in ("none", "retrieved"):
            b = real.get(lab, {}).get(cond)
            if not b:
                continue
            cl = b.get("classes", {})
            vals = [cl.get(c) for c in CLASSES]
            tot = sum(v for v in vals if v is not None)
            L.append(f"| {lab} | {cond} | " + " | ".join(f(v, 1) for v in vals) +
                     f" | {f(tot,1)} |")
    L.append("")

    # ---- v2 lists ----
    v2 = res.get("terms_v2", {})
    if v2 and len(v2) > 1:
        rec = v2.get("recall", {})
        L.append("## With retrieval v2 lists (`research/retrieval_v2`, retrieved only)\n")
        L.append("Same clips, better 30-term lists: recall@30 " +
                 ", ".join(f"{k} {f(v,3)}" for k, v in rec.items()) +
                 ". Unguarded decode, as in `research/retrieval_v2/results.md`.\n")
        L.append("| adapter | new hit (v1) | new hit (v2) | new hit-when-retrieved (v2) | "
                 "unseen hit (v2) | unseen hit-when-retrieved (v2) |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for lab in order:
            b = v2.get(lab)
            if not b:
                continue
            v1new = terms.get(lab, {}).get("new", {}).get("retrieved", {}).get("term_hit_norm")
            L.append(f"| {lab} | {f(v1new)} | "
                     f"{f(b.get('new',{}).get('term_hit_norm'))} | "
                     f"{f(b.get('new',{}).get('hit_term_retrieved'))} | "
                     f"{f(b.get('unseen',{}).get('term_hit_norm'))} | "
                     f"{f(b.get('unseen',{}).get('hit_term_retrieved'))} |")
        L.append("")

    # ---- latency ----
    if lat:
        L.append("## Decode cost (GPU 1, RTX PRO 6000, bf16, LoRA not merged)\n")
        L.append("Real-holdout prompts with the retrieved list, greedy, guard caps, "
                 "48 rows spread over the length distribution.\n")
        L.append("| model | weights GB | bs1 tokens/s | ms/row median | ms/row p90 | "
                 "bs16 rows/s | bs16 tokens/s | adapter MB |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for lab in order:
            b = lat.get(lab)
            if not b:
                continue
            L.append(f"| {lab} | {b['weights_gb']} | {b['decode_tokens_per_s_bs1']} | "
                     f"{b['ms_per_row_median']} | {b['ms_per_row_p90']} | "
                     f"{b.get('rows_per_s_bs16','-')} | {b.get('decode_tokens_per_s_bs16','-')} | "
                     f"{models.get(lab,{}).get('adapter_mb','-')} |")
        L.append("")
    L.append("## Files\n")
    L.append("Scripts `research/sizesweep_v0/` (`run_train_size.sh`, `run_eval_size.sh`, "
             "`bench_latency.py`, `run_bench_size.sh`, `analyze_size.py`, `render_size.py`); "
             "per-row outputs `/data/phonon_sizesweep_v0/refined/`, logs "
             "`/data/phonon_sizesweep_v0/logs/`, adapters "
             "`/data/phonon_corrector_v0/adapters/size_*`.")
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT/'results.md'} ({len('\n'.join(L))} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
