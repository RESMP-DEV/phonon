"""STEP 4: research/nbest_v0/results.md from results.json (+ a prose read file)."""
from __future__ import annotations
import argparse, json
from pathlib import Path

OUT = Path("/home/user/phonon/research/nbest_v0")
LABELS = ["bigrun_mid_r16", "nbest_mid_r16"]
SETS = [("new", "heldout_new (300 new terms, 3 unseen voices, pool 15,661)"),
        ("unseen", "heldold unseen voices (pool 4,738)")]
CONDS = ["retrieved", "oracle"]
VARIANTS = ["noalt", "alt"]


def f(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", type=Path, default=None)
    args = ap.parse_args()
    res = json.loads((OUT / "results.json").read_text())
    L = []
    L.append("# Recogniser n-best alternatives in the refiner prompt\n")
    L.append("Parakeet TDT 0.6b v2 decoded with NeMo `maes` beam search (beam 8, "
             "`return_best_hypothesis=False`), 1-best plus up to 4 distinct alternatives. The "
             "prompt gets one extra line after the user text, `ASR alternatives: <span> | ...`, "
             "holding only the word spans where an alternative differs from the 1-best under a "
             "difflib word alignment (punctuation-only differences dropped), capped at 6 spans. "
             "`nbest_mid_r16` is LFM2.5-1.2B trained on train_mid with that line on the acoustic "
             "rows that have alternatives; `bigrun_mid_r16` is the same corpus without it.\n")
    if args.read and args.read.exists():
        L.append("## Read\n")
        L.append(args.read.read_text().strip() + "\n")

    op = res.get("oracle_presence", {})
    if op:
        L.append("## Step 1: does the term survive in the n-best?\n")
        L.append("Gold-term presence in the recogniser output itself (whisper-normalised, "
                 "word-boundary match). `union` = 1-best plus every beam alternative.\n")
        L.append("| set | subset | n | 1-best | beam 1-best | n-best union | gain | "
                 "1-best loose | union loose | rows with a line | mean spans |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key, _ in SETS:
            s = op.get(key, {})
            for tag in ("all", "hard_acoustic"):
                b = s.get(tag)
                if not b:
                    continue
                L.append(f"| {key} | {tag} | {b['n']} | {f(b['presence_1best_norm'])} | "
                         f"{f(b['presence_beambest_norm'])} | {f(b['presence_nbest_union_norm'])} | "
                         f"{f(b['gain_norm'])} | {f(b['presence_1best_loose'])} | "
                         f"{f(b['presence_nbest_union_loose'])} | {f(b['rows_with_alt_line'])} | "
                         f"{f(b['mean_spans_when_line'],2)} |")
        L.append("")

    tb = res.get("train_build", {})
    if tb:
        L.append("## Step 2: the training file\n")
        L.append(f"`train_mid_nbest.jsonl`: {tb.get('rows')} rows, {tb.get('acoustic')} acoustic "
                 f"and {tb.get('real')} real. {tb.get('acoustic_with_nbest')} acoustic rows had an "
                 f"n-best run ({tb.get('acoustic_with_nbest',0)/max(tb.get('acoustic',1),1):.1%} of "
                 f"them) and {tb.get('acoustic_with_line')} carry an alternatives line "
                 f"({tb.get('line_share_of_acoustic',0):.1%} of acoustic rows), "
                 f"{tb.get('mean_spans_when_line',0):.2f} spans each. Real rows are untouched. The gap is the "
                 f"beam decode itself: maes beam 8 runs at 31.2 clips/s on one RTX PRO 6000 "
                 f"against 246 clips/s greedy, so the train_mid n-best pass was stopped at the "
                 f"60-minute cap after 89,280 of 174k clips.\n")

    L.append("## Step 3: term-clip eval, v2 lists, guard on\n")
    for key, title in SETS:
        block = res["sets"].get(key) or {}
        if not block:
            continue
        L.append(f"### {title}\n")
        L.append("| adapter | cond | alts | n | hit norm | hit loose | fair WER | damage | "
                 "unrel ins | hit when retrieved | hit when missing | hit on hard-acoustic |")
        L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k in ["raw|-|-"] + [f"{a}|{c}|{v}" for a in LABELS for c in CONDS for v in VARIANTS]:
            b = block.get(k)
            if not b:
                continue
            a, c, v = k.split("|")
            L.append(f"| {a} | {c} | {v} | {b['n']} | {f(b['term_hit_norm'])} | "
                     f"{f(b['term_hit_loose'])} | {b['fair_wer']:.4f} | "
                     f"{b['damage_rate']*100:.1f}% | "
                     f"{b['false_insert_rate_unrelated']*100:.2f}% | "
                     f"{f(b['hit_term_retrieved'])} | {f(b['hit_term_missing'])} | "
                     f"{f(b['hard_acoustic_hit'])} (n={b['hard_acoustic_n']}) |")
        L.append("")

    real = res.get("real")
    if real:
        L.append("### real audio, 580-row date-split holdout\n")
        L.append("Alternatives line built from the n-best of the user's own clips; only clips "
                 "short enough for beam decoding got one (the rest keep the bare hypothesis).\n")
        L.append("| adapter | cond | alts | n | fair WER | damage |")
        L.append("|---|---|---|---:|---:|---:|")
        for k in sorted(real):
            b = real[k]
            lab, rest = k.split("|", 1)
            parts = rest.split("_")
            cond = parts[0] if parts[0] != "raw" else "-"
            var = parts[1] if len(parts) > 1 else "-"
            L.append(f"| {lab} | {cond} | {var} | {b['n']} | {b['fair_wer']:.4f} | "
                     f"{b['damage_rate']*100:.1f}% |")
        L.append("")

    st = res.get("stats", {})
    if st:
        L.append("## Prompt length and latency\n")
        L.append("| pass | rows | mean prompt tokens | rows/s | guard fallbacks |")
        L.append("|---|---:|---:|---:|---:|")
        for k in sorted(st):
            b = st[k]
            L.append(f"| {k} | {b['rows']} | {b['mean_prompt_tokens']} | {b['rows_per_s']} | "
                     f"{b['guard_fired']} |")
        L.append("")
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT/'results.md'} ({len('\n'.join(L))} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
