"""bigrun_v0 eval step 4: results.json -> research/bigrun_v0/results.md."""
from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path("/home/user/phonon/research/bigrun_v0")
RES = OUT / "results.json"
READ = OUT / "read.md"
CLASSES = ["ENTITY", "FUNCTION", "ORTHOGRAPHY", "NEAR_MISS", "DROP", "INSERT", "OTHER"]
SETS = [("seen", "heldout_old / seen voices"), ("unseen", "heldout_old / unseen voices"),
        ("new", "heldout_new / new terms + new voices")]
CONDS = ["none", "retrieved", "oracle"]
ORDER = ["raw", "scale_n36000", "bigrun_mid_r16", "bigrun_mid_r64", "bigrun_big_r16",
         "bigrun_big_r16_personal", "bigrun_big_350m", "bigrun_mid_generic",
         "bigrun_mid_generic_personal"]
DESC = {
    "raw": ("-", "-", "Parakeet TDT 0.6b v2 hypothesis, no refiner"),
    "scale_n36000": ("LFM2.5-1.2B", "2,107", "36,000 acoustic rows, scaling_v0 best"),
    "bigrun_mid_r16": ("LFM2.5-1.2B", "14,537", "train_mid 179,633 rows, r=16, 2 epochs"),
    "bigrun_mid_r64": ("LFM2.5-1.2B", "14,537", "train_mid 179,633 rows, r=64, 2 epochs"),
    "bigrun_big_r16": ("LFM2.5-1.2B", "14,537", "train_big 359,345 rows, r=16"),
    "bigrun_big_r16_personal": ("LFM2.5-1.2B", "14,537",
                                "stage 2: bigrun_big_r16 + 3,827 real rows, 1 epoch, lr 1e-4"),
    "bigrun_big_350m": ("LFM2.5-350M", "14,537", "train_big 359,345 rows, r=16, 2 epochs"),
}


def f(x, p=3):
    return "-" if x is None else f"{x:.{p}f}"


def pct(x, p=1):
    return "-" if x is None else f"{x*100:.{p}f}%"


def main() -> int:
    res = json.loads(RES.read_text())
    T, R = res["terms"], res["real"]
    labels = [l for l in ORDER if l in T or l in R]
    L: list[str] = []
    A = L.append

    A("# The big run: a 14,538-term acoustic corpus")
    A("")
    A("`research/bigrun_v0/corpus.md` built the corpus; this file evaluates the adapters trained "
      "on it. The question is whether the scaling law of `research/scaling_v0` "
      "(term hit = 0.468 + 0.107 x log10(distinct terms), fitted over 30 to 2,124 terms) still "
      "holds at 14,537 terms, whether rank 64 buys anything, whether a never-trained term and "
      "voice distribution tracks the old held-out set, and what a personal stage-2 pass does to "
      "the user's own audio.")
    A("")
    A("## Read")
    A("")
    A(READ.read_text().strip() if READ.exists() else "<<READ>>")
    A("")

    A("## Systems")
    A("")
    A("| adapter | base | distinct acoustic terms | corpus | train rows | epochs | r | lr | "
      "train min | peak VRAM |")
    A("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for l in labels:
        b, t, d = DESC.get(l, ("?", "?", "?"))
        m = {}
        mp = Path(f"/data/phonon_corrector_v0/adapters/{l}/train_meta.json")
        if mp.exists():
            try:
                m = json.loads(mp.read_text())
            except Exception:
                m = {}
        cells = ("-", "-", "-", "-", "-", "-")
        if m:
            cells = (f"{m['train_rows']:,}", f"{m['epochs']:g}", str(m["lora_r"]),
                     f"{m['lr']:g}", f"{m['seconds']/60:.0f}",
                     f"{m['peak_vram_bytes']/1e9:.0f} GB")
        A(f"| {l} | {b} | {t} | {d} | " + " | ".join(cells) + " |")
    A("")

    A("## Retrieval on the three eval sets")
    A("")
    A("Lists are computed from the raw hypothesis alone. `heldout_new` uses the full-lexicon + "
      "both-held-out-sets pool of `step5b_lists.py` (15,661 terms), the same pool the training "
      "lists came from; `heldout_old` keeps the pools its conditions were built with, so its "
      "recall is not directly comparable.")
    A("")
    A("| eval set | rows | pool | recall@10 | recall@30 |")
    A("|---|---:|---:|---:|---:|")
    for k, name in SETS:
        r = res["retrieval"].get(k, {})
        pool = r.get("pool") or r.get("pool_eval")
        if pool is None and k == "seen":
            pool = res["retrieval"].get("unseen", {}).get("pool")
        A(f"| {name} | {r.get('rows', 2700)} | {pool or '-'} | "
          f"{f(r.get('recall@10'))} | {f(r.get('recall@30'))} |")
    A("")

    A("## Headline: term hit (norm) with the retrieved list")
    A("")
    A("| adapter | seen ret | unseen ret | new ret | seen none | new none | seen oracle | "
      "new oracle | real fair WER (ret) |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for l in labels:
        def h(s, c):
            b = T.get(l, {}).get(s, {}).get(c if l != "raw" else "none")
            return f(b["term_hit_norm"]) if b else "-"
        rb = R.get(l, {}).get("retrieved") or R.get(l, {}).get("none")
        A(f"| {l} | {h('seen','retrieved')} | {h('unseen','retrieved')} | {h('new','retrieved')} "
          f"| {h('seen','none')} | {h('new','none')} | {h('seen','oracle')} | "
          f"{h('new','oracle')} | {f(rb['fair_wer'], 4) if rb else '-'} |")
    A("")

    A("## Full term-clip table")
    A("")
    A("`hit norm` = whisper-normalised term present in the output; `hit loose` = "
      "punctuation-squashed substring. `damage` = clips whose fair WER is worse than the raw "
      "hypothesis. `unrel` = a list term written that the reference lacks and that is not a "
      "spelling variant of the clip's own term. `spur` = an unrelated insertion on a clip whose "
      "own term was recovered anyway.")
    A("")
    for k, name in SETS:
        A(f"### {name}")
        A("")
        A("| adapter | cond | n | hit norm | hit loose | fair WER | strict lc WER | damage | "
          "unrel | spur | hit when term retrieved | hit when term missing |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for l in labels:
            if l not in T:
                continue
            for c in (["none"] if l == "raw" else CONDS):
                b = T[l].get(k, {}).get(c)
                if not b:
                    continue
                A(f"| {l} | {c} | {b['n']} | {f(b['term_hit_norm'])} | {f(b['term_hit_loose'])} | "
                  f"{f(b['fair_wer'], 4)} | {f(b['strict_lc_wer'], 4)} | {pct(b['damage_rate'])} | "
                  f"{pct(b['false_insert_rate_unrelated'], 2)} | "
                  f"{pct(b['false_insert_spurious'], 2)} | {f(b['hit_term_retrieved'])} | "
                  f"{f(b['hit_term_missing'])} |")
        A("")

    if R:
        A("## Real audio: the date-split holdout (580 rows of the user's own dictation)")
        A("")
        A("Parakeet on the user's recordings, reference = the text he accepted, split by date so "
          "every holdout row is later than every training row. `win`/`loss` count rows whose fair "
          "WER beats or trails the raw hypothesis.")
        A("")
        A("| system | cond | fair WER | worst row dropped | strict lc WER | damage | win | tie "
          "| loss | ENTITY/1k | false insert | not in raw | runaway | truncated |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for l in labels:
            if l not in R:
                continue
            for c, b in R[l].items():
                A(f"| {l} | {c} | {f(b['fair_wer'], 4)} | "
                  f"{f(b.get('fair_wer_drop_worst_row'), 4)} | {f(b['strict_lc_wer'], 4)} | "
                  f"{pct(b['damage_rate'])} | {b['win']} | {b['tie']} | {b['loss']} | "
                  f"{b['classes']['ENTITY']:.1f} | {pct(b['false_insert_rate'], 2)} | "
                  f"{pct(b['false_insert_rate_strict'], 2)} | {b.get('runaway_rows', '-')} | "
                  f"{b.get('truncated_rows', '-')} |")
        A("")
        A("`worst row dropped` is the same WER with each system's single largest error "
          "contributor removed; `runaway` counts outputs over twice the reference length and "
          "`truncated` outputs under half of it. The 580 references are 36,915 normalised words, "
          "so one lost 358-word row is about 0.010 WER.")
        A("")
        A("### Error classes on the real-audio holdout (errors per 1000 reference words)")
        A("")
        A("| system | cond | " + " | ".join(CLASSES) + " | total |")
        A("|---|---|" + "---:|" * (len(CLASSES) + 1))
        for l in labels:
            if l not in R:
                continue
            for c, b in R[l].items():
                cl = b["classes"]
                A(f"| {l} | {c} | " + " | ".join(f"{cl[x]:.1f}" for x in CLASSES)
                  + f" | {sum(cl.values()):.1f} |")
        A("")

    ft = res.get("fit_terms", {})
    if ft:
        A("## The scaling curve, extended to 14,537 terms")
        A("")
        A("Seen-voice term hit with a retrieved list against distinct acoustic training terms.")
        A("")
        A("| adapter | distinct terms | acoustic rows | seen hit ret | source |")
        A("|---|---:|---:|---:|---|")
        rows_by_label = {"scale_n500": 500, "scale_n1000": 1000, "scale_n2000": 2000,
                         "scale_n4000": 4000, "scale_n9000": 9000, "scale_n18000": 18000,
                         "scale_n36000": 36000, "scale_axis_terms": 8900,
                         "scale_axis_sentences": 8900, "scale_axis_voices": 8900,
                         "bigrun_mid_r16": 164325, "bigrun_mid_r64": 164325,
                         "bigrun_big_r16": 328729, "bigrun_big_350m": 328729,
                         "bigrun_mid_generic": 164325}
        for name, terms, hit in sorted(ft.get("scaling_v0_points", []), key=lambda p: p[1]):
            A(f"| {name} | {terms} | {rows_by_label.get(name, '-')} | {f(hit)} | scaling_v0 |")
        for name, terms, hit in sorted(ft.get("bigrun_points", []), key=lambda p: p[1]):
            A(f"| {name} | {terms} | {rows_by_label.get(name, '-')} | {f(hit)} | this run |")
        A("")
        a = ft["scaling_v0_only"]
        A(f"Prior fit (10 adapters, 30 to 2,124 terms): hit = {a['intercept']:.3f} + "
          f"{a['slope_per_decade']:.3f} x log10(terms), R^2 {a['r2']:.3f}, "
          f"RMSE {a['rmse']:.4f}.")
        A("")
        if "extended" in ft:
            e = ft["extended"]
            A(f"Extended fit (+{len(e['points'])-len(ft['scaling_v0_points'])} LFM2.5-1.2B "
              f"adapters at 14,537 terms): hit = {e['intercept']:.3f} + "
              f"{e['slope_per_decade']:.3f} x log10(terms), R^2 {e['r2']:.3f}, "
              f"RMSE {e['rmse']:.4f}.")
            A("")
            if "projected_at_14537_from_scaling_v0" in ft:
                A(f"The prior fit projected {ft['projected_at_14537_from_scaling_v0']:.3f} at "
                  f"14,537 terms. The measured slope from 2,107 to 14,537 terms is "
                  f"{ft['slope_2107_to_14537']:.3f} per decade against "
                  f"{a['slope_per_decade']:.3f} over the earlier range.")
                A("")

    A("## Files")
    A("")
    A("Scripts `research/bigrun_v0/` (`build_new_conditions.py`, `refine_bigrun.py`, "
      "`step6_personal.py`, `analyze_bigrun.py`, `render_bigrun.py`, runners `run_eval.sh` and "
      "`run_train2.sh`), corpus and eval data "
      "`/data/phonon_bigrun_v0/` (`eval_conditions_new.jsonl`, `train_personal.jsonl`, "
      "per-row outputs `refined/`), adapters `/data/phonon_corrector_v0/adapters/bigrun_*`, "
      "logs `/data/phonon_bigrun_v0/logs/`.")
    A("")
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT/'results.md'} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
