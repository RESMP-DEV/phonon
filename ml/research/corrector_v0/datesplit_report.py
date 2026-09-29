"""Write research/corrector_v0/results_datesplit.{md,json} from the date-split eval."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SPLIT = Path("/data/phonon_corrector_v0/datesplit")
OUT = Path("/home/user/phonon/research/corrector_v0")

# id-split numbers, copied from the files named in each value's "src"
ID = {
    "raw": {"audio": 0.1886, "audio_strict": 0.2705, "text": 0.1393, "text_strict": 0.2157,
            "src": "results_bases.json raw_input / results_curve2.md"},
    "date_lfm12_full_e2": {"peer": "lfm2.5-1.2b", "audio": 0.0918, "audio_strict": 0.1459,
                           "text": 0.0823, "text_strict": None, "dmg_audio": 0.0583,
                           "src": "results_bases.json"},
    "date_lfm12_n250_e12": {"peer": "curve_lfm12_n250_e12", "audio": 0.1478, "audio_strict": 0.2096,
                            "text": 0.1192, "text_strict": 0.1700, "dmg_audio": 0.1000,
                            "src": "results_coldctl.md"},
    "date_lfm12_n500_e8": {"peer": "curve_lfm12_n500_e8", "audio": 0.1275, "audio_strict": 0.1959,
                           "text": 0.1222, "text_strict": 0.1702, "dmg_audio": 0.0750,
                           "src": "results_coldctl.md"},
    "date_lfm12_n1000_e2": {"peer": "curve_lfm12_n1000_e2", "audio": 0.1159, "audio_strict": 0.1847,
                            "text": 0.0812, "text_strict": 0.1263, "dmg_audio": 0.0500,
                            "src": "results_coldctl.md"},
    "date_q06_full_e2": {"peer": "qwen3-0.6b", "audio": 0.0865, "audio_strict": 0.1388,
                         "text": 0.0778, "text_strict": None, "dmg_audio": 0.0750,
                         "src": "results_bases.json"},
}
ORDER = ["raw", "date_lfm12_n250_e12", "date_lfm12_n500_e8", "date_lfm12_n1000_e2",
         "date_lfm12_full_e2", "date_q06_full_e2",
         "idsplit_lfm2.5-1.2b_CONTAMINATED", "idsplit_qwen3-0.6b_CONTAMINATED"]


def f(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ev = json.loads((SPLIT / "eval_datesplit.json").read_text())
    counts = json.loads((SPLIT / "datesplit_counts.json").read_text())
    by = {m["label"]: m for m in ev["models"]}
    raw_t = by["raw"]["sets"]["holdout_date_text"]["baseline"]
    raw_a = by["raw"]["sets"]["holdout_date_audio"]["baseline"]
    old_ids = {r["id"][3:] if r["id"].startswith("pk_") else r["id"]
               for r in (json.loads(ln) for ln in
                         open("/data/phonon_corrector_v0/train.jsonl", encoding="utf-8") if ln.strip())}
    hold_ids = {json.loads(ln)["id"] for ln in
                open(SPLIT / "holdout_date_text.jsonl", encoding="utf-8") if ln.strip()}
    leak_in_holdout = len(old_ids & hold_ids)

    L = []
    A = L.append
    A("# Date-split re-check of the personal refiner\n")
    A("The corrector numbers in `results_bases.md`, `results_curve*.md` and `results_coldctl.md` "
      "sit on an id-hash split: 100 percent of the holdout rows fall on days that also have "
      "training rows (`holdout_provenance.md`). This run rebuilds the split by date and "
      "retrains, so nothing the model saw comes from a day it is scored on.\n")

    A("## The split\n")
    c = counts
    td = c["train_date"]
    A(f"Cutoff **{c['cutoff']}**: the last {c['holdout_days']} of {c['total_days']} dictation days "
      f"({c['holdout_day_range'][0]} .. {c['holdout_day_range'][1]}) are holdout, everything "
      f"strictly earlier may train.\n")
    A("| set | rows | days | first | last |")
    A("|---|---:|---:|---|---|")
    A(f"| train_date | {td['n']} | {td['days']} | {td['date_range'][0]} | {td['date_range'][1]} |")
    for name in ("holdout_date_text", "holdout_date_audio"):
        h = c["holdouts"][name]
        A(f"| {name} | {h['n']} | {h['days']} | {h['date_range'][0]} | {h['date_range'][1]} |")
    A("")
    A(f"`train_date.jsonl` is {td['wispr_asr_rows']} Wispr-ASR-input rows plus "
      f"{td['parakeet_rows']} Parakeet-raw-input rows over the same pre-cutoff audio clips, "
      f"deduped on (input, target) exactly as `build_train.py` does.\n")
    A("Two deviations from the task sketch, both forced by the data:\n")
    A("1. The 519 `parakeet:wispr_train` rows in `train.jsonl` are **not** synthetic TTS and are "
      "**not** date-free: they are Parakeet transcriptions of the user's own audio and carry the "
      "same timestamps. 377 of them fall on or after the cutoff, so they are date-filtered like "
      f"everything else and only {td['parakeet_rows']} survive. `tts_pairs.jsonl` contributed 0 rows to "
      "`train.jsonl` (all were dropped by the (input, target) dedupe), so there is no synthetic "
      "block to carry over.")
    A(f"2. Every one of the {c['export_pairs_on_or_after_cutoff']} post-cutoff export rows has audio "
      "(Wispr keeps wavs only for recent dictations; all 3,539 text-only rows are pre-cutoff). "
      "There is no text-only tail to hold out, so the two holdouts are the **same 580 rows in the "
      "two input modes**: `holdout_date_text` feeds Wispr ASR text (the `wispr_text_holdout` "
      "convention), `holdout_date_audio` feeds Parakeet v2 raw (the `wispr_holdout120` "
      "convention). One row was dropped as a verbatim repeat of a pre-cutoff utterance "
      "(\"Done.\").\n")
    hh = c["holdouts"]["holdout_date_audio"]
    A(f"Overlap check against `train_date.jsonl`: id overlap {hh['id_overlap_with_train']}, "
      f"exact input-text overlap {hh['input_text_overlap_with_train_inputs']}, exact "
      f"reference-text overlap {hh['reference_overlap_with_train_targets']} — same for both "
      "holdouts. Of the 580 holdout rows, "
      f"{hh['from_old_holdout120']} came from the old `wispr_holdout120` and "
      f"{hh['from_old_train_splits']} carried a training split label under the id split, of which "
      f"{leak_in_holdout} are actual rows of `train.jsonl` (the rest were lost to its dedupe).\n")
    A(f"That is the size of the old leak: {c['id_split_train_jsonl']['rows_on_or_after_cutoff (leak under a date split)']} "
      f"of the {c['id_split_train_jsonl']['n']} rows in `train.jsonl` are dictations from the "
      "11 holdout days.\n")
    A("Curve subsets are the earliest N rows by timestamp (`train_date_first{250,500,1000}.jsonl`), "
      "Parakeet duplicates sorted last, the same convention as `make_curve.py`: "
      + ", ".join(f"n{n} spans {c['curve_subsets'][f'n{n}']['span'][0]}..{c['curve_subsets'][f'n{n}']['span'][1]}"
                  for n in (250, 500, 1000)) + ".\n")

    A("## Date split vs id split\n")
    A("Left block: this run, LFM2.5-1.2B (and Qwen3-0.6B) trained on `train_date` and scored on the "
      "580 post-cutoff rows. Right block: the published id-split number for the same recipe, "
      "scored on `wispr_holdout120` (audio) and `wispr_text_holdout` (text).\n")
    A("| system | n | date text fair | date audio fair | date text strict | date audio strict "
      "| date audio damage | id text fair | id audio fair | text delta | audio delta |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in ORDER:
        m = by.get(label)
        if not m:
            continue
        t = m["sets"]["holdout_date_text"]
        a = m["sets"]["holdout_date_audio"]
        ref = ID.get(label, {})
        tf = t["corrector"]["fair_wer"]
        af = a["corrector"]["fair_wer"]
        it, ia = ref.get("text"), ref.get("audio")
        A(f"| {label} | {t['n']} | {f(tf)} | {f(af)} | {f(t['corrector']['strict_lc_wer'])} "
          f"| {f(a['corrector']['strict_lc_wer'])} | {f(a['damage_rate'])} | {f(it)} | {f(ia)} "
          f"| {f(tf - it, 4) if it else '-'} | {f(af - ia, 4) if ia else '-'} |")
    A("")
    A("The two `idsplit_*_CONTAMINATED` rows are the **published** adapters "
      "(`adapters/lfm2.5-1.2b`, `adapters/qwen3-0.6b`, trained on `train.jsonl`) run on the date "
      f"holdout. {hh['from_old_train_splits']} of those 580 rows are in their training set, so those "
      "two numbers are a contamination reading, not a result: they show what memorisation of "
      "these days looks like.\n")

    A("## Paired win / tie / loss vs raw input\n")
    A("| system | set | win | tie | loss | damage rate | mean per-row fair-WER delta |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for label in ORDER:
        m = by.get(label)
        if not m or label == "raw":
            continue
        for sname in ("holdout_date_text", "holdout_date_audio"):
            b = m["sets"][sname]
            A(f"| {label} | {sname.replace('holdout_date_', '')} | {b['win']} | {b['tie']} "
              f"| {b['loss']} | {f(b['damage_rate'])} | {f(b['mean_delta_fair_wer'])} |")
    A("")
    A(f"Raw input on these 580 rows: text fair {raw_t['fair_wer']:.4f} / strict "
      f"{raw_t['strict_lc_wer']:.4f}, audio fair {raw_a['fair_wer']:.4f} / strict "
      f"{raw_a['strict_lc_wer']:.4f}. On the id split the same baselines were 0.1393 (text) and "
      "0.1886 (audio).\n")

    full_t = by["date_lfm12_full_e2"]["sets"]["holdout_date_text"]["corrector"]["fair_wer"]
    full_a = by["date_lfm12_full_e2"]["sets"]["holdout_date_audio"]["corrector"]["fair_wer"]
    n250_t = by["date_lfm12_n250_e12"]["sets"]["holdout_date_text"]["corrector"]["fair_wer"]
    n1000_t = by["date_lfm12_n1000_e2"]["sets"]["holdout_date_text"]["corrector"]["fair_wer"]
    n250_a = by["date_lfm12_n250_e12"]["sets"]["holdout_date_audio"]["corrector"]["fair_wer"]
    cont_t = by["idsplit_lfm2.5-1.2b_CONTAMINATED"]["sets"]["holdout_date_text"]["corrector"]["fair_wer"]
    cont_a = by["idsplit_lfm2.5-1.2b_CONTAMINATED"]["sets"]["holdout_date_audio"]["corrector"]["fair_wer"]

    n500_a = by["date_lfm12_n500_e8"]["sets"]["holdout_date_audio"]["corrector"]["fair_wer"]
    q_t = by["date_q06_full_e2"]["sets"]["holdout_date_text"]["corrector"]["fair_wer"]
    q_a = by["date_q06_full_e2"]["sets"]["holdout_date_audio"]["corrector"]["fair_wer"]
    dmg_t = by["date_lfm12_full_e2"]["sets"]["holdout_date_text"]["damage_rate"]
    dmg_a = by["date_lfm12_full_e2"]["sets"]["holdout_date_audio"]["damage_rate"]
    rel = lambda x, base: 100 * (1 - x / base)  # noqa: E731

    A("## Read\n")
    A(f"The personal refiner still beats raw on a date split, and by more of a margin rather than "
      f"less: LFM2.5-1.2B trained on the {td['n']} pre-cutoff pairs scores {full_t:.4f} fair WER on "
      f"the text path against {raw_t['fair_wer']:.4f} raw "
      f"({rel(full_t, raw_t['fair_wer']):.0f} percent relative, where the published id-split pair was "
      f"{ID['date_lfm12_full_e2']['text']:.4f} against 0.1393 raw, "
      f"{rel(ID['date_lfm12_full_e2']['text'], 0.1393):.0f} percent) and {full_a:.4f} on the Parakeet "
      f"path against {raw_a['fair_wer']:.4f} raw ({rel(full_a, raw_a['fair_wer']):.0f} percent, "
      f"against {ID['date_lfm12_full_e2']['audio']:.4f} / 0.1886 = "
      f"{rel(ID['date_lfm12_full_e2']['audio'], 0.1886):.0f} percent before). "
      f"The absolute headline number barely moves on the text path "
      f"({full_t:.4f} date vs {ID['date_lfm12_full_e2']['text']:.4f} id) and loses "
      f"{full_a - ID['date_lfm12_full_e2']['audio']:.4f} on the Parakeet path "
      f"({full_a:.4f} vs {ID['date_lfm12_full_e2']['audio']:.4f}), and that Parakeet gap is mostly a "
      f"training-data effect rather than a split effect, because the cutoff removes 377 of the 519 "
      f"Parakeet-input rows from training and leaves only {td['parakeet_rows']}. "
      f"So the answer to how much of the id-split number was leakage is: on the text path, "
      f"essentially none — the id split shared days and sessions but never a row, and taking the "
      f"days away costs {abs(full_t - ID['date_lfm12_full_e2']['text']) * 100:.1f} of a point. "
      f"The learning curve survives as well: the earliest 250 pairs still beat raw on both paths "
      f"({n250_t:.4f} vs {raw_t['fair_wer']:.4f} text, {n250_a:.4f} vs {raw_a['fair_wer']:.4f} audio), "
      f"n1000 reaches {n1000_t:.4f} / {by['date_lfm12_n1000_e2']['sets']['holdout_date_audio']['corrector']['fair_wer']:.4f}, "
      f"and the one soft point is n500 on the Parakeet path ({n500_a:.4f}), which nearly ties raw. "
      f"What a day-sharing split really hides is verbatim recall, and its size is now measured: the "
      f"shipped `adapters/lfm2.5-1.2b`, which trained on {leak_in_holdout} of these 580 rows, scores "
      f"{cont_t:.4f} / {cont_a:.4f} here, about half the honest {full_t:.4f} / {full_a:.4f}, so any "
      f"future eval that lets same-row data into training is worth roughly a 2x illusion. "
      f"Product read: the refiner decision stands, quote {full_t:.3f} text / {full_a:.3f} audio fair "
      f"WER and a {dmg_t:.1%} / {dmg_a:.1%} damage rate for unseen days, treat Qwen3-0.6B "
      f"({q_t:.4f} / {q_a:.4f}) and LFM2.5-1.2B as tied as before, and if the acoustic path matters, "
      f"re-transcribe the post-cutoff clips so training has more than {td['parakeet_rows']} "
      f"Parakeet-input rows.\n")

    A("## Files\n")
    A("- split: `/data/phonon_corrector_v0/datesplit/{train_date,holdout_date_text,"
      "holdout_date_audio}.jsonl`, `curve/train_date_first{250,500,1000}.jsonl`, "
      "`datesplit_counts.json`")
    A("- adapters: `/data/phonon_corrector_v0/adapters/date_*`")
    A("- per-row predictions: `/data/phonon_corrector_v0/datesplit/preds/`")
    A("- scripts: `research/corrector_v0/datesplit_build.py`, `datesplit_eval.py`, "
      "`datesplit_report.py`; run driver `/data/phonon_corrector_v0/datesplit_chain.sh`\n")

    payload = {
        "cutoff": c["cutoff"],
        "counts": c,
        "raw_baseline": {"text": raw_t, "audio": raw_a},
        "id_split_reference": ID,
        "models": {
            m["label"]: {
                s: {
                    "n": b["n"],
                    "fair_wer": b["corrector"]["fair_wer"],
                    "strict_lc_wer": b["corrector"]["strict_lc_wer"],
                    "fair_exact": b["corrector"]["fair_exact"],
                    "damage_rate": b["damage_rate"],
                    "win": b["win"], "tie": b["tie"], "loss": b["loss"],
                    "mean_delta_fair_wer": b["mean_delta_fair_wer"],
                }
                for s, b in m["sets"].items()
            }
            for m in ev["models"]
        },
    }
    (OUT / "results_datesplit.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "results_datesplit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("STEP 4 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
