"""Unseen-voice and real-audio analysis for the vocab_v1 refiner.

Writes research/vocab_v1/results_voices.md and results_voices.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
from analyze_vocab1 import (  # noqa: E402
    CLASSES, class_rates, fair, false_inserts, hit_loose, hit_norm, read_jsonl, row_wer,
    strict_lc, wer,
)
from analyze import load_english  # noqa: E402

V1 = Path("/data/phonon_term_eval_v1")
OUT = Path("/home/user/phonon/research/vocab_v1")
SEEN = json.loads((OUT / "results.json").read_text())
SYSTEMS = ["vocab1_real_acoustic", "vocab1_real_acoustic_350m", "vocab_real_acoustic",
           "lfm2.5-1.2b_personal"]
CONDS = [("none", "out_none"), ("oracle", "out_oracle"), ("retrieved", "out_retrieved")]
SEEN_VOICES = "af_heart, am_adam, bf_emma"
UNSEEN_VOICES = "af_bella, am_michael, bm_george"
COMMON = load_english()


def block(idx, refs, raws, terms, hyps, vocabs, raw_row_wer, ret_has, want_classes=True):
    R = [refs[i] for i in idx]
    H = [hyps[i] for i in idx]
    T = [terms[i] for i in idx]
    X = [raws[i] for i in idx]
    V = [vocabs[i] for i in idx]
    hn = [hit_norm(t, h) for t, h in zip(T, H)]
    hl = [hit_loose(t, h) for t, h in zip(T, H)]
    rw = [row_wer(a, b, fair) for a, b in zip(R, H)]
    rr = [raw_row_wer[i] for i in idx]
    fi = [false_inserts(v, h, r, x, tt) for v, h, r, x, tt in zip(V, H, R, X, T)]
    n = len(idx)
    out = {
        "n": n,
        "term_hit_norm": sum(hn) / n,
        "term_hit_loose": sum(hl) / n,
        "fair_wer": wer(R, H, fair),
        "strict_lc_wer": wer(R, H, strict_lc),
        "false_insert_rate": sum(1 for b in fi if b[0]) / n,
        "false_insert_rate_strict": sum(1 for b in fi if b[1]) / n,
        "false_insert_rate_unrelated": sum(1 for b in fi if b[2]) / n,
        "false_insert_spurious": sum(1 for b, h in zip([x[2] for x in fi], hn) if b and h) / n,
        "damage_rate": sum(1 for a, b in zip(rw, rr) if a > b + 1e-9) / n,
    }
    if want_classes:
        out["classes"] = class_rates(list(zip(R, H)), COMMON)
    for name, want in (("term_retrieved", True), ("term_missing", False)):
        sel = [(h, w, r) for h, w, r, i in zip(hn, rw, rr, idx) if ret_has[i] is want]
        out[f"n_{name}"] = len(sel)
        out[f"hit_{name}"] = (sum(1 for s in sel if s[0]) / len(sel)) if sel else None
        out[f"damage_{name}"] = (sum(1 for s in sel if s[1] > s[2] + 1e-9) / len(sel)) if sel else None
    return out


def terms_section(res: dict) -> list[str]:
    rows = {}
    for s in SYSTEMS:
        p = V1 / "refined" / f"{s}__terms.jsonl"
        if p.exists():
            rows[s] = read_jsonl(p)
    if not rows:
        print("no refined term files")
        return []
    base = read_jsonl(V1 / "eval_conditions.jsonl")
    refs = [r["reference"] for r in base]
    raws = [r["hyp"] for r in base]
    terms = [r["term"] for r in base]
    voices = [r["voice"] for r in base]
    ret_has = [r["retrieved_has_term"] for r in base]
    voc_ret = [r["vocab_retrieved"] for r in base]
    voc_ora = [r["vocab_oracle"] for r in base]
    raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, raws)]
    allidx = list(range(len(base)))
    vidx = {}
    for i, v in enumerate(voices):
        vidx.setdefault(v, []).append(i)

    res["voices"] = sorted(vidx)
    res["terms"] = {"raw": {"none": block(allidx, refs, raws, terms, raws, voc_ret, raw_rw, ret_has)}}
    res["terms_by_voice"] = {"raw": {"none": {}}}
    for v, idx in sorted(vidx.items()):
        res["terms_by_voice"]["raw"]["none"] = res["terms_by_voice"]["raw"]["none"] | {
            v: block(idx, refs, raws, terms, raws, voc_ret, raw_rw, ret_has, want_classes=False)}
    for s, rr in rows.items():
        by_id = {r["id"]: r for r in rr}
        res["terms"][s] = {}
        res["terms_by_voice"][s] = {}
        for cond, field in CONDS:
            hyps = [by_id.get(r["id"], {}).get(field, "") for r in base]
            vocabs = voc_ora if cond == "oracle" else voc_ret
            res["terms"][s][cond] = block(allidx, refs, raws, terms, hyps, vocabs, raw_rw, ret_has)
            res["terms_by_voice"][s][cond] = {
                v: block(idx, refs, raws, terms, hyps, vocabs, raw_rw, ret_has, want_classes=False)
                for v, idx in sorted(vidx.items())}
    return []


def real_section(res: dict) -> None:
    D = V1 / "real"
    labels = [("vocab1_real_acoustic", ("none", "retrieved")),
              ("date_lfm12_full_e2", ("none",)),
              ("vocab1_date", ("none", "retrieved"))]
    got = {}
    for lab, conds in labels:
        p = D / f"{lab}.jsonl"
        if p.exists():
            got[lab] = (read_jsonl(p), conds)
    if not got:
        return
    base = next(iter(got.values()))[0]
    refs = [r["reference"] for r in base]
    ins = [r["input"] for r in base]
    vocs = [r["vocab_retrieved"] for r in base]
    raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, ins)]
    fi = [false_inserts(v, h, r, h) for v, h, r in zip(vocs, ins, refs)]
    n = len(base)
    res["real"] = {"raw": {"none": {
        "n": n, "fair_wer": wer(refs, ins, fair), "strict_lc_wer": wer(refs, ins, strict_lc),
        "damage_rate": 0.0, "win": 0, "tie": n, "loss": 0,
        "false_insert_rate": sum(1 for b in fi if b[0]) / n, "false_insert_rate_strict": 0.0,
        "classes": class_rates(list(zip(refs, ins)), COMMON)}}}
    for lab, (rows, conds) in got.items():
        by_id = {r["id"]: r for r in rows}
        res["real"][lab] = {}
        for cond in conds:
            hy = [by_id[r["id"]][f"out_{cond}"] for r in base]
            rw = [row_wer(a, b, fair) for a, b in zip(refs, hy)]
            win = sum(1 for a, b in zip(rw, raw_rw) if a < b - 1e-9)
            loss = sum(1 for a, b in zip(rw, raw_rw) if a > b + 1e-9)
            f2 = [false_inserts(v, h, r, x) for v, h, r, x in zip(vocs, hy, refs, ins)]
            res["real"][lab][cond] = {
                "n": n, "fair_wer": wer(refs, hy, fair), "strict_lc_wer": wer(refs, hy, strict_lc),
                "damage_rate": loss / n, "win": win, "tie": n - win - loss, "loss": loss,
                "false_insert_rate": sum(1 for b in f2 if b[0]) / n,
                "false_insert_rate_strict": sum(1 for b in f2 if b[1]) / n,
                "classes": class_rates(list(zip(refs, hy)), COMMON)}


def fmt(x, p=3):
    return "-" if x is None else f"{x:.{p}f}"


def main() -> int:
    res: dict = {"unseen_voices": UNSEEN_VOICES, "seen_voices": SEEN_VOICES}
    res["retrieval"] = json.loads((V1 / "build_stats.json").read_text())
    terms_section(res)
    real_section(res)
    res["overlap_note"] = {
        "holdout_date_audio_rows": 580,
        "rows_also_in_id_split_train.jsonl": 476,
        "rows_also_in_train_date.jsonl": 0,
    }

    L: list[str] = []
    A = L.append
    A("# Vocabulary refiner v1: unseen voices and real audio")
    A("")
    A(f"The v1 term-clip eval used the same three Kokoro voices as the acoustic training corpus "
      f"({SEEN_VOICES}). This re-runs the identical 900 held-out-term sentences through three "
      f"voices the corpus never saw ({UNSEEN_VOICES}: American female, American male, British "
      f"male), same degradation rule, same Parakeet TDT 0.6b v2 path, and adds the honest "
      f"date-split real-audio holdout.")
    A("")
    A("## Read")
    A("")
    A("<<READ>>")
    A("")

    r = res["retrieval"]
    A("## Step 1: the new clips")
    A("")
    A(f"2,700 clips (900 sentences x 3 unseen voices). Retrieval recall@30 "
      f"**{r['recall@30']:.3f}** ({r['missing_term']} clips miss their term) against 0.964 "
      f"(96 clips) on the seen voices; recall@10 {r['recall@10']:.3f} against 0.914.")
    A("")
    A("| voice | n | raw term hit (norm) | raw term hit (loose string) | raw fair WER | "
      "retrieval recall@30 |")
    A("|---|---:|---:|---:|---:|---:|")
    rbv = res["terms_by_voice"]["raw"]["none"]
    for v in res["voices"]:
        b = rbv[v]
        A(f"| {v} | {b['n']} | {b['term_hit_norm']:.3f} | {b['term_hit_loose']:.3f} | "
          f"{b['fair_wer']:.4f} | {r['recall@30_by_voice'][v]:.3f} |")
    ra = res["terms"]["raw"]["none"]
    A(f"| **unseen (all 3)** | {ra['n']} | **{ra['term_hit_norm']:.3f}** | "
      f"{ra['term_hit_loose']:.3f} | {ra['fair_wer']:.4f} | {r['recall@30']:.3f} |")
    sr = SEEN["terms"]["raw"]["none"]
    A(f"| seen (af_heart, am_adam, bf_emma) | {sr['n']} | {sr['term_hit_norm']:.3f} | "
      f"{sr['term_hit_loose']:.3f} | {sr['fair_wer']:.4f} | 0.964 |")
    A("")

    A("## Step 2: seen voices next to unseen voices")
    A("")
    A("Seen columns are the vocab_v1 results.md numbers; unseen columns are the same metric on "
      "the new clips. `unrel` = a list term written that the reference lacks and that is not a "
      "spelling variant of the clip's own term. `spur` = unrelated insertion on a clip whose true "
      "term was recovered anyway. Damage = clips whose fair WER is worse than raw.")
    A("")
    A("| adapter | cond | hit seen | hit unseen | delta | WER seen | WER unseen | dmg seen | "
      "dmg unseen | unrel seen | unrel unseen | spur seen | spur unseen |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in ["raw"] + SYSTEMS:
        for cond, _ in CONDS:
            u = res["terms"].get(s, {}).get(cond)
            v = SEEN["terms"].get(s, {}).get(cond)
            if not u or not v:
                continue
            A(f"| {s} | {cond} | {v['term_hit_norm']:.3f} | {u['term_hit_norm']:.3f} | "
              f"{u['term_hit_norm']-v['term_hit_norm']:+.3f} | {v['fair_wer']:.4f} | "
              f"{u['fair_wer']:.4f} | {v['damage_rate']*100:.1f}% | {u['damage_rate']*100:.1f}% | "
              f"{v['false_insert_rate_unrelated']*100:.2f}% | "
              f"{u['false_insert_rate_unrelated']*100:.2f}% | "
              f"{v['false_insert_spurious']*100:.2f}% | {u['false_insert_spurious']*100:.2f}% |")
    A("")

    A("### Retrieved-condition split on the unseen voices")
    A("")
    A(f"{ra['n_term_retrieved']} clips retrieve their term, {ra['n_term_missing']} do not.")
    A("")
    A(f"| adapter | cond | retrieved (n={ra['n_term_retrieved']}) | missing "
      f"(n={ra['n_term_missing']}) | damage retrieved | damage missing |")
    A("|---|---|---:|---:|---:|---:|")
    A(f"| raw | - | {fmt(ra['hit_term_retrieved'])} | {fmt(ra['hit_term_missing'])} | - | - |")
    for s in SYSTEMS:
        for cond in ("none", "retrieved", "oracle"):
            b = res["terms"].get(s, {}).get(cond)
            if not b:
                continue
            A(f"| {s} | {cond} | {fmt(b['hit_term_retrieved'])} "
              f"({b['hit_term_retrieved']-ra['hit_term_retrieved']:+.3f}) | "
              f"{fmt(b['hit_term_missing'])} "
              f"({b['hit_term_missing']-ra['hit_term_missing']:+.3f}) | "
              f"{b['damage_term_retrieved']*100:.1f}% | {b['damage_term_missing']*100:.1f}% |")
    A("")

    A("### Per unseen voice (term hit norm / fair WER / damage)")
    A("")
    A("| adapter | cond | " + " | ".join(f"{v}" for v in res["voices"]) + " |")
    A("|---|---|" + "---:|" * len(res["voices"]))
    for s in ["raw"] + SYSTEMS:
        for cond, _ in CONDS:
            d = res["terms_by_voice"].get(s, {}).get(cond)
            if not d:
                continue
            cells = [f"{d[v]['term_hit_norm']:.3f} / {d[v]['fair_wer']:.4f} / "
                     f"{d[v]['damage_rate']*100:.1f}%" for v in res["voices"]]
            A(f"| {s} | {cond} | " + " | ".join(cells) + " |")
    A("")

    A("### Error classes on the unseen-voice term clips (errors per 1000 reference words)")
    A("")
    A("| adapter | cond | " + " | ".join(CLASSES) + " | total |")
    A("|---|---|" + "---:|" * (len(CLASSES) + 1))
    for s in ["raw"] + SYSTEMS:
        for cond, _ in CONDS:
            b = res["terms"].get(s, {}).get(cond)
            if not b:
                continue
            cl = b["classes"]
            A(f"| {s} | {cond} | " + " | ".join(f"{cl[k]:.1f}" for k in CLASSES)
              + f" | {sum(cl.values()):.1f} |")
    A("")

    if "real" in res:
        o = res["overlap_note"]
        A("## Step 3: the user's real audio (date-split holdout, 580 rows)")
        A("")
        A(f"`holdout_date_audio` is Parakeet on the user's own recordings, reference = the text he "
          f"accepted, split by date so every holdout row is later than every training row. "
          f"**{o['rows_also_in_id_split_train.jsonl']} of {o['holdout_date_audio_rows']} rows are "
          f"in `train.jsonl`**, the id-split file `vocab1_real_acoustic` was trained on, so that "
          f"row is contaminated; `vocab1_date` is the same v1 recipe rebuilt on `train_date.jsonl` "
          f"(0 of 580 shared) and is the honest number. `date_lfm12_full_e2` is the plain date-split "
          f"corrector with no vocabulary machinery.")
        A("")
        A("| system | cond | contaminated | fair WER | strict lc WER | damage | win | tie | loss | "
          "ENTITY/1k | false insert | not in raw |")
        A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        contam = {"raw": "-", "vocab1_real_acoustic": "yes (476/580)",
                  "date_lfm12_full_e2": "no", "vocab1_date": "no"}
        for lab in ["raw", "vocab1_real_acoustic", "date_lfm12_full_e2", "vocab1_date"]:
            d = res["real"].get(lab)
            if not d:
                continue
            for cond, b in d.items():
                A(f"| {lab} | {cond} | {contam[lab]} | {b['fair_wer']:.4f} | "
                  f"{b['strict_lc_wer']:.4f} | {b['damage_rate']*100:.1f}% | {b['win']} | "
                  f"{b['tie']} | {b['loss']} | {b['classes']['ENTITY']:.1f} | "
                  f"{b['false_insert_rate']*100:.2f}% | "
                  f"{b['false_insert_rate_strict']*100:.2f}% |")
        A("")
        A("### Error classes on the real-audio holdout (errors per 1000 reference words)")
        A("")
        A("| system | cond | " + " | ".join(CLASSES) + " | total |")
        A("|---|---|" + "---:|" * (len(CLASSES) + 1))
        for lab in ["raw", "vocab1_real_acoustic", "date_lfm12_full_e2", "vocab1_date"]:
            d = res["real"].get(lab)
            if not d:
                continue
            for cond, b in d.items():
                cl = b["classes"]
                A(f"| {lab} | {cond} | " + " | ".join(f"{cl[k]:.1f}" for k in CLASSES)
                  + f" | {sum(cl.values()):.1f} |")
        A("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results_voices.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "results_voices.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT/'results_voices.md'} and results_voices.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
