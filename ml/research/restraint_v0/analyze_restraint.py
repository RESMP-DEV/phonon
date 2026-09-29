"""restraint_v0: variant table against bigrun_mid_r16 and raw Parakeet.

Metrics come from research/bigrun_v0/analyze_bigrun.py (term_block, real_block), which imports
them from research/vocab_v1/analyze_vocab1.py, so nothing here redefines a number. Per-row
generations: /data/phonon_restraint_v0/refined for the variants, /data/phonon_bigrun_v0/refined
for bigrun_mid_r16. The guarded real column applies refine_bigrun.guard_post to those exact
generations (the `post` mode of results_guard.md - the rule with zero decode noise).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
sys.path.insert(0, "/home/user/phonon/research/bigrun_v0")
from analyze_vocab1 import fair, read_jsonl, row_wer  # noqa: E402
from analyze_bigrun import real_block, term_block  # noqa: E402
from refine_bigrun import guard_post  # noqa: E402

D = Path("/data/phonon_restraint_v0")
BIG = Path("/data/phonon_bigrun_v0")
REF = {"restraint": D / "refined", "bigrun": BIG / "refined"}
OUT = Path("/home/user/phonon/research/restraint_v0")
ADAPTERS = Path("/data/phonon_corrector_v0/adapters")

COND_FILE = {
    "seen": Path("/data/phonon_vocab_v1/eval_conditions.jsonl"),
    "unseen": Path("/data/phonon_term_eval_v1/eval_conditions.jsonl"),
    "new": BIG / "eval_conditions_new.jsonl",
}
REAL_FILE = Path("/data/phonon_term_eval_v1/real/eval_holdout_date_audio.jsonl")
CONDS = ["none", "retrieved"]

# label -> (refined dir key, one-line description)
SYSTEMS = [
    ("bigrun_mid_r16", "bigrun", "overnight reference: train_mid, r16, legacy trainer settings"),
    ("bigrun_mid_generic_personal", "bigrun",
     "overnight reference: acoustic-only stage 1 + real rows, NO acoustic replay"),
    ("restr_mid_control", "restraint", "e. train_mid, today's settings (dropout 0, grouped, fused CE)"),
    ("restr_restraint50", "restraint", "a. restraint 50% of acoustic rows, zero-edit cap 45%"),
    ("restr_real12", "restraint", "b. real rows x12 instead of x4"),
    ("restr_keepweight", "restraint", "c. train_mid + per-token loss weight 2.0 on copied tokens"),
    ("restr_generic_replay", "restraint",
     "d. acoustic-only stage 1, then real rows x4 + 25% acoustic replay, 1 epoch lr 1e-4"),
]

# criterion, from research/bigrun_v0/results.md and the overnight read
RAW_ENTITY = 22.6
MAX_DAMAGE = 0.057
TERM_TOL = 0.02
MID_HIT = {"seen": 0.888, "unseen": 0.875, "new": 0.811}


def f4(x):
    return "-" if x is None else f"{x:.4f}"


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def f1(x):
    return "-" if x is None else f"{x:.1f}"


def main() -> int:  # noqa: C901
    res: dict = {"terms": {}, "real": {}, "real_guarded": {}, "train": {}, "criteria": {}}
    try:
        res["build"] = json.loads((D / "build_stats.json").read_text())
    except Exception as exc:
        res["build"] = {"error": str(exc)}

    # ---- term clip sets --------------------------------------------------
    prep = {}
    for k, p in COND_FILE.items():
        rows = read_jsonl(p)
        refs = [r["reference"] for r in rows]
        raws = [r["hyp"] for r in rows]
        terms = [r["term"] for r in rows]
        ret_has = [r["retrieved_has_term"] for r in rows]
        vr = [r["vocab_retrieved"] for r in rows]
        raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, raws)]
        prep[k] = (rows, refs, raws, terms, ret_has, vr, raw_rw)
        res["terms"].setdefault("raw", {})[k] = {
            "none": term_block(refs, raws, terms, raws, vr, raw_rw, ret_has)}

    rrows = read_jsonl(REAL_FILE)
    r_refs = [r["reference"] for r in rrows]
    r_ins = [r["input"] for r in rrows]
    r_voc = [r["vocab_retrieved"] for r in rrows]
    r_raw_rw = [row_wer(a, b, fair) for a, b in zip(r_refs, r_ins)]
    res["real"]["raw"] = {"none": real_block(r_refs, r_ins, r_ins, r_voc, r_raw_rw)}
    res["real"]["raw"]["none"].update({"win": 0, "tie": len(rrows), "loss": 0, "damage_rate": 0.0})
    res["real_guarded"]["raw"] = res["real"]["raw"]

    have = []
    for lab, dirkey, _desc in SYSTEMS:
        d = REF[dirkey]
        got = False
        for k, (rows, refs, raws, terms, ret_has, vr, raw_rw) in prep.items():
            p = d / f"{lab}__{k}.jsonl"
            if not p.exists():
                continue
            got = True
            by_id = {r["id"]: r for r in read_jsonl(p)}
            res["terms"].setdefault(lab, {})[k] = {}
            for cond in CONDS:
                hyps = [by_id.get(r["id"], {}).get(f"out_{cond}", "") for r in rows]
                res["terms"][lab][k][cond] = term_block(
                    refs, raws, terms, hyps, vr, raw_rw, ret_has, classes=(cond == "retrieved"))
        p = d / f"{lab}__real.jsonl"
        if p.exists():
            got = True
            by_id = {r["id"]: r for r in read_jsonl(p)}
            res["real"][lab], res["real_guarded"][lab] = {}, {}
            for cond in CONDS:
                hy = [by_id.get(r["id"], {}).get(f"out_{cond}", "") for r in rrows]
                res["real"][lab][cond] = real_block(r_refs, r_ins, hy, r_voc, r_raw_rw)
                g = [guard_post(i, h) for i, h in zip(r_ins, hy)]
                res["real_guarded"][lab][cond] = real_block(
                    r_refs, r_ins, [x for x, _ in g], r_voc, r_raw_rw)
                res["real_guarded"][lab][cond]["fired"] = sum(1 for _, f in g if f)
        meta = ADAPTERS / lab / "train_meta.json"
        if meta.exists():
            res["train"][lab] = json.loads(meta.read_text())
        if got:
            have.append(lab)

    # ---- criterion -------------------------------------------------------
    for lab in have:
        rb = res["real_guarded"].get(lab, {}).get("retrieved")
        if not rb:
            continue
        ent = rb["classes"]["ENTITY"]
        hits = {k: res["terms"].get(lab, {}).get(k, {}).get("retrieved", {}).get("term_hit_norm")
                for k in ("seen", "unseen", "new")}
        term_ok = all(h is not None and h >= MID_HIT[k] - TERM_TOL for k, h in hits.items())
        res["criteria"][lab] = {
            "entity_per_1000": ent, "entity_ok": ent <= RAW_ENTITY,
            "damage": rb["damage_rate"], "damage_ok": rb["damage_rate"] <= MAX_DAMAGE,
            "term_hit": hits, "term_ok": term_ok,
            "pass": bool(ent <= RAW_ENTITY and rb["damage_rate"] <= MAX_DAMAGE and term_ok),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")

    # ---- markdown --------------------------------------------------------
    L = []
    L.append("# restraint_v0: buying real-dictation precision back at fixed terms\n")
    L.append("Five LFM2.5-1.2B variants on the `train_mid` corpus, evaluated with "
             "`research/bigrun_v0/refine_bigrun.py` metrics (oracle condition dropped). Real "
             "audio is the 580-row date-split holdout; the guarded columns apply the "
             "`results_guard.md` word post-check to the same generations (`post` mode).\n")
    L.append("Target: real-audio ENTITY <= 22.6 per 1000 words (the raw Parakeet floor), damage "
             "<= 0.057, term hit within 0.02 of `bigrun_mid_r16` "
             "(0.888 seen / 0.875 unseen / 0.811 heldout_new).\n")

    L.append("\n## Variants\n")
    L.append("| variant | rows | acoustic | real | restraint rows | zero-edit rows | note |")
    L.append("|---|---|---|---|---|---|---|")
    b = res.get("build", {})
    for key, lab in (("a", "restr_restraint50"), ("b", "restr_real12"), ("d2", "restr_generic_replay")):
        s = b.get(key)
        if not s:
            continue
        br = s.get("branches", {})
        L.append(f"| {lab} | {s['rows']} | {s['rows']-s['real_rows']} | {s['real_rows']} | "
                 f"{br.get('restraint', 0)} | {s['zero_edit_rows']} ({s['zero_edit_frac']:.3f}) | "
                 f"{key} |")
    mr = b.get("mid_reference")
    if mr:
        L.append(f"| train_mid (control / keepweight) | 179633 | 164325 | 15308 | "
                 f"{mr['branches'].get('restraint', 0)} | {mr['zero_edit_rows']} "
                 f"({mr['zero_edit_frac']:.3f}) | mid recipe |")

    L.append("\n## Real audio, 580 rows, retrieved list (shipping condition)\n")
    L.append("| system | fair WER | strict WER | damage | win | tie | loss | ENTITY/1k | "
             "false ins | guarded fair WER | guarded damage | guarded ENTITY/1k | fired |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in ["raw"] + have:
        rb = res["real"].get(lab, {}).get("retrieved") or res["real"].get(lab, {}).get("none")
        gb = res["real_guarded"].get(lab, {}).get("retrieved") or \
            res["real_guarded"].get(lab, {}).get("none")
        if not rb:
            continue
        L.append(f"| {lab} | {f4(rb['fair_wer'])} | {f4(rb['strict_lc_wer'])} | "
                 f"{f3(rb['damage_rate'])} | {rb['win']} | {rb['tie']} | {rb['loss']} | "
                 f"{f1(rb['classes']['ENTITY'])} | {f3(rb['false_insert_rate_unrelated'])} | "
                 f"{f4(gb['fair_wer'])} | {f3(gb['damage_rate'])} | "
                 f"{f1(gb['classes']['ENTITY'])} | {gb.get('fired', 0)} |")

    L.append("\n## Real audio, no vocabulary line\n")
    L.append("| system | fair WER | damage | win | tie | loss | ENTITY/1k |")
    L.append("|---|---|---|---|---|---|---|")
    for lab in ["raw"] + have:
        rb = res["real"].get(lab, {}).get("none")
        if not rb:
            continue
        L.append(f"| {lab} | {f4(rb['fair_wer'])} | {f3(rb['damage_rate'])} | {rb['win']} | "
                 f"{rb['tie']} | {rb['loss']} | {f1(rb['classes']['ENTITY'])} |")

    L.append("\n## Term hit, 2,700 clips per set, v1 lists\n")
    L.append("| system | seen none | seen retr | unseen none | unseen retr | new none | new retr | "
             "unrelated ins (new, retr) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for lab in ["raw"] + have:
        t = res["terms"].get(lab, {})
        cells = []
        for k in ("seen", "unseen", "new"):
            for cond in CONDS:
                blk = t.get(k, {}).get(cond) or (t.get(k, {}).get("none") if lab == "raw" else None)
                cells.append(f3(blk["term_hit_norm"]) if blk else "-")
        ui = t.get("new", {}).get("retrieved") or t.get("new", {}).get("none")
        L.append(f"| {lab} | " + " | ".join(cells) +
                 f" | {f3(ui['false_insert_rate_unrelated']) if ui else '-'} |")

    L.append("\n## Criterion\n")
    L.append("| system | ENTITY/1k <= 22.6 | damage <= 0.057 | term hit within 0.02 | pass |")
    L.append("|---|---|---|---|---|")
    for lab in have:
        c = res["criteria"].get(lab)
        if not c:
            continue
        L.append(f"| {lab} | {f1(c['entity_per_1000'])} {'OK' if c['entity_ok'] else 'FAIL'} | "
                 f"{f3(c['damage'])} {'OK' if c['damage_ok'] else 'FAIL'} | "
                 f"{'OK' if c['term_ok'] else 'FAIL'} | "
                 f"{'**PASS**' if c['pass'] else 'fail'} |")

    L.append("\n## Training\n")
    L.append("| adapter | rows | epochs | lr | seconds | sec/step | fused CE | keep weight |")
    L.append("|---|---|---|---|---|---|---|---|")
    for lab in have:
        m = res["train"].get(lab)
        if not m:
            continue
        L.append(f"| {lab} | {m.get('train_rows')} | {m.get('epochs')} | {m.get('lr')} | "
                 f"{m.get('seconds', 0):.0f} | {m.get('sec_per_step', 0):.3f} | "
                 f"{m.get('fused_ce', '-')} | {m.get('keep_weight', 0)} |")

    read_path = OUT / "read.md"
    if read_path.exists():
        L.append("\n" + read_path.read_text().rstrip() + "\n")
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
