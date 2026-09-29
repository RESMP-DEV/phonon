"""bigrun_v0 eval step 3: tables for every adapter that has landed, plus the extended scaling fit.

Metric definitions are imported unchanged from research/vocab_v1/analyze_vocab1.py.
Writes research/bigrun_v0/results.md and results.json; safe to re-run after each adapter.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
from analyze_vocab1 import (  # noqa: E402
    CLASSES, class_rates, fair, false_inserts, hit_loose, hit_norm, read_jsonl, row_wer,
    strict_lc, wer,
)
from analyze import load_english  # noqa: E402

D = Path("/data/phonon_bigrun_v0")
REF = D / "refined"
OUT = Path("/home/user/phonon/research/bigrun_v0")
SCALING = Path("/home/user/phonon/research/scaling_v0/results.json")
COND_FILE = {
    "seen": Path("/data/phonon_vocab_v1/eval_conditions.jsonl"),
    "unseen": Path("/data/phonon_term_eval_v1/eval_conditions.jsonl"),
    "new": D / "eval_conditions_new.jsonl",
}
REAL_FILE = Path("/data/phonon_term_eval_v1/real/eval_holdout_date_audio.jsonl")
TERM_SETS = ["seen", "unseen", "new"]
CONDS = ["none", "oracle", "retrieved"]
REAL_CONDS = ["none", "retrieved"]
# (label, distinct acoustic terms, acoustic rows, note)
CORPUS = {
    "scale_n36000": (2107, 36000, "scaling_v0 baseline, 3 voices"),
    "bigrun_mid_r16": (14537, 164325, "train_mid, r=16, 2 epochs"),
    "bigrun_mid_r64": (14537, 164325, "train_mid, r=64, 2 epochs"),
    "bigrun_big_r16": (14537, 328729, "train_big, r=16"),
    "bigrun_big_r16_personal": (14537, 328729, "stage 2: train_date only on top of big_r16"),
    "bigrun_big_350m": (14537, 328729, "LFM2.5-350M on train_big"),
    "bigrun_mid_generic": (14537, 164325, "train_mid acoustic rows only, no user data"),
    "bigrun_mid_generic_personal": (14537, 164325,
                                    "stage 2: bigrun_mid_generic + the user's 3,827 real rows"),
}
ORDER = ["raw", "scale_n36000", "bigrun_mid_r16", "bigrun_mid_r64", "bigrun_big_r16",
         "bigrun_big_r16_personal", "bigrun_big_350m", "bigrun_mid_generic",
         "bigrun_mid_generic_personal"]
COMMON = load_english()


def term_block(refs, raws, terms, hyps, vocabs, raw_rw, ret_has, classes=False):
    n = len(refs)
    hn = [hit_norm(t, h) for t, h in zip(terms, hyps)]
    hl = [hit_loose(t, h) for t, h in zip(terms, hyps)]
    rw = [row_wer(a, b, fair) for a, b in zip(refs, hyps)]
    fi = [false_inserts(v, h, r, x, tt)
          for v, h, r, x, tt in zip(vocabs, hyps, refs, raws, terms)]
    out = {
        "n": n,
        "term_hit_norm": sum(hn) / n,
        "term_hit_loose": sum(hl) / n,
        "fair_wer": wer(refs, hyps, fair),
        "strict_lc_wer": wer(refs, hyps, strict_lc),
        "damage_rate": sum(1 for a, b in zip(rw, raw_rw) if a > b + 1e-9) / n,
        "false_insert_rate": sum(1 for b in fi if b[0]) / n,
        "false_insert_rate_strict": sum(1 for b in fi if b[1]) / n,
        "false_insert_rate_unrelated": sum(1 for b in fi if b[2]) / n,
        "false_insert_spurious": sum(1 for b, h in zip([x[2] for x in fi], hn) if b and h) / n,
    }
    for name, want in (("term_retrieved", True), ("term_missing", False)):
        sel = [(a, b, c) for a, b, c, w in zip(hn, rw, raw_rw, ret_has) if w is want]
        out[f"n_{name}"] = len(sel)
        out[f"hit_{name}"] = (sum(1 for s in sel if s[0]) / len(sel)) if sel else None
        out[f"damage_{name}"] = (sum(1 for s in sel if s[1] > s[2] + 1e-9) / len(sel)) if sel \
            else None
    if classes:
        out["classes"] = class_rates(list(zip(refs, hyps)), COMMON)
    return out


def real_block(refs, ins, hyps, vocabs, raw_rw):
    n = len(refs)
    rw = [row_wer(a, b, fair) for a, b in zip(refs, hyps)]
    win = sum(1 for a, b in zip(rw, raw_rw) if a < b - 1e-9)
    loss = sum(1 for a, b in zip(rw, raw_rw) if a > b + 1e-9)
    fi = [false_inserts(v, h, r, x) for v, h, r, x in zip(vocabs, hyps, refs, ins)]
    # length pathologies and the WER with the single largest error contributor removed
    nr = [len(fair(a).split()) for a in refs]
    nh = [len(fair(b).split()) for b in hyps]
    worst = max(range(n), key=lambda i: rw[i] * nr[i])
    keep = [i for i in range(n) if i != worst]
    return {
        "fair_wer_drop_worst_row": wer([refs[i] for i in keep], [hyps[i] for i in keep], fair),
        "runaway_rows": sum(1 for a, b in zip(nr, nh) if b > 2 * max(a, 1)),
        "truncated_rows": sum(1 for a, b in zip(nr, nh) if b < 0.5 * a),
        "n": n, "fair_wer": wer(refs, hyps, fair), "strict_lc_wer": wer(refs, hyps, strict_lc),
        "damage_rate": loss / n, "win": win, "tie": n - win - loss, "loss": loss,
        "false_insert_rate": sum(1 for b in fi if b[0]) / n,
        "false_insert_rate_strict": sum(1 for b in fi if b[1]) / n,
        "false_insert_rate_unrelated": sum(1 for b in fi if b[2]) / n,
        "classes": class_rates(list(zip(refs, hyps)), COMMON),
    }


def fit(points):
    """least squares hit = a + b*log10(terms); returns a, b, r2, rmse."""
    xs = [math.log10(t) for _, t, _ in points]
    ys = [h for _, _, h in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    pred = [a + b * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1 - ss_res / ss_tot, math.sqrt(ss_res / n)


def main() -> int:
    res: dict = {"terms": {}, "real": {}, "retrieval": {}, "corpus": CORPUS}
    base = {k: read_jsonl(p) for k, p in COND_FILE.items() if p.exists()}
    for k in list(base):
        p = D / "eval_new_build_stats.json" if k == "new" else None
        if k == "seen":
            p = Path("/data/phonon_vocab_v1/build_stats.json")
        elif k == "unseen":
            p = Path("/data/phonon_term_eval_v1/build_stats.json")
        try:
            s = json.loads(p.read_text())
            res["retrieval"][k] = {kk: s.get(kk) for kk in
                                   ("rows", "recall@10", "recall@30", "missing_term", "pool")}
            if res["retrieval"][k]["recall@30"] is None and "retrieval" in s:
                res["retrieval"][k] = {kk: s["retrieval"].get(kk) for kk in
                                       ("recall@10", "recall@30", "pool_eval", "pool")}
        except Exception as exc:
            res["retrieval"][k] = {"error": str(exc)}

    # ---- term clip sets -------------------------------------------------
    prep = {}
    for k, rows in base.items():
        refs = [r["reference"] for r in rows]
        raws = [r["hyp"] for r in rows]
        terms = [r["term"] for r in rows]
        ret_has = [r["retrieved_has_term"] for r in rows]
        vr = [r["vocab_retrieved"] for r in rows]
        vo = [r["vocab_oracle"] for r in rows]
        raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, raws)]
        prep[k] = (rows, refs, raws, terms, ret_has, vr, vo, raw_rw)
        res["terms"].setdefault("raw", {})[k] = {
            "none": term_block(refs, raws, terms, raws, vr, raw_rw, ret_has, classes=True)}

    labels = sorted({p.name.split("__")[0] for p in REF.glob("*__*.jsonl")})
    for lab in labels:
        for k, (rows, refs, raws, terms, ret_has, vr, vo, raw_rw) in prep.items():
            p = REF / f"{lab}__{k}.jsonl"
            if not p.exists():
                continue
            by_id = {r["id"]: r for r in read_jsonl(p)}
            res["terms"].setdefault(lab, {})[k] = {}
            for cond in CONDS:
                hyps = [by_id.get(r["id"], {}).get(f"out_{cond}", "") for r in rows]
                vocabs = vo if cond == "oracle" else vr
                res["terms"][lab][k][cond] = term_block(
                    refs, raws, terms, hyps, vocabs, raw_rw, ret_has, classes=(cond == "retrieved"))

    # ---- real audio -----------------------------------------------------
    if REAL_FILE.exists():
        rrows = read_jsonl(REAL_FILE)
        refs = [r["reference"] for r in rrows]
        ins = [r["input"] for r in rrows]
        vocs = [r["vocab_retrieved"] for r in rrows]
        raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, ins)]
        res["real"]["raw"] = {"none": real_block(refs, ins, ins, vocs, raw_rw)}
        res["real"]["raw"]["none"].update({"win": 0, "tie": len(rrows), "loss": 0,
                                           "damage_rate": 0.0})
        for lab in labels:
            p = REF / f"{lab}__real.jsonl"
            if not p.exists():
                continue
            by_id = {r["id"]: r for r in read_jsonl(p)}
            res["real"][lab] = {}
            for cond in REAL_CONDS:
                hy = [by_id.get(r["id"], {}).get(f"out_{cond}", "") for r in rrows]
                res["real"][lab][cond] = real_block(refs, ins, hy, vocs, raw_rw)

    # ---- extended scaling fit -------------------------------------------
    sc = json.loads(SCALING.read_text())
    pts = [(n, t, h) for n, t, h in sc["fit_terms"]["points"]]
    new_pts = []
    for lab in ORDER:
        if lab == "raw" or "personal" in lab or "generic" in lab or lab not in res["terms"]:
            continue
        if lab == "scale_n36000":
            continue
        b = res["terms"][lab].get("seen", {}).get("retrieved")
        if b:
            new_pts.append((lab, CORPUS[lab][0], b["term_hit_norm"]))
    # scale_n36000 re-measured here if present, else the scaling_v0 number already in pts
    fits = {"scaling_v0_only": dict(zip(("intercept", "slope_per_decade", "r2", "rmse"),
                                        fit(pts)))}
    ext = pts + [p for p in new_pts if p[0].startswith("bigrun") and "350m" not in p[0]
                 and "generic" not in p[0]]
    if len(ext) > len(pts):
        a, bb, r2, rmse = fit(ext)
        fits["extended"] = {"intercept": a, "slope_per_decade": bb, "r2": r2, "rmse": rmse,
                            "points": ext}
        big = [p for p in ext if p[1] > 3000]
        if big:
            lo = max((p for p in pts), key=lambda p: p[1])
            hi = max(big, key=lambda p: p[2])
            fits["slope_2107_to_14537"] = (hi[2] - lo[2]) / (math.log10(hi[1]) - math.log10(lo[1]))
            fits["projected_at_14537_from_scaling_v0"] = (
                fits["scaling_v0_only"]["intercept"]
                + fits["scaling_v0_only"]["slope_per_decade"] * math.log10(14537))
    fits["scaling_v0_points"] = pts
    fits["bigrun_points"] = new_pts
    res["fit_terms"] = fits

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"labels": labels,
                      "fit": {k: v for k, v in fits.items()
                              if not isinstance(v, (list, dict))}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
