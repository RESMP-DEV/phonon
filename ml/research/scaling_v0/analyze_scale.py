"""scaling_v0 step 5: size table, axis table, log-linear fit -> results.md + results.json."""
from __future__ import annotations
import json, math, sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
from analyze_vocab1 import (  # noqa: E402
    fair, false_inserts, hit_norm, read_jsonl, row_wer, strict_lc, wer,
)

D = Path("/data/phonon_scaling_v0")
REF = D / "refined"
OUT = Path("/home/user/phonon/research/scaling_v0")
SIZES = [0, 500, 1000, 2000, 4000, 9000, 18000, 36000]
AXES = ["terms", "sentences", "voices"]
CLIPSETS = {"seen": Path("/data/phonon_vocab_v1/eval_conditions.jsonl"),
            "unseen": Path("/data/phonon_term_eval_v1/eval_conditions.jsonl")}
REALSET = Path("/data/phonon_term_eval_v1/real/eval_holdout_date_audio.jsonl")


def clip_block(base, hyps, vocabs, raw_rw):
    refs = [r["reference"] for r in base]
    terms = [r["term"] for r in base]
    raws = [r["hyp"] for r in base]
    n = len(base)
    hn = [hit_norm(t, h) for t, h in zip(terms, hyps)]
    rw = [row_wer(a, b, fair) for a, b in zip(refs, hyps)]
    fi = [false_inserts(v, h, r, x, tt) for v, h, r, x, tt in zip(vocabs, hyps, refs, raws, terms)]
    ret = [r["retrieved_has_term"] for r in base]
    sel = [(h, w, q) for h, w, q, k in zip(hn, rw, raw_rw, ret) if not k]
    return {
        "n": n,
        "term_hit_norm": sum(hn) / n,
        "fair_wer": wer(refs, hyps, fair),
        "strict_lc_wer": wer(refs, hyps, strict_lc),
        "damage_rate": sum(1 for a, b in zip(rw, raw_rw) if a > b + 1e-9) / n,
        "unrelated_insert_rate": sum(1 for b in fi if b[2]) / n,
        "spurious_rate": sum(1 for b, h in zip([x[2] for x in fi], hn) if b and h) / n,
        "n_term_missing": len(sel),
        "hit_term_missing": (sum(1 for s in sel if s[0]) / len(sel)) if sel else None,
        "damage_term_missing": (sum(1 for s in sel if s[1] > s[2] + 1e-9) / len(sel)) if sel else None,
    }


def real_block(base, hyps, raw_rw):
    refs = [r["reference"] for r in base]
    ins = [r["input"] for r in base]
    vocs = [r["vocab_retrieved"] for r in base]
    n = len(base)
    rw = [row_wer(a, b, fair) for a, b in zip(refs, hyps)]
    win = sum(1 for a, b in zip(rw, raw_rw) if a < b - 1e-9)
    loss = sum(1 for a, b in zip(rw, raw_rw) if a > b + 1e-9)
    f2 = [false_inserts(v, h, r, x) for v, h, r, x in zip(vocs, hyps, refs, ins)]
    return {"n": n, "fair_wer": wer(refs, hyps, fair), "strict_lc_wer": wer(refs, hyps, strict_lc),
            "damage_rate": loss / n, "win": win, "tie": n - win - loss, "loss": loss,
            "false_insert_not_in_raw": sum(1 for b in f2 if b[1]) / n}


def fit(xs, ys):
    """least squares y = a + b*log10(x)"""
    lx = [math.log10(x) for x in xs]
    n = len(xs)
    mx, my = sum(lx) / n, sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ys))
    b = sxy / sxx
    a = my - b * mx
    pred = [a + b * t for t in lx]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {"intercept": a, "slope_per_decade": b,
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else None,
            "rmse": math.sqrt(ss_res / n)}


def main() -> int:
    res: dict = {"sizes": SIZES, "axes": AXES, "systems": {}}
    bases = {k: read_jsonl(p) for k, p in CLIPSETS.items()}
    voc = {k: {r["id"]: r["vocab_retrieved"] for r in v} for k, v in bases.items()}
    raw_rw = {k: [row_wer(r["reference"], r["hyp"], fair) for r in v] for k, v in bases.items()}
    realbase = read_jsonl(REALSET)
    real_raw_rw = [row_wer(r["reference"], r["input"], fair) for r in realbase]

    res["raw"] = {}
    for k, v in bases.items():
        res["raw"][k] = {
            "retrieved": clip_block(v, [r["hyp"] for r in v],
                                    [voc[k][r["id"]] for r in v], raw_rw[k]),
            "none": clip_block(v, [r["hyp"] for r in v],
                               [voc[k][r["id"]] for r in v], raw_rw[k]),
        }
    res["raw"]["real"] = real_block(realbase, [r["input"] for r in realbase], real_raw_rw)

    names = [f"scale_n{n}" for n in SIZES] + [f"scale_axis_{a}" for a in AXES]
    for name in names:
        ok = all((REF / f"{name}__{k}.jsonl").exists() for k in ("seen", "unseen", "real"))
        if not ok:
            print(f"missing refined files for {name}", flush=True)
            continue
        entry = {}
        for k, v in bases.items():
            rows = {r["id"]: r for r in read_jsonl(REF / f"{name}__{k}.jsonl")}
            for cond, field in (("none", "out_none"), ("retrieved", "out_retrieved")):
                hyps = [rows.get(r["id"], {}).get(field, "") for r in v]
                vv = [voc[k][r["id"]] if cond == "retrieved" else voc[k][r["id"]] for r in v]
                entry[f"{k}_{cond}"] = clip_block(v, hyps, vv, raw_rw[k])
        rrows = {r["id"]: r for r in read_jsonl(REF / f"{name}__real.jsonl")}
        entry["real_retrieved"] = real_block(
            realbase, [rrows[r["id"]]["out_retrieved"] for r in realbase], real_raw_rw)
        p = D / "mixes" / f"stats_{name}.json"
        if p.exists():
            entry["mix"] = json.loads(p.read_text())
        res["systems"][name] = entry

    got = [n for n in SIZES if f"scale_n{n}" in res["systems"] and n > 0]
    ys = [res["systems"][f"scale_n{n}"]["seen_retrieved"]["term_hit_norm"] for n in got]
    res["fit"] = {}
    if len(got) >= 3:
        res["fit"]["all"] = fit(got, ys) | {"points": list(zip(got, ys))}
        lo = [(n, y) for n, y in zip(got, ys) if n <= 9000]
        hi = [(n, y) for n, y in zip(got, ys) if n >= 9000]
        if len(lo) >= 2:
            res["fit"]["low"] = fit([a for a, _ in lo], [b for _, b in lo])
        if len(hi) >= 2:
            res["fit"]["high"] = fit([a for a, _ in hi], [b for _, b in hi])
    # the same curve against distinct training terms, size series and axis series together
    pts = []
    for name, e in res["systems"].items():
        m = e.get("mix", {}).get("selection", {})
        if m.get("terms"):
            pts.append((name, m["terms"], e["seen_retrieved"]["term_hit_norm"]))
    if len(pts) >= 3:
        res["fit_terms"] = fit([b for _, b, _ in pts], [c for _, _, c in pts]) | {
            "points": [(a, b, c) for a, b, c in pts]}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")

    L: list[str] = []
    A = L.append
    A("## Size table (size series, terms-first nested subsets)")
    A("")
    A("| N acoustic | terms | sent/term | voices | seen hit ret | seen hit none | unseen hit ret | "
      "unseen hit none | seen dmg ret | seen unrel ret | real WER ret | real dmg |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    rs, ru, rr = res["raw"]["seen"]["retrieved"], res["raw"]["unseen"]["retrieved"], res["raw"]["real"]
    A(f"| raw ASR | - | - | - | {rs['term_hit_norm']:.3f} | {rs['term_hit_norm']:.3f} | "
      f"{ru['term_hit_norm']:.3f} | {ru['term_hit_norm']:.3f} | - | - | {rr['fair_wer']:.4f} | - |")
    for n in SIZES:
        e = res["systems"].get(f"scale_n{n}")
        if not e:
            continue
        m = e.get("mix", {}).get("selection", {})
        A(f"| {n} | {m.get('terms', 0)} | {m.get('sentences_per_term', 0):.1f} | "
          f"{len(m.get('voices', []))} | "
          f"{e['seen_retrieved']['term_hit_norm']:.3f} | {e['seen_none']['term_hit_norm']:.3f} | "
          f"{e['unseen_retrieved']['term_hit_norm']:.3f} | {e['unseen_none']['term_hit_norm']:.3f} | "
          f"{e['seen_retrieved']['damage_rate']*100:.1f}% | "
          f"{e['seen_retrieved']['unrelated_insert_rate']*100:.2f}% | "
          f"{e['real_retrieved']['fair_wer']:.4f} | {e['real_retrieved']['damage_rate']*100:.1f}% |")
    A("")
    A("## Axis table (N = 9,000 acoustic rows, three compositions)")
    A("")
    A("| composition | rows | terms | sent/term | voices | seen hit ret | seen hit none | "
      "unseen hit ret | seen dmg ret | seen unrel ret | real WER ret | real dmg |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in AXES + ["n9000"]:
        name = f"scale_axis_{a}" if a in AXES else "scale_n9000"
        e = res["systems"].get(name)
        if not e:
            continue
        m = e.get("mix", {}).get("selection", {})
        A(f"| {name} | {m.get('rows', 0)} | {m.get('terms', 0)} | "
          f"{m.get('sentences_per_term', 0):.1f} | {len(m.get('voices', []))} | "
          f"{e['seen_retrieved']['term_hit_norm']:.3f} | {e['seen_none']['term_hit_norm']:.3f} | "
          f"{e['unseen_retrieved']['term_hit_norm']:.3f} | "
          f"{e['seen_retrieved']['damage_rate']*100:.1f}% | "
          f"{e['seen_retrieved']['unrelated_insert_rate']*100:.2f}% | "
          f"{e['real_retrieved']['fair_wer']:.4f} | {e['real_retrieved']['damage_rate']*100:.1f}% |")
    A("")
    if res["fit"]:
        f = res["fit"]["all"]
        A("## Log-linear fit")
        A("")
        A(f"seen-voice term hit (retrieved list) = {f['intercept']:.3f} + "
          f"{f['slope_per_decade']:.3f} x log10(N), R^2 {f['r2']:.3f}, RMSE {f['rmse']:.4f} "
          f"over N in {got}.")
        if "low" in res["fit"] and "high" in res["fit"]:
            A("")
            A(f"slope 500-9,000: {res['fit']['low']['slope_per_decade']:.3f} per decade; "
              f"slope 9,000-36,000: {res['fit']['high']['slope_per_decade']:.3f} per decade.")
        A("")
    if "fit_terms" in res:
        ft = res["fit_terms"]
        A("Against distinct training terms instead, pooling the size series and the axis series "
          f"(10 adapters): hit = {ft['intercept']:.3f} + {ft['slope_per_decade']:.3f} x "
          f"log10(terms), R^2 {ft['r2']:.3f}, RMSE {ft['rmse']:.4f}.")
        A("")
        A("| adapter | distinct terms | acoustic rows | seen hit ret |")
        A("|---|---:|---:|---:|")
        for name, nt, hit in sorted(ft["points"], key=lambda x: x[1]):
            A(f"| {name} | {nt} | {res['systems'][name]['mix']['acoustic_rows']} | {hit:.3f} |")
        A("")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
