"""STEP 2b: adaptive list budget = min(cap, max(30, mult * hypothesis words)) + score threshold.

Pure python over the saved top-100 lists.  heldout_old/seen is the only tuning set: the
threshold is chosen there and applied unchanged to heldout_new, heldout_pool2 and the
real-audio holdout.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from statistics import median

D = Path("/data/phonon_index_v0")
SETS = ["old_seen", "new", "real", "pool2"]
THRESHOLDS = [0, 50, 55, 60, 65, 70, 72, 74, 76, 78, 80, 82, 85]
MULT, CAP = 3, 80


def load(name):
    return json.loads((D / f"lists_{name}.json").read_text())


def budget(nwords, mult=MULT, cap=CAP):
    return min(cap, max(30, mult * nwords))


def evaluate(rows, thr, mult=MULT, cap=CAP, fixed=None):
    """fixed=k -> plain top-k list.  Returns recall over gold pairs + length stats."""
    hit = tot = 0
    lens = []
    ceil_hit = 0
    for r in rows:
        b = fixed if fixed else budget(r["nwords"], mult, cap)
        lst = [t for t, s in r["top"][:b] if s >= thr]
        lens.append(len(lst))
        g = set(r["gold"])
        if not g:
            continue
        tot += len(g)
        hit += len(g & set(lst))
        ceil_hit += min(b, len(g))
    return {
        "recall": hit / max(1, tot), "pairs": tot,
        "mean_len": sum(lens) / max(1, len(lens)),
        "p50_len": median(lens) if lens else 0,
        "p90_len": sorted(lens)[int(0.9 * len(lens))] if lens else 0,
        "max_len": max(lens) if lens else 0,
        "slot_ceiling": ceil_hit / max(1, tot),
    }


data = {s: load(s) for s in SETS}
for s, rows in data.items():
    print(f"{s}: rows={len(rows)} mean_words={sum(r['nwords'] for r in rows)/len(rows):.1f} "
          f"mean_gold={sum(len(r['gold']) for r in rows)/len(rows):.2f}")

RES = {"mult": MULT, "cap": CAP, "sets": {}}

# ---- threshold sweep on the tuning set -------------------------------------
tune = []
for thr in THRESHOLDS:
    e = evaluate(data["old_seen"], thr)
    tune.append({"thr": thr, **{k: round(v, 4) for k, v in e.items()}})
    print(f"tune thr={thr:>3} recall@budget={e['recall']:.4f} mean_len={e['mean_len']:.1f}")
best = max(t["recall"] for t in tune)
T = max(t["thr"] for t in tune if t["recall"] >= best - 0.002)
print(f"chosen threshold (tuned on heldout_old/seen) = {T}  (best recall {best:.4f})")
RES["tuning"] = tune
RES["threshold"] = T

# ---- per set ---------------------------------------------------------------
for s in SETS:
    rows = data[s]
    r10 = evaluate(rows, 0, fixed=10)
    r30 = evaluate(rows, 0, fixed=30)
    rb0 = evaluate(rows, 0)
    rbT = evaluate(rows, T)
    hist = Counter()
    for r in rows:
        lst = [t for t, sc in r["top"][:budget(r["nwords"])] if sc >= T]
        hist[min(80, (len(lst) // 10) * 10)] += 1
    RES["sets"][s] = {
        "rows": len(rows),
        "mean_words": round(sum(r["nwords"] for r in rows) / len(rows), 1),
        "mean_gold": round(sum(len(r["gold"]) for r in rows) / len(rows), 2),
        "fixed10": r10, "fixed30": r30, "budget_thr0": rb0, "budget_thrT": rbT,
        "len_hist_by10": dict(sorted(hist.items())),
    }
    print(f"\n[{s}] r@10={r10['recall']:.4f}  r@30={r30['recall']:.4f} (len 30)  "
          f"r@budget(T=0)={rb0['recall']:.4f} (len {rb0['mean_len']:.1f})  "
          f"r@budget(T={T})={rbT['recall']:.4f} (len {rbT['mean_len']:.1f})  "
          f"30-slot ceiling={r30['slot_ceiling']:.4f} budget ceiling={rb0['slot_ceiling']:.4f}")

# ---- term-dense rows on the real-audio set ---------------------------------
real = data["real"]
buckets = {"1-5": [], "6-15": [], "16-30": [], ">30": []}
for r in real:
    n = len(r["gold"])
    k = "1-5" if n <= 5 else "6-15" if n <= 15 else "16-30" if n <= 30 else ">30"
    buckets[k].append(r)
dense = {}
for k, rows in buckets.items():
    if not rows:
        continue
    dense[k] = {
        "rows": len(rows),
        "gold_pairs": sum(len(r["gold"]) for r in rows),
        "fixed30": round(evaluate(rows, 0, fixed=30)["recall"], 4),
        "budget": round(evaluate(rows, T)["recall"], 4),
        "budget_len": round(evaluate(rows, T)["mean_len"], 1),
        "fixed30_ceiling": round(evaluate(rows, 0, fixed=30)["slot_ceiling"], 4),
        "budget_ceiling": round(evaluate(rows, T)["slot_ceiling"], 4),
    }
    print(f"real gold={k:>6}: rows={dense[k]['rows']:>3} pairs={dense[k]['gold_pairs']:>4} "
          f"r@30={dense[k]['fixed30']:.4f} -> r@budget={dense[k]['budget']:.4f} "
          f"(ceiling {dense[k]['fixed30_ceiling']:.4f} -> {dense[k]['budget_ceiling']:.4f})")
RES["real_by_gold_count"] = dense

# ---- rule sweep ------------------------------------------------------------
sweep = []
for mult in (2, 3, 4):
    for cap in (50, 80, 100):
        row = {"mult": mult, "cap": cap}
        for s in SETS:
            e = evaluate(data[s], T, mult, cap)
            row[s] = round(e["recall"], 4)
            row[f"{s}_len"] = round(e["mean_len"], 1)
        sweep.append(row)
        print(f"mult={mult} cap={cap} " + " ".join(f"{s}={row[s]:.4f}/{row[f'{s}_len']:.0f}"
                                                   for s in SETS))
RES["rule_sweep"] = sweep

# ---- threshold applied to every set (the length/recall trade) --------------
thr_table = []
for thr in (0, 65, 70, 74, 78, 82):
    row = {"thr": thr}
    for s in SETS:
        e = evaluate(data[s], thr)
        row[s] = round(e["recall"], 4)
        row[f"{s}_len"] = round(e["mean_len"], 1)
    thr_table.append(row)
    print(f"thr={thr:>3} " + "  ".join(f"{s}={row[s]:.4f}/len{row[f'{s}_len']:.1f}" for s in SETS))
RES["threshold_table"] = thr_table
(D / "budget.json").write_text(json.dumps(RES, indent=1) + "\n")
print("STEP 2b done")
