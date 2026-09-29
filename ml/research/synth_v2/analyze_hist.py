"""CPU length histograms: real Wispr yardstick vs synth_v1. No GPU."""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    BIN_NAMES,
    BINS,
    DATA_ROOT,
    HIST_PLAN,
    REAL_PAIRS,
    REAL_WERS_BY_BIN,
    V1_PAIRS,
)


def bin_name(n: int) -> str:
    for (a, b), name in zip(BINS, BIN_NAMES, strict=True):
        if a <= n <= b:
            return name
    return "0"


def hist_counts(ns: list[int]) -> dict[str, int]:
    c = Counter(bin_name(n) for n in ns)
    return {k: int(c.get(k, 0)) for k in BIN_NAMES}


def hist_fracs(counts: dict[str, int]) -> dict[str, float]:
    n = sum(counts.values()) or 1
    return {k: counts[k] / n for k in BIN_NAMES}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return float(ys[f])
    return float(ys[f] * (c - k) + ys[c] * (k - f))


def len_stats(ns: list[int]) -> dict:
    if not ns:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    xs = [float(x) for x in ns]
    return {
        "n": len(ns),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p10": percentile(xs, 0.10),
        "p90": percentile(xs, 0.90),
    }


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "tmp").mkdir(parents=True, exist_ok=True)

    real = load_jsonl(REAL_PAIRS)
    real_rows = []
    for row in real:
        target = (row.get("target") or row.get("kept") or "").strip()
        raw = (row.get("asr") or row.get("input") or "").strip()
        if not target or not raw:
            continue
        real_rows.append({"target": target, "input": raw})
    rlen = [len(r["target"].split()) for r in real_rows]
    real_counts = hist_counts(rlen)
    real_fracs = hist_fracs(real_counts)

    identical_lc = defaultdict(lambda: [0, 0])
    for row, n in zip(real_rows, rlen, strict=True):
        b = bin_name(n)
        identical_lc[b][1] += 1
        if row["target"].lower() == row["input"].lower():
            identical_lc[b][0] += 1

    v1 = load_jsonl(V1_PAIRS)
    em = [r for r in v1 if r.get("route") == "error_model"]
    tts = [r for r in v1 if r.get("route") == "tts_asr"]

    def pack(rows: list[dict]) -> dict:
        ns = [len((r.get("target") or "").split()) for r in rows]
        counts = hist_counts(ns)
        return {"counts": counts, "fracs": hist_fracs(counts), "len": len_stats(ns)}

    v1_all = pack(v1)
    v1_em = pack(em)
    v1_tts = pack(tts)

    # empirical 2-25 word-count mix from real (generation prior)
    wc = Counter(n for n in rlen if 2 <= n <= 25)
    wc_n = sum(wc.values()) or 1
    wordcount_2_25 = {str(i): wc[i] / wc_n for i in range(2, 26)}

    # shorts needed so 1-5/6-10/11-20 match real if we later pack N
    # leave packing to build_pairs; record the yardstick here.
    short_real = sum(1 for n in rlen if 2 <= n <= 25)

    out = {
        "real": {
            "n": len(rlen),
            "len": len_stats(rlen),
            "counts": real_counts,
            "fracs": real_fracs,
            "pct": {k: 100.0 * real_fracs[k] for k in BIN_NAMES},
            "n_short_2_25": short_real,
            "frac_short_2_25": short_real / max(len(rlen), 1),
            "identical_lc": {
                k: {
                    "n": identical_lc[k][1],
                    "n_equal": identical_lc[k][0],
                    "frac": (identical_lc[k][0] / identical_lc[k][1] if identical_lc[k][1] else 0.0),
                }
                for k in BIN_NAMES
            },
            "wordcount_2_25": wordcount_2_25,
        },
        "v1_all": v1_all,
        "v1_em": v1_em,
        "v1_tts": v1_tts,
        "note": (
            "v1 has no 1-5 or 6-10 rows and almost no 11-20; median 52 vs real 34. "
            "81+ is 21.7% real vs 2.8% v1 (p90 70 vs 139); v2 shorts cannot fill 81+."
        ),
    }
    HIST_PLAN.write_text(json.dumps(out, indent=2) + "\n")
    print("REAL length histogram (corrector_pairs_v0.jsonl, n={})".format(len(rlen)))
    print(
        "  mean={:.1f} median={:.1f} p10={:.1f} p90={:.1f}".format(
            out["real"]["len"]["mean"],
            out["real"]["len"]["median"],
            out["real"]["len"]["p10"],
            out["real"]["len"]["p90"],
        )
    )
    for k in BIN_NAMES:
        print(f"  {k}: {real_counts[k]} ({100.0 * real_fracs[k]:.2f}%)")
    print("V1 all n={}".format(v1_all["len"]["n"]))
    for k in BIN_NAMES:
        print(f"  {k}: {v1_all['counts'][k]} ({100.0 * v1_all['fracs'][k]:.2f}%)")
    print(f"wrote {HIST_PLAN}")
    if REAL_WERS_BY_BIN.exists():
        print(f"fair WER by bin already at {REAL_WERS_BY_BIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
