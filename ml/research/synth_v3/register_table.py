"""Step 1: register distribution of real targets vs synth v1/v2 (and v3 if present)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register import (  # noqa: E402
    FEATURES_ALL,
    FEATURES_TEXT,
    band_from,
    pair_features,
    read_jsonl,
    summarize,
)

REAL = Path("/data/phonon_corrector_v0/train.jsonl")
V1 = Path("/data/phonon_synth_v1/synth_pairs_v1.jsonl")
V2 = Path("/data/phonon_synth_v2/synth_pairs_v2.jsonl")
V3 = Path("/data/phonon_synth_v3/synth_pairs_v3_em.jsonl")
V3A = Path("/data/phonon_synth_v3/acoustic_pairs.jsonl")
V3F = Path("/data/phonon_synth_v3/synth_pairs_v3.jsonl")
OUT_MD = Path("/home/user/phonon/research/synth_v3/register.md")
OUT_JSON = Path("/data/phonon_synth_v3/register_stats.json")
BAND_JSON = Path("/data/phonon_synth_v3/register_band.json")

NAMES = {
    "func_frac": "function-word fraction",
    "contraction_rate": "contractions / word",
    "first_person_rate": "1st-person pronouns / word",
    "discourse_rate": "discourse markers / word",
    "mean_sent_len": "mean sentence length (words)",
    "np_start_frac": "sentences opening with a noun phrase",
    "edit_per_word": "input->target word edits / word",
}


def collect(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    for i, r in enumerate(read_jsonl(path)):
        if limit and i >= limit:
            break
        f = pair_features(r.get("input") or "", r.get("target") or "")
        if f:
            rows.append(f)
    return rows


def main() -> int:
    corpora = [("real (train.jsonl)", REAL), ("synth v1", V1), ("synth v2", V2)]
    if V3.exists():
        corpora.append(("synth v3 EM", V3))
    if V3A.exists():
        corpora.append(("v3 acoustic", V3A))
    if V3F.exists():
        corpora.append(("synth v3 full", V3F))
    stats = {}
    counts = {}
    for name, path in corpora:
        if not path.exists():
            print(f"missing {path}", flush=True)
            continue
        rows = collect(path)
        counts[name] = len(rows)
        stats[name] = summarize(rows, FEATURES_ALL)
        print(f"{name}: n={len(rows)}", flush=True)

    band = band_from(stats["real (train.jsonl)"], FEATURES_TEXT)
    BAND_JSON.parent.mkdir(parents=True, exist_ok=True)
    BAND_JSON.write_text(json.dumps({k: list(v) for k, v in band.items()}, indent=2) + "\n")

    names = [n for n, _ in corpora if n in stats]
    lines = [
        "# Spoken-register features: real accepted targets vs synthetic targets",
        "",
        "Per-row features of the **target** side of each corpus (what a refiner is trained to emit).",
        "Cells are median (p10 / p90). Rows with fewer than 3 words are skipped.",
        "`edit_per_word` is word-level Levenshtein(input, target) / target words, i.e. how much the",
        "pair asks the model to change; the other six are text-only and are what the v3 filter uses.",
        "",
        "Corpus sizes: " + ", ".join(f"{n} = {counts[n]}" for n in names),
        "",
        "| feature | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for f in FEATURES_ALL:
        cells = []
        for n in names:
            s = stats[n][f]
            cells.append(f"{s['median']:.3f} ({s['p10']:.3f} / {s['p90']:.3f})")
        lines.append(f"| {NAMES[f]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Real p10-p90 keep band (v3 filter)",
        "",
        "| feature | p10 | p90 |",
        "|---|---:|---:|",
    ]
    for f in FEATURES_TEXT:
        lo, hi = band[f]
        lines.append(f"| {NAMES[f]} | {lo:.4f} | {hi:.4f} |")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    OUT_JSON.write_text(json.dumps({"counts": counts, "stats": stats}, indent=1) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
