"""Step 4b: register features of refiner OUTPUTS on a holdout set (raw / reference / each model)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register import FEATURES_TEXT, in_band_count, read_jsonl, summarize, text_features  # noqa: E402

PRED = Path("/data/phonon_corrector_v0/eval_predictions")
BAND = Path("/data/phonon_synth_v3/register_band.json")
NAMES = {
    "func_frac": "func-word frac",
    "contraction_rate": "contractions/word",
    "first_person_rate": "1st person/word",
    "discourse_rate": "discourse/word",
    "mean_sent_len": "mean sent len",
    "np_start_frac": "NP-start frac",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="wispr_holdout120")
    ap.add_argument("--labels", default="")
    ap.add_argument("--out-md", type=Path,
                    default=Path("/home/user/phonon/research/synth_v3/register_outputs.md"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data/phonon_synth_v3/register_outputs.json"))
    args = ap.parse_args()

    band = {k: tuple(v) for k, v in json.loads(BAND.read_text()).items() if k in FEATURES_TEXT}
    labels = [l for l in args.labels.split(",") if l]
    series: list[tuple[str, list[str]]] = []
    first = None
    for lab in labels:
        p = PRED / f"{lab}_{args.set}.jsonl"
        if not p.exists():
            print(f"missing {p}", flush=True)
            continue
        rows = list(read_jsonl(p))
        if first is None:
            first = rows
            series.append(("raw input", [r.get("input") or "" for r in rows]))
            series.append(("reference", [r.get("reference") or "" for r in rows]))
        series.append((lab, [r.get("output") or "" for r in rows]))

    out = {}
    lines = [
        f"# Register of refiner outputs on {args.set}",
        "",
        "Median (p10 / p90) per feature, plus the share of clips whose features land inside the",
        "real p10-p90 band on at least 5 of 6 features. `reference` is what the user accepted.",
        "",
        "| system | " + " | ".join(NAMES[f] for f in FEATURES_TEXT) + " | in-band >=5/6 |",
        "|---|" + "---:|" * (len(FEATURES_TEXT) + 1),
    ]
    for name, texts in series:
        feats = [f for f in (text_features(t) for t in texts) if f]
        if not feats:
            continue
        s = summarize(feats, FEATURES_TEXT)
        good = sum(1 for f in feats if in_band_count(f, band)[0] >= 5) / len(feats)
        out[name] = {"summary": s, "in_band_frac": good, "n": len(feats)}
        cells = [f"{s[f]['median']:.3f} ({s[f]['p10']:.3f}/{s[f]['p90']:.3f})" for f in FEATURES_TEXT]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {good:.2f} |")
    lines.append("")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines))
    args.out_json.write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
