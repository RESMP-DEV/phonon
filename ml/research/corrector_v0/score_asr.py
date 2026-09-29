"""Fair WER of Parakeet raw vs target and Wispr ASR vs target, per split."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ASR_SCORES,
    PARAKEET_JSONL,
    RESEARCH_ROOT,
    exact_match_rate,
    fair_norm,
    read_jsonl,
    score_lists,
)


def per_split(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("split") or "unknown"].append(row)
    report: dict[str, dict] = {}
    for split, items in sorted(grouped.items()):
        items = [r for r in items if (r.get("target") or "").strip()]
        if not items:
            continue
        refs = [r.get("target") or "" for r in items]
        parakeet = [r.get("parakeet_raw") or "" for r in items]
        wispr = [r.get("wispr_asr") or "" for r in items]
        report[split] = {
            "n": len(items),
            "parakeet_vs_target": score_lists(refs, parakeet),
            "wispr_asr_vs_target": score_lists(refs, wispr),
            "parakeet_eq_target_fair": exact_match_rate(parakeet, refs, fair_norm),
            "wispr_eq_target_fair": exact_match_rate(wispr, refs, fair_norm),
            "parakeet_eq_wispr_fair": exact_match_rate(parakeet, wispr, fair_norm),
        }
    return report


def markdown_table(report: dict) -> str:
    lines = [
        "| split | n | parakeet fair WER | wispr asr fair WER | parakeet==target (fair) | wispr==target (fair) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, block in report.items():
        p = block["parakeet_vs_target"]
        w = block["wispr_asr_vs_target"]
        lines.append(
            f"| {split} | {block['n']} | {p['fair_wer']:.4f} | {w['fair_wer']:.4f} | "
            f"{block['parakeet_eq_target_fair']:.4f} | {block['wispr_eq_target_fair']:.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, default=PARAKEET_JSONL)
    parser.add_argument("--out", type=Path, default=ASR_SCORES)
    args = parser.parse_args()
    rows = read_jsonl(args.inp)
    if not rows:
        raise FileNotFoundError(f"no parakeet rows in {args.inp}")
    report = per_split(rows)
    payload = {"source": str(args.inp), "n": len(rows), "splits": report}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    table = markdown_table(report)
    print(table)
    snippet = RESEARCH_ROOT / "asr_scores.md"
    snippet.write_text("# Parakeet v2 vs Wispr ASR\n\n" + table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
