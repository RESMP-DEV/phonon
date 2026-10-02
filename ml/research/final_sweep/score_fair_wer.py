#!/usr/bin/env python3
"""Score a hypothesis JSONL file with Phonon's fair-WER methodology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

NORMALIZER = EnglishTextNormalizer()


def fair_norm(text: str) -> str:
    return NORMALIZER((text or "").strip())


def strict_norm(text: str) -> str:
    return (text or "").strip().lower()


def corpus_wer(refs: list[str], hyps: list[str], norm) -> float:
    pairs = [(norm(ref) or "<empty>", norm(hyp) or "<empty>") for ref, hyp in zip(refs, hyps, strict=True)]
    return float(jiwer.wer([ref for ref, _ in pairs], [hyp for _, hyp in pairs]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hyps", type=Path)
    parser.add_argument("--hyp-field", default="hyp")
    parser.add_argument("--ref-field", default="ref")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.hyps.read_text().splitlines() if line.strip()]
    refs = [row[args.ref_field] for row in rows]
    hyps = [row[args.hyp_field] for row in rows]
    fair_exact = sum(fair_norm(ref) == fair_norm(hyp) for ref, hyp in zip(refs, hyps)) / len(rows)
    print(
        json.dumps(
            {
                "n": len(rows),
                "fair_wer": corpus_wer(refs, hyps, fair_norm),
                "strict_lc_wer": corpus_wer(refs, hyps, strict_norm),
                "fair_exact": fair_exact,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
