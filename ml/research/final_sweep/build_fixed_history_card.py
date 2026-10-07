#!/usr/bin/env python3
"""Build a fixed user-history card for no-oracle prototype inference.

The card is a small, deterministic set of the user's own raw-to-accepted
correction pairs sampled across the training timeline. It is rendered once into
a system prompt that a single-pass model consumes for every request, so no
per-request retrieval query and no oracle transcript are required.

Output stays on controlled storage: the rendered prompt contains personal
transcript text and must never be committed to Git or uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from history_prompt import history_prompt_sha256, render_history_prompt, tokenize

CARD_SELECTION_VERSION = "quantile-timestamp-has-correction-v1"
MAX_TOKENS_PER_SIDE = 55


def load_ids(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    return {
        str(json.loads(line)["audio"])
        for line in path.read_text().splitlines()
        if line.strip()
    }


def card_candidates(
    manifest: Path, excluded: set[str], minimum_duration: float, maximum_duration: float
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        raw = str(row.get("raw") or "").strip()
        corrected = str(row.get("corrected") or "").strip()
        audio = str(row.get("audio") or "")
        duration = float(row.get("duration") or 0)
        if (
            not raw
            or not corrected
            or audio in excluded
            or not row.get("has_correction")
            or raw == corrected
            or not minimum_duration <= duration <= maximum_duration
            or len(tokenize(raw)) > MAX_TOKENS_PER_SIDE
            or len(tokenize(corrected)) > MAX_TOKENS_PER_SIDE
        ):
            continue
        candidates.append(row)
    if not candidates:
        raise RuntimeError("no usable correction pairs for a history card")
    candidates.sort(key=lambda row: (str(row.get("timestamp") or ""), str(row["audio"])))
    return candidates


def select_card(candidates: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    """Sample evenly across the candidate timeline; deterministic for a fixed pool."""

    if count < 1:
        raise ValueError("card count must be positive")
    if count > len(candidates):
        raise RuntimeError(
            f"requested {count} history pairs but only {len(candidates)} are usable"
        )
    if count == 1:
        return [candidates[len(candidates) // 2]]
    step = (len(candidates) - 1) / (count - 1)
    return [candidates[round(index * step)] for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--prompt-id", default="prose_history_dictation_v2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--max-dur", type=float, default=20.0)
    args = parser.parse_args()

    excluded: set[str] = set()
    for path in args.exclude:
        excluded |= load_ids(path)
    candidates = card_candidates(args.manifest, excluded, args.min_dur, args.max_dur)
    selected = select_card(candidates, args.count)
    examples = [
        {
            "audio": str(row["audio"]),
            "raw": str(row["raw"]).strip(),
            "corrected": str(row["corrected"]).strip(),
        }
        for row in selected
    ]
    contract = {
        "selector": CARD_SELECTION_VERSION,
        "target_audio": "<fixed-user-history-card>",
        "history_audio": [example["audio"] for example in examples],
        "examples": examples,
    }
    prompt = render_history_prompt(contract, prompt_id=args.prompt_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(prompt)
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    metadata = {
        "card_selection": CARD_SELECTION_VERSION,
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "dynamic_prompt_sha256": history_prompt_sha256(contract, prompt_id=args.prompt_id),
        "history_audio": contract["history_audio"],
        "history_count": len(examples),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "excluded": [str(path) for path in args.exclude],
        "excluded_ids": len(excluded),
        "candidates": len(candidates),
        "duration_range": [args.min_dur, args.max_dur],
        "prompt_file": str(args.out),
        "non_claims": [
            "does not prove quality until evaluated",
            "rendered prompt contains personal transcript text and stays on controlled storage",
        ],
    }
    (args.out.with_suffix(args.out.suffix + ".meta.json")).write_text(
        json.dumps(metadata, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
