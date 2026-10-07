#!/usr/bin/env python3
"""Build an Aqua pack whose system prompt is one fixed user-history card.

Unlike the dynamic retrieval pack, every row receives the byte-identical card
prompt. That is the layout a single-pass product engine can serve without a
retrieval query, so training on it matches the shippable prompt contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from liquid_audio import LFM2AudioDetokenizer, LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment
from prompts import get_prompt, prompt_sha256


def eligible_rows(
    manifest: Path,
    audio_root: Path,
    excluded: set[str],
    minimum_duration: float,
    maximum_duration: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("corrected") or "").strip()
        duration = float(row.get("duration") or row.get("dur") or 0)
        audio_name = str(row.get("audio") or "")
        if (
            not text
            or audio_name in excluded
            or not minimum_duration <= duration <= maximum_duration
            or not (audio_root / audio_name).is_file()
        ):
            continue
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--prompt-id", default="prose_history_dictation_v2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--max-dur", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=768)
    args = parser.parse_args()

    get_prompt(args.prompt_id)
    system_prompt = args.card.read_text()
    if not system_prompt.strip():
        raise RuntimeError(f"empty history card: {args.card}")

    excluded: set[str] = set()
    for path in args.exclude:
        if path.exists():
            excluded |= {
                json.loads(line)["audio"]
                for line in path.read_text().splitlines()
                if line.strip()
            }
    rows = eligible_rows(
        args.manifest, args.audio_root, excluded, args.min_dur, args.max_dur
    )
    if not rows:
        raise RuntimeError("no eligible Aqua rows")
    selected = rows[: args.limit] if args.limit else rows

    chats = []
    for row in selected:
        audio = args.audio_root / str(row["audio"])
        chats.append(
            [
                ChatMessage(role="system", content=[TextSegment(text=system_prompt)]),
                ChatMessage(role="user", content=[AudioSegment(audio=audio.read_bytes())]),
                ChatMessage(
                    role="assistant",
                    content=[TextSegment(text=str(row["corrected"]))],
                ),
            ]
        )

    LFM2AudioDetokenizer.cuda = lambda self, device=None: self
    processor = LFM2AudioProcessor.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B", device="cpu"
    ).eval()
    preprocess_dataset(
        chats,
        str(args.out),
        LFM2AudioChatMapper(processor),
        max_context_length=args.context_length,
    )

    metadata = {
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "excluded": [str(path) for path in args.exclude],
        "excluded_ids": len(excluded),
        "card": str(args.card),
        "card_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "prompt_id": args.prompt_id,
        "prompt_sha256": prompt_sha256(args.prompt_id),
        "fixed_prompt": True,
        "rows": len(chats),
        "skipped": len(rows) - len(selected),
        "duration_range": [args.min_dur, args.max_dur],
        "context_length": args.context_length,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pack_meta.json").write_text(json.dumps(metadata, indent=1))
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
