#!/usr/bin/env python3
"""Build an Aqua audio pack with a registered prompt baked into every row."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, default=None)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--max-dur", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=768)
    args = parser.parse_args()

    prompt = get_prompt(args.prompt_id)
    excluded = set()
    if args.exclude and args.exclude.exists():
        excluded = {
            json.loads(line)["audio"]
            for line in args.exclude.read_text().splitlines()
            if line.strip()
        }
    chats = []
    skipped = 0
    for line in args.manifest.read_text().splitlines():
        row = json.loads(line)
        text = str(row.get("corrected") or "").strip()
        duration = float(row.get("duration") or row.get("dur") or 0)
        audio_name = row.get("audio")
        audio_path = args.audio_root / str(audio_name)
        if (
            not text
            or audio_name in excluded
            or not args.min_dur <= duration <= args.max_dur
            or not audio_path.is_file()
        ):
            skipped += 1
            continue
        chats.append(
            [
                ChatMessage(role="system", content=[TextSegment(text=prompt)]),
                ChatMessage(role="user", content=[AudioSegment(audio=audio_path.read_bytes())]),
                ChatMessage(role="assistant", content=[TextSegment(text=text)]),
            ]
        )
        if args.limit and len(chats) >= args.limit:
            break
    if not chats:
        raise RuntimeError("no eligible Aqua rows")

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
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    metadata = {
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "excluded": str(args.exclude) if args.exclude else None,
        "prompt_id": args.prompt_id,
        "prompt_sha256": prompt_sha256(args.prompt_id),
        "rows": len(chats),
        "skipped": skipped,
        "duration_range": [args.min_dur, args.max_dur],
        "context_length": args.context_length,
    }
    (args.out / "pack_meta.json").write_text(json.dumps(metadata, indent=1))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
