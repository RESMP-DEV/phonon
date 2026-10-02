#!/usr/bin/env python3
"""Stream public YODAS-Granary audio and bake a registered prompt into rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Audio, load_dataset
from liquid_audio import LFM2AudioDetokenizer, LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment
from prompts import get_prompt, prompt_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--revision", default="969944574ea3f37890beaf67ea651e160cfaf043")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--max-dur", type=float, default=20.0)
    parser.add_argument("--context-length", type=int, default=768)
    args = parser.parse_args()

    prompt = get_prompt(args.prompt_id)
    stream = (
        load_dataset(
            "espnet/yodas-granary",
            "English",
            split="asr_only",
            streaming=True,
            revision=args.revision,
        )
        .cast_column("audio", Audio(decode=False))
    )
    chats = []
    scanned = 0
    for row in stream:
        scanned += 1
        duration = float(row.get("duration") or 0)
        text = str(row.get("text") or "").strip()
        data = (row.get("audio") or {}).get("bytes")
        if not args.min_dur <= duration <= args.max_dur or not text or not data:
            continue
        chats.append(
            [
                ChatMessage(role="system", content=[TextSegment(text=prompt)]),
                ChatMessage(role="user", content=[AudioSegment(audio=data)]),
                ChatMessage(role="assistant", content=[TextSegment(text=text)]),
            ]
        )
        if len(chats) >= args.rows:
            break
    if len(chats) != args.rows:
        raise RuntimeError(f"selected {len(chats)} rows from {scanned}; wanted {args.rows}")

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
        "source": "espnet/yodas-granary English asr_only streaming",
        "revision": args.revision,
        "license": "CC-BY-3.0 backing audio; nvidia/Granary CC-BY-4.0 manifests",
        "prompt_id": args.prompt_id,
        "prompt_sha256": prompt_sha256(args.prompt_id),
        "rows": len(chats),
        "scanned": scanned,
        "duration_range": [args.min_dur, args.max_dur],
        "context_length": args.context_length,
    }
    (args.out / "pack_meta.json").write_text(json.dumps(metadata, indent=1))
    print(json.dumps(metadata), flush=True)
    # The streaming dataset's destructor is unstable in this environment; data
    # is already durable, so bypass it after flushing.
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
