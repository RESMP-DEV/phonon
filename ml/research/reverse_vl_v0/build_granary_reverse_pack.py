#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from datasets import Audio, load_dataset
from liquid_audio import LFM2AudioDetokenizer, LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment

SYSTEM = "You are a transcription engine. Transcribe the user's audio exactly, including punctuation and casing. Text only."
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = sys.argv[2] if len(sys.argv) > 2 else "granary-en-2k-v1"

LFM2AudioDetokenizer.cuda = lambda self, device=None: self
stream = (
    load_dataset("espnet/yodas-granary", "English", split="asr_only", streaming=True)
    .cast_column("audio", Audio(decode=False))
)
chats = []
scanned = 0
for row in stream:
    scanned += 1
    duration = float(row.get("duration") or 0)
    text = str(row.get("text") or "").strip()
    audio = row.get("audio") or {}
    data = audio.get("bytes")
    if not (2 <= duration <= 20) or not text or not data:
        continue
    chats.append([
        ChatMessage(role="system", content=[TextSegment(text=SYSTEM)]),
        ChatMessage(role="user", content=[AudioSegment(audio=data)]),
        ChatMessage(role="assistant", content=[TextSegment(text=text)]),
    ])
    if len(chats) % 100 == 0:
        print(f"selected {len(chats)}/{scanned}", flush=True)
    if len(chats) >= N:
        break
if len(chats) < N:
    raise RuntimeError(f"only selected {len(chats)} rows from {scanned}")
processor = LFM2AudioProcessor.from_pretrained("LiquidAI/LFM2.5-Audio-1.5B", device="cpu").eval()
mapper = LFM2AudioChatMapper(processor)
preprocess_dataset(chats, OUT, mapper, max_context_length=768)
Path(OUT, "pack_meta.json").write_text(json.dumps({
    "source": "espnet/yodas-granary English asr_only streaming",
    "license": "CC-BY-3.0 (backing dataset); nvidia/Granary manifests CC-BY-4.0",
    "target_rows": N,
    "selected_rows": len(chats),
    "scanned_rows": scanned,
    "duration_range": [2, 20],
    "system": SYSTEM,
}, indent=1))
print(json.dumps({"out": OUT, "rows": len(chats), "scanned": scanned}), flush=True)
sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
