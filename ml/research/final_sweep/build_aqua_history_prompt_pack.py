#!/usr/bin/env python3
"""Build an Aqua pack with retrieved historical corrections in each prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from history_prompt import (
    HISTORY_SELECTOR_VERSION,
    build_history_index,
    history_contract,
    history_prompt_sha256,
    render_history_prompt,
)
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
    parser.add_argument("--exclude", type=Path, default=None)
    parser.add_argument("--prompt-id", default="prose_history_dictation_v2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--max-dur", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--history-count", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=768)
    args = parser.parse_args()

    get_prompt(args.prompt_id)
    excluded = set()
    if args.exclude and args.exclude.exists():
        excluded = {
            json.loads(line)["audio"]
            for line in args.exclude.read_text().splitlines()
            if line.strip()
        }
    rows = eligible_rows(
        args.manifest,
        args.audio_root,
        excluded,
        args.min_dur,
        args.max_dur,
    )
    if not rows:
        raise RuntimeError("no eligible Aqua rows")
    selected_rows = rows[: args.limit] if args.limit else rows
    history_index = build_history_index(rows)

    chats = []
    history_ids: dict[str, list[str]] = {}
    dynamic_hashes: dict[str, str] = {}
    for row in selected_rows:
        contract = history_contract(
            row,
            rows,
            count=args.history_count,
            index=history_index,
        )
        prompt = render_history_prompt(contract, prompt_id=args.prompt_id)
        audio_path = args.audio_root / str(row["audio"])
        chats.append(
            [
                ChatMessage(role="system", content=[TextSegment(text=prompt)]),
                ChatMessage(
                    role="user", content=[AudioSegment(audio=audio_path.read_bytes())]
                ),
                ChatMessage(
                    role="assistant",
                    content=[TextSegment(text=str(row["corrected"]))],
                ),
            ]
        )
        audio_id = str(row["audio"])
        history_ids[audio_id] = [str(value) for value in contract["history_audio"]]
        dynamic_hashes[audio_id] = history_prompt_sha256(
            contract, prompt_id=args.prompt_id
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

    history_contract_payload = {
        "selector": HISTORY_SELECTOR_VERSION,
        "history_count": args.history_count,
        "target_to_history": history_ids,
    }
    contract_encoded = json.dumps(
        history_contract_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    metadata = {
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "excluded": str(args.exclude) if args.exclude else None,
        "prompt_id": args.prompt_id,
        "prompt_sha256": prompt_sha256(args.prompt_id),
        "dynamic_prompt": True,
        "history_selector": HISTORY_SELECTOR_VERSION,
        "history_count": args.history_count,
        "history_manifest_sha256": manifest_sha256,
        "history_contract_sha256": hashlib.sha256(contract_encoded).hexdigest(),
        "dynamic_prompt_hashes": dynamic_hashes,
        "rows": len(chats),
        "skipped": len(rows) - len(selected_rows),
        "duration_range": [args.min_dur, args.max_dur],
        "context_length": args.context_length,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "history-contract.json").write_text(
        json.dumps(history_contract_payload, indent=1, sort_keys=True) + "\n"
    )
    (args.out / "pack_meta.json").write_text(json.dumps(metadata, indent=1))
    print(json.dumps({key: value for key, value in metadata.items() if key != "dynamic_prompt_hashes"}), flush=True)


if __name__ == "__main__":
    main()
