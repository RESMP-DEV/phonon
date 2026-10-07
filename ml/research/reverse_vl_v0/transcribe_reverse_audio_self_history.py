#!/usr/bin/env python3
"""Evaluate two-pass self-retrieval for the reverse Audio->VL graft."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from history_prompt import (
    build_history_index,
    history_contract,
    history_prompt_sha256,
    render_history_prompt,
)
from prompts import get_prompt, prompt_sha256
from reverse_audio_vl import ReverseAudioVL
from transcribe_reverse_audio_vl import load_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", default="slice-eval-500.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--audio-root", default="/home/kearm/aqua-training-data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--draft-prompt-id", default="prose_dictation_v1")
    parser.add_argument("--final-prompt-id", default="prose_history_dictation_v2")
    parser.add_argument("--history-manifest", type=Path, required=True)
    parser.add_argument("--history-count", type=int, default=2)
    args = parser.parse_args()
    if args.draft_prompt_id == args.final_prompt_id:
        parser.error("draft and final prompt IDs must differ")
    get_prompt(args.draft_prompt_id)
    get_prompt(args.final_prompt_id)

    history_pool = [
        json.loads(line)
        for line in args.history_manifest.read_text().splitlines()
        if line.strip()
    ]
    history_index = build_history_index(history_pool)
    draft_prompt = get_prompt(args.draft_prompt_id)
    root = Path(args.audio_root).expanduser()
    rows = [
        json.loads(line)
        for line in Path(args.slice).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    output = Path(args.out)
    done = set()
    if output.exists():
        done = {
            json.loads(line)["audio"]
            for line in output.read_text().splitlines()
            if line.strip()
        }

    device = torch.device(args.device)
    model = ReverseAudioVL.from_pretrained(device=device)
    load_adapter(model, Path(args.adapter), rank=args.lora_rank)
    with output.open("a", encoding="utf-8") as sink:
        for index, row in enumerate(rows):
            audio_id = str(row["audio"])
            if audio_id in done:
                continue
            wav, sampling_rate = sf.read(root / audio_id, dtype="float32")
            wave = torch.from_numpy(wav)
            if wave.dim() == 1:
                wave = wave[None, :]

            started = time.perf_counter()
            draft = model.generate(
                wave,
                int(sampling_rate),
                max_new_tokens=args.max_new_tokens,
                system=draft_prompt,
            )
            draft_s = time.perf_counter() - started
            retrieval_started = time.perf_counter()
            # Deliberately pass only the audio ID. The slice row's raw Aqua text
            # and accepted label are unavailable to self-retrieval.
            target: dict[str, object] = {"audio": audio_id}
            contract = history_contract(
                target,
                history_pool,
                count=args.history_count,
                index=history_index,
                query_text=draft,
            )
            final_prompt = render_history_prompt(
                contract, prompt_id=args.final_prompt_id
            )
            retrieval_s = time.perf_counter() - retrieval_started
            final_started = time.perf_counter()
            hypothesis = model.generate(
                wave,
                int(sampling_rate),
                max_new_tokens=args.max_new_tokens,
                system=final_prompt,
            )
            final_s = time.perf_counter() - final_started
            result = {
                "ts": row.get("ts"),
                "audio": audio_id,
                "ref": row.get("ref", ""),
                "draft": draft,
                "hyp": hypothesis,
                "dur": row.get("dur", 0),
                "draft_s": round(draft_s, 4),
                "retrieval_s": round(retrieval_s, 4),
                "final_s": round(final_s, 4),
                "total_s": round(draft_s + retrieval_s + final_s, 4),
                "draft_prompt_id": args.draft_prompt_id,
                "draft_prompt_sha256": prompt_sha256(args.draft_prompt_id),
                "prompt_id": args.final_prompt_id,
                "prompt_sha256": prompt_sha256(args.final_prompt_id),
                "dynamic_prompt_sha256": history_prompt_sha256(
                    contract, prompt_id=args.final_prompt_id
                ),
                "history_audio": contract["history_audio"],
                "oracle_query": False,
            }
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            count = index + 1
            if count % 10 == 0 or count == len(rows):
                print(
                    f"[{count}/{len(rows)}] draft={draft_s:.1f}s "
                    f"retrieve={retrieval_s:.3f}s final={final_s:.1f}s",
                    flush=True,
                )
    print("DONE", output, flush=True)


if __name__ == "__main__":
    main()
