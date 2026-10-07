#!/usr/bin/env python3
"""Evaluate a fused audio-native SALM checkpoint as a standalone model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from liquid_audio import (
    ChatState,
    LFM2AudioDetokenizer,
    LFM2AudioModel,
    LFM2AudioProcessor,
)
from safetensors.torch import load_file

sys.path.insert(0, "/home/kearm/phonon/ml/research/final_sweep")
sys.path.insert(0, "/home/kearm/phonon/ml/research/salm_vision_v0")

from history_prompt import (
    build_history_index,
    history_contract,
    history_prompt_sha256,
    render_history_prompt,
)
from salm_vision import install_vision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--audio-root", type=Path, default=Path("/home/kearm/aqua-training-data")
    )
    parser.add_argument("--history-manifest", type=Path, required=True)
    parser.add_argument("--history-count", type=int, default=2)
    parser.add_argument("--prompt-id", default="prose_history_dictation_v2")
    parser.add_argument("--vision-rows", type=Path, default=None)
    parser.add_argument("--init-variant", default="omp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    LFM2AudioDetokenizer.cuda = lambda self, device=None: self

    model = LFM2AudioModel.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B", device=device, dtype=torch.bfloat16
    )
    target = model.state_dict()
    merged = load_file(str(args.merged))
    adapted = {}
    for key, tensor in merged.items():
        candidate = key if key in target else f"lfm.{key}"
        if candidate in target and target[candidate].shape == tensor.shape:
            adapted[candidate] = tensor
    missing, unexpected = model.load_state_dict(adapted, strict=False)
    unexpected = [k for k in unexpected if "lora_" not in k]
    if unexpected:
        raise RuntimeError(f"unexpected merged keys: {unexpected[:5]}")
    print(
        f"merged weights applied: {len(adapted)}/{len(merged)} keys; unmatched={len(missing)}",
        flush=True,
    )
    if args.vision_rows:
        install_vision(
            model, rows_path=args.vision_rows, init_variant=args.init_variant, verbose=False
        )
    model.to(device).eval()

    history_pool = [
        json.loads(line)
        for line in args.history_manifest.read_text().splitlines()
        if line.strip()
    ]
    index = build_history_index(history_pool)
    proc = LFM2AudioProcessor.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B", device=device
    ).eval()

    rows = [
        json.loads(line) for line in args.slice.read_text().splitlines() if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if args.out.exists():
        done = {
            json.loads(line)["audio"]
            for line in args.out.read_text().splitlines()
            if line.strip()
        }

    with args.out.open("a", encoding="utf-8") as sink:
        for position, row in enumerate(rows):
            if row["audio"] in done:
                continue
            contract = history_contract(
                row, history_pool, count=args.history_count, index=index
            )
            system = render_history_prompt(contract, prompt_id=args.prompt_id)
            prompt_sha = history_prompt_sha256(contract, prompt_id=args.prompt_id)
            wav, rate = sf.read(args.audio_root / row["audio"], dtype="float32")
            wave = torch.from_numpy(wav)
            if wave.dim() == 1:
                wave = wave[None, :]
            started = time.time()
            with torch.no_grad():
                chat = ChatState(proc)
                chat.new_turn("system")
                chat.add_text(system)
                chat.end_turn()
                chat.new_turn("user")
                chat.add_audio(wave.to(device), int(rate))
                chat.end_turn()
                chat.new_turn("assistant")
                tokens = []
                for token in model.generate_sequential(
                    **chat, max_new_tokens=args.max_new_tokens, text_temperature=None
                ):
                    if token.numel() != 1:
                        break
                    if int(token.item()) == 7:
                        break
                    tokens.append(token)
            hyp = (
                proc.text.decode(torch.cat(tokens))
                .removesuffix("<|text_end|>")
                .strip()
                if tokens
                else ""
            )
            sink.write(
                json.dumps(
                    {
                        "audio": row["audio"],
                        "ref": row.get("ref", ""),
                        "hyp": hyp,
                        "dur": row.get("dur", 0),
                        "gen_s": round(time.time() - started, 2),
                        "prompt_id": args.prompt_id,
                        "dynamic_prompt_sha256": prompt_sha,
                        "history_audio": contract["history_audio"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()
            if (position + 1) % 10 == 0 or position + 1 == len(rows):
                print(
                    f"[{position + 1}/{len(rows)}] gen={time.time() - started:.1f}s",
                    flush=True,
                )
    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
