#!/usr/bin/env python3
"""Evaluate the reverse Audio->VL graft on Aqua audio clips."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from peft import LoraConfig, inject_adapter_in_model
from reverse_audio_vl import LORA_TARGETS, ReverseAudioVL
from safetensors.torch import load_file


def load_adapter(model: ReverseAudioVL, path: Path, *, rank: int = 16) -> None:
    for parameter in model.vl.parameters():
        parameter.requires_grad_(False)
    for parameter in model.audio.parameters():
        parameter.requires_grad_(False)
    inject_adapter_in_model(
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
        ),
        model.vl,
    )
    state = load_file(str(path), device="cpu")
    vl_state = {key.removeprefix("vl."): value for key, value in state.items() if key.startswith("vl.")}
    adapter_state = {
        key.removeprefix("audio_adapter."): value
        for key, value in state.items()
        if key.startswith("audio_adapter.")
    }
    result = model.vl.load_state_dict(vl_state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected LoRA keys: {result.unexpected_keys[:5]}")
    model.audio.audio_adapter.load_state_dict(adapter_state, strict=True)
    model.vl.to(model.device).eval()
    model.audio.to(model.device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", default="slice-eval-500.jsonl")
    parser.add_argument("--out", default="hyps_reverse_vl_v1.jsonl")
    parser.add_argument("--adapter", default="reverse-audio-vl-v1/reverse_audio_vl_adapter.safetensors")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--audio-root", default="~/aqua-training-data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = ReverseAudioVL.from_pretrained(device=device)
    load_adapter(model, Path(args.adapter), rank=16)

    rows = [json.loads(line) for line in Path(args.slice).read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    root = Path(args.audio_root).expanduser()
    output = Path(args.out)
    done = set()
    if output.exists():
        done = {json.loads(line)["audio"] for line in output.read_text().splitlines() if line.strip()}
    with output.open("a", encoding="utf-8") as sink:
        for index, row in enumerate(rows):
            if row["audio"] in done:
                continue
            wav, sampling_rate = sf.read(root / row["audio"], dtype="float32")
            wave = torch.from_numpy(wav)
            if wave.dim() == 1:
                wave = wave[None, :]
            t0 = time.time()
            hyp = model.generate(
                wave,
                int(sampling_rate),
                max_new_tokens=args.max_new_tokens,
            )
            result = {
                "ts": row.get("ts"),
                "audio": row["audio"],
                "ref": row.get("ref", ""),
                "raw_aqua": row.get("raw", row.get("raw_aqua", "")),
                "hyp": hyp,
                "dur": row.get("dur", 0),
                "gen_s": round(time.time() - t0, 2),
            }
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(f"[{index + 1}/{len(rows)}] gen={result['gen_s']:.1f}s", flush=True)
    print("DONE", output, flush=True)


if __name__ == "__main__":
    main()
