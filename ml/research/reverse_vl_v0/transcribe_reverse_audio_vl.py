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
from prompts import get_prompt, prompt_sha256
from reverse_audio_vl import LORA_TARGETS, ReverseAudioVL
from safetensors.torch import load_file


def infer_lora_rank(state: dict[str, torch.Tensor]) -> int:
    """Infer the LoRA rank from an adapter's A-matrix shapes."""

    ranks = {
        int(tensor.shape[0])
        for key, tensor in state.items()
        if ".lora_A." in key and tensor.ndim == 2
    }
    if len(ranks) != 1:
        raise RuntimeError(f"adapter does not contain one consistent LoRA rank: {sorted(ranks)}")
    return ranks.pop()


def load_adapter(
    model: ReverseAudioVL, path: Path, *, rank: int = 0
) -> None:
    for parameter in model.vl.parameters():
        parameter.requires_grad_(False)
    for parameter in model.audio.parameters():
        parameter.requires_grad_(False)
    state = load_file(str(path), device="cpu")
    if rank <= 0:
        rank = infer_lora_rank(state)
    inject_adapter_in_model(
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
        ),
        model.vl,
    )
    vl_state = {key.removeprefix("vl."): value for key, value in state.items() if key.startswith("vl.")}
    adapter_state = {
        key.removeprefix("audio_adapter."): value
        for key, value in state.items()
        if key.startswith("audio_adapter.")
    }
    conformer_state = {
        key.removeprefix("conformer."): value
        for key, value in state.items()
        if key.startswith("conformer.")
    }
    result = model.vl.load_state_dict(vl_state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected LoRA keys: {result.unexpected_keys[:5]}")
    model.audio.audio_adapter.load_state_dict(adapter_state, strict=True)
    if conformer_state:
        layers = getattr(model.audio.conformer, "layers", None)
        if not isinstance(layers, torch.nn.ModuleList):
            raise RuntimeError("conformer adapter has no nn.ModuleList layers")
        by_layer: dict[int, dict[str, torch.Tensor]] = {}
        for key, value in conformer_state.items():
            if "." not in key:
                raise RuntimeError(f"malformed conformer tensor key: {key}")
            index_text, parameter_name = key.split(".", 1)
            index = int(index_text)
            if not 0 <= index < len(layers):
                raise RuntimeError(f"conformer layer index out of range: {index}")
            by_layer.setdefault(index, {})[parameter_name] = value
        for index, layer_state in by_layer.items():
            layers[index].load_state_dict(layer_state, strict=True)
    model.vl.to(model.device).eval()
    model.audio.to(model.device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", default="slice-eval-500.jsonl")
    parser.add_argument("--out", default="hyps_reverse_vl_v1.jsonl")
    parser.add_argument("--adapter", default="reverse-audio-vl-v1/reverse_audio_vl_adapter.safetensors")
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="LoRA rank; zero infers the rank from the adapter",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--audio-root", default="~/aqua-training-data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-id", default="prose_dictation_v1")
    args = parser.parse_args()
    get_prompt(args.prompt_id)

    device = torch.device(args.device)
    model = ReverseAudioVL.from_pretrained(device=device)
    load_adapter(model, Path(args.adapter), rank=args.lora_rank)

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
                prompt_id=args.prompt_id,
            )
            result = {
                "ts": row.get("ts"),
                "audio": row["audio"],
                "ref": row.get("ref", ""),
                "raw_aqua": row.get("raw", row.get("raw_aqua", "")),
                "hyp": hyp,
                "dur": row.get("dur", 0),
                "gen_s": round(time.time() - t0, 2),
                "prompt_id": args.prompt_id,
                "prompt_sha256": prompt_sha256(args.prompt_id),
            }
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(f"[{index + 1}/{len(rows)}] gen={result['gen_s']:.1f}s", flush=True)
    print("DONE", output, flush=True)


if __name__ == "__main__":
    main()
