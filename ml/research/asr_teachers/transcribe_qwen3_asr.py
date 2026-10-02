#!/usr/bin/env python3
"""Run the offline Qwen3-ASR teacher over a frozen Phonon audio slice."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
DEFAULT_REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"


def read_slice(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty audio slice: {path}")
    return rows


def completed_keys(output: Path) -> set[str]:
    if not output.exists():
        return set()
    keys: set[str] = set()
    for line in output.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["audio"])
        if str(row.get("hyp", "")).strip():
            keys.add(key)
    return keys


def decode_transcription(value: str | list[str]) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            raise RuntimeError(
                f"expected one transcription, got {len(value)}"
            )
        value = value[0]
    return value.strip()


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    def serialize_set(value: Any) -> list[str]:
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"unsupported metadata value: {type(value).__name__}")

    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=serialize_set) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_slice(args.slice)
    if args.limit:
        rows = rows[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.force and args.out.exists():
        args.out.unlink()
    done = completed_keys(args.out)
    metadata_path = args.out.with_suffix(".meta.json")
    write_metadata(
        metadata_path,
        {
            "status": "running",
            "model": args.model,
            "revision": args.revision,
            "slice": str(args.slice),
            "audio_root": str(args.audio_root),
            "output": str(args.out),
            "selected_rows": len(rows),
            "completed_rows": len(done),
            "language_mode": "auto",
            "started_at_unix": time.time(),
        },
    )

    import torch
    import transformers

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Qwen3-ASR teacher requires CUDA for the frozen-slice run")
    device = torch.device(args.device)
    processor = transformers.AutoProcessor.from_pretrained(
        args.model, revision=args.revision
    )
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    generated_rows = 0
    started = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as sink:
        for index, row in enumerate(rows):
            audio_key = str(row["audio"])
            if audio_key in done:
                continue
            audio_path = args.audio_root / audio_key
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            request_started = time.perf_counter()
            with torch.inference_mode():
                inputs = processor.apply_transcription_request(audio=[str(audio_path)])
                inputs.to(device=device, dtype=model.dtype)
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                continuation = output[:, inputs["input_ids"].shape[1] :]
                hyp = decode_transcription(
                    processor.decode(
                        continuation, return_format="transcription_only"
                    )
                )
            elapsed = time.perf_counter() - request_started
            result = {
                "audio": audio_key,
                "ref": row.get("ref", ""),
                "raw_aqua": row.get("raw", row.get("raw_aqua", "")),
                "hyp": hyp,
                "dur": row.get("dur", 0),
                "gen_s": round(elapsed, 3),
                "model": args.model,
                "revision": args.revision,
                "language_mode": "auto",
            }
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            generated_rows += 1
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(
                    f"[{index + 1}/{len(rows)}] generated={generated_rows} "
                    f"last={elapsed:.2f}s elapsed={time.perf_counter() - started:.0f}s",
                    flush=True,
                )
                write_metadata(
                    metadata_path,
                    {
                        "status": "running",
                        "model": args.model,
                        "revision": args.revision,
                        "slice": str(args.slice),
                        "audio_root": str(args.audio_root),
                        "output": str(args.out),
                        "selected_rows": len(rows),
                        "completed_rows": sorted(completed_keys(args.out)),
                        "language_mode": "auto",
                        "started_at_unix": time.time(),
                    },
                )

    final_done = completed_keys(args.out)
    if final_done != {str(row["audio"]) for row in rows}:
        missing = sorted({str(row["audio"]) for row in rows} - final_done)
        raise RuntimeError(f"incomplete Qwen3-ASR output; first missing: {missing[:5]}")
    write_metadata(
        metadata_path,
        {
            "status": "complete",
            "model": args.model,
            "revision": args.revision,
            "slice": str(args.slice),
            "audio_root": str(args.audio_root),
            "output": str(args.out),
            "selected_rows": len(rows),
            "completed_rows": len(final_done),
            "generated_rows": generated_rows,
            "language_mode": "auto",
            "wall_s": round(time.perf_counter() - started, 3),
            "finished_at_unix": time.time(),
        },
    )
    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
