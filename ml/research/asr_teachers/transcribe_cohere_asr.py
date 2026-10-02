#!/usr/bin/env python3
"""Run offline Cohere Transcribe over a frozen Phonon audio slice."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_REVISION = "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"


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
        if str(row.get("hyp", "")).strip():
            keys.add(str(row["audio"]))
    return keys


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    def serialize_set(value: Any) -> list[str]:
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"unsupported metadata value: {type(value).__name__}")

    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=serialize_set) + "\n"
    )


def decode_transcription(value: str | list[str]) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            raise RuntimeError(f"expected one transcription, got {len(value)}")
        value = value[0]
    return value.strip()


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


def metadata(
    *,
    status: str,
    args: argparse.Namespace,
    selected_rows: int,
    completed: set[str],
    started_at_unix: float,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "model": args.model,
        "revision": args.revision,
        "slice": str(args.slice),
        "audio_root": str(args.audio_root),
        "output": str(args.out),
        "selected_rows": selected_rows,
        "completed_rows": completed,
        "language": "en",
        "started_at_unix": started_at_unix,
        **extra,
    }


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
    started_at_unix = time.time()
    write_metadata(
        metadata_path,
        metadata(
            status="running",
            args=args,
            selected_rows=len(rows),
            completed=done,
            started_at_unix=started_at_unix,
        ),
    )

    import librosa
    import torch
    import transformers

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Cohere Transcribe teacher requires CUDA")
    device = torch.device(args.device)
    processor = transformers.AutoProcessor.from_pretrained(
        args.model, revision=args.revision
    )
    model = transformers.CohereAsrForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map=args.device,
    ).eval()

    generated = 0
    started = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as sink:
        for index, row in enumerate(rows):
            key = str(row["audio"])
            if key in done:
                continue
            path = args.audio_root / key
            if not path.is_file():
                raise FileNotFoundError(path)
            wave, _sample_rate = librosa.load(
                path, sr=16000, mono=True, dtype="float32"
            )
            request_started = time.perf_counter()
            with torch.inference_mode():
                inputs = processor(
                    [wave],
                    sampling_rate=16000,
                    return_tensors="pt",
                    language="en",
                )
                chunk_index = inputs.get("audio_chunk_index")
                inputs.to(device=device, dtype=model.dtype)
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                hyp = decode_transcription(
                    processor.decode(
                        output,
                        skip_special_tokens=True,
                        audio_chunk_index=chunk_index,
                        language="en",
                    )
                )
            elapsed = time.perf_counter() - request_started
            result = {
                "audio": key,
                "ref": row.get("ref", ""),
                "raw_aqua": row.get("raw", row.get("raw_aqua", "")),
                "hyp": hyp,
                "dur": row.get("dur", 0),
                "gen_s": round(elapsed, 3),
                "model": args.model,
                "revision": args.revision,
                "language": "en",
            }
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            generated += 1
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(
                    f"[{index + 1}/{len(rows)}] generated={generated} "
                    f"last={elapsed:.2f}s elapsed={time.perf_counter() - started:.0f}s",
                    flush=True,
                )
                write_metadata(
                    metadata_path,
                    metadata(
                        status="running",
                        args=args,
                        selected_rows=len(rows),
                        completed=completed_keys(args.out),
                        started_at_unix=started_at_unix,
                    ),
                )

    expected = {str(row["audio"]) for row in rows}
    final_done = completed_keys(args.out)
    if final_done != expected:
        missing = sorted(expected - final_done)
        raise RuntimeError(f"incomplete Cohere output; first missing: {missing[:5]}")
    write_metadata(
        metadata_path,
        metadata(
            status="complete",
            args=args,
            selected_rows=len(rows),
            completed=final_done,
            started_at_unix=started_at_unix,
            generated_rows=generated,
            wall_s=round(time.perf_counter() - started, 3),
            finished_at_unix=time.time(),
        ),
    )
    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
