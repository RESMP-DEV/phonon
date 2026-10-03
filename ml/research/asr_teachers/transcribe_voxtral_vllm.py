#!/usr/bin/env python3
"""Run a local vLLM Voxtral teacher over a frozen Phonon audio slice."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def read_slice(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty audio slice: {path}")
    return rows


def completed_keys(output: Path) -> set[str]:
    if not output.exists():
        return set()
    keys: set[str] = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("hyp", "")).strip():
            keys.add(str(row["audio"]))
    return keys


def validate_local_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Voxtral vLLM transcription URL must be HTTP on a loopback host")


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=sorted) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8302/v1/audio/transcriptions")
    parser.add_argument("--language", default="en")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_local_url(args.url)
    rows = read_slice(args.slice)
    if args.limit:
        rows = rows[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.force and args.out.exists():
        args.out.unlink()
    done = completed_keys(args.out)
    metadata_path = args.out.with_suffix(".meta.json")
    base_metadata = {
        "status": "running",
        "model": args.model,
        "url": args.url,
        "language": args.language,
        "slice": str(args.slice),
        "audio_root": str(args.audio_root),
        "output": str(args.out),
        "selected_rows": len(rows),
        "started_at_unix": time.time(),
    }
    write_metadata(metadata_path, base_metadata)

    import requests

    generated = 0
    started = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as sink:
        for index, source in enumerate(rows):
            audio_key = str(source["audio"])
            if audio_key in done:
                continue
            audio_path = args.audio_root / audio_key
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            request_started = time.perf_counter()
            with audio_path.open("rb") as audio:
                response = requests.post(
                    args.url,
                    data={"model": args.model, "language": args.language},
                    files={"file": (audio_path.name, audio, "audio/flac")},
                    timeout=1800,
                )
            response.raise_for_status()
            payload = response.json()
            hyp = str(payload.get("text", "")).strip()
            if not hyp:
                raise RuntimeError(f"empty transcription response for {audio_key}")
            elapsed = time.perf_counter() - request_started
            result = {
                "audio": audio_key,
                "ref": source.get("ref", ""),
                "raw_aqua": source.get("raw", source.get("raw_aqua", "")),
                "hyp": hyp,
                "dur": source.get("dur", source.get("duration", 0)),
                "usage": payload.get("usage", {}),
                "gen_s": round(elapsed, 3),
                "model": args.model,
                "provider": "local-vllm",
            }
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            generated += 1
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                write_metadata(
                    metadata_path,
                    {
                        **base_metadata,
                        "completed_rows": len(completed_keys(args.out)),
                        "generated_rows": generated,
                    },
                )

    final_done = completed_keys(args.out)
    expected = {str(row["audio"]) for row in rows}
    if final_done != expected:
        missing = sorted(expected - final_done)
        raise RuntimeError(f"incomplete Voxtral output; first missing: {missing[:5]}")
    write_metadata(
        metadata_path,
        {
            **base_metadata,
            "status": "complete",
            "completed_rows": len(final_done),
            "generated_rows": generated,
            "wall_s": round(time.perf_counter() - started, 3),
            "finished_at_unix": time.time(),
        },
    )
    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
