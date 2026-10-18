#!/usr/bin/env python3
"""Derive optional voice-isolation sidecars without changing raw Phonon audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RNNN_MODEL_SHAS = {
    "mp.rnnn": "4e84a448a4baf937992aaf4d10c8258007ec5d24219b6647dfd5fb4b563ad231",
    "bd.rnnn": "ae3f7411e1e6a884f839a4a145c394408398f09854dbc1216ee02faafc98a17b",
}
VARIANTS = ("rnnoise-mp", "rnnoise-bd", "dfn-pf", "dfn-pf-atten6")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_parameters(path: Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as wav:
        return {
            "sample_rate": wav.getframerate(),
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "duration_seconds": round(wav.getnframes() / wav.getframerate(), 6),
        }


def validate_source(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing source WAV: {path}")
    params = wav_parameters(path)
    if params["channels"] != 1:
        raise ValueError(f"expected mono source, got {params['channels']} channels")
    if params["sample_width_bytes"] != 2:
        raise ValueError("expected PCM16 source WAV")
    if params["sample_rate"] != 48000:
        raise ValueError(
            f"expected native 48 kHz source, got {params['sample_rate']} Hz"
        )


def run_checked(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        tail = completed.stderr.strip()[-1000:]
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command[0]}\n{tail}"
        )


def derive_rnnoise(
    source: Path,
    output: Path,
    *,
    model: Path,
    ffmpeg: str,
) -> list[str]:
    if not model.is_file():
        raise ValueError(f"missing RNNoise model: {model}")
    expected_sha = RNNN_MODEL_SHAS.get(model.name)
    if expected_sha is None:
        raise ValueError(f"unregistered RNNoise model: {model.name}")
    actual_sha = sha256(model)
    if actual_sha != expected_sha:
        raise ValueError(f"RNNoise model hash mismatch for {model.name}: {actual_sha}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-af",
        f"arnndn=model={model}",
        "-ar",
        "48000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(output),
    ]
    run_checked(command)
    return command


def derive_deep_filter(
    source: Path,
    output: Path,
    *,
    binary: str,
    attenuation_limit_db: int | None,
) -> list[str]:
    if not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phonon-dfn-") as temporary:
        temporary_dir = Path(temporary)
        command = [str(binary)]
        if attenuation_limit_db is None:
            command.append("--pf")
        else:
            command.extend(
                ["--pf", "-a", str(attenuation_db_command_value(attenuation_limit_db))]
            )
        command.extend(["-o", str(temporary_dir), str(source)])
        run_checked(command)
        produced = temporary_dir / source.name
        if not produced.is_file():
            raise RuntimeError(f"DeepFilterNet did not produce {produced.name}")
        shutil.copyfile(produced, output)
    return command


def attenuation_db_command_value(value: int) -> str:
    if value <= 0:
        raise ValueError("DeepFilterNet attenuation limit must be positive")
    return str(value)


def derive_variant(
    source: Path,
    variant: str,
    output_root: Path,
    *,
    ffmpeg: str,
    deep_filter: str,
    rnnoise_root: Path,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    output = output_root / f"{source.stem}-{variant}.wav"
    if variant == "rnnoise-mp":
        model = rnnoise_root / "mp.rnnn"
        command = derive_rnnoise(source, output, model=model, ffmpeg=ffmpeg)
    elif variant == "rnnoise-bd":
        model = rnnoise_root / "bd.rnnn"
        command = derive_rnnoise(source, output, model=model, ffmpeg=ffmpeg)
    elif variant == "dfn-pf":
        command = derive_deep_filter(
            source, output, binary=deep_filter, attenuation_limit_db=None
        )
    elif variant == "dfn-pf-atten6":
        command = derive_deep_filter(
            source, output, binary=deep_filter, attenuation_limit_db=6
        )
    else:  # pragma: no cover - guarded above
        raise AssertionError(variant)
    return {
        "variant": variant,
        "output": str(output),
        "source_sha256": sha256(source),
        "output_sha256": sha256(output),
        "parameters": wav_parameters(output),
        "command": command,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="raw mono PCM16 48 kHz WAV")
    parser.add_argument(
        "--out", type=Path, default=None, help="derived-sidecar directory"
    )
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--deep-filter",
        type=Path,
        default=Path.home() / ".local/share/phonon-research/bin/deep-filter",
    )
    parser.add_argument(
        "--rnnoise-root",
        type=Path,
        default=Path.home() / ".local/share/phonon-research/models/arnndn",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output_root = (
        args.out.expanduser() if args.out else source.parent / "voice-isolation"
    ).resolve()
    validate_source(source)
    source_hash_before = sha256(source)
    variants = args.variant or ["rnnoise-mp"]
    output_root.mkdir(parents=True, exist_ok=True)
    derived = [
        derive_variant(
            source,
            variant,
            output_root,
            ffmpeg=args.ffmpeg,
            deep_filter=args.deep_filter,
            rnnoise_root=args.rnnoise_root.expanduser().resolve(),
        )
        for variant in variants
    ]
    if sha256(source) != source_hash_before:
        raise RuntimeError("source WAV changed during derivation")
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "source_sha256": source_hash_before,
        "raw_is_canonical": True,
        "variants": derived,
    }
    receipt = output_root / "metadata.json"
    receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, wave.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
