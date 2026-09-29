from __future__ import annotations

import hashlib
import json
import re
import subprocess
import wave
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_info(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        duration = frames / sample_rate if sample_rate else 0.0
    return sample_rate, duration


def ffprobe_duration(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        data = json.loads(subprocess.check_output(cmd, text=True))
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None
    duration = data.get("format", {}).get("duration")
    return float(duration) if duration is not None else None


def ensure_wav_16k_mono(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".wav":
        try:
            sample_rate, _ = wav_info(src)
            with wave.open(str(src), "rb") as wav:
                channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
            if sample_rate == 16000 and channels == 1 and sampwidth == 2:
                if src.resolve() != dst.resolve():
                    dst.write_bytes(src.read_bytes())
                return
        except wave.Error:
            pass

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def audio_struct(path: Path) -> dict[str, bytes | str]:
    return {"bytes": path.read_bytes(), "path": path.name}


def youtube_segment_path(dataset_root: Path, row: dict) -> Path | None:
    source_id = row.get("source_id")
    row_id = row.get("id") or ""
    if row.get("source_kind") != "youtube" or not source_id:
        return None
    match = re.search(r"seg(\d+)$", row_id)
    if not match:
        return None
    return dataset_root / "segments" / "youtube" / source_id / f"{source_id}_seg{match.group(1)}.wav"


def local_audio_path_for_row(dataset_root: Path, row: dict) -> Path | None:
    if row.get("source_kind") == "youtube":
        segment_path = youtube_segment_path(dataset_root, row)
        if segment_path and segment_path.exists():
            return segment_path
    original = row.get("original_audio")
    if original:
        path = Path(original)
        if path.exists() and path.name.startswith(str(row.get("source_id", ""))):
            return path
    return None


def ensure_audio_struct(dataset_root: Path, row: dict) -> dict[str, bytes | str]:
    audio = row.get("audio") or {}
    if audio.get("bytes") is not None:
        return audio
    local_path = local_audio_path_for_row(dataset_root, row)
    if local_path is None:
        raise FileNotFoundError(f"cannot find local audio for row {row.get('id')}")
    return audio_struct(local_path)


def write_audio_bytes(audio: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = audio.get("bytes")
    if data is None:
        raise ValueError("audio row does not contain embedded bytes")
    out_path.write_bytes(data)
    return out_path
