from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from phonon.audio import audio_struct, ensure_audio_struct, ffprobe_duration, sha256_file, wav_info
from phonon.schema import DATASET_SCHEMA, json_dumps, now_iso, write_parquet_dataset


def fetch_youtube_audio(url: str, intake_dir: Path, source_id: str | None = None) -> dict:
    intake_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(intake_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--write-info-json",
        "-o",
        output_template,
        url,
    ]
    subprocess.run(cmd, check=True)
    info_files = sorted(intake_dir.glob("*.info.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not info_files:
        raise FileNotFoundError(f"yt-dlp did not write info json in {intake_dir}")
    info = json.loads(info_files[0].read_text())
    video_id = source_id or info.get("id") or info_files[0].stem.replace(".info", "")
    return {
        "source_id": video_id,
        "title": info.get("title", ""),
        "channel": info.get("channel", "") or info.get("uploader", ""),
        "url": url,
        "info_json": str(info_files[0]),
    }


def _download_video_audio(url: str, intake_root: Path, source_id: str | None = None) -> tuple[dict, Path]:
    meta = fetch_youtube_audio(url, intake_root, source_id)
    video_id = meta["source_id"]
    audio_candidates = sorted(intake_root.glob(f"{video_id}*.wav"))
    if not audio_candidates:
        audio_candidates = sorted(intake_root.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not audio_candidates:
        raise FileNotFoundError(f"yt-dlp did not write wav audio in {intake_root}")
    return meta, audio_candidates[0]


def _segment_audio(src_audio: Path, segments_dir: Path, video_id: str, segment_seconds: int) -> list[Path]:
    segments_dir.mkdir(parents=True, exist_ok=True)
    for old in segments_dir.glob(f"{video_id}_seg*.wav"):
        old.unlink()
    pattern = str(segments_dir / f"{video_id}_seg%05d.wav")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    return sorted(segments_dir.glob(f"{video_id}_seg*.wav"))


def _read_existing_rows(dataset_root: Path, dataset_dir: Path) -> list[dict[str, Any]]:
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    metadata_columns = [field.name for field in DATASET_SCHEMA if field.name != "audio"]
    for parquet_path in sorted(data_dir.glob("*.parquet")):
        for row in pq.read_table(parquet_path, columns=metadata_columns).to_pylist():
            row["audio"] = ensure_audio_struct(dataset_root, row)
            rows.append(row)
    return rows


def _write_source_record(dataset_root: Path, meta: dict[str, Any], url: str, dataset: str) -> None:
    sources_dir = dataset_root / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    path = sources_dir / "youtube_sources.jsonl"
    record = {
        "dataset": dataset,
        "source_kind": "youtube",
        "source_id": meta["source_id"],
        "source_url": url,
        "title": meta.get("title", ""),
        "channel": meta.get("channel", ""),
        "info_json": meta.get("info_json", ""),
        "created_at": now_iso(),
    }
    existing = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                old = json.loads(line)
                existing.add((old.get("dataset"), old.get("source_id")))
    if (dataset, meta["source_id"]) not in existing:
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stable_download_dir(dataset_root: Path, source_id: str | None) -> Path:
    return dataset_root / "intake" / "youtube" / (source_id or "downloads")


def ingest_youtube_video(
    url: str,
    dataset_root: Path,
    dataset: str = "youtube_technical_v0",
    split: str = "yt_technical_hidden_v0",
    source_id: str | None = None,
    segment_seconds: int = 30,
    max_segments: int | None = None,
    domain_tag: list[str] | None = None,
    remove_raw_audio: bool = False,
) -> dict[str, Any]:
    intake_root = _stable_download_dir(dataset_root, source_id)
    meta, raw_audio = _download_video_audio(url, intake_root, source_id)
    video_id = meta["source_id"]
    if source_id is None:
        stable_intake = dataset_root / "intake" / "youtube" / video_id
        stable_intake.mkdir(parents=True, exist_ok=True)
        if raw_audio.parent != stable_intake:
            target_audio = stable_intake / raw_audio.name
            target_audio.write_bytes(raw_audio.read_bytes())
            raw_audio = target_audio
            for info in intake_root.glob("*.info.json"):
                target_info = stable_intake / info.name
                if not target_info.exists():
                    target_info.write_bytes(info.read_bytes())
        intake_root = stable_intake

    segments_dir = dataset_root / "segments" / "youtube" / video_id
    segments = _segment_audio(raw_audio, segments_dir, video_id, segment_seconds)
    if max_segments is not None:
        segments = segments[:max_segments]

    dataset_dir = dataset_root / "datasets" / dataset
    existing_by_id = {row["id"]: row for row in _read_existing_rows(dataset_root, dataset_dir)}
    created = now_iso()
    tags = sorted(set(["technical", "youtube", *(domain_tag or [])]))
    total_duration = ffprobe_duration(raw_audio)
    new_rows: list[dict[str, Any]] = []

    for index, segment in enumerate(segments):
        sample_rate, duration = wav_info(segment)
        if duration < 1.0:
            continue
        row_id = f"youtube:{video_id}:seg{index:05d}"
        start = index * segment_seconds
        end = min(start + duration, total_duration) if total_duration else start + duration
        row = {
            "id": row_id,
            "audio": audio_struct(segment),
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "sha256_audio": sha256_file(segment),
            "source_kind": "youtube",
            "source_id": video_id,
            "source_url": url,
            "clip_start": float(start),
            "clip_end": float(end) if end is not None else None,
            "speaker_id": meta.get("channel", "") or "unknown",
            "domain_tags": tags,
            "difficulty_tags": ["needs_label", "technical_video"],
            "leakage_group": f"youtube:{video_id}",
            "split": split,
            "train_allowed": False,
            "eval_allowed": False,
            "label_status": "draft",
            "verbatim_text": "",
            "insert_text": "",
            "normalized_text": "",
            "teacher_transcripts": json_dumps({}),
            "term_spans": [],
            "notes": f"{meta.get('title', '')} | {meta.get('channel', '')}".strip(),
            "created_at": created,
            "updated_at": created,
            "seen_by_mobile_voice": False,
            "original_audio": str(raw_audio),
        }
        existing_by_id[row_id] = row
        new_rows.append(row)

    summary = write_parquet_dataset(list(existing_by_id.values()), dataset_dir, dataset)
    _write_source_record(dataset_root, meta, url, dataset)
    summary["video"] = {
        "source_id": video_id,
        "title": meta.get("title", ""),
        "channel": meta.get("channel", ""),
        "url": url,
        "raw_audio": str(raw_audio),
        "raw_audio_removed": False,
        "intake_dir": str(intake_root),
        "segments_added_or_replaced": len(new_rows),
        "segment_seconds": segment_seconds,
        "duration_seconds": total_duration,
    }
    if remove_raw_audio and raw_audio.exists():
        intake_base = (dataset_root / "intake" / "youtube").resolve()
        raw_resolved = raw_audio.resolve()
        if raw_resolved.is_relative_to(intake_base):
            raw_audio.unlink()
            summary["video"]["raw_audio_removed"] = True
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
