from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phonon.ingest_youtube import ingest_youtube_video
from phonon.schema import now_iso


@dataclass(frozen=True)
class YoutubeCandidate:
    channel_group: str
    rank: int
    source_id: str
    url: str
    title: str
    duration_seconds: float | None
    content_category: str
    label_style: str


def load_youtube_candidates(path: Path, groups: list[str] | None = None) -> list[YoutubeCandidate]:
    selected_groups = set(groups or [])
    candidates: list[YoutubeCandidate] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        channel_group = str(record.get("channel_group", "")).strip()
        if selected_groups and channel_group not in selected_groups:
            continue
        source_id = str(record.get("id", "")).strip()
        url = str(record.get("url", "")).strip()
        if not source_id or not url:
            raise ValueError(f"candidate {path}:{line_number} needs non-empty id and url")
        candidates.append(
            YoutubeCandidate(
                channel_group=channel_group,
                rank=int(record.get("rank") or 0),
                source_id=source_id,
                url=url,
                title=str(record.get("title", "")),
                duration_seconds=record.get("duration_seconds"),
                content_category=str(record.get("content_category") or "technical_video"),
                label_style=str(record.get("label_style") or ""),
            )
        )
    return candidates


def existing_youtube_source_ids(dataset_root: Path, dataset: str) -> set[str]:
    path = dataset_root / "sources" / "youtube_sources.jsonl"
    if not path.exists():
        return set()
    source_ids: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("dataset") == dataset and record.get("source_id"):
            source_ids.add(str(record["source_id"]))
    return source_ids


def candidate_domain_tags(candidate: YoutubeCandidate) -> list[str]:
    tags = ["hard_eval", "technical_video", candidate.channel_group, candidate.content_category]
    if candidate.label_style:
        tags.append(f"style_{candidate.label_style}")
    if candidate.channel_group == "elliot":
        tags.append("owned_channel")
    return sorted(set(tags))


def append_progress(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"created_at": now_iso(), **record}, ensure_ascii=False) + "\n")


def ingest_youtube_candidates(
    candidate_file: Path,
    dataset_root: Path,
    dataset: str,
    split: str,
    segment_seconds: int,
    groups: list[str] | None = None,
    limit: int | None = None,
    progress_log: Path | None = None,
    skip_existing: bool = True,
    stop_on_error: bool = False,
    remove_raw_audio: bool = False,
) -> dict[str, Any]:
    candidates = load_youtube_candidates(candidate_file, groups)
    if limit is not None:
        candidates = candidates[:limit]
    progress_log = progress_log or dataset_root / "logs" / "youtube_candidate_ingest.progress.jsonl"
    existing = existing_youtube_source_ids(dataset_root, dataset) if skip_existing else set()
    results: list[dict[str, Any]] = []

    for candidate in candidates:
        base = {
            "dataset": dataset,
            "split": split,
            "source_id": candidate.source_id,
            "channel_group": candidate.channel_group,
            "title": candidate.title,
            "url": candidate.url,
        }
        if candidate.source_id in existing:
            append_progress(progress_log, {**base, "status": "skipped_existing"})
            results.append({**base, "status": "skipped_existing"})
            continue
        append_progress(progress_log, {**base, "status": "started"})
        try:
            summary = ingest_youtube_video(
                candidate.url,
                dataset_root,
                dataset=dataset,
                split=split,
                source_id=candidate.source_id,
                segment_seconds=segment_seconds,
                domain_tag=candidate_domain_tags(candidate),
                remove_raw_audio=remove_raw_audio,
            )
        except Exception as exc:  # noqa: BLE001
            error = {**base, "status": "error", "error": repr(exc)}
            append_progress(progress_log, error)
            results.append(error)
            if stop_on_error:
                raise
            continue
        existing.add(candidate.source_id)
        done = {
            **base,
            "status": "done",
            "segments": summary.get("video", {}).get("segments_added_or_replaced"),
            "duration_seconds": summary.get("video", {}).get("duration_seconds"),
        }
        append_progress(progress_log, done)
        results.append(done)

    return {
        "candidate_file": str(candidate_file),
        "dataset": dataset,
        "split": split,
        "progress_log": str(progress_log),
        "rows": len(results),
        "done": sum(1 for result in results if result["status"] == "done"),
        "skipped": sum(1 for result in results if result["status"] == "skipped_existing"),
        "errors": sum(1 for result in results if result["status"] == "error"),
        "results": results,
    }
