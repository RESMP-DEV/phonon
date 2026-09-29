from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from phonon.audio import local_audio_path_for_row
from phonon.eval import _audio_temp_files, transcribe_nemo_detailed, transcribe_whisper
from phonon.schema import json_dumps, now_iso, read_parquet_rows, write_parquet_dataset


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    adapter: str
    model: str


def parse_teacher_spec(value: str) -> TeacherSpec:
    if "=" not in value:
        raise ValueError("teacher spec must be NAME=ADAPTER:MODEL")
    name, rest = value.split("=", 1)
    if ":" not in rest:
        raise ValueError("teacher spec must be NAME=ADAPTER:MODEL")
    adapter, model = rest.split(":", 1)
    name = name.strip()
    adapter = adapter.strip()
    model = model.strip()
    if not name or not adapter or not model:
        raise ValueError("teacher spec must contain non-empty name, adapter, and model")
    if adapter not in {"nemo", "whisper"}:
        raise ValueError(f"unknown teacher adapter {adapter!r}")
    return TeacherSpec(name=name, adapter=adapter, model=model)


def default_teacher_file(dataset_root: Path, dataset: str, split: str) -> Path:
    return dataset_root / "labels" / "teachers" / f"{dataset}_{split}.jsonl"


def load_teacher_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        row_id = str(record["id"])
        if row_id not in records:
            records[row_id] = record
            continue
        merged = dict(records[row_id].get("teacher_transcripts") or {})
        merged.update(record.get("teacher_transcripts") or {})
        records[row_id] = {**records[row_id], **record, "teacher_transcripts": merged}
    return records


def write_teacher_records(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row_id in sorted(records):
            f.write(json.dumps(records[row_id], ensure_ascii=False, sort_keys=True) + "\n")


def row_teacher_transcripts(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("teacher_transcripts")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    return {}


def _read_all_dataset_rows(dataset_root: Path, dataset_dir: Path) -> list[dict[str, Any]]:
    data_dir = dataset_dir / "data"
    rows: list[dict[str, Any]] = []
    for parquet_path in sorted(data_dir.glob("*.parquet")):
        metadata_columns = [name for name in pq.read_schema(parquet_path).names if name != "audio"]
        for row in pq.read_table(parquet_path, columns=metadata_columns).to_pylist():
            local_audio = local_audio_path_for_row(dataset_root, row)
            if local_audio is None:
                raise FileNotFoundError(f"cannot find local audio for row {row.get('id')}")
            row["audio"] = {"bytes": local_audio.read_bytes(), "path": local_audio.name}
            rows.append(row)
    return rows


def _teacher_audio_files(rows: list[dict[str, Any]], dataset_root: Path, tmpdir: Path) -> list[str]:
    local_paths = [local_audio_path_for_row(dataset_root, row) for row in rows]
    if all(path is not None for path in local_paths):
        return [str(path) for path in local_paths if path is not None]
    return _audio_temp_files(rows, tmpdir)


def _select_rows(
    rows: list[dict[str, Any]],
    source_id: str | None,
    offset: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    if source_id:
        rows = [row for row in rows if row.get("source_id") == source_id]
    rows = sorted(rows, key=lambda row: (row.get("clip_start") or 0, row.get("id") or ""))
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _transcribe_teacher(
    spec: TeacherSpec,
    audio_files: list[str],
    batch_size: int,
) -> tuple[list[str], float, float]:
    records, model_load_seconds, elapsed_seconds = _transcribe_teacher_records(
        spec, audio_files, batch_size
    )
    return [str(record.get("text") or "") for record in records], model_load_seconds, elapsed_seconds


def _transcribe_teacher_records(
    spec: TeacherSpec,
    audio_files: list[str],
    batch_size: int,
) -> tuple[list[dict[str, Any]], float, float]:
    started = time.perf_counter()
    if spec.adapter == "nemo":
        records, model_load_seconds = transcribe_nemo_detailed(
            spec.model, audio_files, batch_size=batch_size
        )
    elif spec.adapter == "whisper":
        texts, model_load_seconds = transcribe_whisper(spec.model, audio_files)
        records = [{"text": text} for text in texts]
    else:
        raise ValueError(f"unknown adapter {spec.adapter!r}")
    elapsed_seconds = time.perf_counter() - started
    return records, model_load_seconds, elapsed_seconds


def _base_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "split": row["split"],
        "source_id": row.get("source_id", ""),
        "source_url": row.get("source_url", ""),
        "clip_start": row.get("clip_start"),
        "clip_end": row.get("clip_end"),
        "duration_seconds": row.get("duration_seconds"),
        "sha256_audio": row.get("sha256_audio"),
        "teacher_transcripts": {},
        "updated_at": now_iso(),
    }


def merge_teacher_sidecar_into_dataset(
    dataset_root: Path,
    dataset: str,
    teacher_file: Path,
) -> dict[str, Any]:
    dataset_dir = dataset_root / "datasets" / dataset
    teacher_records = load_teacher_records(teacher_file)
    all_rows = _read_all_dataset_rows(dataset_root, dataset_dir)
    updated = 0
    for row in all_rows:
        record = teacher_records.get(row["id"])
        if not record:
            continue
        existing = row_teacher_transcripts(row)
        merged = {**existing, **(record.get("teacher_transcripts") or {})}
        row["teacher_transcripts"] = json_dumps(merged)
        row["updated_at"] = now_iso()
        updated += 1
    summary = write_parquet_dataset(all_rows, dataset_dir, dataset)
    summary["teacher_file"] = str(teacher_file)
    summary["rows_updated_with_teachers"] = updated
    return summary


def generate_teacher_transcripts(
    dataset_root: Path,
    dataset: str,
    split: str,
    teacher_specs: list[str],
    out: Path | None = None,
    source_id: str | None = None,
    offset: int = 0,
    limit: int | None = None,
    batch_size: int = 8,
    force: bool = False,
    update_dataset: bool = False,
) -> dict[str, Any]:
    teachers = [parse_teacher_spec(spec) for spec in teacher_specs]
    teacher_names = {teacher.name for teacher in teachers}
    if len(teacher_names) != len(teachers):
        raise ValueError("teacher names must be unique")

    dataset_dir = dataset_root / "datasets" / dataset
    rows = _select_rows(
        read_parquet_rows(dataset_dir, split, include_audio=False),
        source_id,
        offset,
        limit,
    )
    if not rows:
        raise ValueError("no rows selected for teacher transcript generation")

    out = out or default_teacher_file(dataset_root, dataset, split)
    records = load_teacher_records(out)
    if not force:
        rows = [
            row
            for row in rows
            if not teacher_names.issubset(
                set((records.get(row["id"], {}).get("teacher_transcripts") or {}).keys())
            )
        ]

    if not rows:
        return {
            "dataset": dataset,
            "split": split,
            "teacher_file": str(out),
            "selected_rows": 0,
            "teachers": [teacher.name for teacher in teachers],
            "skipped_existing": True,
            "updated_dataset": False,
        }

    created_at = now_iso()
    with tempfile.TemporaryDirectory(prefix="phonon-teachers-") as tmp:
        audio_files = _teacher_audio_files(rows, dataset_root, Path(tmp))
        for teacher in teachers:
            transcript_records, model_load_seconds, elapsed_seconds = _transcribe_teacher_records(
                teacher, audio_files, batch_size
            )
            for row, transcript_record in zip(rows, transcript_records, strict=True):
                record = records.setdefault(row["id"], _base_record(row))
                transcripts = record.setdefault("teacher_transcripts", {})
                transcripts[teacher.name] = {
                    "adapter": teacher.adapter,
                    "model": teacher.model,
                    **transcript_record,
                    "created_at": created_at,
                    "model_load_seconds": model_load_seconds,
                    "run_elapsed_seconds": elapsed_seconds,
                }
                record["updated_at"] = now_iso()
            write_teacher_records(out, records)

    write_teacher_records(out, records)
    dataset_summary: dict[str, Any] | None = None
    if update_dataset:
        dataset_summary = merge_teacher_sidecar_into_dataset(dataset_root, dataset, out)

    return {
        "dataset": dataset,
        "split": split,
        "teacher_file": str(out),
        "selected_rows": len(rows),
        "teachers": [teacher.name for teacher in teachers],
        "updated_dataset": update_dataset,
        "dataset_summary": dataset_summary,
    }
