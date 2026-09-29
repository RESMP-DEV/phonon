from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CANONICAL_COLUMNS = [
    "id",
    "audio",
    "sample_rate",
    "duration_seconds",
    "sha256_audio",
    "source_kind",
    "source_id",
    "source_url",
    "clip_start",
    "clip_end",
    "speaker_id",
    "domain_tags",
    "difficulty_tags",
    "leakage_group",
    "split",
    "train_allowed",
    "eval_allowed",
    "label_status",
    "verbatim_text",
    "insert_text",
    "normalized_text",
    "teacher_transcripts",
    "term_spans",
    "notes",
    "created_at",
    "updated_at",
    "seen_by_mobile_voice",
    "original_audio",
]


DATASET_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("audio", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
        pa.field("sample_rate", pa.int32()),
        pa.field("duration_seconds", pa.float64()),
        pa.field("sha256_audio", pa.string()),
        pa.field("source_kind", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("source_url", pa.string()),
        pa.field("clip_start", pa.float64()),
        pa.field("clip_end", pa.float64()),
        pa.field("speaker_id", pa.string()),
        pa.field("domain_tags", pa.list_(pa.string())),
        pa.field("difficulty_tags", pa.list_(pa.string())),
        pa.field("leakage_group", pa.string()),
        pa.field("split", pa.string()),
        pa.field("train_allowed", pa.bool_()),
        pa.field("eval_allowed", pa.bool_()),
        pa.field("label_status", pa.string()),
        pa.field("verbatim_text", pa.string()),
        pa.field("insert_text", pa.string()),
        pa.field("normalized_text", pa.string()),
        pa.field("teacher_transcripts", pa.string()),
        pa.field("term_spans", pa.list_(pa.string())),
        pa.field("notes", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("updated_at", pa.string()),
        pa.field("seen_by_mobile_voice", pa.bool_()),
        pa.field("original_audio", pa.string()),
    ]
)

DEFAULT_PARQUET_SHARD_ROWS = 1000


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {column: row.get(column) for column in CANONICAL_COLUMNS}
    out.setdefault("source_url", "")
    out.setdefault("clip_start", None)
    out.setdefault("clip_end", None)
    out.setdefault("speaker_id", "")
    out.setdefault("domain_tags", [])
    out.setdefault("difficulty_tags", [])
    out.setdefault("teacher_transcripts", "{}")
    out.setdefault("term_spans", [])
    out.setdefault("notes", "")
    out.setdefault("created_at", now_iso())
    out.setdefault("updated_at", out["created_at"])
    out.setdefault("seen_by_mobile_voice", False)
    out.setdefault("original_audio", "")
    for column in ("domain_tags", "difficulty_tags", "term_spans"):
        if out[column] is None:
            out[column] = []
    return out


def write_parquet_dataset(
    rows: list[dict[str, Any]],
    out_dir: Path,
    dataset_name: str,
    shard_rows: int = DEFAULT_PARQUET_SHARD_ROWS,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        norm = normalize_row(row)
        by_split.setdefault(norm["split"], []).append(norm)

    summary: dict[str, Any] = {"dataset": dataset_name, "splits": {}, "rows": len(rows)}
    for split, split_rows in sorted(by_split.items()):
        shard_count = max(1, (len(split_rows) + shard_rows - 1) // shard_rows)
        old_paths = sorted(data_dir.glob(f"{split}-*.parquet"))
        final_paths: list[Path] = []
        pending_replacements: list[tuple[Path, Path]] = []
        for shard_index in range(shard_count):
            start = shard_index * shard_rows
            shard = split_rows[start : start + shard_rows]
            table = pa.Table.from_pylist(shard, schema=DATASET_SCHEMA)
            path = data_dir / f"{split}-{shard_index:05d}-of-{shard_count:05d}.parquet"
            tmp_path = data_dir / f".{path.name}.{os.getpid()}.tmp"
            pq.write_table(table, tmp_path, compression="zstd")
            pending_replacements.append((tmp_path, path))
            final_paths.append(path)

        for tmp_path, final_path in pending_replacements:
            tmp_path.replace(final_path)
        for old_path in old_paths:
            if old_path not in final_paths and old_path.exists():
                old_path.unlink()

        parquet_files = [str(path) for path in final_paths]
        summary["splits"][split] = {
            "rows": len(split_rows),
            "duration_seconds": round(sum(float(r["duration_seconds"]) for r in split_rows), 3),
            "parquet": parquet_files[0],
            "parquet_files": parquet_files,
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_dataset_readme(out_dir, dataset_name, sorted(by_split))
    return summary


def write_dataset_readme(out_dir: Path, dataset_name: str, splits: list[str]) -> None:
    split_yaml = "\n".join(
        f"  - split: {split}\n    path: data/{split}-*.parquet" for split in splits
    )
    readme = f"""---
configs:
- config_name: default
  data_files:
{split_yaml}
---

# {dataset_name}

Canonical Phonon ASR dataset. Audio is embedded in Parquet as a Hugging
Face-compatible `audio` struct with `bytes` and `path`.

Hidden eval rows must not be used for training or synthetic prompt generation.
"""
    (out_dir / "README.md").write_text(readme)


def read_parquet_rows(dataset_dir: Path, split: str, include_audio: bool = True) -> list[dict[str, Any]]:
    data_dir = dataset_dir / "data"
    paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards for split {split!r} in {data_dir}")
    rows: list[dict[str, Any]] = []
    columns = None if include_audio else [column for column in CANONICAL_COLUMNS if column != "audio"]
    for path in paths:
        read_rows = pq.read_table(path, columns=columns).to_pylist()
        if not include_audio:
            for row in read_rows:
                row["audio"] = {"bytes": None, "path": ""}
        rows.extend(read_rows)
    return rows
