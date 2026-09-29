from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

from jiwer import wer

from phonon.audio import local_audio_path_for_row
from phonon.schema import read_parquet_rows
from phonon.teachers import load_teacher_records, row_teacher_transcripts


TECHNICAL_RE = re.compile(
    r"\b("
    r"CUDA|cuDNN|NVIDIA|GPU|GPUs|H100|Blackwell|Triton|PyTorch|torch|kernel|kernels|"
    r"warp|warps|SM\d+|tensor cores?|FP8|BF16|NVLink|NVCC|LLVM|compiler|compilers|"
    r"LLM|LLMs|transformer|attention|embedding|tokenizer|logits?|RLHF|PPO|LoRA|"
    r"OpenAI|Claude|Gemini|Llama|Qwen|Python|NumPy|JAX|Docker|Kubernetes|Linux"
    r")\b",
    re.IGNORECASE,
)
COMMAND_RE = re.compile(
    r"(`[^`]+`|/[\w./-]+|[\w.-]+\.(?:py|cu|cpp|cuh|h|hpp|rs|js|ts|json|toml|yaml|yml)|"
    r"\b(?:uv|pip|git|cmake|make|ninja|nvcc|pytest|python|docker|ssh|scp|rsync|npm|pnpm)\b)",
    re.IGNORECASE,
)


def _parse_teacher_texts(row: dict[str, Any], sidecar: dict[str, dict[str, Any]]) -> dict[str, str]:
    merged = row_teacher_transcripts(row)
    record = sidecar.get(row["id"])
    if record:
        merged = {**merged, **(record.get("teacher_transcripts") or {})}
    texts: dict[str, str] = {}
    for name, value in merged.items():
        if isinstance(value, dict):
            text = str(value.get("text") or "").strip()
        else:
            text = str(value or "").strip()
        if text:
            texts[str(name)] = text
    return texts


def pairwise_disagreement(texts: list[str]) -> float:
    usable = [text for text in texts if text.strip()]
    if len(usable) < 2:
        return 0.0
    scores = []
    for left, right in itertools.combinations(usable, 2):
        scores.append(min(float(wer(left, right)), 3.0))
    return sum(scores) / len(scores)


def review_score(row: dict[str, Any], teacher_texts: dict[str, str]) -> dict[str, Any]:
    combined_text = " ".join(teacher_texts.values())
    technical_terms = TECHNICAL_RE.findall(combined_text)
    command_hits = COMMAND_RE.findall(combined_text)
    disagreement = pairwise_disagreement(list(teacher_texts.values()))
    duration = float(row.get("duration_seconds") or 0)
    duration_quality = min(duration / 30.0, 1.0)
    score = disagreement * 10.0 + len(technical_terms) * 0.35 + len(command_hits) * 0.8
    score += duration_quality * 0.1
    return {
        "review_score": round(score, 4),
        "teacher_disagreement": round(disagreement, 4),
        "technical_term_hits": len(technical_terms),
        "command_path_package_hits": len(command_hits),
        "duration_quality": round(duration_quality, 4),
    }


def make_label_queue(
    dataset_root: Path,
    dataset: str,
    split: str,
    out: Path | None = None,
    empty: bool = False,
    teacher_file: Path | None = None,
    scored: bool = False,
    limit: int | None = None,
    source_id: str | None = None,
    only_with_teachers: bool = False,
) -> dict[str, Any]:
    if out is None:
        out = dataset_root / "labels" / "queues" / f"{dataset}_{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    sidecar = load_teacher_records(teacher_file) if teacher_file else {}
    if not empty:
        rows = read_parquet_rows(dataset_root / "datasets" / dataset, split, include_audio=False)
        if source_id:
            rows = [row for row in rows if row.get("source_id") == source_id]
        if only_with_teachers:
            rows = [row for row in rows if _parse_teacher_texts(row, sidecar)]
        if scored:
            rows.sort(
                key=lambda row: (
                    review_score(row, _parse_teacher_texts(row, sidecar))["review_score"],
                    row.get("duration_seconds") or 0,
                    row.get("id") or "",
                ),
                reverse=True,
            )
        else:
            rows.sort(
                key=lambda row: (
                    len(row.get("term_spans") or []),
                    row.get("duration_seconds") or 0,
                    row.get("id") or "",
                ),
                reverse=True,
            )
        if limit is not None:
            rows = rows[:limit]

    with out.open("w") as f:
        for row in rows:
            teacher_texts = _parse_teacher_texts(row, sidecar)
            audio_path = local_audio_path_for_row(dataset_root, row)
            item = {
                "id": row["id"],
                "split": row["split"],
                "source_id": row.get("source_id", ""),
                "source_url": row.get("source_url", ""),
                "clip_start": row.get("clip_start"),
                "clip_end": row.get("clip_end"),
                "audio_path": str(audio_path) if audio_path else "",
                "sha256_audio": row["sha256_audio"],
                "duration_seconds": row["duration_seconds"],
                "label_status": row["label_status"],
                "verbatim_text": row["verbatim_text"],
                "insert_text": row["insert_text"],
                "normalized_text": row["normalized_text"],
                "teacher_transcripts": teacher_texts,
                "term_spans": row.get("term_spans") or [],
                "difficulty_tags": row.get("difficulty_tags") or [],
                "review_verdict": "pending",
                "review_notes": "",
            }
            if scored:
                item.update(review_score(row, teacher_texts))
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    meta = {
        "queue": str(out),
        "dataset": dataset,
        "split": split,
        "rows": len(rows),
        "empty_template": empty,
        "teacher_file": str(teacher_file) if teacher_file else None,
        "scored": scored,
    }
    out.with_suffix(".summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
