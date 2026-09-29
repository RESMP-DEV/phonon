from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from phonon.audio import audio_struct, ensure_wav_16k_mono, sha256_file, wav_info
from phonon.schema import json_dumps, now_iso, write_parquet_dataset


TECH_TOKEN_RE = re.compile(r"([A-Z]{2,}|[A-Za-z]+[A-Z][A-Za-z]*|\d|[./_+#-])")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def profile_terms(export_dir: Path) -> list[str]:
    path = export_dir / "aqua_profile_context.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return sorted({str(term) for term in data.get("dictionary", []) if str(term).strip()}, key=len, reverse=True)


def term_hits(text: str, terms: list[str]) -> list[str]:
    hits: list[str] = []
    lowered = text.lower()
    for term in terms:
        if not term:
            continue
        if re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", lowered):
            hits.append(term)
    return sorted(set(hits), key=str.lower)


def difficulty_tags(text: str, terms: list[str]) -> list[str]:
    tags: set[str] = set()
    if term_hits(text, terms):
        tags.add("profile_term")
    if re.search(r"\b[A-Z]{2,}\b", text):
        tags.add("acronym")
    if re.search(r"\b[A-Za-z]+[0-9]+[A-Za-z0-9]*\b", text):
        tags.add("alphanumeric")
    if re.search(r"[/_.-][A-Za-z0-9]", text):
        tags.add("path_or_symbol")
    if re.search(r"\b(?:uv|ruff|pytest|git|ssh|nvidia-smi|CUDA|Triton|PyTorch|OpenAI|Claude|Codex)\b", text):
        tags.add("developer_term")
    if len(text.split()) > 80:
        tags.add("long_form")
    return sorted(tags)


def domain_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = {"developer_dictation"}
    if any(term in lowered for term in ("cuda", "gpu", "triton", "pytorch", "kernel")):
        tags.add("cuda")
    if any(term in lowered for term in ("openai", "claude", "codex", "llm", "asr", "whisper")):
        tags.add("llm")
    if any(term in lowered for term in ("github", "pr", "commit", "repo")):
        tags.add("repo")
    return sorted(tags)


def split_for_row(row: dict[str, Any], audio_sha: str, holdout_mod: int) -> str:
    source = row.get("source") or ""
    if source.startswith("old-export-2026-05-15"):
        return "aqua_seen"
    bucket = int(audio_sha[:8], 16) % 100
    return "aqua_new_holdout" if bucket < holdout_mod else "aqua_new_dev"


def ingest_aqua(
    export_dir: Path,
    dataset_root: Path,
    out_name: str,
    holdout_percent: int = 20,
) -> dict[str, Any]:
    utterances_path = export_dir / "aqua_utterances.jsonl"
    if not utterances_path.exists():
        raise FileNotFoundError(utterances_path)
    out_dir = dataset_root / "datasets" / out_name
    clean_audio_dir = out_dir / "clean_audio"
    rows = read_jsonl(utterances_path)
    terms = profile_terms(export_dir)

    canonical_rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    created = now_iso()
    source_id = export_dir.name

    for row in rows:
        raw_text = (row.get("raw_text") or "").strip()
        normalized_text = (row.get("normalized_text") or raw_text).strip()
        audio_rel = row.get("audio") or ""
        src_audio = export_dir / audio_rel
        if row.get("audio_present") is False or not src_audio.exists():
            rejects.append({"id": row.get("id"), "reason": "missing_audio", "audio": audio_rel})
            continue
        if not raw_text and not normalized_text:
            rejects.append({"id": row.get("id"), "reason": "empty_label", "audio": audio_rel})
            continue

        clean_path = clean_audio_dir / f"{row.get('id')}.wav"
        ensure_wav_16k_mono(src_audio, clean_path)
        sample_rate, duration = wav_info(clean_path)
        audio_sha = sha256_file(clean_path)
        split = split_for_row(row, audio_sha, holdout_percent)
        text_for_tags = normalized_text or raw_text
        seen = split == "aqua_seen"
        train_allowed = split == "aqua_new_dev"
        eval_allowed = split in {"aqua_seen", "aqua_new_dev", "aqua_new_holdout"}

        canonical_rows.append(
            {
                "id": f"aqua:{row.get('id')}",
                "audio": audio_struct(clean_path),
                "sample_rate": sample_rate,
                "duration_seconds": float(row.get("duration_seconds") or duration),
                "sha256_audio": audio_sha,
                "source_kind": "aqua",
                "source_id": source_id,
                "source_url": "",
                "clip_start": None,
                "clip_end": None,
                "speaker_id": "elliot",
                "domain_tags": domain_tags(text_for_tags),
                "difficulty_tags": difficulty_tags(text_for_tags, terms),
                "leakage_group": f"aqua:{row.get('sessionId') or row.get('source') or source_id}",
                "split": split,
                "train_allowed": train_allowed,
                "eval_allowed": eval_allowed,
                "label_status": "model_assisted",
                "verbatim_text": raw_text,
                "insert_text": normalized_text,
                "normalized_text": normalized_text,
                "teacher_transcripts": json_dumps(
                    {"aqua_raw": raw_text, "aqua_normalized": normalized_text}
                ),
                "term_spans": term_hits(text_for_tags, terms),
                "notes": "Imported from local Aqua Voice training export.",
                "created_at": created,
                "updated_at": created,
                "seen_by_mobile_voice": seen,
                "original_audio": str(src_audio),
            }
        )

    summary = write_parquet_dataset(canonical_rows, out_dir, out_name)
    summary["rejects"] = len(rejects)
    summary["reject_reasons"] = {}
    for reject in rejects:
        summary["reject_reasons"][reject["reason"]] = summary["reject_reasons"].get(
            reject["reason"], 0
        ) + 1
    (out_dir / "rejects.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejects)
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
