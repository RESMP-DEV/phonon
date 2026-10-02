from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .curate import CuratedRow, curate_rows


def read_teacher_map(path: Path, id_key: str, text_key: str) -> dict[str, str]:
    teachers: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            value = json.loads(line)
            identifier = str(value[id_key])
            text = str(value.get(text_key, "")).strip()
            if identifier in teachers:
                raise ValueError(f"duplicate teacher row id: {identifier}")
            if text:
                teachers[identifier] = text
    return teachers


def read_rows(
    path: Path,
    raw_key: str,
    final_key: str,
    id_key: str | None,
    limit: int,
    teacher_maps: dict[str, dict[str, str]] | None = None,
    omit_final: bool = False,
) -> list[CuratedRow]:
    rows: list[CuratedRow] = []
    teacher_maps = teacher_maps or {}
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            value = json.loads(line)
            identifier = str(value[id_key]) if id_key else str(index)
            raw = str(value.get(raw_key, "")).strip()
            final = str(value.get(final_key, "")).strip()
            if raw and (final or omit_final):
                teachers = {
                    name: transcripts[identifier]
                    for name, transcripts in teacher_maps.items()
                    if identifier in transcripts
                }
                rows.append(
                    CuratedRow(identifier, raw, "" if omit_final else final, teachers)
                )
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate dictation corrections with an API teacher")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("PHONON_CURATION_ENDPOINT", ""))
    parser.add_argument("--api-key-env", default="PHONON_CURATION_API_KEY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--raw-key", default="raw")
    parser.add_argument("--final-key", default="corrected")
    parser.add_argument("--id-key")
    parser.add_argument(
        "--teacher",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Named ASR-hypothesis JSONL; repeatable. Its ID field defaults to audio.",
    )
    parser.add_argument("--teacher-id-key", default="audio")
    parser.add_argument("--teacher-text-key", default="hyp")
    parser.add_argument(
        "--omit-final",
        action="store_true",
        help="Do not send the accepted final text; reconcile from raw and named ASR hypotheses only",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    api_key = os.environ[args.api_key_env]
    teacher_maps: dict[str, dict[str, str]] = {}
    for spec in args.teacher:
        try:
            name, path = spec.split("=", 1)
        except ValueError as error:
            raise SystemExit("--teacher requires NAME=PATH") from error
        name = name.strip()
        if not name:
            raise SystemExit("teacher name must not be empty")
        if name in teacher_maps:
            raise SystemExit(f"duplicate teacher name: {name}")
        teacher_maps[name] = read_teacher_map(
            Path(path), args.teacher_id_key, args.teacher_text_key
        )
    rows = read_rows(
        args.input,
        args.raw_key,
        args.final_key,
        args.id_key,
        args.limit,
        teacher_maps,
        args.omit_final,
    )
    path = curate_rows(
        rows,
        args.output,
        resume=not args.no_resume,
        endpoint=endpoint,
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
    )
    print(path)


if __name__ == "__main__":
    main()
