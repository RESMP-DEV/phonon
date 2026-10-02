from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .curate import CuratedRow, curate_rows


def read_rows(path: Path, raw_key: str, final_key: str, id_key: str | None, limit: int) -> list[CuratedRow]:
    rows: list[CuratedRow] = []
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            value = json.loads(line)
            identifier = str(value[id_key]) if id_key else str(index)
            raw = str(value.get(raw_key, "")).strip()
            final = str(value.get(final_key, "")).strip()
            if raw and final:
                rows.append(CuratedRow(identifier, raw, final))
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    api_key = os.environ[args.api_key_env]
    rows = read_rows(args.input, args.raw_key, args.final_key, args.id_key, args.limit)
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
