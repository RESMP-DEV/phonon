#!/usr/bin/env python3
"""Resolve likely-hardware ASR terms using Exa retrieval only.

This tool never calls ZCode, an LLM route, or any personal-data source. It sends
only the caller-supplied technical phrase to native `exq` (Exa search) and
returns a reviewable JSON evidence file the owner can approve into the dictionary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_LIMIT = 8
OFFICIAL_DOMAIN_PRIORITY = (
    "nvidia.com",
    "nvidianews.nvidia.com",
    "resources.nvidia.com",
    "amd.com",
    "intel.com",
    "apple.com",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phrase", action="append", required=True, help="repeatable technical phrase"
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exq", default="exq")
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def candidate_from_result(result: dict[str, Any]) -> str:
    title = str(result.get("title") or "")
    title = re.sub(r"\s*[|\-–—:]\s*[^|\-–—:]+\s*$", "", title).strip()
    return title


def search(phrase: str, args: argparse.Namespace) -> dict[str, Any]:
    command = [
        args.exq,
        "search",
        phrase,
        "-n",
        str(args.limit),
        "--text",
        "--highlights",
        "--json",
    ]
    for domain in args.domain:
        command.extend(["-d", domain])
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=180, check=False
    )
    if completed.returncode != 0:
        return {
            "query": phrase,
            "status": "error",
            "exit_code": completed.returncode,
            "stderr": completed.stderr.strip()[:500],
            "results": [],
        }
    payload = json.loads(completed.stdout)
    results = payload.get("results") or []
    scored = []
    phrase_key = normalize(phrase)
    for index, result in enumerate(results):
        url = str(result.get("url") or "")
        official = any(domain in url for domain in OFFICIAL_DOMAIN_PRIORITY)
        title = candidate_from_result(result)
        title_key = normalize(title)
        score = (8 if official else 0) + (
            6 if phrase_key and phrase_key in title_key else 0
        )
        if title.lower().startswith(("what is", "how to", "guide to", "explained")):
            score -= 3
        scored.append(
            {
                "rank": index,
                "title": title,
                "url": url,
                "official": official,
                "score": score,
                "highlights": [
                    str(value)[:400] for value in (result.get("highlights") or [])[:2]
                ],
            }
        )
    scored.sort(key=lambda value: (-value["score"], value["rank"]))
    return {
        "query": phrase,
        "status": "complete",
        "provider": "exa",
        "rerank_provider": None,
        "results": scored,
    }


def main() -> None:
    args = parse_args()
    if not shutil.which(args.exq):
        raise SystemExit(f"missing {args.exq}; install the native Exa CLI")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for phrase in args.phrase:
        record = search(phrase.strip(), args)
        record["phrase_sha256"] = hashlib.sha256(phrase.strip().encode()).hexdigest()
        top = record["results"][0] if record["results"] else {}
        record["suggested_canonical"] = top.get("title", "")
        record["suggested_sources"] = [
            value.get("url") for value in record["results"] if value.get("official")
        ][:3]
        records.append(record)
    payload = {
        "schema_version": 1,
        "providers": ["exa"],
        "excluded_providers": ["zcode", "openrouter", "glm"],
        "records": records,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(args.out)
    for record in records:
        print(f"{record['query']} -> {record['suggested_canonical']}", file=sys.stderr)


if __name__ == "__main__":
    main()
