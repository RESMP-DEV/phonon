#!/usr/bin/env python3
"""Sync same-ID offline W&B transactions from the credential authority machine."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def offline_run_directories(root: Path, run_id: str) -> list[Path]:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"unsafe W&B run ID {run_id!r}")
    candidates = sorted(root.glob(f"offline-run-*-{run_id}"))
    return [
        path
        for path in candidates
        if path.is_dir() and next(path.glob("*.wandb"), None) is not None
    ]


def sync_command(run_id: str, run_dirs: list[Path]) -> list[str]:
    return [
        "wandb",
        "sync",
        "--yes",
        "--id",
        run_id,
        *(str(path) for path in run_dirs),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = offline_run_directories(args.root, args.run_id)
    if not run_dirs:
        raise SystemExit(f"no offline W&B transactions for {args.run_id} in {args.root}")
    command = sync_command(args.run_id, run_dirs)
    print(json.dumps({"run_id": args.run_id, "transactions": len(run_dirs), "command": command}))
    if args.dry_run:
        return
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error
