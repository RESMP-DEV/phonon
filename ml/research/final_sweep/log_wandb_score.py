#!/usr/bin/env python3
"""Log one aggregate format-trial score as an offline W&B transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from prompts import prompt_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--mode", choices=("online", "offline"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score = json.loads((args.root / "score.json").read_text(encoding="utf-8"))
    params: dict[str, Any] = json.loads(
        (args.root / "params.json").read_text(encoding="utf-8")
    )
    prompt_id = str(params["prompt_id"])
    adapter_path = args.root / "aqua/reverse_audio_vl_adapter.safetensors"
    adapter_sha256 = sha256_file(adapter_path) if adapter_path.is_file() else None

    import wandb

    wandb_dir = args.root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    run = wandb.init(
        project=args.project,
        mode=args.mode,
        id=args.run_id,
        name=args.name,
        resume="must",
        job_type="format_score",
        config={
            "prompt_id": prompt_id,
            "prompt_sha256": prompt_sha256(prompt_id),
            "fixed_params": params,
            "eval_rows": score.get("n"),
            "adapter_sha256": adapter_sha256,
        },
        tags=["phonon", "format-bakeoff", "final-score"],
        allow_val_change=True,
    )
    run.summary.update(
        {
            "eval/fair_wer": score.get("fair_wer"),
            "eval/strict_wer": score.get("strict_lc_wer"),
            "eval/exact": score.get("fair_exact"),
            "eval/rows": score.get("n"),
            "artifact/adapter_sha256": adapter_sha256,
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
