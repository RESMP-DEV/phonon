#!/usr/bin/env python3
"""Run matched full-data Liquid prompt-format trials."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from optuna_sweep import SweepConfig, build_trial_plan, execute_plan
from prompts import PROMPTS, prompt_sha256

HERE = Path(__file__).resolve().parent

BEST_TRIAL_4 = {
    "lr": 6.505720091093967e-05,
    "adapter_lr": 4.943429131224935e-05,
    "rank": 32,
    "lora_dropout": 0.033694424584220395,
    "context_length": 512,
    "warmup": 50,
}


def make_config(args: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        research_python=args.research_python,
        work_root=args.work_root,
        manifest=args.manifest,
        audio_root=args.audio_root,
        exclude=args.exclude,
        eval_slice=args.eval_slice,
        public_rows=args.public_rows,
        public_revision=args.public_revision,
        public_steps=args.public_steps,
        aqua_steps=args.aqua_steps,
        eval_limit=args.eval_limit,
        prompt_ids=tuple(PROMPTS),
        timeout=args.timeout,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
    )


def record_score_wandb(
    args: argparse.Namespace, plan: dict[str, Any], score: dict[str, Any]
) -> None:
    """Append an aggregate-score transaction for later same-ID W&B sync."""

    if not args.wandb_project or args.wandb_mode == "disabled":
        return
    score_path = Path(plan["score"])
    if not score_path.is_file():
        raise RuntimeError(f"missing score file for W&B score transaction: {score_path}")
    if score != json.loads(score_path.read_text(encoding="utf-8")):
        raise RuntimeError("in-memory score does not match the durable score receipt")
    subprocess.run(
        [
            str(plan["research_python"]),
            str(HERE / "log_wandb_score.py"),
            "--root",
            str(plan["trial_root"]),
            "--project",
            args.wandb_project,
            "--mode",
            args.wandb_mode,
            "--run-id",
            str(plan["wandb_run_id"]),
            "--name",
            str(plan["wandb_run_id"]),
        ],
        check=True,
    )


def write_receipt(
    path: Path,
    *,
    status: str,
    prompt_id: str,
    trial_root: Path,
    started: float,
    score: dict[str, Any] | None = None,
    error: str | None = None,
    wandb: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "prompt_id": prompt_id,
        "prompt_sha256": prompt_sha256(prompt_id),
        "fixed_params": BEST_TRIAL_4,
        "trial_root": str(trial_root),
        "started_at_unix": started,
        "updated_at_unix": time.time(),
        "score": score,
        "error": error,
        "wandb": wandb,
        "non_claims": [
            "does not vary hyperparameters",
            "does not prove product latency on macOS",
            "uses accepted historical text as ground truth",
        ],
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def wandb_receipt(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(args.wandb_project) and args.wandb_mode != "disabled"
    return {
        "enabled": enabled,
        "project": args.wandb_project if enabled else None,
        "mode": args.wandb_mode if enabled else "disabled",
        "run_id": str(plan["wandb_run_id"]) if enabled else None,
        "credential_boundary": "offline capture on B550; sync only from the Mac",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-id",
        action="append",
        required=True,
        help="prompt ID to run; repeat the flag for a sequential lane",
    )
    parser.add_argument("--trial-base", type=int, default=100)
    parser.add_argument(
        "--research-python",
        type=Path,
        default=Path("/home/kearm/envs/salm-lora/bin/python"),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/home/kearm/salm-lora/build/format-bakeoff"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/kearm/salm-lora/hq-train-manifest.jsonl"),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("/home/kearm/aqua-training-data"),
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=Path("/home/kearm/salm-lora/slice-eval-500.jsonl"),
    )
    parser.add_argument(
        "--eval-slice",
        type=Path,
        default=Path("/home/kearm/salm-lora/slice-eval-500.jsonl"),
    )
    parser.add_argument("--public-rows", type=int, default=10_000)
    parser.add_argument(
        "--public-revision",
        default="969944574ea3f37890beaf67ea651e160cfaf043",
    )
    parser.add_argument("--public-steps", type=int, default=10_000)
    parser.add_argument("--aqua-steps", type=int, default=1_000)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=86_400)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="offline",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(args.prompt_id) - set(PROMPTS))
    if unknown:
        raise SystemExit(f"unknown prompt IDs: {unknown}")
    return args


def main() -> None:
    args = parse_args()
    config = make_config(args)
    plans = []
    for offset, prompt_id in enumerate(args.prompt_id):
        params = {**BEST_TRIAL_4, "prompt_id": prompt_id}
        plan = build_trial_plan(args.trial_base + offset, params, config)
        plans.append(plan)
    if args.dry_run:
        print(json.dumps({"fixed_params": BEST_TRIAL_4, "plans": plans}, indent=2, default=str))
        return

    for plan in plans:
        prompt_id = plan["prompt_id"]
        trial_root = Path(plan["trial_root"])
        receipt = trial_root / "format-receipt.json"
        started = time.time()
        trial_root.mkdir(parents=True, exist_ok=True)
        (trial_root / "params.json").write_text(
            json.dumps({**BEST_TRIAL_4, "prompt_id": prompt_id}, indent=1) + "\n",
            encoding="utf-8",
        )
        telemetry = wandb_receipt(args, plan)
        write_receipt(
            receipt,
            status="running",
            prompt_id=prompt_id,
            trial_root=trial_root,
            started=started,
            wandb=telemetry,
        )
        try:
            score = execute_plan(plan, config.timeout)
            record_score_wandb(args, plan, score)
            write_receipt(
                receipt,
                status="complete",
                prompt_id=prompt_id,
                trial_root=trial_root,
                started=started,
                score=score,
                wandb=telemetry,
            )
            print("FORMAT_DONE", prompt_id, json.dumps(score, sort_keys=True), flush=True)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            write_receipt(
                receipt,
                status="failed",
                prompt_id=prompt_id,
                trial_root=trial_root,
                started=started,
                error=f"{type(error).__name__}: {error}",
                wandb=telemetry,
            )
            raise


if __name__ == "__main__":
    main()
