#!/usr/bin/env python3
"""Persistent Optuna sweep for the reverse Audio-to-VL final-training recipe.

Prompt formatting is part of each trial's training contract. Every trial builds
public and Aqua packs with the selected prompt, trains with that exact pack,
evaluates with the same prompt ID, and records the prompt hash in Optuna.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompts import PROMPTS, prompt_sha256

HERE = Path(__file__).resolve().parent
REVERSE_DIR = HERE.parent / "reverse_vl_v0"


@dataclass(frozen=True)
class SweepConfig:
    research_python: Path
    work_root: Path
    manifest: Path
    audio_root: Path
    exclude: Path
    eval_slice: Path
    public_rows: int
    public_revision: str
    public_steps: int
    aqua_steps: int
    eval_limit: int
    prompt_ids: tuple[str, ...]
    timeout: int


def command(python: Path, script: Path, *arguments: str) -> list[str]:
    return [str(python), str(script), *(str(value) for value in arguments)]


def build_trial_plan(trial_number: int, params: dict[str, Any], config: SweepConfig) -> dict[str, Any]:
    """Return a deterministic command plan for one trial."""

    trial_root = config.work_root / f"trial-{trial_number:04d}"
    context_key = int(params["context_length"])
    public_pack = config.work_root / (
        f"granary-{params['prompt_id']}-{config.public_rows}-"
        f"{config.public_revision[:12]}-ctx{context_key}"
    )
    manifest_material = (
        config.manifest.read_bytes()
        if config.manifest.exists()
        else str(config.manifest).encode()
    )
    manifest_hash = hashlib.sha256(manifest_material).hexdigest()[:16]
    aqua_pack = (
        config.work_root
        / f"aqua-{params['prompt_id']}-{manifest_hash}-ctx{context_key}"
    )
    public_output = trial_root / "public"
    aqua_output = trial_root / "aqua"
    hyps = trial_root / "hyps.jsonl"
    score = trial_root / "score.json"
    public_checkpoint = public_output / f"ckpt_step{config.public_steps:05d}.pt"

    build_public = command(
        config.research_python,
        HERE / "build_granary_prompt_pack.py",
        "--prompt-id", params["prompt_id"],
        "--rows", config.public_rows,
        "--revision", config.public_revision,
        "--out", public_pack,
        "--context-length", params["context_length"],
    )
    build_aqua = command(
        config.research_python,
        HERE / "build_aqua_prompt_pack.py",
        "--manifest", config.manifest,
        "--audio-root", config.audio_root,
        "--exclude", config.exclude,
        "--prompt-id", params["prompt_id"],
        "--out", aqua_pack,
        "--context-length", params["context_length"],
    )
    train_public = command(
        config.research_python,
        REVERSE_DIR / "train_reverse_audio_vl.py",
        "--dataset", public_pack,
        "--steps", config.public_steps,
        "--context-length", params["context_length"],
        "--warmup", params["warmup"],
        "--lr", params["lr"],
        "--adapter-lr", params["adapter_lr"],
        "--rank", params["rank"],
        "--lora-dropout", params["lora_dropout"],
        "--ckpt-every", max(100, config.public_steps // 10),
        "--keep-ckpts", 3,
        "--out", public_output,
    )
    train_aqua = command(
        config.research_python,
        REVERSE_DIR / "train_reverse_audio_vl.py",
        "--dataset", aqua_pack,
        "--resume", public_checkpoint,
        "--steps", config.public_steps + config.aqua_steps,
        "--context-length", params["context_length"],
        "--warmup", params["warmup"],
        "--lr", params["lr"],
        "--adapter-lr", params["adapter_lr"],
        "--rank", params["rank"],
        "--lora-dropout", params["lora_dropout"],
        "--ckpt-every", max(100, config.aqua_steps // 2),
        "--keep-ckpts", 3,
        "--out", aqua_output,
    )
    evaluate = command(
        config.research_python,
        REVERSE_DIR / "transcribe_reverse_audio_vl.py",
        "--slice", config.eval_slice,
        "--audio-root", config.audio_root,
        "--adapter", aqua_output / "reverse_audio_vl_adapter.safetensors",
        "--lora-rank", params["rank"],
        "--limit", config.eval_limit,
        "--prompt-id", params["prompt_id"],
        "--out", hyps,
    )
    score_command = command(
        config.research_python,
        HERE / "score_fair_wer.py",
        hyps,
    )
    return {
        "prompt_id": params["prompt_id"],
        "context_length": params["context_length"],
        "trial_root": trial_root,
        "public_pack": public_pack,
        "aqua_pack": aqua_pack,
        "public_checkpoint": public_checkpoint,
        "hyps": hyps,
        "score": score,
        "commands": {
            "build_public": build_public,
            "build_aqua": build_aqua,
            "train_public": train_public,
            "train_aqua": train_aqua,
            "evaluate": evaluate,
            "score": score_command,
        },
    }


def run_step(name: str, argv: list[str], log_path: Path, timeout: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    if process.returncode:
        tail = log_path.read_text(errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"{name} failed with exit code {process.returncode}\n" + "\n".join(tail)
        )


def pack_is_valid(path: Path, prompt_id: str, context_length: int) -> bool:
    metadata_path = path / "pack_meta.json"
    if not metadata_path.is_file() or not (path / "dataset_info.json").is_file():
        return False
    metadata = json.loads(metadata_path.read_text())
    return (
        metadata.get("prompt_id") == prompt_id
        and metadata.get("prompt_sha256") == prompt_sha256(prompt_id)
        and metadata.get("context_length") == context_length
    )


def execute_plan(plan: dict[str, Any], timeout: int) -> dict[str, Any]:
    trial_root = plan["trial_root"]
    logs = trial_root / "logs"
    trial_root.mkdir(parents=True, exist_ok=True)
    steps: list[tuple[str, list[str], Path]] = [
        ("build_public", plan["commands"]["build_public"], logs / "build_public.log"),
        ("build_aqua", plan["commands"]["build_aqua"], logs / "build_aqua.log"),
        ("train_public", plan["commands"]["train_public"], logs / "train_public.log"),
        ("train_aqua", plan["commands"]["train_aqua"], logs / "train_aqua.log"),
        ("evaluate", plan["commands"]["evaluate"], logs / "evaluate.log"),
        ("score", plan["commands"]["score"], logs / "score.log"),
    ]
    for name, argv, log_path in steps:
        if name == "build_public" and pack_is_valid(
            plan["public_pack"], plan["prompt_id"], int(plan["context_length"])
        ):
            continue
        if name == "build_aqua" and pack_is_valid(
            plan["aqua_pack"], plan["prompt_id"], int(plan["context_length"])
        ):
            continue
        run_step(name, argv, log_path, timeout)
    score_path = plan["score"]
    if not score_path.exists():
        # The score subprocess writes JSON to its own log; persist a durable copy.
        parsed = json.loads((logs / "score.log").read_text())
        score_path.write_text(json.dumps(parsed, indent=1))
    return json.loads(score_path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--storage", default="sqlite:///build/optuna/final_sweep.db")
    parser.add_argument("--research-python", type=Path, default=Path("/home/kearm/envs/salm-lora/bin/python"))
    parser.add_argument("--work-root", type=Path, default=Path("build/optuna"))
    parser.add_argument("--manifest", type=Path, default=Path("hq-train-manifest.jsonl"))
    parser.add_argument("--audio-root", type=Path, default=Path("~/aqua-training-data"))
    parser.add_argument("--exclude", type=Path, default=Path("slice-eval-500.jsonl"))
    parser.add_argument("--eval-slice", type=Path, default=Path("slice-eval-500.jsonl"))
    parser.add_argument("--public-rows", type=int, default=10_000)
    parser.add_argument("--public-revision", default="969944574ea3f37890beaf67ea651e160cfaf043")
    parser.add_argument("--public-steps", type=int, default=10_000)
    parser.add_argument("--aqua-steps", type=int, default=1_000)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--prompt-ids", default=",".join(PROMPTS))
    parser.add_argument("--timeout", type=int, default=86_400)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> SweepConfig:
    prompt_ids = tuple(value.strip() for value in args.prompt_ids.split(",") if value.strip())
    unknown = sorted(set(prompt_ids) - set(PROMPTS))
    if unknown:
        raise SystemExit(f"unknown prompt IDs: {unknown}")
    return SweepConfig(
        research_python=args.research_python,
        work_root=args.work_root,
        manifest=args.manifest,
        audio_root=args.audio_root.expanduser(),
        exclude=args.exclude,
        eval_slice=args.eval_slice,
        public_rows=args.public_rows,
        public_revision=args.public_revision,
        public_steps=args.public_steps,
        aqua_steps=args.aqua_steps,
        eval_limit=args.eval_limit,
        prompt_ids=prompt_ids,
        timeout=args.timeout,
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    if args.dry_run:
        params = {
            "prompt_id": config.prompt_ids[0],
            "lr": 1e-4,
            "adapter_lr": 1e-4,
            "rank": 16,
            "lora_dropout": 0.05,
            "context_length": 768,
            "warmup": 100,
        }
        plan = build_trial_plan(0, params, config)
        print(json.dumps({"params": params, "plan": plan}, indent=2, default=str))
        return
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")
    if not args.execute:
        raise SystemExit("refusing to train; pass --execute (or --dry-run)")

    import optuna

    def objective(trial: optuna.Trial) -> float:
        params = {
            "prompt_id": trial.suggest_categorical("prompt_id", list(config.prompt_ids)),
            "lr": trial.suggest_float("lr", 3e-5, 2e-4, log=True),
            "adapter_lr": trial.suggest_float("adapter_lr", 3e-5, 2e-4, log=True),
            "rank": trial.suggest_categorical("rank", [8, 16, 32]),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.1),
            "context_length": trial.suggest_categorical("context_length", [512, 768, 1024]),
            "warmup": trial.suggest_int("warmup", 25, 200, step=25),
        }
        trial.set_user_attr("prompt_sha256", prompt_sha256(params["prompt_id"]))
        trial.set_user_attr("work_root", str(config.work_root))
        plan = build_trial_plan(trial.number, params, config)
        trial.set_user_attr("trial_root", str(plan["trial_root"]))
        (plan["trial_root"]).mkdir(parents=True, exist_ok=True)
        (plan["trial_root"] / "params.json").write_text(json.dumps(params, indent=1))
        score = execute_plan(plan, config.timeout)
        for key in ("fair_wer", "strict_lc_wer", "fair_exact", "n"):
            trial.set_user_attr(key, score.get(key))
        trial.set_user_attr("score_path", str(plan["score"]))
        return float(score["fair_wer"])

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=args.trials,
        gc_after_trial=True,
        catch=(RuntimeError, subprocess.TimeoutExpired),
    )
    print(json.dumps({"best_trial": study.best_trial.number, "best_value": study.best_value}, indent=2))


if __name__ == "__main__":
    main()
