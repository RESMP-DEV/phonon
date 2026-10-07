#!/usr/bin/env python3
"""Persistent Optuna study for post-CPT vision-on-audio ASR SFT.

The search owns the learning rate. Other contract fields are fixed so trials
remain comparable: vision-on-audio graft, rank 16, 4:1:1 audio/corrector/vision
mix, context 768/768/1024, and initialization from the measured CPT adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import optuna

REPO = Path("/home/kearm/phonon")
SALM = Path("/home/kearm/salm-lora")
PY = Path("/home/kearm/envs/salm-lora/bin/python")
TRAINER = REPO / "ml/research/salm_vision_v0/train_salm_vision_cpt.py"
TRANSCRIBER = SALM / "transcribe_lfm25audio.py"
SCORER = REPO / "ml/research/final_sweep/score_fair_wer.py"
VISION_EVAL = SALM / "eval_vision_cpt.py"
INIT_ADAPTER = SALM / "salm-vision-cpt-v1/cpt_adapter.safetensors"
ENV_MANIFEST = REPO / "ml/research/salm_vision_v0/environment-b550.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> None:
    started = log.parent / f"{log.stem}.started"
    completed = log.parent / f"{log.stem}.completed"
    started.write_text(json.dumps({"command": command, "started_at_unix": __import__("time").time()}) + "\n")
    with log.open("w") as sink:
        process = subprocess.run(command, stdout=sink, stderr=subprocess.STDOUT, check=False, text=True)
    completed.write_text(json.dumps({"command": command, "returncode": process.returncode, "completed_at_unix": __import__("time").time()}) + "\n")
    if process.returncode:
        raise optuna.TrialPruned(f"command failed ({process.returncode}): {command}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
    root = args.study_root / f"trial-{trial.number:04d}"
    train = root / "train"
    root.mkdir(parents=True, exist_ok=False)
    training_command = [
        str(PY), str(TRAINER),
        "--audio-pack", str(SALM / "aqua-sft-dataset-v3"),
        "--corrector-pack", str(SALM / "aqua-correction-v4"),
        "--vision-pack", str(SALM / "aqua-vision-dataset-v1"),
        "--ratio", "4:1:1",
        "--steps", str(args.steps),
        "--lr", repr(lr),
        "--rank", "16", "--warmup", "50",
        "--init-variant", "omp",
        "--init-adapter", str(INIT_ADAPTER),
        "--transplant-dir", str(SALM / "vision-transplant"),
        "--audio-context", "768", "--text-context", "768", "--vision-context", "1024",
        "--out", str(train), "--device", "cuda", "--ckpt-every", "200", "--keep-ckpts", "1",
    ]
    trial.set_user_attr("training_command", training_command)
    trial.set_user_attr("init_adapter_sha256", sha256(INIT_ADAPTER))
    trial.set_user_attr("environment_manifest", str(ENV_MANIFEST))
    run(training_command, root / "train.log")

    hyps = root / "hyps.jsonl"
    audio_command = [
        str(PY), str(TRANSCRIBER),
        "--slice", str(SALM / "slice-eval-500.jsonl"),
        "--out", str(hyps), "--mode", "seq",
        "--adapter", str(train / "cpt_adapter.safetensors"),
        "--lora-rank", "16", "--device", "cuda",
        "--aqua-root", str(SALM / "aqua-training-data"),
    ]
    trial.set_user_attr("audio_command", audio_command)
    run(audio_command, root / "audio-eval.log")
    score_path = root / "audio-score.json"
    with score_path.open("w") as sink:
        process = subprocess.run([str(PY), str(SCORER), str(hyps)], stdout=sink, stderr=subprocess.STDOUT, check=True, text=True)
    audio = load_json(score_path)

    vision_summary_path = root / "vision-summary.json"
    vision_command = [
        str(PY), str(VISION_EVAL),
        "--adapter", str(train / "cpt_adapter.safetensors"),
        "--pack", str(SALM / "aqua-vision-dataset-v1-eval"),
        "--limit", str(args.vision_limit), "--out", str(root / "vision-eval.json"),
    ]
    trial.set_user_attr("vision_command", vision_command)
    run(vision_command, root / "vision-eval.log")
    vision = load_json(root / "vision-eval.json")["summary"]["real"]
    # Penalize catastrophic forgetting without making a small vision delta dominate ASR.
    vision_penalty = max(0.0, vision["wer"] - args.vision_wer_ceiling)
    objective_value = float(audio["fair_wer"] + args.vision_penalty_weight * vision_penalty)
    trial.set_user_attr("audio_score", audio)
    trial.set_user_attr("vision_score", vision)
    trial.set_user_attr("vision_penalty", vision_penalty)
    trial.set_user_attr("adapter_sha256", sha256(train / "cpt_adapter.safetensors"))
    trial.set_user_attr("hyps_sha256", sha256(hyps))
    trial.set_user_attr("score_sha256", sha256(score_path))
    trial.set_user_attr("vision_summary_sha256", sha256(vision_summary_path))
    return objective_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="vision-on-audio-asr-sft-lr-v1")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1.5e-4)
    parser.add_argument("--vision-limit", type=int, default=50)
    parser.add_argument("--vision-wer-ceiling", type=float, default=0.04)
    parser.add_argument("--vision-penalty-weight", type=float, default=2.0)
    parser.add_argument("--study-root", type=Path, default=SALM / "build/optuna/vision-on-audio-asr-sft-lr-v1")
    args = parser.parse_args()
    args.study_root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.study_root / 'study.db'}"
    study = optuna.create_study(direction="minimize", study_name=args.study, storage=storage, load_if_exists=True)
    study.set_user_attr("environment", load_json(ENV_MANIFEST))
    study.set_user_attr("fixed_contract", {"rank": 16, "ratio": "4:1:1", "steps": args.steps, "contexts": [768, 768, 1024]})
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)
    print(json.dumps({"best_trial": study.best_trial.number, "value": study.best_value, "params": study.best_params}, indent=2))

if __name__ == "__main__":
    main()
