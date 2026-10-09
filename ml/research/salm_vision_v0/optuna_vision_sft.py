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
SCORER = REPO / "ml/research/final_sweep/score_fair_wer.py"
FUSE = REPO / "ml/research/salm_vision_v0/fuse_salm_vision.py"
FUSED_AUDIO_EVAL = REPO / "ml/research/salm_vision_v0/eval_fused_salm.py"
FUSED_VISION_EVAL = REPO / "ml/research/salm_vision_v0/eval_fused_vision.py"
DEFAULT_INIT_ADAPTER = SALM / "salm-vision-cpt-v1/cpt_adapter.safetensors"
AUDIO_ROOT = Path("/home/kearm/aqua-training-data")
ENV_MANIFEST = REPO / "ml/research/salm_vision_v0/environment-b550.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log: Path, *, cwd: Path | None = None) -> None:
    started = log.parent / f"{log.stem}.started"
    completed = log.parent / f"{log.stem}.completed"
    started.write_text(json.dumps({"command": command, "started_at_unix": __import__("time").time()}) + "\n")
    with log.open("w") as sink:
        process = subprocess.run(
            command, stdout=sink, stderr=subprocess.STDOUT, check=False, text=True, cwd=cwd,
            env={**os.environ, "PYTHONPATH": str(REPO / "ml/research/salm_vision_v0"), "CUDA_VISIBLE_DEVICES": "0"},
        )
    completed.write_text(json.dumps({"command": command, "returncode": process.returncode, "completed_at_unix": __import__("time").time()}) + "\n")
    if process.returncode:
        raise optuna.TrialPruned(f"command failed ({process.returncode}): {command}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def kernel_mode() -> dict[str, object]:
    import importlib.util
    from importlib import metadata

    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "torch": version("torch"),
        "flash_attn": version("flash-attn"),
        "causal_conv1d": version("causal-conv1d"),
        "flash_attn_importable": importlib.util.find_spec("flash_attn") is not None,
        "causal_conv1d_importable": importlib.util.find_spec("causal_conv1d") is not None,
    }


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
    root = args.study_root / f"trial-{trial.number:04d}"
    train = root / "train"
    root.mkdir(parents=True, exist_ok=False)
    training_command = [
        str(PY), str(TRAINER),
        "--audio-pack", str(args.audio_pack),
        "--corrector-pack", str(SALM / "aqua-correction-v4"),
        "--vision-pack", str(SALM / "aqua-vision-dataset-v1"),
        "--ratio", "4:1:1",
        "--steps", str(args.steps),
        "--lr", repr(lr),
        "--rank", "16", "--warmup", "50",
        "--init-variant", "omp",
        "--init-adapter", str(args.init_adapter),
        "--transplant-dir", str(SALM / "vision-transplant"),
        "--audio-context", "768", "--text-context", "768", "--vision-context", "1024",
        "--out", str(train), "--device", "cuda", "--ckpt-every", "200", "--keep-ckpts", "1",
    ]
    trial.set_user_attr("training_command", training_command)
    trial.set_user_attr("init_adapter_sha256", sha256(args.init_adapter))
    trial.set_user_attr("init_adapter", str(args.init_adapter))
    trial.set_user_attr("environment_manifest", str(ENV_MANIFEST))
    trial.set_user_attr("source_revision", subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())
    trial.set_user_attr("kernel_mode", kernel_mode())
    run(training_command, root / "train.log")

    fused = root / "fused"
    fuse_command = [
        str(PY), str(FUSE),
        "--adapter", str(train / "cpt_adapter.safetensors"),
        "--out", str(fused),
        "--vision-rows", str(SALM / "vision-transplant/rows_omp.safetensors"),
        "--init-variant", "omp", "--rank", "16", "--device", "cuda",
    ]
    trial.set_user_attr("fuse_command", fuse_command)
    run(fuse_command, root / "fuse.log")

    hyps = root / "hyps.jsonl"
    audio_command = [
        str(PY), str(FUSED_AUDIO_EVAL),
        "--merged", str(fused / "merged_lfm.safetensors"),
        "--slice", str(SALM / "slice-eval-500.jsonl"),
        "--out", str(hyps),
        "--history-manifest", str(args.history_manifest),
        "--prompt-id", str(args.prompt_id),
        "--device", "cuda",
    ]
    trial.set_user_attr("audio_command", audio_command)
    run(audio_command, root / "audio-eval.log")
    score_path = root / "audio-score.json"
    with score_path.open("w") as sink:
        subprocess.run(
            [str(PY), str(SCORER), str(hyps)],
            stdout=sink,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    audio = load_json(score_path)

    vision_command = [
        str(PY), str(FUSED_VISION_EVAL),
        "--merged", str(fused / "merged_lfm.safetensors"),
        "--limit", str(args.vision_limit),
        "--out", str(root / "vision-eval.json"),
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
    trial.set_user_attr("vision_summary_sha256", sha256(root / "vision-eval.json"))
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
    parser.add_argument("--audio-pack", type=Path, default=SALM / "aqua-sft-dataset-v3")
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=DEFAULT_INIT_ADAPTER,
        help=(
            "adapter each trial starts from; point this at the best prior stage "
            "to continue optimizing from that point instead of the CPT baseline"
        ),
    )
    parser.add_argument("--history-manifest", type=Path, default=None)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--history-count", type=int, default=2)
    args = parser.parse_args()
    args.study_root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.study_root / 'study.db'}"
    study = optuna.create_study(direction="minimize", study_name=args.study, storage=storage, load_if_exists=True)
    study.set_user_attr("environment", load_json(ENV_MANIFEST))
    study.set_user_attr(
        "fixed_contract",
        {
            "rank": 16,
            "ratio": "4:1:1",
            "steps": args.steps,
            "contexts": [768, 768, 1024],
            "init_adapter": str(args.init_adapter),
            "init_adapter_sha256": sha256(args.init_adapter),
            "audio_pack": str(args.audio_pack),
            "audio_pack_meta": str(args.audio_pack / "pack_meta.json"),
        },
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print(json.dumps({"status": "no_complete_trials", "states": [str(trial.state) for trial in study.trials]}, indent=2))
        raise SystemExit(2)
    print(json.dumps({"best_trial": study.best_trial.number, "value": study.best_value, "params": study.best_params}, indent=2))

if __name__ == "__main__":
    main()
