"""GPU stages for corrector v0. Run only after overnight-compute wait acquired."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_ROOT,
    GEMMA_ID,
    PARAKEET_JSONL,
    QWEN_FALLBACK_ID,
    QWEN_ID,
    RESEARCH_ROOT,
    TTS_PAIRS,
    record_failure,
)

REPO = Path("/home/user/phonon")
LOGS = DATA_ROOT / "logs"
AGENT = "grok-corrector"
TTS_MANIFEST = DATA_ROOT / "tts_manifest.jsonl"
TTS_PARAKEET = DATA_ROOT / "tts_parakeet.jsonl"


def heartbeat() -> None:
    subprocess.run(
        ["overnight-compute", "heartbeat", "--agent", AGENT, "--ttl", "30m"],
        check=False,
    )


def run(cmd: list[str], cwd: Path | None = None, extra_env: dict | None = None) -> int:
    import os

    LOGS.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(cmd), flush=True)
    heartbeat()
    env = os.environ.copy()
    env["HF_HOME"] = "/data/hf"
    env["HF_HUB_CACHE"] = "/data/hf/hub"
    env["HUGGINGFACE_HUB_CACHE"] = "/data/hf/hub"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["UV_TORCH_BACKEND"] = "cu130"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    heartbeat()
    return proc.returncode


def retry(step: str, fn) -> None:
    last = None
    for attempt in range(1, 3):
        try:
            fn()
            return
        except Exception as exc:
            last = exc
            record_failure(step, f"{type(exc).__name__}: {exc}", attempt)
            print(f"{step} attempt {attempt} failed: {exc}", flush=True)
    print(f"{step} failed twice, continuing: {last}", flush=True)


def parakeet_real() -> None:
    code = run(
        [
            "uv",
            "run",
            "python",
            str(RESEARCH_ROOT / "transcribe_parakeet.py"),
            "--out",
            str(PARAKEET_JSONL),
        ],
        cwd=REPO,
    )
    if code != 0:
        raise RuntimeError(f"transcribe real audio exit {code}")


def score_asr() -> None:
    code = run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "jiwer",
            "--with",
            "whisper-normalizer",
            "python",
            str(RESEARCH_ROOT / "score_asr.py"),
        ]
    )
    if code != 0:
        raise RuntimeError(f"score_asr exit {code}")


def wait_for_tts(timeout_s: float = 8 * 3600) -> None:
    t0 = time.time()
    marker = DATA_ROOT / "tts_complete.json"
    while time.time() - t0 < timeout_s:
        if marker.exists() and TTS_MANIFEST.exists():
            meta = json.loads(marker.read_text())
            print(f"tts ready {meta}", flush=True)
            return
        print("waiting for TTS wavs...", flush=True)
        heartbeat()
        time.sleep(60)
    raise TimeoutError("TTS did not finish in time")


def parakeet_tts() -> None:
    marker = DATA_ROOT / "tts_complete.json"
    while True:
        if TTS_MANIFEST.exists():
            code = run(
                [
                    "uv",
                    "run",
                    "python",
                    str(RESEARCH_ROOT / "transcribe_parakeet.py"),
                    "--manifest",
                    str(TTS_MANIFEST),
                    "--out",
                    str(TTS_PARAKEET),
                ],
                cwd=REPO,
            )
            if code != 0:
                raise RuntimeError(f"transcribe tts exit {code}")
        if marker.exists():
            break
        print("TTS still running; another Parakeet pass after 60s", flush=True)
        heartbeat()
        time.sleep(60)
    code = run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "jiwer",
            "--with",
            "whisper-normalizer",
            "python",
            str(RESEARCH_ROOT / "filter_tts_pairs.py"),
            "--in",
            str(TTS_PARAKEET),
            "--out",
            str(TTS_PAIRS),
        ]
    )
    if code != 0:
        raise RuntimeError(f"filter_tts_pairs exit {code}")


def build_train() -> None:
    code = run(
        [
            "uv",
            "run",
            "--no-project",
            "python",
            str(RESEARCH_ROOT / "build_train.py"),
        ]
    )
    if code != 0:
        raise RuntimeError(f"build_train exit {code}")


def train(model: str, name: str) -> None:
    code = run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.12",
            "--with",
            "torch",
            "--with",
            "transformers",
            "--with",
            "peft",
            "--with",
            "trl",
            "--with",
            "datasets",
            "--with",
            "accelerate",
            "--with",
            "safetensors",
            "--with",
            "protobuf",
            "--with",
            "sentencepiece",
            "python",
            str(RESEARCH_ROOT / "train_lora.py"),
            "--model",
            model,
            "--name",
            name,
        ]
    )
    if code != 0:
        raise RuntimeError(f"train {name} exit {code}")


def evaluate() -> None:
    code = run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.12",
            "--with",
            "torch",
            "--with",
            "transformers",
            "--with",
            "peft",
            "--with",
            "datasets",
            "--with",
            "accelerate",
            "--with",
            "jiwer",
            "--with",
            "whisper-normalizer",
            "--with",
            "protobuf",
            "--with",
            "sentencepiece",
            "python",
            str(RESEARCH_ROOT / "eval_corrector.py"),
        ]
    )
    if code != 0:
        raise RuntimeError(f"eval exit {code}")


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    retry("parakeet_real", parakeet_real)
    retry("score_asr", score_asr)
    retry("parakeet_tts", parakeet_tts)
    retry("build_train", build_train)
    retry("train_qwen", lambda: train(QWEN_ID, "qwen3-0.6b"))
    try:
        train(GEMMA_ID, "gemma-4-e2b-it")
    except Exception as exc:
        record_failure("train_gemma", f"{type(exc).__name__}: {exc}", 1)
        print(f"gemma failed once: {exc}; retry then fallback", flush=True)
        try:
            train(GEMMA_ID, "gemma-4-e2b-it")
        except Exception as exc2:
            record_failure("train_gemma", f"{type(exc2).__name__}: {exc2}", 2)
            print(f"gemma failed twice, falling back to {QWEN_FALLBACK_ID}", flush=True)
            retry("train_qwen17", lambda: train(QWEN_FALLBACK_ID, "qwen3-1.7b"))
    retry("eval", evaluate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
