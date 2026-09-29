"""Paths and system registry for the multi-ASR error breakdown."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO = Path("/home/user/phonon")
OPTION0 = REPO / "scripts" / "option0"
OPTION0_PREDS = REPO / "runs" / "reports" / "option0_sweep_20260915" / "preds"
RESEARCH = REPO / "research" / "asr_errors_v0"
DATA = Path("/data/phonon_asr_errors_v0")
LATENT = Path("/data/phonon_latent_v0")
WISPR_ROOT = Path("/data/phonon_personal/wispr_20260915")
JULY_ROOT = Path("/data/phonon_personal/dictation_eval_20260719")
V0_PARAKEET = Path("/data/phonon_corrector_v0/parakeet_v2_wispr.jsonl")
V1_UNIFIED = Path("/data/phonon_corrector_v1/hyps_parakeet_unified.jsonl")
ENGLISH_WORDS = DATA / "english_words.txt"

CLIPS_PATH = DATA / "clips.jsonl"
HYPS_DIR = DATA / "hyps"
LOG_DIR = DATA / "logs"
CHUNK_DIR = DATA / "chunks"

SYNTHV2 = (
    REPO
    / "runs/finetune/parakeet_v3_synthv2_jo_lr1em7_interp_a0p05_best_20260629"
    / "checkpoint.nemo"
)

SPLITS = ("holdout120", "wispr_train640", "edit25", "aqua19")
WISPR_SPLITS = ("holdout120", "wispr_train640")

AGENT = "grok-asr"

# option0 jsonl tag -> our tag (same string except the personal fine-tune).
OPTION0_GATES = ("wispr_holdout120", "wispr_edit25", "aqua_new_holdout")
GATE_TO_SPLIT = {
    "wispr_holdout120": "holdout120",
    "wispr_edit25": "edit25",
    "aqua_new_holdout": "aqua19",
}

SYSTEMS = (
    {
        "tag": "wispr_asr",
        "family": None,
        "model": "wispr_asrText",
        "gpu": False,
    },
    {
        "tag": "parakeet-tdt-0.6b-v2",
        "family": "nemo",
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "gpu": True,
        "batch_size": 16,
    },
    {
        "tag": "parakeet-tdt-0.6b-v3",
        "family": "nemo",
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "gpu": True,
        "batch_size": 16,
    },
    {
        "tag": "parakeet-unified-en-0.6b",
        "family": "nemo",
        "model": "nvidia/parakeet-unified-en-0.6b",
        "gpu": True,
        "batch_size": 16,
    },
    {
        "tag": "parakeet_v3_synthv2_a0p05",
        "family": "nemo",
        "model": str(SYNTHV2),
        "gpu": True,
        "batch_size": 16,
    },
    {
        "tag": "Qwen3-ASR-0.6B-hf",
        "family": "transformers-qwen3asr",
        "model": "Qwen/Qwen3-ASR-0.6B-hf",
        "gpu": True,
        "batch_size": 4,
    },
    {
        "tag": "Qwen3-ASR-1.7B-hf",
        "family": "transformers-qwen3asr",
        "model": "Qwen/Qwen3-ASR-1.7B-hf",
        "gpu": True,
        "batch_size": 2,
    },
    {
        "tag": "granite-speech-4.1-2b",
        "family": "transformers-granite",
        "model": "ibm-granite/granite-speech-4.1-2b",
        "gpu": True,
        "batch_size": 1,
    },
    {
        "tag": "granite-speech-5.0-470m-turboctc",
        "family": "transformers-granite-ctc",
        "model": "ibm-granite/granite-speech-5.0-470m-turboctc",
        "gpu": True,
        "batch_size": 8,
    },
)

LATENT_TAGS = {
    "parakeet-tdt-0.6b-v2": "parakeet-tdt-0.6b-v2",
    "parakeet-unified-en-0.6b": "parakeet-unified-en-0.6b",
    "Qwen3-ASR-0.6B-hf": "Qwen3-ASR-0.6B-hf",
    "granite-speech-4.1-2b": "granite-speech-4.1-2b",
    "granite-speech-5.0-470m-turboctc": "granite-speech-5.0-470m-turboctc",
}

OPTION0_TAGS = {
    "parakeet-tdt-0.6b-v2": "parakeet-tdt-0.6b-v2",
    "parakeet-tdt-0.6b-v3": "parakeet-tdt-0.6b-v3",
    "parakeet-unified-en-0.6b": "parakeet-unified-en-0.6b",
    "Qwen3-ASR-0.6B-hf": "Qwen3-ASR-0.6B-hf",
    "Qwen3-ASR-1.7B-hf": "Qwen3-ASR-1.7B-hf",
    "granite-speech-4.1-2b": "granite-speech-4.1-2b",
}


def ensure_dirs() -> None:
    for path in (DATA, HYPS_DIR, LOG_DIR, CHUNK_DIR, DATA / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    RESEARCH.mkdir(parents=True, exist_ok=True)


def set_hf_env() -> None:
    os.environ["HF_HOME"] = "/data/hf"
    os.environ["HF_HUB_CACHE"] = "/data/hf/hub"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/hf/hub"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OVERNIGHT_AGENT", AGENT)


def heartbeat(ttl: str = "30m") -> None:
    subprocess.run(
        ["overnight-compute", "heartbeat", "--agent", AGENT, "--ttl", ttl],
        check=False,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path: Path, key: str = "id") -> set[str]:
    if not path.exists():
        return set()
    return {str(row[key]) for row in iter_jsonl(path) if row.get(key)}


def hyp_path(tag: str) -> Path:
    return HYPS_DIR / f"{tag}.jsonl"


def system_by_tag(tag: str) -> dict[str, Any]:
    for system in SYSTEMS:
        if system["tag"] == tag:
            return system
    raise KeyError(tag)
