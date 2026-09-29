"""Where everything lives. One import so no module hard-codes a second copy."""
from __future__ import annotations

import os
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2]          # research/harness
RESEARCH = HARNESS.parent                               # research
REPO = Path(os.environ.get("PHONON_REPO_ROOT", str(RESEARCH.parent)))
DATA = Path(os.environ.get("PHONON_DATA_ROOT", "/data"))

REGISTRY_DIR = HARNESS / "registry"
ADAPTERS_YAML = REGISTRY_DIR / "adapters.yaml"
SETS_YAML = REGISTRY_DIR / "sets.yaml"
CONFIGS = HARNESS / "configs"

QUEUE_ROOT = Path(os.environ.get("PHONON_QUEUE_ROOT", str(DATA / "phonon_queue")))
CORPUS_ROOT = Path(os.environ.get("PHONON_CORPUS_ROOT", str(DATA / "phonon_harness")))

TRAIN_LORA = RESEARCH / "corrector_v0" / "train_lora.py"
GEN_SENTENCES = RESEARCH / "scaling_v0" / "gen_more.py"
TTS_JOBS = RESEARCH / "bigrun_v0" / "tts_jobs.py"
ASR_CLIPS = RESEARCH / "term_eval_v0" / "asr_clips.py"
BENCH_PROJECT = RESEARCH / "bench_v0"

ADAPTER_ROOT = Path(os.environ.get("PHONON_ADAPTER_ROOT", str(DATA / "phonon_corrector_v0/adapters")))

# research packages the corpus modules import from (numerics live there, not here)
VOCAB_V0 = RESEARCH / "vocab_v0"
VOCAB_V1 = RESEARCH / "vocab_v1"
RETRIEVAL_V2 = RESEARCH / "retrieval_v2"


def hf_env(gpu: int | str | None = None) -> dict[str, str]:
    """The env every GPU step on gpubox runs with."""
    env = dict(os.environ)
    env.update({
        "HF_HOME": str(DATA / "hf"),
        "HF_HUB_CACHE": str(DATA / "hf/hub"),
        "HUGGINGFACE_HUB_CACHE": str(DATA / "hf/hub"),
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "UV_TORCH_BACKEND": "cu130",
    })
    if gpu is not None and str(gpu) != "any":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def research_syspath() -> None:
    """Make the existing research modules importable (numerics are reused, never copied)."""
    import sys

    for p in (VOCAB_V0, VOCAB_V1, RETRIEVAL_V2, RESEARCH / "corrector_v0"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
