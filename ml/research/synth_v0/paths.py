"""Shared paths for synth_v0. Data lives under /data; results under research/synth_v0."""
from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("/data/phonon_synth_v0")
REPO_ROOT = Path("/home/user/phonon")
RESEARCH_ROOT = REPO_ROOT / "research" / "synth_v0"
HF_HOME = Path("/data/hf")
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"

LEXICON_PATH = DATA_ROOT / "lexicon.jsonl"
TOPICS_PATH = DATA_ROOT / "topics.jsonl"
CLEAN_PATH = DATA_ROOT / "clean_utterances.jsonl"
PAIRS_PATH = DATA_ROOT / "synth_pairs.jsonl"
ERROR_MODEL_SIBLING = Path("/data/phonon_asr_errors_v0/error_model.json")
ERROR_MODEL_LOCAL = DATA_ROOT / "error_model.json"
TRAIN_JSONL = Path("/data/phonon_corrector_v0/train.jsonl")
REAL_PAIRS = Path("/data/phonon_personal/wispr_20260915/corrector_pairs_v0.jsonl")

TTS_WAV_DIR = DATA_ROOT / "tts_wavs"
TTS_MANIFEST = DATA_ROOT / "tts_manifest.jsonl"
TTS_PARAKEET = DATA_ROOT / "tts_parakeet.jsonl"
LOG_DIR = DATA_ROOT / "logs"

REPO_ROOTS = [
    Path("/home/user/cuda"),
    Path("/home/user/dev"),
    Path("/home/user/phonon"),
    Path("/home/user/experiments"),
    Path("/home/user/benchmarks"),
    Path("/home/user/persona-gym"),
    Path("/home/user/box-install"),
]

SYSTEM_PROMPT = (
    "Rewrite the raw dictation transcript into the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)
