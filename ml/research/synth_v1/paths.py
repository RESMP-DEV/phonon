"""Shared paths for synth_v1. Data under /data; results under research/synth_v1."""
from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("/data/phonon_synth_v1")
V0_ROOT = Path("/data/phonon_synth_v0")
REPO_ROOT = Path("/home/user/phonon")
RESEARCH_ROOT = REPO_ROOT / "research" / "synth_v1"
V0_RESEARCH = REPO_ROOT / "research" / "synth_v0"

CLEAN_PATH = V0_ROOT / "clean_utterances.jsonl"
V0_LEXICON = V0_ROOT / "lexicon.jsonl"
V0_RANKED = V0_ROOT / "lexicon_ranked.jsonl"
V0_TTS = V0_ROOT / "tts_pairs.jsonl"
V0_PAIRS = V0_ROOT / "synth_pairs.jsonl"
V0_RESULTS = V0_RESEARCH / "results.json"

LEXICON_PATH = DATA_ROOT / "lexicon_ranked.jsonl"
PAIRS_PATH = DATA_ROOT / "synth_pairs_v1.jsonl"
FILTER_STATS = DATA_ROOT / "filter_stats.json"
REAL_WERS = DATA_ROOT / "tmp" / "real_wers.json"
LOG_DIR = DATA_ROOT / "logs"

ERROR_MODEL = Path("/data/phonon_asr_errors_v0/error_model.json")
ERROR_MODEL_SIBLING = ERROR_MODEL
ERROR_MODEL_LOCAL = DATA_ROOT / "error_model.json"
TRAIN_JSONL = Path("/data/phonon_corrector_v0/train.jsonl")
REAL_PAIRS = Path("/data/phonon_personal/wispr_20260915/corrector_pairs_v0.jsonl")

SYSTEM_PROMPT = (
    "Rewrite the raw dictation transcript into the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)

# real_wispr row from synth_v0/results.md (asr_errors_v0.align_errors)
REAL_RATES = {
    "ENTITY": 0.015996784565916397,
    "FUNCTION": 0.06772699433471138,
    "ORTHOGRAPHY": 0.29108865411116214,
    "NEAR_MISS": 0.0012210993722247742,
    "DROP": 0.0009225233501760833,
    "INSERT": 0.0281427040269484,
}
REAL_WER_MEAN = 0.08435670531545003
REAL_WER_MEDIAN = 0.05
REAL_WER_P90 = 0.210461338531514
