"""Shared paths for synth_v2. Data under the data root; results under research/synth_v2."""
from __future__ import annotations

import os
from pathlib import Path

DATA = Path(os.environ.get("PHONON_DATA_ROOT", "/data"))
DATA_ROOT = DATA / "phonon_synth_v2"
V1_ROOT = DATA / "phonon_synth_v1"
V0_ROOT = DATA / "phonon_synth_v0"
REPO_ROOT = Path(os.environ.get("PHONON_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
RESEARCH_ROOT = REPO_ROOT / "research" / "synth_v2"
V1_RESEARCH = REPO_ROOT / "research" / "synth_v1"
V0_RESEARCH = REPO_ROOT / "research" / "synth_v0"
HF_HOME = DATA / "hf"
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"

LEXICON_PATH = V1_ROOT / "lexicon_ranked.jsonl"
TOPICS_PATH = V0_ROOT / "topics.jsonl"
V1_PAIRS = V1_ROOT / "synth_pairs_v1.jsonl"
CLEAN_PATH = DATA_ROOT / "clean_short.jsonl"
PAIRS_PATH = DATA_ROOT / "synth_pairs_v2.jsonl"
FILTER_STATS = DATA_ROOT / "filter_stats.json"
HIST_PLAN = DATA_ROOT / "tmp" / "hist_plan.json"
REAL_WERS_BY_BIN = DATA_ROOT / "tmp" / "real_wers_by_bin.json"
LOG_DIR = DATA_ROOT / "logs"

ERROR_MODEL = DATA / "phonon_asr_errors_v0/error_model.json"
REAL_PAIRS = DATA / "phonon_personal/wispr_20260915/corrector_pairs_v0.jsonl"
V1_REAL_WERS = V1_ROOT / "tmp" / "real_wers.json"
V1_FILTER_STATS = V1_ROOT / "filter_stats.json"

SYSTEM_PROMPT = (
    "Rewrite the raw dictation transcript into the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)

BINS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 10**9))
BIN_NAMES = ("1-5", "6-10", "11-20", "21-40", "41-80", "81+")

KIND_MIX = (
    ("question", 0.35),
    ("instruction", 0.25),
    ("ack", 0.15),
    ("slack_commit", 0.15),
    ("self_correction", 0.10),
)
