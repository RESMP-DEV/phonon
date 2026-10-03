#!/usr/bin/env bash
set -euo pipefail

repo_root="${PHONON_REPO_ROOT:-/home/kearm/phonon}"
research_root="${PHONON_RESEARCH_ROOT:-/home/kearm/salm-lora}"
audio_root="${PHONON_AUDIO_ROOT:-/home/kearm/aqua-training-data}"
base="${PHONON_VOXTRAL_BASE:-$research_root/build/asr-teachers/voxtral-small-24b-hfd-da5b424}"
python_bin="${PHONON_VOXTRAL_PYTHON:-$research_root/build/asr-teachers/venvs/voxtral-mxfp4/bin/python}"
limit="${PHONON_VOXTRAL_CALIBRATION_ROWS:-2}"
name="${PHONON_VOXTRAL_RUN_NAME:-voxtral-mxfp4-smoke3}"

test -d "$repo_root"
test -d "$research_root"
test -d "$audio_root"
test -d "$base"
test -x "$python_bin"
test ! -e "$research_root/build/asr-teachers/$name-calibration.json"
test ! -e "$research_root/build/asr-teachers/$name"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec "$python_bin" "$repo_root/ml/research/asr_teachers/quantize_voxtral_mxfp4.py" \
	--manifest "$research_root/hq-train-manifest.jsonl" \
	--eval-slice "$research_root/slice-eval-500.jsonl" \
	--audio-root "$audio_root" \
	--calibration-manifest-out "$research_root/build/asr-teachers/$name-calibration.json" \
	--limit "$limit" \
	--base "$base" \
	--output "$research_root/build/asr-teachers/$name" \
	--receipt "$research_root/build/asr-teachers/$name-receipt.json"
