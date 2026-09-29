#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd /home/user/phonon
exec uv run python research/term_eval_v0/asr_clips.py "$@"
