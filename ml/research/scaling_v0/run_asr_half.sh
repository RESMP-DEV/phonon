#!/usr/bin/env bash
set -uo pipefail
H=$1
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export CUDA_VISIBLE_DEVICES=$H OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd /home/user/phonon
uv run python research/term_eval_v0/asr_clips.py \
  --manifest /data/phonon_scaling_v0/tts_manifest_half${H}.jsonl \
  --out /data/phonon_scaling_v0/asr_half${H}.jsonl --batch-size 64
echo "ASR half${H} rc=$?"
