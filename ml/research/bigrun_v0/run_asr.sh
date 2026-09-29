#!/usr/bin/env bash
# $1 = stage  $2 = half index -> GPU $2
set -uo pipefail
S=$1; H=$2
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export CUDA_VISIBLE_DEVICES=$H OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd /home/user/phonon
uv run python research/term_eval_v0/asr_clips.py \
  --manifest /data/phonon_bigrun_v0/tts_manifest_${S}_half${H}.jsonl \
  --out /data/phonon_bigrun_v0/asr_${S}_half${H}.jsonl --batch-size 96
echo "ASR ${S} half${H} rc=$?"
