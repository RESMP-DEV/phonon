#!/usr/bin/env bash
# $1 = tag  $2 = shard  $3 = gpu  $4 = batch
set -uo pipefail
S=$1; H=$2; G=$3; B=${4:-96}
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd /home/user/phonon
date +"%H:%M:%S ASR3 ${S} s${H} START"
uv run python research/term_eval_v0/asr_clips.py \
  --manifest /data/phonon_pool3_v0/tts_manifest_${S}_s${H}.jsonl \
  --out /data/phonon_pool3_v0/asr_${S}_s${H}.jsonl --batch-size $B
rc=$?
echo "ASR3 ${S} s${H} rc=$rc"
date +"%H:%M:%S ASR3 ${S} s${H} DONE"
[ "$rc" -eq 0 ] || exit 1
