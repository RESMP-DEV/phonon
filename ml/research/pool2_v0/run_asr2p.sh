#!/usr/bin/env bash
# $1 = part tag  $2 = GPU
set -uo pipefail
P=$1; G=$2
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd /home/user/phonon
date +"%H:%M:%S ASRP $P gpu$G START"
uv run python research/term_eval_v0/asr_clips.py \
  --manifest /data/phonon_pool2_v0/tts_manifest_pool_${P}.jsonl \
  --out /data/phonon_pool2_v0/asr_pool_${P}.jsonl --batch-size 96
rc=$?
echo "ASRP $P rc=$rc"
[ "$rc" -eq 0 ] || exit 1
date +"%H:%M:%S ASRP $P DONE"
