#!/usr/bin/env bash
# $1 = GPU  $2 = adapter spec  $3 = sets  $4 = batch size
set -uo pipefail
G=$1; A=$2; S=${3:-seen,unseen,new,pool2,real}; B=${4:-32}
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8
cd /home/user/phonon
date +"%H:%M:%S EVAL2 $A gpu=$G sets=$S bs=$B"
uv run python research/pool2_v0/refine_pool2.py --adapter "$A" --sets "$S" \
  --batch-size "$B" --real-batch-size 16
echo "EVAL2 $A rc=$?"
date +"%H:%M:%S EVAL2 $A END"
