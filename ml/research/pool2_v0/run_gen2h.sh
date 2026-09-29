#!/usr/bin/env bash
# $1 = half (0|1) -> GPU $1
set -uo pipefail
H=$1
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$H OMP_NUM_THREADS=8
cd /home/user/phonon
date +"%H:%M:%S GEN2 half$H START"
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  python research/scaling_v0/gen_more.py \
  --terms /data/phonon_pool2_v0/terms_pool2_half${H}.jsonl \
  --out /data/phonon_pool2_v0/sentences_4_half${H}.jsonl \
  --per-term 4 --batch-size 128 --rounds 10 --seed $((2918+H)) --agent opus-pool2
echo "GEN2 half$H rc=$?"
date +"%H:%M:%S GEN2 half$H END"
