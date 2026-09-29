#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
cd /home/user/phonon
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  python research/scaling_v0/gen_more.py \
  --terms /data/phonon_bigrun_v0/terms_pool.jsonl \
  --out /data/phonon_bigrun_v0/sentences_8.jsonl \
  --per-term 8 --batch-size 128 --rounds 10 --seed 918 --agent opus-bigcorpus
echo "GEN rc=$?"
