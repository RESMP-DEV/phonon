#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8
cd /home/user/phonon
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  python research/scaling_v0/gen_more.py \
  --terms /data/phonon_bigrun_v0/terms_heldout_new.jsonl \
  --out /data/phonon_bigrun_v0/sentences_heldout_new.jsonl \
  --per-term 3 --batch-size 100 --rounds 10 --seed 919 --agent opus-bigcorpus
echo "GEN_HELDOUT rc=$?"
