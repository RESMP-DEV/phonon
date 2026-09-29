#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
cd /home/user/phonon
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  python research/scaling_v0/gen_more.py \
  --terms /data/phonon_synth_v3/terms_scale.jsonl \
  --existing /data/phonon_synth_v3/sentences_v3.jsonl \
  --out /data/phonon_scaling_v0/sentences_8.jsonl \
  --per-term 8 --batch-size 64 --rounds 10 --seed 131 --agent opus-scaling
echo "GEN8 rc=$?"
