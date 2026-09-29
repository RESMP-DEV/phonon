#!/usr/bin/env bash
# GPU 0 only. Caller must already hold the grok-synth gpu0 lease.
set -euo pipefail
export HF_HOME=/data/hf
export HF_HUB_CACHE=/data/hf/hub
export HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export UV_TORCH_BACKEND=cu130
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OVERNIGHT_AGENT=grok-synth
cd /home/user/phonon
exec uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  python research/synth_v0/generate_utterances.py \
  --n 22000 --batch-size 16 --target-min 8000 --target-max 12000 --temperature 0.7
