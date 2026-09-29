#!/usr/bin/env bash
# GPU 0 only, share with the refiner training chain, stay under 10 GB. GPU 1 off limits.
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /home/user/phonon
exec uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  python research/synth_v2/generate_short.py "$@"
