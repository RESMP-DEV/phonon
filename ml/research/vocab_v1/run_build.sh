#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16
cd /home/user/phonon
uv run --no-project --python 3.12 --with numpy --with rapidfuzz --with metaphone \
  python research/vocab_v1/build_data_v1.py
echo "BUILD rc=$?"
