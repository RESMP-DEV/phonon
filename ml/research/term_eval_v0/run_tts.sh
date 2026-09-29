#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
cd /home/user/phonon
exec uv run --no-project --python 3.12 --with 'kokoro>=0.9.4' --with soundfile \
  --with numpy --with scipy --with torch \
  python research/term_eval_v0/tts_clips.py "$@"
