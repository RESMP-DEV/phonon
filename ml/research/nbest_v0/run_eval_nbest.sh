#!/usr/bin/env bash
# nbest_v0: eval nbest_mid_r16 (new, unseen, real) with and without the alternatives line.
set -uo pipefail
GPU=${1:-0}
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=16
export PATH=$PATH:$HOME/.local/bin
cd /home/user/phonon
echo "=== EVAL nbest_mid_r16 gpu=$GPU $(date +%T)"
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  --with peft --with jiwer --with whisper-normalizer --with rapidfuzz --with metaphone \
  python research/nbest_v0/refine_nbest.py --batch-size 24 --real-batch-size 16 \
  --sets new,unseen,real --adapter nbest_mid_r16
echo "EVAL rc=$? $(date +%T)"
echo "EVAL_NBEST_DONE"
