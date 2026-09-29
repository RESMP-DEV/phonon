#!/usr/bin/env bash
# usage: run_bench_size.sh <gpu> <adapter-spec> [more specs...]
set -uo pipefail
GPU=$1; shift
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=16
export PATH=$PATH:$HOME/.local/bin
cd /home/user/phonon
ARGS=""
for s in "$@"; do ARGS="$ARGS --adapter $s"; done
echo "=== BENCH $* gpu=$GPU $(date +%T)"
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  --with peft --with jiwer --with whisper-normalizer --with rapidfuzz --with metaphone \
  python research/sizesweep_v0/bench_latency.py $ARGS
echo "BENCH rc=$? $(date +%T)"
