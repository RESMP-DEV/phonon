#!/usr/bin/env bash
# usage: run_eval.sh <gpu> <batch> <realbatch> <adapter-spec> [more specs...]
set -uo pipefail
GPU=$1; BS=$2; RBS=$3; shift 3
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=16
export PATH=$PATH:$HOME/.local/bin
cd /home/user/phonon
ARGS=""
for s in "$@"; do ARGS="$ARGS --adapter $s"; done
echo "=== EVAL $* gpu=$GPU bs=$BS $(date +%T)"
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  --with peft --with jiwer --with whisper-normalizer --with rapidfuzz --with metaphone \
  python research/bigrun_v0/refine_bigrun.py --batch-size "$BS" --real-batch-size "$RBS" $ARGS
echo "EVAL rc=$? $(date +%T)"
overnight-compute heartbeat --agent opus-bigeval --ttl 30m >/dev/null 2>&1
echo "EVAL_DONE"
