#!/usr/bin/env bash
# vocab_v1 adapters, sequentially, on GPU 1.
set -uo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8
export OVERNIGHT_AGENT=opus-vocab1
cd /home/user/phonon
D=/data/phonon_vocab_v1
L=$D/logs
mkdir -p "$L"
T=$D/train_vocab1_real_acoustic.jsonl

run () {  # name model
  echo "=== TRAIN $1 $(date +%T)"
  uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
    --with peft --with trl --with datasets \
    python research/corrector_v0/train_lora.py --model "$2" --name "$1" --train "$T" --epochs 2 \
    > "$L/train_$1.log" 2>&1
  echo "=== TRAIN $1 rc=$? $(date +%T)"
  tail -2 "$L/train_$1.log"
  overnight-compute heartbeat --agent opus-vocab1 --ttl 30m >/dev/null 2>&1
}

run vocab1_real_acoustic      LiquidAI/LFM2.5-1.2B-Instruct
run vocab1_real_acoustic_350m LiquidAI/LFM2.5-350M
echo "ALL TRAIN DONE $(date +%T)"
