#!/usr/bin/env bash
# $1 = GPU index, $2..$N = adapter names (mix must already exist at mixes/train_<name>.jsonl)
set -uo pipefail
G=$1; shift
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8
cd /home/user/phonon
D=/data/phonon_scaling_v0
L=$D/logs
M=LiquidAI/LFM2.5-1.2B-Instruct

for NAME in "$@"; do
  T=$D/mixes/train_${NAME}.jsonl
  if [ ! -s "$T" ]; then echo "=== MISSING MIX $T"; continue; fi
  if [ -f /data/phonon_corrector_v0/adapters/${NAME}/adapter_config.json ]; then
    echo "=== TRAIN $NAME skipped (adapter exists) $(date +%T)"
  else
    echo "=== TRAIN $NAME gpu$G rows=$(wc -l < $T) $(date +%T)"
    uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
      --with peft --with trl --with datasets \
      python research/corrector_v0/train_lora.py --model "$M" --name "$NAME" --train "$T" \
      --epochs 2 > "$L/train_${NAME}.log" 2>&1
    echo "=== TRAIN $NAME rc=$? $(date +%T)"
    tail -1 "$L/train_${NAME}.log"
  fi
  overnight-compute heartbeat --agent opus-scaling --ttl 30m >/dev/null 2>&1
  echo "=== EVAL $NAME gpu$G $(date +%T)"
  uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
    --with peft --with jiwer --with whisper-normalizer \
    python research/scaling_v0/refine_scale.py --adapters "$NAME" --batch-size 64 \
    > "$L/eval_${NAME}.log" 2>&1
  echo "=== EVAL $NAME rc=$? $(date +%T)"
  tail -2 "$L/eval_${NAME}.log"
  overnight-compute heartbeat --agent opus-scaling --ttl 30m >/dev/null 2>&1
done
echo "CHAIN gpu$G DONE $(date +%T)"
