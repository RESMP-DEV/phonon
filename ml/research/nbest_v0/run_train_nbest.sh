#!/usr/bin/env bash
# nbest_v0: LFM2.5-1.2B LoRA on train_mid_nbest.jsonl, bigrun_mid_r16 settings.
set -uo pipefail
GPU=${1:-0}
cd /home/user/phonon
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 UV_TORCH_BACKEND=cu130
export CUDA_VISIBLE_DEVICES=$GPU PATH=$PATH:$HOME/.local/bin
D=/data/phonon_corrector_v0
NAME=nbest_mid_r16
echo "=== TRAIN $NAME gpu=$GPU $(date +%T) rows=$(wc -l < /data/phonon_nbest_v0/train_mid_nbest.jsonl)"
rm -rf "$D/adapters/$NAME"
uv run --no-project --with torch --with transformers --with peft --with trl \
  --with accelerate --with datasets --with jiwer --with whisper-normalizer \
  python research/corrector_v0/train_lora.py --model LiquidAI/LFM2.5-1.2B-Instruct \
  --name $NAME --train /data/phonon_nbest_v0/train_mid_nbest.jsonl --epochs 2 --lora-r 16
rc=$?
echo "TRAIN_EXIT $NAME $rc $(date +%T)"
rm -rf "$D/adapters/$NAME"/checkpoint-* "$D/adapters/$NAME/trainer"
overnight-compute heartbeat --agent opus-nbest --ttl 30m >/dev/null 2>&1
echo "NBEST_TRAIN_DONE $NAME rc=$rc"
