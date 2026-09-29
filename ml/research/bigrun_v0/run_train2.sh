#!/usr/bin/env bash
# usage: run_train2.sh <gpu> <name> <train.jsonl> <epochs> <lr> [--init-adapter dir | --model X --lora-r R ...]
set -uo pipefail
GPU=$1; NAME=$2; TRAIN=$3; EPOCHS=$4; LR=$5; shift 5
export HF_HOME=/data/hf CUDA_VISIBLE_DEVICES=$GPU UV_TORCH_BACKEND=cu130
export HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH=$PATH:$HOME/.local/bin
cd /home/user/phonon
D=/data/phonon_corrector_v0
echo "=== TRAIN $NAME gpu=$GPU $(date +%T)"
rm -rf "$D/adapters/$NAME"
uv run --no-project --with torch --with transformers --with peft --with trl \
  --with accelerate --with datasets --with jiwer --with whisper-normalizer \
  python research/corrector_v0/train_lora.py --name "$NAME" --train "$TRAIN" \
  --epochs "$EPOCHS" --lr "$LR" "$@"
rc=$?
echo "TRAIN_EXIT $NAME $rc $(date +%T)"
rm -rf "$D/adapters/$NAME"/checkpoint-* "$D/adapters/$NAME/trainer"
overnight-compute heartbeat --agent opus-bigeval --ttl 30m >/dev/null 2>&1
echo "TRAIN2_DONE $NAME rc=$rc"
