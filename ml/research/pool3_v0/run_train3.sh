#!/usr/bin/env bash
# $1 = GPU  $2 = max seconds
set -uo pipefail
G=$1; MS=${2:-0}
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8
export PATH=$PATH:$HOME/.local/bin
cd /home/user/phonon
D=/data/phonon_corrector_v0
NAME=bigrun_xxl_r16
rm -rf "$D/adapters/$NAME"
date +"%H:%M:%S TRAIN3 START rows=$(wc -l < /data/phonon_pool3_v0/train_pool3.jsonl) max_s=$MS"
uv run --no-project --with torch --with transformers --with peft --with trl \
  --with accelerate --with datasets --with jiwer --with whisper-normalizer \
  python research/corrector_v0/train_lora.py \
  --model LiquidAI/LFM2.5-1.2B-Instruct \
  --name "$NAME" \
  --train /data/phonon_pool3_v0/train_pool3.jsonl \
  --dev /data/phonon_corrector_v0/dev.jsonl \
  --epochs 1 --lr 1e-4 --batch-size 16 --max-seq-len 1024 --lora-dropout 0.05 \
  --init-adapter "$D/adapters/bigrun_xl_r16" \
  --max-seconds "$MS" --force
rc=$?
echo "TRAIN3 rc=$rc"
rm -rf "$D/adapters/$NAME"/checkpoint-* "$D/adapters/$NAME/trainer"
date +"%H:%M:%S TRAIN3 END"
[ "$rc" -eq 0 ] || exit 1
