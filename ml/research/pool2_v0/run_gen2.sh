#!/usr/bin/env bash
# $1 = pool|heldout  $2 = gpu
set -uo pipefail
W=$1; G=$2
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8
cd /home/user/phonon
if [ "$W" = "pool" ]; then
  T=/data/phonon_pool2_v0/terms_pool2.jsonl; O=/data/phonon_pool2_v0/sentences_4.jsonl; N=4; S=2918
else
  T=/data/phonon_pool2_v0/terms_heldout_pool2.jsonl; O=/data/phonon_pool2_v0/sentences_heldout_pool2.jsonl; N=3; S=2919
fi
date +"%H:%M:%S GEN2 $W START"
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  python research/scaling_v0/gen_more.py \
  --terms $T --out $O --per-term $N --batch-size 128 --rounds 10 --seed $S --agent opus-pool2
echo "GEN2 $W rc=$?"
date +"%H:%M:%S GEN2 $W END"
