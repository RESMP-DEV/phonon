#!/usr/bin/env bash
# $1 = pool|heldout  $2 = gpu  $3 = per-term
set -uo pipefail
W=$1; G=$2; N=$3
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8
export PATH=$PATH:$HOME/.local/bin
cd /home/user/phonon
D=/data/phonon_pool3_v0
if [ "$W" = "pool" ]; then
  T=$D/terms_pool3.jsonl; O=$D/sentences_pool3.jsonl; S=3918
else
  T=$D/terms_heldout_pool3.jsonl; O=$D/sentences_heldout_pool3.jsonl; S=3919
fi
date +"%H:%M:%S GEN3 $W START n=$N"
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  python research/scaling_v0/gen_more.py \
  --terms $T --out $O --per-term $N --batch-size 128 --rounds 10 --seed $S --agent opus-pool3
rc=$?
echo "GEN3 $W rc=$rc"
date +"%H:%M:%S GEN3 $W END"
[ "$rc" -eq 0 ] || exit 1
