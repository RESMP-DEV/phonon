#!/usr/bin/env bash
# $1 = GPU  $2... = adapter specs
set -uo pipefail
G=$1; shift
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=16
export PATH=$PATH:$HOME/.local/bin
cd /home/user/phonon
SETS=${SETS:-seen,unseen,new,pool2,pool3,real}
date +"%H:%M:%S EVAL3 gpu=$G sets=$SETS specs=$*"
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  --with peft --with jiwer --with whisper-normalizer --with rapidfuzz --with metaphone \
  python research/pool3_v0/refine_pool3.py --sets "$SETS" --guard \
  --batch-size 32 --real-batch-size 16 --out-dir /data/phonon_pool3_v0/refined "$@"
rc=$?
echo "EVAL3 rc=$rc"
date +"%H:%M:%S EVAL3 END"
[ "$rc" -eq 0 ] || exit 1
