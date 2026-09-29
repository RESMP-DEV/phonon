#!/usr/bin/env bash
# Pool-3 held-out sentences: 3 per term for the 300 terms in terms_heldout_pool3.jsonl.
# NOT RUN by opus-index (CPU-only lease, both GPUs were taken).  $1 = GPU index, default 1.
# Afterwards: research/index_v0/make_pool3_jobs.py -> TTS on the 3 unseen voices, as in
# research/bigrun_v0/make_heldout_jobs.py + run_tts.sh.
set -uo pipefail
G=${1:-1}
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8
cd /home/user/phonon
date +"%H:%M:%S GEN_POOL3 START gpu=$G"
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  python research/scaling_v0/gen_more.py \
  --terms /data/phonon_pool3_v0/terms_heldout_pool3.jsonl \
  --out /data/phonon_pool3_v0/sentences_heldout_pool3.jsonl \
  --per-term 3 --batch-size 100 --rounds 10 --seed 3918 --agent opus-index
echo "GEN_POOL3 rc=$?"
date +"%H:%M:%S GEN_POOL3 END"
