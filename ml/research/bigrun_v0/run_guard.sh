#!/usr/bin/env bash
# bigrun_v0 length-guard eval: the three adapters on heldout_new + the real-audio holdout,
# unguarded into refined/ and guarded into refined_guard/. usage: run_guard.sh <gpu>
set -uo pipefail
GPU=${1:-0}
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=16
export PATH=$PATH:$HOME/.local/bin
cd /home/user/phonon
SPECS="--adapter bigrun_big_350m:lfm350 --adapter bigrun_mid_r16 --adapter bigrun_xl_r16"
UV="uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  --with peft --with jiwer --with whisper-normalizer --with rapidfuzz --with metaphone"
echo "=== UNGUARDED (fills any missing baseline) gpu=$GPU $(date +%T)"
$UV python research/bigrun_v0/refine_bigrun.py --sets new,real --batch-size 24 \
  --real-batch-size 16 $SPECS
echo "unguarded rc=$? $(date +%T)"
echo "=== GUARDED gpu=$GPU $(date +%T)"
$UV python research/bigrun_v0/refine_bigrun.py --sets new,real --batch-size 24 \
  --real-batch-size 16 --guard --out-dir /data/phonon_bigrun_v0/refined_guard $SPECS
echo "guarded rc=$? $(date +%T)"
echo "GUARD_EVAL_DONE"
