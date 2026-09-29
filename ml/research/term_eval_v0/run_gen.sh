#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8
cd /home/user/phonon
UV="uv run --no-project --python 3.12 --with torch --with transformers --with accelerate python research/term_eval_v0/gen_sentences.py"
echo "=== eval sentences (300 terms x3)"
$UV --terms /data/phonon_term_eval_v0/terms_heldout.jsonl \
    --out /data/phonon_term_eval_v0/sentences.jsonl \
    --per-term 3 --batch-size 48 --rounds 8 --seed 17
echo "=== scale sentences (2175 terms x2)"
$UV --terms /data/phonon_synth_v3/terms_scale.jsonl \
    --out /data/phonon_synth_v3/sentences_v3.jsonl \
    --per-term 2 --batch-size 48 --rounds 8 --seed 31
echo "GEN ALL DONE"
