#!/usr/bin/env bash
# vocab_v1 inference: 4 systems x (2700 term clips x 3 conds + 2 real holdouts x 2 conds). GPU 1.
set -uo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8
cd /home/user/phonon
L=/data/phonon_vocab_v1/logs

for s in vocab1_real_acoustic vocab1_real_acoustic_350m vocab_real_acoustic lfm2.5-1.2b_personal; do
  echo "=== EVAL $s $(date +%T)"
  uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
    --with peft --with jiwer --with whisper-normalizer \
    python research/vocab_v1/refine_vocab1.py --only "$s" --batch-size 48 > "$L/eval_$s.log" 2>&1
  echo "=== EVAL $s rc=$? $(date +%T)"
  tail -2 "$L/eval_$s.log"
  overnight-compute heartbeat --agent opus-vocab1 --ttl 30m >/dev/null 2>&1
done
echo "ALL EVAL DONE $(date +%T)"
