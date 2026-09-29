#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16
export OVERNIGHT_AGENT=opus-voices
cd /home/user/phonon
L=/data/phonon_term_eval_v1/logs
UVGPU="uv run --no-project --python 3.12 --with torch --with transformers --with accelerate --with peft --with jiwer --with whisper-normalizer --with rapidfuzz --with metaphone"

echo "=== REAL AUDIO all $(date +%T)"
$UVGPU python research/vocab_v1/real_audio_refine.py \
  --only vocab1_real_acoustic,date_lfm12_full_e2,vocab1_date --batch-size 24 \
  > "$L/real_audio.log" 2>&1
echo "=== REAL AUDIO rc=$? $(date +%T)"; tail -5 "$L/real_audio.log"
overnight-compute heartbeat --agent opus-voices --ttl 30m >/dev/null 2>&1
echo "REAL DONE $(date +%T)"
