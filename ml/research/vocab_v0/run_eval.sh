#!/usr/bin/env bash
# Eval each adapter in 3 vocabulary conditions as soon as it is trained. GPU 1.
set -uo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8
cd /home/user/phonon
A=/data/phonon_corrector_v0/adapters
L=/data/phonon_vocab_v0/logs

wait_for () {  # adapter-dir-name, max wait seconds
  local n=0
  until [ -f "$A/$1/adapter_config.json" ]; do
    n=$((n+15)); [ $n -ge "$2" ] && { echo "TIMEOUT waiting for $1"; return 1; }
    sleep 15
  done
  sleep 5
  return 0
}

evalone () {  # label adapterdir
  wait_for "$2" 3600 || return 1
  echo "=== EVAL $1 $(date +%T)"
  uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
    --with peft --with jiwer --with whisper-normalizer \
    python research/vocab_v0/refine_vocab.py --only "$1" --batch-size 48 > "$L/eval_$1.log" 2>&1
  echo "=== EVAL $1 rc=$? $(date +%T)"
  overnight-compute heartbeat --agent opus-vocab --ttl 30m >/dev/null 2>&1
}

evalone lfm2.5-1.2b_personal      lfm2.5-1.2b
evalone vocab_real                vocab_real
evalone vocab_real_acoustic       vocab_real_acoustic
evalone novocab_real_acoustic     vocab_novocab_real_acoustic
evalone vocab_real_acoustic_350m  vocab_real_acoustic_350m
echo "ALL EVAL DONE $(date +%T)"
