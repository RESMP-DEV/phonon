#!/usr/bin/env bash
# $1 = half index (0/1) -> GPU $1
set -uo pipefail
H=$1
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$H OMP_NUM_THREADS=4
cd /home/user/phonon
uv run --no-project --python 3.12 --with 'kokoro>=0.9.4' --with soundfile \
  --with numpy --with scipy --with torch \
  python research/term_eval_v0/tts_clips.py \
  --sentences /data/phonon_scaling_v0/sentences_8_half${H}.jsonl \
  --clips-dir /data/phonon_scaling_v0/clips \
  --manifest /data/phonon_scaling_v0/tts_manifest_half${H}.jsonl \
  --voices af_heart,am_adam,bf_emma,af_bella,am_michael,bm_george \
  --workers 7
echo "TTS half${H} rc=$?"
