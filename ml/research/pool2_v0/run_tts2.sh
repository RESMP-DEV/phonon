#!/usr/bin/env bash
# $1 = tag (pool|heldout)  $2 = half -> GPU
set -uo pipefail
S=$1; H=$2
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$H OMP_NUM_THREADS=4
cd /home/user/phonon
uv run --no-project --python 3.12 --with 'kokoro>=0.9.4' --with soundfile \
  --with numpy --with scipy --with torch \
  python research/bigrun_v0/tts_jobs.py \
  --jobs /data/phonon_pool2_v0/jobs_${S}_half${H}.jsonl \
  --clips-dir /data/phonon_pool2_v0/clips \
  --manifest /data/phonon_pool2_v0/tts_manifest_${S}_half${H}.jsonl \
  --shard-dir /data/phonon_pool2_v0/tts_shards \
  --workers 7
echo "TTS2 ${S} half${H} rc=$?"
