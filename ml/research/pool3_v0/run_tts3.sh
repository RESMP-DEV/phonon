#!/usr/bin/env bash
# $1 = tag (pool|heldout)  $2 = shard  $3 = gpu  $4 = workers
set -uo pipefail
S=$1; H=$2; G=$3; W=${4:-7}
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4
cd /home/user/phonon
date +"%H:%M:%S TTS3 ${S} s${H} START workers=$W"
uv run --no-project --python 3.12 --with 'kokoro>=0.9.4' --with soundfile \
  --with numpy --with scipy --with torch \
  python research/bigrun_v0/tts_jobs.py \
  --jobs /data/phonon_pool3_v0/jobs_${S}_s${H}.jsonl \
  --clips-dir /data/phonon_pool3_v0/clips \
  --manifest /data/phonon_pool3_v0/tts_manifest_${S}_s${H}.jsonl \
  --shard-dir /data/phonon_pool3_v0/tts_shards \
  --workers $W
rc=$?
echo "TTS3 ${S} s${H} rc=$rc"
date +"%H:%M:%S TTS3 ${S} s${H} END"
[ "$rc" -eq 0 ] || exit 1
