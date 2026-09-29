#!/usr/bin/env bash
# Step 1: TTS the 900 held-out-term sentences with 3 UNSEEN Kokoro voices, then Parakeet ASR.
set -uo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8
cd /home/user/phonon
D=/data/phonon_term_eval_v1
L=$D/logs
mkdir -p "$L" "$D/clips"

echo "=== TTS $(date +%T)"
uv run --no-project --python 3.12 --with 'kokoro>=0.9.4' --with soundfile \
  --with numpy --with scipy --with torch \
  python research/term_eval_v0/tts_clips.py \
  --sentences /data/phonon_term_eval_v0/sentences.jsonl \
  --clips-dir "$D/clips" --manifest "$D/tts_manifest.jsonl" \
  --voices af_bella,am_michael,bm_george --workers 3 > "$L/tts.log" 2>&1
echo "=== TTS rc=$? $(date +%T)"
tail -3 "$L/tts.log"

echo "=== ASR $(date +%T)"
CUDA_VISIBLE_DEVICES=1 uv run python research/term_eval_v0/asr_clips.py \
  --manifest "$D/tts_manifest.jsonl" --out "$D/asr.jsonl" --batch-size 64 > "$L/asr.log" 2>&1
echo "=== ASR rc=$? $(date +%T)"
tail -3 "$L/asr.log"
overnight-compute heartbeat --agent opus-voices --ttl 30m >/dev/null 2>&1
echo "STEP1 TTSASR DONE $(date +%T)"
