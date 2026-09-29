#!/usr/bin/env bash
# Steps 1b-3: conditions for the unseen-voice clips, refiner inference, real-audio run,
# vocab1_date build+train, real-audio run for vocab1_date.
set -uo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16
export OVERNIGHT_AGENT=opus-voices
cd /home/user/phonon
L=/data/phonon_term_eval_v1/logs
mkdir -p "$L" /data/phonon_term_eval_v1/refined /data/phonon_term_eval_v1/real

hb () { overnight-compute heartbeat --agent opus-voices --ttl 30m >/dev/null 2>&1; }
UVCPU="uv run --no-project --python 3.12 --with numpy --with rapidfuzz --with metaphone"
UVGPU="uv run --no-project --python 3.12 --with torch --with transformers --with accelerate --with peft --with jiwer --with whisper-normalizer"

echo "=== BUILD CONDITIONS $(date +%T)"
$UVCPU python research/vocab_v1/build_voices_conditions.py > "$L/build_cond.log" 2>&1
echo "=== BUILD CONDITIONS rc=$? $(date +%T)"; tail -3 "$L/build_cond.log"; hb

echo "=== BUILD DATE TRAIN $(date +%T)"
$UVCPU python research/vocab_v1/build_vocab1_date.py > "$L/build_date.log" 2>&1
echo "=== BUILD DATE TRAIN rc=$? $(date +%T)"; tail -3 "$L/build_date.log"; hb

for s in vocab1_real_acoustic vocab1_real_acoustic_350m vocab_real_acoustic lfm2.5-1.2b_personal; do
  echo "=== VOICES EVAL $s $(date +%T)"
  $UVGPU python research/vocab_v1/refine_vocab1.py --only "$s" --batch-size 48 \
    --skip-real --term-file /data/phonon_term_eval_v1/eval_conditions.jsonl \
    --out-dir /data/phonon_term_eval_v1/refined > "$L/eval_$s.log" 2>&1
  echo "=== VOICES EVAL $s rc=$? $(date +%T)"; tail -2 "$L/eval_$s.log"; hb
done

echo "=== REAL AUDIO (v1 + date adapter) $(date +%T)"
$UVGPU python research/vocab_v1/real_audio_refine.py --only vocab1_real_acoustic,date_lfm12_full_e2 \
  --batch-size 24 > "$L/real_audio.log" 2>&1
echo "=== REAL AUDIO rc=$? $(date +%T)"; tail -3 "$L/real_audio.log"; hb

echo "=== TRAIN vocab1_date $(date +%T)"
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  --with peft --with trl --with datasets \
  python research/corrector_v0/train_lora.py --model LiquidAI/LFM2.5-1.2B-Instruct \
  --name vocab1_date --train /data/phonon_vocab_v1/train_vocab1_date.jsonl --epochs 2 \
  > "$L/train_vocab1_date.log" 2>&1
echo "=== TRAIN vocab1_date rc=$? $(date +%T)"; tail -3 "$L/train_vocab1_date.log"; hb

echo "=== REAL AUDIO vocab1_date $(date +%T)"
$UVGPU python research/vocab_v1/real_audio_refine.py --only vocab1_date --batch-size 24 \
  > "$L/real_audio_date.log" 2>&1
echo "=== REAL AUDIO vocab1_date rc=$? $(date +%T)"; tail -3 "$L/real_audio_date.log"; hb

echo "CHAIN DONE $(date +%T)"
