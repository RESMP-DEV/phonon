#!/usr/bin/env bash
# Sequential GPU transcription of missing hypotheses on CUDA device 0.
set -u
cd /home/user/phonon
export HF_HOME=/data/hf
export HF_HUB_CACHE=/data/hf/hub
export HUGGINGFACE_HUB_CACHE=/data/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UV_TORCH_BACKEND=cu130
export TOKENIZERS_PARALLELISM=false
export OVERNIGHT_AGENT=grok-asr
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/user/phonon/research/asr_errors_v0:/home/user/phonon/scripts/option0:/home/user/phonon/src${PYTHONPATH:+:$PYTHONPATH}"
LOG=/data/phonon_asr_errors_v0/logs
FAIL=/data/phonon_asr_errors_v0/logs/failures.jsonl
HYPS=/data/phonon_asr_errors_v0/hyps
CLIPS=/data/phonon_asr_errors_v0/clips.jsonl
V3_NEMO=/data/hf/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.nemo
# Hub snapshot for granite-5 is incomplete (processor json only). Latent extract kept a full copy.
GRANITE5=/data/phonon_latent_v0/models/granite-speech-5.0-470m-turboctc
mkdir -p "$LOG"

missing_count() {
  local tag=$1
  python3 - "$CLIPS" "$HYPS/${tag}.jsonl" <<'PY'
import json, sys
from pathlib import Path
wanted = {json.loads(line)["id"] for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()}
path = Path(sys.argv[2])
have = set()
if path.exists():
    have = {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}
print(len(wanted - have))
PY
}

common_tf=(--with torch==2.14.0 --with accelerate --with soundfile --with scipy
           --with pyarrow --with jiwer --with librosa --with sentencepiece --with protobuf)

run_nemo() {
  local model=$1 tag=$2 bs=${3:-16}
  local miss
  miss=$(missing_count "$tag")
  local log="$LOG/transcribe_${tag}.log"
  if [[ "$miss" == "0" ]]; then
    echo "$(date -Is) SKIP $tag already complete" | tee -a "$log"
    return 0
  fi
  echo "$(date -Is) START nemo $tag missing=$miss CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$log"
  overnight-compute heartbeat --agent grok-asr --ttl 30m >>"$log" 2>&1 || true
  if cd /home/user/phonon && HF_HOME=/data/hf CUDA_VISIBLE_DEVICES=0 UV_TORCH_BACKEND=cu130 \
      PYTHONUNBUFFERED=1 uv run python research/asr_errors_v0/transcribe_manifest.py \
        --family nemo --model "$model" --tag "$tag" --batch-size "$bs" \
        >>"$log" 2>&1; then
    echo "$(date -Is) DONE $tag" | tee -a "$log"
    return 0
  fi
  echo "$(date -Is) FAIL $tag" | tee -a "$log"
  printf '{"tag":"%s","family":"nemo","error":"see %s"}\n' "$tag" "$log" >>"$FAIL"
  return 1
}

run_tf() {
  local family=$1 model=$2 tag=$3 bs=$4
  shift 4
  local miss
  miss=$(missing_count "$tag")
  local log="$LOG/transcribe_${tag}.log"
  if [[ "$miss" == "0" ]]; then
    echo "$(date -Is) SKIP $tag already complete" | tee -a "$log"
    return 0
  fi
  echo "$(date -Is) START $family $tag missing=$miss CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$log"
  overnight-compute heartbeat --agent grok-asr --ttl 30m >>"$log" 2>&1 || true
  if cd /home/user/phonon && HF_HOME=/data/hf CUDA_VISIBLE_DEVICES=0 UV_TORCH_BACKEND=cu130 \
      PYTHONUNBUFFERED=1 uv run --no-project --isolated --python 3.12 "${common_tf[@]}" "$@" python \
        research/asr_errors_v0/transcribe_manifest.py \
        --family "$family" --model "$model" --tag "$tag" --batch-size "$bs" \
        >>"$log" 2>&1; then
    echo "$(date -Is) DONE $tag" | tee -a "$log"
    return 0
  fi
  echo "$(date -Is) FAIL $tag" | tee -a "$log"
  printf '{"tag":"%s","family":"%s","error":"see %s"}\n' "$tag" "$family" "$log" >>"$FAIL"
  return 1
}

echo "$(date -Is) GPU runner start device=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG/run_gpu.log"
nvidia-smi | tee -a "$LOG/run_gpu.log"

# Fast remaining jobs first so error_model.json can be refreshed before granite-4.1.
run_nemo "$V3_NEMO" parakeet-tdt-0.6b-v3 16
run_tf transformers-qwen3asr Qwen/Qwen3-ASR-1.7B-hf Qwen3-ASR-1.7B-hf 2 --with 'transformers==5.17.0'
run_tf transformers-granite-ctc "$GRANITE5" granite-speech-5.0-470m-turboctc 4 \
  --with 'transformers==5.17.0' --with torchaudio
run_tf transformers-granite ibm-granite/granite-speech-4.1-2b granite-speech-4.1-2b 1 \
  --with 'transformers==5.17.0' --with peft --with torchaudio==2.11.0

echo "$(date -Is) GPU runner finished" | tee -a "$LOG/run_gpu.log"
nvidia-smi | tee -a "$LOG/run_gpu.log"
