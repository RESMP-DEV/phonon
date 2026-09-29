#!/usr/bin/env bash
set -u
cd /home/user/phonon
export HF_HOME=/data/hf CUDA_VISIBLE_DEVICES=0
export TMPDIR=/data/phonon_latent_v0/tmp
export MPLCONFIGDIR=/data/phonon_latent_v0/matplotlib
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
base=/home/user/phonon/research/latent_v0
nvidia-smi > "$base/gpu_preflight.log"
overnight-compute wait --agent codex-latent --poll 5s > "$base/lease.log" 2>&1
if ! rg -q '^acquired' "$base/lease.log"; then
  cat "$base/lease.log"
  exit 1
fi
nvidia-smi >> "$base/gpu_preflight.log"
(while sleep 180; do overnight-compute heartbeat --agent codex-latent --ttl 30m >> "$base/lease.log"; done) &
heartbeat_pid=$!
trap 'kill "$heartbeat_pid" 2>/dev/null; overnight-compute release --agent codex-latent --status done >> "$base/lease.log" 2>&1' EXIT
common=(--with torch==2.14.0 --with accelerate --with soundfile --with scipy --with pyarrow
        --with jiwer --with librosa --with sentencepiece --with protobuf --with transformers==5.17.0
        --with peft --with torchaudio==2.11.0)
for tag in parakeet-tdt-0.6b-v2 parakeet-unified-en-0.6b Qwen3-ASR-0.6B-hf cohere-transcribe-03-2026 granite-speech-5.0-470m-turboctc granite-speech-4.1-2b; do
  overnight-compute wait --agent codex-latent --poll 5s >> "$base/lease.log" 2>&1
  overnight-compute heartbeat --agent codex-latent --ttl 30m >> "$base/lease.log"
  command=(uv run --no-sync python)
  if [[ "$tag" != parakeet* ]]; then
    command=(uv run --no-project --isolated "${common[@]}" python)
  fi
  echo "starting $tag $(date -Is)"
  "${command[@]}" "$base/extract.py" --encoder "$tag" > "$base/extract_${tag}_attempt1.log" 2>&1
  result=$?
  echo "$tag attempt1 exit=$result $(date -Is)"
  if [[ "$result" != 0 ]]; then
    # Parent reviews exact traceback and writes retry.ready after a targeted fix.
    echo "needs reviewed retry: $tag"
    while [[ ! -f "$base/retry_${tag}.ready" ]]; do
      sleep 15
      overnight-compute heartbeat --agent codex-latent --ttl 30m >> "$base/lease.log"
    done
    overnight-compute wait --agent codex-latent --poll 5s >> "$base/lease.log" 2>&1
    "${command[@]}" "$base/extract.py" --encoder "$tag" > "$base/extract_${tag}_attempt2.log" 2>&1
    echo "$tag attempt2 exit=$? $(date -Is)"
  fi
done
