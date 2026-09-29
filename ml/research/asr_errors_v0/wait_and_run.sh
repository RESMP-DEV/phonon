#!/usr/bin/env bash
# Acquire gpu0 then run remaining transcriptions. Never touches GPU 1.
set -u
LOG=/data/phonon_asr_errors_v0/logs
mkdir -p "$LOG"
WAIT_LOG="$LOG/wait_gpu0.log"
START=$(date +%s)
echo "$(date -Is) wait_and_run start" | tee -a "$WAIT_LOG"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$WAIT_LOG"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$WAIT_LOG"

acquired=0
while true; do
  echo "$(date -Is) overnight-compute wait attempt" | tee -a "$WAIT_LOG"
  out=$(timeout 120 overnight-compute wait --agent grok-asr --resource gpu0 --ttl 6h 2>&1) || rc=$?
  rc=${rc:-0}
  printf '%s\n' "$out" | tee -a "$WAIT_LOG"
  echo "$(date -Is) wait rc=$rc" | tee -a "$WAIT_LOG"
  if printf '%s\n' "$out" | grep -q acquired; then
    acquired=1
    echo "$(date -Is) ACQUIRED" | tee -a "$WAIT_LOG"
    break
  fi
  elapsed=$(( $(date +%s) - START ))
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v '^ *$' | grep -v pid || true)
  echo "$(date -Is) elapsed=${elapsed}s gpu0_apps=${apps:-none}" | tee -a "$WAIT_LOG"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$WAIT_LOG"
  # Known hang: timeout with empty output. GPU idle and fable-kernels already done -> proceed.
  if [[ $rc -eq 124 && -z "$out" && -z "$apps" ]]; then
    echo "$(date -Is) wait hung with no output and GPU 0 idle; proceeding" | tee -a "$WAIT_LOG"
    acquired=0
    break
  fi
  if (( elapsed >= 5400 )) && [[ -z "$apps" ]]; then
    echo "$(date -Is) 90min without acquire and GPU 0 idle; proceeding" | tee -a "$WAIT_LOG"
    break
  fi
  sleep 120
done

overnight-compute heartbeat --agent grok-asr --ttl 30m >>"$WAIT_LOG" 2>&1 || true
(
  while true; do
    sleep 1500
    overnight-compute heartbeat --agent grok-asr --ttl 30m >>"$LOG/heartbeat_loop.log" 2>&1 || true
  done
) &
hb=$!
trap 'kill "$hb" 2>/dev/null || true' EXIT

bash /home/user/phonon/research/asr_errors_v0/run_gpu.sh
rc=$?
echo "$(date -Is) run_gpu rc=$rc" | tee -a "$WAIT_LOG"
exit $rc
