#!/usr/bin/env bash
# per-GPU utilisation and memory every 20 s, for the queue-timing table
while true; do
  ts=$(date +%s)
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | sed "s/^/$ts, /"
  sleep 20
done
