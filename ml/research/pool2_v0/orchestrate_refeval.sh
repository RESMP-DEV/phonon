#!/usr/bin/env bash
set -uo pipefail
cd /home/user/phonon
for i in $(seq 1 120); do
  grep -q "ASR_POOL_DONE half1" /data/phonon_pool2_v0/logs/chain2_h1.log 2>/dev/null && break
  sleep 15
done
date +"%H:%M:%S REF EVAL pool2 on gpu1"
bash research/pool2_v0/run_eval2.sh 1 bigrun_big_r16 pool2 32
