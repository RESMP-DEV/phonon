#!/usr/bin/env bash
set -uo pipefail
cd /home/user/phonon
for i in $(seq 1 120); do
  [ -s /data/phonon_pool2_v0/refined/bigrun_big_r16__pool2.jsonl ] && break
  sleep 15
done
date +"%H:%M:%S MID EVAL (parent adapter) on gpu1"
bash research/pool2_v0/run_eval2.sh 1 bigrun_mid_r16 seen,unseen,new,pool2,real 32
