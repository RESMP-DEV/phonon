#!/usr/bin/env bash
set -uo pipefail
cd /home/user/phonon
for i in $(seq 1 160); do
  [ -s /data/phonon_pool2_v0/asr_heldout_half0.jsonl ] && \
  [ -s /data/phonon_pool2_v0/asr_heldout_half1.jsonl ] && break
  sleep 15
done
date +"%H:%M:%S BUILD POOL2 CONDITIONS"
export OMP_NUM_THREADS=32
uv run --no-project --python 3.12 --with numpy --with rapidfuzz --with metaphone \
  python research/pool2_v0/build_pool2_conditions.py
echo "COND rc=$?"
date +"%H:%M:%S COND END"
