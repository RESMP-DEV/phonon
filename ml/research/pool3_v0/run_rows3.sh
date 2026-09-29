#!/usr/bin/env bash
# waits for both ASR shards, then 4a/4b/4c
set -uo pipefail
cd /home/user/phonon
L=/data/phonon_pool3_v0/logs
for i in $(seq 1 480); do
  grep -q "ASR_POOL_DONE_0" $L/chain_pool_0.log 2>/dev/null && \
  grep -q "ASR_POOL_DONE_1" $L/chain_pool_1.log 2>/dev/null && break
  sleep 15
done
grep -q "ASR_POOL_DONE_1" $L/chain_pool_1.log 2>/dev/null || { echo "ROWS: asr not done"; exit 1; }
date +"%H:%M:%S ROWS START"
export OMP_NUM_THREADS=64
uv run --no-project --python 3.12 --with whisper-normalizer \
  python research/pool3_v0/step4a_pool3.py || exit 1
date +"%H:%M:%S 4a done"
uv run --no-project --python 3.12 --with numpy --with rapidfuzz --with metaphone \
  python research/pool3_v0/step4b_lists3.py || exit 1
date +"%H:%M:%S 4b done"
uv run --no-project --python 3.12 --with numpy --with rapidfuzz --with metaphone \
  python research/pool3_v0/step4c_mix3.py || exit 1
date +"%H:%M:%S 4c done ROWS rc=0"
