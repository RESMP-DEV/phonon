#!/usr/bin/env bash
set -uo pipefail
cd /home/user/phonon
for i in $(seq 1 160); do
  grep -q "ASRP p0 DONE" /data/phonon_pool2_v0/logs/asr_p0.log 2>/dev/null && \
  grep -q "ASRP p2 DONE" /data/phonon_pool2_v0/logs/asr_p2.log 2>/dev/null && break
  sleep 10
done
date +"%H:%M:%S ROWS START"
export OMP_NUM_THREADS=64
uv run --no-project --python 3.12 --with whisper-normalizer \
  python research/pool2_v0/step4a_pool.py || exit 1
date +"%H:%M:%S 4a done"
uv run --no-project --python 3.12 --with numpy --with rapidfuzz --with metaphone \
  python research/pool2_v0/step4b_lists.py || exit 1
date +"%H:%M:%S 4b done"
uv run --no-project --python 3.12 --with numpy --with rapidfuzz --with metaphone \
  python research/pool2_v0/step4c_mix.py || exit 1
date +"%H:%M:%S 4c done ROWS rc=0"
MS=$(( $(date -d "today 09:45" +%s) - $(date +%s) ))
[ "$MS" -lt 300 ] && MS=300
nohup bash research/pool2_v0/run_train2.sh 1 "$MS" > /data/phonon_pool2_v0/logs/train_xl.log 2>&1 &
date +"%H:%M:%S TRAIN LAUNCHED max_seconds=$MS"
