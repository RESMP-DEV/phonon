#!/usr/bin/env bash
# waits for rows, trains bigrun_xxl_r16 with a wall-clock cap, then eval + analyze + render.
set -uo pipefail
cd /home/user/phonon
L=/data/phonon_pool3_v0/logs
export PATH=$PATH:$HOME/.local/bin
for i in $(seq 1 480); do
  grep -qa "ROWS rc=0" $L/rows.log 2>/dev/null && break
  grep -qa "ROWS: asr not done" $L/rows.log 2>/dev/null && { echo "ORCH: rows failed"; exit 1; }
  sleep 15
done
grep -qa "ROWS rc=0" $L/rows.log 2>/dev/null || { echo "ORCH: rows never finished"; exit 1; }
# training must be over by 01:00 so the eval and the write-up fit before 02:30
MS=$(( $(date -d "tomorrow 01:00" +%s) - $(date +%s) ))
[ "$MS" -gt 9000 ] && MS=9000
[ "$MS" -lt 600 ] && MS=600
date +"%H:%M:%S ORCH launching training max_seconds=$MS"
bash research/pool3_v0/run_train3.sh 1 "$MS" > $L/train.log 2>&1
echo "TRAIN_CHAIN rc=$?"
date +"%H:%M:%S ORCH launching eval"
bash research/pool3_v0/run_eval3.sh 1 --adapter bigrun_xxl_r16 --adapter bigrun_xl_r16 > $L/eval.log 2>&1
echo "EVAL rc=$?"
export OMP_NUM_THREADS=32
uv run --no-project --python 3.12 python research/pool3_v0/throughput3.py > $L/throughput.log 2>&1
echo "THROUGHPUT rc=$?"
uv run --no-project --python 3.12 --with jiwer --with whisper-normalizer --with rapidfuzz \
  --with metaphone python research/pool3_v0/analyze_pool3.py > $L/analyze.log 2>&1
echo "ANALYZE rc=$?"
uv run --no-project --python 3.12 python research/pool3_v0/render_pool3.py > $L/render.log 2>&1
echo "RENDER rc=$?"
date +"%H:%M:%S ORCH END"
