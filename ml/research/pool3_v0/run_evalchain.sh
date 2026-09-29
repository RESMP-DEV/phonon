#!/usr/bin/env bash
# waits for training, then refine + analyze + render. $1 = gpu
set -uo pipefail
G=${1:-1}
cd /home/user/phonon
L=/data/phonon_pool3_v0/logs
for i in $(seq 1 720); do
  grep -qa "TRAIN3 rc=" $L/train.log 2>/dev/null && break
  sleep 15
done
date +"%H:%M:%S EVALCHAIN START"
bash research/pool3_v0/run_eval3.sh $G --adapter bigrun_xxl_r16 --adapter bigrun_xl_r16 \
  > $L/eval.log 2>&1
echo "EVAL rc=$?"
export OMP_NUM_THREADS=32
uv run --no-project --python 3.12 --with jiwer --with whisper-normalizer --with rapidfuzz \
  --with metaphone python research/pool3_v0/throughput3.py > $L/throughput.log 2>&1
uv run --no-project --python 3.12 --with jiwer --with whisper-normalizer --with rapidfuzz \
  --with metaphone python research/pool3_v0/analyze_pool3.py > $L/analyze.log 2>&1
echo "ANALYZE rc=$?"
uv run --no-project --python 3.12 python research/pool3_v0/render_pool3.py > $L/render.log 2>&1
echo "RENDER rc=$?"
date +"%H:%M:%S EVALCHAIN END"
