#!/usr/bin/env bash
set -uo pipefail
cd /home/user/phonon
for i in $(seq 1 600); do
  grep -q "TRAIN2 rc=" /data/phonon_pool2_v0/logs/train_xl.log 2>/dev/null && break
  sleep 20
done
date +"%H:%M:%S TRAIN FINISHED: $(grep -o 'TRAIN2 rc=[0-9]*' /data/phonon_pool2_v0/logs/train_xl.log | tail -1)"
if [ ! -s /data/phonon_corrector_v0/adapters/bigrun_xl_r16/adapter_config.json ]; then
  echo "NO ADAPTER; stopping"; exit 1
fi
bash research/pool2_v0/run_eval2.sh 0 bigrun_xl_r16 seen,unseen,new,pool2,real 32
date +"%H:%M:%S XL EVAL DONE"
uv run --no-project --python 3.12 python research/pool2_v0/throughput.py > /dev/null
uv run --no-project --python 3.12 --with jiwer --with whisper-normalizer --with numpy \
  python research/pool2_v0/analyze_pool2.py
uv run --no-project --python 3.12 python research/pool2_v0/render_pool2.py
date +"%H:%M:%S ANALYZE+RENDER DONE"
