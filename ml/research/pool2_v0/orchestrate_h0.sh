#!/usr/bin/env bash
set -uo pipefail
cd /home/user/phonon
for i in $(seq 1 120); do
  grep -q "GEN2 half0 END" /data/phonon_pool2_v0/logs/gen2_h0.log 2>/dev/null && break
  sleep 15
done
date +"%H:%M:%S GEN2 both done; building pool jobs"
uv run --no-project --python 3.12 python research/pool2_v0/make_jobs2.py --which pool || exit 1
date +"%H:%M:%S POOL JOBS BUILT"
nohup bash research/pool2_v0/run_chain2.sh 0 > /data/phonon_pool2_v0/logs/chain2_h0.log 2>&1 &
echo "chain2 half0 launched"
