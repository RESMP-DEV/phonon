#!/usr/bin/env bash
# $1 = half / GPU. Waits for the pool jobs file before the pool stage.
set -uo pipefail
H=$1
cd /home/user/phonon
date +"%H:%M:%S CHAIN2 START half$H"
bash research/pool2_v0/run_tts2.sh heldout $H || exit 1
date +"%H:%M:%S TTS_HELDOUT_DONE half$H"
bash research/pool2_v0/run_asr2.sh heldout $H || exit 1
date +"%H:%M:%S ASR_HELDOUT_DONE half$H"
for i in $(seq 1 240); do
  [ -s /data/phonon_pool2_v0/jobs_pool_half${H}.jsonl ] && break
  sleep 15
done
date +"%H:%M:%S POOL_JOBS_READY half$H"
bash research/pool2_v0/run_tts2.sh pool $H || exit 1
date +"%H:%M:%S TTS_POOL_DONE half$H"
bash research/pool2_v0/run_asr2.sh pool $H || exit 1
date +"%H:%M:%S ASR_POOL_DONE half$H"
echo "CHAIN2 half$H rc=0"
