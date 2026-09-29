#!/usr/bin/env bash
# $1 = shard index. TTS then ASR for that shard, GPU 1.
set -uo pipefail
H=$1
cd /home/user/phonon
L=/data/phonon_pool3_v0/logs
bash research/pool3_v0/run_tts3.sh pool $H 1 12 > $L/tts_pool_s$H.log 2>&1 || { echo "TTS_FAIL_$H"; exit 1; }
echo "TTS_POOL_DONE_$H"
bash research/pool3_v0/run_asr3.sh pool $H 1 96 > $L/asr_pool_s$H.log 2>&1 || { echo "ASR_FAIL_$H"; exit 1; }
echo "ASR_POOL_DONE_$H"
