#!/usr/bin/env bash
# $1 = half index / GPU
set -uo pipefail
H=$1
cd /home/user/phonon
date +"%H:%M:%S START half$H"
bash research/bigrun_v0/run_tts.sh mid $H || exit 1
date +"%H:%M:%S TTS_MID_DONE half$H"
bash research/bigrun_v0/run_asr.sh mid $H || exit 1
date +"%H:%M:%S ASR_MID_DONE half$H"
bash research/bigrun_v0/run_tts.sh big $H || exit 1
date +"%H:%M:%S TTS_BIG_DONE half$H"
bash research/bigrun_v0/run_asr.sh big $H || exit 1
date +"%H:%M:%S ASR_BIG_DONE half$H"
echo "CHAIN half$H rc=0"
