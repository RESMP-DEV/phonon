#!/usr/bin/env bash
set -uo pipefail
cd /home/user/phonon
uv run --no-project --python 3.12 python research/bigrun_v0/make_heldout_jobs.py || exit 1
bash research/bigrun_v0/run_tts.sh heldout 1 || exit 1
cp /data/phonon_bigrun_v0/jobs_heldout_half0.jsonl /data/phonon_bigrun_v0/jobs_heldoutb_half1.jsonl
sed -i 's/heldout_half1/heldoutb_half1/' /dev/null 2>/dev/null
bash research/bigrun_v0/run_tts.sh heldoutb 1 || exit 1
bash research/bigrun_v0/run_asr.sh heldout 1 || exit 1
bash research/bigrun_v0/run_asr.sh heldoutb 1 || exit 1
cat /data/phonon_bigrun_v0/asr_heldout_half1.jsonl /data/phonon_bigrun_v0/asr_heldoutb_half1.jsonl \
  > /data/phonon_bigrun_v0/heldout_new_asr.jsonl
wc -l /data/phonon_bigrun_v0/heldout_new_asr.jsonl
echo "HELDOUT CHAIN rc=0"
