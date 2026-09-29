#!/usr/bin/env bash
set -euo pipefail
R=/home/user/phonon/research/term_eval_v0
echo "=== scale TTS"
bash $R/run_tts.sh --sentences /data/phonon_synth_v3/sentences_v3.jsonl \
  --clips-dir /data/phonon_synth_v3/clips --manifest /data/phonon_synth_v3/tts_manifest.jsonl --workers 6
echo "=== scale ASR"
bash $R/run_asr.sh --manifest /data/phonon_synth_v3/tts_manifest.jsonl \
  --out /data/phonon_synth_v3/asr_v3.jsonl --batch-size 64
echo "SCALE DONE"
