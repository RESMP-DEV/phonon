#!/usr/bin/env bash
set -uo pipefail
cd /home/user/phonon
export OMP_NUM_THREADS=64
date +"%H:%M:%S ROWS START"
uv run --no-project --python 3.12 --with whisper-normalizer \
  python research/bigrun_v0/step5a_pool.py || exit 1
date +"%H:%M:%S 5a done"
uv run --no-project --python 3.12 --with numpy --with rapidfuzz --with metaphone \
  python research/bigrun_v0/step5b_lists.py || exit 1
date +"%H:%M:%S 5b done"
uv run --no-project --python 3.12 --with numpy --with rapidfuzz --with metaphone \
  python research/bigrun_v0/step5c_mix.py || exit 1
date +"%H:%M:%S 5c done"
echo "ROWS rc=0"
