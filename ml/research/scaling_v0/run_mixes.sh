#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/data/hf TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=32
cd /home/user/phonon
UV="uv run --no-project --python 3.12 --with numpy --with rapidfuzz --with metaphone python"
$UV research/scaling_v0/cache_lists.py || exit 1
for N in 0 500 1000 2000 4000 9000 18000 36000; do
  $UV research/scaling_v0/build_mix.py --spec size:$N --name scale_n${N}
done
for A in terms sentences voices; do
  $UV research/scaling_v0/build_mix.py --spec axis:$A --name scale_axis_${A}
done
echo "MIXES DONE"
