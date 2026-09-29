#!/usr/bin/env bash
set -uo pipefail
cd /home/user/phonon
L=/data/phonon_pool3_v0/logs
bash research/pool3_v0/run_gen3.sh heldout 1 3 > $L/gen_heldout.log 2>&1 || { echo GEN_HELDOUT_FAIL; exit 1; }
echo "GEN_HELDOUT_DONE"
bash research/pool3_v0/run_gen3.sh pool 1 2 > $L/gen_pool.log 2>&1 || { echo GEN_POOL_FAIL; exit 1; }
echo "GEN_POOL_DONE"
