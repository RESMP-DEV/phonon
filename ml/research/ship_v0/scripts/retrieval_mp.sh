#!/usr/bin/env bash
set -uo pipefail
export PYTHONUNBUFFERED=1
/data/venvs/phonon_bench/bin/python /data/phonon_ship_v0/scripts/retrieval_mp.py
