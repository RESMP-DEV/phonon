"""Where the bench lives. One resolution so every module agrees.

`PHONON_DATA_ROOT` moves the whole /data tree (default `/data`, the GPU box
layout); `PHONON_BENCH_ROOT` moves only the bench directory and wins over it.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA = Path(os.environ.get("PHONON_DATA_ROOT", "/data"))
BENCH = Path(os.environ.get("PHONON_BENCH_ROOT", str(DATA / "phonon_bench_v0")))
HF = Path(os.environ.get("PHONON_HF_HOME", str(DATA / "hf")))
