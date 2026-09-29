"""pool2 eval step 2: refine_bigrun over the four term sets plus heldout_pool2 and real audio."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/bigrun_v0")
import refine_bigrun as RB  # noqa: E402

RB.TERM_SETS["pool2"] = Path("/data/phonon_pool2_v0/eval_conditions_pool2.jsonl")
RB.OUT = Path("/data/phonon_pool2_v0/refined")

if __name__ == "__main__":
    raise SystemExit(RB.main())
