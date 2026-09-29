"""restraint_v0 eval: bigrun_v0/refine_bigrun.py with the oracle condition dropped.

The restraint question is precision under the shipping conditions, so the term sets are run in
`none` and `retrieved` only (oracle costs a third of the decode time and is not in the criterion).
Everything else - prompts, guard, metrics, per-row output format - is refine_bigrun's, imported.
"""
from __future__ import annotations
import sys

sys.path.insert(0, "/home/user/phonon/research/bigrun_v0")
import refine_bigrun  # noqa: E402

refine_bigrun.TERM_CONDS = ("none", "retrieved")

if __name__ == "__main__":
    if not any(a.startswith("--out-dir") for a in sys.argv):
        sys.argv += ["--out-dir", "/data/phonon_restraint_v0/refined"]
    raise SystemExit(refine_bigrun.main())
