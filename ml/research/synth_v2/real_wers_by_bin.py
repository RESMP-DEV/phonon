"""CPU: fair WER by length bin on real Wispr pairs. Writes real_wers_by_bin.json."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BIN_NAMES, BINS, DATA_ROOT, REAL_PAIRS, REAL_WERS_BY_BIN  # noqa: E402

_spec_path = Path("/home/user/phonon/research/corrector_v0/common.py")
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("corrector_v0_common", _spec_path)
assert _spec and _spec.loader
_common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_common)
fair_norm = _common.fair_norm
pair_wer = _common.pair_wer


def bin_name(n: int) -> str:
    for (a, b), name in zip(BINS, BIN_NAMES, strict=True):
        if a <= n <= b:
            return name
    return "0"


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    by_bin: dict[str, list[float]] = defaultdict(list)
    all_w = []
    n = 0
    with REAL_PAIRS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target = (row.get("target") or "").strip()
            raw = (row.get("asr") or row.get("input") or "").strip()
            if not target or not raw:
                continue
            n_words = len(target.split())
            w = pair_wer(target, raw, fair_norm)
            by_bin[bin_name(n_words)].append(w)
            all_w.append(w)
            n += 1
            if n % 500 == 0:
                print(f"scored {n}", flush=True)
    out = {"n": n, "bins": {}}
    for name in BIN_NAMES:
        xs = by_bin[name]
        out["bins"][name] = {
            "n": len(xs),
            "wers": xs,
            "mean": statistics.fmean(xs) if xs else 0.0,
            "median": statistics.median(xs) if xs else 0.0,
            "zero": (sum(1 for x in xs if x == 0.0) / len(xs) if xs else 0.0),
            "le05": (sum(1 for x in xs if x <= 0.05) / len(xs) if xs else 0.0),
        }
    out["all_mean"] = statistics.fmean(all_w) if all_w else 0.0
    out["all_zero"] = sum(1 for x in all_w if x == 0.0) / max(len(all_w), 1)
    REAL_WERS_BY_BIN.write_text(json.dumps(out) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "wers"} for k, v in out["bins"].items()}, indent=2))
    print(f"wrote {REAL_WERS_BY_BIN} n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
