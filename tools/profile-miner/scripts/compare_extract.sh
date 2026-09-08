#!/bin/sh
# Compare Python `extract --source code` with `phonon-mine extract-code`.
# Usage: compare_extract.sh [HOME_OVERRIDE]
# Env: UNTIL (optional), OUT_BASE (default mktemp), PHONON_MINE_BIN (optional).
# `seconds` in counts.json is wall-clock and cannot match; it is zeroed before diff.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
MINER="$ROOT/tools/profile-miner"

if [ "${1:-}" != "" ]; then
  HOME=$1
  export HOME
fi

OUT_BASE="${OUT_BASE:-$(mktemp -d "${TMPDIR:-/tmp}/phonon-mine-compare.XXXXXX")}"
OUT_PY="$OUT_BASE/py"
OUT_RS="$OUT_BASE/rs"
NORM="$OUT_BASE/norm"
mkdir -p "$OUT_PY" "$OUT_RS" "$NORM/py/extract" "$NORM/rs/extract"

echo "home=$HOME"
echo "out=$OUT_BASE"

if [ -z "${PHONON_MINE_BIN:-}" ]; then
  cargo build -p phonon-mine --release --manifest-path "$ROOT/Cargo.toml"
  PHONON_MINE_BIN="$ROOT/target/release/phonon-mine"
fi

now() {
  python3 -c 'import time; print(time.perf_counter())'
}

set --
if [ -n "${UNTIL:-}" ]; then
  set -- --until "$UNTIL"
fi

t0=$(now)
(
  cd "$MINER"
  PHONON_MINER_OUT="$OUT_PY" uv run --offline --python 3.12 python -m profile_miner extract --source code "$@"
)
t1=$(now)

t2=$(now)
"$PHONON_MINE_BIN" extract-code --out "$OUT_RS" "$@"
t3=$(now)

python3 -c 'import sys; print("python_seconds=%.3f" % (float(sys.argv[1]) - float(sys.argv[2])))' "$t1" "$t0"
python3 -c 'import sys; print("rust_seconds=%.3f" % (float(sys.argv[1]) - float(sys.argv[2])))' "$t3" "$t2"

python3 - "$OUT_PY" "$OUT_RS" "$NORM" <<'PY'
import re, shutil, sys
from pathlib import Path

# Zero only the seconds value so diff still sees indent, key order, and other fields.
sec = re.compile(r'"seconds": [0-9.]+')

def copy_norm(src_root: Path, dst_extract: Path) -> None:
    src = src_root / "extract"
    dst_extract.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "code.txt", dst_extract / "code.txt")
    text = (src / "counts.json").read_text()
    (dst_extract / "counts.json").write_text(sec.sub('"seconds": 0.0', text, count=1))

py, rs, norm = map(Path, sys.argv[1:4])
copy_norm(py, norm / "py" / "extract")
copy_norm(rs, norm / "rs" / "extract")
PY

echo "diff -r (extract/, seconds zeroed):"
if diff -r "$NORM/py/extract" "$NORM/rs/extract"; then
  echo "diff: empty"
else
  echo "diff: NOT empty"
  exit 1
fi
