#!/bin/sh
# Run the whole miner. Output dir: $PHONON_MINER_OUT (default ./out).
# Stages needing a model run under uv with the pinned package, offline.
set -eu
cd "$(dirname "$0")"
export PHONON_MINER_OUT="${PHONON_MINER_OUT:-$PWD/out}"
export HF_HUB_OFFLINE=1
UNTIL="${UNTIL:-}"
TOP="${ORACLE_TOP:-200}"   # live non-lexicon terms; lexicon spoken forms ship in lexicon/spoken.jsonl.gz
PY="uv run --offline --python 3.12"

# Code identifiers come from the Rust port when it is built (cargo build --release -p phonon-mine; byte-identical
# output, about twice as fast); it writes extract/counts.json fresh, so it runs before the Python sources merge in.
MINE="$(cd ../.. && pwd)/target/release/phonon-mine"
if [ -x "$MINE" ]; then
  "$MINE" extract-code --out "$PHONON_MINER_OUT" ${UNTIL:+--until "$UNTIL"}
  $PY python -m profile_miner extract --source claude --source codex --source grok --source repos ${UNTIL:+--until "$UNTIL"}
else
  $PY python -m profile_miner extract ${UNTIL:+--until "$UNTIL"}
fi
$PY python -m profile_miner seed
$PY python -m profile_miner candidates
$PY --with parakeet-mlx==0.5.2 python -m profile_miner oracle --top "$TOP"
$PY python -m profile_miner rank
# Judge: LoRA keep/drop classifier over the live top-N. Needs an mlx_lm adapter dir in $PHONON_JUDGE_ADAPTER
# (recipe in judge/); without one the dictionary falls back to the top LIVE_TOP live terms by rank.
if [ -n "${PHONON_JUDGE_ADAPTER:-}" ]; then
  $PY --with mlx-lm==0.31.3 python -m profile_miner judge --top "${JUDGE_TOP:-300}" || echo "judge failed (optional)"
fi
# Lexicon (lexicon/lexicon.json.gz + lexicon/spoken.jsonl.gz) is built once from public lists: lexicon/fetch.sh, then
# `profile_miner lexicon build`, oracled on a Linux GPU box with lexicon/oracle_linux.py. Both ship in the repo.
$PY python -m profile_miner dictionary --live-top "${LIVE_TOP:-100}"
if [ "${GEMMA:-1}" = 1 ]; then
  $PY --with mlx-lm==0.31.3 python -m profile_miner gemma --minutes "${GEMMA_MINUTES:-20}" || echo "gemma pass failed (optional)"
fi
echo "done: $PHONON_MINER_OUT/mined/dictionary.json (ranked candidates in mined/candidates.json)"
