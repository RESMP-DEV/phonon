#!/bin/bash
# pool3 step A: shallow-clone one public repo, run the symbol index over it, delete the clone.
# usage: clone_extract.sh "<lang> <owner/name>"
set -u
LANG_="${1%% *}"
REPO="${1#* }"
NAME="$(echo "$REPO" | tr '/' '__')"
BASE=/data/phonon_pool3_v0
CL="$BASE/clones/$NAME"
OUT="$BASE/per_repo/${LANG_}__${NAME}.jsonl"
LOG="$BASE/logs/${LANG_}__${NAME}.log"
mkdir -p "$BASE/clones" "$BASE/per_repo" "$BASE/logs"
if [ -s "$OUT" ]; then echo "SKIP $REPO (done)"; exit 0; fi
rm -rf "$CL"
{
  echo "=== $REPO $(date -Is)"
  timeout 900 git clone --depth 1 --single-branch --quiet \
     "https://github.com/$REPO.git" "$CL" || { echo "CLONE_FAIL $REPO"; rm -rf "$CL"; exit 3; }
  du -sh "$CL"
  CUDA_VISIBLE_DEVICES= HF_HOME=/data/hf timeout 1800 \
    uv run --no-project --python 3.12 --with tree-sitter --with tree-sitter-language-pack \
    python /home/user/phonon/research/index_v0/symbol_index.py \
      --roots "$CL" --out "$OUT.tmp" --workers 6 --batch 200
  rc=$?
  rm -rf "$CL"
  if [ $rc -ne 0 ]; then echo "EXTRACT_FAIL $REPO rc=$rc"; exit 4; fi
  mv "$OUT.tmp" "$OUT"
  echo "OK $REPO $(wc -l < "$OUT") terms"
} >> "$LOG" 2>&1
tail -1 "$LOG"
