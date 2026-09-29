#!/usr/bin/env bash
# quant.sh <adapter-name>: merge, bf16 reference, MLX custom_attn6emb8, GGUF IQ4_XS with the
# mixed imatrix, and a phonon-bench run for each against the bf16 reference.
set -uo pipefail
NAME=${1:?adapter name}
BASE=LiquidAI/LFM2.5-1.2B-Instruct
MERGED=/data/phonon_bench_v0/merged/$NAME
MLXDIR=/data/phonon_bench_v0/mlx/${NAME}_attn6emb8
Q=/data/phonon_ship_v0/quant/$NAME
LOG=/data/phonon_ship_v0/logs
LL=/data/llama.cpp/build/bin
PY=/data/venvs/phonon_bench/bin/python
PB=/data/venvs/phonon_bench/bin/phonon-bench
SETS=real_580,term_new
PORT=$(( 8090 + RANDOM % 50 ))
mkdir -p "$Q" "$LOG" /data/phonon_bench_v0/merged /data/phonon_bench_v0/mlx
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 UV_PROJECT_ENVIRONMENT=/data/venvs/phonon_bench
step () { echo "=== $(date +%H:%M:%S) $* ==="; }

step "merge $NAME"
$PY /home/user/phonon/research/quant_v0/merge_adapter.py \
    --base "$BASE" --adapter "/data/phonon_corrector_v0/adapters/$NAME" --out "$MERGED" \
    || exit 11
# transformers 5.17 drops block_ff_dim when it saves a merged LFM2 checkpoint
cp /data/hf/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/*/config.json "$MERGED/config.json"

step "bf16 reference (hf)"
$PB run --backend hf --base "$MERGED" --label "ship_${NAME}_bf16" --sets "$SETS" \
    --build-reference --ref-key "${NAME}_bf16" --no-retrieval --no-latency || exit 12

step "mlx convert custom_attn6emb8"
rm -rf "$MLXDIR"
$PY /data/phonon_ship_v0/scripts/mlx_convert.py "$MERGED" "$MLXDIR" || exit 13

step "mlx bench"
$PB run --backend mlx --model-dir "$MLXDIR" --base "$MERGED" \
    --label "ship_${NAME}_mlx_attn6emb8" --sets "$SETS" --ref-key "${NAME}_bf16" \
    --no-retrieval --no-latency || echo "MLX BENCH FAILED rc=$?"

step "gguf f16"
cd /data/llama.cpp
uv run --no-project --python 3.12 --with gguf --with torch --with transformers \
    --with sentencepiece --with protobuf \
    python /data/llama.cpp/convert_hf_to_gguf.py "$MERGED" \
    --outfile "$Q/${NAME}-f16.gguf" --outtype f16 || exit 14

step "imatrix (calib.txt + calib_resmp.txt = calib_mix.txt)"
$LL/llama-imatrix -m "$Q/${NAME}-f16.gguf" -f /data/phonon_quant_v0/gguf/calib_mix.txt \
    -o "$Q/imatrix_mix.gguf" -ngl 99 -c 2048 --parse-special || exit 15

step "quantize IQ4_XS"
$LL/llama-quantize --imatrix "$Q/imatrix_mix.gguf" "$Q/${NAME}-f16.gguf" \
    "$Q/${NAME}-IQ4_XS.gguf" IQ4_XS || exit 16

step "llama-server on port $PORT"
$LL/llama-server -m "$Q/${NAME}-IQ4_XS.gguf" --port "$PORT" -ngl 99 -c 32768 --parallel 8 \
    --jinja --temp 0 --seed 0 > "$LOG/server_${NAME}.log" 2>&1 &
SRV=$!
for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
    sleep 2
done
step "llama bench"
$PB run --backend llama --base-url "http://127.0.0.1:$PORT" \
    --label "ship_${NAME}_gguf_IQ4_XS" --sets "$SETS" --ref-key "${NAME}_bf16" \
    --no-retrieval --no-latency --max-batch 8 || echo "LLAMA BENCH FAILED rc=$?"
kill "$SRV" 2>/dev/null          # started by this script, nothing else is touched
wait "$SRV" 2>/dev/null

step "sizes"
$PY - "$NAME" "$MERGED" "$MLXDIR" "$Q" <<'PYEOF'
import json, sys
from pathlib import Path
name, merged, mlxdir, q = sys.argv[1:5]
def mb(p):
    p = Path(p)
    if p.is_dir():
        return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1)
    return round(p.stat().st_size / 1e6, 1) if p.exists() else None
out = {"adapter": name, "merged_bf16_MB": mb(merged), "mlx_attn6emb8_MB": mb(mlxdir),
       "gguf_f16_MB": mb(f"{q}/{name}-f16.gguf"),
       "gguf_IQ4_XS_MB": mb(f"{q}/{name}-IQ4_XS.gguf")}
Path(f"{q}/sizes.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
PYEOF
step "quant.sh $NAME done"
