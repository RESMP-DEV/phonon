#!/usr/bin/env bash
set -euo pipefail
cd /home/user/phonon
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export CUDA_VISIBLE_DEVICES=0 TMPDIR=/data/phonon_fusion_v0/tmp NUMBA_CACHE_DIR=/data/phonon_fusion_v0/numba
export MPLCONFIGDIR=/data/phonon_fusion_v0/matplotlib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export CUDA_CACHE_PATH=/data/phonon_fusion_v0/cuda-cache
export TORCH_EXTENSIONS_DIR=/data/phonon_fusion_v0/torch-extensions
export TORCH_HOME=/data/phonon_fusion_v0/cache/torch XDG_CACHE_HOME=/data/phonon_fusion_v0/cache
export NEMO_CACHE_DIR=/data/phonon_fusion_v0/cache/nemo
export UV_CACHE_DIR=/data/phonon_fusion_v0/cache/uv
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$NEMO_CACHE_DIR"
heartbeat_pid=''
lease_held=0
finish() {
    if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
    if [[ "$lease_held" == 1 ]]; then
        overnight-compute release --agent codex-fusion --status done
    fi
}
trap finish EXIT
while true; do
    nvidia-smi
    overnight-compute schedule --agent codex-fusion --start now --duration 10m --resource gpu0 --note 'Fusion v0; release between steps'
    lease_log=/data/phonon_fusion_v0/logs/lease_latest.log
    overnight-compute wait --agent codex-fusion --poll 5s | tee "$lease_log"
    if ! rg -q 'acquired codex-fusion' "$lease_log"; then exit 1; fi
    lease_held=1
    compute_pids=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ -z "$compute_pids" ]]; then break; fi
    overnight-compute release --agent codex-fusion --status done
    lease_held=0
    printf 'GPU remains occupied after acquisition; released lease and waiting for PIDs: %s\n' "$compute_pids"
    busy_checks=0
    while [[ -n "$compute_pids" ]]; do
        sleep 5
        compute_pids=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)
        busy_checks=$((busy_checks + 1))
        if (( busy_checks % 12 == 0 )); then
            printf 'Still waiting without a lease for GPU processes: %s\n' "$compute_pids"
        fi
    done
done
(
    while sleep 60; do overnight-compute heartbeat --agent codex-fusion --ttl 10m; done
) &
heartbeat_pid=$!
"$@"
