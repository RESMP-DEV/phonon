#!/usr/bin/env bash
# pool2 step 1a: collect raw term sources that do NOT come from the repo walk.
set -uo pipefail
R=/data/phonon_pool2_v0/raw
mkdir -p "$R"
UA="phonon-research/1.0 (local research; contact <maintainer-email>)"

echo "== pypi"
timeout 120 curl -sSL -A "$UA" -o "$R/pypi_top.json" \
  https://hugovk.github.io/top-pypi-packages/top-pypi-packages.json
echo "pypi bytes=$(stat -c%s "$R/pypi_top.json" 2>/dev/null || echo 0)"

echo "== crates"
: > "$R/crates_raw.jsonl"
for p in 1 2 3 4 5 6 7 8 9 10; do
  timeout 60 curl -sSL -A "$UA" \
    -o "$R/crates_p$p.json" \
    "https://crates.io/api/v1/crates?sort=downloads&per_page=100&page=$p"
  echo "  crates page $p bytes=$(stat -c%s "$R/crates_p$p.json" 2>/dev/null || echo 0)"
  sleep 1
done

echo "== npm"
timeout 90 curl -sSL -A "$UA" -o "$R/npm_rank.json" \
  https://raw.githubusercontent.com/anvaka/npmrank/master/online/npmrank.json
echo "npm bytes=$(stat -c%s "$R/npm_rank.json" 2>/dev/null || echo 0)"

echo "== executables on PATH"
: > "$R/exes.txt"
for d in /usr/bin /usr/local/bin "$HOME/.local/bin" "$HOME/.cargo/bin" "$HOME/.bun/bin"; do
  [ -d "$d" ] || continue
  find "$d" -maxdepth 1 -type f -executable -printf '%f\n' 2>/dev/null >> "$R/exes.txt"
  find "$d" -maxdepth 1 -type l -printf '%f\n' 2>/dev/null >> "$R/exes.txt"
done
sort -u "$R/exes.txt" -o "$R/exes.txt"
echo "exes=$(wc -l < "$R/exes.txt")"

echo "== long flags from --help"
TOOLS="git docker cargo uv pip npm bun ssh rsync ffmpeg tmux nvidia-smi ncu nsys gcc g++ cmake make kubectl curl jq rg fd bat eza fzf htop btop tar zip unzip sed awk grep find xargs systemctl journalctl apt dpkg python3 node deno go rustc rustup clang ld nm objdump strace perf ip ss netstat iptables lsof ps top du df mount nvcc cuobjdump nvprof sqlite3 ffprobe yt-dlp croc hyperfine tokei"
: > "$R/help_flags.txt"
: > "$R/help_ok.txt"
for t in $TOOLS; do
  command -v "$t" >/dev/null 2>&1 || continue
  out=$(timeout 2 "$t" --help 2>&1; timeout 2 "$t" help 2>&1)
  n=$(printf '%s' "$out" | grep -oE -- '--[a-z][a-z0-9-]{2,}' | sort -u | tee -a "$R/help_flags.txt" | wc -l)
  echo "$t $n" >> "$R/help_ok.txt"
done
sort -u "$R/help_flags.txt" -o "$R/help_flags.txt"
echo "flags=$(wc -l < "$R/help_flags.txt") tools=$(wc -l < "$R/help_ok.txt")"

echo "== cuda api names"
: > "$R/cuda_api.txt"
H=/usr/local/cuda/include
grep -hoE 'cuda[A-Z][A-Za-z0-9_]{2,}' "$H/cuda_runtime_api.h" "$H/cuda.h" "$H/driver_types.h" 2>/dev/null >> "$R/cuda_api.txt"
grep -hoE 'cublas[A-Za-z0-9_]{3,}' "$H/cublas_api.h" "$H/cublasLt.h" 2>/dev/null >> "$R/cuda_api.txt"
grep -hoE 'cudnn[A-Za-z0-9_]{3,}' /usr/include/x86_64-linux-gnu/cudnn*.h 2>/dev/null >> "$R/cuda_api.txt"
grep -hoE 'cu(Blas|Rand|Sparse|Solver|Fft)[A-Za-z0-9_]{2,}|curand[A-Za-z0-9_]{3,}|cusparse[A-Za-z0-9_]{3,}|cusolver[A-Za-z0-9_]{3,}|cufft[A-Za-z0-9_]{3,}|nvml[A-Za-z0-9_]{3,}|nccl[A-Za-z0-9_]{3,}' \
  "$H"/*.h 2>/dev/null >> "$R/cuda_api.txt"
sort -u "$R/cuda_api.txt" -o "$R/cuda_api.txt"
echo "cuda_api=$(wc -l < "$R/cuda_api.txt")"

echo "== hf hub dirs"
ls /data/hf/hub > "$R/hf_hub.txt" 2>/dev/null
echo "hf=$(wc -l < "$R/hf_hub.txt")"

echo "COLLECT done"
