#!/usr/bin/env bash
set -euo pipefail

repo_root="${PHONON_REPO_ROOT:-/home/kearm/phonon}"
research_root="${PHONON_RESEARCH_ROOT:-/home/kearm/salm-lora}"
device="${PHONON_PROFILE_DEVICE:-0}"
steps="${PHONON_PROFILE_STEPS:-300}"
pack="${PHONON_PROFILE_PACK:-$research_root/build/format-bakeoff/granary-prose_dictation_v1-10000-969944574ea3-ctx512}"
out_root="${PHONON_PROFILE_ROOT:-$research_root/build/nsys/reverse-training-v1}"
python_bin="${PHONON_RESEARCH_PYTHON:-/home/kearm/envs/salm-lora/bin/python}"

test -d "$pack"
test -x "$python_bin"
test -x /usr/bin/nsys
mkdir -p "$out_root"

run_variant() {
	local variant="$1"
	local output="$out_root/$variant"
	local extra_args=()
	mkdir -p "$output"
	if [[ "$variant" == triton ]]; then
		extra_args+=("--torch-compile")
	fi
	CUDA_VISIBLE_DEVICES="$device" \
		TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}" \
		nsys profile \
		--trace=cuda,nvtx \
		--sample=none \
		--cpuctxsw=none \
		--force-overwrite=true \
		--output "$output/report" \
		"$python_bin" "$repo_root/ml/research/reverse_vl_v0/train_reverse_audio_vl.py" \
		--dataset "$pack" \
		--steps "$steps" \
		--context-length 512 \
		--warmup 50 \
		--lr 6.505720091093967e-05 \
		--adapter-lr 4.943429131224935e-05 \
		--rank 32 \
		--lora-dropout 0.033694424584220395 \
		--ckpt-every 0 \
		--keep-ckpts 0 \
		--out "$output/train" \
		"${extra_args[@]+"${extra_args[@]}"}" \
		2>&1 | tee "$output/run.log"

	/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
		-i "$output/report.qdstrm" \
		-o "$output/report.nsys-rep" \
		-f >"$output/import.log" 2>&1

	nsys stats \
		--report cuda_gpu_kern_sum \
		--report cuda_gpu_mem_time_sum \
		--format csv \
		--output "$output/stats" \
		"$output/report.nsys-rep" >"$output/stats-stdout.txt"
}

printf '%s\n' "pack=$pack" "device=$device" "steps=$steps" >"$out_root/profile-meta.txt"
run_variant baseline "$steps"
run_variant triton "$steps"
printf 'DONE %s\n' "$out_root"
