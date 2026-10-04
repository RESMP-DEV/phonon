#!/usr/bin/env bash
set -euo pipefail

repo_root="${PHONON_REPO_ROOT:-/home/kearm/phonon}"
research_root="${PHONON_RESEARCH_ROOT:-/home/kearm/salm-lora}"
device="${PHONON_PROFILE_DEVICE:-0}"
steps="${PHONON_PROFILE_STEPS:-300}"
pack="${PHONON_PROFILE_PACK:-$research_root/build/format-bakeoff/granary-prose_dictation_v1-10000-969944574ea3-ctx512}"
out_root="${PHONON_PROFILE_ROOT:-$research_root/build/nsys/reverse-training-v1}"
variants="${PHONON_PROFILE_VARIANTS:-baseline,triton}"
python_bin="${PHONON_RESEARCH_PYTHON:-/home/kearm/envs/salm-lora/bin/python}"
nsys_root="${PHONON_NSYS_ROOT:-/usr/local/cuda-12.8/nsight-systems-2024.6.2}"
nsys_bin="$nsys_root/bin/nsys"
wandb_project="${PHONON_PROFILE_WANDB_PROJECT:-phonon}"
wandb_mode="${PHONON_PROFILE_WANDB_MODE:-offline}"
wandb_suffix="${PHONON_PROFILE_WANDB_SUFFIX:-}"

test -d "$pack"
test -x "$python_bin"
test -x "$nsys_bin"
if [[ -n "$wandb_suffix" && ! "$wandb_suffix" =~ ^[A-Za-z0-9_.-]+$ ]]; then
	printf 'invalid PHONON_PROFILE_WANDB_SUFFIX: %s\n' "$wandb_suffix" >&2
	exit 2
fi
mkdir -p "$out_root"

run_variant() {
	local variant="$1"
	local output="$out_root/$variant"
	local extra_args=()
	case "$variant" in
	baseline | triton | liger | triton-liger) ;;
	*)
		printf 'unknown profile variant: %s\n' "$variant" >&2
		return 2
		;;
	esac
	if [[ -e "$output/report.nsys-rep" || -e "$output/run.log" ]]; then
		printf 'refusing to overwrite existing profile: %s\n' "$output" >&2
		return 3
	fi
	mkdir -p "$output"
	if [[ "$variant" == triton ]]; then
		extra_args+=("--torch-compile")
	fi
	if [[ "$variant" == liger ]]; then
		extra_args+=("--liger-cross-entropy")
	fi
	if [[ "$variant" == triton-liger ]]; then
		extra_args+=("--torch-compile" "--liger-cross-entropy")
	fi
	if [[ -n "$wandb_project" && "$wandb_mode" != disabled ]]; then
		extra_args+=(
			"--wandb-project" "$wandb_project"
			"--wandb-mode" "$wandb_mode"
			"--wandb-run-id" "reverse-profile-${variant}${wandb_suffix:+-${wandb_suffix}}"
			"--wandb-name" "reverse-profile-${variant}${wandb_suffix:+-${wandb_suffix}}"
			"--wandb-stage" "profile"
		)
	fi
	CUDA_VISIBLE_DEVICES="$device" \
		TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}" \
		"$nsys_bin" profile \
		--trace=cuda,nvtx \
		--nvtx-capture=hierarchical \
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

	if [[ -f "$output/report.qdstrm" ]]; then
		"$nsys_root/host-linux-x64/QdstrmImporter" \
			-i "$output/report.qdstrm" \
			-o "$output/report.nsys-rep" \
			-f >"$output/import.log" 2>&1
	fi

	if [[ ! -f "$output/report.sqlite" ||
		"$output/report.nsys-rep" -nt "$output/report.sqlite" ]]; then
		"$nsys_bin" export \
			--type sqlite \
			--force-overwrite=true \
			--output "$output/report.sqlite" \
			"$output/report.nsys-rep" >"$output/export.log" 2>&1
	fi

	"$nsys_bin" stats \
		--report cuda_gpu_kern_sum \
		--report cuda_gpu_mem_time_sum \
		--format csv \
		--output "$output/stats" \
		"$output/report.nsys-rep" >"$output/stats-stdout.txt"
}

IFS=, read -r -a requested_variants <<<"$variants"
printf '%s\n' "pack=$pack" "device=$device" "steps=$steps" \
	"variants=$variants" >"$out_root/profile-meta.txt"
for variant in "${requested_variants[@]}"; do
	run_variant "$variant" "$steps"
done
printf 'DONE %s\n' "$out_root"
