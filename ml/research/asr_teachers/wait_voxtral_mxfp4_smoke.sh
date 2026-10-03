#!/usr/bin/env bash
set -euo pipefail

research_root="${PHONON_RESEARCH_ROOT:-/home/kearm/salm-lora}"
repo_root="${PHONON_REPO_ROOT:-/home/kearm/phonon}"
download_exit="${PHONON_VOXTRAL_DOWNLOAD_EXIT:-$research_root/build/asr-teachers/voxtral-small-24b-hfd-da5b424.exit}"
run_name="${PHONON_VOXTRAL_RUN_NAME:-voxtral-mxfp4-smoke3}"
run_exit="$research_root/build/asr-teachers/$run_name.exit"

while [ ! -f "$download_exit" ]; do
	date -Is
	sleep 30
done

download_status="$(cat "$download_exit")"
echo "verified base download exit=$download_status"
if [ "$download_status" != 0 ]; then
	echo "$download_status" >"$run_exit"
	exit "$download_status"
fi

set +e
"$repo_root/ml/research/asr_teachers/run_voxtral_mxfp4_smoke.sh"
status=$?
set -e
echo "$status" >"$run_exit"
exit "$status"
