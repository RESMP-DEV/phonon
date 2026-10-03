#!/usr/bin/env bash
set -euo pipefail

repo_root="${PHONON_REPO_ROOT:-/home/kearm/phonon}"
research_root="${PHONON_RESEARCH_ROOT:-/home/kearm/salm-lora}"
audio_root="${PHONON_AUDIO_ROOT:-/home/kearm/aqua-training-data}"
run_name="${PHONON_VOXTRAL_RUN_NAME:-voxtral-mxfp4-audio256-v1}"
port="${PHONON_VOXTRAL_VLLM_PORT:-8303}"
base_dir="$research_root/build/asr-teachers"
vllm_model="$base_dir/$run_name-vllm"
slice="$research_root/slice-eval-500.jsonl"
output="$base_dir/$run_name-eval500.jsonl"
score="$base_dir/$run_name-eval500-score.json"
server_log="$base_dir/$run_name-eval500-server.log"
eval_exit="$base_dir/$run_name-eval500.exit"
receipt="$base_dir/$run_name-eval500-receipt.json"
server_pid=""

finish() {
	if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
		kill "$server_pid" 2>/dev/null || true
		wait "$server_pid" 2>/dev/null || true
	fi
}
trap finish EXIT

write_receipt() {
	local status="$1"
	local error="${2-}"
	STATUS="$status" ERROR_STRING="$error" RUN_NAME="$run_name" PORT="$port" \
		MODEL="$vllm_model" OUTPUT="$output" SCORE="$score" SERVER_LOG="$server_log" \
		EVAL_EXIT="$eval_exit" python3 - "$receipt" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

receipt = Path(sys.argv[1])
paths = {
    "output": Path(os.environ["OUTPUT"]),
    "score": Path(os.environ["SCORE"]),
    "server_log": Path(os.environ["SERVER_LOG"]),
}
payload = {
    "schema_version": 1,
    "status": os.environ["STATUS"],
    "run_name": os.environ["RUN_NAME"],
    "port": int(os.environ["PORT"]),
    "model": os.environ["MODEL"],
    **{name: str(path) for name, path in paths.items()},
    **{
        f"{name}_sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        for name, path in paths.items()
    },
    "score": json.loads(paths["score"].read_text()) if paths["score"].is_file() else None,
    "non_claims": [
        "no BF16 Voxtral reference was run",
        "accepted historical text is the ground truth",
        "technical-term damage needs a separate audit",
    ],
    "error": os.environ.get("ERROR_STRING") or None,
}
receipt.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
Path(os.environ["EVAL_EXIT"]).write_text(
    "0\n" if payload["status"] == "complete" else "1\n"
)
PY
}

test -d "$vllm_model"
test -f "$slice"
test ! -e "$output"
test ! -e "$eval_exit"
: >"$server_log"

CUDA_VISIBLE_DEVICES=0 HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
	VLLM_USE_FLASHINFER_SAMPLER=0 \
	"$base_dir/venvs/vllm-0p30/bin/vllm" serve "$vllm_model" \
	--tokenizer-mode mistral \
	--served-model-name "$run_name" \
	--max-model-len 4096 \
	--max-num-seqs 1 \
	--gpu-memory-utilization 0.90 \
	--no-enable-log-requests \
	--disable-uvicorn-access-log \
	--host 127.0.0.1 \
	--port "$port" >>"$server_log" 2>&1 &
server_pid=$!

healthy=false
for _ in $(seq 1 240); do
	if ! kill -0 "$server_pid" 2>/dev/null; then
		wait "$server_pid" || true
		write_receipt failed "vLLM evaluation server exited before health"
		exit 1
	fi
	if curl --silent --output /dev/null --max-time 2 "http://127.0.0.1:$port/health"; then
		healthy=true
		break
	fi
	sleep 5
done
if [ "$healthy" != true ]; then
	write_receipt failed "vLLM evaluation health timeout"
	exit 1
fi

if ! "$base_dir/venvs/vllm-0p30/bin/python" \
	"$repo_root/ml/research/asr_teachers/transcribe_voxtral_vllm.py" \
	--slice "$slice" \
	--audio-root "$audio_root" \
	--out "$output" \
	--model "$run_name" \
	--url "http://127.0.0.1:$port/v1/audio/transcriptions"; then
	write_receipt failed "Voxtral transcription harness failed"
	exit 1
fi

/home/kearm/.local/bin/uv run --no-project --python 3.12 \
	--with "jiwer==3.0.4" --with whisper-normalizer \
	python "$repo_root/ml/research/final_sweep/score_fair_wer.py" "$output" >"$score"

finish
server_pid=""
write_receipt complete ""
echo "DONE $score"
