#!/usr/bin/env bash
set -euo pipefail

repo_root="${PHONON_REPO_ROOT:-/home/kearm/phonon}"
research_root="${PHONON_RESEARCH_ROOT:-/home/kearm/salm-lora}"
run_name="${PHONON_VOXTRAL_RUN_NAME:-voxtral-mxfp4-smoke3}"
port="${PHONON_VOXTRAL_VLLM_PORT:-8301}"
base_dir="$research_root/build/asr-teachers"
source_model="$base_dir/$run_name"
vllm_model="$base_dir/$run_name-vllm"
quant_exit="$base_dir/$run_name.exit"
convert_exit="$base_dir/$run_name-vllm-convert.exit"
verify_exit="$base_dir/$run_name-vllm-verify.exit"
receipt="$base_dir/$run_name-vllm-verify-receipt.json"
server_log="$base_dir/$run_name-vllm-server.log"
response_dir="$base_dir/$run_name-vllm-responses"
server_pid=""
mkdir -p "$response_dir"

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
	STATUS="$status" ERROR_STRING="$error" VLLM_PORT="$port" RESPONSE_DIR="$response_dir" \
		python3 - "$receipt" "$source_model" \
		"$vllm_model" "$server_log" "$verify_exit" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

receipt, source, model, server_log, verify_exit = map(Path, sys.argv[1:])
log_hash = None
if server_log.exists():
    log_hash = hashlib.sha256(server_log.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "status": os.environ["STATUS"],
    "source_model": str(source),
    "vllm_model": str(model),
    "server_log": str(server_log),
    "server_log_sha256": log_hash,
    "port": int(os.environ["VLLM_PORT"]),
    "response_dir": os.environ["RESPONSE_DIR"],
    "response_sha256": {},
    "error": os.environ.get("ERROR_STRING") or None,
}
response_root = Path(os.environ["RESPONSE_DIR"])
payload["response_sha256"] = {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(response_root.glob("transcript-*.json"))
    if path.is_file()
}
receipt.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
if len(sys.argv) > 5:
    Path(verify_exit).write_text("0\n" if payload["status"] == "complete" else "1\n")
PY
}

while [ ! -f "$quant_exit" ]; do
	date -Is
	sleep 30
done
quant_status="$(cat "$quant_exit")"
echo "quantization exit=$quant_status"
if [ "$quant_status" != 0 ]; then
	write_receipt blocked "source quantization exited $quant_status"
	echo "$quant_status" >"$verify_exit"
	exit "$quant_status"
fi

if [ ! -f "$convert_exit" ]; then
	if "$repo_root/ml/research/asr_teachers/convert_voxtral_mxfp4_vllm.py" \
		--model "$source_model" \
		--output "$vllm_model" \
		--receipt "$base_dir/$run_name-vllm-convert-receipt.json"; then
		convert_status=0
	else
		convert_status=$?
	fi
else
	convert_status="$(cat "$convert_exit")"
fi
echo "$convert_status" >"$convert_exit"
if [ "$convert_status" != 0 ]; then
	write_receipt failed "Mistral-name conversion exited $convert_status"
	exit "$convert_status"
fi

: >"$server_log"
CUDA_VISIBLE_DEVICES=0 HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
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
		write_receipt failed "vLLM server exited before health"
		echo 1 >"$verify_exit"
		exit 1
	fi
	if curl --silent --output /dev/null --max-time 2 "http://127.0.0.1:$port/health"; then
		healthy=true
		break
	fi
	sleep 5
done
if [ "$healthy" != true ]; then
	write_receipt failed "vLLM health timeout"
	echo 1 >"$verify_exit"
	exit 1
fi

row_index=0
while IFS= read -r audio_path; do
	row_index=$((row_index + 1))
	response="$response_dir/transcript-$row_index.json"
	if ! curl --fail-with-body --silent --show-error --max-time 600 \
		-o "$response" \
		-F "model=$run_name" \
		-F "file=@$audio_path" \
		-F "language=en" \
		"http://127.0.0.1:$port/v1/audio/transcriptions"; then
		write_receipt failed "transcription request failed for row $row_index"
		echo 1 >"$verify_exit"
		exit 1
	fi
	if ! RESPONSE_PATH="$response" python3 - <<'PY'
import json
import os
from pathlib import Path

value = json.loads(Path(os.environ["RESPONSE_PATH"]).read_text())
if not str(value.get("text", "")).strip():
    raise RuntimeError("empty vLLM transcription response")
PY
	then
		write_receipt failed "empty or invalid transcription JSON for row $row_index"
		echo 1 >"$verify_exit"
		exit 1
	fi
done < <(
	python3 - "$base_dir/$run_name-calibration.json" <<'PY'
import json
import sys
from pathlib import Path

for row in json.loads(Path(sys.argv[1]).read_text())["rows"]:
    print(row["audio_path"])
PY
)

if [ "$row_index" -ne 2 ]; then
	write_receipt failed "expected two transcription responses, got $row_index"
	echo 1 >"$verify_exit"
	exit 1
fi

finish
server_pid=""
write_receipt complete ""
echo 0 >"$verify_exit"
echo "DONE $vllm_model"
