#!/usr/bin/env bash
# Zero-cost local sweep. Never changes pyproject.toml or uv.lock.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/data/hf
export HF_HUB_CACHE=/data/hf/hub
export HUGGINGFACE_HUB_CACHE=/data/hf/hub
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
REPORT=runs/reports/option0_sweep_20260915
mkdir -p "$REPORT/logs" "$REPORT/preds"
exec 9>"$REPORT/sweep.lock"
flock -n 9 || { echo 'Another Option 0 sweep holds the run lock'; exit 1; }
nvidia-smi
overnight-compute schedule --agent codex-option0 --start now --duration 4h --resource machine
lease=$(overnight-compute wait --agent codex-option0)
printf '%s\n' "$lease"
[[ "$lease" == *acquired* ]] || exit 1
sweep_pid=$$
(
    while kill -0 "$sweep_pid" 2>/dev/null; do
        overnight-compute heartbeat --agent codex-option0 --ttl 30m
        sleep 45
    done
) > "$REPORT/logs/heartbeat.log" 2>&1 &
heartbeat_pid=$!
finish() {
    rc=$?
    kill "$heartbeat_pid" 2>/dev/null || true
    if (( rc == 0 )); then
        overnight-compute release --agent codex-option0 --status done
    else
        overnight-compute release --agent codex-option0 --status failed
    fi
}
trap finish EXIT
# Restore optional runtime packages without changing project metadata or the NeMo pin.
if ! uv run --no-sync python -c 'import importlib.metadata as m; assert m.version("nemo_toolkit") == "2.3.0"; import torch, torchaudio'; then
    uv pip install --python .venv/bin/python 'nemo_toolkit[asr]==2.3.0' \
        'torch==2.14.0' 'torchaudio==2.11.0' 'transformers==5.9.0' \
        'dill==0.3.8' 'multiprocess<0.70.17'
fi
models=(
    nvidia/parakeet-tdt-0.6b-v2
    nvidia/parakeet-tdt-0.6b-v3
    nvidia/parakeet-unified-en-0.6b
    /home/user/phonon/runs/finetune/parakeet_v3_a0p4375_x_jointout_lr1em7_interp_fine_save_20260527/a0p5625/checkpoint.nemo
    Qwen/Qwen3-ASR-0.6B-hf
    Qwen/Qwen3-ASR-1.7B-hf
    CohereLabs/cohere-transcribe-03-2026
    ibm-granite/granite-speech-4.1-2b
    ibm-granite/granite-speech-4.1-2b-plus
    ibm-granite/granite-speech-5.0-470m-turboctc
)
# Each transformer family has a separate uv resolved environment; NeMo stays at 2.3.0.
command_for() {
    model=$1
    tag=${model##*/}
    family=nemo
    if [[ "$model" == *.nemo ]]; then tag=parakeet_v3_localbest_a0p5625; fi
    command=(uv run --no-sync python)
    common=(--with torch==2.14.0 --with accelerate
            --with soundfile --with scipy --with pyarrow --with jiwer
            --with librosa --with sentencepiece --with protobuf)
    if [[ "$model" == Qwen/* ]]; then
        family=transformers-qwen3asr
        command=(uv run --no-project --isolated "${common[@]}" --with 'transformers==5.17.0' python)
    elif [[ "$model" == CohereLabs/* ]]; then
        family=transformers-cohere
        command=(uv run --no-project --isolated "${common[@]}" --with 'transformers==5.17.0' python)
    elif [[ "$model" == *granite-speech-5.0-470m-turboctc ]]; then
        family=transformers-granite-ctc
        command=(uv run --no-project --isolated "${common[@]}" --with 'transformers==5.17.0' python)
    elif [[ "$model" == ibm-granite/* ]]; then
        family=transformers-granite
        command=(uv run --no-project --isolated "${common[@]}" --with 'transformers==5.17.0' --with peft --with torchaudio==2.11.0 python)
    fi
}
score() {
    uv run --no-sync --with whisper-normalizer --with 'jiwer>=4' python scripts/option0/score_gates.py \
        --pred-dir "$REPORT/preds"
}
run_gate() {
    local gate=$1
    local out="$REPORT/preds/$gate/$tag.jsonl"
    if uv run --no-sync python scripts/option0/transcribe_gate.py --family "$family" \
        --model "$model" --gate "$gate" --out "$out" --check-complete; then
        echo "SKIP complete $gate/$tag"
        return
    fi
    if [[ " ${failed_models[*]:-} " == *" $tag "* ]]; then return; fi
    for attempt in 1; do
        local log="$REPORT/logs/${gate}_${tag}_attempt${attempt}.log"
        echo "$(date -Is) START $gate/$tag attempt=$attempt"
        # Conservative common batch size 1. Failed attempts retain their exact logs.
        if "${command[@]}" scripts/option0/transcribe_gate.py --family "$family" \
            --model "$model" --gate "$gate" --out "$out" --batch-size 1 > "$log" 2>&1; then
            echo "$(date -Is) DONE $gate/$tag"
            score
            return
        fi
        {
            printf '\n### %s / %s / attempt %s\n\n' "$tag" "$gate" "$attempt"
            printf 'Command family: `%s`. Exact output: `%s`.\n\n```text\n' "$family" "$log"
            tail -n 35 "$log"
            printf '\n```\n'
        } >> "$REPORT/failures.md"
        echo "FAILED $gate/$tag attempt=$attempt (see $log)"
        # Diagnose a failed process before retrying; preserve every completed prediction.
    done
    failed_models+=("$tag")
}
# Restored shards are local. Preserve validated predictions and append only missing IDs.
failed_models=()
uv run --no-sync python scripts/option0/materialize_gate_audio.py \
    > "$REPORT/logs/materialize_latest.log" 2>&1
ready=$(uv run --no-sync python scripts/option0/gates.py --ready)
for model in "${models[@]}"; do
    command_for "$model"
    for gate in $ready; do run_gate "$gate"; done
done
uv run ruff check scripts/option0 > "$REPORT/logs/ruff_option0_final.log" 2>&1
uv run pytest -q > "$REPORT/logs/pytest_final.log" 2>&1
uv run phonon doctor > "$REPORT/logs/doctor_final.log" 2>&1
score
uv run --no-sync python - <<'PY_DEVLOG'
import json
from pathlib import Path
root = Path('runs/reports/option0_sweep_20260915')
report = json.loads((root / 'results.json').read_text())
markdown = (root / 'results.md').read_text()
tables = []
inside = False
for line in markdown.splitlines():
    if line.startswith('|'):
        tables.append(line)
        inside = True
    elif inside:
        tables.append('')
        inside = False
count = sum(gate.get('clip_count', 0) for model in report['models']
            for gate in model['gates'].values())
missing = [(model['tag'], gate) for model in report['models']
           for gate in ('course91', 'uncertain48', 'wispr_holdout120', 'wispr_edit25')
           if not model['gates'].get(gate, {}).get('complete')]
summary = (f"Follow-up results: {count} scored predictions across {len(report['models'])} candidates. "
           f"Scoring errors: {len(report['errors'])}. "
           + (f"Incomplete model/gate pairs: {missing}." if missing else "All requested labeled gates are complete.")
           + " Full metadata, missing IDs, metric definitions, and reference limitations are in results.md and results.json. "
           "Fair and strict WER and term errors below are fractions. The five-gate mean weights each gate equally. "
           "The final script Ruff, full pytest, and phonon doctor checks passed; their logs are in the report directory.")
p = Path('DEVLOG.md')
s = p.read_text()
a = '<!-- option0-followup-tables:start -->'
b = '<!-- option0-followup-tables:end -->'
start = s.index(a) + len(a)
end = s.index(b, start)
p.write_text(s[:start] + '\n' + summary + '\n\n' + '\n'.join(tables) + '\n' + s[end:])
PY_DEVLOG
uv run --no-sync python scripts/option0/gates.py
cat "$REPORT/results.md"
uv run --no-sync python - <<'PY_CHECK'
import json
from pathlib import Path
report = json.loads(Path('runs/reports/option0_sweep_20260915/results.json').read_text())
assert not report['errors'], report['errors']
assert len(report['models']) == 10
required = ('aqua_new_holdout', 'course91', 'novel180', 'hard77', 'uncertain48',
            'wispr_holdout120', 'wispr_edit25', 'personal_cuda')
incomplete = [(model['tag'], gate) for model in report['models'] for gate in required
              if not model['gates'].get(gate, {}).get('complete')]
assert not incomplete, f'Incomplete model/gate predictions: {incomplete}'
assert all(model['gates']['hard77']['technical_terms'] > 0 for model in report['models'])
PY_CHECK
