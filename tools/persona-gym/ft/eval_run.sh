#!/bin/bash
# Serve one candidate with vLLM and run the held-out persona eval.
# usage: eval_run.sh MODEL_PATH TAG [TOOL_PARSER] [MAX_TURNS] [CONCURRENCY]
set -uo pipefail
MODEL=$1; TAG=$2; PARSER=${3:-qwen3_xml}; TURNS=${4:-32}; PAR=${5:-10}; EXTRA=${6:-}
export PATH="$HOME/.local/bin:$PATH"
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd ~/gym
tmux kill-session -t serve 2>/dev/null
rm -f serve.log
tmux new-session -d -s serve ". ~/.venv-vllm/bin/activate && vllm serve $MODEL --served-model-name cand --port 8399 --max-model-len 65536 --max-num-seqs 32 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser $PARSER --reasoning-parser qwen3 2>&1 | tee serve.log"
for i in $(seq 1 120); do
  grep -q "Application startup complete" serve.log 2>/dev/null && break
  grep -q "EngineCore failed to start\|Error" serve.log 2>/dev/null && { echo SERVE_FAILED; grep -m3 "Error" serve.log; exit 1; }
  sleep 5
done
grep -q "Application startup complete" serve.log || { echo SERVE_TIMEOUT; exit 1; }
rm -rf rollouts-$TAG rollouts-$TAG.*.log
cat > one.sh <<EOF
#!/bin/bash
cd ~/gym/persona-gym && exec python3 -m persona_gym rollout --personas ~/gym/split/\$1 --endpoint http://localhost:8399/v1 --model cand --out ~/gym/rollouts-$TAG --max-turns $TURNS $EXTRA >> ~/gym/rollouts-$TAG.\$1.log 2>&1
EOF
chmod +x one.sh
ls split | xargs -P $PAR -n 1 ./one.sh
grep -h "\[rollout\]" rollouts-$TAG.*.log | grep -v transport | sort
cd persona-gym && python3 -m persona_gym grade --personas ~/gym/personas --rollouts ~/gym/rollouts-$TAG 2>&1 | tail -24
tmux kill-session -t serve 2>/dev/null
echo EVAL_DONE_$TAG
