#!/usr/bin/env bash
# gemma-4-E2B (1 epoch, batch 8, grad checkpointing on - the fastest config that fits),
# then Qwen3-0.6B (2 epochs), each followed by its guarded eval and v2-list pass;
# finally the batch-1 latency bench over every model in the sweep.
set -uo pipefail
cd /home/user/phonon
LOGS=/data/phonon_sizesweep_v0/logs
R=research/sizesweep_v0
hb() { overnight-compute heartbeat --agent opus-size --ttl 30m >/dev/null 2>&1; }
v2() {
  export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
  export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16
  export PATH=$PATH:$HOME/.local/bin
  uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
    --with peft --with jiwer --with whisper-normalizer --with rapidfuzz --with metaphone \
    python research/retrieval_v2/refine_v2.py --batch-size 24 --adapter "$1" > "$2" 2>&1
}

bash $R/run_train_size.sh 1 size_gemma4-e2b /data/phonon_bigrun_v0/train_mid.jsonl 1 2e-4 \
  --model google/gemma-4-E2B-it --lora-r 16 --lora-dropout 0.0 --batch-size 8 --grad-accum 1 \
  --grad-checkpointing on --max-seconds 7200 > $LOGS/train_gemma4-e2b.log 2>&1
echo "CHAIN2 train_gemma rc=$? $(date +%T)"; hb
bash $R/run_eval_size.sh 1 24 16 size_gemma4-e2b:google/gemma-4-E2B-it \
  > $LOGS/eval_gemma.log 2>&1
echo "CHAIN2 eval_gemma rc=$? $(date +%T)"; hb
v2 size_gemma4-e2b:google/gemma-4-E2B-it $LOGS/v2_gemma.log
echo "CHAIN2 v2_gemma rc=$? $(date +%T)"; hb

bash $R/run_train_size.sh 1 size_qwen3-0.6b /data/phonon_bigrun_v0/train_mid.jsonl 2 2e-4 \
  --model Qwen/Qwen3-0.6B --lora-r 16 --lora-dropout 0.0 --batch-size 16 --grad-accum 1 \
  > $LOGS/train_qwen3-0.6b.log 2>&1
echo "CHAIN2 train_06b rc=$? $(date +%T)"; hb
bash $R/run_eval_size.sh 1 24 16 size_qwen3-0.6b:Qwen/Qwen3-0.6B > $LOGS/eval_06b.log 2>&1
echo "CHAIN2 eval_06b rc=$? $(date +%T)"; hb
v2 size_qwen3-0.6b:Qwen/Qwen3-0.6B $LOGS/v2_06b.log
echo "CHAIN2 v2_06b rc=$? $(date +%T)"; hb

bash $R/run_bench_size.sh 1 bigrun_big_350m:lfm350 size_qwen3-0.6b:Qwen/Qwen3-0.6B \
  bigrun_mid_r16:lfm12 size_qwen3-1.7b:Qwen/Qwen3-1.7B \
  size_gemma4-e2b:google/gemma-4-E2B-it > $LOGS/bench.log 2>&1
echo "CHAIN2 bench rc=$? $(date +%T)"; hb
echo "CHAIN2 DONE $(date +%T)"
