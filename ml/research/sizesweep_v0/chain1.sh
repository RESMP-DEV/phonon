#!/usr/bin/env bash
# after the 1.7B train: guarded eval of the new adapter and the two carried-over baselines,
# the v2-list pass, then the two gemma throughput probes.
set -uo pipefail
cd /home/user/phonon
LOGS=/data/phonon_sizesweep_v0/logs
R=research/sizesweep_v0
bash $R/run_eval_size.sh 1 24 16 size_qwen3-1.7b:Qwen/Qwen3-1.7B bigrun_mid_r16:lfm12 \
  bigrun_big_350m:lfm350 > $LOGS/eval_round1.log 2>&1
echo "CHAIN1 eval rc=$?"
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16
export PATH=$PATH:$HOME/.local/bin
uv run --no-project --python 3.12 --with torch --with transformers --with accelerate \
  --with peft --with jiwer --with whisper-normalizer --with rapidfuzz --with metaphone \
  python research/retrieval_v2/refine_v2.py --batch-size 24 \
  --adapter size_qwen3-1.7b:Qwen/Qwen3-1.7B --adapter bigrun_big_350m:lfm350 \
  > $LOGS/v2_round1.log 2>&1
echo "CHAIN1 v2 rc=$?"
overnight-compute heartbeat --agent opus-size --ttl 30m >/dev/null 2>&1
bash $R/run_train_size.sh 1 probe_gemma_bs8 /data/phonon_bigrun_v0/train_mid.jsonl 1 2e-4 \
  --model google/gemma-4-E2B-it --lora-r 16 --lora-dropout 0.0 --batch-size 8 --grad-accum 1 \
  --grad-checkpointing on --max-seconds 240 > $LOGS/probe_gemma_bs8.log 2>&1
echo "CHAIN1 probe8 rc=$?"
bash $R/run_train_size.sh 1 probe_gemma_bs16 /data/phonon_bigrun_v0/train_mid.jsonl 1 2e-4 \
  --model google/gemma-4-E2B-it --lora-r 16 --lora-dropout 0.0 --batch-size 16 --grad-accum 1 \
  --grad-checkpointing on --max-seconds 240 > $LOGS/probe_gemma_bs16.log 2>&1
echo "CHAIN1 probe16 rc=$?"
overnight-compute heartbeat --agent opus-size --ttl 30m >/dev/null 2>&1
echo "CHAIN1 DONE"
