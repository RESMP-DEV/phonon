#!/bin/bash
# Refiner v3: LoRA fine-tunes on the spoken-register synthetic corpus, synth-only and synth+real,
# three bases. Copy of refiner_v2_chain.sh with ref3_ names and the duplicate --model _real:: /
# ref__synth entries (which always failed with an empty repo id) removed.
# Usage: GPU=0 bash refiner_v3_chain.sh /data/phonon_synth_v3/synth_pairs_v3.jsonl
set -u
cd /home/user/phonon
SYNTH=${1:-/data/phonon_synth_v3/synth_pairs_v3.jsonl}
export HF_HOME=/data/hf CUDA_VISIBLE_DEVICES=${GPU:-0} UV_TORCH_BACKEND=cu130
W="--with torch --with transformers --with peft --with trl --with accelerate --with datasets --with jiwer --with whisper-normalizer"
LOG=/data/phonon_corrector_v0/logs/refiner_v3
mkdir -p $LOG /data/phonon_corrector_v0/refiner
COMBO=/data/phonon_corrector_v0/refiner/synth_v3_plus_real.jsonl
cat $SYNTH /data/phonon_corrector_v0/train.jsonl > $COMBO
echo "SYNTH=$SYNTH rows=$(wc -l < $SYNTH) combo=$(wc -l < $COMBO)" >> $LOG/chain.log
MODELS=""
for base in "LiquidAI/LFM2.5-350M:lfm2.5-350m" "Qwen/Qwen3-0.6B:qwen3-0.6b" "LiquidAI/LFM2.5-1.2B-Instruct:lfm2.5-1.2b"; do
  mid=${base%%:*}; short=${base##*:}
  for data in "synth:$SYNTH" "synthreal:$COMBO"; do
    tag=${data%%:*}; path=${data#*:}; name=ref3_${short}_${tag}
    rm -rf /data/phonon_corrector_v0/adapters/$name
    uv run --no-project --python 3.12 $W python research/corrector_v0/train_lora.py \
      --model $mid --name $name --train $path --epochs 2 ${EXTRA_TRAIN_FLAGS:-} \
      > $LOG/train_$name.log 2>&1
    echo "TRAIN_EXIT $name $?" >> $LOG/chain.log
    overnight-compute heartbeat --agent opus-register --ttl 30m >/dev/null 2>&1
    MODELS="$MODELS --model $name:$mid:/data/phonon_corrector_v0/adapters/$name"
  done
done
echo "EVAL_MODELS $MODELS" >> $LOG/chain.log
uv run --no-project --python 3.12 $W python research/corrector_v0/eval_corrector.py \
  --max-new-tokens 2048 --batch-size 32 $MODELS \
  --out-json research/corrector_v0/results_refiner_v3.json \
  --out-md research/corrector_v0/results_refiner_v3.md > $LOG/eval.log 2>&1
echo "EVAL_EXIT $?" >> $LOG/chain.log
echo CHAIN_DONE >> $LOG/chain.log
