#!/bin/bash
# Remote setup + launch for the phonon LoRA run. Idempotent.
set -euo pipefail
cd ~/ft
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv .venv --python 3.12
. .venv/bin/activate
uv pip install -q torch --index-url https://download.pytorch.org/whl/cu126
uv pip install -q "transformers>=4.57" peft datasets accelerate sentencepiece huggingface_hub hf_transfer
[ -d sft ] || tar -xzf sft.tgz
python prep_generic.py sft 140 || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,transformers,peft;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__,'peft',peft.__version__)"
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "cd ~/ft && . .venv/bin/activate && python train_lora.py --data sft --model Qwen/Qwen3.5-4B --out out 2>&1 | tee train.log; echo TRAIN_EXIT=\$? >> train.log; sleep 86400"
echo SETUP_DONE
