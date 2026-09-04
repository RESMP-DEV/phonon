#!/bin/bash
# One-time env for lexicon/oracle_linux.py on a CUDA box. Creates ./.venv-oracle next to this script's copy.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv-oracle
. .venv-oracle/bin/activate
uv pip install "torch" "torchaudio" --index-url https://download.pytorch.org/whl/cu130
uv pip install "nemo_toolkit[asr]" "kokoro>=0.9" "misaki[en]" soundfile
python -c "import nemo, kokoro, misaki; print('env ok')"
