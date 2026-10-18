#!/usr/bin/env bash
# One-time env for lexicon/oracle_linux.py on a CUDA box. Creates ./.venv-oracle next to this script's copy.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv-oracle
export VIRTUAL_ENV="$PWD/.venv-oracle"
export PATH="$VIRTUAL_ENV/bin:$PATH"
uv pip install "torch" "torchaudio" --index-url https://download.pytorch.org/whl/cu130
uv pip install "nemo_toolkit[asr]" "kokoro>=0.9" "misaki[en]" soundfile
uv pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
python -c "import nemo, kokoro, misaki, spacy; spacy.load('en_core_web_sm'); print('env ok')"
