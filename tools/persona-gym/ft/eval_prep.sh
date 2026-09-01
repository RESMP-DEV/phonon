#!/bin/bash
# Rebuild the 20 held-out personas on the box (repos re-cloned from GitHub)
# and install a vLLM venv for serving candidates. Idempotent.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/gym && cd ~/gym
[ -d persona-gym ] || tar -xzf ~/ft/persona-gym-code.tgz
[ -f heldout_manifest.json ] || tar -xzf ~/ft/heldout.tgz
python3 - <<'EOF'
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "gym/persona-gym"))
from persona_gym import repos
G = Path.home() / "gym"
manifest = json.loads((G / "heldout_manifest.json").read_text())
cache = G / ".cache" / "repos"
fail = 0
for pname, rows in manifest.items():
    for r in rows:
        dest = G / "personas" / pname / "repos" / r["name"]
        if dest.exists():
            continue
        c = repos.clone_into_cache(r["slug"], cache)
        if c is None:
            fail += 1; print("FAIL", pname, r["slug"]); continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(c, dest)
print("reconstructed; failures:", fail)
EOF
rm -rf split && mkdir split
for p in personas/persona-*; do n=$(basename $p); mkdir -p split/$n; ln -s $PWD/$p split/$n/$n; done
[ -f /usr/share/dict/words ] || (sudo apt-get install -y -qq wamerican >/dev/null 2>&1 || true)
[ -d ~/.venv-vllm ] || (cd ~ && uv venv .venv-vllm --python 3.12 && . ~/.venv-vllm/bin/activate && uv pip install -q vllm)
echo EVAL_PREP_DONE
