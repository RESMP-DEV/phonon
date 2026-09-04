#!/bin/bash
# Fetch the public name lists the lexicon is built from into $1 (default: lexicon/cache).
set -euo pipefail
OUT=${1:-$(dirname "$0")/cache}; mkdir -p "$OUT"; cd "$OUT"
curl -sL https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json -o pypi.json
brew formulae > brew_formulae.txt 2>/dev/null || true
brew casks > brew_casks.txt 2>/dev/null || true
python3 - <<'PY'
import json, re, time, urllib.request
UA = {"User-Agent": "phonon-lexicon (https://github.com/Infatoshi/phonon)"}
crates = []
for page in range(1, 31):
    req = urllib.request.Request(f"https://crates.io/api/v1/crates?page={page}&per_page=100&sort=downloads", headers=UA)
    crates += [c["name"] for c in json.loads(urllib.request.urlopen(req, timeout=30).read())["crates"]]
    time.sleep(1.0)
json.dump(crates, open("crates.json", "w"))
models, url = [], "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=1000"
for _ in range(8):
    r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60)
    models += [m["id"] for m in json.loads(r.read())]
    m = re.search(r'<([^>]+)>; rel="next"', r.headers.get("Link", ""))
    if not m:
        break
    url = m.group(1)
json.dump(models, open("hf_models.json", "w"))
print("crates", len(crates), "hf models", len(models))
PY
