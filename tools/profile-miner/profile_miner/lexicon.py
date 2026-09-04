"""Tech lexicon: public names a developer might dictate, with spoken forms precomputed offline.

Build (dev): python -m profile_miner lexicon build --cache DIR  (DIR holds pypi.json, crates.json,
hf_models.json, brew_formulae.txt, brew_casks.txt from tools/profile-miner/lexicon/fetch.sh) plus
lexicon/curated.txt. Writes lexicon/lexicon.json: {key: {"term", "cats"}}.
Only public lists and the curated glossary go in here, never a user's mined candidates.
"""
import json
import re
import sys
from pathlib import Path

LEX_DIR = Path(__file__).resolve().parent.parent / "lexicon"
MIN_LEN, MAX_LEN = 2, 40
# Model-id tokens that are sizes, variants or noise, not names.
HF_TOKEN_SKIP = re.compile(r"^(?:\d+(?:\.\d+)?[bBmMkK]?|v\d+(?:\.\d+)*|instruct|chat|base|hf|gguf|awq|gptq|fp8|fp16|bf16|int4|int8|"
                           r"uncased|cased|en|multilingual|small|base|large|xl|xxl|tiny|mini|medium|preview|it|sft|dpo|rl|"
                           r"merged|lora|adapter|onnx|q\d.*|ft|finetuned?|distilled?|exp|test|demo|model|models)$", re.I)


def norm(t: str) -> str:
    return t.strip().lower()


def load_words():
    """The miner's English class: system dictionary plus google-10000 with suffix rules (files, tools, runs)."""
    from .candidates import English
    return English()


def add(lex, term, cat, english, allow_english=False):
    term = term.strip()
    if not (MIN_LEN <= len(term) <= MAX_LEN) or not any(c.isalpha() for c in term):
        return
    key = norm(term)
    if not allow_english and " " not in key and key in english:
        return  # "requests", "rich", "click": presence in sources says nothing
    e = lex.setdefault(key, {"term": term, "cats": []})
    if cat not in e["cats"]:
        e["cats"].append(cat)


def build(cache: Path):
    english = load_words()
    lex = {}
    cat = "misc"
    for line in open(LEX_DIR / "curated.txt"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.match(r"#\s*([a-z][\w-]*)\s*$", line)
            if m:
                cat = m.group(1)
            continue
        add(lex, line, cat, english, allow_english=True)
    n_curated = len(lex)
    p = cache / "pypi.json"
    if p.exists():
        for r in json.load(open(p))["rows"]:
            name = r["project"]
            add(lex, name, "pypi", english)
            m = re.match(r"nvidia-(.+?)(-cu\d+)?$", name)
            if m:
                add(lex, m.group(1), "pypi", english)
    p = cache / "crates.json"
    if p.exists():
        for name in json.load(open(p)):
            add(lex, name, "crate", english)
    for fn, cat2 in (("brew_formulae.txt", "brew"), ("brew_casks.txt", "cask")):
        p = cache / fn
        if p.exists():
            for line in open(p):
                name = line.strip().split("/")[-1]
                add(lex, name, cat2, english)
    p = cache / "hf_models.json"
    if p.exists():
        for mid in json.load(open(p)):
            org, _, name = mid.partition("/")
            add(lex, org, "hf-org", english)
            add(lex, name, "hf-model", english)
            for tok in re.split(r"[-_]", name):
                if len(tok) >= 3 and not HF_TOKEN_SKIP.match(tok):
                    add(lex, tok, "hf-model", english)
    out = LEX_DIR / "lexicon.json"
    json.dump(lex, open(out, "w"), indent=0, ensure_ascii=False)
    cats = {}
    for e in lex.values():
        for c in e["cats"]:
            cats[c] = cats.get(c, 0) + 1
    print(f"[lexicon] {len(lex)} entries ({n_curated} curated) -> {out}; {cats}", file=sys.stderr)
    return lex


def load():
    return json.load(open(LEX_DIR / "lexicon.json"))
