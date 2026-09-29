"""Calibrated ASR-style corruption matching real_wispr class rates and WER mix."""
from __future__ import annotations

import importlib.util
import random
from collections import defaultdict
from pathlib import Path

from paths import REAL_RATES


def _load_v0(name: str):
    import sys

    v0 = Path("/home/user/phonon/research/synth_v0")
    if str(v0) not in sys.path:
        sys.path.append(str(v0))
    path = v0 / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"synth_v0_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_v0_corrupt = _load_v0("corrupt")
_v0_errors = _load_v0("errors")
HOMOPHONES = _v0_corrupt.HOMOPHONES
apply_orthography = _v0_corrupt.apply_orthography
lookup_sub = _v0_corrupt.lookup_sub
phonetic_near_miss = _v0_corrupt.phonetic_near_miss
import re

CLASSES = _v0_errors.CLASSES
is_entity_shape = _v0_errors.is_entity_shape
is_function = _v0_errors.is_function
# Keep snake_case / kebab identifiers as one token so we do not invent ENTITY splits.
TOKEN_RE = re.compile(
    r"--[A-Za-z][A-Za-z0-9-]*|[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+|[A-Za-z0-9']+"
)


def word_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")

FUNCTION_SUBS = {
    "the": ["a", "that", "this"],
    "a": ["the", "uh"],
    "an": ["a", "the"],
    "to": ["too", "the"],
    "too": ["to", "two"],
    "of": ["a", "off"],
    "in": ["on", "and"],
    "on": ["in", "of"],
    "it": ["that", "this"],
    "that": ["it", "the"],
    "we": ["i", "they"],
    "i": ["we", "you"],
    "and": ["an", "in"],
    "is": ["it's", "was"],
    "was": ["is", "were"],
    "for": ["four", "the"],
    "this": ["the", "that"],
    "with": ["the", "which"],
}

# extra filler events from real wispr pairs (input minus target)
FILLER_WEIGHTS = [
    ("like", 9743),
    ("um", 736),
    ("uh", 430),
    ("you know", 465),
]
VISIBLE = ("ENTITY", "FUNCTION", "NEAR_MISS", "DROP", "INSERT")
VISIBLE_SUM = sum(REAL_RATES[c] for c in VISIBLE)


def stochastic_round(expected: float, rng: random.Random) -> int:
    if expected <= 0:
        return 0
    base = int(expected)
    frac = expected - base
    return base + (1 if rng.random() < frac else 0)


def pick_filler(rng: random.Random) -> str:
    labels, weights = zip(*FILLER_WEIGHTS)
    return rng.choices(list(labels), weights=list(weights), k=1)[0]


def rates_for_target_wer(target_wer: float, hit_scale: float) -> dict[str, float]:
    rates = dict(REAL_RATES)
    if target_wer <= 0.005:
        for c in VISIBLE:
            rates[c] = 0.0
        return rates
    # Scale visible classes so expected ops/n ≈ target_wer * hit_scale.
    scale = (target_wer * hit_scale) / max(VISIBLE_SUM, 1e-9)
    for c in VISIBLE:
        rates[c] = REAL_RATES[c] * scale
    return rates


def _sample_idxs(idxs: list[int], k: int, rng: random.Random, taken: set[int]) -> list[int]:
    avail = [i for i in idxs if i not in taken]
    if not avail or k <= 0:
        return []
    k = min(k, len(avail))
    chosen = rng.sample(avail, k)
    taken.update(chosen)
    return chosen


def corrupt_text(
    target: str,
    rng: random.Random,
    terms: list[str],
    substitutions: dict,
    target_wer: float,
    hit_scale: float = 1.0,
) -> tuple[str, list[str]]:
    words = word_tokens(target)
    if not words:
        return target.lower(), ["ORTHOGRAPHY"]
    n = len(words)
    rates = rates_for_target_wer(target_wer, hit_scale)
    budget = {c: stochastic_round(float(rates[c]) * n, rng) for c in CLASSES}
    term_set = set(terms or [])
    entity_idxs = [i for i, w in enumerate(words) if is_entity_shape(w) or w in term_set]
    function_idxs = [i for i, w in enumerate(words) if is_function(w)]
    content_idxs = [i for i, w in enumerate(words) if not is_function(w)]
    all_idxs = list(range(n))
    taken: set[int] = set()
    drop: set[int] = set()
    replace: dict[int, str] = {}
    inserts_after: dict[int, list[str]] = defaultdict(list)
    applied: list[str] = []

    def _hyp_or_phonetic(tok: str) -> str | None:
        hyp = lookup_sub(tok, substitutions, rng)
        if hyp is None:
            return None
        h = str(hyp).strip()
        if not h or h in {"<eps>", "<unk>", "<pad>", "<blank>", "ε"}:
            return ""
        if "<" in h or ">" in h:
            return None
        return h

    def _gentle_entity(tok: str) -> str:
        # One surface token, not a camelCase split (those inflate ENTITY ops ~4x).
        hyp = _hyp_or_phonetic(tok)
        if hyp == "":
            return ""
        if hyp:
            parts = str(hyp).split()
            return parts[0] if parts else tok.lower()
        t = tok
        roll = rng.random()
        if "_" in t or "-" in t:
            return t.replace("_", "").replace("-", "") if roll < 0.5 else t.lower()
        if roll < 0.55:
            return t.lower()
        if roll < 0.8 and len(t) > 4:
            i = rng.randrange(1, len(t) - 1)
            return t[:i] + t[i + 1 :]
        return phonetic_near_miss(t, rng).split()[0]

    for i in _sample_idxs(entity_idxs, budget["ENTITY"], rng, taken):
        hyp = _gentle_entity(words[i])
        if hyp == "":
            drop.add(i)
        else:
            replace[i] = hyp
        applied.append("ENTITY")

    near_pool = [i for i in all_idxs if i not in taken]
    for i in _sample_idxs(near_pool, budget["NEAR_MISS"], rng, taken):
        hyp = _hyp_or_phonetic(words[i])
        if hyp == "":
            drop.add(i)
        elif hyp:
            replace[i] = hyp
        elif words[i].lower() in HOMOPHONES:
            replace[i] = HOMOPHONES[words[i].lower()]
        else:
            replace[i] = phonetic_near_miss(words[i], rng)
        applied.append("NEAR_MISS")

    fn_pool = [i for i in function_idxs if i not in taken]
    for i in _sample_idxs(fn_pool, budget["FUNCTION"], rng, taken):
        applied.append("FUNCTION")
        roll = rng.random()
        if roll < 0.45:
            drop.add(i)
        elif roll < 0.70:
            replace[i] = "like"
        else:
            opts = FUNCTION_SUBS.get(words[i].lower())
            replace[i] = rng.choice(opts) if opts else rng.choice(["the", "a", "to"])

    drop_pool = [i for i in content_idxs if i not in taken]
    for i in _sample_idxs(drop_pool, budget["DROP"], rng, taken):
        drop.add(i)
        applied.append("DROP")

    ortho_pool = [i for i in all_idxs if i not in taken and i not in drop]
    for i in _sample_idxs(ortho_pool, budget["ORTHOGRAPHY"], rng, set()):
        replace[i] = apply_orthography(replace.get(i, words[i]), rng)
        applied.append("ORTHOGRAPHY")

    insert_budget = budget["INSERT"]
    insert_pos = list(range(n))
    rng.shuffle(insert_pos)
    for pos in insert_pos:
        if insert_budget <= 0:
            break
        if rng.random() < 0.45:
            filler = pick_filler(rng)
            tokens = filler.split()
        else:
            # Stutter a content word so the sibling classifier counts INSERT,
            # not FUNCTION (like/um/uh live in the function-word list).
            content = [words[j] for j in range(n) if not is_function(words[j])]
            tokens = [rng.choice(content)] if content else pick_filler(rng).split()
        if len(tokens) > insert_budget:
            continue
        inserts_after[pos].extend(tokens)
        insert_budget -= len(tokens)
        applied.append("INSERT")

    out: list[str] = []
    for i, w in enumerate(words):
        if i in drop:
            out.extend(inserts_after.get(i, []))
            continue
        out.append(replace.get(i, w))
        out.extend(inserts_after.get(i, []))
    if not out:
        out = [w.lower() for w in words]
        applied.append("ORTHOGRAPHY")
    text = " ".join(out)
    text = " ".join(text.split()).strip()
    if not applied:
        applied.append("ORTHOGRAPHY")
        text = text.lower()
    return text, sorted(set(applied))
