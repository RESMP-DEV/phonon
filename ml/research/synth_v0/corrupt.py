"""Corrupt clean utterances into ASR-style raw text via the measured error model."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from errors import (  # noqa: E402
    CLASSES,
    CONTRACTIONS,
    DIGIT_WORDS,
    classify_pair,
    get_error_model,
    is_entity_shape,
    is_function,
    word_tokens,
)
from paths import CLEAN_PATH, DATA_ROOT, PAIRS_PATH, SYSTEM_PROMPT  # noqa: E402

DIGIT_TO_WORD = dict(DIGIT_WORDS)
WORD_TO_DIGIT = {v: k for k, v in DIGIT_TO_WORD.items()}
INV_CONTRACTIONS = {v: k for k, v in CONTRACTIONS.items()}

HOMOPHONES = {
    "write": "right",
    "right": "write",
    "for": "four",
    "four": "for",
    "to": "too",
    "too": "to",
    "two": "to",
    "see": "sea",
    "sync": "sink",
    "kernel": "colonel",
    "cache": "cash",
    "queue": "cue",
    "root": "route",
    "route": "root",
    "sum": "some",
    "some": "sum",
    "one": "won",
    "wait": "weight",
    "weight": "wait",
    "break": "brake",
    "serial": "cereal",
    "plain": "plane",
    "base": "bass",
    "cell": "sell",
    "site": "sight",
    "which": "witch",
    "their": "there",
    "there": "their",
    "they're": "there",
}

LETTER_WORDS = {
    "a": "ay",
    "b": "bee",
    "c": "see",
    "d": "dee",
    "e": "ee",
    "f": "eff",
    "g": "jee",
    "h": "aitch",
    "i": "eye",
    "j": "jay",
    "k": "kay",
    "l": "ell",
    "m": "em",
    "n": "en",
    "o": "oh",
    "p": "pee",
    "q": "cue",
    "r": "are",
    "s": "ess",
    "t": "tee",
    "u": "you",
    "v": "vee",
    "w": "double you",
    "x": "ex",
    "y": "why",
    "z": "zee",
}

FILLERS = ["uh", "um", "like", "you know"]


def split_ident(term: str) -> list[str]:
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", term)
    parts = parts.replace("_", " ").replace("-", " ").replace(".", " ")
    return [p for p in parts.split() if p]


def spell_acronym(term: str) -> str:
    if re.fullmatch(r"[A-Z]{2,8}", term):
        return " ".join(LETTER_WORDS.get(c.lower(), c.lower()) for c in term)
    return term.lower()


def phonetic_near_miss(term: str, rng: random.Random) -> str:
    if re.fullmatch(r"[A-Z]{2,8}", term) and rng.random() < 0.6:
        return spell_acronym(term)
    parts = split_ident(term)
    if len(parts) > 1 and rng.random() < 0.7:
        lowered = [p.lower() for p in parts]
        if rng.random() < 0.4:
            return " ".join(lowered)
        return "".join(lowered)
    t = term.lower()
    if t in HOMOPHONES and rng.random() < 0.5:
        return HOMOPHONES[t]
    t = t.replace("ph", "f").replace("ck", "k")
    t = re.sub(r"([aeiou])\1", r"\1", t)
    if t != term.lower():
        return t
    if len(t) > 4:
        i = rng.randrange(1, len(t) - 1)
        return t[:i] + t[i + 1 :]
    return t


def lookup_sub(term: str, substitutions: dict, rng: random.Random) -> str | None:
    stripped = term.strip(".,;:!?\"'`()[]{}")
    keys = [term, term.lower(), stripped, stripped.lower()]
    for key in keys:
        table = substitutions.get(key)
        if not table:
            continue
        pairs: list[tuple[str, int]] = []
        if isinstance(table, dict):
            pairs = [(str(h), max(int(c), 1)) for h, c in table.items()]
        elif isinstance(table, list):
            for item in table:
                if isinstance(item, tuple) and item:
                    pairs.append((str(item[0]), max(int(item[1]) if len(item) > 1 else 1, 1)))
                elif isinstance(item, dict) and item.get("hyp"):
                    pairs.append((str(item["hyp"]), max(int(item.get("count") or 1), 1)))
                elif isinstance(item, str):
                    pairs.append((item, 1))
        if pairs:
            hyps, wts = zip(*pairs)
            return rng.choices(list(hyps), weights=list(wts), k=1)[0]
    return None


def apply_orthography(tok: str, rng: random.Random) -> str:
    t = tok
    if rng.random() < 0.85:
        t = t.lower()
    if t.isdigit() and t in DIGIT_TO_WORD and rng.random() < 0.6:
        t = DIGIT_TO_WORD[t]
    elif t.lower() in WORD_TO_DIGIT and rng.random() < 0.3:
        t = WORD_TO_DIGIT[t.lower()]
    low = t.lower()
    if low in CONTRACTIONS and rng.random() < 0.5:
        t = CONTRACTIONS[low]
    elif low in INV_CONTRACTIONS and rng.random() < 0.3:
        t = INV_CONTRACTIONS[low]
    t = t.strip(".,;:!?")
    return t


def corrupt_text(target: str, model: dict, rng: random.Random, terms: list[str]) -> tuple[str, list[str]]:
    rates = model["rates"]
    substitutions = model.get("substitutions") or {}
    words = word_tokens(target)
    if not words:
        return target.lower(), ["ORTHOGRAPHY"]
    applied: list[str] = []
    out: list[str] = []
    n = len(words)
    budget = {}
    for c in CLASSES:
        expected = float(rates.get(c, 0.0)) * n
        base = int(expected)
        frac = expected - base
        budget[c] = max(0, base + (1 if rng.random() < frac else 0))
    used = {c: 0 for c in CLASSES}

    def can(cls: str) -> bool:
        return used[cls] < budget[cls]

    for i, w in enumerate(words):
        rolled = rng.random()
        entity_p = rates.get("ENTITY", 0.0)
        near_p = rates.get("NEAR_MISS", 0.0)
        ortho_p = rates.get("ORTHOGRAPHY", 0.0)
        drop_p = rates.get("DROP", 0.0)
        fn_p = rates.get("FUNCTION", 0.0)
        cls = None
        if (is_entity_shape(w) or w in terms) and can("ENTITY") and rolled < entity_p * 4:
            cls = "ENTITY"
        elif can("NEAR_MISS") and rolled < entity_p + near_p * 3:
            cls = "NEAR_MISS"
        elif can("ORTHOGRAPHY") and rolled < entity_p + near_p + ortho_p * 2:
            cls = "ORTHOGRAPHY"
        elif is_function(w) and can("FUNCTION") and rolled < entity_p + near_p + ortho_p + fn_p * 2:
            cls = "FUNCTION"
        elif (not is_function(w)) and can("DROP") and rolled < entity_p + near_p + ortho_p + fn_p + drop_p:
            cls = "DROP"

        if cls == "DROP":
            used["DROP"] += 1
            applied.append("DROP")
            continue
        if cls == "FUNCTION" and is_function(w):
            used["FUNCTION"] += 1
            applied.append("FUNCTION")
            if rng.random() < 0.5:
                continue
            out.append(rng.choice(FILLERS))
            continue
        if cls == "ENTITY":
            used["ENTITY"] += 1
            applied.append("ENTITY")
            hyp = lookup_sub(w, substitutions, rng)
            out.append(hyp if hyp else phonetic_near_miss(w, rng))
        elif cls == "NEAR_MISS":
            used["NEAR_MISS"] += 1
            applied.append("NEAR_MISS")
            hyp = lookup_sub(w, substitutions, rng)
            out.append(hyp if hyp else phonetic_near_miss(w, rng))
        elif cls == "ORTHOGRAPHY":
            used["ORTHOGRAPHY"] += 1
            applied.append("ORTHOGRAPHY")
            out.append(apply_orthography(w, rng))
        else:
            out.append(w)

        if can("INSERT") and rng.random() < rates.get("INSERT", 0.0):
            used["INSERT"] += 1
            applied.append("INSERT")
            out.append(rng.choice(FILLERS))
        if can("FUNCTION") and rng.random() < rates.get("FUNCTION", 0.0) * 0.5:
            used["FUNCTION"] += 1
            applied.append("FUNCTION")
            out.append(rng.choice(["uh", "like", "the", "a"]))

    if not out:
        out = [w.lower() for w in words]
        applied.append("ORTHOGRAPHY")
    text = " ".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    if not applied:
        applied.append("ORTHOGRAPHY")
        text = text.lower()
    return text, sorted(set(applied))


def chat_messages(raw: str, target: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw},
        {"role": "assistant", "content": target},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "error_model_pairs.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    model = get_error_model(prefer_sibling=True)
    print(f"error_model source={model.get('source')} rates={model.get('rates')}", flush=True)
    rng = random.Random(args.seed)
    n = 0
    with CLEAN_PATH.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            target = (row.get("text") or "").strip()
            if not target:
                continue
            raw, applied = corrupt_text(target, model, rng, row.get("terms") or [])
            n += 1
            out = {
                "id": f"synth_em_{row.get('id', n)}",
                "source": "synth_error_model",
                "input": raw,
                "target": target,
                "messages": chat_messages(raw, target),
                "route": "error_model",
                "terms": row.get("terms") or [],
                "error_classes_applied": applied,
                "prompt_id": row.get("prompt_id"),
                "setting": row.get("setting"),
            }
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            if args.limit and n >= args.limit:
                break
    print(f"wrote {n} error-model pairs -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
