"""Spoken-register scorer: per-row features of dictation targets, real vs synthetic.

Seven features. Six are text-only (computable on a clean utterance); the seventh,
edit_per_word, needs an (input, target) pair and is only defined for corpora that
have both sides.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ~150 English function words (determiners, prepositions, conjunctions, auxiliaries, pronouns).
FUNCTION_WORDS = set("""
a an the this that these those my your his her its our their some any each every no none
i me you he him she it we us they them mine yours hers ours theirs myself yourself himself
herself itself ourselves yourselves themselves who whom whose which what where when why how
and or but nor so yet for because although though while whereas unless until since if then
than as whether either neither both
in on at by to from with without within into onto out up down over under above below
through across behind between among around against during before after off about near per
be am is are was were been being do does did doing done have has had having
can could shall should will would may might must ought need dare used
not n't never always also just only even still yet again too very really quite rather much
more most less least many few several such same other another all
there here now then once soon back well up out
of off onto upon via
""".split())

CONTRACTION_RE = re.compile(
    r"\b\w+['’](?:s|t|re|ve|ll|d|m|em)\b|\b(?:can|won|don|doesn|didn|isn|aren|wasn|weren|"
    r"haven|hasn|hadn|couldn|shouldn|wouldn|ain|gonna|wanna|gotta|kinda|sorta|lemme|gimme)\b",
    re.I,
)

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", "i'm",
                "i've", "i'll", "i'd", "we're", "we've", "we'll", "we'd"}

DISCOURSE_SINGLE = {"so", "like", "okay", "ok", "basically", "actually", "right"}
DISCOURSE_MULTI = ["kind of", "sort of", "i think", "i mean"]

# spoken sentence openers: pronouns, conjunctions, discourse markers
SPOKEN_OPENERS = set("""
i we you he she it they there here this that these those
and but so or because if when while then also plus though although yeah yep ok okay
like just now well actually basically right maybe let lets let's what how why who where
do does did don't doesn't can can't could would should will won't is are was were am have has had
my our your his her their its me us them one two
""".split())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
HEADING_RE = re.compile(r"^\s*#{1,6}\s|^\s*\*\*[^*]+\*\*\s*:?\s*$", re.M)
LIST_RE = re.compile(r"(?:^|\n)\s*(?:[-*•]\s+|\d+[.)]\s+)")
COLON_END_RE = re.compile(r":\s*(?:\n|$)")

FEATURES_TEXT = [
    "func_frac",
    "contraction_rate",
    "first_person_rate",
    "discourse_rate",
    "mean_sent_len",
    "np_start_frac",
]
FEATURES_ALL = FEATURES_TEXT + ["edit_per_word"]


def words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split((text or "").strip()) if p.strip()]
    return parts or ([text.strip()] if (text or "").strip() else [])


def text_features(text: str) -> dict[str, float] | None:
    ws = words(text)
    n = len(ws)
    if n < 3:
        return None
    low = [w.lower() for w in ws]
    lowtext = " " + " ".join(low) + " "

    func = sum(1 for w in low if w in FUNCTION_WORDS) / n
    contr = len(CONTRACTION_RE.findall(text)) / n
    fp = sum(1 for w in low if w in FIRST_PERSON) / n
    disc = sum(1 for w in low if w in DISCOURSE_SINGLE)
    for phrase in DISCOURSE_MULTI:
        disc += lowtext.count(" " + phrase + " ")
    disc = disc / n

    sents = sentences(text)
    mean_len = n / max(len(sents), 1)
    np_start = 0
    for s in sents:
        sw = words(s)
        if not sw:
            continue
        first = sw[0]
        if first.lower() in SPOKEN_OPENERS:
            continue
        if first[:1].isupper() or first.lower() not in FUNCTION_WORDS:
            np_start += 1
    np_frac = np_start / max(len(sents), 1)

    return {
        "func_frac": func,
        "contraction_rate": contr,
        "first_person_rate": fp,
        "discourse_rate": disc,
        "mean_sent_len": mean_len,
        "np_start_frac": np_frac,
    }


def word_edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def pair_features(inp: str, target: str) -> dict[str, float] | None:
    feats = text_features(target)
    if feats is None:
        return None
    tw = [w.lower() for w in words(target)]
    iw = [w.lower() for w in words(inp or "")]
    feats["edit_per_word"] = word_edit_distance(iw, tw) / max(len(tw), 1)
    return feats


def structural_reject(text: str) -> str | None:
    """Hard rejects: headings, list markers, more than one colon-terminated line."""
    if not text or not text.strip():
        return "empty"
    if HEADING_RE.search(text):
        return "heading"
    if LIST_RE.search(text):
        return "list_marker"
    if len(COLON_END_RE.findall(text)) > 1:
        return "colon_lines"
    if text.count(":") > 2:
        return "colons"
    if re.search(r"```|\|\s*---", text):
        return "markdown"
    return None


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * q
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    return ys[f] if f == c else ys[f] * (c - k) + ys[c] * (k - f)


def summarize(rows: list[dict[str, float]], keys: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for k in keys:
        xs = [r[k] for r in rows if k in r and r[k] == r[k]]
        out[k] = {
            "p10": quantile(xs, 0.10),
            "median": quantile(xs, 0.50),
            "p90": quantile(xs, 0.90),
            "mean": sum(xs) / len(xs) if xs else float("nan"),
            "n": len(xs),
        }
    return out


def band_from(summary: dict[str, dict[str, float]], keys: list[str]) -> dict[str, tuple[float, float]]:
    return {k: (summary[k]["p10"], summary[k]["p90"]) for k in keys}


def in_band_count(feats: dict[str, float], band: dict[str, tuple[float, float]]) -> tuple[int, list[str]]:
    n = 0
    misses = []
    for k, (lo, hi) in band.items():
        v = feats.get(k)
        if v is not None and lo <= v <= hi:
            n += 1
        else:
            misses.append(k)
    return n, misses


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
