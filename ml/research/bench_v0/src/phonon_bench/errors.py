"""Word-error classifier, vendored verbatim from research/asr_errors_v0/analyze.py.

Lines 22-302 of that file are copied unchanged so the ENTITY/FUNCTION/DROP/INSERT rates this
benchmark reports are the same numbers the research tree reports. The only edit is the header:
the english word list is read from a fixture path instead of /data/phonon_asr_errors_v0/.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

ENGLISH_WORDS = Path("/data/phonon_bench_v0/sets/english_words.txt")


def set_english_words(path) -> None:
    global ENGLISH_WORDS, _ENGLISH_CACHE
    ENGLISH_WORDS = Path(path)
    _ENGLISH_CACHE = None


_ENGLISH_CACHE: set[str] | None = None


def english() -> set[str]:
    global _ENGLISH_CACHE
    if _ENGLISH_CACHE is None:
        _ENGLISH_CACHE = load_english()
    return _ENGLISH_CACHE


CLASSES = (
    "ENTITY",
    "FUNCTION",
    "ORTHOGRAPHY",
    "NEAR_MISS",
    "DROP",
    "INSERT",
    "OTHER",
)

FUNCTION_WORDS = frozenset(
    """
    a an the and or but if then so because as than that this these those
    i me my mine myself you your yours yourself he him his himself she her hers herself
    it its itself we us our ours ourselves they them their theirs themselves
    is am are was were be been being do does did doing done have has had having
    will would shall should can could may might must need
    to of in on for with at by from up about into over after under above below
    between through during before without within along across behind beyond plus versus vs
    not no nor too very just also only even still already yet
    um uh uhh umm er ah like yeah yep yup okay ok right well actually basically
    there here where when what which who whom whose how why
    some any each every all both few more most other such own
    out off down back away around again
    """.split()
)

_CAMEL = re.compile(r"(?:[a-z][A-Z]|[A-Z]{2,}[a-z]|[A-Za-z]\d|\d[A-Za-z])")
_STRIP = re.compile(r"^[^\w+#]+|[^\w+#]+$")
_NORM = None


def fair_norm(text: str) -> str:
    global _NORM
    if _NORM is None:
        from whisper_normalizer.english import EnglishTextNormalizer

        _NORM = EnglishTextNormalizer()
    return _NORM((text or "").strip())


def strict_lc(text: str) -> str:
    return (text or "").strip().lower()


def load_english() -> set[str]:
    words: set[str] = set()
    if ENGLISH_WORDS.exists():
        for line in ENGLISH_WORDS.read_text(encoding="utf-8").splitlines():
            token = line.strip().lower()
            if token:
                words.add(token)
    words.update(FUNCTION_WORDS)
    return words


def core_token(surface: str) -> str:
    return _STRIP.sub("", surface or "")


def is_camel(token: str) -> bool:
    return bool(_CAMEL.search(token))


def is_acronym(token: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", token)
    return len(letters) >= 2 and letters.isupper()


def is_entity(surface: str, common: set[str]) -> bool:
    """Technical term / identifier / name / acronym / code-like, or not common English.

    Rule: a reference token is ENTITY if after stripping wrapping punctuation it
    (1) contains a digit, underscore, or camelCase / letter-digit mix,
    (2) is a 2+ letter all-caps acronym,
    (3) contains an internal hyphen plus a digit or extra hyphen (code-like),
    or (4) its alphabetic core is length >= 2 and not in the top-20k English
    word list (wordfreq `en`, plus the function-word list).
    """
    token = core_token(surface)
    if not token:
        return False
    if any(ch.isdigit() for ch in token) or "_" in token or is_camel(token):
        return True
    if is_acronym(token):
        return True
    if token.count("-") >= 1 and (any(ch.isdigit() for ch in token) or token.count("-") >= 2):
        return True
    letters = re.sub(r"[^A-Za-z]", "", token).lower()
    if len(letters) >= 2 and letters not in common:
        return True
    return False


def is_function(surface: str) -> bool:
    token = core_token(surface).lower()
    return token in FUNCTION_WORDS


def soundex(word: str) -> str:
    token = re.sub(r"[^a-z]", "", word.lower())
    if not token:
        return ""
    first = token[0].upper()
    mapping = str.maketrans("bfpvcgjkqsxzdtlmnr", "111122222222334556")
    coded = token.translate(mapping)
    digits = []
    last = ""
    for ch, raw in zip(coded, token):
        if raw in "aeiouyhw":
            last = "0"
            continue
        digit = ch if ch.isdigit() else "0"
        if digit != "0" and digit != last:
            digits.append(digit)
        last = digit
    if digits and token[0] not in "aeiouyhw":
        first_digit = token[0].translate(mapping)
        if digits and digits[0] == first_digit:
            digits = digits[1:]
    return (first + "".join(digits) + "000")[:4]


def metaphone_key(word: str) -> str:
    token = re.sub(r"[^a-z]", "", word.lower())
    if not token:
        return ""
    token = token.replace("ph", "f").replace("kn", "n").replace("gn", "n")
    token = token.replace("wr", "r").replace("ck", "k").replace("x", "ks")
    out = []
    prev = ""
    for i, ch in enumerate(token):
        if ch in "aeiou" and i != 0:
            continue
        if ch == prev:
            continue
        if ch in "wy" and i != 0:
            continue
        out.append(ch)
        prev = ch
    return "".join(out)[:6]


def edit_distance(left: str, right: str) -> int:
    a, b = left.lower(), right.lower()
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
        if min(prev) > 2:
            return 3
    return prev[-1]


def is_near_miss(ref: str, hyp: str) -> bool:
    a = core_token(ref).lower()
    b = core_token(hyp).lower()
    if not a or not b:
        return False
    if edit_distance(a, b) <= 2:
        return True
    if len(a) >= 3 and len(b) >= 3 and (soundex(a) == soundex(b) or metaphone_key(a) == metaphone_key(b)):
        return True
    return False


def safe_wer(refs: list[str], hyps: list[str], normalizer) -> float:
    import jiwer

    if not refs:
        return 0.0
    nrefs, nhyps = [], []
    for ref, hyp in zip(refs, hyps, strict=True):
        r = normalizer(ref) or "<empty>"
        h = normalizer(hyp) or "<empty>"
        nrefs.append(r)
        nhyps.append(h)
    return float(jiwer.wer(nrefs, nhyps))


def tokenize(text: str) -> list[str]:
    return (text or "").strip().split()


def classify_chunk(
    ref_toks: list[str],
    hyp_toks: list[str],
    kind: str,
    common: set[str],
) -> str:
    if kind == "equal":
        return "NONE"
    if kind == "substitute":
        if fair_norm(" ".join(ref_toks)) == fair_norm(" ".join(hyp_toks)):
            return "ORTHOGRAPHY"
        if any(is_entity(tok, common) for tok in ref_toks):
            return "ENTITY"
        if any(is_function(tok) for tok in ref_toks) or any(is_function(tok) for tok in hyp_toks):
            return "FUNCTION"
        if (
            len(ref_toks) == 1
            and len(hyp_toks) == 1
            and not is_entity(ref_toks[0], common)
            and is_near_miss(ref_toks[0], hyp_toks[0])
        ):
            return "NEAR_MISS"
        return "OTHER"
    if kind == "delete":
        if any(is_entity(tok, common) for tok in ref_toks):
            return "ENTITY"
        if all(is_function(tok) for tok in ref_toks):
            return "FUNCTION"
        return "DROP"
    if kind == "insert":
        if all(is_function(tok) for tok in hyp_toks) and hyp_toks:
            return "FUNCTION"
        return "INSERT"
    return "OTHER"


def align_errors(ref: str, hyp: str, common: set[str]) -> list[dict[str, Any]]:
    import jiwer

    ref_toks = tokenize(ref)
    hyp_toks = tokenize(hyp)
    if not ref_toks and not hyp_toks:
        return []
    rjoin = " ".join(t.lower() for t in ref_toks) or "<empty>"
    hjoin = " ".join(t.lower() for t in hyp_toks) or "<empty>"
    processed = jiwer.process_words(rjoin, hjoin)
    chunks = processed.alignments[0]
    errors = []
    for chunk in chunks:
        kind = chunk.type
        if hasattr(kind, "value"):
            kind = kind.value
        kind = str(kind).lower()
        rspan = ref_toks[chunk.ref_start_idx : chunk.ref_end_idx]
        hspan = hyp_toks[chunk.hyp_start_idx : chunk.hyp_end_idx]
        if kind in {"equal", "match"}:
            # Lowercased alignment treats case-only diffs as equal; those are ORTHOGRAPHY.
            if rspan != hspan and (rspan or hspan):
                errors.append(
                    {
                        "type": "equal",
                        "class": "ORTHOGRAPHY",
                        "ref": " ".join(rspan),
                        "hyp": " ".join(hspan),
                        "ref_start": chunk.ref_start_idx,
                        "ref_end": chunk.ref_end_idx,
                        "hyp_start": chunk.hyp_start_idx,
                        "hyp_end": chunk.hyp_end_idx,
                        "ref_words": len(rspan),
                    }
                )
            continue
        klass = classify_chunk(rspan, hspan, kind, common)
        errors.append(
            {
                "type": kind,
                "class": klass,
                "ref": " ".join(rspan),
                "hyp": " ".join(hspan),
                "ref_start": chunk.ref_start_idx,
                "ref_end": chunk.ref_end_idx,
                "hyp_start": chunk.hyp_start_idx,
                "hyp_end": chunk.hyp_end_idx,
                "ref_words": len(rspan),
            }
        )
    return errors


CLASS_LIST = ["ENTITY", "FUNCTION", "ORTHOGRAPHY", "NEAR_MISS", "DROP", "INSERT", "OTHER"]


def class_rates(pairs, common=None) -> dict[str, float]:
    """Errors per 1000 reference words, identical to analyze_vocab1.class_rates."""
    common = english() if common is None else common
    c, ntok = Counter(), 0
    for ref, hyp in pairs:
        ntok += len(tokenize(ref))
        for e in align_errors(ref, hyp, common):
            c[e["class"]] += len(e["hyp"].split()) if e["type"] == "insert" else max(1, e["ref_words"])
    return {k: 1000.0 * c[k] / max(ntok, 1) for k in CLASS_LIST}
