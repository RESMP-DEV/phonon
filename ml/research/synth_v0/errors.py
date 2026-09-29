"""ASR error classes and error-model load/derive.

Classes match the grok-asr brief: ENTITY, FUNCTION, ORTHOGRAPHY, NEAR_MISS, DROP, INSERT.
If research/asr_errors_v0 lands a classifier, import it; otherwise use these rules.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ERROR_MODEL_LOCAL, ERROR_MODEL_SIBLING, TRAIN_JSONL  # noqa: E402

FUNCTION_WORDS = frozenset(
    """
    a an the and or but if then else when where why how what who whom whose which
    this that these those i you he she it we they me him her us them my your his
    its our their mine yours hers ours theirs is am are was were be been being
    do does did doing done have has had having can could should would may might
    must will shall need not no nor none never nothing of to in on at for from
    with without into onto upon over under out up down off about around between
    through during before after while since until as than so too very just also
    like uh um hmm ah oh yeah yes yep yup ok okay actually basically anyway
    anyways kinda sort maybe perhaps well still even only really just please
    wait like
    """.split()
)

DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}

CONTRACTIONS = {
    "i'm": "i am",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "i've": "i have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "let's": "let us",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
}

CLASSES = ("ENTITY", "FUNCTION", "ORTHOGRAPHY", "NEAR_MISS", "DROP", "INSERT")


def try_import_sibling_classifier():
    sibling = Path("/home/user/phonon/research/asr_errors_v0")
    if not sibling.exists():
        return None
    sys.path.insert(0, str(sibling))
    for name in ("classify", "classifier", "errors", "error_classes"):
        try:
            mod = __import__(name)
        except Exception:
            continue
        for attr in ("classify_pair", "classify_alignment", "classify"):
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
    return None


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']", text or "") if t.strip()]


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text or "")


def ortho_norm(tok: str) -> str:
    t = tok.lower().strip(".,;:!?\"'`()[]{}")
    t = CONTRACTIONS.get(t, t)
    if t.isdigit() and t in DIGIT_WORDS:
        t = DIGIT_WORDS[t]
    t = t.replace("'", "")
    return t


def is_function(tok: str) -> bool:
    return ortho_norm(tok) in FUNCTION_WORDS or tok.lower() in FUNCTION_WORDS


_SIBLING_ANALYZE = None
_SIBLING_COMMON = None


def _load_sibling():
    global _SIBLING_ANALYZE, _SIBLING_COMMON
    if _SIBLING_ANALYZE is not None:
        return _SIBLING_ANALYZE, _SIBLING_COMMON
    sibling = Path("/home/user/phonon/research/asr_errors_v0")
    if not (sibling / "analyze.py").exists():
        _SIBLING_ANALYZE = False
        return None, None
    if str(sibling) in sys.path:
        sys.path.remove(str(sibling))
    sys.path.insert(0, str(sibling))
    cached = sys.modules.get("common")
    if cached is not None and not hasattr(cached, "CLIPS_PATH"):
        del sys.modules["common"]
    import analyze as _analyze  # type: ignore

    _SIBLING_ANALYZE = _analyze
    _SIBLING_COMMON = _analyze.load_english()
    return _SIBLING_ANALYZE, _SIBLING_COMMON


def is_entity_shape(tok: str) -> bool:
    analyze, common = _load_sibling()
    if analyze and common is not None:
        return bool(analyze.is_entity(tok, common))
    if not tok:
        return False
    if re.search(r"[a-z][A-Z]", tok) or "_" in tok or re.search(r"\d", tok):
        return True
    if re.fullmatch(r"[A-Z]{2,8}", tok):
        return True
    if "-" in tok and not is_function(tok) and len(tok) > 3:
        return True
    return False


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if abs(len(a) - len(b)) > 8:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def classify_sub(ref: str, hyp: str) -> str:
    if ortho_norm(ref) == ortho_norm(hyp):
        return "ORTHOGRAPHY"
    if is_function(ref) and is_function(hyp):
        return "FUNCTION"
    if is_entity_shape(ref) or is_entity_shape(hyp):
        return "ENTITY"
    ra, ha = ortho_norm(ref), ortho_norm(hyp)
    dist = levenshtein(ra, ha)
    if dist <= 2 or (min(len(ra), len(ha)) and dist / max(len(ra), len(ha)) <= 0.35):
        return "NEAR_MISS"
    if is_function(ref) or is_function(hyp):
        return "FUNCTION"
    return "NEAR_MISS"


def align_words(ref_words: list[str], hyp_words: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Needleman-Wunsch on lowercase keys; keep original tokens."""
    n, m = len(ref_words), len(hyp_words)
    if n + m == 0:
        return []
    INF = 10**9
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"
    for i in range(1, n + 1):
        ri = ref_words[i - 1].lower()
        for j in range(1, m + 1):
            hj = hyp_words[j - 1].lower()
            sub = dp[i - 1][j - 1] + (0 if ri == hj else 1)
            delete = dp[i - 1][j] + 1
            ins = dp[i][j - 1] + 1
            best = sub
            op = "E" if ri == hj else "S"
            if delete < best:
                best, op = delete, "D"
            if ins < best:
                best, op = ins, "I"
            dp[i][j] = best
            bt[i][j] = op
    ops: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in {"E", "S"}:
            ops.append((op, ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            ops.append(("D", ref_words[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp_words[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def classify_pair(target: str, raw: str) -> dict[str, Any]:
    """Classify errors of raw (hyp/input) vs target (ref). Prefer grok-asr rules."""
    analyze, common = _load_sibling()
    if analyze and common is not None:
        errors = analyze.align_errors(target, raw, common)
        counts = Counter()
        details = []
        for err in errors:
            cls = err.get("class") or "OTHER"
            n = int(err.get("ref_words") or 1)
            counts[cls] += n
            details.append(
                {
                    "op": err.get("type"),
                    "cls": cls,
                    "ref": err.get("ref"),
                    "hyp": err.get("hyp"),
                }
            )
        ref_w = analyze.tokenize(target)
        hyp_w = analyze.tokenize(raw)
        n_ref = max(len(ref_w), 1)
        rates = {c: counts[c] / n_ref for c in CLASSES}
        return {
            "n_ref": len(ref_w),
            "n_hyp": len(hyp_w),
            "counts": dict(counts),
            "rates": rates,
            "details": details,
            "classifier": "asr_errors_v0.align_errors",
        }
    ref_w = word_tokens(target)
    hyp_w = word_tokens(raw)
    ops = align_words(ref_w, hyp_w)
    counts = Counter()
    details = []
    for op, ref, hyp in ops:
        if op == "E":
            if ref is not None and hyp is not None and ref != hyp:
                cls = "ORTHOGRAPHY"
                counts[cls] += 1
                details.append({"op": "sub", "cls": cls, "ref": ref, "hyp": hyp})
            continue
        if op == "S":
            cls = classify_sub(ref or "", hyp or "")
            counts[cls] += 1
            details.append({"op": "sub", "cls": cls, "ref": ref, "hyp": hyp})
        elif op == "D":
            cls = "FUNCTION" if is_function(ref or "") else "DROP"
            counts[cls] += 1
            details.append({"op": "del", "cls": cls, "ref": ref, "hyp": None})
        elif op == "I":
            cls = "FUNCTION" if is_function(hyp or "") else "INSERT"
            counts[cls] += 1
            details.append({"op": "ins", "cls": cls, "ref": None, "hyp": hyp})
    n_ref = max(len(ref_w), 1)
    rates = {c: counts[c] / n_ref for c in CLASSES}
    return {
        "n_ref": len(ref_w),
        "n_hyp": len(hyp_w),
        "counts": dict(counts),
        "rates": rates,
        "details": details,
        "classifier": "local_fallback",
    }


def load_error_model(path: Path | None = None) -> dict[str, Any] | None:
    for candidate in (path, ERROR_MODEL_SIBLING, ERROR_MODEL_LOCAL):
        if candidate is None:
            continue
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            data["_loaded_from"] = str(candidate)
            return data
    return None


def _subs_from_unknown_schema(data: dict) -> dict[str, list[tuple[str, int]]]:
    out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    skip_src = set(CLASSES) | {"OTHER", "NONE"}
    for key in ("substitutions", "substitution_table", "entity_substitutions", "subs"):
        table = data.get(key)
        if not isinstance(table, dict):
            continue
        for src, val in table.items():
            if src in skip_src:
                continue
            if isinstance(val, str):
                out[src].append((val, 1))
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        out[src].append((item, 1))
                    elif isinstance(item, dict):
                        hyp = item.get("hyp") or item.get("to") or item.get("raw")
                        cnt = int(item.get("count") or item.get("n") or 1)
                        if hyp:
                            out[src].append((str(hyp), cnt))
            elif isinstance(val, dict):
                for hyp, cnt in val.items():
                    try:
                        c = int(cnt)
                    except (TypeError, ValueError):
                        c = 1
                    out[src].append((str(hyp), c))
    return dict(out)


def _class_rate_map(src: dict) -> dict[str, float]:
    rates: dict[str, float] = {}
    for c in CLASSES:
        cell = src.get(c)
        if isinstance(cell, dict) and "per_word" in cell:
            rates[c] = float(cell["per_word"])
        elif isinstance(cell, (int, float)):
            rates[c] = float(cell)
    return rates


def _per_word_rates(blob: Any) -> dict[str, float]:
    if not isinstance(blob, dict):
        return {}
    if "ENTITY" in blob:
        return _class_rate_map(blob)
    for key in ("parakeet-tdt-0.6b-v2_wispr", "pooled_wispr"):
        if isinstance(blob.get(key), dict):
            return _class_rate_map(blob[key])
    for val in blob.values():
        if isinstance(val, dict) and "ENTITY" in val:
            return _class_rate_map(val)
    return {}


def normalize_error_model(data: dict[str, Any]) -> dict[str, Any]:
    rates = _per_word_rates(data.get("rates") or data.get("class_rates") or data.get("per_class_rates") or {})
    subs = _subs_from_unknown_schema(data)
    nested = data.get("substitutions")
    if isinstance(nested, dict) and ("ENTITY" in nested or "NEAR_MISS" in nested):
        for klass in ("ENTITY", "NEAR_MISS"):
            table = nested.get(klass) or {}
            if not isinstance(table, dict):
                continue
            for src, val in table.items():
                if src not in subs:
                    subs[src] = []
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and item.get("hyp"):
                            subs[src].append((str(item["hyp"]), int(item.get("count") or 1)))
                        elif isinstance(item, str):
                            subs[src].append((item, 1))
    folded: dict[str, list[tuple[str, int]]] = {}
    for src, pairs in subs.items():
        folded.setdefault(src, []).extend(pairs)
        folded.setdefault(src.lower(), []).extend(pairs)
    return {
        "source": data.get("_loaded_from") or data.get("source") or "unknown",
        "rates": {c: float(rates.get(c, 0.0)) for c in CLASSES},
        "substitutions": folded,
        "raw_keys": sorted(str(k) for k in data.keys()),
    }


def derive_from_wispr_train(path: Path = TRAIN_JSONL) -> dict[str, Any]:
    counts = Counter()
    n_ref = 0
    n_pairs = 0
    subs: dict[str, Counter] = defaultdict(Counter)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("source") or "")
            if not source.startswith("wispr"):
                continue
            target = row.get("target") or ""
            raw = row.get("input") or ""
            if not target.strip() or not raw.strip():
                continue
            cls = classify_pair(target, raw)
            n_ref += cls["n_ref"]
            n_pairs += 1
            for k, v in cls["counts"].items():
                counts[k] += v
            for d in cls["details"]:
                if d["op"] == "sub" and d.get("ref") and d.get("hyp"):
                    subs[d["ref"]][d["hyp"]] += 1
    denom = max(n_ref, 1)
    rates = {c: counts[c] / denom for c in CLASSES}
    sub_table = {
        src: dict(cnt.most_common(8))
        for src, cnt in subs.items()
        if sum(cnt.values()) >= 2 or is_entity_shape(src)
    }
    model = {
        "source": f"derived_from_corrector_train_wispr:{path}",
        "n_pairs": n_pairs,
        "n_ref_tokens": n_ref,
        "counts": dict(counts),
        "rates": rates,
        "substitutions": sub_table,
    }
    ERROR_MODEL_LOCAL.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    return model


def get_error_model(prefer_sibling: bool = True) -> dict[str, Any]:
    if prefer_sibling:
        loaded = load_error_model(ERROR_MODEL_SIBLING)
        if loaded:
            return normalize_error_model(loaded)
    local = load_error_model(ERROR_MODEL_LOCAL)
    if local and "derived_from" in str(local.get("source") or local.get("_loaded_from") or ""):
        return normalize_error_model(local)
    derived = derive_from_wispr_train()
    return normalize_error_model({**derived, "_loaded_from": str(ERROR_MODEL_LOCAL)})
