"""Metric definitions, carried over unchanged from research/vocab_v1/analyze_vocab1.py.

fair  = whisper_normalizer EnglishTextNormalizer on both sides, then jiwer
strict = lowercase and strip only
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import class_rates, english

_NORM = None
_cache: dict[str, str] = {}


def fair(t: str) -> str:
    global _NORM
    if _NORM is None:
        from whisper_normalizer.english import EnglishTextNormalizer

        _NORM = EnglishTextNormalizer()
    t = (t or "").strip()
    if t not in _cache:
        _cache[t] = _NORM(t)
    return _cache[t]


def strict_lc(t: str) -> str:
    return (t or "").strip().lower()


def squash(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def hit_norm(term: str, text: str) -> bool:
    nt, nx = fair(term), fair(text)
    return bool(nt) and re.search(r"(?<![a-z0-9])" + re.escape(nt) + r"(?![a-z0-9])",
                                  nx) is not None


def hit_loose(term: str, text: str) -> bool:
    s = squash(term)
    return bool(s) and s in squash(text)


def wer(refs, hyps, norm) -> float:
    import jiwer

    if not refs:
        return 0.0
    nr = [norm(r) or "<empty>" for r in refs]
    nh = [norm(h) or "<empty>" for h in hyps]
    return float(jiwer.wer(nr, nh))


def row_wer(ref: str, hyp: str, norm) -> float:
    import jiwer

    r, h = norm(ref) or "<empty>", norm(hyp) or "<empty>"
    return float(jiwer.wer(r, h))


_pat_cache: dict[str, re.Pattern | None] = {}


def term_pat(term: str):
    if term not in _pat_cache:
        t = (term or "").strip()
        if not t or len(squash(t)) < 2 or (t.isalpha() and t.islower() and t.lower() in english()):
            _pat_cache[term] = None      # plain English word: not a term insertion
        else:
            _pat_cache[term] = re.compile(
                r"(?<![A-Za-z0-9_./-])" + re.escape(t) + r"(?![A-Za-z0-9_./-])")
    return _pat_cache[term]


def false_inserts(vocab, out_text, reference, raw_text, true_term: str = ""):
    """Terms written into the output that the reference does not contain.

    bad        - the literal metric
    bad_strict - and absent from the ASR hypothesis, so the refiner really wrote it
    bad_unrel  - and not a spelling variant of the clip's own term
    """
    bad, bad_strict, bad_unrel = [], [], []
    st = squash(true_term)
    for t in vocab or []:
        p = term_pat(t)
        if p is None or not p.search(out_text or ""):
            continue
        if p.search(reference or ""):
            continue
        bad.append(t)
        if p.search(raw_text or ""):
            continue
        bad_strict.append(t)
        sq = squash(t)
        if st and (sq in st or st in sq):
            continue
        bad_unrel.append(t)
    return bad, bad_strict, bad_unrel


def read_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p, rows) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- blocks
def term_block(refs, raws, terms, hyps, vocabs, raw_rw, ret_has, raw_fi=None):
    n = len(refs)
    hn = [hit_norm(t, h) for t, h in zip(terms, hyps)]
    hl = [hit_loose(t, h) for t, h in zip(terms, hyps)]
    rw = [row_wer(a, b, fair) for a, b in zip(refs, hyps)]
    fi = [false_inserts(v, h, r, x, tt)
          for v, h, r, x, tt in zip(vocabs, hyps, refs, raws, terms)]
    unrel = sum(1 for b in fi if b[2]) / n
    out = {
        "n": n,
        "term_hit_norm": sum(hn) / n,
        "term_hit_loose": sum(hl) / n,
        "fair_wer": wer(refs, hyps, fair),
        "strict_lc_wer": wer(refs, hyps, strict_lc),
        "damage_rate": sum(1 for a, b in zip(rw, raw_rw) if a > b + 1e-9) / n,
        "win": sum(1 for a, b in zip(rw, raw_rw) if a < b - 1e-9),
        "loss": sum(1 for a, b in zip(rw, raw_rw) if a > b + 1e-9),
        "false_insert_rate": sum(1 for b in fi if b[0]) / n,
        "false_insert_rate_strict": sum(1 for b in fi if b[1]) / n,
        "false_insert_rate_unrelated": unrel,
        "false_insert_spurious": sum(1 for b, h in zip([x[2] for x in fi], hn) if b and h) / n,
        "classes": class_rates(list(zip(refs, hyps))),
    }
    out["tie"] = n - out["win"] - out["loss"]
    if raw_fi is not None:
        out["false_insert_raw_floor_unrelated"] = raw_fi
        out["false_insert_excess_unrelated"] = unrel - raw_fi
    for name, want in (("term_retrieved", True), ("term_missing", False)):
        sel = [(a, b, c) for a, b, c, w in zip(hn, rw, raw_rw, ret_has) if bool(w) is want]
        out[f"n_{name}"] = len(sel)
        out[f"hit_{name}"] = (sum(1 for s in sel if s[0]) / len(sel)) if sel else None
        out[f"damage_{name}"] = (sum(1 for s in sel if s[1] > s[2] + 1e-9) / len(sel)) if sel \
            else None
    return out


def real_block(refs, ins, hyps, vocabs, raw_rw, raw_fi=None):
    n = len(refs)
    rw = [row_wer(a, b, fair) for a, b in zip(refs, hyps)]
    win = sum(1 for a, b in zip(rw, raw_rw) if a < b - 1e-9)
    loss = sum(1 for a, b in zip(rw, raw_rw) if a > b + 1e-9)
    fi = [false_inserts(v, h, r, x) for v, h, r, x in zip(vocabs, hyps, refs, ins)]
    nr = [len(fair(a).split()) for a in refs]
    nh = [len(fair(b).split()) for b in hyps]
    worst = max(range(n), key=lambda i: rw[i] * nr[i])
    keep = [i for i in range(n) if i != worst]
    unrel = sum(1 for b in fi if b[2]) / n
    out = {
        "n": n,
        "fair_wer": wer(refs, hyps, fair),
        "strict_lc_wer": wer(refs, hyps, strict_lc),
        "fair_wer_drop_worst_row": wer([refs[i] for i in keep], [hyps[i] for i in keep], fair),
        "runaway_rows": sum(1 for a, b in zip(nr, nh) if b > 2 * max(a, 1)),
        "truncated_rows": sum(1 for a, b in zip(nr, nh) if b < 0.5 * a),
        "damage_rate": loss / n, "win": win, "tie": n - win - loss, "loss": loss,
        "false_insert_rate": sum(1 for b in fi if b[0]) / n,
        "false_insert_rate_strict": sum(1 for b in fi if b[1]) / n,
        "false_insert_rate_unrelated": unrel,
        "classes": class_rates(list(zip(refs, hyps))),
    }
    if raw_fi is not None:
        out["false_insert_raw_floor_unrelated"] = raw_fi
        out["false_insert_excess_unrelated"] = unrel - raw_fi
    return out
