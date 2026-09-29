"""Step 1: is the held-out term PRESENT PHONETICALLY in the raw ASR hypothesis?

For every row of /data/phonon_term_eval_v0/asr.jsonl compute a phonetic
representation of the term and of every hypothesis word n-gram whose syllable
count is within +/-2 of the term's, then call the term phonetically present if
the best normalized phoneme edit distance is below 0.35.

Primary phonetics: g2p_en (CMUdict + neural fallback), word-level cached.
Cross-check: jellyfish metaphone on the squashed strings.
"""
from __future__ import annotations
import json, re, sys
from collections import defaultdict
from pathlib import Path

import jellyfish
from g2p_en import G2p
from whisper_normalizer.english import EnglishTextNormalizer

DATA = Path("/data/phonon_term_eval_v0")
OUT = Path("/home/user/phonon/research/term_eval_v0")
THRESH = 0.35
MAX_NGRAM = 8

_NORM = EnglishTextNormalizer()
_g2p = G2p()

LETTER = {
    "a": "EY", "b": "B IY", "c": "S IY", "d": "D IY", "e": "IY", "f": "EH F",
    "g": "JH IY", "h": "EY CH", "i": "AY", "j": "JH EY", "k": "K EY",
    "l": "EH L", "m": "EH M", "n": "EH N", "o": "OW", "p": "P IY",
    "q": "K Y UW", "r": "AA R", "s": "EH S", "t": "T IY", "u": "Y UW",
    "v": "V IY", "w": "D AH B AH L Y UW", "x": "EH K S", "y": "W AY",
    "z": "Z IY",
    "0": "Z IH R OW", "1": "W AH N", "2": "T UW", "3": "TH R IY",
    "4": "F AO R", "5": "F AY V", "6": "S IH K S", "7": "S EH V AH N",
    "8": "EY T", "9": "N AY N",
}

_PH2CH: dict[str, str] = {}


def ph_chars(phones: list[str]) -> str:
    """Map an ARPAbet sequence (stress stripped) to a string for edit distance."""
    out = []
    for p in phones:
        p = re.sub(r"\d", "", p).strip()
        if not p or p == " ":
            continue
        if p not in _PH2CH:
            _PH2CH[p] = chr(0x4E00 + len(_PH2CH))
        out.append(_PH2CH[p])
    return "".join(out)


_word_cache: dict[str, str] = {}


def word_ph(w: str) -> str:
    if w not in _word_cache:
        try:
            _word_cache[w] = ph_chars(_g2p(w))
        except Exception:
            _word_cache[w] = ph_chars([LETTER.get(c, "") for c in w.lower()])
    return _word_cache[w]


_syl_cache: dict[str, int] = {}


def word_syl(w: str) -> int:
    if w not in _syl_cache:
        try:
            _syl_cache[w] = sum(1 for p in _g2p(w) if re.search(r"\d", p))
        except Exception:
            _syl_cache[w] = max(1, len(re.findall(r"[aeiouy]+", w.lower())))
    return _syl_cache[w]


def split_ident(term: str) -> list[str]:
    """cudaDeviceSynchronize / snake_case / kebab-case -> word list."""
    parts = re.split(r"[^A-Za-z0-9]+", term)
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        for chunk in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", p):
            out.append(chunk)
    return out or [term]


def letters_ph(s: str) -> str:
    seq: list[str] = []
    for c in s.lower():
        if c in LETTER:
            seq.extend(LETTER[c].split())
    return ph_chars(seq)


def term_variants(term: str) -> list[str]:
    """Plausible spoken realizations of the term, as phoneme-char strings."""
    words = split_ident(term)
    v = set()
    # a) each sub-word pronounced as a word
    v.add("".join(word_ph(w) for w in words))
    # b) whole thing spelled letter by letter
    v.add(letters_ph(re.sub(r"[^A-Za-z0-9]", "", term)))
    # c) short all-caps chunks spelled out, the rest pronounced
    mixed = []
    for w in words:
        if w.isupper() and len(w) <= 6:
            mixed.append(letters_ph(w))
        elif w.isdigit():
            mixed.append(letters_ph(w))
        else:
            mixed.append(word_ph(w))
    v.add("".join(mixed))
    return [x for x in v if x]


def term_syl(term: str) -> int:
    words = split_ident(term)
    n = 0
    for w in words:
        if w.isupper() and len(w) <= 6:
            n += sum(1 for c in w if c.isalnum())
        else:
            n += max(1, word_syl(w))
    return max(1, n)


def ratio(a: str, b: str) -> float:
    if not a or not b:
        return 1.0
    return jellyfish.levenshtein_distance(a, b) / max(len(a), len(b))


def squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def hit_norm(term: str, text: str) -> bool:
    nt = _NORM(term or "")
    nx = _NORM(text or "")
    if not nt:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(nt) + r"(?![a-z0-9])", nx) is not None


def hit_loose(term: str, text: str) -> bool:
    st = squash(term)
    return bool(st) and st in squash(text)


_mp_cache: dict[str, str] = {}


def mp(s: str) -> str:
    if s not in _mp_cache:
        try:
            _mp_cache[s] = jellyfish.metaphone(s)
        except Exception:
            _mp_cache[s] = ""
    return _mp_cache[s]


def spelled_word_ph(w: str) -> str:
    """How the span word sounds if it is an acronym read letter by letter."""
    return letters_ph(w)


def best_match(term: str, hyp: str):
    """(best phoneme ratio, best metaphone ratio, matched hypothesis span)."""
    words = re.findall(r"[A-Za-z0-9']+", hyp or "")
    if not words:
        return 1.0, 1.0, ""
    tvars = term_variants(term)
    tsyl = term_syl(term)
    tmp = mp(squash(term))
    wph = [word_ph(w) for w in words]
    wlet = [spelled_word_ph(w) for w in words]
    # mixed: a short ALL-CAPS or digit token in the hypothesis is read out letter by letter
    wmix = [wlet[i] if ((w.isupper() and len(w) <= 6) or w.isdigit()) else wph[i]
            for i, w in enumerate(words)]
    wsy = [max(1, word_syl(w)) for w in words]
    wch = [max(1, len(re.sub(r"[^A-Za-z0-9]", "", w))) for w in words]
    best_p, best_m, best_span = 1.0, 1.0, ""
    for i in range(len(words)):
        a_ph = a_let = a_mix = ""
        a_sy = a_ch = 0
        for j in range(i, min(i + MAX_NGRAM, len(words))):
            a_ph += wph[j]
            a_let += wlet[j]
            a_mix += wmix[j]
            a_sy += wsy[j]
            a_ch += wch[j]
            if a_sy > tsyl + 2 and a_ch > tsyl + 2:
                break
            in_word = tsyl - 2 <= a_sy <= tsyl + 2
            in_let = tsyl - 2 <= a_ch <= tsyl + 2
            if not (in_word or in_let):
                continue
            svars = []
            if in_word:
                svars.append(a_ph)
                svars.append(a_mix)
            if in_let:
                svars.append(a_let)
                svars.append(a_mix)
            span = " ".join(words[i:j + 1])
            r = min(ratio(t, s) for t in tvars for s in svars if s)
            if r < best_p:
                best_p, best_span = r, span
            rm = ratio(tmp, mp(squash(span)))
            if rm < best_m:
                best_m = rm
    return best_p, best_m, best_span


def main() -> int:
    rows = [json.loads(l) for l in (DATA / "asr.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"rows={len(rows)}", flush=True)
    out_rows = []
    for k, r in enumerate(rows):
        term, hyp = r["term"], r.get("hyp", "")
        bp, bm, span = best_match(term, hyp)
        out_rows.append({
            "id": r["id"], "term": term, "kind": r.get("kind"), "voice": r.get("voice"),
            "reference": r.get("reference", ""), "hyp": hyp,
            "phon_ratio": round(bp, 4), "mp_ratio": round(bm, 4), "span": span,
            "phon_present": bp < THRESH, "mp_present": bm < THRESH,
            "hit_norm": hit_norm(term, hyp), "hit_loose": hit_loose(term, hyp),
        })
        if (k + 1) % 250 == 0:
            print(f"  {k+1}/{len(rows)}", flush=True)

    n = len(out_rows)
    pres = [r for r in out_rows if r["phon_present"]]
    absent = [r for r in out_rows if not r["phon_present"]]
    summary = {
        "n_rows": n,
        "threshold": THRESH,
        "phonetics": "g2p_en ARPAbet (word-level cache), jellyfish.metaphone cross-check",
        "phonetic_presence_rate": len(pres) / n,
        "metaphone_presence_rate": sum(r["mp_present"] for r in out_rows) / n,
        "string_hit_norm_overall": sum(r["hit_norm"] for r in out_rows) / n,
        "string_hit_loose_overall": sum(r["hit_loose"] for r in out_rows) / n,
        "hit_norm_given_present": (sum(r["hit_norm"] for r in pres) / len(pres)) if pres else None,
        "hit_norm_given_absent": (sum(r["hit_norm"] for r in absent) / len(absent)) if absent else None,
        "hit_loose_given_present": (sum(r["hit_loose"] for r in pres) / len(pres)) if pres else None,
        "hit_loose_given_absent": (sum(r["hit_loose"] for r in absent) / len(absent)) if absent else None,
        "n_present": len(pres), "n_absent": len(absent),
    }

    # per-term aggregation
    by_term: dict[str, list[dict]] = defaultdict(list)
    for r in out_rows:
        by_term[r["term"]].append(r)
    term_stats = {}
    for t, rs in by_term.items():
        term_stats[t] = {
            "kind": rs[0]["kind"], "clips": len(rs),
            "hit_norm": sum(x["hit_norm"] for x in rs) / len(rs),
            "hit_loose": sum(x["hit_loose"] for x in rs) / len(rs),
            "phon_present": sum(x["phon_present"] for x in rs) / len(rs),
            "mp_present": sum(x["mp_present"] for x in rs) / len(rs),
            "median_ratio": sorted(x["phon_ratio"] for x in rs)[len(rs) // 2],
        }

    # "dead" = the 58 terms of results.md: zero normalizer-level string recovery on all 9 clips
    dead = sorted([t for t, s in term_stats.items() if s["hit_norm"] == 0.0])
    dead_loose_zero = [t for t in dead if term_stats[t]["hit_loose"] == 0.0]
    dead_present = [t for t in dead if term_stats[t]["phon_present"] >= 0.5]
    dead_absent = [t for t in dead if term_stats[t]["phon_present"] < 0.5]
    summary["n_terms"] = len(term_stats)
    summary["n_dead_terms"] = len(dead)
    summary["n_dead_also_loose_zero"] = len(dead_loose_zero)
    summary["n_dead_present_respelled"] = len(dead_present)
    summary["n_dead_genuinely_absent"] = len(dead_absent)

    # by kind
    kind_stats: dict[str, dict] = {}
    for r in out_rows:
        k = r["kind"] or "?"
        d = kind_stats.setdefault(k, {"n": 0, "phon": 0, "hit": 0})
        d["n"] += 1
        d["phon"] += r["phon_present"]
        d["hit"] += r["hit_norm"]

    res = {
        "summary": summary, "terms": term_stats,
        "dead_present_respelled": dead_present, "dead_genuinely_absent": dead_absent,
        "kind_stats": kind_stats,
        "rows": out_rows,
    }
    (OUT / "phonetic_split.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ---- markdown ----
    L = []
    A = L.append
    A("# Term eval v0: phonetic presence vs string recovery\n")
    A(f"2,700 raw Parakeet-TDT-0.6b-v2 hypotheses over {summary['n_terms']} held-out terms. "
      "A term counts as PHONETICALLY PRESENT when the best normalized phoneme edit distance "
      f"between the term and any hypothesis word n-gram of comparable syllable count is < {THRESH}.\n")
    A("Phonetics: **g2p_en** (CMUdict with the neural OOV fallback), word-level cache, ARPAbet with stress "
      "stripped; three spoken realizations per term (sub-words as words, whole term spelled letter by letter, "
      "short all-caps chunks spelled and the rest pronounced), best of the three. "
      "`jellyfish.metaphone` on squashed strings is reported as a cross-check "
      "(jellyfish >= 1.0 no longer ships double metaphone).\n")
    A("## Headline\n")
    A("| metric | value |")
    A("|---|---:|")
    A(f"| rows | {n} |")
    A(f"| phonetic presence rate (g2p) | {summary['phonetic_presence_rate']:.3f} |")
    A(f"| phonetic presence rate (metaphone) | {summary['metaphone_presence_rate']:.3f} |")
    A(f"| string hit, normalizer | {summary['string_hit_norm_overall']:.3f} |")
    A(f"| string hit, loose | {summary['string_hit_loose_overall']:.3f} |")
    A("")
    A("## String recovery conditioned on phonetic presence\n")
    A("| phonetically present | n | string hit (norm) | string hit (loose) |")
    A("|---|---:|---:|---:|")
    A(f"| yes | {len(pres)} | {summary['hit_norm_given_present']:.3f} | {summary['hit_loose_given_present']:.3f} |")
    A(f"| no | {len(absent)} | {summary['hit_norm_given_absent']:.3f} | {summary['hit_loose_given_absent']:.3f} |")
    A("")
    A("## Phonetic presence by lexicon kind\n")
    A("| kind | clips | phonetic presence | string hit (norm) |")
    A("|---|---:|---:|---:|")
    for k in sorted(kind_stats, key=lambda x: -kind_stats[x]["n"]):
        d = kind_stats[k]
        A(f"| {k} | {d['n']} | {d['phon']/d['n']:.3f} | {d['hit']/d['n']:.3f} |")
    A("")
    A(f"## The {len(dead)} dead terms (normalizer-level string hit 0.00 on all 9 clips)\n")
    A(f"{len(dead_loose_zero)} of the {len(dead)} are also 0.00 under the loose (squashed-substring) match.\n")
    A(f"- present but respelled or split (phonetically present on >= 5/9 clips): **{len(dead_present)}**")
    A(f"- genuinely absent (the acoustics are gone): **{len(dead_absent)}**\n")
    A("### Respelled or split, 10 examples\n")
    for t in dead_present[:10]:
        ex = max(by_term[t], key=lambda r: -r["phon_ratio"] if False else r["phon_present"])
        ex = sorted(by_term[t], key=lambda r: r["phon_ratio"])[0]
        A(f"- `{t}` ({term_stats[t]['kind']}, phon {term_stats[t]['phon_present']:.2f}, "
          f"best ratio {ex['phon_ratio']:.2f}) heard as **{ex['span']}**")
        A(f"  - HYP: {ex['hyp']}")
    A("")
    A(f"### Genuinely absent, all {len(dead_absent)} (a failing clip shown for each)\n")
    for t in dead_absent:
        miss = [r for r in by_term[t] if not r["phon_present"]] or by_term[t]
        ex = sorted(miss, key=lambda r: r["phon_ratio"])[len(miss) // 2]
        A(f"- `{t}` ({term_stats[t]['kind']}, phonetically present on "
          f"{term_stats[t]['phon_present']*9:.0f}/9 clips) closest span on this clip "
          f"**{ex['span']}** (ratio {ex['phon_ratio']:.2f})")
        A(f"  - HYP: {ex['hyp']}")
    A("")
    A("### Full dead-term classification\n")
    A("| term | kind | phonetic presence | median ratio | class |")
    A("|---|---|---:|---:|---|")
    for t in dead:
        s = term_stats[t]
        cls = "respelled/split" if t in dead_present else "absent"
        A(f"| `{t}` | {s['kind']} | {s['phon_present']:.2f} | {s['median_ratio']:.2f} | {cls} |")
    A("")
    (OUT / "phonetic_split.md").write_text("\n".join(L), encoding="utf-8")

    print(json.dumps(summary, indent=1))
    print("STEP 1 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
