"""Step 3: damage-visible reporting for the corrector systems.

Per system and eval set: fair WER, strict lowercase WER, paired row-level
win/tie/loss vs raw, damage rate, median per-row delta; plus what share of the
ORTHOGRAPHY error mass is sentence-initial capitalization or terminal
punctuation (i.e. fixable by a two-line rule), plus a filler probe on the
whisper normalizer.
"""
from __future__ import annotations
import json, re, statistics, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/asr_errors_v0")
from analyze import align_errors, load_english, tokenize  # noqa: E402

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

PRED = Path("/data/phonon_corrector_v0/eval_predictions")
VOCAB = Path("/data/phonon_vocab_v0/refined")
OUT = Path("/home/user/phonon/research/corrector_v0")
SETS = ["wispr_holdout120", "wispr_text_holdout"]
CLASSES = ["ENTITY", "FUNCTION", "ORTHOGRAPHY", "NEAR_MISS", "DROP", "INSERT", "OTHER"]

_NORM = EnglishTextNormalizer()
_c: dict[str, str] = {}


def fair(t):
    t = (t or "").strip()
    if t not in _c:
        _c[t] = _NORM(t)
    return _c[t]


def strict(t):
    return (t or "").strip().lower()


def corpus_wer(refs, hyps, norm):
    nr = [norm(r) or "<empty>" for r in refs]
    nh = [norm(h) or "<empty>" for h in hyps]
    return float(jiwer.wer(nr, nh))


def row_wer(ref, hyp, norm):
    r = norm(ref) or "<empty>"
    h = norm(hyp) or "<empty>"
    return float(jiwer.wer(r, h))


TERM_PUNCT = ".!?"
ALL_PUNCT = ".,!?;:…"


def strip_term(tok):
    return tok.rstrip(ALL_PUNCT)


def sentence_initial(ref_toks, i):
    if i == 0:
        return True
    prev = ref_toks[i - 1]
    return bool(prev) and prev[-1] in TERM_PUNCT


def ortho_kind(ref_toks, hyp_toks, rstart):
    """Classify one ORTHOGRAPHY chunk: is every difference sentence-initial
    capitalization or terminal punctuation?"""
    if len(ref_toks) != len(hyp_toks):
        return "other"
    kinds = set()
    for k, (rt, ht) in enumerate(zip(ref_toks, hyp_toks)):
        if rt == ht:
            continue
        i = rstart + k
        rs, hs = strip_term(rt), strip_term(ht)
        rsuf, hsuf = rt[len(rs):], ht[len(hs):]
        if rs != hs and rs.lower() != hs.lower():
            return "other"
        if rsuf != hsuf:
            if set(rsuf) | set(hsuf) <= set(TERM_PUNCT):
                kinds.add("terminal_punct")
            else:
                kinds.add("other_punct")
        if rs != hs:  # same letters, different case
            kinds.add("sentence_initial_cap" if sentence_initial(ALL_REF_TOKS, i) else "other_case")
    if not kinds:
        return "none"
    if kinds <= {"terminal_punct", "sentence_initial_cap"}:
        return "rule_fixable"
    return "other"


ALL_REF_TOKS: list[str] = []


def token_kind(rt, ht, i):
    """Why does this one reference token differ from the hypothesis token?"""
    if rt == ht:
        return "none"
    rs, hs = strip_term(rt), strip_term(ht)
    rsuf, hsuf = rt[len(rs):], ht[len(hs):]
    if rs != hs and rs.lower() != hs.lower():
        return "other"
    tags = []
    if rsuf != hsuf:
        tags.append("terminal_punct" if set(rsuf) | set(hsuf) <= set(TERM_PUNCT) else "other_punct")
    if rs != hs:
        tags.append("sentence_initial_cap" if sentence_initial(ALL_REF_TOKS, i) else "other_case")
    if not tags:
        return "other"
    if all(t in ("terminal_punct", "sentence_initial_cap") for t in tags):
        return "+".join(sorted(set(tags)))
    return "+".join(sorted(set(tags)))


RULE_FIXABLE = {"terminal_punct", "sentence_initial_cap",
                "sentence_initial_cap+terminal_punct", "none"}


def ortho_breakdown(pairs, common):
    """ORTHOGRAPHY error mass, chunk-level (as class_breakdown.py counts it) and
    token-level (only the tokens that actually differ), split into the part a
    sentence-initial-capitalization + terminal-punctuation rule would fix."""
    global ALL_REF_TOKS
    chunk_tot = chunk_fix = 0
    tok_tot = tok_fix = 0
    kindc = Counter()
    examples = []
    for ref, hyp in pairs:
        ALL_REF_TOKS = tokenize(ref)
        htoks = tokenize(hyp)
        for e in align_errors(ref, hyp, common):
            if e["class"] != "ORTHOGRAPHY":
                continue
            chunk_tot += max(1, e["ref_words"])
            rt = ALL_REF_TOKS[e["ref_start"]:e["ref_end"]]
            ht = htoks[e["hyp_start"]:e["hyp_end"]]
            k = ortho_kind(rt, ht, e["ref_start"])
            if k in ("rule_fixable", "none"):
                chunk_fix += max(1, e["ref_words"])
            elif len(examples) < 8:
                examples.append({"ref": " ".join(rt), "hyp": " ".join(ht)})
            # token level
            if len(rt) == len(ht):
                for j, (a, b) in enumerate(zip(rt, ht)):
                    if a == b:
                        continue
                    tok_tot += 1
                    tk = token_kind(a, b, e["ref_start"] + j)
                    kindc[tk] += 1
                    if tk in RULE_FIXABLE:
                        tok_fix += 1
            else:
                n = max(len(rt), len(ht))
                tok_tot += n
                kindc["length_mismatch"] += n
    return chunk_tot, chunk_fix, tok_tot, tok_fix, dict(kindc), examples


def class_rates(pairs, common):
    c = Counter()
    ntok = 0
    for ref, hyp in pairs:
        ntok += len(tokenize(ref))
        for e in align_errors(ref, hyp, common):
            c[e["class"]] += len(e["hyp"].split()) if e["type"] == "insert" else max(1, e["ref_words"])
    return {k: 1000.0 * c[k] / max(ntok, 1) for k in CLASSES}, ntok


def load_set(s):
    rows = {}
    for label, fn, key in [
        ("lfm2.5-1.2b", PRED / f"lfm2.5-1.2b_{s}.jsonl", "output"),
        ("lfm2.5-1.2b_real", PRED / f"lfm2.5-1.2b_real_{s}.jsonl", "output"),
        ("ref1_lfm2.5-1.2b_synthreal", PRED / f"ref1_lfm2.5-1.2b_synthreal_{s}.jsonl", "output"),
        ("ref3_lfm2.5-1.2b_synth", PRED / f"ref3_lfm2.5-1.2b_synth_{s}.jsonl", "output"),
        ("vocab_real_acoustic", VOCAB / f"vocab_real_acoustic__{s}.jsonl", "out_vocabref"),
        ("vocab_real_acoustic_novocab", VOCAB / f"vocab_real_acoustic__{s}.jsonl", "out_none"),
    ]:
        if not fn.exists():
            print(f"MISSING {fn}", flush=True)
            continue
        data = [json.loads(l) for l in fn.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows[label] = {r["id"]: r for r in data}
        rows.setdefault("_ref", {}).update(
            {r["id"]: (r["reference"], r["input"]) for r in data})
        rows["_key_" + label] = key
    return rows


FILLERS = ["um", "uh", "like", "you know", "I mean", "sort of"]


def filler_probe(pairs20):
    out = {"token_level": {}, "rows": [], "n_rows": len(pairs20)}
    for f in FILLERS:
        out["token_level"][f] = {"fair_norm": fair(f), "survives": bool(fair(f).strip())}
    refs = [r for r, _ in pairs20]
    for f in FILLERS:
        hyps = []
        n_ins = 0
        for r in refs:
            toks = r.split()
            new = []
            for i, t in enumerate(toks):
                new.append(t)
                if i % 8 == 7:
                    new.append(f)
                    n_ins += 1
            hyps.append(" ".join(new))
        out["rows"].append({
            "filler": f, "insertions": n_ins,
            "fair_wer": corpus_wer(refs, hyps, fair),
            "strict_wer": corpus_wer(refs, hyps, strict),
        })
    return out


def real_filler_counts(pairs):
    """how often the raw ASR emits a filler the reference does not keep"""
    pat = re.compile(r"\b(um+|uh+|hmm+|mm+|er+|ah+|like|you know)\b", re.I)
    c_raw = Counter()
    c_ref = Counter()
    for ref, raw in pairs:
        for m in pat.findall(raw or ""):
            c_raw[m.lower()] += 1
        for m in pat.findall(ref or ""):
            c_ref[m.lower()] += 1
    return dict(c_raw), dict(c_ref)


def main() -> int:
    common = load_english()
    report = {}
    L = []
    A = L.append
    A("# Damage-visible reporting for the corrector on the Wispr holdouts\n")
    A("Fair WER = whisper_normalizer EnglishTextNormalizer on both sides, then jiwer. "
      "Strict = lowercase only. Row-level comparison is per-row fair WER against `raw` "
      "(the ASR `input` field of the same prediction file), paired by id; a row is a LOSS "
      "(damage) when its fair WER is strictly worse than raw's.\n")

    for s in SETS:
        R = load_set(s)
        refmap = R["_ref"]
        ids = sorted(refmap)
        refs = [refmap[i][0] for i in ids]
        raws = [refmap[i][1] for i in ids]
        raw_row = {i: row_wer(refmap[i][0], refmap[i][1], fair) for i in ids}
        report[s] = {"n": len(ids), "systems": {}}

        A(f"## {s} (n={len(ids)})\n")
        A("| system | fair WER | strict lc WER | win | tie | loss | damage rate | median row delta | mean row delta |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        sysnames = ["raw"] + [k for k in R if not k.startswith("_")]
        for label in sysnames:
            if label == "raw":
                hyps = raws
            else:
                key = R["_key_" + label]
                m = R[label]
                hyps = [m[i][key] if i in m else "" for i in ids]
            fw = corpus_wer(refs, hyps, fair)
            sw = corpus_wer(refs, hyps, strict)
            deltas = []
            win = tie = loss = 0
            for i, h in zip(ids, hyps):
                d = row_wer(refmap[i][0], h, fair) - raw_row[i]
                deltas.append(d)
                if d < -1e-9:
                    win += 1
                elif d > 1e-9:
                    loss += 1
                else:
                    tie += 1
            ot, ofix, otok, ofixtok, okinds, oex = ortho_breakdown(list(zip(refs, hyps)), common)
            cr, ntok = class_rates(list(zip(refs, hyps)), common)
            report[s]["systems"][label] = {
                "fair_wer": fw, "strict_wer": sw, "win": win, "tie": tie, "loss": loss,
                "damage_rate": loss / len(ids), "median_delta": statistics.median(deltas),
                "mean_delta": sum(deltas) / len(deltas),
                "ortho_chunk_mass": ot, "ortho_chunk_rule_fixable": ofix,
                "ortho_chunk_rule_fixable_frac": (ofix / ot) if ot else None,
                "ortho_token_mass": otok, "ortho_token_rule_fixable": ofixtok,
                "ortho_rule_fixable_frac": (ofixtok / otok) if otok else None,
                "ortho_token_kinds": okinds,
                "ortho_per_1000_words": cr["ORTHOGRAPHY"], "ref_tokens": ntok,
                "ortho_hard_examples": oex,
                "classes": cr,
            }
            A(f"| {label} | {fw:.4f} | {sw:.4f} | {win} | {tie} | {loss} | "
              f"{loss/len(ids):.3f} | {statistics.median(deltas):+.4f} | {sum(deltas)/len(deltas):+.4f} |")
        A("")
        A("### ORTHOGRAPHY error mass: how much a two-line rule would fix\n")
        A("Rule = capitalize the first word of a sentence, and match the reference's terminal "
          "punctuation `.!?`. `class_breakdown.py` charges the WHOLE aligned chunk when any token "
          "inside it differs in case or punctuation, so the chunk column overstates the mass badly; "
          "the token column counts only the tokens that actually differ and is the honest one.\n")
        A("| system | ORTHOGRAPHY err/1000 ref words (chunk) | chunk mass | token mass | rule-fixable tokens | fraction |")
        A("|---|---:|---:|---:|---:|---:|")
        for label in sysnames:
            d = report[s]["systems"][label]
            f = d["ortho_rule_fixable_frac"]
            A(f"| {label} | {d['ortho_per_1000_words']:.1f} | {d['ortho_chunk_mass']} | "
              f"{d['ortho_token_mass']} | {d['ortho_token_rule_fixable']} | "
              f"{(f if f is not None else 0):.3f} |")
        A("")
        A("Token-level ORTHOGRAPHY reasons, raw: " +
          ", ".join(f"{k}={v}" for k, v in sorted(
              report[s]["systems"]["raw"]["ortho_token_kinds"].items(), key=lambda x: -x[1])) + "\n")
        ex = report[s]["systems"]["raw"]["ortho_hard_examples"]
        if ex:
            A("Residual (not rule-fixable) orthography examples on raw: " +
              "; ".join(f"`{e['ref']}` -> `{e['hyp']}`" for e in ex[:6]) + "\n")

    # ---- filler probe ----
    R = load_set("wispr_holdout120")
    refmap = R["_ref"]
    ids = sorted(refmap)[:20]
    pairs20 = [refmap[i] for i in ids]
    fp = filler_probe(pairs20)
    allpairs = [refmap[i] for i in sorted(refmap)]
    craw, cref = real_filler_counts(allpairs)
    nref_words = sum(len((fair(r) or "").split()) for r, _ in allpairs)
    deleted = re.compile(r"\b(um+|uh+|hmm+|mm+|er|ah)\b", re.I)
    free_ins = sum(len(deleted.findall(raw or "")) - len(deleted.findall(ref or ""))
                   for ref, raw in allpairs)
    kept = re.compile(r"\b(like|you know|i mean)\b", re.I)
    charged_ins = sum(len(kept.findall(raw or "")) - len(kept.findall(ref or ""))
                      for ref, raw in allpairs)
    report["filler_probe"] = {
        "synthetic": fp, "real_raw_counts": craw, "real_ref_counts": cref,
        "normalized_reference_words": nref_words,
        "filler_insertions_deleted_by_normalizer": free_ins,
        "hidden_wer_points": free_ins / max(nref_words, 1),
        "filler_insertions_still_charged": charged_ins,
        "charged_wer_points": charged_ins / max(nref_words, 1),
    }

    A("## Does fair WER hide filler insertions?\n")
    A("Probe: take the references of 20 real wispr_holdout120 rows, insert one filler every 8 words, "
      "score the result against the untouched reference.\n")
    A("| filler | normalizer output | insertions | fair WER | strict lc WER |")
    A("|---|---|---:|---:|---:|")
    for r in fp["rows"]:
        tl = fp["token_level"][r["filler"]]
        A(f"| {r['filler']} | `{tl['fair_norm'] or '(deleted)'}` | {r['insertions']} | "
          f"{r['fair_wer']:.4f} | {r['strict_wer']:.4f} |")
    A("")
    A("Filler tokens actually present in the wispr_holdout120 raw ASR vs in the references: ")
    A("")
    A("| token | in raw ASR | in reference |")
    A("|---|---:|---:|")
    for k in sorted(set(craw) | set(cref), key=lambda x: -(craw.get(x, 0) + cref.get(x, 0))):
        A(f"| {k} | {craw.get(k,0)} | {cref.get(k,0)} |")
    A("")
    d = report["filler_probe"]
    A(f"Over all 120 rows the reference normalizes to {d['normalized_reference_words']} words. "
      f"{d['filler_insertions_deleted_by_normalizer']} um/uh/hmm/er/ah insertions that raw ASR makes "
      f"and the reference does not keep are deleted by the normalizer on both sides, so fair WER "
      f"hides {100*d['hidden_wer_points']:.2f} WER points of real INSERT error. "
      f"{d['filler_insertions_still_charged']} like / you know / I mean insertions survive the "
      f"normalizer and are charged ({100*d['charged_wer_points']:.2f} WER points).\n")

    (OUT / "results_damage.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (OUT / "results_damage.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("STEP 3 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
