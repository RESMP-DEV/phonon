"""Step 4: term-hit x condition table, real-holdout table, class breakdown, examples, read."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/asr_errors_v0")
from analyze import align_errors, load_english, tokenize  # noqa: E402

import jiwer  # noqa: E402
from whisper_normalizer.english import EnglishTextNormalizer  # noqa: E402

DATA = Path("/data/phonon_vocab_v0")
REF = DATA / "refined"
OUT = Path("/home/user/phonon/research/vocab_v0")
CLASSES = ["ENTITY", "FUNCTION", "ORTHOGRAPHY", "NEAR_MISS", "DROP", "INSERT", "OTHER"]
SYSTEMS = ["vocab_real", "vocab_real_acoustic", "novocab_real_acoustic",
           "vocab_real_acoustic_350m", "lfm2.5-1.2b_personal"]
CONDS = [("none", "out_none"), ("oracle", "out_oracle"), ("realistic", "out_realistic")]
REAL_SETS = ["wispr_holdout120", "wispr_text_holdout"]

_NORM = EnglishTextNormalizer()
_cache: dict[str, str] = {}


def fair(t: str) -> str:
    t = (t or "").strip()
    if t not in _cache:
        _cache[t] = _NORM(t)
    return _cache[t]


def strict_lc(t: str) -> str:
    return (t or "").strip().lower()


def hit_norm(term: str, text: str) -> bool:
    nt, nx = fair(term), fair(text)
    return bool(nt) and re.search(r"(?<![a-z0-9])" + re.escape(nt) + r"(?![a-z0-9])", nx) is not None


def squash(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def hit_loose(term: str, text: str) -> bool:
    s = squash(term)
    return bool(s) and s in squash(text)


def wer(refs, hyps, norm) -> float:
    nr = [norm(r) or "<empty>" for r in refs]
    nh = [norm(h) or "<empty>" for h in hyps]
    return float(jiwer.wer(nr, nh))


def class_rates(pairs, common) -> dict[str, float]:
    c, ntok = Counter(), 0
    for ref, hyp in pairs:
        ntok += len(tokenize(ref))
        for e in align_errors(ref, hyp, common):
            c[e["class"]] += len(e["hyp"].split()) if e["type"] == "insert" else max(1, e["ref_words"])
    return {k: 1000.0 * c[k] / max(ntok, 1) for k in CLASSES}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    common = load_english()
    res: dict = {"terms": {}, "real": {}, "classes": {}, "kinds": {}}
    term_rows: dict[str, list[dict]] = {}
    for s in SYSTEMS:
        p = REF / f"{s}__terms.jsonl"
        if p.exists():
            term_rows[s] = read_jsonl(p)
    if not term_rows:
        print("no refined term files yet")
        return 1
    base_rows = next(iter(term_rows.values()))
    refs = [r["reference"] for r in base_rows]
    terms = [r["term"] for r in base_rows]
    kinds = [r["kind"] for r in base_rows]
    has_term = [r["realistic_has_term"] for r in base_rows]

    def block(hyps, label):
        hn = [hit_norm(t, h) for t, h in zip(terms, hyps)]
        hl = [hit_loose(t, h) for t, h in zip(terms, hyps)]
        by_kind: dict[str, float] = {}
        for k in sorted(set(kinds)):
            sel = [v for v, kk in zip(hn, kinds) if kk == k]
            by_kind[k] = sum(sel) / len(sel) if sel else None
        out = {
            "n": len(hyps),
            "term_hit_norm": sum(hn) / len(hn),
            "term_hit_loose": sum(hl) / len(hl),
            "fair_wer": wer(refs, hyps, fair),
            "strict_lc_wer": wer(refs, hyps, strict_lc),
            "classes": class_rates(list(zip(refs, hyps)), common),
            "by_kind": by_kind,
        }
        sel_in = [v for v, m in zip(hn, has_term) if m]
        sel_out = [v for v, m in zip(hn, has_term) if not m]
        out["hit_term_in_list"] = sum(sel_in) / len(sel_in)
        out["n_term_in_list"] = len(sel_in)
        out["hit_term_missing"] = sum(sel_out) / len(sel_out)
        out["n_term_missing"] = len(sel_out)
        return out

    res["terms"]["raw"] = {"none": block([r["hyp"] for r in base_rows], "none")}
    for s, rows in term_rows.items():
        by_id = {r["id"]: r for r in rows}
        res["terms"][s] = {}
        for cond, field in CONDS:
            hyps = [by_id.get(r["id"], {}).get(field, "") for r in base_rows]
            res["terms"][s][cond] = block(hyps, cond)

    # ---- real dictation holdouts ----------------------------------------
    for name in REAL_SETS:
        res["real"][name] = {}
        got = False
        for s in SYSTEMS:
            p = REF / f"{s}__{name}.jsonl"
            if not p.exists():
                continue
            rows = read_jsonl(p)
            rr = [r["reference"] for r in rows]
            if not got:
                res["real"][name]["raw"] = {
                    "none": {"fair_wer": wer(rr, [r["input"] for r in rows], fair),
                             "strict_lc_wer": wer(rr, [r["input"] for r in rows], strict_lc),
                             "n": len(rows)}}
                got = True
            res["real"][name][s] = {
                "none": {"fair_wer": wer(rr, [r["out_none"] for r in rows], fair),
                         "strict_lc_wer": wer(rr, [r["out_none"] for r in rows], strict_lc),
                         "n": len(rows)},
                "vocab_from_reference": {
                    "fair_wer": wer(rr, [r["out_vocabref"] for r in rows], fair),
                    "strict_lc_wer": wer(rr, [r["out_vocabref"] for r in rows], strict_lc),
                    "n": len(rows)},
            }

    # ---- markdown --------------------------------------------------------
    L = ["# Vocabulary-in-prompt refiner v0", "",
         "Does putting the user's vocabulary in the refiner prompt recover spoken technical terms?",
         "2,700 term-eval clips (300 held-out terms x 3 sentences x 3 voices, Parakeet TDT 0.6b v2 "
         "hypotheses), three prompt conditions per adapter.", "",
         "Conditions: **none** = no vocabulary line; **oracle** = the true term plus 9 "
         "nearest-rank distractors; **realistic** = 30 same-kind nearest-rank terms with the true "
         "term present with probability 0.8 (551 of 2,700 rows have no true term in the list).", "",
         "## Term hit rate by adapter x condition", "",
         "| adapter | condition | term hit (norm) | vs its own none | vs raw | term hit (loose) | "
         "fair WER | strict lc WER |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    raw_hit = res["terms"]["raw"]["none"]["term_hit_norm"]
    for s in ["raw"] + [x for x in SYSTEMS if x in res["terms"]]:
        own = res["terms"][s].get("none", {}).get("term_hit_norm")
        for cond, _ in CONDS:
            b = res["terms"][s].get(cond)
            if not b:
                continue
            d_own = "-" if own is None else f"{b['term_hit_norm']-own:+.3f}"
            L.append(f"| {s} | {cond} | {b['term_hit_norm']:.3f} | {d_own} | "
                     f"{b['term_hit_norm']-raw_hit:+.3f} | {b['term_hit_loose']:.3f} | "
                     f"{b['fair_wer']:.4f} | {b['strict_lc_wer']:.4f} |")
    L += ["", "Reference points from research/term_eval_v0/results.md (no vocabulary anywhere): "
          "raw 0.584 / 0.708, lfm2.5-1.2b personal adapter 0.579 / 0.706, best refiner "
          "ref_lfm2.5-350m_synthreal 0.588 / 0.726.", ""]

    rawb = res["terms"]["raw"]["none"]
    L += ["## Realistic condition split: is the true term in the 30-term list?", "",
          f"The two subsets are not equally hard for the ASR (raw hit "
          f"{rawb['hit_term_in_list']:.3f} on the {rawb['n_term_in_list']} rows whose term is in "
          f"the list vs {rawb['hit_term_missing']:.3f} on the {rawb['n_term_missing']} rows where "
          f"it is missing), so read the delta against raw inside a column.", "",
          "| adapter | condition | term in list (n=%d) | term missing (n=%d) |"
          % (rawb["n_term_in_list"], rawb["n_term_missing"]),
          "|---|---|---:|---:|",
          f"| raw | - | {rawb['hit_term_in_list']:.3f} | {rawb['hit_term_missing']:.3f} |"]
    for s in [x for x in SYSTEMS if x in res["terms"]]:
        for cond in ("none", "realistic", "oracle"):
            b = res["terms"][s][cond]
            L.append(f"| {s} | {cond} | {b['hit_term_in_list']:.3f} "
                     f"({b['hit_term_in_list']-rawb['hit_term_in_list']:+.3f}) | "
                     f"{b['hit_term_missing']:.3f} "
                     f"({b['hit_term_missing']-rawb['hit_term_missing']:+.3f}) |")
    L.append("")

    L += ["## Real dictation holdouts (fair WER; vocab from reference is an upper bound)", "",
          "| set | system | no vocab | vocab from reference |", "|---|---|---:|---:|"]
    for name in REAL_SETS:
        d = res["real"].get(name, {})
        if "raw" in d:
            L.append(f"| {name} | raw ASR input | {d['raw']['none']['fair_wer']:.4f} | - |")
        for s in SYSTEMS:
            if s in d:
                L.append(f"| {name} | {s} | {d[s]['none']['fair_wer']:.4f} | "
                         f"{d[s]['vocab_from_reference']['fair_wer']:.4f} |")
    L.append("")

    L += ["## Error classes on the term clips (errors per 1000 reference words, lower is better)",
          "", "| adapter | condition | " + " | ".join(CLASSES) + " | total |",
          "|---|---|" + "---:|" * (len(CLASSES) + 1)]
    for s in ["raw"] + [x for x in SYSTEMS if x in res["terms"]]:
        for cond, _ in CONDS:
            b = res["terms"][s].get(cond)
            if not b:
                continue
            c = b["classes"]
            L.append(f"| {s} | {cond} | " + " | ".join(f"{c[k]:.1f}" for k in CLASSES)
                     + f" | {sum(c.values()):.1f} |")
    L.append("")

    kinds_order = sorted({k for k in set(kinds)})
    L += ["## Term hit by lexicon kind (norm match)", "",
          "| adapter | condition | " + " | ".join(kinds_order) + " |",
          "|---|---|" + "---:|" * len(kinds_order)]
    for s in ["raw"] + [x for x in SYSTEMS if x in res["terms"]]:
        for cond, _ in CONDS:
            b = res["terms"][s].get(cond)
            if not b:
                continue
            L.append(f"| {s} | {cond} | " + " | ".join(
                ("-" if b["by_kind"].get(k) is None else f"{b['by_kind'][k]:.2f}")
                for k in kinds_order) + " |")
    L.append("")

    # ---- 15 examples: oracle recovered what none did not -----------------
    best = "vocab_real_acoustic" if "vocab_real_acoustic" in term_rows else list(term_rows)[0]
    rows = term_rows[best]
    ex = [r for r in rows if not hit_norm(r["term"], r["hyp"])
          and hit_norm(r["term"], r["out_oracle"])][:15]
    if len(ex) < 15:
        ex += [r for r in rows if not hit_norm(r["term"], r["hyp"])][: 15 - len(ex)]
    L += [f"## 15 example rows ({best})", ""]
    for i, r in enumerate(ex[:15], 1):
        L += [f"{i}. term `{r['term']}` ({r['kind']}, {r['voice']})",
              f"   - REF:  {r['reference']}",
              f"   - RAW:  {r['hyp']}",
              f"   - vocab(oracle): {', '.join(r['vocab_oracle'])}",
              f"   - none:      {r['out_none']}",
              f"   - oracle:    {r['out_oracle']}",
              f"   - realistic: {r['out_realistic']}", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:80]))
    print(f"\nwrote {OUT/'results.md'} and results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
