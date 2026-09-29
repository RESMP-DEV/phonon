"""Step 4: term-recovery metric, fair WER, error-class rates for raw + each refiner."""
from __future__ import annotations
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/asr_errors_v0")
from analyze import align_errors, load_english, tokenize  # noqa: E402

from whisper_normalizer.english import EnglishTextNormalizer
import jiwer

DATA = Path("/data/phonon_term_eval_v0")
OUT_DIR = Path("/home/user/phonon/research/term_eval_v0")
CLASSES = ["ENTITY", "FUNCTION", "ORTHOGRAPHY", "NEAR_MISS", "DROP", "INSERT", "OTHER"]
SYSTEMS = ["lfm2.5-1.2b", "ref1_lfm2.5-1.2b_synthreal", "lfm2.5-350m",
           "ref_lfm2.5-350m_synthreal", "qwen3-0.6b"]
# which training corpus each system saw
SYNTH_SYSTEMS = {"ref1_lfm2.5-1.2b_synthreal", "ref_lfm2.5-350m_synthreal"}

_NORM = EnglishTextNormalizer()
_norm_cache: dict[str, str] = {}


def fair(text: str) -> str:
    t = (text or "").strip()
    if t not in _norm_cache:
        _norm_cache[t] = _NORM(t)
    return _norm_cache[t]


def strict_lc(text: str) -> str:
    return (text or "").strip().lower()


def hit_norm(term: str, text: str) -> bool:
    nt = fair(term)
    nx = fair(text)
    if not nt:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(nt) + r"(?![a-z0-9])", nx) is not None


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def hit_loose(term: str, text: str) -> bool:
    st = squash(term)
    return bool(st) and st in squash(text)


def corpus_wer(refs, hyps, norm):
    nr = [norm(r) or "<empty>" for r in refs]
    nh = [norm(h) or "<empty>" for h in hyps]
    return float(jiwer.wer(nr, nh))


def class_rates(pairs, common):
    c = Counter()
    ntok = 0
    for ref, hyp in pairs:
        ntok += len(tokenize(ref))
        for e in align_errors(ref, hyp, common):
            c[e["class"]] += len(e["hyp"].split()) if e["type"] == "insert" else max(1, e["ref_words"])
    return {k: 1000.0 * c[k] / max(ntok, 1) for k in CLASSES}


def main() -> int:
    asr = [json.loads(l) for l in (DATA / "asr.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    terms = {r["term"]: r for r in
             (json.loads(l) for l in (DATA / "terms_heldout.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    print(f"asr rows={len(asr)} terms={len(terms)}", flush=True)
    common = load_english()

    outputs: dict[str, list[str]] = {"raw": [r["hyp"] for r in asr]}
    for label in SYSTEMS:
        p = DATA / "refined" / f"{label}.jsonl"
        if not p.exists():
            print(f"missing {p}", flush=True)
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_id = {r["id"]: r["output"] for r in rows}
        outputs[label] = [by_id.get(r["id"], "") for r in asr]

    refs = [r["reference"] for r in asr]
    term_list = [r["term"] for r in asr]

    def in_train(t):
        return bool(terms.get(t, {}).get("in_train"))

    def in_synth(t):
        m = terms.get(t, {})
        return bool(m.get("in_synth_v1") or m.get("in_synth_v2") or m.get("in_train"))

    results = {}
    for label, hyps in outputs.items():
        hits_n = [hit_norm(t, h) for t, h in zip(term_list, hyps)]
        hits_l = [hit_loose(t, h) for t, h in zip(term_list, hyps)]
        seen_fn = in_synth if label in SYNTH_SYSTEMS else in_train
        seen = [seen_fn(t) for t in term_list]
        def rate(mask, vals):
            sel = [v for v, m in zip(vals, mask) if m]
            return (sum(sel) / len(sel)) if sel else None
        block = {
            "n": len(asr),
            "term_hit_norm": sum(hits_n) / len(hits_n),
            "term_hit_loose": sum(hits_l) / len(hits_l),
            "term_hit_norm_seen": rate(seen, hits_n),
            "term_hit_norm_unseen": rate([not s for s in seen], hits_n),
            "n_seen": sum(seen),
            "n_unseen": len(seen) - sum(seen),
            "fair_wer": corpus_wer(refs, hyps, fair),
            "strict_lc_wer": corpus_wer(refs, hyps, strict_lc),
            "classes": class_rates(list(zip(refs, hyps)), common),
            "by_kind": {},
            "in_synth_split": {
                "seen_synth_hit": rate([in_synth(t) for t in term_list], hits_n),
                "unseen_synth_hit": rate([not in_synth(t) for t in term_list], hits_n),
            },
        }
        by_kind = defaultdict(list)
        for t, h in zip(term_list, hits_n):
            by_kind[terms.get(t, {}).get("kind", "?")].append(h)
        block["by_kind"] = {k: {"n": len(v), "hit": sum(v) / len(v)} for k, v in sorted(by_kind.items())}
        results[label] = block
        print(f"{label}: hit_norm={block['term_hit_norm']:.3f} hit_loose={block['term_hit_loose']:.3f} "
              f"fair_wer={block['fair_wer']:.4f}", flush=True)

    # per-term raw hit rate
    per_term = defaultdict(lambda: {"n": 0, "raw_hit": 0, "raw_loose": 0})
    for r, h in zip(asr, outputs["raw"]):
        d = per_term[r["term"]]
        d["n"] += 1
        d["raw_hit"] += int(hit_norm(r["term"], h))
        d["raw_loose"] += int(hit_loose(r["term"], h))
    per_term_rows = []
    for t, d in per_term.items():
        meta = terms.get(t, {})
        per_term_rows.append({
            "term": t, "kind": meta.get("kind"), "n": d["n"],
            "raw_hit_rate": d["raw_hit"] / d["n"], "raw_loose_rate": d["raw_loose"] / d["n"],
            "mangle_score": meta.get("mangle_score"), "rank_score": meta.get("rank_score"),
            "in_synth": in_synth(t), "in_train": in_train(t),
        })
    per_term_rows.sort(key=lambda r: (r["raw_hit_rate"], r["raw_loose_rate"], -(r["mangle_score"] or 0), r["term"]))
    hardest = per_term_rows[:20]
    n_zero = sum(1 for r in per_term_rows if r["raw_hit_rate"] == 0)

    # best refiner by term hit
    ref_labels = [l for l in outputs if l != "raw"]
    best = max(ref_labels, key=lambda l: results[l]["term_hit_norm"]) if ref_labels else None

    # 15 examples: prefer rows where raw missed the term
    examples = []
    for i, r in enumerate(asr):
        raw_h = outputs["raw"][i]
        if hit_norm(r["term"], raw_h):
            continue
        examples.append({
            "term": r["term"], "voice": r["voice"], "reference": r["reference"],
            "raw": raw_h, "best_label": best,
            "best_output": outputs[best][i] if best else "",
            "best_hit": hit_norm(r["term"], outputs[best][i]) if best else False,
        })
        if len(examples) >= 15:
            break

    payload = {
        "n_rows": len(asr), "n_terms": len(per_term), "best_refiner": best,
        "systems": results, "hardest_terms": hardest, "n_terms_zero_raw_hit": n_zero,
        "examples": examples, "per_term": per_term_rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=1) + "\n")

    L = []
    A = L.append
    A("# Term-recovery eval v0")
    A("")
    A(f"{len(per_term)} held-out lexicon terms, 3 LFM-generated spoken sentences each, "
      f"Kokoro TTS in 3 voices, Parakeet TDT 0.6b v2 ASR: {len(asr)} clips.")
    A("`term hit` = the term is present in the text after whisper_normalizer EnglishTextNormalizer "
      "on both sides (word-boundary match); `loose` = both sides lowercased with every non-alphanumeric "
      "character stripped, then substring. Fair WER is the same normalizer plus jiwer.")
    A("")
    A("## Term hit rate and WER")
    A("")
    A("| system | n | term hit (norm) | term hit (loose) | hit, term in its train data | hit, term not in it | fair WER | strict lc WER |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label in ["raw"] + [l for l in SYSTEMS if l in results]:
        b = results[label]
        seen = f"{b['term_hit_norm_seen']:.3f} (n={b['n_seen']})" if b["term_hit_norm_seen"] is not None else "-"
        unseen = f"{b['term_hit_norm_unseen']:.3f} (n={b['n_unseen']})" if b["term_hit_norm_unseen"] is not None else "-"
        if label == "raw":
            seen = unseen = "-"
        A(f"| {label} | {b['n']} | {b['term_hit_norm']:.3f} | {b['term_hit_loose']:.3f} | {seen} | {unseen} | "
          f"{b['fair_wer']:.4f} | {b['strict_lc_wer']:.4f} |")
    A("")
    A("Split column: for the personal-only adapters the term counts as seen if it appears in "
      "/data/phonon_corrector_v0/train.jsonl targets; for the synth+real adapters if it appears in "
      "synth_pairs_v1/v2 or train.jsonl targets.")
    A("")
    A("## Same split for every system: term present in synth_pairs_v1/v2 or train.jsonl targets")
    A("")
    A("The 24 held-out terms absent from every training corpus are also the easier ones for raw ASR, "
      "so compare each system against the raw row inside a column, not across columns.")
    A("")
    A("| system | hit, in synth/train targets (n=2484) | hit, in none (n=216) | delta vs raw (in) | delta vs raw (out) |")
    A("|---|---:|---:|---:|---:|")
    raw_in = results["raw"]["in_synth_split"]["seen_synth_hit"]
    raw_out = results["raw"]["in_synth_split"]["unseen_synth_hit"]
    for label in ["raw"] + [l for l in SYSTEMS if l in results]:
        b = results[label]["in_synth_split"]
        A(f"| {label} | {b['seen_synth_hit']:.3f} | {b['unseen_synth_hit']:.3f} | "
          f"{b['seen_synth_hit']-raw_in:+.3f} | {b['unseen_synth_hit']-raw_out:+.3f} |")
    A("")
    A("## Error classes (errors per 1000 reference words, lower is better)")
    A("")
    A("| system | " + " | ".join(CLASSES) + " | total |")
    A("|---|" + "---:|" * (len(CLASSES) + 1))
    for label in ["raw"] + [l for l in SYSTEMS if l in results]:
        c = results[label]["classes"]
        A(f"| {label} | " + " | ".join(f"{c[k]:.1f}" for k in CLASSES) + f" | {sum(c.values()):.1f} |")
    A("")
    A("## Term hit by lexicon kind (norm match)")
    A("")
    kinds = sorted({k for b in results.values() for k in b["by_kind"]})
    A("| system | " + " | ".join(kinds) + " |")
    A("|---|" + "---:|" * len(kinds))
    for label in ["raw"] + [l for l in SYSTEMS if l in results]:
        bk = results[label]["by_kind"]
        A(f"| {label} | " + " | ".join(f"{bk[k]['hit']:.2f}" if k in bk else "-" for k in kinds) + " |")
    A("")
    A(f"Clip counts by kind: " + ", ".join(f"{k}={results['raw']['by_kind'][k]['n']}" for k in kinds if k in results["raw"]["by_kind"]))
    A("")
    A(f"## 20 hardest terms (lowest raw hit rate; {n_zero} of {len(per_term)} terms are at 0.00)")
    A("")
    A("| term | kind | clips | raw hit | raw loose | mangle | in synth targets |")
    A("|---|---|---:|---:|---:|---:|---|")
    for r in hardest:
        A(f"| `{r['term']}` | {r['kind']} | {r['n']} | {r['raw_hit_rate']:.2f} | {r['raw_loose_rate']:.2f} | "
          f"{r['mangle_score']} | {'yes' if r['in_synth'] else 'no'} |")
    A("")
    A(f"## 15 example rows where raw ASR lost the term (best refiner by term hit = {best})")
    A("")
    for i, ex in enumerate(examples, 1):
        A(f"{i}. term `{ex['term']}` ({ex['voice']}) best_hit={ex['best_hit']}")
        A(f"   REF:  {ex['reference']}")
        A(f"   RAW:  {ex['raw']}")
        A(f"   {ex['best_label']}: {ex['best_output']}")
        A("")
    (OUT_DIR / "results.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_DIR/'results.md'} and results.json", flush=True)
    print("STEP 4 done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
