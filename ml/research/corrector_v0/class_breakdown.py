"""Per-error-class breakdown of refiner predictions: does synthetic training fix entities or add fluency edits?"""
import json, sys, glob
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path.home() / "phonon/research/asr_errors_v0"))
from analyze import align_errors, load_english, tokenize  # noqa

PRED = Path("/data/phonon_corrector_v0/eval_predictions")
SETS = ["wispr_holdout120", "wispr_text_holdout"]
LABELS = ["lfm2.5-350m_real", "ref_lfm2.5-350m_synth", "ref_lfm2.5-350m_synthreal", "ref1_lfm2.5-350m_synthreal", "ref2_lfm2.5-350m_synthreal",
          "qwen3-0.6b_real", "ref1_qwen3-0.6b_synth", "ref1_qwen3-0.6b_synthreal",
          "lfm2.5-1.2b_real", "ref1_lfm2.5-1.2b_synth", "ref2_lfm2.5-1.2b_synth", "ref1_lfm2.5-1.2b_synthreal", "ref2_lfm2.5-1.2b_synthreal",
          "curve_lfm12_n250_e12", "cold_lfm2.5-1.2b_synthv1_n250",
          "ref3_lfm2.5-350m_synth", "ref3_lfm2.5-350m_synthreal",
          "ref3_qwen3-0.6b_synth", "ref3_qwen3-0.6b_synthreal",
          "ref3_lfm2.5-1.2b_synth", "ref3_lfm2.5-1.2b_synthreal", "ref3em_lfm2.5-1.2b_synth"]
CLASSES = ["ENTITY", "FUNCTION", "ORTHOGRAPHY", "NEAR_MISS", "DROP", "INSERT", "OTHER"]
common = load_english()

def rates(pairs):
    c = Counter(); ntok = 0
    for ref, hyp in pairs:
        ntok += len(tokenize(ref))
        for e in align_errors(ref, hyp, common):
            c[e["class"]] += max(1, e["ref_words"]) if e["type"] != "insert" else len(e["hyp"].split())
    return {k: 1000.0 * c[k] / max(ntok, 1) for k in CLASSES}, ntok

out = {}
lines = ["# Error-class breakdown of refiner outputs (errors per 1000 reference words; lower is better)", "",
         "Classifier: asr_errors_v0.align_errors. `raw` is the ASR input scored against the reference. Synthetic-only rows show which classes the paraphrase failure lands in.", ""]
for s in SETS:
    lines += [f"## {s}", "", "| system | " + " | ".join(CLASSES) + " | total |", "|---|" + "---:|" * (len(CLASSES) + 1)]
    raw_done = False
    for label in LABELS:
        p = PRED / f"{label}_{s}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not raw_done:
            r, n = rates([(x["reference"], x["input"]) for x in rows])
            lines.append(f"| raw | " + " | ".join(f"{r[k]:.1f}" for k in CLASSES) + f" | {sum(r.values()):.1f} |"); raw_done = True
            out.setdefault(s, {})["raw"] = r
        r, n = rates([(x["reference"], x["output"]) for x in rows])
        out[s][label] = r
        lines.append(f"| {label} | " + " | ".join(f"{r[k]:.1f}" for k in CLASSES) + f" | {sum(r.values()):.1f} |")
    lines.append("")
Path.home().joinpath("phonon/research/corrector_v0/results_classes.md").write_text("\n".join(lines))
Path.home().joinpath("phonon/research/corrector_v0/results_classes.json").write_text(json.dumps(out, indent=1))
print("\n".join(lines))
