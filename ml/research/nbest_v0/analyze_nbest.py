"""STEP 3/4: tables for the n-best alternatives eval. Metrics imported unchanged from
research/vocab_v1/analyze_vocab1.py via research/bigrun_v0/analyze_bigrun.py."""
from __future__ import annotations
import json, sys
from pathlib import Path

for p in ("/home/user/phonon/research/vocab_v1", "/home/user/phonon/research/bigrun_v0",
          "/home/user/phonon/research/nbest_v0"):
    sys.path.insert(0, p)
from analyze_bigrun import term_block  # noqa: E402
from analyze_vocab1 import fair, hit_norm, read_jsonl, row_wer, wer  # noqa: E402
from altline import load_nbest  # noqa: E402

REF = Path("/data/phonon_nbest_v0/refined")
OUT = Path("/home/user/phonon/research/nbest_v0")
MISSES = Path("/data/phonon_retrieval_v2/misses.json")
SETS = ["new", "unseen"]
NB = {"new": "/data/phonon_nbest_v0/nbest_heldout_new.jsonl",
      "unseen": "/data/phonon_nbest_v0/nbest_unseen.jsonl"}
CONDS = ["retrieved", "oracle"]
VARIANTS = ["noalt", "alt"]
LABELS = ["bigrun_mid_r16", "nbest_mid_r16"]


def hard_ids():
    m = json.loads(MISSES.read_text())
    out = {}
    for key, name in (("new", "new"), ("unseen", "old_unseen")):
        out[key] = {r["id"] for r in m[name]["per_miss"] if r["cat"] == "hard_acoustic_loss"}
    return out


def main() -> int:
    hard = hard_ids()
    res = {"sets": {}, "stats": {}}
    sp = Path("/data/phonon_nbest_v0/refine_stats.json")
    if sp.exists():
        res["stats"] = json.loads(sp.read_text())
    op = Path("/data/phonon_nbest_v0/oracle_presence.json")
    if op.exists():
        res["oracle_presence"] = json.loads(op.read_text())
    bp = Path("/data/phonon_nbest_v0/train_build_stats.json")
    if bp.exists():
        res["train_build"] = json.loads(bp.read_text())

    for key in SETS:
        block = {}
        base_rows = None
        nb = load_nbest(Path(NB[key]))
        for label in LABELS:
            p = REF / f"{label}__{key}.jsonl"
            if not p.exists():
                print(f"missing {p}", flush=True)
                continue
            rows = read_jsonl(p)
            base_rows = rows
            refs = [r["reference"] for r in rows]
            raws = [r["hyp"] for r in rows]
            terms = [r["term"] for r in rows]
            reth = [r["retrieved_has_term"] for r in rows]
            raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, raws)]
            hidx = [i for i, r in enumerate(rows) if r["id"] in hard[key]]
            nbonly = []
            for i, r in enumerate(rows):
                rec = nb.get(r["id"]) or {}
                union = " \n ".join([rec.get("best", "")] + list(rec.get("alts") or []))
                if union and hit_norm(r["term"], union) and not hit_norm(r["term"], r["hyp"]):
                    nbonly.append(i)
            for cond in CONDS:
                vocabs = [r["vocab_oracle" if cond == "oracle" else "vocab_retrieved"]
                          for r in rows]
                for var in VARIANTS:
                    col = f"out_{cond}_{var}"
                    if col not in rows[0]:
                        continue
                    hyps = [r[col] for r in rows]
                    b = term_block(refs, raws, terms, hyps, vocabs, raw_rw, reth)
                    b["hard_acoustic_n"] = len(hidx)
                    b["hard_acoustic_hit"] = (
                        sum(hit_norm(terms[i], hyps[i]) for i in hidx) / len(hidx)) if hidx else None
                    b["nbest_only_n"] = len(nbonly)
                    b["nbest_only_hit"] = (
                        sum(hit_norm(terms[i], hyps[i]) for i in nbonly) / len(nbonly)
                    ) if nbonly else None
                    b["echo_rate"] = sum(
                        1 for h in hyps if "alternatives:" in h.lower()) / len(hyps)
                    b["alt_line_rows"] = sum(1 for r in rows if r.get("alt_line"))
                    sel = [i for i, r in enumerate(rows) if r.get("alt_line")]
                    b["hit_rows_with_line"] = (
                        sum(hit_norm(terms[i], hyps[i]) for i in sel) / len(sel)) if sel else None
                    block[f"{label}|{cond}|{var}"] = b
        if base_rows is not None:
            refs = [r["reference"] for r in base_rows]
            raws = [r["hyp"] for r in base_rows]
            terms = [r["term"] for r in base_rows]
            reth = [r["retrieved_has_term"] for r in base_rows]
            raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, raws)]
            vocabs = [r["vocab_retrieved"] for r in base_rows]
            hidx = [i for i, r in enumerate(base_rows) if r["id"] in hard[key]]
            b = term_block(refs, raws, terms, raws, vocabs, raw_rw, reth)
            b["hard_acoustic_n"] = len(hidx)
            b["hard_acoustic_hit"] = (
                sum(hit_norm(terms[i], raws[i]) for i in hidx) / len(hidx)) if hidx else None
            block["raw|-|-"] = b
        res["sets"][key] = block

    rp = REF / "nbest_mid_r16__real.jsonl"
    if rp.exists():
        real = {}
        for label in LABELS:
            p = REF / f"{label}__real.jsonl"
            if not p.exists():
                continue
            rows = read_jsonl(p)
            refs = [r["reference"] for r in rows]
            ins = [r["input"] for r in rows]
            for col in [c for c in rows[0] if c.startswith("out_")]:
                hyps = [r[col] for r in rows]
                real[f"{label}|{col[4:]}"] = {
                    "n": len(rows), "fair_wer": wer(refs, hyps, fair),
                    "damage_rate": sum(1 for a, b in zip(
                        [row_wer(x, y, fair) for x, y in zip(refs, hyps)],
                        [row_wer(x, y, fair) for x, y in zip(refs, ins)]) if a > b + 1e-9) / len(rows),
                }
            real[f"{label}|raw"] = {"n": len(rows), "fair_wer": wer(refs, ins, fair),
                                    "damage_rate": 0.0}
        res["real"] = real

    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")

    lines = []
    for key in SETS:
        block = res["sets"].get(key) or {}
        if not block:
            continue
        lines.append(f"\n### {key}\n")
        lines.append("| adapter | cond | alts | n | hit norm | hit loose | fair WER | damage | "
                     "unrel | hit when retrieved | hit when missing | hard-acoustic hit | "
                     "hit on n-best-only | echo |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k in ["raw|-|-"] + [f"{a}|{c}|{v}" for a in LABELS for c in CONDS for v in VARIANTS]:
            b = block.get(k)
            if not b:
                continue
            a, c, v = k.split("|")
            hm = b["hit_term_missing"]
            hmv = "-" if hm is None else f"{hm:.3f}"
            nbo = b.get("nbest_only_hit")
            nbov = "-" if nbo is None else f"{nbo:.3f}"
            lines.append(
                f"| {a} | {c} | {v} | {b['n']} | {b['term_hit_norm']:.3f} | "
                f"{b['term_hit_loose']:.3f} | {b['fair_wer']:.4f} | {b['damage_rate']*100:.1f}% | "
                f"{b['false_insert_rate_unrelated']*100:.2f}% | {b['hit_term_retrieved']:.3f} | "
                f"{hmv} | {b['hard_acoustic_hit']:.3f} (n={b['hard_acoustic_n']}) | "
                f"{nbov} (n={b.get('nbest_only_n', 0)}) | {b.get('echo_rate', 0)*100:.1f}% |")
    table = "\n".join(lines)
    (OUT / "table.md").write_text(table + "\n")
    print(table, flush=True)
    print("\nSTEP 3 analyze done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
