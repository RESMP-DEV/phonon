"""STEP 1 metric: is the gold term in the recogniser n-best when it is not in the 1-best?"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
sys.path.insert(0, "/home/user/phonon/research/nbest_v0")
from analyze_vocab1 import hit_loose, hit_norm  # noqa: E402
from altline import line_for, load_nbest  # noqa: E402

D = Path("/data/phonon_nbest_v0")
SETS = {
    "new": (Path("/data/phonon_retrieval_v2/cond_new_v2.jsonl"), D / "nbest_heldout_new.jsonl"),
    "unseen": (Path("/data/phonon_retrieval_v2/cond_unseen_v2.jsonl"), D / "nbest_unseen.jsonl"),
}
MISSES = Path("/data/phonon_retrieval_v2/misses.json")


def main() -> int:
    misses = json.loads(MISSES.read_text())
    hard = {"new": set(), "unseen": set()}
    for key, name in (("new", "new"), ("unseen", "old_unseen")):
        for rec in misses[name]["per_miss"]:
            if rec["cat"] == "hard_acoustic_loss":
                hard[key].add(rec["id"])
    out = {}
    for key, (cond_path, nb_path) in SETS.items():
        rows = [json.loads(l) for l in cond_path.read_text().splitlines() if l.strip()]
        nb = load_nbest(nb_path)
        n = len(rows)
        cov = sum(1 for r in rows if r["id"] in nb)
        stat = {"n": n, "nbest_coverage": cov / n if n else 0.0}
        for tag, ids in (("all", None), ("hard_acoustic", hard[key])):
            sel = [r for r in rows if ids is None or r["id"] in ids]
            if not sel:
                continue
            b_norm = b_loose = u_norm = u_loose = 0
            beam_norm = 0
            nspans = 0
            nline = 0
            for r in sel:
                rec = nb.get(r["id"])
                one = r["hyp"]
                alts = list((rec or {}).get("alts") or [])
                beam_best = (rec or {}).get("best", "")
                union = " \n ".join([one, beam_best] + alts)
                b_norm += hit_norm(r["term"], one)
                b_loose += hit_loose(r["term"], one)
                u_norm += hit_norm(r["term"], union)
                u_loose += hit_loose(r["term"], union)
                beam_norm += hit_norm(r["term"], beam_best)
                line = line_for(rec, one)
                if line:
                    nline += 1
                    nspans += len(line.split(" | "))
            m = len(sel)
            stat[tag] = {
                "n": m,
                "presence_1best_norm": b_norm / m,
                "presence_1best_loose": b_loose / m,
                "presence_beambest_norm": beam_norm / m,
                "presence_nbest_union_norm": u_norm / m,
                "presence_nbest_union_loose": u_loose / m,
                "gain_norm": (u_norm - b_norm) / m,
                "rows_with_alt_line": nline / m,
                "mean_spans_when_line": (nspans / nline) if nline else 0.0,
            }
        out[key] = stat
        print(json.dumps({key: stat}, indent=1), flush=True)
    (D / "oracle_presence.json").write_text(json.dumps(out, indent=1) + "\n")
    print("STEP 1 metric done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
