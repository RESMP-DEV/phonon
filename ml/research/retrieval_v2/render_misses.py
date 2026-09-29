"""Render research/retrieval_v2/misses.md from misses.json (+ optional recovery column)."""
import json, sys
from pathlib import Path
D = Path("/data/phonon_retrieval_v2")
OUT = Path("/home/user/phonon/research/retrieval_v2/misses.md")
R = json.loads((D / "misses.json").read_text())
rec = {}
p = D / "recovered.json"
if p.exists():
    rec = json.loads(p.read_text())

CATS = ["wrong_ngram_window", "split_into_words", "spelled_letters",
        "phoneme_not_metaphone", "hard_acoustic_loss"]
DESC = {
 "wrong_ngram_window": "the whole term is in the hypothesis as >3 consecutive words, so the 1-3 word window never sees it",
 "split_into_words": "heard as separate words that the joined form does not match, but a sub-word realisation does",
 "spelled_letters": "heard as spelled-out letters (`C W R B`), matched only by the spelled realisation",
 "phoneme_not_metaphone": "g2p phoneme distance is close but metaphone and characters are not",
 "hard_acoustic_loss": "no realisation and no window gets near it -- the acoustics are gone",
}
L = []
L.append("# Retrieval v1 misses at 30: what is actually failing\n")
L.append("2,700 raw Parakeet-TDT-0.6b-v2 hypotheses per set. A clip is a *miss* when its gold term")
L.append("is not in the top 30 the v1 scorer (0.5 double-metaphone + 0.5 character similarity, max")
L.append("over 1-3 word hypothesis n-grams) returns. Each miss is labelled by the *cheapest fix that")
L.append("would reach it*, tested in the order below; the score quoted is 0.5*char + 0.5*metaphone of")
L.append("the gold term's best-matching realisation against the best hypothesis span.\n")
L.append("| category | meaning |\n|---|---|")
for c in CATS:
    L.append(f"| `{c}` | {DESC[c]} |")
L.append("")
for s in ("new", "old_unseen"):
    d = R[s]
    L.append(f"## {s} (pool {d['pool']}, {d['rows']} clips)\n")
    b = d["baseline"]
    L.append(f"v1 recall@10 {b['recall@10']:.4f}, recall@30 {b['recall@30']:.4f}, "
             f"misses {d['misses']} ({d['misses']/d['rows']:.3f}).\n")
    L.append("| category | misses | share | rank 31-100 | absent from top 100 |"
             + (" recovered@30 by v2 |" if rec else ""))
    L.append("|---|---:|---:|---:|---:|" + ("---:|" if rec else ""))
    for c in CATS:
        n = d["cats"].get(c, 0)
        if not n:
            continue
        cb = d["cat_by_rank"].get(c, {})
        row = (f"| `{c}` | {n} | {n/d['misses']:.2f} | {cb.get('31_100',0)} "
               f"| {cb.get('beyond_100',0)} |")
        if rec:
            rr = rec.get(s, {}).get(c)
            row += f" {rr['n_rec']}/{rr['n']} ({rr['n_rec']/max(1,rr['n']):.2f}) |" if rr else " - |"
        L.append(row)
    tot31 = d["rank_axis"]["31_100"]
    tot100 = d["rank_axis"]["beyond_100"]
    row = f"| **all** | {d['misses']} | 1.00 | {tot31} | {tot100} |"
    if rec:
        rr = rec.get(s, {}).get("__all__")
        row += f" {rr['n_rec']}/{rr['n']} ({rr['n_rec']/max(1,rr['n']):.2f}) |" if rr else " - |"
    L.append(row)
    L.append("")
    for c in CATS:
        ex = d["examples"].get(c, [])
        if not ex:
            continue
        L.append(f"### {c}, {min(10,len(ex))} examples\n")
        for e in ex[:10]:
            L.append(f"- `{e['term']}` ({e['kind']}, v1 rank {e['rank']}, joined "
                     f"{e['joined']:.0f} / split {e['split']:.0f} / spelled {e['spelled']:.0f} / "
                     f"phon {e['ph']:.0f}; 1-3gram {e['best_1_3']:.0f} vs 4-6gram "
                     f"{e['best_4_6']:.0f}) best span **{e['best_span']}** vs realisation "
                     f"`{e['best_real']}`")
            L.append(f"  - HYP: {e['hyp']}")
        L.append("")
OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT} ({len('\n'.join(L))} bytes)")
