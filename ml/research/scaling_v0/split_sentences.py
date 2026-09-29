"""Split sentences_8.jsonl into two halves by term, for one TTS chain per GPU."""
import json
from pathlib import Path
SRC = Path("/data/phonon_scaling_v0/sentences_8.jsonl")
rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
terms = sorted({r["term"] for r in rows})
side = {t: i % 2 for i, t in enumerate(terms)}
halves = [[], []]
for r in rows:
    halves[side[r["term"]]].append(r)
for i, h in enumerate(halves):
    p = SRC.parent / f"sentences_8_half{i}.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in h:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"half{i} sentences={len(h)} terms={len({r['term'] for r in h})} -> {p}")
print(f"total sentences={len(rows)} terms={len(terms)}")
by_term = {}
for r in rows:
    by_term.setdefault(r["term"], 0)
    by_term[r["term"]] += 1
from collections import Counter
print("sentences-per-term histogram:", dict(sorted(Counter(by_term.values()).items())))
