"""Order the train_mid acoustic clips terms-first (round robin over distinct terms)."""
import json, random
from collections import defaultdict
from pathlib import Path

D = Path("/data/phonon_bigrun_v0")
out = Path("/data/phonon_nbest_v0/mid_order.jsonl")
by_term = defaultdict(list)
n = 0
for half in (0, 1):
    for line in (D / f"tts_manifest_mid_half{half}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("id") in (None, "_meta"):
            continue
        by_term[r.get("term") or ""].append({"id": r["id"], "wav": r["wav"], "term": r.get("term")})
        n += 1
rng = random.Random(918)
for v in by_term.values():
    rng.shuffle(v)
terms = sorted(by_term)
rng.shuffle(terms)
order = []
i = 0
while True:
    added = 0
    for t in terms:
        v = by_term[t]
        if i < len(v):
            order.append(v[i])
            added += 1
    if not added:
        break
    i += 1
with out.open("w", encoding="utf-8") as h:
    for r in order:
        h.write(json.dumps(r) + "\n")
print(f"clips={n} distinct_terms={len(terms)} ordered={len(order)} -> {out}")
