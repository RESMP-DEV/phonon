"""Which v1 misses does v2 recover at 30?"""
import json
from pathlib import Path
D = Path("/data/phonon_retrieval_v2")
M = json.loads((D / "misses.json").read_text())
COND = {"new": D / "cond_new_v2.jsonl", "old_unseen": D / "cond_unseen_v2.jsonl"}
out = {}
for s, p in COND.items():
    has = {}
    for l in p.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            has[r["id"]] = bool(r["retrieved_has_term"])
    agg = {}
    for rec in M[s]["per_miss"]:
        for k in (rec["cat"], "__all__"):
            a = agg.setdefault(k, {"n": 0, "n_rec": 0})
            a["n"] += 1
            a["n_rec"] += int(has.get(rec["id"], False))
    out[s] = agg
    print(s, {k: f"{v['n_rec']}/{v['n']}" for k, v in agg.items()}, flush=True)
(D / "recovered.json").write_text(json.dumps(out, indent=1))
print("recovered done")
