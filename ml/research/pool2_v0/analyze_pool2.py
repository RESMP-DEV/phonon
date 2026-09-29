"""pool2 eval step 3: tables for every adapter refined into /data/phonon_pool2_v0/refined."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/bigrun_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
import analyze_bigrun as AB  # noqa: E402

P2 = Path("/data/phonon_pool2_v0")
AB.REF = P2 / "refined"
AB.COND_FILE["pool2"] = P2 / "eval_conditions_pool2.jsonl"
AB.TERM_SETS = ["seen", "unseen", "new", "pool2"]
AB.OUT = Path("/home/user/phonon/research/pool2_v0")
AB.CORPUS = dict(AB.CORPUS)
try:
    _mx = json.loads((P2 / "mix2_stats.json").read_text())
    AB.CORPUS["bigrun_xl_r16"] = (
        int(_mx["distinct_terms_total"]),
        int(_mx["pool2_rows"]) + int(_mx["replay_rows"]),
        "continue-train of bigrun_mid_r16 on pool 2 + 25 percent replay",
    )
except Exception as _exc:
    print(f"no mix2_stats yet: {_exc}", flush=True)
AB.ORDER = AB.ORDER + ["bigrun_xl_r16"]

rc = AB.main()

# attach the pool2 retrieval stats the wrapped analyzer cannot find
rp = AB.OUT / "results.json"
res = json.loads(rp.read_text())
try:
    s = json.loads((P2 / "eval_pool2_build_stats.json").read_text())
    res["retrieval"]["pool2"] = {k: s.get(k) for k in
                                 ("rows", "recall@10", "recall@30", "missing_term", "pool")}
except Exception as exc:
    res["retrieval"]["pool2"] = {"error": str(exc)}
for k, p in (("pool2_pool", P2 / "pool2_stats.json"), ("pool2_mix", P2 / "mix2_stats.json"),
             ("pool2_pairs", P2 / "pool2_pairs_stats.json"),
             ("pool2_jobs", P2 / "jobs_stats_pool.json"),
             ("pool2_throughput", P2 / "throughput.json")):
    try:
        res[k] = json.loads(p.read_text())
    except Exception:
        pass
rp.write_text(json.dumps(res, indent=1) + "\n")
print("analyze_pool2 wrote", rp, flush=True)
raise SystemExit(rc)
