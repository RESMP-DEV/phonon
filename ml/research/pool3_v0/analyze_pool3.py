"""pool3 eval step 3: tables for every adapter refined into /data/phonon_pool3_v0/refined."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/bigrun_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
import analyze_bigrun as AB  # noqa: E402

P2 = Path("/data/phonon_pool2_v0")
P3 = Path("/data/phonon_pool3_v0")
AB.REF = P3 / "refined"
AB.COND_FILE["pool2"] = P2 / "eval_conditions_pool2.jsonl"
AB.COND_FILE["pool3"] = P3 / "eval_conditions_pool3.jsonl"
AB.TERM_SETS = ["seen", "unseen", "new", "pool2", "pool3"]
AB.OUT = Path("/home/user/phonon/research/pool3_v0")
AB.CORPUS = dict(AB.CORPUS)


def _mix(p, keys):
    s = json.loads(Path(p).read_text())
    return {k: s.get(k) for k in keys}


try:
    mx2 = json.loads((P2 / "mix2_stats.json").read_text())
    AB.CORPUS["bigrun_xl_r16"] = (
        int(mx2["distinct_terms_total"]),
        int(mx2["pool2_rows"]) + int(mx2["replay_rows"]),
        "continue-train of bigrun_mid_r16 on pool 2 + 25 percent replay",
    )
except Exception as exc:
    print(f"no mix2_stats: {exc}", flush=True)
try:
    mx3 = json.loads((P3 / "mix3_stats.json").read_text())
    AB.CORPUS["bigrun_xxl_r16"] = (
        int(mx3["distinct_terms_total"]),
        int(mx3["pool3_rows"]) + int(mx3["replay_p2_rows"]) + int(mx3["replay_p1_rows"]),
        "continue-train of bigrun_xl_r16 on pool 3 + 25/15 percent replay",
    )
except Exception as exc:
    print(f"no mix3_stats: {exc}", flush=True)
AB.ORDER = AB.ORDER + ["bigrun_xl_r16", "bigrun_xxl_r16"]

rc = AB.main()

rp = AB.OUT / "results.json"
res = json.loads(rp.read_text())
for key, p in (("pool2", P2 / "eval_pool2_build_stats.json"),
               ("pool3", P3 / "eval_pool3_build_stats.json")):
    try:
        s = json.loads(p.read_text())
        res["retrieval"][key] = {k: s.get(k) for k in
                                 ("rows", "recall@10", "recall@30", "recall@budget",
                                  "budget_mean_len", "missing_term", "pool")}
    except Exception as exc:
        res["retrieval"][key] = {"error": str(exc)}
for k, p in (("pool3_pool", P3 / "pool3_stats.json"),
             ("pool3_mix", P3 / "mix3_stats.json"),
             ("pool3_pairs", P3 / "pool3_pairs_stats.json"),
             ("pool3_jobs", P3 / "jobs_stats_pool.json"),
             ("pool3_jobs_heldout", P3 / "jobs_stats_heldout.json"),
             ("pool3_throughput", P3 / "throughput.json"),
             ("pool2_mix", P2 / "mix2_stats.json")):
    try:
        res[k] = json.loads(p.read_text())
    except Exception:
        pass
res["order"] = AB.ORDER
rp.write_text(json.dumps(res, indent=1) + "\n")
print("analyze_pool3 wrote", rp, flush=True)
raise SystemExit(rc)
