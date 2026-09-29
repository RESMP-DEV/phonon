"""Build research/ship_v0/results.md and results.json from what the queue produced."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

BENCH = Path("/data/phonon_bench_v0/bench.jsonl")
STATE = Path("/data/phonon_queue/ship_v0/state.json")
SHIP = Path("/data/phonon_ship_v0")
OUT = Path("/home/user/phonon/research/ship_v0")
SETS = ["real_580", "term_old_seen", "term_old_unseen", "term_new", "term_pool2", "term_pool3"]
ORDER = [
    ("ship_bigrun_mid_r16", "bigrun_mid_r16 (ref)"),
    ("ship_restr_real12", "restr_real12 (ref)"),
    ("ship_ship_lfm12_v0", "ship_lfm12_v0"),
    ("ship_ship_lfm12_wide_v0", "ship_lfm12_wide_v0"),
    ("ship_ship_qwen17_v0", "ship_qwen17_v0"),
    ("ship_ship_lfm12_generic_v0", "ship_lfm12_generic_v0"),
    ("ship_ship_lfm12_generic_personal_v0", "ship_lfm12_generic_personal_v0"),
]
JOBS = ["corpus_wide", "corpus_gen_personal", "ship_lfm12", "ship_qwen17", "ship_lfm12_gen",
        "ship_lfm12_gen_personal", "ship_lfm12_wide", "retrieval_mp", "bench_bigrun_mid_r16",
        "bench_restr_real12", "bench_ship_lfm12", "bench_ship_qwen17", "bench_ship_lfm12_gen",
        "bench_ship_lfm12_gen_personal", "bench_ship_lfm12_wide", "quant_lfm12",
        "quant_lfm12_wide", "bench_retrieval_check"]


def lines():
    out = {}
    if BENCH.exists():
        for l in BENCH.read_text().splitlines():
            if not l.strip():
                continue
            d = json.loads(l)
            out[d["label"]] = d          # last run under a label wins
    return out


def f(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"


def table(rows, head, align=None):
    align = align or ["---"] * len(head)
    w = [max(len(str(head[i])), *(len(str(r[i])) for r in rows)) if rows else len(str(head[i]))
         for i in range(len(head))]
    o = ["| " + " | ".join(str(head[i]).ljust(w[i]) for i in range(len(head))) + " |",
         "|" + "|".join(("-" * (w[i] + 2)) if align[i] == "---" else
                        ("-" * (w[i] + 1)) + ":" for i in range(len(head))) + "|"]
    for r in rows:
        o.append("| " + " | ".join(str(r[i]).ljust(w[i]) for i in range(len(head))) + " |")
    return "\n".join(o)


def main():
    B = lines()
    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "ship": {}, "quant": {},
           "queue": {}, "retrieval": {}}

    # ---------------------------------------------------------------- ship table
    rows = []
    for label, name in ORDER:
        d = B.get(label)
        if not d:
            rows.append([name, "MISSING", "-", "-", "-", "-", "-", "-"])
            continue
        res["ship"][name] = d["sets"]
        for s in SETS:
            v = d["sets"].get(s)
            if not v:
                continue
            rows.append([name if s == SETS[0] else "", s, f(v.get("fair_wer")),
                         f(v.get("damage_rate"), 3), f(v.get("ENTITY_per_1k"), 1),
                         f(v.get("term_hit_norm"), 3), f(v.get("hit_term_retrieved"), 3),
                         f(v.get("false_insert_rate_unrelated"), 4)])
    ship_tbl = table(rows, ["adapter", "set", "fair WER", "damage", "ENTITY/1k",
                            "term hit (ret)", "hit when retrieved", "false ins (unrel)"],
                     ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:"])

    # ---------------------------------------------------------------- quant table
    qrows = []
    for base in ("ship_lfm12_v0", "ship_lfm12_wide_v0"):
        sz = {}
        p = SHIP / "quant" / base / "sizes.json"
        if p.exists():
            sz = json.loads(p.read_text())
        for label, what, mb in ((f"ship_{base}_bf16", "bf16 (merged, hf)",
                                 sz.get("merged_bf16_MB")),
                                (f"ship_{base}_mlx_attn6emb8", "MLX custom_attn6emb8",
                                 sz.get("mlx_attn6emb8_MB")),
                                (f"ship_{base}_gguf_IQ4_XS", "GGUF IQ4_XS (mixed imatrix)",
                                 sz.get("gguf_IQ4_XS_MB"))):
            d = B.get(label)
            if not d:
                qrows.append([base, what, mb or "-", "MISSING", "-", "-", "-", "-", "-", "-"])
                continue
            res["quant"][label] = {"sets": d["sets"], "numerics": d.get("numerics"), "MB": mb}
            n = d.get("numerics") or {}
            g = (n.get("greedy") or {})
            r5 = d["sets"].get("real_580", {})
            tn = d["sets"].get("term_new", {})
            qrows.append([base, what, mb or "-", f(r5.get("fair_wer")),
                          f(r5.get("ENTITY_per_1k"), 1), f(tn.get("term_hit_norm"), 3),
                          f(g.get("word_agreement_vs_ref")), f(g.get("exact_vs_ref"), 3),
                          f(n.get("kl_mean"), 6), f(n.get("top1_agreement"))])
    quant_tbl = table(qrows, ["adapter", "build", "MB", "real_580 fair WER",
                              "real ENTITY/1k", "term_new hit", "word agree vs bf16",
                              "exact vs bf16", "KL mean", "top-1 agree"],
                      ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"])

    # ---------------------------------------------------------------- queue timing
    st = json.loads(STATE.read_text()) if STATE.exists() else {"jobs": {}}
    t0 = min((j.get("started_at") or 1e18) for j in st["jobs"].values()) if st["jobs"] else 0
    jrows = []
    for n in JOBS:
        j = st["jobs"].get(n)
        if not j:
            continue
        res["queue"][n] = {k: j.get(k) for k in ("status", "gpu", "attempts", "wall_s",
                                                 "inner_s", "overhead_s", "exit_code",
                                                 "started_at", "ended_at")}
        s = j.get("started_at")
        e = j.get("ended_at")
        jrows.append([n, j["kind"], j.get("gpu", "-"), j["status"], j.get("attempts", 0),
                      f"{j['wall_s']:.0f}" if j.get("wall_s") else "-",
                      f"{j['inner_s']:.0f}" if j.get("inner_s") else "-",
                      f"{j['overhead_s']:.2f}" if j.get("overhead_s") is not None else "-",
                      f"{(s - t0) / 60:.1f}" if s else "-",
                      f"{(e - t0) / 60:.1f}" if e else "-"])
    queue_tbl = table(jrows, ["job", "kind", "gpu", "status", "try", "wall_s", "inner_s",
                              "harness_s", "start min", "end min"],
                      ["---", "---", "---", "---", "--:", "--:", "--:", "--:", "--:", "--:"])

    # ---------------------------------------------------------------- gpu utilisation
    samples = defaultdict(list)
    mem = defaultdict(list)
    p = SHIP / "gpu_samples.csv"
    if p.exists():
        for l in p.read_text().splitlines():
            parts = [x.strip() for x in l.split(",")]
            if len(parts) == 4 and parts[1].isdigit():
                samples[int(parts[1])].append(int(parts[2]))
                mem[int(parts[1])].append(int(parts[3]))
    urows = []
    for g in sorted(samples):
        v = samples[g]
        m = mem[g]
        urows.append([g, len(v), f"{sum(v) / len(v):.1f}", f"{sum(1 for x in v if x >= 50) / len(v):.2f}",
                      f"{max(m) / 1024:.1f}", f"{sum(m) / len(m) / 1024:.1f}"])
    util_tbl = table(urows, ["gpu", "samples", "mean util %", "frac >=50%", "peak GB",
                             "mean GB"], ["--:"] * 6)
    res["gpu_util"] = {g: {"mean": sum(samples[g]) / len(samples[g]),
                           "peak_mem_MiB": max(mem[g])} for g in samples}

    # ---------------------------------------------------------------- retrieval
    rp = SHIP / "retrieval_mp.json"
    rtbl = "not produced"
    if rp.exists():
        r = json.loads(rp.read_text())
        res["retrieval"] = r
        rr = []
        for k, v in r.get("timing", {}).items():
            if k.startswith("procs_"):
                rr.append([k.split("_")[1], f"{v['total_s']:.1f}",
                           " ".join(f"{a}={b}" for a, b in v["per_set"].items())])
        rtbl = table(rr, ["procs", "four sets, s", "per set"], ["--:", "--:", "---"])
    res["retrieval_table"] = rtbl

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "tables.md").write_text(
        "## Ship table\n\n" + ship_tbl + "\n\n## Quantization\n\n" + quant_tbl
        + "\n\n## Queue\n\n" + queue_tbl + "\n\n## GPU utilisation (all four, 20 s samples)\n\n"
        + util_tbl + "\n\n## Retrieval\n\n" + rtbl + "\n")
    print((OUT / "tables.md").read_text())


if __name__ == "__main__":
    main()
