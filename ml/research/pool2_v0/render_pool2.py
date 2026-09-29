"""pool2: results.json -> research/pool2_v0/results.md (tables only; the read is hand-written)."""
from __future__ import annotations
import json, math
from pathlib import Path

OUT = Path("/home/user/phonon/research/pool2_v0")
RES = OUT / "results.json"
P2 = Path("/data/phonon_pool2_v0")
SETS = ["seen", "unseen", "new", "pool2"]
SET_LABEL = {"seen": "heldout_old seen", "unseen": "heldout_old unseen",
             "new": "heldout_new", "pool2": "heldout_pool2"}


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def main() -> int:
    r = json.loads(RES.read_text())
    L: list[str] = []
    A = L.append

    A("# pool2_v0: a second term pool from outside the repo walk")
    A("")
    A("Built 2026-09-18 on gpubox, both GPUs. Scripts `research/pool2_v0/`, data "
      "`/data/phonon_pool2_v0/`, logs `/data/phonon_pool2_v0/logs/`.")
    A("")
    A("The repo-walk lexicon (15,881 terms) was exhausted by `bigrun_v0`. This build draws a "
      "second pool from sources the walk never saw: symbols of installed packages, the CUDA and "
      "cuBLAS/cuDNN headers, public package registries, the executables and long flags on this "
      "box, and the local HF model cache.")
    A("")

    # ---- pool sources ---------------------------------------------------
    ps = r.get("pool2_pool", {})
    A("## Pool 2 sources")
    A("")
    raw = ps.get("raw_by_source", {})
    kept = ps.get("pool_sources", {})
    kbs = ps.get("kind_by_source", {})
    A("| source | raw | kept | kinds |")
    A("|---|---:|---:|---|")
    for s in sorted(kept, key=lambda k: -kept[k]):
        kinds = ", ".join(f"{k} {v}" for k, v in
                          sorted(kbs.get(s, {}).items(), key=lambda kv: -kv[1]))
        A(f"| `{s}` | {raw.get(s, 0):,} | {kept[s]:,} | {kinds} |")
    A(f"| **total** | **{ps.get('raw_candidates', 0):,}** | **{ps.get('pool_terms', 0):,}** | |")
    A("")
    dr = ps.get("drops", {})
    A(f"Drops: " + ", ".join(f"{k} {v:,}" for k, v in sorted(dr.items(), key=lambda kv: -kv[1]))
      + ". `already_in_pool1` is the case-insensitive overlap with `terms_pool.jsonl`, "
        "`terms_heldout_new.jsonl` and `terms_heldout.jsonl`, so **pool 2 shares no term with "
        "pool 1 or any held-out set**.")
    A("")
    A("Kinds in the training pool: " + ", ".join(
        f"{k} {v:,}" for k, v in ps.get("pool_kinds", {}).items()) + ".")
    A(f"300 terms held out stratified by (kind, source) as `terms_heldout_pool2.jsonl`; "
      f"identifiers were capped at {ps.get('identifier_cap', 0):,} "
      f"(before the cap {ps.get('identifiers_before_cap', 0):,}).")
    A("")

    # ---- throughput -----------------------------------------------------
    tp = r.get("pool2_throughput", {})
    if tp:
        A("## Throughput")
        A("")
        A("| stage | clips / sentences | failed | wall | rate |")
        A("|---|---:|---:|---:|---:|")
        for row in tp.get("stages", []):
            A(f"| {row['stage']} | {row['n']:,} | {row.get('failed', '-')} | "
              f"{row['wall_min']:.1f} min | {row['rate']} |")
        A("")

    # ---- rows -----------------------------------------------------------
    mx = r.get("pool2_mix", {})
    pp = r.get("pool2_pairs", {})
    if mx:
        A("## Training rows")
        A("")
        A("| slice | rows | distinct terms |")
        A("|---|---:|---:|")
        A(f"| pool-2 acoustic | {mx.get('pool2_rows', 0):,} | {mx.get('pool2_terms', 0):,} |")
        A(f"| replay of `train_big` acoustic ({mx.get('replay_fraction', 0):.0%}) | "
          f"{mx.get('replay_rows', 0):,} | {mx.get('replay_terms', 0):,} |")
        A(f"| real `train_date` x{mx.get('real_repeat', 0)} | {mx.get('real_rows', 0):,} | "
          f"{mx.get('real_unique', 0):,} unique |")
        A(f"| **`train_pool2.jsonl`** | **{mx.get('rows', 0):,}** | "
          f"**{mx.get('distinct_terms_total', 0):,}** |")
        A("")
        A("Branches: " + ", ".join(f"{k} {v:,}" for k, v in
                                   sorted(mx.get("branches", {}).items(), key=lambda kv: -kv[1]))
          + f". Zero-edit rows capped at 30 percent per (sentence_idx, voice) bucket: "
            f"{pp.get('zero_edit_kept', 0):,} kept of {pp.get('zero_edit_available', 0):,} "
            f"available ({pp.get('zero_edit_fraction_pool', 0):.1%} of the pool).")
        A("")

    # ---- retrieval ------------------------------------------------------
    ret = r.get("retrieval", {})
    A("## Retrieval recall over the union lexicon")
    A("")
    A("| set | pool | recall@10 | recall@30 | missing |")
    A("|---|---:|---:|---:|---:|")
    for s in SETS:
        d = ret.get(s) or {}
        if "error" in d:
            continue
        A(f"| {SET_LABEL[s]} | {d.get('pool', '-')} | {f3(d.get('recall@10'))} | "
          f"{f3(d.get('recall@30'))} | {d.get('missing_term', '-')} |")
    A("")

    # ---- eval -----------------------------------------------------------
    terms = r.get("terms", {})
    labels = [l for l in r.get("order", []) if l in terms] or list(terms)
    want = [l for l in ("raw", "bigrun_mid_r16", "bigrun_big_r16", "bigrun_xl_r16") if l in terms]
    A("## Term hit, damage and unrelated insertions")
    A("")
    A("`hit` is normalised term hit, `dmg` the fraction of clips whose fair WER got worse than "
      "the raw Parakeet hypothesis, `ins` the unrelated-insertion rate.")
    A("")
    for s in SETS:
        if not any(s in terms.get(l, {}) for l in want):
            continue
        A(f"### {SET_LABEL[s]}")
        A("")
        A("| adapter | cond | hit | hit(retrieved) | hit(missing) | dmg | ins | fair WER |")
        A("|---|---|---:|---:|---:|---:|---:|---:|")
        for lab in want:
            b = terms.get(lab, {}).get(s, {})
            for cond in ("none", "oracle", "retrieved"):
                if cond not in b:
                    continue
                d = b[cond]
                A(f"| {lab} | {cond} | {f3(d.get('term_hit_norm'))} | "
                  f"{f3(d.get('hit_term_retrieved'))} | {f3(d.get('hit_term_missing'))} | "
                  f"{f3(d.get('damage_rate'))} | {f3(d.get('false_insert_rate_unrelated'))} | "
                  f"{f3(d.get('fair_wer'))} |")
        A("")

    real = r.get("real", {})
    if real:
        A("### Real audio (date-split holdout)")
        A("")
        A("| adapter | cond | fair WER | strict WER | win | tie | loss | dmg | ins |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for lab in [l for l in ("raw", "bigrun_mid_r16", "bigrun_big_r16", "bigrun_xl_r16") if l in real]:
            for cond in ("none", "retrieved"):
                d = real[lab].get(cond)
                if not d:
                    continue
                A(f"| {lab} | {cond} | {f3(d.get('fair_wer'))} | {f3(d.get('strict_lc_wer'))} | "
                  f"{d.get('win', '-')} | {d.get('tie', '-')} | {d.get('loss', '-')} | "
                  f"{f3(d.get('damage_rate'))} | {f3(d.get('false_insert_rate_unrelated'))} |")
        A("")

    # ---- scaling --------------------------------------------------------
    ft = r.get("fit_terms", {})
    A("## The scaling curve with a pool-2 point")
    A("")
    A("| run | distinct acoustic terms | seen-voice term hit (retrieved) |")
    A("|---|---:|---:|")
    for name, t, h in ft.get("scaling_v0_points", []):
        A(f"| {name} | {t:,} | {h:.3f} |")
    for name, t, h in ft.get("bigrun_points", []):
        A(f"| {name} | {t:,} | {h:.3f} |")
    A("")
    for k in ("scaling_v0_only", "extended"):
        d = ft.get(k)
        if d:
            A(f"- fit `{k}`: hit = {d['intercept']:.3f} + "
              f"{d['slope_per_decade']:.3f} x log10(terms), r2 = {d['r2']:.3f}, "
              f"rmse = {d['rmse']:.4f}")
    if "slope_2107_to_14537" in ft:
        A(f"- measured slope 2,107 -> 14,537 terms: {ft['slope_2107_to_14537']:.3f} per decade")
    A("")

    md = OUT / "results.md"
    head = md.read_text() if md.exists() else ""
    read = ""
    if "## Read" in head:
        read = head[head.index("## Read"):]
    md.write_text("\n".join(L) + ("\n" + read if read else "\n"))
    print(f"wrote {md} ({len(L)} lines)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
