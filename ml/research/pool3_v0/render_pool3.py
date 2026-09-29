"""pool3: results.json -> research/pool3_v0/results.md (tables only; the read is hand-written)."""
from __future__ import annotations
import json, math
from pathlib import Path

OUT = Path("/home/user/phonon/research/pool3_v0")
RES = OUT / "results.json"
SETS = ["seen", "unseen", "new", "pool2", "pool3"]
SET_LABEL = {"seen": "heldout_old seen", "unseen": "heldout_old unseen",
             "new": "heldout_new", "pool2": "heldout_pool2", "pool3": "heldout_pool3"}
WANT = ["raw", "bigrun_mid_r16", "bigrun_big_r16", "bigrun_xl_r16", "bigrun_xxl_r16"]


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def f4(x):
    return "-" if x is None else f"{x:.4f}"


def main() -> int:
    r = json.loads(RES.read_text())
    L: list[str] = []
    A = L.append

    A("# pool3_v0: pool-3 audio and the xxl continue-train")
    A("")
    A("Built 2026-09-18/19 on gpubox, GPU 1 only. Scripts `research/pool3_v0/`, data "
      "`/data/phonon_pool3_v0/`, logs `/data/phonon_pool3_v0/logs/`.")
    A("")
    ps = r.get("pool3_pool", {})
    A(f"Pool 3 is {ps.get('pool_terms', 0):,} terms plus {ps.get('heldout_terms', 0)} held out, "
      f"extracted by `research/index_v0` from 150 shallow-cloned public repositories "
      f"(25 each across Python, Rust, C/C++, CUDA, JS/TS, Go) plus "
      f"{ps.get('grammar_keywords', 0)} tree-sitter grammar keywords, deduped against pools 1-2 "
      f"and every held-out set. It had no audio until this run.")
    A("")
    A("**Scale deviation.** Pool 2 gave every training term 4 sentences x 3 of the 9 Kokoro "
      "training voices (12 clips per term). One GPU and a 02:30 stop do not fit that at 36k "
      "terms, so pool 3 uses 2 sentences x 2 voices (4 clips per term) and keeps every distinct "
      "term. `bigrun_v0` measured that doubling acoustic rows at fixed term coverage is worth "
      "+0.007 seen term hit, while distinct terms carry ~0.096 per decade, so the axis under "
      "test is the one that was protected.")
    A("")

    tp = r.get("pool3_throughput", {})
    if tp:
        A("## Throughput")
        A("")
        A("| stage | clips / sentences | failed | wall | rate |")
        A("|---|---:|---:|---:|---:|")
        for row in tp.get("stages", []):
            A(f"| {row['stage']} | {row['n']:,} | {row.get('failed', '-')} | "
              f"{row['wall_min']:.1f} min | {row['rate']} |")
        A("")

    mx = r.get("pool3_mix", {})
    pp = r.get("pool3_pairs", {})
    jb = r.get("pool3_jobs", {})
    if mx:
        A("## Corpus")
        A("")
        A("| slice | rows | distinct terms |")
        A("|---|---:|---:|")
        A(f"| pool-3 acoustic | {mx.get('pool3_rows', 0):,} | {mx.get('pool3_terms', 0):,} |")
        A(f"| replay of the pool-2 acoustic rows ({mx.get('replay_p2_fraction', 0):.0%}) | "
          f"{mx.get('replay_p2_rows', 0):,} | {mx.get('replay_p2_terms', 0):,} |")
        A(f"| replay of the pool-1 (`train_big`) acoustic rows "
          f"({mx.get('replay_p1_fraction', 0):.0%}) | {mx.get('replay_p1_rows', 0):,} | "
          f"{mx.get('replay_p1_terms', 0):,} |")
        A(f"| real `train_date` x{mx.get('real_repeat', 0)} | {mx.get('real_rows', 0):,} | "
          f"{mx.get('real_unique', 0):,} unique |")
        A(f"| **`train_pool3.jsonl`** | **{mx.get('rows', 0):,}** | "
          f"**{mx.get('distinct_terms_total', 0):,}** |")
        A("")
        A(f"Clips: {jb.get('clips', 0):,} for {jb.get('terms', 0):,} terms "
          f"({jb.get('sentences', 0):,} sentences x {jb.get('voices_per_term', 0)} voices). "
          "Branches: " + ", ".join(f"{k} {v:,}" for k, v in
                                   sorted(mx.get("branches", {}).items(),
                                          key=lambda kv: -(kv[1] or 0)) if k)
          + f". Zero-edit rows capped at 30 percent per (sentence_idx, voice) bucket: "
            f"{pp.get('zero_edit_kept', 0):,} kept of {pp.get('zero_edit_available', 0):,} "
            f"available ({pp.get('zero_edit_fraction_pool', 0):.1%} of the pool).")
        A("")

    ret = r.get("retrieval", {})
    A("## Retrieval recall")
    A("")
    A("Training lists: the shipped `vocab_v1` retriever over the union lexicon "
      "(pools 1-3 + every held-out set), top-30, as pool 2 built them. `heldout_pool3` "
      "conditions: the two-stage v2 retriever of `research/retrieval_v2/retrieve2.py` over the "
      "same union pool, with the `index_v0` adaptive budget `min(80, max(30, 3 x words))` at "
      "threshold 65 reported alongside the fixed top-30 that the term-hit numbers use.")
    A("")
    A("| set | pool | recall@10 | recall@30 | recall@budget | mean budget list | missing |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for s in SETS:
        d = ret.get(s) or {}
        if "error" in d or not d:
            continue
        bl = d.get("budget_mean_len")
        A(f"| {SET_LABEL[s]} | {d.get('pool', '-')} | {f3(d.get('recall@10'))} | "
          f"{f3(d.get('recall@30'))} | {f3(d.get('recall@budget'))} | "
          f"{'-' if bl is None else f'{bl:.1f}'} | {d.get('missing_term', '-')} |")
    A("")

    terms = r.get("terms", {})
    want = [l for l in WANT if l in terms]
    A("## Term hit, damage and unrelated insertions")
    A("")
    A("`hit` is normalised term hit, `dmg` the fraction of clips whose fair WER got worse than "
      "the raw Parakeet hypothesis, `ins` the unrelated-insertion rate. Decode-time length "
      "guard on.")
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
        A("### Real audio (580-row date-split holdout)")
        A("")
        A("| adapter | cond | fair WER | strict WER | ENTITY /1000 | win | tie | loss | dmg | ins |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for lab in [l for l in WANT if l in real]:
            for cond in ("none", "retrieved"):
                d = real[lab].get(cond)
                if not d:
                    continue
                cls = (d.get("classes") or {})
                ent = cls.get("ENTITY_per_1000", cls.get("ENTITY"))
                A(f"| {lab} | {cond} | {f4(d.get('fair_wer'))} | {f4(d.get('strict_lc_wer'))} | "
                  f"{'-' if ent is None else f'{ent:.1f}'} | "
                  f"{d.get('win', '-')} | {d.get('tie', '-')} | {d.get('loss', '-')} | "
                  f"{f3(d.get('damage_rate'))} | {f3(d.get('false_insert_rate_unrelated'))} |")
        A("")

    ft = r.get("fit_terms", {})
    A("## The scaling curve extended to the pool-3 point")
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
    ext = ft.get("extended")
    if ext:
        a, b = ext["intercept"], ext["slope_per_decade"]
        A("")
        A("| distinct terms | fit prediction | measured |")
        A("|---:|---:|---:|")
        meas = {int(t): h for _n, t, h in ext.get("points", [])}
        for t in sorted(meas):
            A(f"| {t:,} | {a + b * math.log10(t):.3f} | {meas[t]:.3f} |")
    A("")
    A("Ceilings: a held-out set can only score as high as its own retrieval recall lets it, so "
      "the `heldout_*` rows below the seen-voice curve are read against the recall table above, "
      "not against each other.")
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
