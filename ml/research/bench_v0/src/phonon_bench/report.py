"""run.md, `compare A B`, and `show`."""
from __future__ import annotations

import json
from pathlib import Path

CLASSES = ["ENTITY", "FUNCTION", "ORTHOGRAPHY", "NEAR_MISS", "DROP", "INSERT", "OTHER"]
NOISE = {"fair_wer": 0.002, "term_hit_norm": 0.01}


def _f(x, d=4):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def render_md(res: dict) -> str:
    L = [f"# phonon-bench {res['label']} ({res['backend']}, {res['condition']} lists)", "",
         f"host {res['host']} - started {res['started_utc']} - wall "
         f"{res['stages']['wall']:.1f}s", ""]
    eng = res.get("engine", {})
    L += ["```", json.dumps(eng, indent=1)[:1200], "```", ""]

    L += ["## Stage timing (s)", "", "| stage | seconds |", "|---|---:|"]
    for k, v in res["stages"].items():
        L.append(f"| {k} | {v:.2f} |")
    L.append("")

    L += ["## Quality", "",
          "| set | n | fair WER | raw | strict lc | term hit | loose | hit when retrieved | "
          "damage | win/tie/loss | guard fired | capped | rows/s | tok/s |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, b in res["sets"].items():
        g = b["gen"]
        L.append(
            f"| {name} | {b['n']} | {_f(b['fair_wer'])} | {_f(b.get('raw_fair_wer'))} | "
            f"{_f(b['strict_lc_wer'])} | {_f(b.get('term_hit_norm'), 3)} | "
            f"{_f(b.get('term_hit_loose'), 3)} | {_f(b.get('hit_term_retrieved'), 3)} | "
            f"{_f(b['damage_rate'], 3)} | {b.get('win')}/{b.get('tie')}/{b.get('loss')} | "
            f"{b['guard_fired']} | {b['guard_capped']} | {g['rows_per_s']:.1f} | "
            f"{g['tokens_per_s']:.0f} |")
    L.append("")

    L += ["## False insertions (per clip) and length pathologies", "",
          "| set | insert | not in raw | unrelated | raw floor | excess | spurious | runaway | "
          "truncated |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, b in res["sets"].items():
        L.append(
            f"| {name} | {_f(b['false_insert_rate'], 4)} | "
            f"{_f(b['false_insert_rate_strict'], 4)} | "
            f"{_f(b['false_insert_rate_unrelated'], 4)} | "
            f"{_f(b.get('false_insert_raw_floor_unrelated'), 4)} | "
            f"{_f(b.get('false_insert_excess_unrelated'), 4)} | "
            f"{_f(b.get('false_insert_spurious'), 4)} | {b.get('runaway_rows', '-')} | "
            f"{b.get('truncated_rows', '-')} |")
    L.append("")

    L += ["## Error classes (per 1000 reference words)", "",
          "| set | " + " | ".join(CLASSES) + " | total |",
          "|---|" + "---:|" * (len(CLASSES) + 1)]
    for name, b in res["sets"].items():
        c = b["classes"]
        L.append(f"| {name} | " + " | ".join(f"{c[k]:.1f}" for k in CLASSES)
                 + f" | {sum(c.values()):.1f} |")
    L.append("")

    if res.get("retrieval") and "sets" in res["retrieval"]:
        r = res["retrieval"]
        L += [f"## Retrieval ({r['lexicon']}, {r['pool_terms']} terms, "
              f"two-stage={r['two_stage']}, index {r['index_seconds']:.1f}s)", "",
              "| set | rows | gold pairs | r@10 | r@30 | r@budget | mean list | clips/s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, b in r["sets"].items():
            L.append(f"| {name} | {b['rows']} | {b['gold_pairs']} | {b['recall@10']:.4f} | "
                     f"{b['recall@30']:.4f} | {b['recall@budget']:.4f} | "
                     f"{b['budget_mean_len']:.1f} | {b['clips_per_s']:.1f} |")
        L.append("")

    n = res.get("numerics")
    if n:
        L += ["## Numerics (teacher forced on numerics_500, vs the cached bf16 reference)", ""]
        if n.get("error"):
            L += [f"FAILED: {n['error']}", ""]
        elif n.get("is_reference"):
            L += [f"This run **is** the reference ({n['reference_key']}): "
                  f"{n['build']['n_rows']} rows, {n['build']['n_tokens']} target tokens, "
                  f"{n['build']['hidden_rows']} rows with hidden states, "
                  f"{n['build']['seconds']:.0f}s.", ""]
        else:
            L += ["| metric | value |", "|---|---:|"]
            for k in ("rows_scored", "tokens", "topk", "kl_mean", "kl_p99", "kl_max",
                      "top1_agreement", "hidden_layers", "hidden_max_abs", "hidden_rel_mean"):
                if n.get(k) is not None:
                    L.append(f"| {k} | {_f(n[k], 6) if isinstance(n[k], float) else n[k]} |")
            g = n.get("greedy") or {}
            for k in ("word_agreement_vs_ref", "exact_vs_ref"):
                if g.get(k) is not None:
                    L.append(f"| greedy {k} | {g[k]:.4f} |")
            if n.get("note"):
                L.append(f"| note | {n['note']} |")
            L.append("")
            if n.get("hidden_rel_by_layer"):
                L += ["Per-layer hidden state (mean over rows): max-abs / relative", "",
                      "| layer | max abs | rel |", "|---:|---:|---:|"]
                for i, (a, b) in enumerate(zip(n["hidden_max_abs_by_layer"],
                                               n["hidden_rel_by_layer"])):
                    L.append(f"| {i} | {a:.4g} | {b:.4g} |")
                L.append("")

    lat = res.get("latency")
    if lat:
        L += ["## Latency", "", "| metric | value |", "|---|---:|"]
        for k, v in lat.items():
            L.append(f"| {k} | {_f(v, 2) if isinstance(v, float) else v} |")
        L.append("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- compare
def _load(path_or_label: str, bench_jsonl: Path):
    p = Path(path_or_label)
    if p.is_dir() and (p / "run.json").exists():
        return json.loads((p / "run.json").read_text())
    if p.suffix == ".json" and p.exists():
        return json.loads(p.read_text())
    rows = [json.loads(l) for l in bench_jsonl.read_text().splitlines() if l.strip()]
    hits = [r for r in rows if r["label"] == path_or_label
            or f"{r['label']}/{r['backend']}" == path_or_label or r["ts"] == path_or_label]
    if not hits:
        raise SystemExit(f"no run matching {path_or_label!r} in {bench_jsonl}")
    return json.loads((Path(hits[-1]["run_dir"]) / "run.json").read_text())


def compare(a_key: str, b_key: str, bench_jsonl: Path) -> str:
    a, b = _load(a_key, bench_jsonl), _load(b_key, bench_jsonl)
    L = [f"# phonon-bench compare: {b['label']}/{b['backend']} vs {a['label']}/{a['backend']}",
         "", f"Noise floors: fair WER +/-{NOISE['fair_wer']}, term hit "
         f"+/-{NOISE['term_hit_norm']}. `.` = inside noise.", "",
         "| set | metric | A | B | delta | verdict |", "|---|---|---:|---:|---:|---|"]
    for name in sorted(set(a["sets"]) & set(b["sets"])):
        for metric, floor in (("fair_wer", NOISE["fair_wer"]),
                              ("term_hit_norm", NOISE["term_hit_norm"]),
                              ("damage_rate", None), ("hit_term_retrieved", None)):
            av, bv = a["sets"][name].get(metric), b["sets"][name].get(metric)
            if av is None or bv is None:
                continue
            d = bv - av
            if floor is None:
                verdict = "-"
            elif abs(d) <= floor:
                verdict = "."
            else:
                better = d < 0 if metric.endswith("wer") or metric == "damage_rate" else d > 0
                verdict = "BETTER" if better else "WORSE"
            L.append(f"| {name} | {metric} | {av:.4f} | {bv:.4f} | {d:+.4f} | {verdict} |")
        ac, bc = a["sets"][name]["classes"]["ENTITY"], b["sets"][name]["classes"]["ENTITY"]
        L.append(f"| {name} | ENTITY/1k | {ac:.1f} | {bc:.1f} | {bc - ac:+.1f} | - |")
    L += ["", "| stage | A s | B s | delta |", "|---|---:|---:|---:|"]
    for k in sorted(set(a["stages"]) | set(b["stages"])):
        av, bv = a["stages"].get(k), b["stages"].get(k)
        if av is None or bv is None:
            continue
        L.append(f"| {k} | {av:.2f} | {bv:.2f} | {bv - av:+.2f} |")
    for side, r in (("A", a), ("B", b)):
        n = r.get("numerics") or {}
        if n.get("kl_mean") is not None:
            L.append("")
            L.append(f"{side} numerics: KL mean {n['kl_mean']:.6f} p99 {n['kl_p99']:.6f} "
                     f"top1 {n['top1_agreement']:.4f} "
                     f"greedy word agree {(n.get('greedy') or {}).get('word_agreement_vs_ref')}")
    return "\n".join(L) + "\n"


def show(bench_jsonl: Path, limit: int = 40) -> str:
    rows = [json.loads(l) for l in bench_jsonl.read_text().splitlines() if l.strip()][-limit:]
    L = ["| ts | label | backend | cond | wall s | real_580 fair | real ENTITY | new hit | "
         "unseen hit | pool2 hit | pool3 hit |",
         "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        s = r.get("sets", {})

        def g(k, m, d=3):
            v = s.get(k, {}).get(m)
            return f"{v:.{d}f}" if isinstance(v, (int, float)) else "-"
        L.append(f"| {r['ts']} | {r['label']} | {r['backend']} | {r['condition']} | "
                 f"{r['wall_s']:.0f} | {g('real_580', 'fair_wer', 4)} | "
                 f"{g('real_580', 'ENTITY_per_1k', 1)} | {g('term_new', 'term_hit_norm')} | "
                 f"{g('term_old_unseen', 'term_hit_norm')} | "
                 f"{g('term_pool2', 'term_hit_norm')} | {g('term_pool3', 'term_hit_norm')} |")
    return "\n".join(L) + "\n"
