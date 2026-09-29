"""Compare synth_v2 pairs to real Wispr and to synth_v1. Length hist + WER table."""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

V1 = Path("/home/user/phonon/research/synth_v1")
sys.path.insert(0, str(V1))
from corrupt import _load_v0  # noqa: E402
from pollution import has_identifier_dump  # noqa: E402

import importlib.util as _ilu

_v2_spec = _ilu.spec_from_file_location("synth_v2_paths", Path(__file__).resolve().parent / "paths.py")
assert _v2_spec and _v2_spec.loader
_v2p = _ilu.module_from_spec(_v2_spec)
_v2_spec.loader.exec_module(_v2p)
BIN_NAMES = _v2p.BIN_NAMES
BINS = _v2p.BINS
FILTER_STATS = _v2p.FILTER_STATS
LEXICON_PATH = _v2p.LEXICON_PATH
PAIRS_PATH = _v2p.PAIRS_PATH
REAL_PAIRS = _v2p.REAL_PAIRS
RESEARCH_ROOT = _v2p.RESEARCH_ROOT
V1_PAIRS = _v2p.V1_PAIRS
V1_RESEARCH = _v2p.V1_RESEARCH

_errors = _load_v0("errors")
CLASSES = _errors.CLASSES
classify_pair = _errors.classify_pair
try_import_sibling_classifier = _errors.try_import_sibling_classifier

_spec = _ilu.spec_from_file_location(
    "corrector_v0_common",
    Path("/home/user/phonon/research/corrector_v0/common.py"),
)
assert _spec and _spec.loader
_common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_common)
fair_norm = _common.fair_norm
pair_wer = _common.pair_wer

try:
    sibling_fn = try_import_sibling_classifier()
except Exception:
    sibling_fn = None


def bin_name(n: int) -> str:
    for (a, b), name in zip(BINS, BIN_NAMES, strict=True):
        if a <= n <= b:
            return name
    return "0"


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    return float(ys[f] * (c - k) + ys[c] * (k - f))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def classify_row(target: str, raw: str) -> dict:
    if sibling_fn is not None:
        try:
            out = sibling_fn(target, raw)
            if isinstance(out, dict) and "rates" in out:
                return out
        except Exception:
            pass
    return classify_pair(target, raw)


def length_hist(rows: list[dict]) -> dict:
    ns = [len((r.get("target") or "").split()) for r in rows]
    counts = Counter(bin_name(n) for n in ns)
    n = len(ns) or 1
    return {
        "n": len(ns),
        "counts": {k: int(counts.get(k, 0)) for k in BIN_NAMES},
        "fracs": {k: counts.get(k, 0) / n for k in BIN_NAMES},
        "pct": {k: 100.0 * counts.get(k, 0) / n for k in BIN_NAMES},
        "mean": statistics.fmean(ns) if ns else 0.0,
        "median": statistics.median(ns) if ns else 0.0,
        "p10": percentile([float(x) for x in ns], 0.10) if ns else 0.0,
        "p90": percentile([float(x) for x in ns], 0.90) if ns else 0.0,
    }


def summarize(name: str, rows: list[dict], lexicon_terms: set[str]) -> dict:
    wers = []
    n_words = []
    class_counts = Counter()
    n_ref = 0
    entity_terms_real = Counter()
    entity_in_lex = 0
    entity_total = 0
    n_dump = 0
    n_zero = 0
    n_le05 = 0
    for i, row in enumerate(rows):
        target = (row.get("target") or "").strip()
        raw = (row.get("input") or row.get("asr") or "").strip()
        if not target or not raw:
            continue
        try:
            w = pair_wer(target, raw, fair_norm)
        except Exception:
            continue
        wers.append(w)
        if w == 0:
            n_zero += 1
        if w <= 0.05:
            n_le05 += 1
        n_words.append(len(target.split()))
        if has_identifier_dump(target):
            n_dump += 1
        cls = classify_row(target, raw)
        n_ref += cls.get("n_ref") or 0
        for k, v in (cls.get("counts") or {}).items():
            class_counts[k] += v
        for d in cls.get("details") or []:
            if d.get("cls") == "ENTITY" and d.get("ref"):
                entity_total += 1
                ref = d["ref"]
                entity_terms_real[ref] += 1
                if ref in lexicon_terms or ref.lower() in lexicon_terms:
                    entity_in_lex += 1
        if (i + 1) % 2000 == 0:
            print(f"  {name} scored {i + 1}/{len(rows)}", flush=True)
    rates = {c: (class_counts[c] / n_ref if n_ref else 0.0) for c in CLASSES}
    return {
        "name": name,
        "n": len(wers),
        "wer_mean": statistics.fmean(wers) if wers else 0.0,
        "wer_median": statistics.median(wers) if wers else 0.0,
        "wer_p90": percentile(wers, 0.90),
        "len_mean": statistics.fmean(n_words) if n_words else 0.0,
        "len_median": statistics.median(n_words) if n_words else 0.0,
        "len_p90": percentile([float(x) for x in n_words], 0.90) if n_words else 0.0,
        "class_rates": rates,
        "class_counts": dict(class_counts),
        "n_ref_tokens": n_ref,
        "entity_overlap": (entity_in_lex / entity_total) if entity_total else 0.0,
        "entity_total": entity_total,
        "entity_in_lex": entity_in_lex,
        "top_entity_terms": entity_terms_real.most_common(20),
        "n_wer_zero": n_zero,
        "n_wer_le05": n_le05,
        "frac_wer_zero": (n_zero / len(wers) if wers else 0.0),
        "frac_wer_le05": (n_le05 / len(wers) if wers else 0.0),
        "n_identifier_dump": n_dump,
    }


def fmt(x: float) -> str:
    return f"{x:.4f}"


def table_row(s: dict) -> str:
    r = s["class_rates"]
    return (
        f"| {s['name']} | {s['n']} | {fmt(s['wer_mean'])} | {fmt(s['wer_median'])} | "
        f"{fmt(s['wer_p90'])} | {fmt(s['len_mean'])} | {fmt(s['len_median'])} | {fmt(s['len_p90'])} | "
        f"{fmt(r.get('ENTITY', 0.0))} | {fmt(r.get('FUNCTION', 0.0))} | {fmt(r.get('ORTHOGRAPHY', 0.0))} | "
        f"{fmt(r.get('NEAR_MISS', 0.0))} | {fmt(r.get('DROP', 0.0))} | {fmt(r.get('INSERT', 0.0))} |"
    )


def hist_table(hists: dict[str, dict]) -> list[str]:
    lines = [
        "| set | n | 1-5 | 6-10 | 11-20 | 21-40 | 41-80 | 81+ | median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, h in hists.items():
        p = h["pct"]
        lines.append(
            f"| {name} | {h['n']} | {p['1-5']:.1f}% | {p['6-10']:.1f}% | {p['11-20']:.1f}% | "
            f"{p['21-40']:.1f}% | {p['41-80']:.1f}% | {p['81+']:.1f}% | {h['median']:.1f} |"
        )
    return lines


def v1_as_summary(blob: dict, name: str) -> dict:
    rates = blob.get("class_rates") or {}
    return {
        "name": name,
        "n": blob.get("n") or 0,
        "wer_mean": blob.get("wer_mean") or 0.0,
        "wer_median": blob.get("wer_median") or 0.0,
        "wer_p90": blob.get("wer_p90") or 0.0,
        "len_mean": blob.get("len_mean") or 0.0,
        "len_median": blob.get("len_median") or 0.0,
        "len_p90": blob.get("len_p90") or 0.0,
        "class_rates": {c: float(rates.get(c) or 0.0) for c in CLASSES},
        "class_counts": blob.get("class_counts") or {},
        "n_ref_tokens": blob.get("n_ref_tokens") or 0,
        "frac_wer_zero": blob.get("frac_wer_zero") or 0.0,
        "frac_wer_le05": blob.get("frac_wer_le05") or 0.0,
    }


def main() -> int:
    RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    lexicon = load_jsonl(LEXICON_PATH)
    lex_terms = {r["term"] for r in lexicon} | {r["term"].lower() for r in lexicon}

    real_raw = load_jsonl(REAL_PAIRS)
    real_rows = []
    for row in real_raw:
        target = (row.get("target") or row.get("kept") or "").strip()
        raw = (row.get("asr") or row.get("input") or row.get("raw") or "").strip()
        if not target or not raw:
            continue
        real_rows.append({"input": raw, "asr": raw, "target": target})

    v1 = load_jsonl(V1_PAIRS)
    v2 = load_jsonl(PAIRS_PATH)
    v2_short = [r for r in v2 if r.get("synth_version") == "v2_short"]
    v2_em = [r for r in v2 if r.get("route") == "error_model"]
    v2_tts = [r for r in v2 if r.get("route") == "tts_asr"]

    print("summarize real", flush=True)
    stats_real = summarize("real_wispr", real_rows, lex_terms)
    print("summarize v1", flush=True)
    stats_v1 = summarize("synth_v1_all", v1, lex_terms)
    print("summarize v2 all", flush=True)
    stats_v2 = summarize("synth_v2_all", v2, lex_terms)
    print("summarize v2 short", flush=True)
    stats_v2s = summarize("synth_v2_short", v2_short, lex_terms)

    v1_blob = json.loads((V1_RESEARCH / "results.json").read_text()) if (V1_RESEARCH / "results.json").exists() else {}
    stats_v1_em = v1_as_summary(v1_blob.get("synth_v1_error_model") or {}, "synth_v1_error_model")
    stats_v1_tts = v1_as_summary(v1_blob.get("synth_v1_tts_asr") or {}, "synth_v1_tts_asr")

    h_real = length_hist(real_rows)
    h_v1 = length_hist(v1)
    h_v2 = length_hist(v2)
    h_v2s = length_hist(v2_short)

    sample = []
    for row in v2_short[:20]:
        sample.append(
            {
                "kind": row.get("kind"),
                "n_words": len((row.get("target") or "").split()),
                "input": (row.get("input") or "")[:240],
                "target": (row.get("target") or "")[:240],
            }
        )
    if len(sample) < 20:
        extra = [r for r in v2_short[20:] if r.get("kind") == "question"]
        for row in extra:
            if len(sample) >= 20:
                break
            sample.append(
                {
                    "kind": row.get("kind"),
                    "n_words": len((row.get("target") or "").split()),
                    "input": (row.get("input") or "")[:240],
                    "target": (row.get("target") or "")[:240],
                }
            )

    # stratified sample: 4 of each kind
    by_kind: dict[str, list] = {}
    for row in v2_short:
        by_kind.setdefault(row.get("kind") or "?", []).append(row)
    stratified = []
    for kind in ("question", "instruction", "ack", "slack_commit", "self_correction"):
        for row in by_kind.get(kind, [])[:4]:
            stratified.append(
                {
                    "kind": kind,
                    "n_words": len((row.get("target") or "").split()),
                    "input": (row.get("input") or "")[:240],
                    "target": (row.get("target") or "")[:240],
                }
            )
    if len(stratified) >= 16:
        sample = stratified[:20]

    filter_stats = json.loads(FILTER_STATS.read_text()) if FILTER_STATS.exists() else {}
    delta_pp = {k: h_v2["pct"][k] - h_real["pct"][k] for k in BIN_NAMES}

    results = {
        "real": stats_real,
        "synth_v1_all": stats_v1,
        "synth_v1_error_model": stats_v1_em,
        "synth_v1_tts_asr": stats_v1_tts,
        "synth_v2_all": stats_v2,
        "synth_v2_short": stats_v2s,
        "hist": {"real": h_real, "v1": h_v1, "v2": h_v2, "v2_short": h_v2s},
        "delta_pp": delta_pp,
        "short_sample": sample,
        "pairs_path": str(PAIRS_PATH),
        "n_real": len(real_rows),
        "n_v1": len(v1),
        "n_v2": len(v2),
        "n_v2_short": len(v2_short),
        "n_v2_em": len(v2_em),
        "n_v2_tts": len(v2_tts),
        "filter_stats": filter_stats,
        "classifier": (classify_row("CUDA kernel", "cuda colonel").get("classifier") or ("sibling" if sibling_fn else "local_errors.py")),
    }
    (RESEARCH_ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    md = []
    md.append("# Synthetic correction corpus v2")
    md.append("")
    md.append("Date: 2026-09-17. Machine: gpubox GPU 0 (shared, <10 GB; GPU 1 off limits). Agent: grok-synth-v2.")
    md.append("")
    md.append(
        "v1 plus the missing short-utterance half. Real Wispr pairs "
        "(`/data/phonon_personal/wispr_20260915/corrector_pairs_v0.jsonl`) are the length yardstick only. "
        "LFM2.5-1.2B-Instruct generated 2-25 word spoken utterances grounded in the cleaned v1 lexicon "
        "and user-authored topics, then the v1 calibrated corruption and target normalize were applied unchanged. "
        "Long v1 error-model rows were subsampled; every TTS-ASR row was kept."
    )
    md.append("")
    md.append("## Length histogram")
    md.append("")
    md.extend(hist_table({"real_wispr": h_real, "synth_v1": h_v1, "synth_v2": h_v2, "synth_v2_short": h_v2s}))
    md.append("")
    md.append(
        "Delta v2 minus real (percentage points): "
        + ", ".join(f"{k} {delta_pp[k]:+.1f}" for k in BIN_NAMES)
        + "."
    )
    md.append(
        f"Real median {h_real['median']:.0f} words (p10 {h_real['p10']:.0f}, p90 {h_real['p90']:.0f}); "
        f"v1 median {h_v1['median']:.0f}; v2 median {h_v2['median']:.0f}; "
        f"v2 short n={h_v2s['n']} median {h_v2s['median']:.0f}."
    )
    md.append(
        "v1 has no 1-5 or 6-10 rows (median 52, p10 35). Real 81+ is 21.7% with p90 139; "
        "v1 p90 is 70 so only a few hundred 81+ rows exist to keep. Short generation cannot fill that tail."
    )
    md.append("")
    md.append("Fair WER is whisper_normalizer EnglishTextNormalizer then jiwer. Class rates are ops / reference tokens.")
    md.append("")
    md.append(
        "| set | n | WER mean | WER median | WER p90 | len mean | len median | len p90 | ENTITY | FUNCTION | ORTHOGRAPHY | NEAR_MISS | DROP | INSERT |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in (stats_real, stats_v1, stats_v2s, stats_v2):
        md.append(table_row(s))
    md.append("")
    md.append(
        f"WER mix: real zero={stats_real['frac_wer_zero']:.3f} le0.05={stats_real['frac_wer_le05']:.3f}; "
        f"v2_all zero={stats_v2['frac_wer_zero']:.3f} le0.05={stats_v2['frac_wer_le05']:.3f}; "
        f"v2_short zero={stats_v2s['frac_wer_zero']:.3f} le0.05={stats_v2s['frac_wer_le05']:.3f}."
    )
    md.append(f"Classifier: {results['classifier']}.")
    md.append(f"Pairs: `{PAIRS_PATH}`.")
    pack = filter_stats.get("pack") or filter_stats
    md.append(
        f"Packed n={filter_stats.get('n_pairs', len(v2))} "
        f"v2_short={filter_stats.get('n_v2_short', len(v2_short))} "
        f"v1_kept={filter_stats.get('n_v1')} tts={filter_stats.get('n_tts')} "
        f"kind_counts={filter_stats.get('kind_counts')}."
    )
    md.append("")
    md.append("## Short pair sample")
    md.append("")
    for i, row in enumerate(sample, 1):
        md.append(f"**{i}.** kind={row.get('kind')} n={row.get('n_words')}")
        md.append(f"- raw: {row['input']}")
        md.append(f"- target: {row['target']}")
        md.append("")
    md.append("## What v2 still cannot reproduce")
    md.append("")
    md.append(
        f"v2 all mean fair WER {stats_v2['wer_mean']:.4f} vs real {stats_real['wer_mean']:.4f} "
        f"(v1 {stats_v1['wer_mean']:.4f}); short slice {stats_v2s['wer_mean']:.4f}."
    )
    md.append(
        f"Length: real median {h_real['median']:.0f} vs v1 {h_v1['median']:.0f} vs v2 {h_v2['median']:.0f}. "
        f"Short bins are the point of this recut; 81+ remains a v1 hole ({h_v2['pct']['81+']:.1f}% vs real {h_real['pct']['81+']:.1f}%)."
    )
    md.append(
        "Corruption is still the v1 per-word rate model, not Elliot's microphone. "
        "Short questions are now in the targets so a refiner can copy them instead of answering them; "
        "the generator can still drop a few that sound like answers, and 1.2B prose is not Elliot."
    )
    (RESEARCH_ROOT / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:40]))
    print(f"wrote {RESEARCH_ROOT / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
