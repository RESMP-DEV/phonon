"""Compare synth_v1 pairs to real Wispr and to synth_v0."""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util as _ilu

from paths import (  # noqa: E402
    FILTER_STATS,
    LEXICON_PATH,
    PAIRS_PATH,
    REAL_PAIRS,
    RESEARCH_ROOT,
    V0_RESULTS,
)
from pollution import has_identifier_dump  # noqa: E402
from corrupt import _load_v0  # noqa: E402

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
    for row in rows:
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


def side_by_side(real: list[dict], synth: list[dict], n: int = 20) -> list[dict]:
    out = []
    for i in range(min(n, len(real), len(synth))):
        r = real[i]
        s = synth[i]
        out.append(
            {
                "real_input": (r.get("asr") or r.get("input") or "")[:240],
                "real_target": (r.get("target") or "")[:240],
                "synth_input": (s.get("input") or "")[:240],
                "synth_target": (s.get("target") or "")[:240],
                "synth_route": s.get("route"),
            }
        )
    return out


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


def v0_as_summary(blob: dict, name: str) -> dict:
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
        "entity_overlap": blob.get("entity_overlap") or 0.0,
        "entity_total": blob.get("entity_total") or 0,
        "entity_in_lex": blob.get("entity_in_lex") or 0,
        "top_entity_terms": blob.get("top_entity_terms") or [],
    }


def filler_extras(rows: list[dict]) -> dict[str, float]:
    import re

    word_re = re.compile(r"[A-Za-z0-9']+")
    extra = Counter()
    n_ref = 0

    def bigrams(ws):
        return sum(1 for i in range(len(ws) - 1) if ws[i] == "you" and ws[i + 1] == "know")

    for row in rows:
        raw = (row.get("input") or "").lower()
        tgt = (row.get("target") or "").lower()
        rw, tw = word_re.findall(raw), word_re.findall(tgt)
        n_ref += len(tw)
        rc, tc = Counter(rw), Counter(tw)
        for f in ("um", "uh", "like"):
            extra[f] += max(0, rc[f] - tc[f])
        extra["you know"] += max(0, bigrams(rw) - bigrams(tw))
    denom = max(n_ref, 1)
    return {k: extra[k] / denom for k in ("um", "uh", "like", "you know")} | {
        "n_ref_words": float(n_ref),
        **{f"count_{k}": float(extra[k]) for k in extra},
    }


def main() -> int:
    lexicon = load_jsonl(LEXICON_PATH)
    lex_terms = {r["term"] for r in lexicon} | {r["term"].lower() for r in lexicon}

    real_raw = load_jsonl(REAL_PAIRS)
    real_rows = []
    for row in real_raw:
        target = (row.get("target") or row.get("kept") or "").strip()
        raw = (row.get("asr") or row.get("input") or row.get("raw") or "").strip()
        if not target or not raw:
            continue
        real_rows.append({"input": raw, "asr": raw, "target": target, "source": row.get("source") or "wispr"})

    synth = load_jsonl(PAIRS_PATH)
    em = [r for r in synth if r.get("route") == "error_model"]
    tts = [r for r in synth if r.get("route") == "tts_asr"]

    stats_real = summarize("real_wispr", real_rows, lex_terms)
    stats_em = summarize("synth_v1_error_model", em, lex_terms)
    stats_tts = summarize("synth_v1_tts_asr", tts, lex_terms)
    stats_all = summarize("synth_v1_all", synth, lex_terms)

    v0 = json.loads(V0_RESULTS.read_text()) if V0_RESULTS.exists() else {}
    stats_v0_em = v0_as_summary(v0.get("synth_error_model") or {}, "synth_v0_error_model")
    stats_v0_all = v0_as_summary(v0.get("synth_all") or {}, "synth_v0_all")
    stats_v0_tts = v0_as_summary(v0.get("synth_tts_asr") or {}, "synth_v0_tts_asr")

    real_stride = real_rows[:: max(len(real_rows) // 20, 1)]
    synth_for_sample = (em[:15] + tts[:5]) if (em and tts) else (em or tts or synth)
    sample = side_by_side(real_stride, synth_for_sample, 20)

    filter_stats = json.loads(FILTER_STATS.read_text()) if FILTER_STATS.exists() else {}
    fillers_real = filler_extras(real_rows)
    fillers_em = filler_extras(em)
    fillers_all = filler_extras(synth)

    probe = classify_row("CUDA kernel", "cuda colonel")
    closer = "error-model"
    if tts and em:
        d_em = abs(stats_em["wer_mean"] - stats_real["wer_mean"])
        d_tts = abs(stats_tts["wer_mean"] - stats_real["wer_mean"])
        closer = "error-model" if d_em <= d_tts else "TTS-ASR"

    results = {
        "real": stats_real,
        "synth_v0_error_model": stats_v0_em,
        "synth_v0_tts_asr": stats_v0_tts,
        "synth_v0_all": stats_v0_all,
        "synth_v1_error_model": stats_em,
        "synth_v1_tts_asr": stats_tts,
        "synth_v1_all": stats_all,
        "lexicon_n": len(lexicon),
        "lexicon_path": str(LEXICON_PATH),
        "classifier": probe.get("classifier") or ("sibling" if sibling_fn else "local_errors.py"),
        "closer_route": closer,
        "side_by_side": sample,
        "pairs_path": str(PAIRS_PATH),
        "n_real": len(real_rows),
        "n_synth": len(synth),
        "n_em": len(em),
        "n_tts": len(tts),
        "filter_stats": filter_stats,
        "fillers_real": fillers_real,
        "fillers_v1_em": fillers_em,
        "fillers_v1_all": fillers_all,
    }
    (RESEARCH_ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    lex = filter_stats.get("lexicon") or {}
    emf = filter_stats.get("error_model") or {}
    ttf = filter_stats.get("tts") or {}

    md = []
    md.append("# Synthetic correction corpus v1")
    md.append("")
    md.append("Date: 2026-09-17. Machine: gpubox CPU (GPU 0 shared, GPU 1 off limits). Agent: grok-synth-v1.")
    md.append("")
    md.append(
        "Calibrated recut of synth_v0: same clean utterances, no LLM regen. "
        "Pollution filter on lexicon paths, Elliot-style target normalize, "
        "error-model route recorrupted at real_wispr per-word rates with the real WER mix "
        "(38% fair-WER 0). TTS-ASR rows kept after the same target/pollution filters."
    )
    md.append("")
    md.append("Fair WER is whisper_normalizer EnglishTextNormalizer then jiwer. Class rates are ops / reference tokens.")
    md.append("")
    md.append(
        "| set | n | WER mean | WER median | WER p90 | len mean | len median | len p90 | ENTITY | FUNCTION | ORTHOGRAPHY | NEAR_MISS | DROP | INSERT |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in (stats_real, stats_v0_em, stats_v0_all, stats_em, stats_tts, stats_all):
        md.append(table_row(s))
    md.append("")
    md.append(
        f"Entity-term overlap (real ENTITY refs present in v1 lexicon): "
        f"{stats_real['entity_overlap']:.4f} ({stats_real['entity_in_lex']}/{stats_real['entity_total']}); "
        f"v0 was 0.0507 (140/2764)."
    )
    md.append(
        f"WER mix: real zero={stats_real['frac_wer_zero']:.3f} le0.05={stats_real['frac_wer_le05']:.3f}; "
        f"v1_em zero={stats_em['frac_wer_zero']:.3f} le0.05={stats_em['frac_wer_le05']:.3f}."
    )
    md.append(f"Classifier: {results['classifier']}. Closer v1 route by mean fair WER: {closer}.")
    md.append(f"Pairs: `{PAIRS_PATH}`.")
    md.append("")
    md.append("## Pollution filter")
    md.append("")
    md.append(
        f"Ranked lexicon in {lex.get('ranked_in', 0)} -> keep {lex.get('keep', 0)}, "
        f"weak {lex.get('weak', 0)}, polluted {lex.get('polluted', 0)}, "
        f"lexicon out {lex.get('lexicon_out', 0)} (keep+weak). "
        "Third-party path classes: `curation/context_resources` (scraped transformers/px4 docs), "
        "`cuda-oxide` / `cutile-rs` (NVlabs bindings), `nccl-tests`, `tools/codex`, "
        "`tModLoader`, `spiny` (Mario Kart decomp), `nsmbw-wt`, `kernelbench.com/benchmarks`, "
        "generated `*_pb2` / bindgen. Keep: phonon scripts/src/research/tests/docs/macos/hard_terms, "
        "experiments, box-install, DEVLOG/AGENTS/SPEC/README, scripts dirs, person/machine/cli_tool/product."
    )
    md.append(
        f"Clean utterances in {emf.get('in', 0)} -> kept {emf.get('kept', 0)}; "
        f"drop reasons {emf.get('drop_reason', {})}. "
        f"TTS in {ttf.get('in', 0)} -> kept {ttf.get('kept', 0)}; drop reasons {ttf.get('drop_reason', {})}."
    )
    md.append(
        "Worked examples: PersimmonModel/Blip2VisionConfig/FlightGear from materialized transformers/px4 markdown "
        "(polluted); KOOPA from mariokart `spiny/tests` (polluted); GatherGetCollByteCount from nccl-tests (polluted); "
        "Grok/KernelBench/Gpubox kept from DEVLOG and hard_terms."
    )
    md.append(
        f"Filler extras per ref word, real: like={fillers_real.get('like', 0):.4f} um={fillers_real.get('um', 0):.4f} "
        f"uh={fillers_real.get('uh', 0):.4f} you_know={fillers_real.get('you know', 0):.4f}; "
        f"v1_em: like={fillers_em.get('like', 0):.4f} um={fillers_em.get('um', 0):.4f} "
        f"uh={fillers_em.get('uh', 0):.4f} you_know={fillers_em.get('you know', 0):.4f}."
    )
    md.append("")
    md.append("## Side-by-side sample")
    md.append("")
    for i, row in enumerate(sample, 1):
        md.append(f"**{i}.**")
        md.append(f"- real raw: {row['real_input']}")
        md.append(f"- real target: {row['real_target']}")
        md.append(f"- synth raw ({row['synth_route']}): {row['synth_input']}")
        md.append(f"- synth target: {row['synth_target']}")
        md.append("")
    md.append("## What v1 still cannot reproduce")
    md.append("")
    gap = [
        f"v1 error-model mean fair WER is {stats_em['wer_mean']:.4f} against real {stats_real['wer_mean']:.4f} "
        f"(median {stats_em['wer_median']:.4f} vs {stats_real['wer_median']:.4f}, p90 {stats_em['wer_p90']:.4f} vs {stats_real['wer_p90']:.4f}); "
        f"TTS-ASR remains a frozen Kokoro-Parakeet residual at {stats_tts['wer_mean']:.4f}.",
        f"ENTITY is {stats_em['class_rates'].get('ENTITY', 0):.4f} vs real {stats_real['class_rates'].get('ENTITY', 0):.4f} "
        f"and FUNCTION {stats_em['class_rates'].get('FUNCTION', 0):.4f} vs {stats_real['class_rates'].get('FUNCTION', 0):.4f}; "
        "the recut matches the class mix better than v0 but still plants errors by token budget rather than by Elliot's microphone confusions.",
        f"Entity-term overlap with real spoken errors is {stats_real['entity_overlap']:.4f} "
        f"({stats_real['entity_in_lex']}/{stats_real['entity_total']}); names like Wispr, Tavus, DFlash, and Kimi barely appear in the repo walk, so the lexicon still cannot teach those repairs.",
        "Targets are ASCII-normalized and dump-filtered, but mixed utterances can still mention a third-party identifier when the generating prompt stuffed extra lexicon terms beside a kept one.",
        "v1 has no self-corrections with real timing, no room-mic acoustics, and no owner-specific length tail (real median 34 words, synth still ~50); it is a new-user prior, not a replacement for recorded Wispr pairs.",
    ]
    md.extend(gap)
    (RESEARCH_ROOT / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:28]))
    print(f"wrote {RESEARCH_ROOT / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
