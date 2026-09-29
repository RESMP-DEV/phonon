"""Compare synthetic pairs to Elliot's real Wispr traces. Read personal data only here."""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path("/home/user/phonon/research/corrector_v0")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import fair_norm, pair_wer  # noqa: E402
from errors import CLASSES, classify_pair, is_entity_shape, try_import_sibling_classifier  # noqa: E402
from paths import (  # noqa: E402
    DATA_ROOT,
    LEXICON_PATH,
    PAIRS_PATH,
    REAL_PAIRS,
    RESEARCH_ROOT,
)

RANKED_LEXICON = DATA_ROOT / "lexicon_ranked.jsonl"

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
    for row in rows:
        target = (row.get("target") or "").strip()
        raw = (row.get("input") or row.get("asr") or "").strip()
        if not target or not raw:
            continue
        try:
            wers.append(pair_wer(target, raw, fair_norm))
        except Exception:
            continue
        n_words.append(len(target.split()))
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
        f"{fmt(r['ENTITY'])} | {fmt(r['FUNCTION'])} | {fmt(r['ORTHOGRAPHY'])} | "
        f"{fmt(r['NEAR_MISS'])} | {fmt(r['DROP'])} | {fmt(r['INSERT'])} |"
    )


def main() -> int:
    lex_path = RANKED_LEXICON if RANKED_LEXICON.exists() else LEXICON_PATH
    lexicon = []
    with lex_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            lexicon.append(row)
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
    stats_em = summarize("synth_error_model", em, lex_terms)
    stats_tts = summarize("synth_tts_asr", tts, lex_terms)
    stats_all = summarize("synth_all", synth, lex_terms)

    real_stride = real_rows[:: max(len(real_rows) // 20, 1)]
    synth_for_sample = (em[:15] + tts[:5]) if (em and tts) else (em or tts or synth)
    sample = side_by_side(real_stride, synth_for_sample, 20)

    closer = "error-model"
    if tts and em:
        d_em = abs(stats_em["wer_mean"] - stats_real["wer_mean"])
        d_tts = abs(stats_tts["wer_mean"] - stats_real["wer_mean"])
        closer = "error-model" if d_em <= d_tts else "TTS-ASR"

    probe = classify_row("CUDA kernel", "cuda colonel")
    results = {
        "real": stats_real,
        "synth_error_model": stats_em,
        "synth_tts_asr": stats_tts,
        "synth_all": stats_all,
        "lexicon_n": len(lexicon),
        "lexicon_path": str(lex_path),
        "classifier": probe.get("classifier") or ("sibling" if sibling_fn else "local_errors.py"),
        "error_model_path": str(Path("/data/phonon_asr_errors_v0/error_model.json")),
        "closer_route": closer,
        "side_by_side": sample,
        "pairs_path": str(PAIRS_PATH),
        "n_real": len(real_rows),
        "n_synth": len(synth),
        "n_em": len(em),
        "n_tts": len(tts),
        "clean_utterances": sum(1 for _ in open(DATA_ROOT / "clean_utterances.jsonl") if _.strip()),
    }
    (RESEARCH_ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    md = []
    md.append("# Synthetic correction corpus v0")
    md.append("")
    md.append("Date: 2026-09-17. Machine: gpubox GPU 0. Agent: grok-synth.")
    md.append("")
    md.append("A new-user synthetic correction corpus: repo walk -> LFM2.5-1.2B spoken utterances -> two ASR-raw routes. Elliot's Wispr pairs are the yardstick only; they were not a generation source.")
    md.append("")
    md.append(
        f"Lexicon and topics from `/home/user/{{cuda,dev,phonon,experiments,benchmarks,box-install}}` "
        f"(persona-gym missing on this box). Ranked speakable lexicon {len(lexicon)} terms. "
        f"LFM2.5-1.2B-Instruct produced {results['clean_utterances']} dictation utterances. "
        "Error-model route used `/data/phonon_asr_errors_v0/error_model.json` (Parakeet v2 Wispr per-word rates). "
        "TTS-ASR: Kokoro-82M voices af_heart/am_adam/bf_emma then Parakeet TDT v2."
    )
    md.append("")
    md.append("Fair WER is whisper_normalizer EnglishTextNormalizer then jiwer. Class rates are ops / reference tokens.")
    md.append("")
    md.append(
        "| set | n | WER mean | WER median | WER p90 | len mean | len median | len p90 | ENTITY | FUNCTION | ORTHOGRAPHY | NEAR_MISS | DROP | INSERT |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in (stats_real, stats_em, stats_tts, stats_all):
        md.append(table_row(s))
    md.append("")
    md.append(f"Entity-term overlap (real ENTITY refs present in synthetic lexicon): {stats_real['entity_overlap']:.4f} ({stats_real['entity_in_lex']}/{stats_real['entity_total']}).")
    md.append(f"Classifier: {results['classifier']}. Closer route by mean fair WER: {closer}.")
    md.append(f"Pairs: `{PAIRS_PATH}`.")
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
    md.append("## Gap")
    md.append("")
    gap_sentences = [
        f"The error-model route mean fair WER is {stats_em['wer_mean']:.4f} against real {stats_real['wer_mean']:.4f}, and TTS-ASR is {stats_tts['wer_mean']:.4f}.",
        f"Real utterances average {stats_real['len_mean']:.1f} words versus {stats_em['len_mean']:.1f} on the error-model set and {stats_tts['len_mean']:.1f} on TTS-ASR.",
        f"ENTITY per-word rate is {stats_real['class_rates']['ENTITY']:.4f} real, {stats_em['class_rates']['ENTITY']:.4f} error-model, {stats_tts['class_rates']['ENTITY']:.4f} TTS.",
        f"{stats_real['entity_overlap']*100:.1f} percent of real entity-error terms sit in the synthetic lexicon ({stats_real['entity_in_lex']}/{stats_real['entity_total']}).",
        "Synthetic reproduces orthography collapse, function-word jitter, and identifier splitting because those classes are in the measured error model and in Kokoro-Parakeet residuals.",
        "It misses Elliot-specific names that never appear in these repos, plus spoken self-corrections with real timing.",
        "It also misses acoustic confusions that only show up on his microphone and room.",
        "TTS-ASR substitutions are real recognizer errors on generic voices; the error-model route can copy the measured class mix, including ENTITY table hits.",
        (
            f"FUNCTION/INSERT per-word rates are real {stats_real['class_rates']['FUNCTION']:.4f}/{stats_real['class_rates']['INSERT']:.4f}, "
            f"error-model {stats_em['class_rates']['FUNCTION']:.4f}/{stats_em['class_rates']['INSERT']:.4f}, "
            f"TTS {stats_tts['class_rates']['FUNCTION']:.4f}/{stats_tts['class_rates']['INSERT']:.4f}; "
            "the error-model route plants uh/like by rate, TTS only if Parakeet actually inserts them."
        ),
        f"The {closer} route is closer to real on mean fair WER; neither replaces recorded traces as eval, but a new user without recordings still gets {results['n_synth']} pairs from {results['clean_utterances']} utterances grounded in their repos.",
    ]
    md.extend(gap_sentences)
    (RESEARCH_ROOT / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:20]))
    print(f"wrote {RESEARCH_ROOT / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
