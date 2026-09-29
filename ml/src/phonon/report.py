from __future__ import annotations

import glob
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any

from jiwer import wer

from phonon.metrics import normalize_for_wer, normalize_words_for_wer
from phonon.schema import now_iso


FORMULAS = {
    "wer": "(substitutions + deletions + insertions) / reference_words",
    "word_wer": "WER after tokenizing ASR words and ignoring standalone punctuation/casing",
    "cer": "(character_substitutions + character_deletions + character_insertions) / reference_chars",
    "technical_term_error_rate": "missed_annotated_technical_terms / annotated_technical_terms",
    "casing_error_rate": "cased_reference_tokens_not_reproduced_exactly / cased_reference_tokens",
    "punctuation_error_rate": "unmatched_reference_punctuation_marks / reference_punctuation_marks",
    "realtime_factor": "elapsed_seconds / audio_seconds",
    "throughput_x_realtime": "audio_seconds / elapsed_seconds",
    "teacher_disagreement": "mean pairwise WER between teacher transcripts; triage only, not accuracy",
}


def expand_metric_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
        if matches:
            paths.extend(matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    return sorted(set(paths))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values: list[float], pct: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[lower]
    weight = rank - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def short_model_name(metrics: dict[str, Any]) -> str:
    model = str(metrics.get("model") or "")
    adapter = str(metrics.get("adapter") or "")
    output = Path(str(metrics.get("output_jsonl") or "")).stem
    if adapter == "external" and model:
        return model
    if adapter == "openai-compatible" and (
        "api.aqua.sh" in str(metrics.get("base_url") or "")
        or "api.aquavoice.com" in str(metrics.get("base_url") or "")
    ):
        return f"aqua_{model.replace('-', '_').replace('.', '_')}"
    if adapter == "openai-compatible" and model:
        return model
    if "canary-qwen-2.5b" in model:
        return "canary_qwen_2_5b"
    if "canary-1b-flash" in model:
        return "canary_1b_flash"
    if "canary-1b-v2" in model:
        return "canary_1b_v2"
    if "canary-180m-flash" in model:
        return "canary_180m_flash"
    if "canary-1b" in model:
        return "canary_1b"
    if "nemotron-speech-streaming-en-0.6b" in model:
        return "nemotron_streaming_0_6b"
    if "cohere-transcribe-03-2026" in model:
        return "cohere_transcribe_03_2026"
    if "parakeet-unified-en-0.6b" in model:
        return "parakeet_unified_0_6b"
    if "parakeet-tdt_ctc-1.1b" in model:
        return "parakeet_tdt_ctc_1_1b"
    if "parakeet-tdt-1.1b" in model:
        return "parakeet_tdt_1_1b"
    if "parakeet-tdt-0.6b-v2" in model:
        return "parakeet_v2"
    if "parakeet-ctc-1.1b" in model:
        return "parakeet_ctc_1_1b"
    if "parakeet-tdt-0.6b-v3" in model and not model.endswith(".nemo"):
        return "parakeet_v3"
    if "mobile-voice" in model or model.endswith(".nemo"):
        return "mobile_voice"
    if "large-v3-turbo" in model:
        return "whisper_turbo"
    return output or f"{adapter}:{Path(model).name or model}"


def row_wer(ref: str, hyp: str) -> float:
    ref_norm = normalize_for_wer(ref)
    hyp_norm = normalize_for_wer(hyp)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    return float(wer(ref_norm, hyp_norm))


def row_word_wer(ref: str, hyp: str) -> float:
    ref_norm = normalize_words_for_wer(ref)
    hyp_norm = normalize_words_for_wer(hyp)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    return float(wer(ref_norm, hyp_norm))


def prediction_stats(metrics: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(metrics.get("output_jsonl") or ""))
    rows = load_jsonl(path)
    insert_wers = [row_wer(row.get("insert_text") or "", row.get("hypothesis") or "") for row in rows]
    insert_word_wers = [
        row_word_wer(row.get("insert_text") or "", row.get("hypothesis") or "") for row in rows
    ]
    verbatim_wers = [
        row_wer(row.get("verbatim_text") or "", row.get("hypothesis") or "") for row in rows
    ]
    durations = [float(row.get("duration_seconds") or 0) for row in rows]
    return {
        "prediction_rows": len(rows),
        "insert_wer_p50": percentile(insert_wers, 0.50),
        "insert_wer_p90": percentile(insert_wers, 0.90),
        "insert_wer_p95": percentile(insert_wers, 0.95),
        "insert_wer_max": max(insert_wers) if insert_wers else None,
        "insert_word_wer_p90": percentile(insert_word_wers, 0.90),
        "verbatim_wer_p50": percentile(verbatim_wers, 0.50),
        "verbatim_wer_p90": percentile(verbatim_wers, 0.90),
        "duration_seconds_p50": percentile(durations, 0.50),
        "duration_seconds_p90": percentile(durations, 0.90),
    }


def load_metric_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        metrics = json.loads(path.read_text())
        metrics["metrics_path"] = str(path)
        metrics["model_label"] = short_model_name(metrics)
        metrics["prediction_stats"] = prediction_stats(metrics)
        summaries.append(metrics)
    return sorted(summaries, key=lambda item: item.get("insert_wer", 99))


def load_queue_summary(path: Path | None, limit: int = 50) -> dict[str, Any]:
    if path is None:
        return {"path": None, "rows": 0, "top_items": []}
    rows = load_jsonl(path)
    scored = [row for row in rows if row.get("review_score") is not None]
    scored.sort(key=lambda row: float(row.get("review_score") or 0), reverse=True)
    disagreements = [float(row.get("teacher_disagreement") or 0) for row in scored]
    review_scores = [float(row.get("review_score") or 0) for row in scored]
    technical_hits = [int(row.get("technical_term_hits") or 0) for row in scored]
    command_hits = [int(row.get("command_path_package_hits") or 0) for row in scored]
    return {
        "path": str(path),
        "rows": len(rows),
        "scored_rows": len(scored),
        "review_score_mean": statistics.fmean(review_scores) if review_scores else None,
        "teacher_disagreement_mean": statistics.fmean(disagreements) if disagreements else None,
        "teacher_disagreement_p90": percentile(disagreements, 0.90),
        "technical_term_hits_total": sum(technical_hits),
        "command_path_package_hits_total": sum(command_hits),
        "top_items": scored[:limit],
    }


def pct(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) * 100.0
    except (TypeError, ValueError):
        return None


def fmt_pct(value: Any) -> str:
    converted = pct(value)
    return "n/a" if converted is None else f"{converted:.2f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def bar_svg(
    rows: list[dict[str, Any]],
    key: str,
    label_key: str,
    title: str,
    lower_is_better: bool = True,
    width: int = 760,
    row_height: int = 34,
) -> str:
    if not rows:
        return "<p>No data.</p>"
    values = [float(row.get(key) or 0) for row in rows]
    max_value = max(values) or 1.0
    height = 56 + len(rows) * row_height
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="0" y="20" class="chart-title">{html.escape(title)}</text>',
    ]
    x0 = 180
    chart_width = width - x0 - 90
    for index, row in enumerate(rows):
        y = 44 + index * row_height
        value = float(row.get(key) or 0)
        bar_width = value / max_value * chart_width
        color = "#2f6f73" if lower_is_better else "#8d5a1f"
        parts.extend(
            [
                f'<text x="0" y="{y + 14}" class="axis-label">'
                f'{html.escape(str(row.get(label_key) or ""))}</text>',
                f'<line x1="{x0}" y1="{y + 9}" x2="{x0 + chart_width}" y2="{y + 9}" '
                'stroke="#e6e6e6" stroke-width="2" />',
                f'<line x1="{x0}" y1="{y + 9}" x2="{x0 + bar_width}" y2="{y + 9}" '
                f'stroke="{color}" stroke-width="4" />',
                f'<circle cx="{x0 + bar_width}" cy="{y + 9}" r="6" fill="{color}" />',
                f'<text x="{x0 + chart_width + 14}" y="{y + 14}" class="value-label">'
                f'{fmt_pct(value) if key.endswith("wer") or "error_rate" in key else fmt_num(value)}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


def scatter_svg(rows: list[dict[str, Any]], width: int = 760, height: int = 360) -> str:
    points = [
        row
        for row in rows
        if row.get("insert_wer") is not None and row.get("throughput_x_realtime") is not None
    ]
    if not points:
        return "<p>No speed/accuracy data.</p>"
    min_x = 0.0
    max_x = max(float(row["insert_wer"]) for row in points) or 1.0
    min_y = 0.0
    max_y = max(float(row["throughput_x_realtime"]) for row in points) or 1.0
    pad_left = 70
    pad_bottom = 54
    chart_w = width - pad_left - 28
    chart_h = height - 46 - pad_bottom
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" xmlns="http://www.w3.org/2000/svg">',
        '<text x="0" y="20" class="chart-title">Speed vs insert WER</text>',
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - 28}" '
        f'y2="{height - pad_bottom}" stroke="#888" />',
        f'<line x1="{pad_left}" y1="38" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#888" />',
        f'<text x="{width / 2 - 70}" y="{height - 12}" class="axis-label">insert WER, lower is better</text>',
        '<text x="4" y="38" class="axis-label">x realtime</text>',
    ]
    for row in points:
        x = pad_left + (float(row["insert_wer"]) - min_x) / (max_x - min_x or 1.0) * chart_w
        y = height - pad_bottom - (float(row["throughput_x_realtime"]) - min_y) / (
            max_y - min_y or 1.0
        ) * chart_h
        label = html.escape(str(row.get("model_label") or "model"))
        parts.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#7b3f88" opacity="0.85" />',
                f'<text x="{x + 12:.1f}" y="{y + 4:.1f}" class="value-label">{label}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


def heatmap_table(items: list[dict[str, Any]], limit: int = 25) -> str:
    if not items:
        return "<p>No teacher-disagreement queue data.</p>"
    top = items[:limit]
    max_score = max(float(item.get("review_score") or 0) for item in top) or 1.0
    rows = []
    for item in top:
        score = float(item.get("review_score") or 0)
        intensity = min(score / max_score, 1.0)
        bg = f"rgba(47, 111, 115, {0.12 + intensity * 0.55:.3f})"
        teachers = item.get("teacher_transcripts") or {}
        rows.append(
            "<tr>"
            f'<td style="background:{bg}">{html.escape(str(item.get("id") or ""))}</td>'
            f"<td>{fmt_num(item.get('clip_start'), 1)}</td>"
            f"<td>{fmt_num(score, 2)}</td>"
            f"<td>{fmt_num(item.get('teacher_disagreement'), 3)}</td>"
            f"<td>{html.escape(str(item.get('technical_term_hits') or 0))}</td>"
            f"<td>{html.escape(str(item.get('command_path_package_hits') or 0))}</td>"
            f"<td>{html.escape(', '.join(sorted(teachers)))}</td>"
            f"<td>{html.escape(str(item.get('audio_path') or ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>clip</th><th>start s</th><th>review score</th>"
        "<th>teacher disagreement</th><th>technical hits</th><th>cmd/path hits</th>"
        "<th>teachers</th><th>audio</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def metrics_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No gold-label eval metrics found.</p>"
    body = []
    for row in rows:
        stats = row.get("prediction_stats") or {}
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('model_label') or ''))}</td>"
            f"<td>{html.escape(str(row.get('dataset') or ''))}</td>"
            f"<td>{html.escape(str(row.get('split') or ''))}</td>"
            f"<td>{fmt_pct(row.get('insert_wer'))}</td>"
            f"<td>{fmt_pct(row.get('insert_word_wer'))}</td>"
            f"<td>{fmt_pct(row.get('verbatim_wer'))}</td>"
            f"<td>{fmt_pct(row.get('insert_cer'))}</td>"
            f"<td>{fmt_pct(row.get('technical_term_error_rate'))}</td>"
            f"<td>{fmt_pct(row.get('insert_casing_error_rate'))}</td>"
            f"<td>{fmt_pct(row.get('insert_punctuation_error_rate'))}</td>"
            f"<td>{fmt_num(row.get('throughput_x_realtime'), 2)}</td>"
            f"<td>{fmt_pct(stats.get('insert_wer_p90'))}</td>"
            f"<td>{fmt_pct(stats.get('insert_word_wer_p90'))}</td>"
            f"<td>{html.escape(str(row.get('utterances') or ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>model</th><th>dataset</th><th>split</th><th>insert WER</th>"
        "<th>insert word WER</th><th>verbatim WER</th><th>insert CER</th><th>term error</th>"
        "<th>case error</th><th>punct error</th><th>x realtime</th><th>row p90 WER</th>"
        "<th>row p90 word WER</th><th>rows</th>"
        "</tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table>"
    )


def formulas_html() -> str:
    rows = [
        f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(formula)}</code></td></tr>"
        for name, formula in FORMULAS.items()
    ]
    return "<table><thead><tr><th>metric</th><th>formula</th></tr></thead><tbody>" + "\n".join(
        rows
    ) + "</tbody></table>"


def render_html(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    queue = summary["teacher_queue"]
    style = """
    body { font-family: system-ui, sans-serif; margin: 32px; color: #202124; }
    h1, h2 { letter-spacing: 0; }
    .note { color: #555; max-width: 900px; }
    .grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 28px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #e6e6e6; padding: 8px; text-align: left; vertical-align: top; }
    th { background: #f7f7f7; }
    code { background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }
    svg { max-width: 100%; height: auto; display: block; }
    .chart-title { font-weight: 700; font-size: 16px; }
    .axis-label { font-size: 12px; fill: #555; }
    .value-label { font-size: 12px; fill: #333; }
    """
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Phonon Metrics Dashboard</title>
<style>{style}</style>
</head>
<body>
<h1>Phonon Metrics Dashboard</h1>
<p class="note">Generated {html.escape(summary["created_at"])}. Gold-label metrics are real accuracy
measurements. Teacher-disagreement metrics are hard-eval triage signals until human labels are locked.</p>
<div class="grid">
<section>
<h2>Gold-label Baselines</h2>
{metrics_table(metrics)}
</section>
<section>
<h2>Insert WER Lollipop</h2>
{bar_svg(metrics, "insert_wer", "model_label", "Insert WER, lower is better")}
</section>
<section>
<h2>Speed Accuracy Map</h2>
{scatter_svg(metrics)}
</section>
<section>
<h2>Technical Term Error</h2>
{bar_svg(metrics, "technical_term_error_rate", "model_label", "Technical term error rate")}
</section>
<section>
<h2>Teacher Disagreement Triage</h2>
<p class="note">Queue: {html.escape(str(queue.get("path") or ""))}. Rows: {queue.get("rows", 0)}.
Mean disagreement: {fmt_num(queue.get("teacher_disagreement_mean"), 3)}.
P90 disagreement: {fmt_num(queue.get("teacher_disagreement_p90"), 3)}.</p>
{heatmap_table(queue.get("top_items") or [])}
</section>
<section>
<h2>Metric Formulas</h2>
{formulas_html()}
</section>
</div>
</body>
</html>
"""


def build_report(
    metric_patterns: list[str],
    queue_file: Path | None,
    out: Path,
    json_out: Path | None = None,
) -> dict[str, Any]:
    metric_paths = expand_metric_paths(metric_patterns)
    metrics = load_metric_summaries(metric_paths)
    queue = load_queue_summary(queue_file)
    summary = {
        "created_at": now_iso(),
        "metrics": metrics,
        "teacher_queue": queue,
        "formulas": FORMULAS,
        "outputs": {
            "html": str(out),
            "json": str(json_out) if json_out else None,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(summary))
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary
