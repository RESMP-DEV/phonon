#!/usr/bin/env python3
"""Score an Option 0 sweep using the existing metrics and fair normalization."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jiwer  # noqa: E402
from whisper_normalizer.english import EnglishTextNormalizer  # noqa: E402

from gates import FIVE_GATES, GATE_NAMES, gate_summary, load_gate  # noqa: E402
from phonon.metrics import (  # noqa: E402
    compute_wer_cer,
    compute_word_wer_cer,
    normalize_word_token,
    technical_term_error_rate,
)

LOCAL_BEST = (
    ROOT / "runs/finetune/parakeet_v3_a0p4375_x_jointout_lr1em7_interp_fine_save_20260527"
    / "a0p5625/checkpoint.nemo"
)
# Parameter counts here are nominal model-name sizes, not measured tensor counts.
CANDIDATES = (
    ("parakeet-tdt-0.6b-v2", "nvidia/parakeet-tdt-0.6b-v2", 600_000_000),
    ("parakeet-tdt-0.6b-v3", "nvidia/parakeet-tdt-0.6b-v3", 600_000_000),
    ("parakeet-unified-en-0.6b", "nvidia/parakeet-unified-en-0.6b", 600_000_000),
    ("parakeet_v3_localbest_a0p5625", str(LOCAL_BEST), 600_000_000),
    ("Qwen3-ASR-0.6B-hf", "Qwen/Qwen3-ASR-0.6B-hf", 600_000_000),
    ("Qwen3-ASR-1.7B-hf", "Qwen/Qwen3-ASR-1.7B-hf", 1_700_000_000),
    ("cohere-transcribe-03-2026", "CohereLabs/cohere-transcribe-03-2026", 2_000_000_000),
    ("granite-speech-4.1-2b", "ibm-granite/granite-speech-4.1-2b", 2_000_000_000),
    ("granite-speech-4.1-2b-plus", "ibm-granite/granite-speech-4.1-2b-plus", 2_000_000_000),
    ("granite-speech-5.0-470m-turboctc",
     "ibm-granite/granite-speech-5.0-470m-turboctc", 470_000_000),
)
NORMALIZE = EnglishTextNormalizer()
PERSONAL_GATES = ("wispr_holdout120", "wispr_edit25", "personal_cuda")


def strict_term_tokens(text: str) -> list[str]:
    """Keep case and code punctuation; ignore sentence-final periods on word tokens."""
    return [normalize_word_token(token)
            for token in re.findall(r"[\w./+#*=-]+|[^\w\s]", text, flags=re.UNICODE)]


def strict_technical_term_error_rate(
    term_lists: list[list[str]], hypotheses: list[str],
) -> dict[str, float | int | None]:
    total = missed = 0
    for terms, hyp in zip(term_lists, hypotheses, strict=True):
        hyp_tokens = strict_term_tokens(hyp)
        for term in terms:
            tokens = strict_term_tokens(term)
            if not tokens:
                continue
            total += 1
            if not any(hyp_tokens[start:start + len(tokens)] == tokens
                       for start in range(len(hyp_tokens) - len(tokens) + 1)):
                missed += 1
    return {
        "strict_technical_terms": total,
        "strict_technical_term_misses": missed,
        "strict_technical_term_error_rate": missed / total if total else None,
    }


def term_metrics(term_lists: list[list[str]], hyps: list[str]) -> dict[str, Any]:
    metrics = technical_term_error_rate(term_lists, hyps)
    # An unannotated gate has no term measurement, rather than a perfect score.
    if not metrics["technical_terms"]:
        metrics["technical_term_error_rate"] = None
    return {**metrics, **strict_technical_term_error_rate(term_lists, hyps)}


def text_metrics(refs: list[str], hyps: list[str]) -> dict[str, float]:
    strict = compute_wer_cer(refs, hyps)
    word = compute_word_wer_cer(refs, hyps)
    return {
        "fair_wer": jiwer.wer([NORMALIZE(ref) for ref in refs], [NORMALIZE(hyp) for hyp in hyps]),
        "strict_lowercase_wer": strict["wer"],
        "strict_lowercase_cer": strict["cer"],
        "project_word_wer": word["wer"],
        "project_word_cer": word["cer"],
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError(f"{path}:{line_no}: expected an object with a nonempty string id")
            if not isinstance(row.get("hypothesis"), str):
                raise ValueError(f"{path}:{line_no}: hypothesis must be a string")
            rows.append(row)
    return rows


def score_prediction(
    gate: str, rows: list[dict[str, Any]], path: Path, meta: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = read_jsonl(path)
    expected = {str(row["id"]): row for row in rows}
    if len(expected) != len(rows):
        raise ValueError(f"gate {gate} has duplicate reference ids")
    counts = Counter(row["id"] for row in predictions)
    duplicates = sorted(row_id for row_id, count in counts.items() if count > 1)
    extra = sorted(set(counts) - set(expected))
    missing = sorted(set(expected) - set(counts))
    metadata_errors = []
    if meta.get("complete") is not True:
        metadata_errors.append("sidecar does not mark the transcription complete")
    if meta.get("gate") != gate:
        metadata_errors.append("sidecar gate does not match prediction gate")
    if meta.get("clips") != len(predictions):
        metadata_errors.append("sidecar clip count does not match prediction rows")
    recorded_ids = meta.get("prediction_ids")
    if recorded_ids is not None and (
        not isinstance(recorded_ids, list)
        or Counter(recorded_ids) != counts
    ):
        metadata_errors.append("sidecar prediction ids do not match prediction rows")
    recorded_sha256 = meta.get("prediction_sha256")
    if recorded_sha256 is not None and recorded_sha256 != hashlib.sha256(path.read_bytes()).hexdigest():
        metadata_errors.append("sidecar prediction SHA256 does not match prediction file")
    if recorded_ids is None and recorded_sha256 is None:
        metadata_errors.append("sidecar has no prediction ids or SHA256 to validate identity")
    metrics: dict[str, Any] = {
        "prediction_file": str(path),
        "gate": gate,
        "expected_clips": len(rows),
        "prediction_rows": len(predictions),
        "missing_prediction_ids": missing,
        "extraneous_prediction_ids": extra,
        "duplicate_prediction_ids": duplicates,
        "metadata": meta,
        "metadata_valid": not metadata_errors,
        "metadata_errors": metadata_errors,
        "realtime_factor": None,
        "peak_vram_gb": None,
        "complete": False,
    }
    if duplicates or extra:
        metrics.update(status="invalid", clip_count=0,
                       error="duplicate or extraneous prediction ids; file excluded from scoring")
        return metrics, []
    if meta.get("family") == "transformers-qwen3asr" and meta.get("language_mode") != "auto":
        metrics.update(status="invalid", clip_count=0,
                       error="superseded English-forced Qwen configuration; automatic detection required")
        return metrics, []
    if (str(meta.get("model", "")).endswith("granite-speech-4.1-2b-plus")
            and meta.get("prompt_policy") != "granite_plus_model_card_asr"):
        metrics.update(status="invalid", clip_count=0,
                       error="superseded Granite-plus prompt; model-card ASR prompt required")
        return metrics, []
    if any(30 < float(row.get("audio_seconds", 0)) < 31 and row.get("chunks", 1) > 1
           for row in predictions):
        metrics.update(status="invalid", clip_count=0,
                       error="obsolete millisecond-tail chunking; prediction must be rerun")
        return metrics, []
    by_id = {row["id"]: row for row in predictions}
    matched = [row for row in rows if row["id"] in by_id]
    metrics["clip_count"] = len(matched)
    metrics["complete"] = bool(rows) and not missing and not metadata_errors
    metrics["status"] = "complete" if metrics["complete"] else "partial" if matched else "empty"
    if matched and not metadata_errors:
        metrics["realtime_factor"] = meta.get("realtime_factor")
        metrics["peak_vram_gb"] = meta["peak_vram_bytes"] / 1e9 if meta.get("peak_vram_bytes") else None
    if not meta:
        metrics["metadata_missing"] = True
    if not matched:
        return metrics, []
    refs = [row["reference"] for row in matched]
    hyps = [by_id[row["id"]]["hypothesis"] for row in matched]
    metrics.update(text_metrics(refs, hyps))
    metrics.update(term_metrics([row.get("technical_terms") or [] for row in matched], hyps))
    if gate == "aqua_new_holdout":
        metrics["reference_field"] = "verbatim_text"
        insert_pairs = [(row.get("insert_text"), hyp) for row, hyp in zip(matched, hyps, strict=True)
                        if (row.get("insert_text") or "").strip()]
        metrics["insert_clip_count"] = len(insert_pairs)
        if insert_pairs:
            metrics["insert_metrics"] = text_metrics(
                [ref for ref, _ in insert_pairs], [hyp for _, hyp in insert_pairs]
            )
    diffs = []
    for row, ref, hyp in zip(matched, refs, hyps, strict=True):
        prediction = by_id[row["id"]]
        diff = {
            "gate": gate, "id": row["id"], "reference": ref, "hypothesis": hyp,
            "normalized_reference": NORMALIZE(ref), "normalized_hypothesis": NORMALIZE(hyp),
            **text_metrics([ref], [hyp]),
            "technical_terms": row.get("technical_terms") or [],
            "audio_seconds": prediction.get("audio_seconds"),
            "elapsed_seconds": prediction.get("elapsed_seconds"),
        }
        diff.update({key: value for key, value in term_metrics(
            [row.get("technical_terms") or []], [hyp]).items() if key != "technical_terms"})
        if gate == "aqua_new_holdout":
            diff["insert_text"] = row.get("insert_text", "")
        diffs.append(diff)
    return metrics, diffs


def identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def model_record(tag: str, model: str, nominal: int | None) -> dict[str, Any]:
    return {"tag": tag, "model": model, "nominal_parameter_count": nominal, "gates": {}}


def summarize_model(record: dict[str, Any]) -> None:
    gates = record["gates"]
    complete = all(gates.get(gate, {}).get("complete") for gate in FIVE_GATES)
    record["five_gate_mean_fair_wer"] = (
        sum(gates[gate]["fair_wer"] for gate in FIVE_GATES) / len(FIVE_GATES) if complete else None
    )
    record["five_gate_mean_project_word_wer"] = (
        sum(gates[gate]["project_word_wer"] for gate in FIVE_GATES) / len(FIVE_GATES) if complete else None
    )
    elapsed = 0.0
    audio = 0.0
    peak = []
    parameter_counts = set()
    timed_gates = []
    for gate, result in gates.items():
        if (not result.get("metadata_valid") or not result.get("clip_count")
                or result.get("status") == "invalid"):
            continue
        meta = result.get("metadata", {})
        if meta.get("parameter_count"):
            parameter_counts.add(int(meta["parameter_count"]))
        if meta.get("peak_vram_bytes") is not None:
            peak.append(meta["peak_vram_bytes"])
        if meta.get("audio_seconds", 0) > 0 and meta.get("elapsed_seconds") is not None:
            elapsed += meta["elapsed_seconds"]
            audio += meta["audio_seconds"]
            timed_gates.append(gate)
    record["measured_parameter_counts"] = sorted(parameter_counts)
    record["parameter_count"] = next(iter(parameter_counts)) if len(parameter_counts) == 1 else None
    record["aggregate_audio_seconds"] = audio
    record["aggregate_elapsed_seconds"] = elapsed
    record["realtime_factor"] = elapsed / audio if audio else None
    record["timed_gates"] = timed_gates
    record["peak_vram_gb"] = max(peak) / 1e9 if peak else None


def fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "-"


def gate_cell(record: dict[str, Any], gate: str, metric: str = "fair_wer") -> str:
    result = record["gates"].get(gate, {})
    value = fmt(result.get(metric))
    if result.get("status") == "invalid":
        return "invalid"
    if result and not result.get("complete"):
        return f"{value} ({result.get('clip_count', 0)}/{result.get('expected_clips', '?')}, partial)"
    return value


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "Option 0 base-model sweep. WER values are fractions; lower is better.", "",
        "Fair WER applies Whisper EnglishTextNormalizer to both reference and hypothesis, "
        "then corpus-level jiwer WER. Aqua uses verbatim_text; insert-text scores and strict "
        "lowercase/project word WER are in results.json. No text repair is applied.", "",
        "Parameter counts are measured from the loaded models, including their speech encoders.", "",
    ]
    main_gates = [gate for gate in GATE_NAMES if gate not in PERSONAL_GATES]
    headers = ["Model", "Params", *main_gates, "Five-gate mean fair WER", "hard77 term error",
               "hard77 strict term error",
               "Realtime factor", "Peak VRAM GB"]
    lines += ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for record in report["models"]:
        measured = record.get("parameter_count")
        params = measured or record.get("nominal_parameter_count")
        param_cell = f"{params / 1e9:.3f}B" + ("" if measured else " nominal") if params else "-"
        cells = [record["tag"], param_cell]
        cells += [gate_cell(record, gate) for gate in main_gates]
        cells += [fmt(record["five_gate_mean_fair_wer"]),
                  gate_cell(record, "hard77", "technical_term_error_rate"),
                  gate_cell(record, "hard77", "strict_technical_term_error_rate"),
                  fmt(record["realtime_factor"]), fmt(record["peak_vram_gb"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "The five-gate mean weights aqua_new_holdout, course91, novel180, hard77 and "
              "uncertain48 equally and is withheld until all five gates have complete predictions "
              "and metadata. A dash means "
              "no measured result; see failures.md for attempted candidates.", "",
              "Timing is descriptive single-pass inference including preprocessing and cold inference, "
              "excluding model loading. Realtime factor is summed inference seconds divided by summed "
              "audio seconds across each model's measured gates, including finalized subsets with "
              "verified metadata. Incomplete or stale transcription metadata supplies no timing. "
              "Different gate coverage or batch sizes "
              "prevent a controlled speed comparison. Peak VRAM is the largest allocated CUDA memory "
              "reported by torch, including model loading, in decimal GB; non-torch allocations "
              "are excluded. Full timing, "
              "batch sizes, versions and coverage are retained in results.json.", ""]
    personal = [gate for gate in PERSONAL_GATES if gate in GATE_NAMES]
    if personal:
        lines += ["Personal gates are eval-only. Strict WER lowercases and collapses whitespace "
                  "but preserves punctuation and number formatting, using the project's existing "
                  "compute_wer_cer metric. Wispr references are formatted dictations or owner edits; "
                  "they are not independent verbatim human transcripts.", ""]
        headers = ["Model", *[f"{gate} {kind} WER" for gate in personal
                              for kind in ("fair", "strict")]]
        lines += ["| " + " | ".join(headers) + " |",
                  "| " + " | ".join(["---"] * len(headers)) + " |"]
        for record in report["models"]:
            cells = [record["tag"], *[gate_cell(record, gate, metric) for gate in personal
                     for metric in ("fair_wer", "strict_lowercase_wer")]]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    annotated = [gate for gate in GATE_NAMES if any(
        record["gates"].get(gate, {}).get("technical_terms", 0) for record in report["models"])]
    if annotated:
        lines += ["Term error is missed annotations divided by all annotations. The existing "
                  "lenient metric accepts case-insensitive substrings. Strict term error requires "
                  "an exact case-sensitive contiguous token sequence. Tokens preserve developer "
                  "punctuation (. / + # * = - and underscores); other punctuation forms separate "
                  "tokens, and sentence-final periods on word tokens are ignored. Thus CUDA "
                  "matches CUDA, but not cuda or CUDAs; torch does not match torch.compile, and "
                  "API does not match APIs. Duplicate annotations retain their existing weight. "
                  "Gates without annotations have no term score.", ""]
        headers = ["Model", *[f"{gate} {kind} term error" for gate in annotated
                              for kind in ("lenient", "strict")]]
        lines += ["| " + " | ".join(headers) + " |",
                  "| " + " | ".join(["---"] * len(headers)) + " |"]
        for record in report["models"]:
            cells = [record["tag"], *[gate_cell(record, gate, metric) for gate in annotated
                     for metric in ("technical_term_error_rate", "strict_technical_term_error_rate")]]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if any(result.get("status") == "partial" for record in report["models"]
           for result in record["gates"].values()):
        lines += ["Partial scores are not comparable to full-gate scores.", ""]
    aqua = report["gate_availability"].get("aqua_new_holdout", {})
    if aqua.get("labeled_rows") == 19:
        lines += ["The canonical Aqua split currently contains 19 labeled clips, versus the 17 "
                  "expected in the request; this sweep scores all 19 and does not select a subset.", ""]
    hidden = report["gate_availability"].get("yt_hidden120", {})
    if hidden.get("rows") == 120 and hidden.get("dropped_unlabeled") == 120:
        lines += ["All 120 hidden YouTube rows lack eligible human or consensus labels and are "
                  "excluded from scoring. Hidden labels are never used for training.", ""]
    lines += ["Every family uses the same non-overlapping, nominal 30-second audio chunks. "
              "A final tail shorter than one second is merged into the preceding chunk (maximum "
              "31 seconds), avoiding separate decoding of millisecond WAV padding. Long clips "
              "retain all audio, and chunk transcripts are joined before scoring the original clip.", ""]
    lines += ["Qwen uses its model-card default automatic language detection. Cohere uses the "
              "documented English processor setting; no per-clip language or vocabulary is supplied "
              "from labels. NeMo and Granite receive no explicit language token.", "",
              "The two Granite 4.1 variants use their respective model-card prompts: the base model's "
              "punctuation/capitalization prompt and the plus model's plain ASR prompt with its "
              "documented system message. Granite 5.0 TurboCTC uses its "
              "[model-card CTC decoder](https://huggingface.co/ibm-granite/"
              "granite-speech-5.0-470m-turboctc) without a text prompt.", "",
              "Reference-quality limitation: hard77 includes Chinese, and some uncertain48 labels "
              "are edited drafts based on Parakeet outputs. For example, the curator note for "
              "youtube:rrTts5x8JSY:seg00027 describes rewriting Hindi/English speech using Parakeet "
              "drafts. This can favor Parakeet and penalize faithful multilingual transcription. "
              "These are the requested existing-gate scores, with labels unchanged.", ""]
    summaries = []
    for gate, summary in report["gate_availability"].items():
        if summary.get("error"):
            summaries.append(f"{gate}: unavailable")
        else:
            summaries.append(
                f"{gate}: {summary.get('found', 0)}/{summary.get('labeled_rows', 0)} audio present, "
                f"{summary.get('dropped_unlabeled', 0)} unlabeled dropped"
            )
    lines.append("Gate availability: " + "; ".join(summaries) + ". Missing IDs are in results.json.")
    if report["errors"]:
        lines += ["", "Scoring errors: " + "; ".join(report["errors"])]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, default=ROOT / "runs/reports/option0_sweep_20260915/preds")
    args = parser.parse_args()
    pred_dir = args.pred_dir.resolve()
    report_dir = pred_dir.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    diff_dir = report_dir / "diffs"
    diff_dir.mkdir(exist_ok=True)
    records = {tag: model_record(tag, model, nominal) for tag, model, nominal in CANDIDATES}
    aliases = {}
    for tag, model, _ in CANDIDATES:
        for alias in (tag, model, model.rsplit("/", 1)[-1]):
            aliases[identity(alias)] = tag
    errors = []
    availability = {}
    all_diffs: dict[str, list[dict[str, Any]]] = {}
    for gate in GATE_NAMES:
        try:
            rows = load_gate(gate, include_audio=False)
            availability[gate] = gate_summary(gate)
        except Exception as exc:
            message = f"{gate}: {type(exc).__name__}: {exc}"
            errors.append(message)
            availability[gate] = {"error": message}
            continue
        for path in sorted((pred_dir / gate).glob("*.jsonl")):
            meta = {}
            try:
                meta_path = Path(str(path) + ".meta.json")
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                tag = aliases.get(identity(str(meta.get("model", ""))),
                                  aliases.get(identity(path.stem), path.stem))
                if tag not in records:
                    records[tag] = model_record(tag, meta.get("model", path.stem), None)
                if gate in records[tag]["gates"]:
                    records[tag]["gates"][gate].update(
                        complete=False, status="invalid", clip_count=0,
                        error="multiple prediction files map to this model and gate",
                    )
                    all_diffs[tag] = [diff for diff in all_diffs.get(tag, []) if diff["gate"] != gate]
                    raise ValueError(f"multiple prediction files map to {tag}/{gate}")
                metrics, diffs = score_prediction(gate, rows, path, meta)
                records[tag]["gates"][gate] = metrics
                all_diffs.setdefault(tag, []).extend(diffs)
                if metrics.get("error"):
                    errors.append(f"{path}: {metrics['error']}")
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    for tag, record in records.items():
        summarize_model(record)
        diffs = sorted(all_diffs.get(tag, []), key=lambda row: (-row["fair_wer"], row["gate"], row["id"]))
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
        with (diff_dir / f"{safe_tag}.jsonl").open("w") as dest:
            for row in diffs:
                dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        with (diff_dir / f"{safe_tag}.txt").open("w") as dest:
            for row in diffs:
                dest.write(f"{row['gate']} | {row['id']} | fair WER {row['fair_wer']:.4f}\n")
                dest.write(f"REF: {row['reference']}\nHYP: {row['hypothesis']}\n")
                if row['technical_terms']:
                    dest.write("TERMS: " + ", ".join(row['technical_terms']) + "\n")
                dest.write("\n")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prediction_directory": str(pred_dir),
        "normalizer": "whisper_normalizer.english.EnglishTextNormalizer",
        "metric_aggregation": "corpus WER within gates; unweighted five-gate mean",
        "term_metrics": {
            "lenient": "project technical_term_error_rate: case-insensitive substring presence",
            "strict": "case-sensitive contiguous strict_term_tokens sequence presence; "
                      "preserves developer punctuation, ignores sentence-final word periods",
            "denominator": "annotation count; repeated annotations retain their weight",
            "unannotated": "null",
        },
        "versions": {name: importlib.metadata.version(name) for name in ("jiwer", "whisper-normalizer")},
        "five_gates": list(FIVE_GATES), "gate_availability": availability,
        "models": list(records.values()), "errors": errors,
    }
    (report_dir / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    result_md = markdown(report)
    (report_dir / "results.md").write_text(result_md)
    print(result_md)


if __name__ == "__main__":
    main()
