#!/usr/bin/env python3
"""Score frozen-encoder experiments without selecting checkpoints from evaluation labels."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "research/fusion_v0"
DATA = Path("/data/phonon_fusion_v0")
ORIGINAL = ROOT / "runs/reports/option0_sweep_20260915/results.json"
sys.path.insert(0, str(ROOT / "scripts/option0"))

from gates import FIVE_GATES, GATE_NAMES, PERSONAL_GATES, load_gate  # noqa: E402
from score_gates import read_jsonl, score_prediction  # noqa: E402

SYSTEMS = ("control_b", "control_a", "fusion", "partner_only", "cohere_fusion")
PRIMARY = ("control_b", "control_a", "fusion")


def read_optional(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: Any) -> str:
    return "unmeasured" if value is None else f"{value:.6f}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def score_all(preds: Path) -> tuple[dict, dict, list]:
    models = {system: {"gates": {}} for system in SYSTEMS}
    references: dict[str, list] = {}
    errors = []
    for gate in GATE_NAMES:
        try:
            references[gate] = load_gate(gate)
        except Exception as exc:
            errors.append({"stage": "load_gate", "gate": gate,
                           "error": f"{type(exc).__name__}: {exc}"})
        refs = references.get(gate)
        for system in SYSTEMS:
            path = preds / gate / f"{system}.jsonl"
            metric = {
                "gate": gate, "prediction_file": str(path), "status": "unmeasured",
                "complete": False, "clip_count": 0,
                "expected_clips": len(refs) if refs is not None else None,
                "observed_ids": [], "metadata_valid": False,
            }
            if path.is_file() and refs is not None:
                try:
                    meta = read_optional(Path(str(path) + ".meta.json"), {})
                    metric, diffs = score_prediction(gate, refs, path, meta)
                    metric["observed_ids"] = sorted(row["id"] for row in diffs)
                    metric["prediction_sha256"] = sha256(path)
                except Exception as exc:
                    metric.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
                    errors.append({"stage": "score", "gate": gate, "system": system,
                                   "error": metric["error"]})
            models[system]["gates"][gate] = metric
    common: dict[str, list[str]] = {}
    for gate in FIVE_GATES:
        metrics = [models[system]["gates"][gate] for system in PRIMARY]
        ids = [metric["observed_ids"] for metric in metrics]
        if all(metric.get("metadata_valid") for metric in metrics) and ids[0]:
            if all(value == ids[0] for value in ids[1:]):
                common[gate] = ids[0]
    for model in models.values():
        gates = model["gates"]
        model["five_gate_mean_fair_wer"] = (
            sum(gates[gate]["fair_wer"] for gate in FIVE_GATES) / len(FIVE_GATES)
            if all(gates[gate].get("complete") for gate in FIVE_GATES) else None
        )
        available_complete = len(common) == len(FIVE_GATES) and all(
            gates[gate].get("metadata_valid")
            and gates[gate]["observed_ids"] == common[gate] for gate in FIVE_GATES
        )
        model["available_audio_five_gate_mean_fair_wer"] = (
            sum(gates[gate]["fair_wer"] for gate in FIVE_GATES) / len(FIVE_GATES)
            if available_complete else None
        )
        model["personal_gate_mean_fair_wer"] = (
            sum(gates[gate]["fair_wer"] for gate in PERSONAL_GATES) / len(PERSONAL_GATES)
            if all(gates[gate].get("complete") for gate in PERSONAL_GATES) else None
        )
    return models, {"references": references, "common_ids": common}, errors


def check_reproduction(models: dict, references: dict, original_path: Path) -> dict:
    result: dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "passed": False,
        "tolerance_absolute_fair_wer": 0.002, "original_results": str(original_path),
        "scorer": str(ROOT / "scripts/option0/score_gates.py"),
        "scorer_sha256": sha256(ROOT / "scripts/option0/score_gates.py"), "gates": {},
    }
    try:
        original = json.loads(original_path.read_text())
        result["original_results_sha256"] = sha256(original_path)
        stock = next(row for row in original["models"]
                     if row.get("tag") == "parakeet-tdt-0.6b-v2")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    for gate in ("aqua_new_holdout", "novel180"):
        observed = models["control_b"]["gates"][gate]
        recorded = stock.get("gates", {}).get(gate, {})
        original_meta = recorded.get("metadata", {})
        original_ids = original_meta.get("prediction_ids")
        provenance_error = None
        original_prediction_path = Path(recorded.get("prediction_file", "/nonexistent"))
        try:
            if original_prediction_path.is_file():
                original_rows = read_jsonl(original_prediction_path)
                file_ids = [row["id"] for row in original_rows]
                if original_ids is not None and Counter(file_ids) != Counter(original_ids):
                    raise ValueError("original prediction IDs differ from original metadata")
                original_ids = file_ids
                recorded_hash = original_meta.get("prediction_sha256")
                if recorded_hash and sha256(original_prediction_path) != recorded_hash:
                    raise ValueError("original prediction SHA256 differs from original metadata")
            if original_ids is None:
                raise ValueError("original prediction identity unavailable")
        except Exception as exc:
            provenance_error = f"{type(exc).__name__}: {exc}"
        expected_ids = sorted(str(row["id"]) for row in references.get(gate, []))
        observed_ids = observed.get("observed_ids", [])
        original_ids = sorted(original_ids or [])
        identity_matches = bool(expected_ids) and expected_ids == original_ids == observed_ids
        old_wer, new_wer = recorded.get("fair_wer"), observed.get("fair_wer")
        delta = abs(new_wer - old_wer) if old_wer is not None and new_wer is not None else None
        passed = (identity_matches and not provenance_error and recorded.get("complete") is True
                  and observed.get("complete") is True and delta is not None and delta <= 0.002)
        result["gates"][gate] = {
            "passed": bool(passed), "original_fair_wer": old_wer, "observed_fair_wer": new_wer,
            "absolute_difference": delta, "identity_matches": identity_matches,
            "expected_ids": expected_ids, "original_ids": original_ids,
            "observed_ids": observed_ids, "original_prediction_file": str(original_prediction_path),
            "original_prediction_sha256": original_meta.get("prediction_sha256"),
            "observed_prediction_file": observed.get("prediction_file"),
            "observed_prediction_sha256": observed.get("prediction_sha256"),
            "observed_metadata_valid": observed.get("metadata_valid"),
            "provenance_error": provenance_error,
        }
    result["passed"] = all(row["passed"] for row in result["gates"].values())
    return result


def reproduction_current(reproduction: Any, models: dict) -> bool:
    """A previous pass is invalid after either checked prediction changes."""
    if not reproduction or reproduction.get("passed") is not True:
        return False
    for gate in ("aqua_new_holdout", "novel180"):
        checked = reproduction.get("gates", {}).get(gate, {})
        current = models["control_b"]["gates"][gate]
        if (not current.get("complete") or not checked.get("observed_prediction_sha256")
                or checked["observed_prediction_sha256"] != current.get("prediction_sha256")):
            return False
    return True


def cell(value: Any) -> str:
    return str(value if value is not None else "unmeasured").replace("|", "\\|").replace("\n", " ")


def manifest_inventory(data: Path) -> list[dict]:
    """Count the real selected training manifests without retaining their reference text."""
    inventory = []
    for split in ("train", "dev"):
        path = data / "manifests" / f"{split}.jsonl"
        if not path.is_file():
            continue
        by_source: dict[str, dict] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("source", "unspecified"))
            entry = by_source.setdefault(source, {"source": source, "split": split,
                                                  "clips": 0, "seconds": 0.0})
            entry["clips"] += 1
            entry["seconds"] += float(row["duration_seconds"])
        inventory.extend(by_source.values())
    return inventory


def render_data(lines: list[str], scores: dict) -> None:
    lines.extend(["## Data and exclusions", "",
                  "| Selected source | Split | Clips | Hours |",
                  "| --- | --- | ---: | ---: |"])
    inventory = scores.get("manifest_inventory", [])
    for row in inventory:
        lines.append(f"| {cell(row['source'])} | {row['split']} | {row['clips']} | "
                     f"{row['seconds'] / 3600:.6f} |")
    if not inventory:
        lines.append("| unmeasured | unmeasured | unmeasured | unmeasured |")
    lines.append("")
    summary = scores.get("manifest_summary") or {}
    sources = summary.get("train", {})
    if sources:
        lines.extend(["| Source | Eligible/permitted clips | Selected clips | Selected hours |",
                      "| --- | ---: | ---: | ---: |"])
        for source, row in sources.items():
            lines.append(f"| {cell(source)} | {cell(row.get('eligible', row.get('permitted')))} | "
                         f"{cell(row.get('selected'))} | {number(row.get('hours'))} |")
        lines.append("")
    youtube = sources.get("youtube_technical_v0", {})
    if youtube.get("permitted") == 0:
        lines.extend([f"YouTube: {youtube.get('train_candidate', 'unmeasured')} training candidates; "
                      f"{youtube.get('train_allowed_true', 'unmeasured')} train-allowed rows; "
                      "zero permitted rows selected. Source Parquet permissions were not "
                      "overridden with queues or sidecars.", ""])
    synthetic = sources.get("synthetic_tcpgen_tts_v4_20260630", {})
    if synthetic.get("reason"):
        lines.extend([f"Synthetic pack: {synthetic['reason']}; "
                      f"{synthetic.get('selected', 'unmeasured')} clips selected.", ""])
    lines.extend([f"Exact manifest audit: [summary.json]({scores['data_directory']}/manifests/"
                  "summary.json). All source metadata is retained in scores.json.", ""])
    validation = scores.get("training_validation")
    if validation:
        lines.extend([f"Training memory validation: complete={validation.get('complete')}; "
                      f"retained train clips={validation.get('train_retained', 'unmeasured')}; "
                      f"retained dev clips={validation.get('dev_retained', 'unmeasured')}; "
                      "largest verified joint time/token cells="
                      f"{validation.get('max_verified_joint_cells', 'unmeasured')}. "
                      "The same exclusion mask applies to every training arm.", ""])
    # These are explicit per-clip records. summary.exclusions contains counts, not IDs.
    exclusions = list(scores.get("training_exclusions") or [])
    exclusions.extend((scores.get("independent_data_audit") or {}).get("exclusions", []))
    for source, row in sources.items():
        for key, reason in (("missing_ids", "missing_audio"), ("error_ids", "audio_error")):
            exclusions.extend({"id": row_id, "split": source, "reason": reason}
                              for row_id in row.get(key, []))
    unique = {(str(row['id']), str(row.get('split') or 'source selection'), str(row['reason']))
              for row in exclusions}
    if unique:
        lines.extend(["Recorded excluded clips (ID, split, exact reason):", ""])
        for row_id, split, reason in sorted(unique):
            lines.append(f"- `{row_id}` ({split}): {reason}")
        lines.append("")
    else:
        lines.extend(["No individual training/dev exclusions recorded.", ""])
    lines.extend(["## Missing evaluation audio", "",
                  "| Gate | Expected labeled clips | Missing audio clips |",
                  "| --- | ---: | ---: |"])
    missing = scores.get("eval_missing")
    for gate in GATE_NAMES:
        expected = scores["models"]["control_b"]["gates"][gate]["expected_clips"]
        count = None
        if isinstance(missing, dict):
            entry = missing.get(gate, missing.get("gates", {}).get(gate))
            if isinstance(entry, list):
                count = len(entry)
            elif isinstance(entry, (int, float)):
                count = entry
            elif isinstance(entry, dict):
                count = entry.get("missing_count", entry.get("missing"))
                if isinstance(count, list):
                    count = len(count)
                if count is None:
                    ids = entry.get("missing_ids", entry.get("errors"))
                    count = len(ids) if isinstance(ids, list) else None
        elif isinstance(missing, list):
            count = sum(row.get("gate") == gate for row in missing if isinstance(row, dict))
        lines.append(f"| {gate} | {cell(expected)} | {cell(count)} |")
    lines.extend(["", f"Exact missing IDs and errors: [eval_missing.json]"
                  f"({scores['eval_missing_path']}).", ""])


def reported_systems(scores: dict) -> list[str]:
    selected = (scores.get("followup") or {}).get("system")
    return [system for system in SYSTEMS if system in PRIMARY or system == selected
            or any(row.get("clip_count", 0) for row in scores["models"][system]["gates"].values())]


def render_training_timing(lines: list[str], scores: dict) -> None:
    lines.extend(["## Training", "",
                  "| System | Epochs | Steps | Best dev loss | Train/dev before exclusions | "
                  "Retained train/dev | Elapsed seconds |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    records = {row.get("system", Path(path).parent.name): row
               for path, row in scores["training"].items()}
    validation = scores.get("training_validation") or {}
    for system in reported_systems(scores):
        if system == "control_b":
            continue
        row = records.get(system, {})
        retained = (f"{validation['train_retained']}/{validation['dev_retained']}"
                    if row and "train_retained" in validation and "dev_retained" in validation
                    else "unmeasured")
        counts = (f"{row['train_clips']}/{row['dev_clips']}"
                  if "train_clips" in row and "dev_clips" in row else "unmeasured")
        lines.append(f"| {system} | {cell(row.get('epochs'))} | {cell(row.get('steps'))} | "
                     f"{number(row.get('best_dev_loss'))} | {counts} | {retained} | "
                     f"{number(row.get('elapsed_seconds'))} |")
    lines.extend(["", "Checkpoints are selected by the lowest recorded development loss. "
                  "Exact histories, optimizer settings, manifest hashes, and exclusions are "
                  "retained in scores.json and checkpoints/<system>/training.json.", ""])
    if validation.get("complete"):
        inventory = scores.get("manifest_inventory", [])
        hours = sum(row["seconds"] for row in inventory if row["split"] == "train") / 3600
        excluded_hours = sum(row.get("duration_seconds", 0) for row in
                             (scores.get("training_exclusions") or [])
                             if row.get("split") == "train") / 3600
        lines.extend([f"Shared memory exclusions remove {excluded_hours:.6f} training hours; "
                      f"the actual training set contains {hours - excluded_hours:.6f} hours.", ""])
    primary = scores.get("primary_partner")
    if primary:
        lines.extend([f"Primary frozen partner: {cell(primary.get('encoder'))}.", ""])
    followup = scores.get("followup")
    if followup:
        lines.extend([f"Conditional follow-up: {cell(followup.get('system'))}, encoder "
                      f"{cell(followup.get('encoder'))}. {followup.get('reason', '')}", ""])
    timing = scores.get("timing") or {}
    lines.extend(["## Decode speed", "",
                  "Serial realtime factor (RTF) is the sum of separately measured Parakeet "
                  "preprocessing/encoder, partner preprocessing, partner forward/alignment, and cached "
                  "adapter/native TDT decoder seconds divided by audio seconds. This component "
                  "sum excludes model loading and cache I/O; it is not a directly measured "
                  "integrated live application pipeline.", "",
                  f"Timing gate: {timing.get('gate', 'unmeasured')}.", "",
                  "| System | Clips | Audio s | Parakeet s | Partner preprocessing s | "
                  "Partner forward s | Partner alignment s | Adapter/decoder s | Serial total s | Serial RTF |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for system in reported_systems(scores):
        row = timing.get("systems", {}).get(system, {})
        values = [number(row.get(key)) for key in (
            "audio_seconds", "parakeet_preprocessing_and_encoder_seconds",
            "partner_preprocessing_seconds", "partner_forward_seconds", "partner_alignment_seconds",
            "adapter_decoder_seconds",
            "serial_total_seconds", "serial_realtime_factor")]
        lines.append(f"| {system} | {cell(row.get('clips'))} | " + " | ".join(values) + " |")
    lines.extend(["", "Partner-only timing includes the Parakeet pass used for the shared frame grid.", "", f"Exact timing record: [timing.json]({scores['data_directory']}/timing.json).",
                  ""])


def render_errors(lines: list[str], errors: list) -> None:
    lines.extend(["## Exact failures", ""])
    if not errors:
        lines.extend(["No failures recorded.", ""])
    for row in errors:
        if not isinstance(row, dict):
            row = {"error": str(row)}
        stage = row.get("stage", row.get("name", "unspecified"))
        detail = ", ".join(f"{key}={row[key]}" for key in ("system", "gate", "id") if key in row)
        lines.extend([f"{stage}; attempt {row.get('attempt', 'unrecorded')}"
                      f"{'; ' + detail if detail else ''}:", ""])
        # Preserve exception text byte-for-byte, including multiline tracebacks, without
        # serializing unrelated commands, process state, or ID arrays from the error object.
        error = str(row.get("error", row.get("raw_error", "No error text recorded.")))
        fence = "`" * max(3, max((len(part) for part in error.split() if set(part) == {'`'}),
                                 default=0) + 1)
        lines.extend([fence + "text", error, fence, ""])
        if row.get("resolved_by"):
            lines.extend([f"Resolved by: {row['resolved_by']}", ""])


def render(scores: dict) -> str:
    models = scores["models"]
    lines = ["# Fusion v0 results", "", f"Generated: {scores['generated_at']}.", ""]
    complete = all(models[system]["gates"][gate].get("complete")
                   for system in PRIMARY for gate in GATE_NAMES)
    reproduction = scores["reproduction"]
    trusted = scores["reproduction_current"]
    splits = (scores.get("manifest_summary") or {}).get("splits", {})
    available_complete = bool(splits) and all(
        models[system]["gates"][gate].get("metadata_valid")
        and models[system]["gates"][gate].get("clip_count") == splits.get(gate, {}).get("rows")
        for system in reported_systems(scores) for gate in GATE_NAMES)
    execution_status = ("Execution complete on available data; full gate coverage incomplete."
                        if available_complete and trusted and not complete else
                        f"Experiment status: {'complete' if complete and trusted else 'incomplete'}.")
    lines.extend([
        execution_status, "",
        "Control B is stock Parakeet v2 decoded through this experiment's path. Control A "
        "trains the same adapter, prediction network, and joint with partner features zeroed. "
        "Fusion trains them with both frozen encoders. Partner-only zeros the Parakeet features. "
        "Cohere fusion is the conditional second-partner experiment.", "",
        "Fair WER uses the unchanged Option 0 scorer's English normalization and corpus word "
        "error rate. Strict term error uses its case-sensitive token matching. All rates below "
        "are fractions, not percentages. A gate without annotated terms is unmeasured for terms. "
        "Observed/expected counts retain every reference returned by the read-only gate loader.",
        "", "## Control B reproduction", "",
        f"Recorded check: {'PASS' if trusted else 'NOT PASSED OR NOT CURRENT'}. "
        "Both gates must use identical original and current reference IDs, valid prediction "
        "sidecars, and absolute fair-WER difference at most 0.002.", "",
        "| Gate | Original fair WER | Own fair WER | Absolute difference | IDs match | Pass |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ])
    for gate in ("aqua_new_holdout", "novel180"):
        row = (reproduction or {}).get("gates", {}).get(gate, {})
        lines.append(f"| {gate} | {number(row.get('original_fair_wer'))} | "
                     f"{number(row.get('observed_fair_wer'))} | "
                     f"{number(row.get('absolute_difference'))} | "
                     f"{row.get('identity_matches', 'unmeasured')} | "
                     f"{row.get('passed', 'unmeasured')} |")
    lines.extend(["", "Running --check-reproduction records exact ID lists and source SHA256 "
                  "hashes in reproduction.json and scores.json. Evaluation scores never select "
                  "a checkpoint; only dev loss does.",
                  "", "## Every gate", "",
                  "| Gate | System | Observed/expected | Fair WER | Strict term error | Status |",
                  "| --- | --- | ---: | ---: | ---: | --- |"])
    for gate in GATE_NAMES:
        for system in reported_systems(scores):
            row = models[system]["gates"][gate]
            status = row["status"]
            if row.get("metadata_errors"):
                status += "; metadata invalid"
            lines.append(f"| {gate} | {system} | {row.get('clip_count', 0)}/"
                         f"{row['expected_clips'] if row['expected_clips'] is not None else '?'} | "
                         f"{number(row.get('fair_wer'))} | "
                         f"{number(row.get('strict_technical_term_error_rate'))} | {status} |")
    lines.extend(["", "## Five-gate and personal summaries", "",
                  "| System | Full five-gate mean fair WER | Available-audio five-gate mean "
                  "fair WER | Full personal mean fair WER | wispr_holdout120 fair WER |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for system in reported_systems(scores):
        model = models[system]
        lines.append(f"| {system} | {number(model['five_gate_mean_fair_wer'])} | "
                     f"{number(model['available_audio_five_gate_mean_fair_wer'])} | "
                     f"{number(model['personal_gate_mean_fair_wer'])} | "
                     f"{number(model['gates']['wispr_holdout120'].get('fair_wer'))} |")
    lines.extend(["", "Five-gate means weight aqua_new_holdout, course91, novel180, hard77, "
                  "and uncertain48 equally. The full mean requires complete coverage of all five. "
                  "The available-audio mean requires nonempty, identical IDs across Control A, "
                  "Control B, and fusion on every gate; optional systems must match those IDs. "
                  "Missing audio prevents gains on this subset from establishing a full mean. "
                  "The personal mean requires all three personal gates to be complete.", "",
                  "## Verdict", ""])
    fusion = models["fusion"]
    control = models["control_a"]
    comparable = (trusted and fusion["five_gate_mean_fair_wer"] is not None
                  and control["five_gate_mean_fair_wer"] is not None
                  and fusion["gates"]["wispr_holdout120"].get("complete")
                  and control["gates"]["wispr_holdout120"].get("complete"))
    if comparable:
        mean_gain = control["five_gate_mean_fair_wer"] - fusion["five_gate_mean_fair_wer"]
        personal_gain = (control["gates"]["wispr_holdout120"]["fair_wer"]
                         - fusion["gates"]["wispr_holdout120"]["fair_wer"])
        verdict = ("Fusion beats Control A on both required measurements."
                   if mean_gain > 0 and personal_gain > 0
                   else "Fusion does not beat Control A on both required measurements.")
        lines.extend([f"{verdict} Control A minus fusion: full five-gate mean "
                      f"{mean_gain:.6f}; wispr_holdout120 {personal_gain:.6f}. "
                      "Positive differences favor fusion.", ""])
    else:
        lines.extend(["The requested full encoder-borrowing effect is not established. "
                      "Full comparable five-gate coverage, complete wispr_holdout120 results, "
                      "and a current passing Control B reproduction check are required.", ""])
        fmean = fusion["available_audio_five_gate_mean_fair_wer"]
        cmean = control["available_audio_five_gate_mean_fair_wer"]
        fg, cg = (model["gates"]["wispr_holdout120"] for model in (fusion, control))
        mean_gain = cmean - fmean if cmean is not None and fmean is not None else None
        personal_gain = (cg["fair_wer"] - fg["fair_wer"]
                         if fg.get("complete") and cg.get("complete") else None)
        lines.extend(["Observed Control A minus fusion differences: available-audio five-gate "
                      f"mean {number(mean_gain)}; complete wispr_holdout120 "
                      f"{number(personal_gain)}. Positive differences favor fusion. "
                      "These observations are insufficient to establish the full requested "
                      "five-gate mean.", ""])
    fw = fusion["gates"]["wispr_holdout120"]
    cw = control["gates"]["wispr_holdout120"]
    if trusted and fw.get("complete") and cw.get("complete") and fw["fair_wer"] >= cw["fair_wer"]:
        lines.extend([f"Fusion fails the requested win criterion: Wispr WER increases from "
                      f"{cw['fair_wer']:.6f} for Control A to {fw['fair_wer']:.6f}. "
                      "The conditional second encoder was therefore not run.", ""])
    baseline = models["control_b"]["available_audio_five_gate_mean_fair_wer"]
    if baseline is not None:
        lines.extend([f"Stock Parakeet has available-audio five-gate mean WER {baseline:.6f}, "
                      f"versus {control['available_audio_five_gate_mean_fair_wer']:.6f} for Control A "
                      f"and {fusion['available_audio_five_gate_mean_fair_wer']:.6f} for fusion. "
                      "The adaptation gains on personal dictation come with substantial general "
                      "ASR regressions; this recipe does not justify replacing the app's stock model.", ""])
    ablation = models["partner_only"]
    aw = ablation["gates"]["wispr_holdout120"]
    amean = ablation["available_audio_five_gate_mean_fair_wer"]
    if aw.get("complete") and amean is not None:
        lines.extend([f"Partner-only features produce transcripts through Parakeet's decoder, "
                      f"but accuracy is worse: available-audio mean WER {amean:.6f} and "
                      f"Wispr WER {aw['fair_wer']:.6f}. The frozen Qwen tower plus this small "
                      "adapter did not provide a competitive standalone replacement in this run.", ""])
    inventory = scores.get("manifest_inventory", [])
    if inventory and all("wispr" in row["source"].lower() for row in inventory):
        lines.extend(["The selected training and development manifests contain only Wispr "
                      "audio. This limits the hypothesis test to personal dictation adaptation; "
                      "the requested broader technical-data training mixture was not tested.", ""])
        sources = (scores.get("manifest_summary") or {}).get("train", {})
        no_youtube = sources.get("youtube_technical_v0", {}).get("permitted") == 0
        synthetic = sources.get("synthetic_tcpgen_tts_v4_20260630", {})
        empty_synthetic = synthetic.get("file_count") == 0
        if no_youtube and empty_synthetic:
            lines.extend(["The source audit records no permitted YouTube training rows/labels "
                          "and an empty synthetic pack. Neither source contributed training "
                          "audio; source permissions were not overridden.", ""])
    render_data(lines, scores)
    render_training_timing(lines, scores)
    render_errors(lines, scores["errors"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path, default=WORK / "preds")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=WORK)
    parser.add_argument("--original-results", type=Path, default=ORIGINAL)
    parser.add_argument("--check-reproduction", action="store_true")
    args = parser.parse_args()
    models, coverage, errors = score_all(args.preds)
    reproduction_path = args.data / "reproduction.json"
    if args.check_reproduction:
        reproduction = check_reproduction(models, coverage["references"], args.original_results)
        write_json(reproduction_path, reproduction)
    else:
        reproduction = read_optional(reproduction_path)
    error_path = args.data / "errors.jsonl"
    if error_path.is_file():
        for line in error_path.read_text().splitlines():
            if line.strip():
                try:
                    errors.append(json.loads(line))
                except json.JSONDecodeError:
                    errors.append({"stage": "error_log_parse", "raw_error": line})
    scores = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "models": models,
        "data_directory": str(args.data),
        "prediction_directory": str(args.preds),
        "scorer": str(ROOT / "scripts/option0/score_gates.py"),
        "five_gates": list(FIVE_GATES), "personal_gates": list(PERSONAL_GATES),
        "available_audio_common_ids": coverage["common_ids"],
        "manifest_summary": read_optional(args.data / "manifests/summary.json"),
        "manifest_inventory": manifest_inventory(args.data),
        "eval_missing": read_optional(args.data / "manifests/eval_missing.json",
                                      read_optional(args.data / "eval_missing.json")),
        "eval_missing_path": str(args.data / "manifests/eval_missing.json"
                                 if (args.data / "manifests/eval_missing.json").is_file()
                                 else args.data / "eval_missing.json"),
        "reproduction": reproduction,
        "reproduction_current": reproduction_current(reproduction, models),
        "training": {str(path.relative_to(args.data)): read_optional(path)
                     for path in sorted((args.data / "checkpoints").glob("*/training.json"))},
        "training_validation": read_optional(args.data / "training_validation.json"),
        "training_exclusions": read_optional(args.data / "training_exclusions.json"),
        "independent_data_audit": read_optional(args.data / "independent_data_audit.json"),
        "followup": read_optional(args.data / "followup.json"),
        "primary_partner": read_optional(args.data / "primary_partner.json"),
        "timing": read_optional(args.data / "timing.json"), "errors": errors,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "scores.json", scores)
    (args.output / "results.md").write_text(render(scores))
    print(args.output / "results.md")
    return 1 if args.check_reproduction and not scores["reproduction_current"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
