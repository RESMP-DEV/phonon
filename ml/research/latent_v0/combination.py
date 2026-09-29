"""Reproducible, CPU-only evaluation of existing ASR hypothesis combinations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import matplotlib
import numpy as np
from rapidfuzz.distance import Levenshtein

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/option0"))
from gates import load_gate  # noqa: E402
from score_gates import CANDIDATES, NORMALIZE, score_prediction  # noqa: E402

OUT = Path(__file__).resolve().parent
PREDS = ROOT / "runs/reports/option0_sweep_20260915/preds"
GATES = (
    "aqua_new_holdout", "novel180", "hard77", "uncertain48", "course91",
    "wispr_holdout120", "wispr_edit25", "personal_cuda",
)
MODELS = [tag for tag, _, _ in CANDIDATES]
BASE = "parakeet-tdt-0.6b-v2"
FAST = [BASE, "parakeet-unified-en-0.6b", "cohere-transcribe-03-2026",
        "granite-speech-5.0-470m-turboctc"]
SHORT = ["Parakeet v2", "Parakeet v3", "Unified", "Local v3", "Qwen 0.6B",
         "Qwen 1.7B", "Cohere", "Granite 4.1", "Granite 4.1+", "Granite 5.0"]


def mbr_index(hypotheses: list[str]) -> int:
    """Medoid under symmetric word edit distance / max word count."""
    tokens = [hyp.split() for hyp in hypotheses]
    costs = np.zeros(len(tokens))
    for i, a in enumerate(tokens):
        for j in range(i):
            b = tokens[j]
            distance = Levenshtein.distance(a, b) / max(len(a), len(b), 1)
            costs[i] += distance
            costs[j] += distance
    return int(np.argmin(costs))


def winner(votes: Counter, preferred: str = "") -> str:
    maximum = max(votes.values())
    if votes[preferred] == maximum:
        return preferred
    return next(token for token, count in votes.items() if count == maximum)


def rover(hypotheses: list[str], backbone_index: int) -> str:
    """Progressive word confusion network, including explicit epsilon votes.

    Align each additional hypothesis to one nonempty representative per column.
    Insertions introduce columns with epsilon votes for every prior system.
    Every column ends with exactly one vote per system. Equal votes retain the
    original backbone token, or epsilon for newly introduced insertion columns.
    """
    backbone = hypotheses[backbone_index].split()
    columns = [Counter({word: 1}) for word in backbone]
    preferences = list(backbone)
    processed = 1
    order = [i for i in range(len(hypotheses)) if i != backbone_index]
    for index in order:
        representative = [winner(Counter({k: v for k, v in col.items() if k}))
                          for col in columns]
        hyp = hypotheses[index].split()
        if not representative:
            columns = [Counter({"": processed, word: 1}) for word in hyp]
            preferences = [""] * len(hyp)
            processed += 1
            continue
        alignment = jiwer.process_words(" ".join(representative), " ".join(hyp))
        new_columns, new_preferences = [], []
        for chunk in alignment.alignments[0]:
            a, b = chunk.ref_start_idx, chunk.ref_end_idx
            c, d = chunk.hyp_start_idx, chunk.hyp_end_idx
            if chunk.type == "insert":
                for word in hyp[c:d]:
                    new_columns.append(Counter({"": processed, word: 1}))
                    new_preferences.append("")
            else:
                for offset, old_index in enumerate(range(a, b)):
                    col = columns[old_index].copy()
                    col["" if chunk.type == "delete" else hyp[c + offset]] += 1
                    new_columns.append(col)
                    new_preferences.append(preferences[old_index])
        columns, preferences = new_columns, new_preferences
        processed += 1
        assert all(sum(col.values()) == processed for col in columns)
    return " ".join(word for col, preference in zip(columns, preferences, strict=True)
                    if (word := winner(col, preference)))


def alignment_checks() -> None:
    assert rover(["a b", "a x b", "a x b"], 0) == "a x b"
    assert rover(["a x b", "a b", "a b"], 0) == "a b"
    assert rover(["a b c", "a d c", "a d c"], 0) == "a d c"
    assert rover(["", "a", "a"], 0) == "a"
    assert rover(["a", "", ""], 0) == ""
    assert rover(["a b", "a c"], 0) == "a b"
    assert rover(["a b", "a x b"], 0) == "a b"
    assert rover(["a", "x a y", "x a y"], 0) == "x a y"
    assert mbr_index(["a b", "a b", "x y z"]) in (0, 1)


def load_predictions(gate: str, rows: list[dict], supplementary: Path | None = None) -> tuple[dict, dict]:
    predictions, coverage = {}, {}
    for model in MODELS:
        path = PREDS / gate / f"{model}.jsonl"
        provenance = "existing option0 sweep predictions"
        if not path.exists() and supplementary is not None:
            fallback = supplementary / gate / f"{model}.jsonl"
            if fallback.exists():
                path = fallback
                provenance = "new latent extraction CTC decode"
        if not path.exists():
            coverage[model] = {"status": "missing", "expected_clips": len(rows),
                               "source_path": str(path)}
            continue
        before = path.read_bytes()
        digest = hashlib.sha256(before).hexdigest()
        sidecar = path.with_suffix(".jsonl.meta.json")
        meta_bytes = sidecar.read_bytes() if sidecar.exists() else b"{}"
        try:
            meta = json.loads(meta_bytes)
            metrics, diffs = score_prediction(gate, rows, path, meta)
        except (ValueError, TypeError, KeyError) as exc:
            coverage[model] = {"status": "invalid", "error": str(exc), "sha256": digest,
                               "source_path": str(path), "provenance": provenance}
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Prediction changed during scoring: {path}")
        if sidecar.exists() and sidecar.read_bytes() != meta_bytes:
            raise RuntimeError(f"Prediction sidecar changed during scoring: {sidecar}")
        coverage[model] = {
            key: value for key, value in metrics.items()
            if key in {"status", "complete", "expected_clips", "prediction_rows", "clip_count",
                       "missing_prediction_ids", "extraneous_prediction_ids",
                       "duplicate_prediction_ids", "metadata_valid", "metadata_errors", "error"}
        }
        coverage[model].update(sha256=digest, source_path=str(path), provenance=provenance,
                               sidecar_sha256=hashlib.sha256(meta_bytes).hexdigest())
        if metrics["status"] != "invalid":
            predictions[model] = {row["id"]: row["normalized_hypothesis"] for row in diffs}
    return predictions, coverage


def evaluate_subset(refs: dict, predictions: dict, models: list[str]) -> dict:
    available = [model for model in models if model in predictions and predictions[model]]
    ids = sorted(set(refs).intersection(*(set(predictions[m]) for m in available)))
    if not available or not ids:
        return {"models": available, "clips": 0}
    outputs = {name: [] for name in ("baseline", "oracle", "mbr", "rover")}
    choices = {"oracle": Counter(), "mbr": Counter()}
    clips = []
    references = [refs[row_id] for row_id in ids]
    for row_id, ref in zip(ids, references, strict=True):
        hyps = [predictions[m][row_id] for m in available]
        errors = [Levenshtein.distance(ref.split(), hyp.split()) for hyp in hyps]
        oracle = int(np.argmin(errors))
        mbr = mbr_index(hyps)
        vote = rover(hyps, mbr)
        outputs["baseline"].append(predictions[BASE][row_id])
        outputs["oracle"].append(hyps[oracle])
        outputs["mbr"].append(hyps[mbr])
        outputs["rover"].append(vote)
        choices["oracle"][available[oracle]] += 1
        choices["mbr"][available[mbr]] += 1
        clips.append({"id": row_id, "oracle": available[oracle], "mbr": available[mbr],
                      "rover_hypothesis": vote})
    return {
        "requested_models": models, "models": available, "clips": len(ids),
        "missing_models": [m for m in models if m not in available], "ids": ids,
        "reference_words": sum(len(ref.split()) for ref in references),
        "fair_wer": {name: jiwer.wer(references, hyps) for name, hyps in outputs.items()},
        "selection_counts": choices, "clip_outputs": clips,
    }


def pairwise(refs: dict, predictions: dict) -> dict:
    numerator = np.zeros((len(MODELS), len(MODELS)), dtype=np.int64)
    denominator = numerator.copy()
    for i, a in enumerate(MODELS):
        for j, b in enumerate(MODELS):
            if a not in predictions or b not in predictions:
                continue
            ids = set(refs) & set(predictions[a]) & set(predictions[b])
            denominator[i, j] = len(ids)
            numerator[i, j] = sum(predictions[a][r] != refs[r] and
                                   predictions[b][r] == refs[r] for r in ids)
    return {"counts": numerator.tolist(), "denominators": denominator.tolist()}


def aggregate_gates(result: dict, gates: list[str]) -> dict:
    """Pool corpus error numerators, retaining original-only comparisons."""
    data = [result["gates"][gate] for gate in gates]
    aggregate = {"gates": gates}
    for group in ("all_requested", "fast_requested"):
        selected = [row[group] for row in data if row[group].get("reference_words")]
        words = sum(row["reference_words"] for row in selected)
        scores = {metric: sum(row["fair_wer"][metric] * row["reference_words"]
                              for row in selected) / words
                  for metric in ("baseline", "oracle", "mbr", "rover")} if words else {}
        aggregate[group] = {"clips": sum(row["clips"] for row in selected),
                            "reference_words": words, "fair_wer": scores}
        if any("original_existing" in row for row in data):
            original = [row.get("original_existing", row)[group] for row in data]
            original = [row for row in original if row.get("reference_words")]
            original_words = sum(row["reference_words"] for row in original)
            original_scores = {
                metric: sum(row["fair_wer"][metric] * row["reference_words"]
                            for row in original) / original_words
                for metric in ("baseline", "oracle", "mbr", "rover")
            } if original_words else {}
            aggregate[group]["original_existing"] = {
                "clips": sum(row["clips"] for row in original),
                "reference_words": original_words, "fair_wer": original_scores,
            }
            same_support = all(row[group].get("ids") ==
                               row.get("original_existing", row)[group].get("ids") for row in data)
            aggregate[group]["same_support_as_original"] = same_support
            if same_support and scores and original_scores:
                aggregate[group]["delta_percentage_points_vs_original"] = {
                    metric: 100 * (scores[metric] - original_scores[metric]) for metric in scores}
    aggregate["pairwise"] = {
        key: np.sum([row["pairwise"][key] for row in data], axis=0).tolist()
        for key in ("counts", "denominators")
    }
    return aggregate


def plot_matrix(result: dict) -> None:
    counts = np.asarray(result["pairwise"]["counts"])
    denoms = np.asarray(result["pairwise"]["denominators"])
    fractions = np.divide(counts * 100.0, denoms, out=np.full(counts.shape, np.nan),
                          where=denoms > 0)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12.5, 10.5), facecolor="#111827")
    ax.set_facecolor("#111827")
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#283142")
    im = ax.imshow(fractions, vmin=0, cmap=cmap)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            label = "missing" if not denoms[i, j] else f"{fractions[i, j]:.1f}%\n{counts[i, j]}/{denoms[i, j]}"
            color = "black" if denoms[i, j] and fractions[i, j] > np.nanmax(fractions) * 0.65 else "white"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.5, color=color)
    ax.set_xticks(range(len(MODELS)), SHORT, rotation=45, ha="right")
    ax.set_yticks(range(len(MODELS)), SHORT)
    ax.set_xlabel("B: exact after fair normalization")
    ax.set_ylabel("A: at least one word error after fair normalization")
    ax.set_title("Directional clip complementarity on eight gates\nA wrong and B exact / all paired clips", pad=16)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025, label="Percent of paired clips")
    fig.tight_layout()
    fig.savefig(OUT / "complementarity.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def markdown(result: dict) -> str:
    lines = [
        "# Combination of existing ASR predictions", "",
        "Fair WER is corpus word error rate after the exact `NORMALIZE` object from "
        "`scripts/option0/score_gates.py` is applied to both sides. Values below are percentages; "
        "lower is better. No labels train or tune any selector.", "",
        "The requested groups are all ten systems and the fast four: Parakeet v2, unified, "
        "Cohere and Granite 5.0. Each row states the actual system count and compares exactly "
        "the same clip IDs within its group. A smaller count means the full requested group "
        "is unavailable on that gate.", "",
        "Original prediction files take precedence. Only a missing original file can be supplied "
        "from `research/latent_v0/preds`. Supplementary Granite 5.0 transcripts are new greedy "
        "CTC decodes from the latent extraction, with the source path, provenance and SHA256 "
        "recorded per file in `combination.json`. They are not pre-existing sweep predictions.", "",
        "| Gate | Group | Systems | Expected | Paired clips | Parakeet v2 | Oracle | MBR | ROVER |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for gate, data in result["gates"].items():
        for key, label in [("all_requested", "all"), ("fast_requested", "fast")]:
            group = data[key]
            values = group.get("fair_wer", {})
            cells = [gate, label, str(len(group["models"])), str(data["expected_clips"]),
                     str(group["clips"])]
            cells += [f"{100 * values[name]:.3f}" if name in values else "unavailable"
                      for name in ("baseline", "oracle", "mbr", "rover")]
            lines.append("| " + " | ".join(cells) + " |")
    if result.get("aggregates"):
        lines += ["", "## Corpus aggregates", "",
                  "These pool word-error counts, rather than averaging gate WERs. The six-gate "
                  "subset excludes course91 and personal_cuda and matches the feature-extraction "
                  "clip set. Deltas are supplemented minus original-only WER in percentage points; "
                  "negative means an improvement. A delta is reported only on identical support.", "",
                  "| Scope | Group | Clips | Reference words | Parakeet v2 | Oracle | MBR | ROVER | Oracle delta | MBR delta | ROVER delta |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for scope, data in result["aggregates"].items():
            for key, label in [("all_requested", "all"), ("fast_requested", "fast")]:
                group = data[key]
                scores = group["fair_wer"]
                delta = group.get("delta_percentage_points_vs_original", {})
                cells = [scope, label, str(group["clips"]), str(group["reference_words"])]
                cells += [f"{100 * scores[name]:.4f}" if name in scores else "unavailable"
                          for name in ("baseline", "oracle", "mbr", "rover")]
                cells += [f"{delta[name]:+.4f}" if name in delta else "unavailable"
                          for name in ("oracle", "mbr", "rover")]
                lines.append("| " + " | ".join(cells) + " |")
    original = {gate: data["original_existing"] for gate, data in result["gates"].items()
                if "original_existing" in data}
    if original:
        lines += ["", "## Original files without supplementary transcripts", "",
                  "These comparisons retain the original available systems, preserving the nine-system "
                  "and three-system baseline where Granite 5.0 was absent. Original-only and supplemented "
                  "groups may have different clip coverage; their paired counts are shown explicitly.", "",
                  "| Gate | Group | Systems | Paired clips | Parakeet v2 | Oracle | MBR | ROVER |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for gate, groups in original.items():
            for key, label in [("all_requested", "all original"), ("fast_requested", "fast original")]:
                group = groups[key]
                values = group.get("fair_wer", {})
                cells = [gate, label, str(len(group["models"])), str(group["clips"])]
                cells += [f"{100 * values[name]:.3f}" if name in values else "unavailable"
                          for name in ("baseline", "oracle", "mbr", "rover")]
                lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "The oracle chooses the minimum-reference-WER hypothesis per clip and is an upper "
              "bound requiring the answer. MBR (minimum Bayes risk) chooses a hypothesis whose mean "
              "word edit distance to the other hypotheses is smallest. Each pairwise distance is "
              "divided by the larger hypothesis word count, making it symmetric. Ties use the "
              "fixed model order from `score_gates.CANDIDATES`. Systems have equal votes; the "
              "four related Parakeet systems are therefore correlated voters.", "",
              "ROVER is a word-level progressive confusion-network vote initialized by the MBR "
              "hypothesis. Alignment uses jiwer; deletions and newly introduced insertion columns "
              "receive explicit empty-word votes, so every system contributes one vote per column. "
              "A tie keeps the original backbone token, or the empty word in a new insertion column. "
              "Other hypotheses enter in fixed model order. This is a ROVER-style implementation, "
              "not an implementation claiming equivalence to a particular ROVER package.", "",
              "## Pairwise rescue of Parakeet v2", "",
              "A rescue requires Parakeet v2 to have any normalized word error and the partner to "
              "match the entire normalized reference. The denominator is all paired clips, "
              "not only the clips Parakeet misses. Repeated audio across gates is counted once "
              "per gate; this aggregate describes the evaluation estate and is not an independent "
              "sample confidence estimate.", "",
              "| Partner | Rescued clips | Paired clips | Fraction (%) |",
              "| --- | ---: | ---: | ---: |"]
    counts, denoms = result["pairwise"]["counts"], result["pairwise"]["denominators"]
    for index, model in enumerate(MODELS):
        denominator = denoms[0][index]
        fraction = f"{100 * counts[0][index] / denominator:.3f}" if denominator else "unavailable"
        lines.append(f"| {model} | {counts[0][index]} | {denominator} | {fraction} |")
    ranked = sorted((i for i in range(1, len(MODELS)) if denoms[0][i]),
                    key=lambda i: counts[0][i] / denoms[0][i], reverse=True)
    leaders = "; ".join(f"{SHORT[i]} {counts[0][i]}/{denoms[0][i]} "
                        f"({100 * counts[0][i] / denoms[0][i]:.3f}%)" for i in ranked[:3])
    qwen = MODELS.index("Qwen3-ASR-0.6B-hf")
    reverse = (f"Qwen 0.6B rescues {counts[0][qwen]}/{denoms[0][qwen]} Parakeet-v2 failures, "
               f"while Parakeet v2 rescues {counts[qwen][0]}/{denoms[qwen][0]} Qwen-0.6B failures. "
               if denoms[0][qwen] and denoms[qwen][0] else "")
    lines += ["", "![Directional clip complementarity](complementarity.png)", "",
              f"The largest exact-rescue fractions in the Parakeet-v2 row are {leaders}. "
              + reverse + "The asymmetric colors describe both unique successes and differences "
              "in overall accuracy; they do not by themselves isolate information unique to an "
              "encoder. Gray cells mean missing paired predictions, not zero complementarity.", "",
              "## Coverage and validation", "",
              "The source scorer's invalid-configuration exclusions are reused, including obsolete "
              "Qwen language forcing, Granite-plus prompts and sub-second tail chunking. "
              "Empty string hypotheses are valid failures; duplicate IDs, extras and invalid "
              "configurations are excluded. Partial but otherwise valid predictions are scored on "
              "their common IDs. Missing IDs, sidecar validation errors, exact source SHA256 hashes "
              "and selected hypotheses are recorded in `combination.json`. Source bytes and "
              "sidecars are checked for concurrent changes during loading.", ""]
    for gate, data in result["gates"].items():
        valid = [c for c in data["coverage"].values() if c.get("sha256")]
        invalid = {m: c for m, c in data["coverage"].items() if c["status"] == "invalid"}
        errors = {m: c.get("metadata_errors") for m, c in data["coverage"].items()
                  if c.get("metadata_errors")}
        lines.append(f"{gate}: {data['all_requested']['clips']}/{data['expected_clips']} paired clips; "
                     f"{len(valid)} source files; {len(invalid)} invalid files; "
                     f"{len(errors)} sidecars with validation warnings.")
        if errors:
            lines.append("Sidecar warnings: " + json.dumps(errors, sort_keys=True) + ".")
        supplemental = [m for m, c in data["coverage"].items()
                        if c.get("provenance") == "new latent extraction CTC decode"]
        if supplemental:
            lines.append("Supplementary CTC predictions: " + ", ".join(supplemental) + ".")
        lines.append("")
    lines += ["## Rerun", "", "```bash",
              "uv run --no-sync --with whisper-normalizer python research/latent_v0/combination.py",
              "uv run ruff check research/latent_v0/combination.py", "```", "",
              "The supplementary directory defaults to `research/latent_v0/preds`; override it "
              "with `--supplementary-preds PATH`, or use `--existing-only` to ignore supplementary files.", "",
              "Small built-in checks cover insertions, deletions, substitutions, empty hypotheses, "
              "tie behavior and MBR consensus. All processing is CPU-only."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplementary-preds", type=Path, default=OUT / "preds")
    parser.add_argument("--existing-only", action="store_true")
    args = parser.parse_args()
    supplementary = None if args.existing_only else args.supplementary_preds.resolve()
    alignment_checks()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "models": MODELS,
              "normalizer": "scripts.option0.score_gates.NORMALIZE", "gates": {},
              "supplementary_directory": str(supplementary) if supplementary else None}
    total_counts = np.zeros((len(MODELS), len(MODELS)), dtype=np.int64)
    total_denoms = total_counts.copy()
    for gate in GATES:
        rows = load_gate(gate)
        refs = {row["id"]: NORMALIZE(row["reference"]) for row in rows}
        empty = [row_id for row_id, ref in refs.items() if not ref.strip()]
        refs = {row_id: ref for row_id, ref in refs.items() if ref.strip()}
        predictions, coverage = load_predictions(gate, rows, supplementary)
        pairs = pairwise(refs, predictions)
        total_counts += np.asarray(pairs["counts"])
        total_denoms += np.asarray(pairs["denominators"])
        data = {"expected_clips": len(rows), "empty_normalized_reference_ids": empty,
                "reference_sha256": hashlib.sha256(json.dumps(refs, sort_keys=True).encode()).hexdigest(),
                "coverage": coverage, "all_requested": evaluate_subset(refs, predictions, MODELS),
                "fast_requested": evaluate_subset(refs, predictions, FAST), "pairwise": pairs}
        original = {model: pred for model, pred in predictions.items()
                    if coverage[model]["provenance"] == "existing option0 sweep predictions"}
        if set(original) != set(predictions):
            data["original_existing"] = {
                "all_requested": evaluate_subset(refs, original, MODELS),
                "fast_requested": evaluate_subset(refs, original, FAST),
                "pairwise": pairwise(refs, original),
            }
        result["gates"][gate] = data
        print(gate, data["all_requested"]["clips"], data["all_requested"].get("fair_wer"), flush=True)
    result["pairwise"] = {"counts": total_counts.tolist(), "denominators": total_denoms.tolist()}
    result["aggregates"] = {
        "eight_gates": aggregate_gates(result, list(GATES)),
        "six_extraction_gates": aggregate_gates(
            result, [gate for gate in GATES if gate not in {"course91", "personal_cuda"}]),
    }
    (OUT / "combination.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "combination.md").write_text(markdown(result))
    plot_matrix(result)


if __name__ == "__main__":
    main()
