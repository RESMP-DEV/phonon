"""CPU-only grouped word-span probes and unsupervised UMAP diagnostics.

Run only after releasing the extraction GPU lease. Gate labels never update an
ASR encoder. Eval-only Aqua/Wispr labels are excluded from every fitted probe.
"""
from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import sys
import warnings

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jiwer  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/option0"))
from gates import load_gate  # noqa: E402
from score_gates import NORMALIZE  # noqa: E402

BASE = "parakeet-tdt-0.6b-v2"
TAGS = [BASE, "parakeet-unified-en-0.6b", "Qwen3-ASR-0.6B-hf", "cohere-transcribe-03-2026",
        "granite-speech-5.0-470m-turboctc", "granite-speech-4.1-2b"]
GATES = ["novel180", "hard77", "uncertain48", "wispr_holdout120", "aqua_new_holdout",
         "wispr_edit25"]
FIT_GATES = {"novel180", "hard77", "uncertain48"}
SEED = 20260915
plt.style.use("dark_background")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalized_timestamps(hypothesis, timestamps):
    """Track suffix normalizer rewrites, retaining only unambiguous time unions.

    A numerical phrase can collapse several original words to one normalized
    token. Prefix normalization carries the union of those word timestamps.
    Rewrites yielding multiple tokens without equal matches are discarded.
    No duration is synthesized from transcript word positions.
    """
    old, origins, surface = [], [], []
    for index, row in enumerate(timestamps):
        surface.append(str(row.get("word", row.get("text", ""))))
        new = NORMALIZE(" ".join(surface)).split()
        updated = [None] * len(new)
        for operation, a, b, c, d in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
            if operation == "equal":
                updated[c:d] = origins[a:b]
            elif operation in {"insert", "replace"}:
                prior = set().union(*(value or set() for value in origins[a:b]))
                if d - c == 1 and all(value is not None for value in origins[a:b]):
                    updated[c] = prior | {index}
                elif operation == "insert":
                    # A single spoken word can expand (e.g. "can't" -> "can not").
                    updated[c:d] = [{index} for _ in range(d - c)]
        old, origins = new, updated
    target = NORMALIZE(hypothesis).split()
    mapped = [None] * len(target)
    for operation, a, b, c, d in SequenceMatcher(None, old, target, autojunk=False).get_opcodes():
        if operation != "equal":
            continue
        for src, dst in zip(range(a, b), range(c, d), strict=True):
            indices = origins[src]
            if not indices:
                continue
            try:
                bounds = [(float(timestamps[k]["start"]), float(timestamps[k]["end"]))
                          for k in indices]
            except (ValueError, KeyError, TypeError):
                continue
            if any(not np.isfinite(s + e) or s < 0 or e <= s for s, e in bounds):
                continue
            mapped[dst] = (min(s for s, _ in bounds), max(e for _, e in bounds))
    return target, mapped


def labels_for_words(reference, hypothesis, terms):
    ref, hyp = NORMALIZE(reference).split(), NORMALIZE(hypothesis).split()
    term_ref = np.zeros(len(ref), dtype=np.int8)
    phrase_matches = 0
    for phrase in terms:
        tokens = NORMALIZE(str(phrase)).split()
        if tokens:
            for start in range(len(ref) - len(tokens) + 1):
                if ref[start:start + len(tokens)] == tokens:
                    term_ref[start:start + len(tokens)] = 1
                    phrase_matches += 1
    error = np.ones(len(hyp), dtype=np.int8)
    term = np.full(len(hyp), -1, dtype=np.int8)
    aligned = jiwer.process_words(" ".join(ref), " ".join(hyp))
    for chunk in aligned.alignments[0]:
        r = slice(chunk.ref_start_idx, chunk.ref_end_idx)
        h = slice(chunk.hyp_start_idx, chunk.hyp_end_idx)
        if chunk.type == "equal":
            error[h] = 0
            term[h] = term_ref[r]
        elif chunk.type == "substitute":
            ref_labels = term_ref[r]
            # jiwer has arbitrary one-to-many substitutions. Assign only when
            # every reference word in the aligned block agrees on term status.
            if len(ref_labels) and np.all(ref_labels == ref_labels[0]):
                term[h] = ref_labels[0]
    return error, term, {"deletions_without_hypothesis_span": aligned.deletions,
                         "reference_words": len(ref), "term_phrase_matches": phrase_matches}


def build_records(cache):
    records, stats = [], {}
    for gate in GATES:
        counter = Counter()
        try:
            rows = load_gate(gate)
        except Exception as exc:
            stats[gate] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        for row in rows:
            sidecar = cache / BASE / gate / f'{row["id"]}.json'
            if not sidecar.exists():
                counter["clips_missing_parakeet"] += 1
                continue
            meta = json.loads(sidecar.read_text())
            hypothesis = meta.get("hypothesis", "")
            timestamps = meta.get("word_timestamps", [])
            words, spans = normalized_timestamps(hypothesis, timestamps)
            errors, terms, extra = labels_for_words(row["reference"], hypothesis,
                                                    row.get("technical_terms") or [])
            counter.update(extra)
            counter["clips"] += 1
            counter["hypothesis_words"] += len(words)
            counter["raw_word_timestamps"] += len(timestamps)
            counter["raw_zero_duration_timestamps"] += sum(
                word.get("start") == word.get("end") for word in timestamps)
            counter["clips_with_nonempty_term_spans"] += bool(row.get("term_spans"))
            source = row.get("source_id") or row.get("video_id")
            if not source and row["id"].startswith("youtube:"):
                source = row["id"].split(":")[1]
            group = f"youtube:{source}" if source and gate in FIT_GATES else f'{gate}:{row["id"]}'
            allowed = (gate in FIT_GATES and row.get("train_allowed") is not False
                       and "hidden" not in str(row.get("split", "")).lower())
            for index, (word, span) in enumerate(zip(words, spans, strict=True)):
                if span is None or span[0] >= float(meta["duration_seconds"]):
                    counter["dropped_invalid_or_ambiguous_span"] += 1
                    continue
                end = min(span[1], float(meta["duration_seconds"]))
                if end <= span[0]:
                    counter["dropped_invalid_or_ambiguous_span"] += 1
                    continue
                records.append({"gate": gate, "id": row["id"], "word_index": index,
                                "word": word, "start": span[0], "end": end,
                                "error": int(errors[index]),
                                "term": int(terms[index]) if gate in FIT_GATES else -1,
                                "group": group, "fit_allowed": allowed})
                counter["valid_spans"] += 1
                counter["error_spans"] += int(errors[index])
                counter["term_spans"] += int(terms[index] == 1 and gate in FIT_GATES)
        stats[gate] = dict(counter)
    return records, stats


def pool_features(cache, tag, records, destination):
    from geometry import align_frames

    grouped = {}
    for index, record in enumerate(records):
        grouped.setdefault((record["gate"], record["id"]), []).append(index)
    output, support = None, np.zeros(len(records), dtype=bool)
    failures = []
    for (gate, clip_id), indices in grouped.items():
        path = cache / tag / gate / f"{clip_id}.npy"
        if not path.exists() or not path.with_suffix(".json").exists():
            continue
        try:
            meta = json.loads(path.with_suffix(".json").read_text())
            array = align_frames(np.load(path, mmap_mode="r"), meta, target_fps=12.5)
            if isinstance(array, tuple):
                array = array[0]
            if output is None:
                output = np.lib.format.open_memmap(destination, mode="w+", dtype="float32",
                                                   shape=(len(records), array.shape[1]))
                output[:] = np.nan
            for index in indices:
                record = records[index]
                left = max(0, int(np.floor(record["start"] * 12.5)))
                right = min(len(array), int(np.ceil(record["end"] * 12.5)))
                if right <= left:
                    continue
                vector = np.asarray(array[left:right], dtype=np.float32).mean(axis=0)
                if np.isfinite(vector).all():
                    output[index] = vector
                    support[index] = True
        except Exception as exc:
            failures.append({"gate": gate, "id": clip_id, "error": f"{type(exc).__name__}: {exc}"})
    if output is not None:
        output.flush()
    return output, support, failures


def metrics(y, probability):
    valid = np.isfinite(probability)
    y, probability = y[valid], probability[valid]
    if not len(y):
        return {"n": 0, "positive": 0, "prevalence": None, "auroc": None,
                "average_precision": None}
    two_classes = len(np.unique(y)) == 2
    return {"n": int(len(y)), "positive": int(y.sum()), "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, probability)) if two_classes else None,
            "average_precision": float(average_precision_score(y, probability))
            if two_classes else None}


def report_metrics(y, probability, records, folds):
    gates = np.array([row["gate"] for row in records])
    allowed = np.array([row["fit_allowed"] for row in records])
    result = {"pooled": metrics(y, probability),
              "youtube_grouped_oof": metrics(y[allowed], probability[allowed]),
              "external_holdout": metrics(y[~allowed], probability[~allowed]),
              "gates": {gate: metrics(y[gates == gate], probability[gates == gate])
                        for gate in GATES},
              "folds": {str(fold): metrics(y[folds == fold], probability[folds == fold])
                        for fold in range(5)}}
    return result


def fit_predict(features, y, support, allowed, folds):
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    external = support & ~allowed & (y >= 0)
    external_predictions = []
    diagnostics = []
    for fold in range(5):
        train = support & allowed & (folds != fold) & (y >= 0)
        valid = support & allowed & (folds == fold) & (y >= 0)
        if len(np.unique(y[train])) < 2 or not valid.any():
            diagnostics.append({"fold": fold, "skipped": "one-class training or empty validation"})
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=600, random_state=SEED))
        with warnings.catch_warnings(record=True) as caught:
            model.fit(np.asarray(features[train]), y[train])
        prediction[valid] = model.predict_proba(np.asarray(features[valid]))[:, 1]
        if external.any():
            external_predictions.append(model.predict_proba(np.asarray(features[external]))[:, 1])
        diagnostics.append({"fold": fold, "train_words": int(train.sum()),
                            "validation_words": int(valid.sum()),
                            "iterations": int(model[-1].n_iter_[0]),
                            "warnings": [str(item.message) for item in caught]})
    if len(external_predictions) == 5:
        prediction[external] = np.mean(external_predictions, axis=0)
    return prediction, diagnostics


def bootstrap_gain(y, base, partner, groups, replicates=200):
    valid = np.isfinite(base) & np.isfinite(partner) & (y >= 0)
    y, base, partner, groups = y[valid], base[valid], partner[valid], groups[valid]
    if len(np.unique(y)) < 2:
        return {"gain": None, "ci95": None, "n": int(len(y))}
    gain = float(roc_auc_score(y, partner) - roc_auc_score(y, base))
    names = np.unique(groups)
    positions = [np.flatnonzero(groups == group) for group in names]
    values = []
    rng = np.random.default_rng(SEED)
    for _ in range(replicates):
        sampled = np.concatenate([positions[k] for k in rng.integers(0, len(names), len(names))])
        if len(np.unique(y[sampled])) == 2:
            values.append(roc_auc_score(y[sampled], partner[sampled])
                          - roc_auc_score(y[sampled], base[sampled]))
    return {"gain": gain, "ci95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
            "n": int(len(y)), "groups": int(len(names)), "bootstrap_replicates": len(values),
            "base_auroc": float(roc_auc_score(y, base)),
            "concat_auroc": float(roc_auc_score(y, partner))}


def probe_plot(results, task, output):
    if task == "error":
        fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
    display_names = {BASE: "Parakeet v2", "parakeet-unified-en-0.6b": "Unified",
                     "Qwen3-ASR-0.6B-hf": "Qwen 0.6B", "cohere-transcribe-03-2026": "Cohere",
                     "granite-speech-5.0-470m-turboctc": "Granite 5.0",
                     "granite-speech-4.1-2b": "Granite 4.1"}
    scope_names = {"youtube_grouped_oof": "Held-out YouTube videos",
                   "external_holdout": "Aqua/Wispr (evaluation only)"}
    tags = list(results[task])
    scopes = ["youtube_grouped_oof", "external_holdout"] if task == "error" else ["youtube_grouped_oof"]
    width = 0.34
    for k, scope in enumerate(scopes):
        values = [results[task][tag][scope]["auroc"] for tag in tags]
        numeric = [value if value is not None else np.nan for value in values]
        positions = np.arange(len(tags)) + (k - (len(scopes) - 1) / 2) * width
        bars = ax.bar(positions, numeric, width, label=scope_names[scope])
        for bar, value in zip(bars, values, strict=True):
            if value is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                        ha="center", fontsize=9)
    if not tags:
        ax.text(0.5, 0.5, "No valid timestamped spans available", transform=ax.transAxes, ha="center")
    ax.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(tags)), [display_names.get(tag, tag) for tag in tags], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC (0.5 is chance)")
    ax.set_title("Erroneous hypothesis word from span features" if task == "error"
                 else "Derived annotated-term membership from span features")
    ax.legend(loc="lower left")
    if task == "error":
        partners = list(results["residual"])
        for k, scope in enumerate(scopes):
            for index, tag in enumerate(partners):
                row = results["residual"][tag][scope]
                value, interval = row["gain"], row["ci95"]
                if value is None:
                    continue
                position = index + (k - 0.5) * 0.20
                color = ["#55aaff", "#ffb85c"][k]
                residual_ax.scatter(position, value, color=color, s=35,
                                    label=scope_names[scope] if index == 0 else None)
                if interval is not None:
                    residual_ax.vlines(position, interval[0], interval[1], color=color)
                residual_ax.annotate(f"{value:+.3f}", (position, value),
                                     xytext=(4, 6), textcoords="offset points", fontsize=8)
        residual_ax.axhline(0, color="#aaaaaa", linestyle="--", linewidth=1)
        residual_ax.set_xticks(np.arange(len(partners)),
                              [display_names.get(tag, tag) for tag in partners], fontsize=8)
        residual_ax.set_ylabel("AUROC gain from concatenating partner")
        residual_ax.set_title("Additional information beyond Parakeet: exact paired support; "
                              "95% grouped bootstrap intervals")
        if partners:
            residual_ax.legend(fontsize=8)
    fig.text(0.5, 0.015, "5 source-grouped folds; scaling fitted on training only; "
             "Aqua/Wispr labels never used to fit probes", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output / f"probe_{task}_auroc.png", dpi=160)
    plt.close(fig)


def umap_plots(records, features, supports, residual, output, cache):
    from umap import UMAP

    ranked = sorted((tag for tag in residual if residual[tag]["youtube_grouped_oof"]["gain"]
                     is not None),
                    key=lambda tag: residual[tag]["youtube_grouped_oof"]["gain"], reverse=True)
    chosen = [BASE, *ranked[:2]] if BASE in features else []
    if not chosen:
        return {"status": "no features"}
    common = np.logical_and.reduce([supports[tag] for tag in chosen])
    candidates = np.flatnonzero(common)
    rng = np.random.default_rng(SEED)
    selected = np.sort(rng.choice(candidates, min(5000, len(candidates)), replace=False))
    labels = {name: np.array([row[name] for row in records])[selected] for name in ("error", "term")}
    output_data = {"encoders": chosen, "points": int(len(selected)),
                   "selection": "top two source-held-out residual AUROC gains; descriptive selection",
                   "random_state": SEED, "pca_components": 50}
    if len(selected) < 20:
        output_data["status"] = "too few common spans"
        return output_data
    np.save(cache / "umap_record_indices.npy", selected)
    for tag in chosen:
        array = np.asarray(features[tag][selected], dtype=np.float32)
        reduced = PCA(n_components=min(50, *array.shape), random_state=SEED).fit_transform(array)
        coordinates = UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=SEED,
                           n_jobs=1).fit_transform(reduced)
        np.save(cache / f"umap_{tag}.npy", coordinates)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, name in zip(axes, ("error", "term"), strict=True):
            classes = [(0, "correct" if name == "error" else "unannotated", "#55aaff"),
                       (1, "error" if name == "error" else "annotated term", "#ff6e70")]
            if name == "term":
                classes.insert(0, (-1, "unknown / external holdout", "#555555"))
            for value, label, color in classes:
                mask = labels[name] == value
                ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=5, alpha=0.55,
                           c=color, label=f"{label} ({int(mask.sum())})", rasterized=True)
            ax.set_title("Parakeet error labels" if name == "error" else "Derived term labels")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(fontsize=8, markerscale=2)
        fig.suptitle(f"{tag}: {len(selected)} identical sampled word spans")
        fig.text(0.5, 0.02, "PCA(50) then UMAP; neither reduction uses labels; visual clusters "
                 "do not establish held-out prediction quality", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.savefig(output / f"umap_{tag}.png", dpi=160)
        plt.close(fig)
    return output_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/data/phonon_latent_v0"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--skip-umap", action="store_true")
    args = parser.parse_args()
    span_cache = args.cache / "probes"
    span_cache.mkdir(parents=True, exist_ok=True)
    records, span_stats = build_records(args.cache)
    save_json(span_cache / "span_records.json", records)
    results = {"span_statistics": span_stats, "error": {}, "term": {}, "residual": {},
               "methods": {"normalizer": "scripts/option0/score_gates.py NORMALIZE",
                           "features": "mean of aligned 12.5 fps frames overlapping timestamp span",
                           "classifier": "StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=600)",
                           "cv": "5 StratifiedGroupKFold folds by global source_id, fixed seed",
                           "interpretation": "diagnostic error detection, not error correction; "
                           "concatenation adds dimensions and has no equal-capacity control; "
                           "encoder ranking is selected on these same diagnostic folds",
                           "external": "Aqua/Wispr excluded from all fitting; average of 5 fold models",
                           "term_labels": "derived normalized reference phrase matches of technical_terms; "
                           "mapped through jiwer alignment; insertions and mixed-label substitution "
                           "blocks excluded; these are annotation membership, not exhaustive term truth",
                           "error_labels": "substitutions and insertions erroneous; deletions have no "
                           "hypothesis span and are not represented",
                           "timestamp_policy": "real timestamp unions for normalization merges; "
                           "zero-duration, invalid and ambiguous spans excluded; no uniform fallback",
                           "bootstrap": "200 source/clip-grouped paired resamples of fixed predictions; "
                           "does not capture probe refitting or model-selection uncertainty",
                           "umap": "same uniform sample of at most 5000 common spans; PCA50 then "
                           "UMAP cosine; no labels supplied to reduction"}}
    if not records:
        save_json(args.output / "probe_results.json", results)
        for task in ("error", "term"):
            probe_plot(results, task, args.output)
        print("No valid spans; recorded unavailable probes", flush=True)
        return
    labels = {name: np.array([row[name] for row in records]) for name in ("error", "term")}
    allowed = np.array([row["fit_allowed"] for row in records])
    groups = np.array([row["group"] for row in records])
    gates = np.array([row["gate"] for row in records])
    folds = np.full(len(records), -1, dtype=np.int8)
    train_indices = np.flatnonzero(allowed)
    if len(np.unique(groups[allowed])) < 5:
        raise ValueError("Fewer than five permitted source groups; cannot fit requested probes")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, heldout) in enumerate(splitter.split(train_indices, labels["error"][allowed],
                                                      groups[allowed])):
        folds[train_indices[heldout]] = fold
    np.save(span_cache / "fold_assignments.npy", folds)
    results["fold_groups"] = {str(fold): sorted(set(groups[folds == fold])) for fold in range(5)}
    features, supports, baseline_predictions = {}, {}, {}
    for tag in TAGS:
        print(f"Pooling {tag}", flush=True)
        array, support, failures = pool_features(args.cache, tag, records, span_cache / f"{tag}.npy")
        if array is None:
            results.setdefault("unavailable", {})[tag] = failures or ["no extracted features"]
            continue
        features[tag], supports[tag] = array, support
        np.save(span_cache / f"{tag}.support.npy", support)
        for task, y in labels.items():
            print(f"Fitting {tag} {task}: {int((support & (y >= 0)).sum())} spans", flush=True)
            probability, diagnostics = fit_predict(array, y, support, allowed, folds)
            np.save(span_cache / f"{tag}.{task}.predictions.npy", probability)
            results[task][tag] = {**report_metrics(y, probability, records, folds),
                                  "fit_diagnostics": diagnostics, "feature_failures": failures}
            if tag == BASE:
                baseline_predictions[task] = probability
        save_json(args.output / "probe_results.json", results)
    if BASE in features:
        for tag in features:
            if tag == BASE:
                continue
            support = supports[BASE] & supports[tag]
            print(f"Fitting paired residual {tag}: {int(support.sum())} spans", flush=True)
            base_probability = baseline_predictions["error"]
            if not np.array_equal(support, supports[BASE]):
                base_probability, _ = fit_predict(features[BASE], labels["error"], support,
                                                   allowed, folds)
            combined = np.concatenate((features[BASE], features[tag]), axis=1)
            probability, diagnostics = fit_predict(combined, labels["error"], support, allowed, folds)
            del combined
            np.save(span_cache / f"{tag}.paired_base_predictions.npy", base_probability)
            np.save(span_cache / f"{tag}.concat_predictions.npy", probability)
            result = {"metrics": report_metrics(labels["error"], probability, records, folds),
                      "fit_diagnostics": diagnostics}
            for scope, mask in [("pooled", np.ones(len(records), dtype=bool)),
                                ("youtube_grouped_oof", allowed), ("external_holdout", ~allowed),
                                *[(gate, gates == gate) for gate in GATES]]:
                result[scope] = bootstrap_gain(labels["error"][mask], base_probability[mask],
                                                probability[mask], groups[mask])
            results["residual"][tag] = result
            save_json(args.output / "probe_results.json", results)
    for task in ("error", "term"):
        probe_plot(results, task, args.output)
    if not args.skip_umap:
        results["umap"] = umap_plots(records, features, supports, results["residual"],
                                     args.output, span_cache)
    save_json(args.output / "probe_results.json", results)
    print(json.dumps({"error": {tag: row["youtube_grouped_oof"] for tag, row in
                                results["error"].items()},
                      "residual": {tag: row["youtube_grouped_oof"] for tag, row in
                                   results["residual"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
    with threadpool_limits(limits=4):
        main()
