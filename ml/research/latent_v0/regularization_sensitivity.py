"""Exploratory regularization sensitivity after observing primary C=1 losses.

All requested strengths are reported. This does not select a hyperparameter,
replace the primary analysis, or fit any Aqua/Wispr evaluation labels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import warnings

from probes import (
    BASE,
    SEED,
    TAGS,
    LogisticRegression,
    StandardScaler,
    bootstrap_gain,
    make_pipeline,
    np,
    report_metrics,
    save_json,
    threadpool_limits,
)


def fit_predict(features, y, support, allowed, folds, strength):
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    external = support & ~allowed & (y >= 0)
    external_predictions, diagnostics = [], []
    for fold in range(5):
        train = support & allowed & (folds != fold) & (y >= 0)
        valid = support & allowed & (folds == fold) & (y >= 0)
        if len(np.unique(y[train])) < 2 or not valid.any():
            diagnostics.append({"fold": fold, "skipped": "one-class training or empty validation"})
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=strength, solver="lbfgs", max_iter=600, random_state=SEED))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/data/phonon_latent_v0/probes"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    records = json.loads((args.cache / "span_records.json").read_text())
    primary = json.loads((args.output / "probe_results.json").read_text())
    y = np.array([row["error"] for row in records])
    allowed = np.array([row["fit_allowed"] for row in records])
    groups = np.array([row["group"] for row in records])
    folds = np.load(args.cache / "fold_assignments.npy")
    features = {tag: np.load(args.cache / f"{tag}.npy", mmap_mode="r") for tag in TAGS
                if (args.cache / f"{tag}.npy").exists()}
    supports = {tag: np.load(args.cache / f"{tag}.support.npy") for tag in features}
    results = {
        "status": "running",
        "methods": {
            "design": "exploratory sensitivity requested after observing the primary C=1 "
            "unified concatenation loss; not a preregistered comparison",
            "strengths": [1.0, 0.01, 0.001],
            "interpretation": "smaller C means stronger L2 regularization; every C is reported, "
            "none selected and primary results unchanged",
            "protocol": "identical cached word vectors, primary source folds, train-only "
            "StandardScaler, lbfgs max_iter=600, seed20260915; C=1 probabilities reused",
            "external": "Aqua/Wispr labels never enter any fit or hyperparameter selection; "
            "predictions average the five source-fold models",
            "bootstrap": "200 paired source/clip-grouped resamples of fixed probabilities "
            "for source-held-out and external scopes; no refitting uncertainty",
            "capacity": "concatenation increases dimensions; no equal-capacity control",
        },
        "by_C": {},
    }
    path = args.output / "regularization_sensitivity.json"
    for strength in (1.0, 0.01, 0.001):
        name = str(strength)
        location = args.cache / "regularization_sensitivity" / f"C{name}"
        location.mkdir(parents=True, exist_ok=True)
        result = {"solo": {}, "residual": {}}
        results["by_C"][name] = result
        predictions = {}
        for tag, array in features.items():
            print(f"C={strength} solo {tag}", flush=True)
            if strength == 1.0:
                probability = np.load(args.cache / f"{tag}.error.predictions.npy")
                diagnostics = primary["error"][tag]["fit_diagnostics"]
            else:
                probability, diagnostics = fit_predict(array, y, supports[tag], allowed,
                                                        folds, strength)
                np.save(location / f"{tag}.solo.npy", probability)
            predictions[tag] = probability
            result["solo"][tag] = {**report_metrics(y, probability, records, folds),
                                    "fit_diagnostics": diagnostics}
            save_json(path, results)
        for tag in features:
            if tag == BASE:
                continue
            print(f"C={strength} concatenation {tag}", flush=True)
            support = supports[BASE] & supports[tag]
            if strength == 1.0:
                base_probability = np.load(args.cache / f"{tag}.paired_base_predictions.npy")
                probability = np.load(args.cache / f"{tag}.concat_predictions.npy")
                diagnostics = primary["residual"][tag]["fit_diagnostics"]
            else:
                base_probability = predictions[BASE]
                if not np.array_equal(support, supports[BASE]):
                    base_probability, _ = fit_predict(features[BASE], y, support, allowed,
                                                        folds, strength)
                combined = np.concatenate((features[BASE], features[tag]), axis=1)
                probability, diagnostics = fit_predict(combined, y, support, allowed,
                                                        folds, strength)
                del combined
                np.save(location / f"{tag}.paired_base.npy", base_probability)
                np.save(location / f"{tag}.concat.npy", probability)
            entry = {"metrics": report_metrics(y, probability, records, folds),
                     "fit_diagnostics": diagnostics}
            for scope, mask in (("youtube_grouped_oof", allowed), ("external_holdout", ~allowed)):
                entry[scope] = (primary["residual"][tag][scope] if strength == 1.0 else
                                bootstrap_gain(y[mask], base_probability[mask], probability[mask],
                                               groups[mask]))
            result["residual"][tag] = entry
            save_json(path, results)
    results["status"] = "complete"
    save_json(path, results)
    print(json.dumps({c: {tag: {scope: entry[scope]["gain"]
                               for scope in ("youtube_grouped_oof", "external_holdout")}
                         for tag, entry in row["residual"].items()}
                      for c, row in results["by_C"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
    with threadpool_limits(limits=4):
        main()
