"""Out-of-fold neural ensembles and descriptive confidence/review diagnostics."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .model import IncidentClassifier, classification_metrics, softmax, tokenize
from .validation import stratified_folds


def _probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("Expected nonempty sample-by-class probabilities")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Probability rows must sum to one")
    return values


def _entropy(values: np.ndarray) -> np.ndarray:
    return -(values * np.log(np.clip(values, 1e-300, 1))).sum(axis=-1)


def ensemble_statistics(member_probabilities: np.ndarray) -> dict:
    values = np.asarray(member_probabilities, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("Expected at least two member-by-sample-by-class arrays")
    for member in values:
        _probabilities(member)
    mean = values.mean(axis=0)
    entropy = _entropy(mean)
    member_entropy = _entropy(values).mean(axis=0)
    votes = values.argmax(axis=2)
    disagreement = np.array([1 - np.bincount(votes[:, i], minlength=mean.shape[1]).max() / len(values)
                             for i in range(mean.shape[0])])
    return {"probabilities": mean, "predictive_entropy": entropy,
            "mean_member_entropy": member_entropy,
            "mutual_information": np.maximum(entropy - member_entropy, 0),
            "vote_disagreement": disagreement}


def review_curve(targets: np.ndarray, probabilities: np.ndarray, coverage: np.ndarray, *,
                 thresholds: list[float] | None = None, min_coverage: float = 0.35) -> list[dict]:
    probabilities = _probabilities(probabilities)
    targets, coverage = np.asarray(targets), np.asarray(coverage, dtype=float)
    if targets.shape != (len(probabilities),) or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Expected one integer target per sample")
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("Target out of range")
    if coverage.shape != targets.shape or not np.isfinite(coverage).all() or np.any(coverage < 0) or np.any(coverage > 1):
        raise ValueError("Expected vocabulary coverage in [0, 1] for each sample")
    thresholds = [0.0, 0.5, 0.7, 0.9, 0.99, 1.0] if thresholds is None else list(thresholds)
    if not thresholds or any(not np.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError("Expected nonempty finite thresholds in [0, 1]")
    if len(set(thresholds)) != len(thresholds) or not 0 <= min_coverage <= 1:
        raise ValueError("Expected distinct thresholds and minimum coverage in [0, 1]")
    wrong = probabilities.argmax(axis=1) != targets
    curve = []
    for threshold in sorted(thresholds):
        kept = (probabilities.max(axis=1) >= threshold) & (coverage >= min_coverage)
        count, errors = int(kept.sum()), int(wrong[kept].sum())
        curve.append({"threshold": float(threshold), "total": len(targets), "kept": count,
                      "reviewed": len(targets) - count, "coverage": count / len(targets),
                      "kept_errors": errors, "kept_error_rate": errors / count if count else None})
    return curve


def evaluate_ensemble(rows: list[dict], *, member_seeds: list[int] | None = None,
                      folds: int = 3, split_seed: int = 7, epochs: int = 180,
                      thresholds: list[float] | None = None, min_coverage: float = 0.35) -> dict:
    seeds = [7, 17, 27] if member_seeds is None else list(member_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Expected at least two distinct nonnegative integer member seeds")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    # Validate review configuration before fitting any model.
    review_curve(np.array([0]), np.array([[0.5, 0.5]]), np.array([1.0]),
                 thresholds=thresholds, min_coverage=min_coverage)
    training = [r for r in rows if r.get("split") == "train"]
    seen = set()
    for row in training:
        if not isinstance(row.get("text"), str) or not tokenize(row["text"]) or not isinstance(row.get("label"), str) or not row["label"]:
            raise ValueError("Training records require nonempty text and labels")
        normalized = " ".join(tokenize(row["text"]))
        if normalized in seen:
            raise ValueError("Duplicate normalized training text")
        seen.add(normalized)
    texts, labels = [r["text"] for r in training], [r["label"] for r in training]
    partitions = stratified_folds(labels, folds, split_seed)
    classes = sorted(set(labels))
    targets = np.array([classes.index(label) for label in labels])
    values = np.zeros((len(seeds), len(texts), len(classes)))
    coverage = np.zeros(len(texts))
    details = []
    for number, held_out in enumerate(partitions):
        fit_indices = np.setdiff1d(np.arange(len(texts)), held_out)
        fit_texts, fit_labels = [texts[i] for i in fit_indices], [labels[i] for i in fit_indices]
        for member, seed in enumerate(seeds):
            model = IncidentClassifier.from_training_data(fit_texts, fit_labels, seed + number)
            model.fit(fit_texts, fit_labels, epochs=epochs)
            features, fold_coverage = model.encode([texts[i] for i in held_out])
            values[member, held_out] = softmax(model.logits(features))
            coverage[held_out] = fold_coverage
        details.append({"fold": number + 1, "fit_indices": fit_indices.tolist(), "held_out_indices": held_out.tolist(),
                        "vocabulary_words": len(model.vocabulary)})
    statistics = ensemble_statistics(values)
    mean = statistics["probabilities"]
    digest = hashlib.sha256(json.dumps(list(zip(texts, labels)), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return {"format_version": 1, "data_scope": "Out-of-fold predictions on the training split only; synthetic research diagnostic",
            "training_records": len(texts), "training_sha256": digest, "labels": classes,
            "configuration": {"member_seeds": seeds, "folds": folds, "split_seed": split_seed, "epochs": epochs,
                              "temperature": 1.0, "learning_rate": 0.015, "min_coverage": min_coverage},
            "folds": details, "members": [{"seed": seed, "metrics": classification_metrics(targets, values[i])}
                                          for i, seed in enumerate(seeds)],
            "single_member_reference": {"seed": seeds[0], "metrics": classification_metrics(targets, values[0]),
                                        "review_curve": review_curve(targets, values[0], coverage, thresholds=thresholds, min_coverage=min_coverage)},
            "ensemble": {"metrics": classification_metrics(targets, mean),
                         "review_curve": review_curve(targets, mean, coverage, thresholds=thresholds, min_coverage=min_coverage)},
            "diagnostics": {name: {"mean": float(statistics[name].mean()), "max": float(statistics[name].max())}
                            for name in statistics if name != "probabilities"},
            "out_of_fold": {"targets": targets.tolist(), "vocabulary_coverage": coverage.tolist(),
                            "member_probabilities": values.tolist(), "ensemble_probabilities": mean.tolist()},
            "limitations": "Threshold curves are descriptive, not a selected deployment policy. Review counts are simulated; no human reviews were performed. Uncertainty scores are uncalibrated and overlapping folds are dependent."}
