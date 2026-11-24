"""Training-only stratified comparison of the neural model and multinomial NB."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .model import IncidentClassifier, classification_metrics, softmax, tokenize


class MultinomialTextBaseline:
    """Token-count naive Bayes with additive smoothing and empirical priors."""

    def __init__(self, texts: list[str], labels: list[str], alpha: float = 1.0):
        if not texts or len(texts) != len(labels) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Expected aligned data and positive finite alpha")
        self.labels = sorted(set(labels))
        self.vocabulary = {word: i for i, word in enumerate(sorted({w for t in texts for w in tokenize(t)}))}
        if len(self.labels) < 2 or not self.vocabulary:
            raise ValueError("Expected at least two classes and a nonempty vocabulary")
        counts = np.full((len(self.labels), len(self.vocabulary)), alpha)
        targets = np.array([self.labels.index(label) for label in labels])
        for text, target in zip(texts, targets):
            for word in tokenize(text):
                counts[target, self.vocabulary[word]] += 1
        self.log_likelihood = np.log(counts / counts.sum(axis=1, keepdims=True))
        self.log_prior = np.log(np.bincount(targets, minlength=len(self.labels)) / len(labels))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        scores = np.tile(self.log_prior, (len(texts), 1))
        for row, text in enumerate(texts):
            for word in tokenize(text):
                if word in self.vocabulary:
                    scores[row] += self.log_likelihood[:, self.vocabulary[word]]
        return softmax(scores) if texts else np.empty((0, len(self.labels)))


def stratified_folds(labels: list[str], folds: int = 3, seed: int = 7) -> list[np.ndarray]:
    if type(folds) is not int or folds < 2:
        raise ValueError("folds must be an integer of at least 2")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError("Cross-validation requires at least two classes")
    rng = np.random.default_rng(seed)
    partitions = [[] for _ in range(folds)]
    for label in classes:
        indices = np.array([i for i, value in enumerate(labels) if value == label])
        if len(indices) < folds:
            raise ValueError("Each class must have at least folds training records")
        rng.shuffle(indices)
        for partition, chunk in zip(partitions, np.array_split(indices, folds)):
            partition.extend(map(int, chunk))
    return [np.array(sorted(partition), dtype=int) for partition in partitions]


def cross_validate(rows: list[dict], *, folds: int = 3, seed: int = 7,
                   epochs: int = 180, alpha: float = 1.0) -> dict:
    # Select first: validation and test text/labels never enter fitting or scoring.
    training = [row for row in rows if row.get("split") == "train"]
    if type(epochs) is not int or epochs < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Expected positive integer epochs and positive finite alpha")
    seen = set()
    for row in training:
        if not isinstance(row.get("text"), str) or not tokenize(row["text"]) or not isinstance(row.get("label"), str) or not row["label"]:
            raise ValueError("Training records require nonempty text and labels")
        normalized = " ".join(tokenize(row["text"]))
        if normalized in seen:
            raise ValueError("Duplicate normalized training text")
        seen.add(normalized)
    texts = [row["text"] for row in training]
    labels = [row["label"] for row in training]
    partitions = stratified_folds(labels, folds, seed)
    classes = sorted(set(labels))
    targets = np.array([classes.index(label) for label in labels])
    probabilities = {name: np.zeros((len(training), len(classes))) for name in ("neural", "multinomial_nb")}
    details = []
    for number, held_out in enumerate(partitions):
        fit_indices = np.setdiff1d(np.arange(len(training)), held_out)
        fit_texts, fit_labels = [texts[i] for i in fit_indices], [labels[i] for i in fit_indices]
        held_texts = [texts[i] for i in held_out]
        neural = IncidentClassifier.from_training_data(fit_texts, fit_labels, seed + number)
        neural.fit(fit_texts, fit_labels, epochs=epochs)
        features, coverage = neural.encode(held_texts)
        baseline = MultinomialTextBaseline(fit_texts, fit_labels, alpha)
        scores = {"neural": softmax(neural.logits(features)), "multinomial_nb": baseline.predict_proba(held_texts)}
        detail = {"fold": number + 1, "fit_indices": fit_indices.tolist(), "held_out_indices": held_out.tolist(),
                  "vocabulary_words": len(neural.vocabulary), "mean_neural_vocabulary_coverage": float(coverage.mean()), "metrics": {}}
        for name, values in scores.items():
            probabilities[name][held_out] = values
            detail["metrics"][name] = classification_metrics(targets[held_out], values)
        details.append(detail)
    summaries = {}
    for name, values in probabilities.items():
        summaries[name] = {"out_of_fold": classification_metrics(targets, values),
                           "mean_fold_accuracy": float(np.mean([d["metrics"][name]["accuracy"] for d in details])),
                           "fold_accuracy_std": float(np.std([d["metrics"][name]["accuracy"] for d in details])),
                           "out_of_fold_probabilities": values.tolist()}
    digest = hashlib.sha256(json.dumps(list(zip(texts, labels)), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return {"format_version": 1, "data_scope": "Training split only; synthetic corpus results are not deployment evidence",
            "training_records": len(training), "training_sha256": digest, "labels": classes,
            "configuration": {"folds": folds, "seed": seed, "epochs": epochs, "alpha": alpha,
                              "neural_temperature": 1.0, "neural_learning_rate": 0.015},
            "folds": details, "models": summaries}
