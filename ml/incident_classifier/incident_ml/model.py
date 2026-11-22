"""Mean embeddings -> two ReLU hidden layers -> temperature-scaled softmax.

All gradients and Adam updates are explicit NumPy; no pretrained weights,
external APIs, GPU, or runtime network access are required.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    """Rows are true labels; columns are predictions. Macro F1 includes all labels."""
    y = np.asarray(y, dtype=int)
    if not len(y) or probabilities.ndim != 2 or len(y) != len(probabilities):
        raise ValueError("Expected nonempty, equally sized labels and predictions")
    classes = probabilities.shape[1]
    if np.any(y < 0) or np.any(y >= classes):
        raise ValueError("Label index out of range")
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((classes, classes), dtype=int)
    np.add.at(confusion, (y, predicted), 1)
    tp = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(classes), where=denominator > 0)
    return {
        "samples": len(y),
        "accuracy": float(np.mean(predicted == y)),
        "macro_f1": float(f1.mean()),
        "negative_log_likelihood": float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "confusion_matrix": confusion.tolist(),
    }


class IncidentClassifier:
    def __init__(self, vocabulary: dict[str, int], labels: list[str], seed: int = 7):
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("At least two distinct labels are required")
        if sorted(vocabulary.values()) != list(range(1, len(vocabulary) + 1)):
            raise ValueError("Vocabulary IDs must be contiguous starting at 1")
        self.vocabulary, self.labels = dict(vocabulary), list(labels)
        self.temperature = 1.0
        rng = np.random.default_rng(seed)
        self.parameters = {
            "embedding": rng.normal(0, 0.15, (len(vocabulary) + 1, 24)),
            "w1": rng.normal(0, math.sqrt(2 / 24), (24, 48)), "b1": np.zeros(48),
            "w2": rng.normal(0, math.sqrt(2 / 48), (48, 24)), "b2": np.zeros(24),
            "w3": rng.normal(0, math.sqrt(2 / 24), (24, len(labels))), "b3": np.zeros(len(labels)),
        }
        # Zero is the unknown-token embedding; low-coverage input triggers review.
        self.parameters["embedding"][0] = 0

    @classmethod
    def from_training_data(cls, texts: list[str], labels: list[str], seed: int = 7):
        words = sorted({word for text in texts for word in tokenize(text)})
        if not words or len(texts) != len(labels):
            raise ValueError("Training requires aligned nonempty texts and labels")
        return cls({word: i + 1 for i, word in enumerate(words)}, sorted(set(labels)), seed)

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        features = np.zeros((len(texts), len(self.vocabulary) + 1))
        coverage = np.zeros(len(texts))
        for row, text in enumerate(texts):
            words = tokenize(text)
            if words:
                ids = [self.vocabulary.get(word, 0) for word in words]
                np.add.at(features[row], ids, 1 / len(ids))
                coverage[row] = sum(i != 0 for i in ids) / len(ids)
        return features, coverage

    def logits(self, features: np.ndarray) -> np.ndarray:
        p = self.parameters
        pooled = features @ p["embedding"]
        hidden1 = np.maximum(pooled @ p["w1"] + p["b1"], 0)
        hidden2 = np.maximum(hidden1 @ p["w2"] + p["b2"], 0)
        return hidden2 @ p["w3"] + p["b3"]

    def loss_and_gradients(self, features: np.ndarray, targets: np.ndarray):
        p = self.parameters
        pooled = features @ p["embedding"]
        z1 = pooled @ p["w1"] + p["b1"]
        h1 = np.maximum(z1, 0)
        z2 = h1 @ p["w2"] + p["b2"]
        h2 = np.maximum(z2, 0)
        probabilities = softmax(h2 @ p["w3"] + p["b3"])
        loss = float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1)).mean())
        delta = probabilities.copy()
        delta[np.arange(len(targets)), targets] -= 1
        delta /= len(targets)
        gradients = {"w3": h2.T @ delta, "b3": delta.sum(axis=0)}
        dz2 = (delta @ p["w3"].T) * (z2 > 0)
        gradients.update(w2=h1.T @ dz2, b2=dz2.sum(axis=0))
        dz1 = (dz2 @ p["w2"].T) * (z1 > 0)
        gradients.update(w1=pooled.T @ dz1, b1=dz1.sum(axis=0))
        gradients["embedding"] = features.T @ (dz1 @ p["w1"].T)
        gradients["embedding"][0] = 0
        return loss, gradients

    def fit(self, texts: list[str], labels: list[str], epochs: int = 180, learning_rate: float = 0.015) -> list[float]:
        if not texts or len(texts) != len(labels) or epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Expected aligned training data, positive epochs and learning rate")
        if any(label not in self.labels for label in labels):
            raise ValueError("Unknown training label")
        features, _ = self.encode(texts)
        targets = np.array([self.labels.index(label) for label in labels])
        moments = {k: np.zeros_like(v) for k, v in self.parameters.items()}
        variances = {k: np.zeros_like(v) for k, v in self.parameters.items()}
        losses = []
        self.temperature = 1.0
        for step in range(1, epochs + 1):
            loss, gradients = self.loss_and_gradients(features, targets)
            losses.append(loss)
            for name, gradient in gradients.items():
                moments[name] = 0.9 * moments[name] + 0.1 * gradient
                variances[name] = 0.999 * variances[name] + 0.001 * gradient ** 2
                m = moments[name] / (1 - 0.9 ** step)
                v = variances[name] / (1 - 0.999 ** step)
                self.parameters[name] -= learning_rate * m / (np.sqrt(v) + 1e-8)
        return losses

    def calibrate(self, texts: list[str], labels: list[str]) -> float:
        """Choose temperature by NLL on separate validation examples, never test data."""
        if not texts or len(texts) != len(labels) or any(label not in self.labels for label in labels):
            raise ValueError("Expected aligned validation data with known labels")
        features, _ = self.encode(texts)
        scores = self.logits(features)
        targets = np.array([self.labels.index(label) for label in labels])
        temperatures = np.geomspace(0.5, 5.0, 50)
        losses = [classification_metrics(targets, softmax(scores / t))["negative_log_likelihood"] for t in temperatures]
        self.temperature = float(temperatures[int(np.argmin(losses))])
        return self.temperature

    def predict(self, texts: list[str], threshold: float = 0.7, min_coverage: float = 0.35) -> list[dict]:
        if not 0 <= threshold <= 1 or not 0 <= min_coverage <= 1:
            raise ValueError("Thresholds must be in [0, 1]")
        if not texts:
            return []
        features, coverage = self.encode(texts)
        probabilities = softmax(self.logits(features) / self.temperature)
        results = []
        for i, probability in enumerate(probabilities):
            reasons = []
            if not tokenize(texts[i]):
                reasons.append("empty_input")
            if coverage[i] < min_coverage:
                reasons.append("low_vocabulary_coverage")
            if probability.max() < threshold:
                reasons.append("low_confidence")
            results.append({
                "suggested_label": self.labels[int(probability.argmax())],
                "confidence": float(probability.max()),
                "probabilities": dict(zip(self.labels, map(float, probability))),
                "vocabulary_coverage": float(coverage[i]),
                "requires_review": bool(reasons), "review_reasons": reasons,
            })
        return results

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {"format_version": 1, "vocabulary": self.vocabulary, "labels": self.labels, "temperature": self.temperature}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.savez_compressed(directory / "weights.npz", **self.parameters)

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported model format")
        model = cls(metadata["vocabulary"], metadata["labels"])
        temperature = float(metadata["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid model temperature")
        with np.load(directory / "weights.npz", allow_pickle=False) as archive:
            for name, default in model.parameters.items():
                weights = archive[name]
                if weights.shape != default.shape or not np.isfinite(weights).all():
                    raise ValueError(f"Invalid weights: {name}")
                model.parameters[name] = weights.copy()
        model.temperature = temperature
        return model
