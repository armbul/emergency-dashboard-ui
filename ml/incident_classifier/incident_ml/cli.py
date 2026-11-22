"""Reproducible local training, held-out evaluation, and JSON inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .model import IncidentClassifier, classification_metrics, softmax, tokenize


def read_dataset(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if not isinstance(row.get("text"), str) or not tokenize(row["text"]) or not isinstance(row.get("label"), str) or not row["label"]:
            raise ValueError("Each record requires nonempty text and label")
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Each record requires train, validation or test split")
        normalized = " ".join(tokenize(row["text"]))
        if normalized in seen:
            raise ValueError("Duplicate text detected: possible split leakage")
        seen.add(normalized)
    partitions = {split: [row for row in rows if row["split"] == split] for split in ["train", "validation", "test"]}
    if not all(partitions.values()):
        raise ValueError("All three data splits must be nonempty")
    train_labels = {row["label"] for row in partitions["train"]}
    if any(row["label"] not in train_labels for row in rows):
        raise ValueError("Validation/test labels must occur in training")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--output", default="artifacts/incident-model")
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--epochs", type=int, default=180)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--model", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--text", required=True)
    predict.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()
    if args.command == "predict":
        model = IncidentClassifier.load(args.model)
        print(json.dumps(model.predict([args.text], threshold=args.threshold)[0], indent=2))
        return
    rows = read_dataset(args.data)
    splits = {split: [row for row in rows if row["split"] == split] for split in ["train", "validation", "test"]}
    if args.command == "train":
        training = splits["train"]
        model = IncidentClassifier.from_training_data([r["text"] for r in training], [r["label"] for r in training], args.seed)
        losses = model.fit([r["text"] for r in training], [r["label"] for r in training], epochs=args.epochs)
        validation = splits["validation"]
        model.calibrate([r["text"] for r in validation], [r["label"] for r in validation])
        model.save(args.output)
    else:
        model = IncidentClassifier.load(args.model)
    testing = splits["test"]
    targets = np.array([model.labels.index(r["label"]) for r in testing])
    features, coverage = model.encode([r["text"] for r in testing])
    report = classification_metrics(targets, softmax(model.logits(features) / model.temperature))
    report.update(labels=model.labels, temperature=model.temperature, mean_vocabulary_coverage=float(coverage.mean()),
                  data_scope="Synthetic smoke-test corpus; not a real-world benchmark")
    if args.command == "train":
        report.update(seed=args.seed, epochs=args.epochs, initial_train_loss=losses[0], final_train_loss=losses[-1])
        Path(args.output, "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
