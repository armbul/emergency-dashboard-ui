"""Reproducible local training, held-out evaluation, and JSON inference."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .model import IncidentClassifier, classification_metrics, softmax, tokenize
from .batch import predict_jsonl
from .validation import cross_validate


def batch_predict(args, parser):
    """Keep predictions on the selected output and a compact summary on stderr."""
    try:
        if args.input != "-" and args.output != "-":
            source, destination = Path(args.input), Path(args.output)
            if source.resolve() == destination.resolve() or (destination.exists() and source.samefile(destination)):
                raise ValueError("Input and output must be different files")
        if args.batch_size < 1 or not 0 <= args.threshold <= 1 or not 0 <= args.min_coverage <= 1:
            raise ValueError("Expected positive batch size and thresholds in [0, 1]")
        model = IncidentClassifier.load(args.model)
        counts = {"records": 0, "succeeded": 0, "failed": 0}
        with ExitStack() as stack:
            source = sys.stdin if args.input == "-" else stack.enter_context(open(args.input, encoding="utf-8"))
            output = sys.stdout if args.output == "-" else stack.enter_context(open(args.output, "w", encoding="utf-8"))
            for result in predict_jsonl(source, model, batch_size=args.batch_size,
                                        threshold=args.threshold, min_coverage=args.min_coverage):
                output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                counts["records"] += 1
                counts["succeeded" if result["status"] == "ok" else "failed"] += 1
        print(json.dumps(counts), file=sys.stderr)
        return 1 if counts["failed"] else 0
    except (OSError, ValueError) as error:
        parser.exit(2, f"batch-predict: {error}\n")


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
    batch = commands.add_parser("batch-predict", help="Predict JSONL with one result per input line")
    batch.add_argument("--model", required=True)
    batch.add_argument("--input", required=True, help="UTF-8 JSONL path, or - for stdin")
    batch.add_argument("--output", default="-", help="Output JSONL path, or - for stdout (default)")
    batch.add_argument("--batch-size", type=int, default=128)
    batch.add_argument("--threshold", type=float, default=0.7)
    batch.add_argument("--min-coverage", type=float, default=0.35)
    comparison = commands.add_parser("cross-validate", help="Compare neural and naive Bayes using training data only")
    comparison.add_argument("--data", required=True)
    comparison.add_argument("--folds", type=int, default=3)
    comparison.add_argument("--seed", type=int, default=7)
    comparison.add_argument("--epochs", type=int, default=180)
    comparison.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "cross-validate":
        try:
            report = cross_validate(read_dataset(args.data), folds=args.folds, seed=args.seed,
                                    epochs=args.epochs, alpha=args.alpha)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    if args.command == "batch-predict":
        return batch_predict(args, parser)
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
    raise SystemExit(main())
