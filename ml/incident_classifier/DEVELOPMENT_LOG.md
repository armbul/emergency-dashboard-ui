# Development log

## 2026-11-22: initial trainable neural baseline

Added a NumPy neural classifier, Adam training, validation-only temperature
selection, test evaluation, JSON inference, safe checkpoint reload, 76 synthetic
examples, documentation, and a model card.

Verification: 11 unit tests passed, including directional finite-difference
checks of all seven parameter tensors, deterministic training, unknown/empty
input review flags, checkpoint round trips, and split-duplicate detection.
The train/evaluate/predict CLI commands were exercised locally.

Seed-7 held-out synthetic result: 15/16 correct, accuracy 0.9375, macro F1
0.9365079365, NLL 0.5179436159. These figures describe only the bundled toy corpus.
This first version has no dashboard integration or real incident benchmark.

Next priority: batch JSONL inference with record-level error reporting.
