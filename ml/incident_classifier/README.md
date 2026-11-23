# Incident ML: an inspectable neural classification baseline

A complete local research pipeline for categorising short incident descriptions into
`fire`, `flood`, `medical`, and `road_collision`. This is a new ML subproject of the
emergency dashboard. It does not yet connect to the dashboard or dispatch services.

## Understand the project

Read [How Incident ML works](PROJECT_EXPLAINED.md) for a detailed walkthrough of
the problem, architecture, training process, data splits, example predictions,
test results and known failure cases.

## Batch inference

Use `batch-predict` to classify a JSONL file with ordered predictions and a clear
error for each invalid line. See [the batch inference guide](BATCH_INFERENCE.md)
for examples, output fields, exit codes, streaming behaviour and limitations.

## What works

- A trainable embedding layer and two nonlinear hidden layers, implemented in NumPy.
- Explicit backpropagation and Adam optimisation, with numerical gradient checks.
- Fixed train/validation/test partitions and a training-only vocabulary.
- Validation-set temperature scaling, JSON predictions, confidence and vocabulary-coverage review flags.
- Held-out accuracy, macro F1, negative log likelihood, and a confusion matrix.
- Reproducible training, checkpoint reloads without pickle, and a standard-library test suite.
- CPU-only operation; no model downloads, accounts, external inference API, or GPU required.

## Run locally

From this directory, using Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m incident_ml.cli train --data data/synthetic_incidents.jsonl --output artifacts/incident-model
python -m incident_ml.cli evaluate --data data/synthetic_incidents.jsonl --model artifacts/incident-model
python -m incident_ml.cli predict --model artifacts/incident-model --text "Smoke and flames coming from a building"
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
Installed environments also expose the same commands through `incident-ml`.

Training writes `metadata.json`, `weights.npz`, and `evaluation.json` to the
output directory. Model artifacts are deliberately ignored by Git; reproduce them
with the training command above. The seed defaults to 7.

## Architecture

Unicode word tokens become vocabulary IDs. Normalised token counts pool learned
24-dimensional embeddings. The pooled vector passes through 48-unit and 24-unit
ReLU layers, followed by four output logits and a temperature-scaled softmax.
The architecture ignores word order. It is an intentionally small deep neural
baseline, not a transformer or large language model.

Unknown words contribute a fixed zero embedding. Their proportion is exposed in
predictions. Empty input, low vocabulary coverage, or confidence below the selected
threshold adds a review flag. These heuristics do not detect all unfamiliar inputs;
the absence of a flag does not establish that a prediction is correct.

## Data and results

The included 76 records are newly authored synthetic examples: 48 training,
12 validation, and 16 test records, evenly divided across four categories.
Some examples are in Swahili, but this tiny sample does not establish multilingual
capability. There are no real incident records or personal identifiers.
Exact normalised duplicate text across the corpus is rejected before training.
Near-duplicates and shared vocabulary still make this an easy, narrow dataset.

The first seed-7 run classified 15 of 16 synthetic test examples correctly
(accuracy 0.9375, macro F1 0.9365). See [the recorded run](reports/baseline.json).
These numbers only verify that the pipeline runs on the toy corpus. No real-world
performance, calibrated reliability, or operational suitability has been established.
Hyperparameters must not be selected against this test partition in future work.

## Intended use and limitations

Use this package to inspect, test, and extend an ML pipeline. It must not decide
emergency priority, give medical advice, replace professional assessment, or trigger
dispatch. Predictions always require human interpretation. Confidence is a model
score, not a validated probability of correctness. Validation data here is too small
to establish calibration. There is no severity model, geographical context, real
deployment evaluation, access control, or dashboard integration.

The synthetic examples are dedicated to the public domain under CC0 1.0; the new
code in this subproject is MIT licensed. Existing dashboard files keep their existing
terms. See [the model card](MODEL_CARD.md) and [development roadmap](ROADMAP.md).
