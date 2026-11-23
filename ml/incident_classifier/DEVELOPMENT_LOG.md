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

## 2026-11-22: project walkthrough

Added PROJECT_EXPLAINED.md with an input-to-output walkthrough, exact matrix
shapes and parameter count, training and validation choices, prediction fields,
reproduction commands, test coverage, baseline interpretation and next steps.
Linked it from the package README. Checked the explanations against the current
implementation and recorded baseline. No model behaviour or parameters changed.

## 2026-11-23: batch JSONL inference

Added batch-predict for UTF-8 JSONL files and stdin. Results preserve physical
line numbers, valid optional IDs and input order. Invalid records produce stable
error codes without hiding later records. The iterator uses bounded chunks and
keeps summaries on stderr. Exit codes distinguish success, record errors and
supported command failures. File-path checks prevent overwriting the input,
including through hard links. Added a synthetic example, a detailed usage guide,
README and roadmap links, and batch execution to the existing CI workflow.

Local verification: all 25 tests passed, including 14 new batch tests. The valid
three-record example produced three predictions. A mixed five-line example
produced three predictions and two error records, in order, with exit code 1.
The errors were invalid_json and empty_line. The unknown-vocabulary example
retained its review flag. The seed-7 training command reproduced the original
baseline report exactly. No training data, neural parameters or inference rules
were changed by this feature.

Limitations: input line size is not capped; memory depends on line length and
vocabulary as well as batch size. Interrupted or failed output is not atomic and
can be partial. No throughput or real-world accuracy improvement is claimed.

Next priority: a simple text baseline and stratified cross-validation on training
data, keeping the fixed test partition out of model selection.
