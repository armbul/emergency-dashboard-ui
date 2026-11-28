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

## 2026-11-24: training-only model comparison

Added a token-count multinomial naive Bayes baseline, seeded stratified folds,
and the cross-validate command. Both models fit fresh vocabularies inside each
fold. Neural temperature remains 1.0 with no held-out-fold calibration. Reports
contain configuration, training-content digest, fold indices, per-fold metrics,
pooled out-of-fold metrics and probabilities. Added a detailed comparison guide,
a recorded report, README links and comparison execution in CI.

Local verification: all 34 tests passed, including nine new evaluation tests.
Tests cover a hand-calculated naive Bayes result, class-stratified fold coverage,
training-only vocabulary, excluded nontraining records, deterministic reports,
pooled metric reproduction, invalid configuration and CLI behaviour. The default
seed-7 run on 48 synthetic training records gave 31/48 correct for the neural
model (accuracy 0.6458333333, macro F1 0.6278983816) and 32/48 for naive Bayes
(accuracy 0.6666666667, macro F1 0.6574603175). The one-record difference does
not establish superiority. No fixed test results were used for this comparison.

Limitations: related synthetic phrases may span folds; fold score variation is
not a confidence interval; the neural model is uncalibrated in this comparison.
The original data, model implementation and baseline report are unchanged.

Next priority: multi-seed summaries with explicit small-sample caveats.

## 2026-11-27: multi-seed comparison

Added compare-seeds to repeat training-only cross-validation with explicit,
distinct seeds. Reports include seed-level folds and pooled metrics, mean/std/
min/max summaries, paired neural-minus-naive-Bayes differences and accuracy win
counts. Duplicate seeds are rejected. Added a detailed guide, a recorded report,
README links and a two-seed CI example. No inference or training data changed.

Verification: 41 tests passed, including seven new tests covering standalone
run equivalence, aggregate arithmetic, win counts, excluded partitions, seed
validation, reproducibility and CLI behaviour. The default five-seed run used
seeds 7, 17, 27, 37 and 47 with three folds and 180 neural epochs per fit. Mean
accuracy was 0.6458333333 for neural and 0.6541666667 for naive Bayes. Neural
accuracy ranged from 0.6041666667 to 0.6875; naive Bayes from 0.6041666667 to
0.75. Neural won two seeds and naive Bayes three. Mean paired accuracy difference
was -0.0083333333 for neural minus naive Bayes.

The same 48 synthetic training records are reused across seeds. Seed standard
deviation and range are descriptive, not independent-sample confidence
intervals. Inferential intervals are deliberately deferred. The original fixed
test result and single-seed comparison report are preserved.

Next priority: ensemble uncertainty with review coverage/error evaluation,
without tuning against the fixed test partition.

## 2026-11-28: out-of-fold ensemble uncertainty

Added ensemble-evaluate with three independently initialised neural members
per fold, mean class probabilities, predictive/member entropy, entropy-gap
mutual information, vote disagreement and confidence/review curves. Both the
first member and ensemble are evaluated on the same held-out training records.
Reports retain probabilities and targets for recomputation. No checkpoint or
existing inference behaviour changes. Added a detailed guide and CI execution.

Verification: 51 tests passed, including ten new tests for analytic uncertainty
cases, review counts and boundaries, empty-kept null error rates, invalid inputs,
fold-local vocabulary, excluded partitions, repeatability, standalone reference
equivalence and CLI behaviour. On 48 synthetic training records with three
folds, split seed 7 and member seeds 7, 17 and 27, the ensemble got 34/48 right
(accuracy 0.7083333333, macro F1 0.7070929071). The first member reproduced
31/48 correct. At threshold 0.9 and minimum vocabulary coverage 0.35, the ensemble
kept 21 predictions with three errors and flagged 27 for hypothetical review.

These are descriptive curves, not a selected policy. No human reviews were
performed. Members are uncalibrated, the corpus is synthetic, and overlapping
folds do not support independent-sample confidence claims. The fixed validation
and test records are excluded from model fitting and evaluation in this command.

Next priority: licensed public text dataset import with provenance tracking.
