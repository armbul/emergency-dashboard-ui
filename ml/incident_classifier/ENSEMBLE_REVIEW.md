# Ensemble uncertainty and review coverage

The earlier multi-seed command measured variation between complete experiments.
It did not combine predictions into an ensemble. The new ensemble-evaluate
command trains several neural members on each fold's fitting records and averages
their class scores for the same held-out records.

It also reports a confidence/review curve. This asks how many predictions remain,
and how many of those are wrong, when low-confidence or low-vocabulary-coverage
records are flagged for review. Review is simulated. Nobody has reviewed or
corrected these predictions as part of the experiment.

This is a research diagnostic. It does not change predict, batch-predict or any
dashboard behaviour. It does not dispatch services, decide urgency or supply
medical advice. Every real output would still require professional interpretation.

## Reproduce the example

From ml/incident_classifier in the installed environment:

```bash
python -m incident_ml.cli ensemble-evaluate --data data/synthetic_incidents.jsonl --member-seeds 7 17 27 --split-seed 7 --folds 3 --epochs 180 --thresholds 0 0.5 0.7 0.9 0.99 1 --min-coverage 0.35 > artifacts/ensemble_review.json
```

Create artifacts first if it does not exist. The command writes JSON to stdout;
redirect it to a new destination rather than the source dataset. The values in
the command are also the defaults. At least two distinct, nonnegative member
seeds are required. Invalid configuration produces a command error.

The dataset reader validates the complete corpus file first. The experiment then
selects only records marked train. The twelve fixed validation records and sixteen
fixed test records never enter fitting, calibration, scoring or threshold selection.
The command reports the supplied thresholds; it does not select a preferred one.

## How the ensemble is evaluated

Split seed 7 creates the same three stratified folds used in the original seed-7
comparison. Each fold fits on 32 descriptions and holds out 16. Within a fold,
all three members see the same fitting descriptions and build their vocabularies
only from those descriptions.

The initialisation seed for a member is its member seed plus the zero-based fold
number. The members use the existing architecture, learning rate and epoch count.
Temperature stays at 1.0. There is no calibration against the held-out fold.

For a held-out description, each member produces four class scores. The ensemble
takes their arithmetic mean and suggests the class with the largest mean score.
Every description is held out from every member that predicts it. The experiment
requires nine neural fits: three members in each of three folds.

The first member is a reference model using exactly the same folds and records.
With the default seeds it reproduces the neural seed-7 cross-validation metrics.
This makes the comparison easier to interpret than comparing ensembles and single
models evaluated on different records.

There is no saved ensemble checkpoint in this feature. The evaluated ensemble
members belong to different fitting sets across folds. Treating all nine models
as one inference ensemble would mix training and held-out roles and invalidate
the reported comparison.

## What the uncertainty diagnostics mean

The report includes four diagnostics. Entropies use natural logarithms, so their
units are nats. Small negative entropy gaps caused by round-off are clipped to
zero.

| Diagnostic | Calculation | What it describes |
| --- | --- | --- |
| Predictive entropy | Entropy of the averaged class scores | How spread out the ensemble output is |
| Mean member entropy | Mean entropy of the members' outputs | How spread out individual outputs are |
| Mutual information | Predictive entropy minus mean member entropy | The entropy gap associated with member disagreement |
| Vote disagreement | One minus the fraction voting for the most common class | How often top-class votes differ |

Two identical members have zero mutual information and zero vote disagreement,
even if both assign uncertain class scores. Two members assigning opposite classes
with scores of one have an averaged output of 0.5/0.5, predictive entropy of
log(2), mutual information of log(2), and vote disagreement of 0.5.

Those cases are checked against hand-calculated values in the tests. They verify
the calculations, not whether the diagnostics reliably identify real-world errors.
Members can agree confidently and still be wrong. These scores are not calibrated
probabilities of failure or validated estimates of every source of uncertainty.

The review curve currently gates on confidence and vocabulary coverage. Entropy,
mutual information and vote disagreement are reported for inspection; they are not
additional review gates in this version.

## Reading the confidence/review curve

A prediction is kept when its largest class score is at least the threshold and
its vocabulary coverage is at least min-coverage. Equality passes both checks.
Thresholds must be distinct and in [0, 1]; the returned curve is sorted ascending.

Coverage here means kept records divided by all evaluated records. This is
different from vocabulary coverage, which measures the fraction of known tokens
inside one description. In the bundled experiment, vocabulary coverage alone
flags eleven records. That is why a zero confidence threshold does not retain
all 48 predictions.

| Report field | Meaning |
| --- | --- |
| total | Number of evaluated descriptions |
| kept | Predictions passing both checks |
| reviewed | Predictions flagged for hypothetical review |
| coverage | Kept divided by total |
| kept_errors | Wrong class suggestions among kept predictions |
| kept_error_rate | Kept errors divided by kept predictions |

When no predictions are kept, kept_error_rate is null. Returning zero would
incorrectly imply measured error-free performance on an empty set. A confidence
score that rounds to one is also not proof of correctness.

The flagged records have not been corrected, and the error curve does not assume
perfect human review. It measures remaining model errors only within the kept
set. The overall classification metrics include all 48 records, before gating.

## Recorded result

See [reports/ensemble_review_comparison.json](reports/ensemble_review_comparison.json)
for configuration, fold indices, per-member metrics, averaged predictions, targets,
vocabulary coverage and both review curves. Indices refer to the training list,
not physical file lines. The training-content digest matches the earlier reports.

| Model | Correct out of 48 | Accuracy | Macro F1 | Negative log likelihood |
| --- | --- | --- | --- | --- |
| First member, seed 7 | 31 | 0.6458333333 | 0.6278983816 | 3.2214185199 |
| Three-member ensemble | 34 | 0.7083333333 | 0.7070929071 | 1.4126425093 |

The ensemble's curve on this run is:

| Confidence threshold | Kept | Flagged for review | Kept errors | Kept error rate |
| --- | --- | --- | --- | --- |
| 0.0 | 37 | 11 | 10 | 0.2702702703 |
| 0.5 | 33 | 15 | 8 | 0.2424242424 |
| 0.7 | 29 | 19 | 7 | 0.2413793103 |
| 0.9 | 21 | 27 | 3 | 0.1428571429 |
| 0.99 | 17 | 31 | 2 | 0.1176470588 |
| 1.0 | 0 | 48 | 0 | null |

These are observations at predefined thresholds, not a selected operating policy.
A higher threshold reduces coverage; it does not mathematically guarantee a lower
error rate on every dataset. The single member's error rate actually rises slightly
between thresholds 0.5 and 0.7 in this report.

The full report retains numerical member probabilities so its averaged scores,
metrics and review counts can be recomputed without retraining. No raw descriptions
are repeated in the report. Existing metrics clip the expected-class score at
1e-12 for negative log likelihood.

## Limits, tests and next work

The ensemble improves this particular synthetic comparison by three records.
That is not evidence of deployment improvement. There are only twelve training
records per class, related phrases can span folds, and fitting sets overlap.
There are no independent-sample confidence intervals or validated clinical claims.
Keep experimental confidence and operational reliability seperate.

Ten new tests cover hand-calculated uncertainty, identical-member agreement,
threshold equality, review counts, the empty-kept case, invalid probability arrays,
configuration errors, fold-local vocabularies, excluded partitions, reproducibility,
standalone reference equivalence and command-line behaviour. The full suite now
contains 51 test methods. CI runs the experiment and checks its report structure.

The next roadmap item is a licensed public text dataset importer with provenance
tracking. Representative data is needed before drawing stronger conclusions about
ensemble quality or choosing a review policy. A future threshold-selection study
should use an explicit inner selection split and an untouched outer evaluation set.
