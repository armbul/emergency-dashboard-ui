# How much does the result depend on the seed?

The first cross-validation report used seed 7. Its one-record difference between
models could have depended on the particular folds or neural initialisation.
The compare-seeds command repeats that experiment with several explicit seeds
and reports every run alongside descriptive summaries.

This feature changes evaluation, not the classifier used by predict or
batch-predict. It does not load or replace a saved checkpoint. All comparisons
still use only the 48 records marked train.

## Reproduce the recorded experiment

From ml/incident_classifier in the installed environment:

```bash
python -m incident_ml.cli compare-seeds --data data/synthetic_incidents.jsonl --seeds 7 17 27 37 47 --folds 3 --epochs 180 --alpha 1.0 > artifacts/multi_seed.json
```

Create the artifacts directory first if needed. Output is JSON on stdout; choose
a new destination rather than redirecting over the input dataset. Omitting
--seeds uses the five seeds above. The command requires at least two distinct,
nonnegative integer seeds. Repeating a seed is rejected because it would give
the same run extra weight.

The fold count, neural epochs and naive Bayes smoothing setting are shared by
all runs. Invalid inputs produce an error instead of a partial success report.
The dataset loader validates the whole corpus file before the comparison selects
the training partition. Validation and test content never enter model fitting,
temperature selection, scoring or the reported training-content digest.

## What changes between runs

Each seed is passed to the existing cross_validate function. It changes the
class-stratified shuffle that assigns training records to folds. It also changes
the neural initialisation: fold number is added to the selected seed, exactly
as in the single-seed command.

Within a seed, both models use the same fitting and held-out records. Each fold
builds new vocabularies from its fitting records. With the bundled data there
are three folds, each fitting 32 descriptions and holding out 16. Each training
description receives one held-out prediction per model per seed.

Five seeds therefore run fifteen neural fits and fifteen naive Bayes fits. The
same 48 descriptions are reused. There are not 240 independent observations.
The neural model stays at temperature 1.0, with no calibration against the
held-out fold. This preserves the existing comparison protocol.

The seeds change fold assignment and neural initialisation together. This
experiment cannot say how much variation comes from each source individually.
An experiment fixing folds while varying neural seeds would answer a different
question and would need its own configuration and report.

## What is in the report

The report records seeds in the supplied order, shared model settings, class
order and the SHA-256 digest of the ordered training text/label pairs. Every
run includes its seed, fold indices, fold metrics and pooled out-of-fold metrics
for both models. Indices refer to the selected training list, not JSONL lines.

The repeated report omits per-record probability arrays to avoid duplicating
large outputs. Use cross-validate with a particular seed to obtain those arrays.
The stored fold definitions and metrics still allow inspection of each run.

For each model, accuracy, macro F1 and negative log likelihood have a mean,
population standard deviation, minimum and maximum across seed-level pooled
metrics. Every seed has equal weight. Macro F1 is calculated from each seed's
complete out-of-fold predictions before those seed-level values are averaged.

The paired_neural_minus_nb section first subtracts naive Bayes from the neural
score within each seed, then summarises those differences. Pairing matters
because the two models share folds within that seed. Positive accuracy or macro
F1 differences favour the neural model. Negative log likelihood has the opposite
direction: a negative difference favours the neural model.

The accuracy_wins field counts seeds where each model has higher accuracy,
plus exact ties. It is a descriptive count, not a statistical significance test.
Neither this count nor the mean difference chooses a model automatically.

## Recorded result

The [recorded report](reports/multi_seed_comparison.json) uses seeds 7, 17, 27,
37 and 47 with three folds and 180 neural epochs.

| Model | Mean accuracy | Accuracy range | Accuracy standard deviation | Mean macro F1 | Mean negative log likelihood |
| --- | --- | --- | --- | --- | --- |
| Neural | 0.6458333333 | 0.6041666667 to 0.6875 | 0.0263523138 | 0.6336702952 | 2.7130585711 |
| Multinomial naive Bayes | 0.6541666667 | 0.6041666667 to 0.75 | 0.0520416500 | 0.6371917781 | 0.8501487138 |

The neural model has higher accuracy on two seeds; naive Bayes has higher
accuracy on three. There are no ties. The mean paired accuracy difference is
-0.0083333333, about -0.83 percentage points for neural minus naive Bayes.
Across individual seeds the difference ranges from about -10.42 to +8.33
percentage points. That spread makes a conclusion from seed 7 alone look fragile.

Naive Bayes also has lower mean negative log likelihood in this experiment.
The neural model can assign strong scores to wrong answers, so reporting only
its winning category would hide useful information. Both use the existing
metric function's clipping floor of 1e-12 for the expected-class score.

These results do not establish that either model is superior on real incidents.
All examples are synthetic, classes have only twelve training records each,
and related phrases can appear across folds. No fixed validation or test scores
were consulted to produce the comparison.

## Why these are not confidence intervals

Minimum, maximum and standard deviation describe these particular seed runs.
They do not give a confidence interval for deployment accuracy. Fits overlap
within and across runs, and every seed reuses the same small corpus. Dividing
the seed standard deviation by the square root of five would not fix the lack
of independent evaluation data.

For that reason the command reports descriptive variation and explicitly avoids
an inferential interval. A later uncertainty analysis should define its target
first, account for dependence, and use appropriately governed representative
data. Keep seed sensitivity and sampling uncertainty seperate when discussing
the model.

## Tests and next steps

Seven new tests check agreement with standalone runs, aggregate arithmetic,
paired differences, win counts, invalid and duplicate seeds, excluded held-out
partitions, repeatability, default settings and CLI success/error behaviour.
The full suite now has 41 test methods. CI runs a two-seed example and validates
the generated report structure alongside the existing pipeline checks.

The next feature is an ensemble uncertainty experiment with an explicit review
coverage/error comparison on suitable held-out data. An ensemble should be
evaluated as a distinct model; averaging these cross-validation metrics is not
an ensemble prediction.
