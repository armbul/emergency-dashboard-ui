# Comparing the neural model with a text baseline

The original test report answers one narrow question: how did one trained model
perform on sixteen synthetic test descriptions? This comparison asks a different
question: does the neural architecture do better than a simple token-count model
when both are trained and evaluated on exactly the same folds?

The new cross-validate command compares the existing neural classifier with
multinomial naive Bayes. It uses only the 48 records marked train. The fixed
validation and test partitions stay out of fitting, temperature selection and
scoring. No additional library or model download is needed.

## Run it

From ml/incident_classifier in the installed environment:

```bash
python -m incident_ml.cli cross-validate --data data/synthetic_incidents.jsonl --folds 3 --seed 7 --epochs 180 --alpha 1.0 > artifacts/comparison.json
```

Create the artifacts directory first if it does not exist. The report goes to
stdout; redirect it to a new output path, never the input dataset. The command
does not save or replace a model checkpoint. Invalid fold counts, negative seeds,
nonpositive epochs and invalid smoothing values produce a command error.

The dataset loader still checks the complete file's schema, partition presence,
label consistency and duplicate text. After that validation, the comparison
selects only training rows. Changing the contents of a valid held-out partition
does not affect its fitted models or scores.

## What happens inside each fold

The splitter groups training records by class, shuffles each group with the
specified seed, and divides each class across the requested folds. Every class
must have at least as many records as there are folds. With the bundled data,
each of three folds has four descriptions from every class: sixteen in total.

For a particular fold, its sixteen records are held out and the other thirty-two
are used for fitting. Both models build new vocabularies from those thirty-two
records alone. Words appearing only in the held-out fold cannot enter either
vocabulary. The models predict that fold, then the process repeats for the other
two folds. Each training record receives one prediction from each model, from
a model that did not fit that record.

Neural initialisation uses seed plus the zero-based fold number. Its dimensions,
learning rate and epoch count otherwise match the existing baseline. Temperature
stays at 1.0. Selecting temperature on a held-out fold and then reporting that
fold's performance would contaminate the comparison. A future calibrated
comparison would need an additional inner split.

## How naive Bayes works here

Naive Bayes counts how often each vocabulary word occurs in each class. Additive
smoothing adds alpha to every word count so a word never receives a zero
likelihood merely because it did not occur in a particular class.

For a new description, the baseline adds the log likelihood of each known token
to the log class prior. Repeated words contribute repeatedly. Class priors are
the proportions of fitted records in each class. The highest resulting score
determines the prediction. Softmax produces normalised per-class scores for the
same metric functions used by the neural model.

Unknown tokens are ignored. A description with no known tokens falls back to
class priors. This baseline has no embedding layer or hidden layers. Its word
independence assumption is crude, but it is a useful comparison on a small text
dataset where a larger model might simply memorise recurring words.

The implementation lives in incident_ml/validation.py. Its probability test
checks a tiny two-word example against a hand-calculated answer. Fold coverage,
reproducibility, excluded partitions, pooled metrics and CLI errors are tested
alongside the existing model and batch tests.

## Reading the report

The report includes the configuration, sorted class order, a SHA-256 digest of
the ordered training text/label pairs, and per-fold fit and held-out indices.
Indices refer to the selected training list, not physical JSONL line numbers.
There are no raw descriptions in the report. The digest tracks the relevant
training content; it is not an authenticity or anonymisation guarantee.

Each model has pooled out-of-fold metrics, mean fold accuracy, population
standard deviation of fold accuracy, and per-record out-of-fold scores. The
pooled confusion matrix includes every training record once. Pooled macro F1
is calculated from all out-of-fold predictions, rather than averaging the fold
F1 values. These quantities can differ.

Fold standard deviation describes variation between these folds. It is not a
confidence interval: their fitting sets overlap, and they share one small corpus.
All scores use the existing metric implementation, which clips the expected
class score at 1e-12 when calculating negative log likelihood.

## Recorded seed-7 run

The report in [reports/cross_validation_seed7.json](reports/cross_validation_seed7.json)
was produced with three folds, 180 neural epochs and alpha 1.0.

| Model | Correct out of 48 | Accuracy | Macro F1 | Negative log likelihood |
| --- | --- | --- | --- | --- |
| Neural classifier | 31 | 0.6458333333 | 0.6278983816 | 3.2214185199 |
| Multinomial naive Bayes | 32 | 0.6666666667 | 0.6574603175 | 0.8434327138 |

Naive Bayes gets one more record right in this run. That difference is too small
to establish superiority. The neural model's higher log loss also shows that its
errors can carry strong scores. Neither result validates probability calibration.

These scores are lower than the original 15/16 synthetic test result, but the
experiments use different evaluation records, training sizes and calibration
settings. They cannot be read as evidence of a regression in the deployed
classifier. The existing train, predict and batch commands retain their previous
behaviour. The original baseline report remains unchanged.

## Limits and next work

The corpus is still synthetic, small and rich in shared vocabulary. Exact
normalised training duplicates are rejected; related phrases can still span
folds. These results do not measure real emergency understanding, subgroup
fairness, dispatch readiness or reliable multilingual performance.

Keep this report and the original held-out result seperate when discussing
performance. Repeatedly choosing settings against these folds would make them
part of model selection. A later claim about generalisation would need suitable
independent data or a properly nested evaluation design.

The next improvement is a multi-seed summary showing whether the comparison
changes under different fold assignments and neural initialisations. Wider
representative data remains more important than a favourable number on this
toy corpus.
