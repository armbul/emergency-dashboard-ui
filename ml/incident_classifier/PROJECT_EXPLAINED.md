# How Incident ML works

Incident ML takes a short description of an incident and suggests a category.
For example, a sentence about smoke coming from a building should usually land
under fire. A sentence about a river overflowing should usually land under flood.
The interesting part of this project is the complete training and evaluation
pipeline behind that suggestion.

The model learns its parameters from labelled examples. The category names are
not hard-coded keyword rules. There is still a strong keyword bias because the
current dataset is small and synthetic. That distinction matters when deciding
what the results actually tell us.

## What the current version can do

There are four categories: fire, flood, medical and road collision. The package
can train a model from a JSONL dataset, save it, evaluate it on a fixed test
partition, and predict a category for a new sentence. It runs locally on a CPU
with Python and NumPy.

The dashboard and the classifier are seperate parts of the repository. Opening
the dashboard does not run this model, and a prediction cannot contact emergency
services. Connecting the two is later work.

The classifier also does not estimate urgency. A medical category is a topic
suggestion, not a judgement about how quickly someone needs help.

## Follow one description through the model

Take this input:

```text
Smoke and flames coming from a building
```

First, the tokenizer converts the sentence to lowercase words:

```text
smoke, and, flames, coming, from, a, building
```

Punctuation and numbers do not become tokens. Unicode letters are preserved,
which allows words containing accents and Swahili words to pass through the same
tokenizer. This does not, on its own, give the model reliable multilingual
understanding.

Each known word maps to an integer in the vocabulary. The vocabulary comes only
from the 48 training examples. Validation and test sentences cannot add words
to it.

Next, the model looks up a learned vector for each word. Each vector has 24
numbers. During training, those numbers change so that useful patterns become
easier for the later layers to distinguish.

The word vectors are averaged to produce one vector for the whole description.
Repeated words count more than words that occur once. Unknown words contribute
a zero vector but still count towards the average.

The sentence vector then passes through two hidden layers. The first produces
48 values; the second produces 24. Both use ReLU, which replaces negative
values with zero. The last layer produces four scores, one per category.

Softmax converts these scores into nonnegative values that sum to one. The
largest value determines the suggested category. A temperature chosen using
validation examples changes how sharp or spread out these scores are.

On the recorded seed-7 model, this example suggests fire. Its score rounds to
1.0. That is a confident model output on familiar vocabulary, not a guarantee
that the interpretation is correct.

## What the matrices represent

Let B be the number of descriptions in a batch. The bundled training vocabulary
contains 175 words. An additional row represents unknown words.

| Stage | Shape | What it contains |
| --- | --- | --- |
| Normalised word counts | B by 176 | Each word's share of the tokens in a description |
| Embedding table | 176 by 24 | A learned vector per known word, plus the fixed zero row |
| Pooled descriptions | B by 24 | One average word vector per description |
| First hidden layer | B by 48 | Nonlinear combinations of the pooled features |
| Second hidden layer | B by 24 | Another learned representation |
| Output logits | B by 4 | One unnormalised score per category |
| Output scores | B by 4 | Temperature-scaled softmax values |

There are 6,700 stored parameter values in this configuration. The unknown-word
embedding contains 24 of them and is kept at zero rather than trained.

Mean pooling makes the model easy to inspect, but it discards word order.
For example, "there is a fire" and "there is no fire" differ by one pooled word
rather than a structured representation of negation. That can produce a serious
classification error. This version has no attention mechanism, recurrence, or
transformer layers.

## How training changes the predictions

Every training example has a text and its expected category. The model first
predicts scores for the examples. Cross-entropy loss penalises low scores on
the expected category.

Backpropagation calculates how each parameter affected the loss. The gradients
pass through the output layer, both hidden layers and the embedding table.
Adam then updates those parameters using moving averages of the gradients and
their squares.

The implementation exposes these calculations in
[model.py](incident_ml/model.py), rather than relying on an automatic
differentiation library. That makes it possible to inspect each update and test
the gradients directly.

The default run performs 180 full-batch updates with a learning rate of 0.015.
Full-batch means all 48 training records participate in each update. Seed 7
controls the random initialisation. Repeating the same run in the same numerical
environment gives the same parameters. Small differences between platforms or
NumPy versions remain possible.

The recorded training loss falls from about 1.3988 to 0.0000000247. This shows
that the model can fit these training records. It does not show that it will
work on unfamiliar real incidents.

## Why the data has three partitions

The data file contains 76 authored synthetic descriptions. Each row explicitly
names its partition.

| Partition | Records | Used for |
| --- | --- | --- |
| Training | 48 | Building the vocabulary and learning neural parameters |
| Validation | 12 | Choosing the softmax temperature |
| Test | 16 | Measuring the finished baseline on held-out records |

Each partition contains the same number of examples per category. The dataset
reader rejects repeated descriptions after normalising case and punctuation.
It also requires all three partitions and checks that validation and test labels
exist in training.

These checks catch exact text reuse, but they do not prove independence.
Closely related phrases and repeated vocabulary still appear across partitions.
That is one reason the current test is an easy smoke test.

The validation examples stay seperate from parameter training. They are used
to compare 50 candidate temperatures between 0.5 and 5.0. The model keeps the
temperature with the lowest validation negative log likelihood.

The selected temperature was 0.5 in the first run. This makes the output
distribution sharper. Twelve synthetic records are far too few to establish
that the resulting scores match real error frequencies.

## Reading a prediction

The prediction result contains several fields:

| Field | Meaning |
| --- | --- |
| `suggested_label` | Category with the largest model score |
| `confidence` | Largest temperature-scaled softmax score |
| `probabilities` | Scores for all four categories |
| `vocabulary_coverage` | Fraction of input tokens seen in the training vocabulary |
| `requires_review` | Whether one of the implemented review checks fired |
| `review_reasons` | The specific checks that fired |

The default confidence threshold is 0.7. The default minimum vocabulary coverage
is 0.35. Empty text adds an empty-input reason. Unfamiliar vocabulary can add a
low-coverage reason, even when the neural output looks confident.

These are diagnostic flags. Every output still needs human interpretation.
A false review flag means the implemented checks did not fire; it does not mean
that the input is safe, well understood, or correctly classified.

An input such as "xyzzy quux" has zero vocabulary coverage and is flagged.
An input made entirely of familiar words can still be wrong. Coverage measures
familiar words, not familiarity with the situation they describe.

## Reproducing the current result

From the repository root, move into the package directory:

```bash
cd ml/incident_classifier
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

On Windows, replace the activation command with
`.venv\Scripts\activate`.

Train a fresh model:

```bash
python -m incident_ml.cli train --data data/synthetic_incidents.jsonl --output artifacts/incident-model --seed 7 --epochs 180
```

The training command also evaluates the test partition. It writes three files:

| File | Contents |
| --- | --- |
| `metadata.json` | Vocabulary, category order, format version and temperature |
| `weights.npz` | Numerical model parameters |
| `evaluation.json` | Training summary and held-out evaluation |

The checkpoint loader disables pickle and checks parameter shapes and finite
values. It also requires a positive, finite temperature. These checks catch some
broken artifacts; they do not authenticate their origin. Use artifacts from a
trusted source.

To evaluate the saved model or try a description:

```bash
python -m incident_ml.cli evaluate --data data/synthetic_incidents.jsonl --model artifacts/incident-model
python -m incident_ml.cli predict --model artifacts/incident-model --text "Smoke and flames coming from a building"
```

The command output is JSON, so a later application can consume it without parsing
a human-readable report.

## What the first result says

The first model got 15 of the 16 synthetic test records right.

| Measure | Recorded value | Interpretation |
| --- | --- | --- |
| Accuracy | 0.9375 | Fraction of test records assigned the expected category |
| Macro F1 | 0.9365079365 | Mean F1 across all four categories |
| Negative log likelihood | 0.5179436159 | Penalty for low scores on the expected category |

The confusion matrix rows represent expected categories and columns represent
predicted categories. Both use this order: fire, flood, medical, road collision.
The single error was a fire description predicted as flood. The other three
categories had all four records classified correctly in this small test.

With only four examples per category, one error changes that category's apparent
performance substantially. These numbers cannot support a production accuracy
claim or a comparison against models evaluated on larger real datasets.

The original values are stored in
[reports/baseline.json](reports/baseline.json). Keep that baseline as a historical
record. New experiments should produce their own reports, with the data,
configuration and limitations stated clearly.

## Why the tests matter

The numerical gradient test perturbs each parameter tensor in a random direction.
It measures how the loss changes, then compares that change with the derivative
calculated by backpropagation. This can catch an incorrect derivative even when
a training run appears to improve.

Other tests check that training learns a simple distinction, seed-controlled
runs reproduce the same parameters, checkpoint reloads preserve predictions,
extreme logits do not break softmax, and empty or unknown inputs trigger the
expected review checks.

The suite also covers duplicate text detection, invalid labels and thresholds,
Unicode tokenisation, temperature validation, and macro F1 when a category has
no examples in the supplied metric input. There are currently 11 test methods.

The GitHub workflow runs the tests and the train, evaluate and predict commands
on Python 3.10 and 3.12. This checks the path from source code to a usable model,
alongside the smaller numerical tests.

## What should change next

Batch JSONL inference is the next feature. It should accept multiple descriptions,
preserve their order, and report malformed records clearly without silently
dropping them. That will make it easier to inspect larger experiments.

The bigger modelling questions need representative data. An appropriate next
comparison is a simple text baseline, followed by multi-seed evaluation and then
attention pooling or a small transformer. Any added model should earn its
complexity through measured results.

The roadmap also includes multi-label outputs, a licensed public dataset importer,
uncertainty experiments, and a local API. Dashboard integration comes after the
model's behaviour and failure cases are better understood. A larger architecture
alone would not resolve the data limitations.
