# Model card

## Identity

Incident ML v0.1.0 is a randomly initialised, trainable mean-embedding multilayer
perceptron. No pretrained models or proprietary data are included.

## Task and outputs

Single-label classification into fire, flood, medical, and road collision.
Outputs include a suggested class, per-class scores, vocabulary coverage, and
review reasons. Real incidents may span multiple categories; this version does
not represent that complexity. There is no severity or urgency prediction.

## Training and evaluation

76 manually authored synthetic descriptions, split before vocabulary construction.
48 examples train the parameters using full-batch Adam for 180 epochs by default.
12 validation examples select a temperature from a fixed grid by negative log
likelihood. 16 test examples are evaluated once by the training CLI. The baseline
report includes the seed, epochs, class order and confusion matrix. Results are
smoke-test evidence only. Do not repeatedly tune on this test split.

## Known failure modes

- Word order and negation are not modelled; "no fire" may resemble "fire".
- Rare categories, code switching, typos and regional language are underrepresented.
- A model can be confidently wrong even with complete vocabulary coverage.
- The small corpus exaggerates the value of shared keywords.
- Calibration on 12 synthetic records does not establish reliable probabilities.
- Unknown incidents cannot be represented by the four-class taxonomy.

## Deployment boundaries

Research and portfolio demonstration only. This package is not integrated into
the existing dashboard. Do not use it for dispatch, triage, medical decisions,
or automatic prioritisation. Human review remains necessary for every output.
Before any proposed deployment, collect appropriately governed representative
data and assess error costs, subgroup performance, privacy, robustness and drift.

## Data provenance

All included examples were newly authored as synthetic project fixtures during
development. They do not describe actual people or incidents. The dataset is
dedicated under CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/.

## Development provenance

The initial package was developed with AI assistance at the repository owner's
request. Recorded metrics come from executed local code. Future changes should
record their verification and limitations in DEVELOPMENT_LOG.md.
