# Classifying a JSONL file

The original predict command handles one description. The new batch-predict
command handles a file or a stream. It produces one JSON result per physical
input line, in the same order. If a line is malformed, its result describes the
problem and processing continues with the next line.

This is useful for inspecting an experiment without writing a Python script or
launching one Python process per description. The neural model is loaded once,
and valid descriptions are passed to it in bounded batches.

## Try the included example

From ml/incident_classifier, train the model if you have not already done so:

```bash
python -m incident_ml.cli train --data data/synthetic_incidents.jsonl --output artifacts/incident-model
python -m incident_ml.cli batch-predict --model artifacts/incident-model --input examples/batch_input.jsonl --output artifacts/predictions.jsonl --batch-size 2
```

The three examples are synthetic. Two describe familiar categories and the last
contains unknown words. Its vocabulary coverage is zero, so its prediction is
flagged for review. It is still a successfully processed record. A review flag
and a record validation error mean different things.

The input has a required text field and an optional id:

```json
{"id":"report-17","text":"Smoke and flames coming from a building"}
```

An id can be a string, integer or null. Duplicate IDs are allowed. The physical
line number remains available even if records have identical IDs. Other fields
are ignored. The text must be a string containing at least one letter according
to the existing tokenizer. Labels and training partitions are not required.

## What comes back

Successful results have line, status, optional id, and prediction fields:

```json
{"line":1,"id":"report-17","status":"ok","prediction":{"suggested_label":"fire","confidence":1.0,"probabilities":{"fire":1.0,"flood":0.0,"medical":0.0,"road_collision":0.0},"vocabulary_coverage":1.0,"requires_review":false,"review_reasons":[]}}
```

This example rounds scores for readability. Actual output preserves the numerical
values returned by the model, including very small nonzero scores.

A malformed JSON line produces this shape:

```json
{"line":2,"status":"error","error":{"code":"invalid_json","message":"Expected valid standard JSON"}}
```

The input text and malformed content are not echoed into output or error
messages. Valid IDs are preserved even when their record has a missing or invalid
text field. IDs themselves can be identifying, so use suitable non-sensitive
identifiers when sharing results.

| Error code | What happened |
| --- | --- |
| empty_line | A physical input line was blank |
| invalid_json | JSON could not be parsed, including NaN or Infinity constants |
| invalid_record | JSON was valid but its top-level value was not an object |
| invalid_id | An id was neither a string, integer nor null |
| missing_text | The object had no text field |
| invalid_text | Text was not a string containing letters |

When several fields are invalid, validation returns the first applicable error
in the order shown above. An empty file has zero results and succeeds. A trailing
newline terminates the last record; an additional blank line produces an error
result. That makes missing or accidentally blank records visible.

## Streams, summaries and exit codes

Set --input to - to read stdin. Output goes to stdout unless --output names a
file. The command keeps its summary on stderr, so stdout remains valid JSONL.

```bash
python -m incident_ml.cli batch-predict --model artifacts/incident-model --input - < examples/batch_input.jsonl > artifacts/predictions.jsonl
```

For the included valid example, stderr contains:

```json
{"records":3,"succeeded":3,"failed":0}
```

| Exit code | Meaning |
| --- | --- |
| 0 | Every supplied record was processed successfully, or the input was empty |
| 1 | At least one record failed validation; the output still contains all line results |
| 2 | A configuration, file access, decoding or supported checkpoint-loading error stopped the command |

The neural checkpoint must load before output files are opened. Invalid batch
configuration is also rejected before an existing output is truncated. Input and
output must be different files, including paths that point at the same file
through symbolic or hard links. Successful file output replaces an existing
destination. Output directories must already exist.

An interrupted run, unexpected model exception, unreadable input, or write failure
can leave partial output. Output is not an atomic transaction and there is no
resume facility. The batch library propagates shared model failures rather than
pretending that they are bad input records. For a production pipeline, retain
partial results separately and check process success before using them.

## How batching works

The implementation reads at most batch-size physical lines into a chunk,
validates each line, collects the valid descriptions, and calls model.predict
once for that group. It places the predictions back beside their original line
numbers, interleaved with any errors, before reading the next chunk.

The default batch size is 128. It limits records per chunk rather than characters
per line. Memory still depends on line length and vocabulary size, and a single
very long line can consume substantial memory. There is no throughput benchmark
claim for this first implementation.

Use --threshold to change the low-confidence review check and --min-coverage for
the vocabulary-coverage check. Both accept values between 0 and 1. These checks
retain the existing model semantics. They do not establish real incident accuracy.

## Verification and limits

The tests compare batch scores and suggested classes against single-call
predictions, exercise several batch sizes, preserve ordering across invalid
records, and verify that the input iterator is consumed in bounded chunks.
They also cover Unicode, IDs, empty input, invalid configuration, stream and file
commands, exit codes and same-file protection.

The model, its weights and its synthetic evaluation baseline did not change.
Batch processing and model quality remain seperate concerns. This feature makes
experiments easier to run and inspect; it does not improve the classifier's
ability to understand emergencies.
