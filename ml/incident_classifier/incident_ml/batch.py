"""Ordered JSONL inference with bounded batches and explicit record errors."""
from __future__ import annotations

import json
from itertools import islice
from typing import Iterable, Iterator

from .model import IncidentClassifier, tokenize


def _reject_constant(value: str):
    raise ValueError("Non-standard JSON constant")


def _record(line: str, number: int) -> tuple[dict, str | None]:
    result = {"line": number, "status": "error"}
    if not line.strip():
        result["error"] = {"code": "empty_line", "message": "Expected a JSON object"}
        return result, None
    try:
        record = json.loads(line, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        result["error"] = {"code": "invalid_json", "message": "Expected valid standard JSON"}
        return result, None
    if not isinstance(record, dict):
        result["error"] = {"code": "invalid_record", "message": "Expected a JSON object"}
        return result, None
    if "id" in record:
        identity = record["id"]
        if identity is not None and type(identity) not in (str, int):
            result["error"] = {"code": "invalid_id", "message": "id must be a string, integer or null"}
            return result, None
        result["id"] = identity
    if "text" not in record:
        result["error"] = {"code": "missing_text", "message": "Missing text field"}
        return result, None
    text = record["text"]
    if not isinstance(text, str) or not tokenize(text):
        result["error"] = {"code": "invalid_text", "message": "text must be a string containing letters"}
        return result, None
    return result, text


def predict_jsonl(lines: Iterable[str], model: IncidentClassifier, *, batch_size: int = 128,
                  threshold: float = 0.7, min_coverage: float = 0.35) -> Iterator[dict]:
    """Yield one result per physical input line, including malformed and blank lines.

    Only input validation errors become record errors. Model failures propagate
    to the caller. Memory scales with batch size, vocabulary and line length,
    rather than the total number of records.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not 0 <= threshold <= 1 or not 0 <= min_coverage <= 1:
        raise ValueError("Thresholds must be in [0, 1]")
    source = enumerate(lines, start=1)
    while True:
        chunk = list(islice(source, batch_size))
        if not chunk:
            return
        parsed = [_record(line, number) for number, line in chunk]
        texts = [text for _, text in parsed if text is not None]
        predictions = iter(model.predict(texts, threshold=threshold, min_coverage=min_coverage)) if texts else iter(())
        for result, text in parsed:
            if text is not None:
                result["status"] = "ok"
                result["prediction"] = next(predictions)
            yield result
