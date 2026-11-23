import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from incident_ml.batch import predict_jsonl
from incident_ml.model import IncidentClassifier


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = IncidentClassifier.from_training_data(["fire smoke", "flood water"], ["fire", "flood"])
        cls.model.fit(["fire smoke", "flood water"], ["fire", "flood"], epochs=30)

    def test_order_errors_and_identifiers(self):
        lines = ['{"id":"a","text":"fire smoke"}\n', '{broken\n', '\n',
                 '{"id":7,"text":"flood water"}\n', '{"id":null,"text":"unknown"}']
        results = list(predict_jsonl(lines, self.model, batch_size=2))
        self.assertEqual([r["line"] for r in results], [1, 2, 3, 4, 5])
        self.assertEqual([r["status"] for r in results], ["ok", "error", "error", "ok", "ok"])
        self.assertEqual([results[i]["id"] for i in (0, 3, 4)], ["a", 7, None])
        self.assertTrue(results[4]["prediction"]["requires_review"])
        self.assertNotIn("text", json.dumps(results))

    def test_prediction_parity_across_batch_sizes(self):
        texts = ["fire smoke", "flood water", "unknown"]
        for size in [1, 2, 100]:
            results = list(predict_jsonl(map(lambda t: json.dumps({"text": t}), texts), self.model, batch_size=size))
            expected = self.model.predict(texts)
            for actual, prediction in zip(results, expected):
                self.assertEqual(actual["prediction"]["suggested_label"], prediction["suggested_label"])
                self.assertAlmostEqual(actual["prediction"]["confidence"], prediction["confidence"], places=12)

    def test_invalid_records(self):
        cases = [('[]', 'invalid_record'), ('null', 'invalid_record'), ('{}', 'missing_text'),
                 ('{"text":false}', 'invalid_text'), ('{"text":"123 !"}', 'invalid_text'),
                 ('{"text":""}', 'invalid_text'), ('{"text":"fire","id":[]}', 'invalid_id'),
                 ('{"text":"fire","id":true}', 'invalid_id'), ('{"text":"fire","id":1.2}', 'invalid_id'),
                 ('{"text":"fire","extra":NaN}', 'invalid_json')]
        for line, code in cases:
            with self.subTest(line=line):
                result = list(predict_jsonl([line], self.model))[0]
                self.assertEqual(result["error"]["code"], code)

    def test_unicode_and_extra_fields(self):
        result = list(predict_jsonl(['{"text":"Moshi café","extra":"ignored"}'], self.model))[0]
        self.assertEqual(result["status"], "ok")

    def test_empty_input(self):
        self.assertEqual(list(predict_jsonl([], self.model)), [])

    def test_invalid_configuration_even_for_empty_input(self):
        for options in [{"batch_size": 0}, {"batch_size": True}, {"threshold": float("nan")}, {"min_coverage": 2}]:
            with self.assertRaises(ValueError):
                list(predict_jsonl([], self.model, **options))

    def test_stream_is_consumed_in_bounded_chunks(self):
        def lines():
            for i in range(20):
                consumed.append(i)
                yield '{"text":"fire"}'
        consumed = []
        results = predict_jsonl(lines(), self.model, batch_size=3)
        next(results)
        self.assertEqual(len(consumed), 3)

    def test_model_failures_propagate(self):
        class BrokenModel:
            def predict(self, *args, **kwargs):
                raise RuntimeError("shared model failure")
        with self.assertRaisesRegex(RuntimeError, "shared model"):
            list(predict_jsonl(['{"text":"fire"}'], BrokenModel()))

    def run_cli(self, *args, input=None):
        return subprocess.run([sys.executable, "-m", "incident_ml.cli", "batch-predict", "--model", self.checkpoint,
                               *args], input=input, text=True, encoding="utf-8", capture_output=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.checkpoint = str(Path(self.tmp.name, "model"))
        self.model.save(self.checkpoint)

    def test_cli_stdin_stdout_and_partial_failure_status(self):
        result = self.run_cli("--input", "-", input='{"text":"fire"}\nBAD\n{"text":"flood"}\n')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 3)
        self.assertEqual(json.loads(result.stderr), {"records": 3, "succeeded": 2, "failed": 1})

    def test_cli_file_io(self):
        source, output = Path(self.tmp.name, "in.jsonl"), Path(self.tmp.name, "out.jsonl")
        source.write_text('{"id":"m","text":"Moshi"}', encoding="utf-8")
        result = self.run_cli("--input", str(source), "--output", str(output), "--batch-size", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(json.loads(output.read_text())["id"], "m")

    def test_cli_prevents_input_overwrite_including_hardlinks(self):
        source = Path(self.tmp.name, "input.jsonl")
        original = '{"text":"fire"}'
        source.write_text(original)
        linked = Path(self.tmp.name, "linked.jsonl")
        os.link(source, linked)
        for destination in [source, linked]:
            result = self.run_cli("--input", str(source), "--output", str(destination))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(source.read_text(), original)

    def test_cli_invalid_config_does_not_truncate_output(self):
        output = Path(self.tmp.name, "output.jsonl")
        output.write_text("keep me")
        result = self.run_cli("--input", "-", "--output", str(output), "--threshold", "nan", input="")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(output.read_text(), "keep me")

    def test_cli_missing_input_reports_failure(self):
        result = self.run_cli("--input", str(Path(self.tmp.name, "missing")))
        self.assertEqual(result.returncode, 2)

    def test_cli_empty_file_summary(self):
        result = self.run_cli("--input", "-", input="")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stderr)["records"], 0)


if __name__ == "__main__":
    unittest.main()
