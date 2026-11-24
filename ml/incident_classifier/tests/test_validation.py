import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from incident_ml.model import classification_metrics
from incident_ml.validation import MultinomialTextBaseline, cross_validate, stratified_folds


class ValidationTests(unittest.TestCase):
    def rows(self):
        return [{"text": text, "label": label, "split": "train"} for text, label in
                [("fire smoke", "fire"), ("fire flames", "fire"), ("burning house", "fire"),
                 ("river flood", "flood"), ("flood water", "flood"), ("heavy rain", "flood")]]

    def test_naive_bayes_matches_hand_calculation(self):
        model = MultinomialTextBaseline(["a a", "b"], ["first", "second"])
        # P(a|first)=3/4, P(a|second)=1/3; equal class priors.
        np.testing.assert_allclose(model.predict_proba(["a"]), [[9/13, 4/13]])
        np.testing.assert_allclose(model.predict_proba(["unseen"]), [[0.5, 0.5]])
        self.assertNotIn("unseen", model.vocabulary)

    def test_repeated_tokens_and_empty_prediction_batch(self):
        model = MultinomialTextBaseline(["a a", "b"], ["first", "second"])
        self.assertGreater(model.predict_proba(["a a"])[0, 0], model.predict_proba(["a"])[0, 0])
        self.assertEqual(model.predict_proba([]).shape, (0, 2))

    def test_folds_cover_each_record_once_and_preserve_classes(self):
        labels = ["a"] * 5 + ["b"] * 7
        folds = stratified_folds(labels, 3)
        self.assertEqual(sorted(np.concatenate(folds)), list(range(12)))
        for fold in folds:
            self.assertEqual({labels[i] for i in fold}, {"a", "b"})
        self.assertEqual([f.tolist() for f in folds], [f.tolist() for f in stratified_folds(labels, 3)])

    def test_invalid_options(self):
        for options in [{"folds": 1}, {"folds": 4}, {"seed": -1}, {"epochs": 0}, {"alpha": float("nan")}, {"alpha": 0}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                cross_validate(self.rows(), **options)

    def test_validation_and_test_records_do_not_affect_results(self):
        expected = cross_validate(self.rows(), epochs=5)
        extra = [{"split": "test", "text": "secret excluded token", "label": "new class"},
                 {"split": "validation", "text": None, "label": None}]
        self.assertEqual(expected, cross_validate(self.rows() + extra, epochs=5))

    def test_fold_vocabulary_excludes_held_out_only_words(self):
        rows = self.rows()
        report = cross_validate(rows, epochs=5)
        for fold in report["folds"]:
            fit_texts = [rows[i]["text"] for i in fold["fit_indices"]]
            expected_words = set(" ".join(fit_texts).split())
            self.assertEqual(fold["vocabulary_words"], len(expected_words))
            self.assertFalse(set(fold["fit_indices"]) & set(fold["held_out_indices"]))

    def test_pooled_metrics_reproduce_and_probabilities_are_finite(self):
        report = cross_validate(self.rows(), epochs=5)
        targets = np.array([0, 0, 0, 1, 1, 1])
        for summary in report["models"].values():
            values = np.array(summary["out_of_fold_probabilities"])
            np.testing.assert_allclose(values.sum(axis=1), 1)
            self.assertTrue(np.isfinite(values).all())
            self.assertEqual(summary["out_of_fold"], classification_metrics(targets, values))
        self.assertEqual(report, cross_validate(self.rows(), epochs=5))

    def test_duplicate_training_text_rejected(self):
        rows = self.rows()
        rows[1]["text"] = "FIRE, smoke!"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cross_validate(rows)

    def test_cli_report_and_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "data.jsonl")
            rows = self.rows() + [{"text": "validation sentence", "label": "fire", "split": "validation"},
                                  {"text": "test sentence", "label": "flood", "split": "test"}]
            path.write_text("\n".join(json.dumps(r) for r in rows))
            command = [sys.executable, "-m", "incident_ml.cli", "cross-validate", "--data", str(path), "--epochs", "5"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["training_records"], 6)
            bad = subprocess.run(command + ["--folds", "4"], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
