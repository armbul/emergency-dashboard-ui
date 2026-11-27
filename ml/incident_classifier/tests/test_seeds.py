import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from incident_ml.validation import compare_seeds, cross_validate


class SeedComparisonTests(unittest.TestCase):
    def rows(self):
        return [{"text": text, "label": label, "split": "train"} for text, label in
                [("fire smoke", "fire"), ("fire flames", "fire"), ("burning house", "fire"),
                 ("river flood", "flood"), ("flood water", "flood"), ("heavy rain", "flood")]]

    def test_each_run_matches_standalone_comparison(self):
        report = compare_seeds(self.rows(), seeds=[7, 17], epochs=5)
        for run in report["runs"]:
            single = cross_validate(self.rows(), seed=run["seed"], epochs=5)
            self.assertEqual(run["folds"], single["folds"])
            for name in run["models"]:
                self.assertEqual(run["models"][name], single["models"][name]["out_of_fold"])

    def test_aggregates_and_paired_differences_recompute(self):
        report = compare_seeds(self.rows(), seeds=[7, 17, 27], epochs=5)
        for name, metrics in report["models"].items():
            for metric, stats in metrics.items():
                values = [run["models"][name][metric] for run in report["runs"]]
                self.assertAlmostEqual(stats["mean"], np.mean(values))
                self.assertAlmostEqual(stats["std"], np.std(values))
                self.assertEqual(stats["min"], min(values))
                self.assertEqual(stats["max"], max(values))
        for metric, summary in report["paired_neural_minus_nb"].items():
            expected = [run["models"]["neural"][metric] - run["models"]["multinomial_nb"][metric]
                        for run in report["runs"]]
            self.assertEqual(summary["per_seed"], expected)
            self.assertAlmostEqual(summary["mean"], np.mean(expected))
        self.assertEqual(sum(report["accuracy_wins"].values()), 3)
        expected_wins = {"neural": 0, "multinomial_nb": 0, "ties": 0}
        for run in report["runs"]:
            a, b = run["models"]["neural"]["accuracy"], run["models"]["multinomial_nb"]["accuracy"]
            expected_wins["neural" if a > b else "multinomial_nb" if b > a else "ties"] += 1
        self.assertEqual(report["accuracy_wins"], expected_wins)

    def test_invalid_seed_lists_rejected(self):
        for seeds in [[], [7], [7, 7], [-1, 7], [True, 7], [1.5, 7]]:
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                compare_seeds(self.rows(), seeds=seeds, epochs=5)

    def test_nontraining_records_are_excluded(self):
        report = compare_seeds(self.rows(), seeds=[7, 17], epochs=5)
        extras = [{"split": "test", "text": "excluded", "label": "unknown"},
                  {"split": "validation", "text": None}]
        self.assertEqual(report, compare_seeds(self.rows() + extras, seeds=[7, 17], epochs=5))
        self.assertEqual(report["training_records"], 6)

    def test_reproducibility_and_order_independent_aggregate(self):
        report = compare_seeds(self.rows(), seeds=[7, 17], epochs=5)
        self.assertEqual(report, compare_seeds(self.rows(), seeds=[7, 17], epochs=5))
        reverse = compare_seeds(self.rows(), seeds=[17, 7], epochs=5)
        self.assertEqual(report["models"], reverse["models"])
        self.assertEqual(report["accuracy_wins"], reverse["accuracy_wins"])

    def test_defaults_and_descriptive_uncertainty_label(self):
        report = compare_seeds(self.rows(), epochs=5)
        self.assertEqual(report["configuration"]["seeds"], [7, 17, 27, 37, 47])
        self.assertNotIn("confidence_interval", report)
        self.assertIn("Descriptive", report["uncertainty_note"])
        self.assertTrue(np.isfinite(report["models"]["neural"]["accuracy"]["std"]))

    def test_cli_example_and_duplicate_seed_error(self):
        data = Path(__file__).resolve().parents[1] / "data/synthetic_incidents.jsonl"
        command = [sys.executable, "-m", "incident_ml.cli", "compare-seeds", "--data", str(data),
                   "--seeds", "7", "17", "--epochs", "5"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(len(report["runs"]), 2)
        self.assertEqual(report["training_records"], 48)
        bad = subprocess.run(command + ["--seeds", "7", "7"], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("distinct", bad.stderr)


if __name__ == "__main__":
    unittest.main()
