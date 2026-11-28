import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from incident_ml.ensemble import ensemble_statistics, evaluate_ensemble, review_curve
from incident_ml.model import classification_metrics, tokenize
from incident_ml.validation import cross_validate


class EnsembleTests(unittest.TestCase):
    def rows(self):
        return [{"text": text, "label": label, "split": "train"} for text, label in
                [("fire smoke", "fire"), ("fire flames", "fire"), ("burning house", "fire"),
                 ("river flood", "flood"), ("flood water", "flood"), ("heavy rain", "flood")]]

    def test_opposing_confident_members_have_known_entropy_and_disagreement(self):
        stats = ensemble_statistics(np.array([[[1., 0.]], [[0., 1.]]]))
        np.testing.assert_allclose(stats["probabilities"], [[0.5, 0.5]])
        self.assertAlmostEqual(stats["predictive_entropy"][0], np.log(2))
        self.assertAlmostEqual(stats["mutual_information"][0], np.log(2))
        self.assertEqual(stats["vote_disagreement"][0], 0.5)

    def test_identical_members_have_zero_information_and_vote_disagreement(self):
        stats = ensemble_statistics(np.array([[[0.8, 0.2]], [[0.8, 0.2]]]))
        np.testing.assert_allclose(stats["mutual_information"], [0])
        np.testing.assert_allclose(stats["vote_disagreement"], [0])

    def test_review_curve_counts_errors_boundaries_and_empty_kept_set(self):
        values = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7]])
        curve = review_curve(np.array([0, 0, 1]), values, np.array([1., 1., 0.]), thresholds=[1., 0., 0.8])
        self.assertEqual([r["threshold"] for r in curve], [0., 0.8, 1.])
        self.assertEqual([r["kept"] for r in curve], [2, 1, 0])
        self.assertEqual(curve[0]["kept_error_rate"], 0.5)
        self.assertEqual(curve[1]["kept_error_rate"], 0)
        self.assertIsNone(curve[2]["kept_error_rate"])
        self.assertEqual(curve[2]["reviewed"], 3)

    def test_invalid_probabilities_and_targets_rejected(self):
        for values in [np.array([[0.2, 0.2]]), np.array([[float("nan"), 0.5]]), np.array([[-1., 2.]])]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ensemble_statistics(np.array([values, values]))
        for targets in [np.array([0.5]), np.array([2]), np.array([0, 1])]:
            with self.assertRaises(ValueError):
                review_curve(targets, np.array([[0.5, 0.5]]), np.array([1.]))

    def test_invalid_configuration(self):
        for options in [{"member_seeds": [7]}, {"member_seeds": [7, 7]}, {"member_seeds": [-1, 7]},
                        {"thresholds": []}, {"thresholds": [0.5, 0.5]}, {"thresholds": [float("nan")]},
                        {"min_coverage": float("nan")}, {"folds": 4}, {"epochs": 0}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluate_ensemble(self.rows(), **options)

    def test_reference_matches_standalone_cv_and_vocabulary_is_fold_local(self):
        rows = self.rows()
        report = evaluate_ensemble(rows, epochs=5)
        reference = cross_validate(rows, epochs=5)
        self.assertEqual(report["single_member_reference"]["metrics"], reference["models"]["neural"]["out_of_fold"])
        np.testing.assert_allclose(report["out_of_fold"]["member_probabilities"][0],
                                   reference["models"]["neural"]["out_of_fold_probabilities"])
        for fold in report["folds"]:
            fit = fold["fit_indices"]
            self.assertFalse(set(fit) & set(fold["held_out_indices"]))
            self.assertEqual(fold["vocabulary_words"], len({w for i in fit for w in tokenize(rows[i]["text"])}))

    def test_ensemble_mean_metrics_and_curve_recompute(self):
        report = evaluate_ensemble(self.rows(), epochs=5)
        raw = report["out_of_fold"]
        mean = np.array(raw["member_probabilities"]).mean(axis=0)
        np.testing.assert_allclose(mean, raw["ensemble_probabilities"])
        targets = np.array(raw["targets"])
        self.assertEqual(report["ensemble"]["metrics"], classification_metrics(targets, mean))
        self.assertEqual(report["ensemble"]["review_curve"], review_curve(targets, mean, np.array(raw["vocabulary_coverage"])))
        for row in report["ensemble"]["review_curve"]:
            self.assertEqual(row["kept"] + row["reviewed"], 6)

    def test_excluded_partitions_and_reproducibility(self):
        report = evaluate_ensemble(self.rows(), epochs=5)
        extras = [{"split": "test", "text": "excluded", "label": "new"}, {"split": "validation", "text": None}]
        self.assertEqual(report, evaluate_ensemble(self.rows() + extras, epochs=5))

    def test_duplicate_training_text_rejected(self):
        rows = self.rows()
        rows[1]["text"] = "FIRE, smoke!"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_ensemble(rows, epochs=5)

    def test_cli_success_and_invalid_seed_error(self):
        data = Path(__file__).resolve().parents[1] / "data/synthetic_incidents.jsonl"
        command = [sys.executable, "-m", "incident_ml.cli", "ensemble-evaluate", "--data", str(data), "--epochs", "5"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["training_records"], 48)
        self.assertEqual(len(report["members"]), 3)
        bad = subprocess.run(command + ["--member-seeds", "7", "7"], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
