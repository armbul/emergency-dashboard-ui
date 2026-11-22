import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from incident_ml.cli import read_dataset
from incident_ml.model import IncidentClassifier, classification_metrics, softmax, tokenize


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.texts = ["fire flames smoke", "burning fire building", "river flood water", "flood rain river"]
        self.labels = ["fire", "fire", "flood", "flood"]
        self.model = IncidentClassifier.from_training_data(self.texts, self.labels)

    def test_unicode_tokenization(self):
        self.assertEqual(tokenize("MOTO! nyumba, maji 123"), ["moto", "nyumba", "maji"])
        self.assertEqual(tokenize("café"), ["café"])

    def test_softmax_stability(self):
        p = softmax(np.array([[10000., 10001.]]))
        np.testing.assert_allclose(p.sum(axis=1), [1])
        self.assertTrue(np.isfinite(p).all())

    def test_gradient_matches_finite_differences(self):
        x, _ = self.model.encode(self.texts)
        y = np.array([0, 0, 1, 1])
        _, gradients = self.model.loss_and_gradients(x, y)
        rng = np.random.default_rng(11)
        # Directional derivative checks cover every parameter, not just the output layer.
        for name, parameter in self.model.parameters.items():
            direction = rng.normal(size=parameter.shape)
            if name == "embedding":
                direction[0] = 0
            epsilon = 1e-6
            parameter += epsilon * direction
            upper, _ = self.model.loss_and_gradients(x, y)
            parameter -= 2 * epsilon * direction
            lower, _ = self.model.loss_and_gradients(x, y)
            parameter += epsilon * direction
            self.assertAlmostEqual((upper - lower) / (2 * epsilon), float((gradients[name] * direction).sum()), places=5, msg=name)

    def test_training_learns_and_is_reproducible(self):
        other = IncidentClassifier.from_training_data(self.texts, self.labels)
        losses = self.model.fit(self.texts, self.labels, epochs=80)
        other.fit(self.texts, self.labels, epochs=80)
        self.assertLess(losses[-1], losses[0] * 0.1)
        self.assertEqual([r["suggested_label"] for r in self.model.predict(self.texts)], self.labels)
        for name in self.model.parameters:
            np.testing.assert_array_equal(self.model.parameters[name], other.parameters[name])

    def test_empty_and_unknown_input_require_review(self):
        self.model.fit(self.texts, self.labels, epochs=30)
        results = self.model.predict(["", "xyzzy quux"], threshold=0)
        self.assertIn("empty_input", results[0]["review_reasons"])
        self.assertIn("low_vocabulary_coverage", results[1]["review_reasons"])
        self.assertEqual(self.model.predict([]), [])
        with self.assertRaises(ValueError):
            self.model.predict(["fire"], threshold=2)

    def test_checkpoint_roundtrip(self):
        self.model.fit(self.texts, self.labels, epochs=40)
        self.model.calibrate(["fire smoke", "river water"], ["fire", "flood"])
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save(tmp)
            loaded = IncidentClassifier.load(tmp)
            self.assertEqual(loaded.predict(self.texts), self.model.predict(self.texts))

    def test_corrupt_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save(tmp)
            path = Path(tmp, "metadata.json")
            data = json.loads(path.read_text())
            data["temperature"] = 0
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                IncidentClassifier.load(tmp)

    def test_metrics_include_missing_classes(self):
        report = classification_metrics(np.array([0, 0]), np.array([[1., 0.], [1., 0.]]))
        self.assertEqual(report["accuracy"], 1)
        self.assertEqual(report["macro_f1"], 0.5)

    def test_validation_does_not_expand_vocabulary(self):
        before = self.model.vocabulary.copy()
        self.model.calibrate(["unknownword fire", "flood"], ["fire", "flood"])
        self.assertEqual(before, self.model.vocabulary)

    def test_dataset_detects_normalized_duplicates(self):
        rows = [{"text": "Fire!", "label": "fire", "split": "train"}, {"text": "fire", "label": "fire", "split": "test"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "data.jsonl")
            path.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_dataset(path)

    def test_unknown_labels_and_invalid_training_rejected(self):
        with self.assertRaises(ValueError):
            self.model.fit(self.texts, self.labels, epochs=0)
        with self.assertRaises(ValueError):
            self.model.fit(["fire"], ["unseen"])
        with self.assertRaises(ValueError):
            self.model.calibrate([], [])


if __name__ == "__main__":
    unittest.main()
