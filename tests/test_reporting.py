import unittest

import numpy as np
import pandas as pd

from mlkt.reporting import aggregate_seed_metrics, paired_transfer_deltas
from mlkt.metrics import (
    classification_metrics,
    joint_prediction_diagnostics,
    multilabel_metrics,
)


class ReportingTests(unittest.TestCase):
    def test_classification_macro_f1_uses_fixed_label_space(self):
        metrics = classification_metrics([1, 1], [1, 1], labels=(1, 2, 3, 4))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_macro"], 0.25)

    def test_multilabel_metrics_keep_emotion_dimension(self):
        true = np.asarray([[1, 0], [0, 1]])
        predicted = np.asarray([[1, 0], [1, 0]])
        scores = np.asarray([[0.9, 0.1], [0.7, 0.4]])
        metrics = multilabel_metrics(true, predicted, scores)
        self.assertEqual(metrics["subset_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["f1_macro"], 1 / 3)

    def test_joint_diagnostics_detect_inconsistency_and_invalid_drop(self):
        metrics, arrays = joint_prediction_diagnostics(
            initial_intensity=[4, 2, 5],
            final_prediction=[2, 2, 1],
            drop_target=[2, 1, 4],
            drop_prediction=[2, 1, 3],
        )
        self.assertAlmostEqual(metrics["joint_consistency_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["derived_drop_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["invalid_derived_drop_rate"], 1 / 3)
        self.assertEqual(arrays["derived_drop_prediction"].tolist(), [2, 0, 4])

    def test_seed_aggregation_and_paired_transfer_delta(self):
        rows = []
        for seed in (42, 52):
            for transfer, score in ((False, 0.40), (True, 0.50)):
                rows.append(
                    {
                        "checkpoint": 0.1,
                        "checkpoint_percent": 10,
                        "split": "test",
                        "target": "final_intensity",
                        "modality": "text",
                        "transfer": transfer,
                        "seed": seed,
                        "f1_macro": score,
                        "accuracy": score,
                        "f1_weighted": score,
                        "mae": 1.0,
                        "quadratic_weighted_kappa": 0.0,
                    }
                )
        frame = pd.DataFrame(rows)
        summary = aggregate_seed_metrics(frame)
        self.assertEqual(set(summary["n_seeds"]), {2})
        deltas = paired_transfer_deltas(frame)
        self.assertTrue(
            np.allclose(
                deltas["f1_macro_delta_transfer_minus_vanilla"], 0.10
            )
        )


if __name__ == "__main__":
    unittest.main()
