import unittest

import pandas as pd

from mlkt.baseline import run_outcome_metadata_baselines
from mlkt.outcome_labels import COARSE_3, relabel_outcome_frame


class OutcomeBaselineTests(unittest.TestCase):
    def test_metadata_baseline_derives_consistent_drop(self):
        rows = []
        labels = [(5, 2, "fear"), (4, 1, "sadness"), (3, 2, "anger")]
        for split, repeats in (("train", 8), ("validation", 2), ("test", 2)):
            for initial, final, emotion in labels:
                for index in range(repeats):
                    rows.append(
                        {
                            "dataset": "esconv",
                            "conversation_id": f"{split}_{initial}_{index}",
                            "checkpoint": 1.0,
                            "split": split,
                            "initial_intensity": initial,
                            "emotion_family": emotion,
                            "problem_type": "test problem",
                            "final_intensity": final,
                            "drop_magnitude": initial - final,
                        }
                    )
        metrics, predictions = run_outcome_metadata_baselines(pd.DataFrame(rows))
        self.assertEqual(set(metrics["task"]), {"final_intensity", "drop_magnitude"})
        self.assertTrue(
            (
                predictions["final_prediction"] + predictions["drop_prediction"]
                == predictions["initial_intensity"]
            ).all()
        )
        self.assertTrue(predictions["drop_prediction"].between(1, 4).all())

    def test_metadata_baseline_supports_coarse_three_class_targets(self):
        rows = []
        labels = [(5, 1, "fear"), (5, 3, "sadness"), (5, 4, "anger")]
        for split, repeats in (("train", 8), ("validation", 2), ("test", 2)):
            for initial, final, emotion in labels:
                for index in range(repeats):
                    rows.append(
                        {
                            "dataset": "esconv",
                            "conversation_id": f"{split}_{initial}_{final}_{index}",
                            "checkpoint": 1.0,
                            "split": split,
                            "initial_intensity": initial,
                            "emotion_family": emotion,
                            "problem_type": "test problem",
                            "final_intensity": final,
                            "drop_magnitude": initial - final,
                        }
                    )
        frame = relabel_outcome_frame(pd.DataFrame(rows), COARSE_3)
        metrics, predictions = run_outcome_metadata_baselines(frame)
        self.assertEqual(set(metrics["label_scheme"]), {COARSE_3})
        self.assertTrue(predictions["final_prediction"].between(1, 3).all())
        self.assertTrue(predictions["drop_prediction"].between(1, 3).all())


if __name__ == "__main__":
    unittest.main()
