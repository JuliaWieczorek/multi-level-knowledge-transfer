import unittest

import pandas as pd

from mlkt.baseline import run_outcome_metadata_baselines


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


if __name__ == "__main__":
    unittest.main()
