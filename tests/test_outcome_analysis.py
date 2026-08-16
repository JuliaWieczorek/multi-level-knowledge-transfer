import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlkt.outcome_analysis import analyze_outcome_errors


class OutcomeErrorAnalysisTests(unittest.TestCase):
    def test_error_analysis_reports_zero_recall_and_joint_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = pd.DataFrame(
                {
                    "conversation_id": ["a", "b", "c", "d"],
                    "initial_intensity": [3, 4, 5, 5],
                    "final_target": [1, 2, 3, 4],
                    "final_prediction": [1, 2, 2, 2],
                    "drop_target": [2, 2, 2, 1],
                    "drop_prediction": [2, 2, 3, 3],
                    "split": ["test"] * 4,
                }
            )
            checkpoint_rows = []
            for split in ("train", "test"):
                for index, (initial, final) in enumerate(
                    ((3, 1), (4, 2), (5, 3), (5, 4))
                ):
                    checkpoint_rows.append(
                        {
                            "conversation_id": (
                                chr(ord("a") + index)
                                if split == "test"
                                else f"train_{index}"
                            ),
                            "checkpoint": 1.0,
                            "split": split,
                            "emotion_family": "fear",
                            "problem_type": "academic pressure",
                            "final_intensity": final,
                            "drop_magnitude": initial - final,
                        }
                    )
            predictions_path = root / "predictions.csv"
            checkpoints_path = root / "checkpoints.csv"
            output = root / "analysis"
            predictions.to_csv(predictions_path, index=False)
            pd.DataFrame(checkpoint_rows).to_csv(checkpoints_path, index=False)

            summary = analyze_outcome_errors(
                predictions_path, checkpoints_path, output, min_group_size=1
            )

            self.assertIn("final_intensity=4", summary["zero_recall_classes"])
            support = pd.read_csv(output / "training_support.csv")
            self.assertIn("joint_pair", set(support["target"]))
            saved = json.loads((output / "summary.json").read_text())
            self.assertEqual(saved["examples"], 4)


if __name__ == "__main__":
    unittest.main()
