import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlkt.outcome_augmentation import (
    DeterministicOutcomeMockGenerator,
    build_outcome_augmentation_plan,
    outcome_rewrite_prompt,
    run_outcome_augmentation,
    validate_outcome_augmentation_frame,
)


def _row(
    conversation_id: str,
    split: str,
    final_intensity: int,
    drop_magnitude: int,
) -> dict:
    return {
        "dataset": "esconv",
        "conversation_id": conversation_id,
        "checkpoint": 1.0,
        "split": split,
        "text": "I feel stuck. Tell me more.",
        "text_role_turns": json.dumps(
            ["seeker: I feel stuck.", "supporter: Tell me more."]
        ),
        "text_seeker": "I feel stuck.",
        "text_seeker_turns": json.dumps(["I feel stuck."]),
        "text_supporter": "Tell me more.",
        "initial_intensity": final_intensity + drop_magnitude,
        "final_intensity": final_intensity,
        "drop_magnitude": drop_magnitude,
        "emotion_family": "sadness",
    }


class OutcomeAugmentationTests(unittest.TestCase):
    def test_plan_prioritises_the_rarest_joint_pair(self):
        rows = [
            _row("rare", "train", 4, 1),
            *[_row(f"common-{index}", "train", 2, 2) for index in range(4)],
        ]
        plan, report = build_outcome_augmentation_plan(
            pd.DataFrame(rows), fraction=0.4, seed=42
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["source_conversation_id"], "rare")
        self.assertEqual((plan[0]["final_intensity"], plan[0]["drop_magnitude"]), (4, 1))
        self.assertEqual(report["planned_conversations"], 2)

    def test_prompt_requests_label_preserving_json_without_supporter_rewrite(self):
        prompt = outcome_rewrite_prompt(["I cannot sleep."], "fear", 5, 3)
        self.assertIn("valid JSON array of exactly 1 strings", prompt)
        self.assertIn("Rewrite only the support seeker's utterances", prompt)
        self.assertIn("initial survey intensity category: 5", prompt)
        self.assertIn("never state these labels or numbers", prompt)

    def test_pilot_can_focus_zero_recall_pairs(self):
        rows = [
            *[_row(f"final-four-{index}", "train", 4, 1) for index in range(2)],
            *[_row(f"drop-three-a-{index}", "train", 1, 3) for index in range(3)],
            *[_row(f"drop-three-b-{index}", "train", 2, 3) for index in range(3)],
            *[_row(f"common-{index}", "train", 2, 2) for index in range(10)],
        ]
        plan, report = build_outcome_augmentation_plan(
            pd.DataFrame(rows),
            fraction=0.5,
            seed=42,
            max_conversations=6,
            focus_pairs=((4, 1), (1, 3), (2, 3)),
        )
        planned_pairs = {
            (item["final_intensity"], item["drop_magnitude"]) for item in plan
        }
        self.assertEqual(planned_pairs, {(4, 1), (1, 3), (2, 3)})
        self.assertEqual(report["focus_pairs"], [[1, 3], [2, 3], [4, 1]])

    def test_mock_pipeline_augments_train_only_and_preserves_labels(self):
        frame = pd.DataFrame(
            [
                _row("train-rare", "train", 4, 1),
                _row("train-common", "train", 2, 2),
                _row("train-common-2", "train", 2, 2),
                _row("validation", "validation", 2, 2),
                _row("test", "test", 1, 4),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "output"
            frame.to_csv(source, index=False)
            manifest = run_outcome_augmentation(
                input_path=source,
                output_dir=output,
                generator=DeterministicOutcomeMockGenerator(),
                fraction=0.5,
                max_conversations=1,
            )
            augmented = pd.read_csv(output / "esconv_outcome_augmented.csv")
        synthetic = augmented[augmented["augmented"].astype(str).str.lower() == "true"]
        self.assertEqual(len(synthetic), 1)
        self.assertEqual(synthetic.iloc[0]["split"], "train")
        self.assertEqual(int(synthetic.iloc[0]["final_intensity"]), 4)
        self.assertEqual(int(synthetic.iloc[0]["drop_magnitude"]), 1)
        self.assertIn("Please understand me", synthetic.iloc[0]["text_seeker"])
        untouched = augmented[augmented["split"].isin(["validation", "test"])]
        self.assertFalse(
            untouched["augmented"].astype(str).str.lower().eq("true").any()
        )
        self.assertEqual(manifest["valid_generated_conversations"], 1)
        with self.assertRaisesRegex(ValueError, "Mock outcome augmentation"):
            validate_outcome_augmentation_frame(augmented)

        augmented.loc[
            augmented["augmented"].astype(str).str.lower() == "true", "generator"
        ] = "llama-test"
        report = validate_outcome_augmentation_frame(augmented)
        self.assertEqual(report["augmented_rows"], 1)


if __name__ == "__main__":
    unittest.main()
