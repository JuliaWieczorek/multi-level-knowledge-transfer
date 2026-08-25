import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlkt.outcome_augmentation import (
    DeterministicOutcomeMockGenerator,
    build_augmented_temporal_checkpoints,
    build_outcome_augmentation_plan,
    outcome_rewrite_prompt,
    run_outcome_augmentation,
    validate_outcome_augmentation_frame,
    _split_seeker_windows,
    _preserves_protected_concepts,
    _short_turn_keeps_content_anchor,
    _apply_novelty_gate,
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
    def test_augmented_temporal_checkpoints_follow_source_boundaries(self):
        half = _row("source", "train", 2, 2)
        half.update(
            {
                "checkpoint": 0.5,
                "augmented": False,
            }
        )
        full = _row("source", "train", 2, 2)
        full.update(
            {
                "text": "I feel stuck. Tell me more. It is still difficult.",
                "text_role_turns": json.dumps(
                    [
                        "seeker: I feel stuck.",
                        "supporter: Tell me more.",
                        "seeker: It is still difficult.",
                    ]
                ),
                "text_seeker": "I feel stuck. It is still difficult.",
                "text_seeker_turns": json.dumps(
                    ["I feel stuck.", "It is still difficult."]
                ),
                "augmented": False,
            }
        )
        synthetic = {
            **full,
            "conversation_id": "source__outcome_aug_0000",
            "source_conversation_id": "source",
            "text_seeker": "I remain trapped. This continues to be difficult.",
            "text_seeker_turns": json.dumps(
                ["I remain trapped.", "This continues to be difficult."]
            ),
            "augmented": True,
            "generation_valid": True,
            "generation_seed": 42,
            "generator": "llama-test",
            "changed_seeker_turn_share": 1.0,
            "source_text_similarity": 0.1,
            "min_changed_turn_share_required": 0.6,
            "max_source_text_similarity_allowed": 0.92,
        }
        checkpoints, report = build_augmented_temporal_checkpoints(
            pd.DataFrame([half, full, synthetic])
        )
        generated = checkpoints[
            checkpoints["augmented"].astype(str).str.lower().eq("true")
        ].sort_values("checkpoint")
        self.assertEqual(len(generated), 2)
        self.assertEqual(
            json.loads(generated.iloc[0]["text_seeker_turns"]),
            ["I remain trapped."],
        )
        self.assertEqual(
            json.loads(generated.iloc[1]["text_seeker_turns"]),
            ["I remain trapped.", "This continues to be difficult."],
        )
        self.assertEqual(report["synthetic_checkpoint_rows"], 2)

    def test_novelty_gate_rejects_near_and_exact_duplicates(self):
        source = pd.Series(_row("source", "train", 4, 1))
        near_duplicate = {
            "text_seeker": "I feel rather stuck.",
            "changed_seeker_turn_share": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "too similar"):
            _apply_novelty_gate(near_duplicate, source, set(), 0.6, 0.5)

        exact_existing = {
            "text_seeker": "A different existing conversation.",
            "changed_seeker_turn_share": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "exact seeker-text duplicate"):
            _apply_novelty_gate(
                exact_existing,
                source,
                {"a different existing conversation."},
                0.6,
                0.99,
            )

    def test_novelty_gate_rejects_too_many_unchanged_turns(self):
        source = pd.Series(_row("source", "train", 4, 1))
        generated = {
            "text_seeker": "I remain trapped in this situation.",
            "changed_seeker_turn_share": 0.4,
        }
        with self.assertRaisesRegex(ValueError, "too few seeker turns changed"):
            _apply_novelty_gate(generated, source, set(), 0.6, 0.99)

    def test_conservative_semantic_guards_reject_action_substitution(self):
        self.assertFalse(
            _short_turn_keeps_content_anchor("riding a motorcyle", "lying in bed")
        )
        self.assertFalse(
            _preserves_protected_concepts("or I could steal one", "I could borrow one")
        )
        self.assertTrue(
            _preserves_protected_concepts("my brother murdered our mom", "he killed her")
        )

    def test_single_turn_windows_preserve_item_alignment(self):
        items = [
            {"seeker": "one", "preceding_supporter": "context"},
            {"seeker": "two", "preceding_supporter": "context"},
        ]
        windows = _split_seeker_windows(
            items, max_prompt_words=100, max_turns_per_window=1
        )
        self.assertEqual([len(window) for window in windows], [1, 1])

    def test_resume_retries_invalid_generation_records(self):
        frame = pd.DataFrame([_row("train-rare", "train", 4, 1)])

        class InvalidOnceGenerator:
            name = "invalid-once"

            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt: str, max_tokens: int, seed: int) -> str:
                self.calls += 1
                if self.calls == 1:
                    return "not-json"
                return '["I still feel stuck."]'

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "output"
            frame.to_csv(source, index=False)
            first_generator = InvalidOnceGenerator()
            first = run_outcome_augmentation(
                input_path=source,
                output_dir=output,
                generator=first_generator,
                fraction=1.0,
                max_generation_attempts=1,
            )
            second_generator = InvalidOnceGenerator()
            second_generator.calls = 1
            second = run_outcome_augmentation(
                input_path=source,
                output_dir=output,
                generator=second_generator,
                fraction=1.0,
                max_generation_attempts=1,
                resume=True,
            )

        self.assertEqual(first["valid_generated_conversations"], 0)
        self.assertEqual(second["valid_generated_conversations"], 1)
        self.assertEqual(second["invalid_loaded_for_retry"], 1)

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

    def test_plan_never_reuses_a_source_conversation(self):
        rows = [
            _row("only-rare", "train", 4, 1),
            *[_row(f"common-{index}", "train", 2, 2) for index in range(4)],
        ]
        plan, report = build_outcome_augmentation_plan(
            pd.DataFrame(rows), fraction=1.0, seed=42
        )
        source_ids = [item["source_conversation_id"] for item in plan]
        self.assertEqual(len(source_ids), 5)
        self.assertEqual(len(source_ids), len(set(source_ids)))
        self.assertFalse(report["source_reuse"])

    def test_prompt_requests_label_preserving_json_without_supporter_rewrite(self):
        prompt = outcome_rewrite_prompt(["I cannot sleep."], "fear", 5, 3)
        self.assertIn("valid JSON array of exactly 1 strings", prompt)
        self.assertIn("Rewrite only the support seeker's utterances", prompt)
        self.assertIn("initial survey intensity category: 5", prompt)
        self.assertIn("never state these labels or numbers", prompt)
        self.assertIn("keep short utterances short", prompt)
        self.assertIn('"minimum_rewrite_words": 1', prompt)
        self.assertIn('"maximum_rewrite_words": 7', prompt)
        self.assertIn("smallest wording", prompt)
        self.assertIn("never answer it", prompt)

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
