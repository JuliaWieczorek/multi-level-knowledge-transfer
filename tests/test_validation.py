import unittest

import pandas as pd

from mlkt.validation import validate_source_training_frame


def _row(
    text: str,
    conversation: str,
    split: str,
    emotion: int,
    intensity: int,
    *,
    augmented: bool = False,
    valid: bool = True,
) -> dict:
    return {
        "Utterances": text,
        "sentiment": "negative",
        "conversation_id": conversation,
        "split": split,
        "emotion__sadness": emotion,
        "intensity__sadness": intensity,
        "augmented": augmented,
        "generation_valid": valid,
        "quality": 0.8,
    }


class SourceValidationTests(unittest.TestCase):
    def test_invalid_duplicates_and_conflicts_are_removed(self):
        frame = pd.DataFrame(
            [
                _row("original", "train-1", "train", 1, 2),
                _row("original", "train-1", "train", 1, 2, augmented=True),
                _row("failed", "train-2", "train", 1, 1, augmented=True, valid=False),
                _row("conflict", "train-3", "train", 1, 1, augmented=True),
                _row("conflict", "train-4", "train", 1, 3, augmented=True),
                _row("validation", "validation-1", "validation", 1, 2),
            ]
        )
        validated, report = validate_source_training_frame(frame, ["sadness"])
        self.assertEqual(validated["Utterances"].tolist(), ["original", "validation"])
        self.assertEqual(report["excluded_invalid_generation_rows"], 1)
        self.assertEqual(report["exact_train_duplicate_rows_dropped"], 1)
        self.assertEqual(report["conflicting_train_rows_dropped"], 2)

    def test_conversation_leakage_is_rejected(self):
        frame = pd.DataFrame(
            [
                _row("train", "shared", "train", 1, 2),
                _row("validation", "shared", "validation", 1, 2),
            ]
        )
        with self.assertRaisesRegex(ValueError, "conversation IDs overlap"):
            validate_source_training_frame(frame, ["sadness"])


if __name__ == "__main__":
    unittest.main()
