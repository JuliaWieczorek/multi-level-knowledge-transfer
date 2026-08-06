import json
import unittest

import pandas as pd

from mlkt.data import (
    assert_checkpoint_integrity,
    build_esconv_checkpoints,
    build_meisd_checkpoints,
)
from mlkt.splits import assign_conversation_splits, assert_no_split_leakage


class DataPipelineTests(unittest.TestCase):
    def test_esconv_prefixes_and_strategy_features(self):
        source = [
            {
                "emotion_type": "anxiety",
                "problem_type": "job crisis",
                "survey_score": {
                    "seeker": {
                        "initial_emotion_intensity": "5",
                        "final_emotion_intensity": "2",
                    }
                },
                "dialog": [
                    {"speaker": "seeker", "annotation": {}, "content": "one"},
                    {
                        "speaker": "supporter",
                        "annotation": {"strategy": "Question"},
                        "content": "two",
                    },
                    {"speaker": "seeker", "annotation": {}, "content": "three"},
                    {
                        "speaker": "supporter",
                        "annotation": {"strategy": "Reflection of feelings"},
                        "content": "four",
                    },
                ],
            }
        ]
        frame = build_esconv_checkpoints(
            source, checkpoints=(0.25, 0.50, 1.00)
        )
        self.assertEqual(frame["n_observed_turns"].tolist(), [1, 2, 4])
        self.assertEqual(
            frame["text"].tolist(), ["one", "one two", "one two three four"]
        )
        self.assertEqual(set(frame["intensity_change"]), {"decrease"})
        self.assertEqual(frame.iloc[-1]["strategy_count__question"], 1)
        self.assertEqual(
            frame.iloc[-1]["strategy_count__reflection_of_feelings"], 1
        )
        self.assertEqual(frame.iloc[-1]["drop_magnitude"], 3)
        self.assertEqual(
            json.loads(frame.iloc[-1]["text_seeker_turns"]),
            ["one", "three"],
        )
        halfway = frame[frame["checkpoint"] == 0.5].iloc[0]
        self.assertEqual(halfway["strategy_sequence"], "Question")
        self.assertEqual(
            halfway["strategy_count__reflection_of_feelings"], 0
        )
        earliest = frame[frame["checkpoint"] == 0.25].iloc[0]
        self.assertEqual(earliest["n_observed_strategies"], 0)
        assert_checkpoint_integrity(
            frame, checkpoints=(0.25, 0.50, 1.00)
        )

    def test_meisd_scalar_targets_and_three_change_classes(self):
        rows = []
        examples = [
            ("A", 1, [3, 1]),
            ("A", 2, [1, 1]),
            ("B", 1, [1, 3]),
        ]
        for series, dialog_id, intensities in examples:
            for turn_id, intensity in enumerate(intensities):
                rows.append(
                    {
                        "TV Series": series,
                        "dialog_ids": dialog_id,
                        "uttr_ids": turn_id,
                        "Utterances": f"turn {turn_id}",
                        "emotion": "sadness",
                        "intensity": intensity,
                    }
                )
        frame = build_meisd_checkpoints(
            pd.DataFrame(rows), checkpoints=(0.5, 1.0)
        )
        final_rows = frame[frame["checkpoint"] == 1.0]
        self.assertEqual(
            set(final_rows["intensity_change"]),
            {"decrease", "same", "increase"},
        )
        assert_checkpoint_integrity(frame, checkpoints=(0.5, 1.0))

    def test_conversation_level_split_has_no_checkpoint_leakage(self):
        rows = []
        for label in (1, 2, 3):
            for conversation in range(10):
                for checkpoint in (0.1, 0.5, 1.0):
                    rows.append(
                        {
                            "dataset": "synthetic",
                            "conversation_id": f"{label}_{conversation}",
                            "checkpoint": checkpoint,
                            "final_intensity": label,
                        }
                    )
        split = assign_conversation_splits(pd.DataFrame(rows), seed=7)
        assert_no_split_leakage(split)
        self.assertEqual(set(split["split"]), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
