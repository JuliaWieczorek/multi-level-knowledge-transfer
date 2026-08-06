import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlkt.transfer import (
    DeterministicMockGenerator,
    attach_segment_conversation_ids,
    augment_source_training,
    extract_style_patterns,
)


class TransferPipelineTests(unittest.TestCase):
    def test_segment_ids_preserve_pairs_and_allow_meisd_singletons(self):
        esconv = pd.DataFrame(
            {"segment": ["start", "end", "start", "end"]}
        )
        attached = attach_segment_conversation_ids(
            esconv, dataset="esconv", require_pairs=True
        )
        self.assertEqual(
            attached["conversation_id"].tolist(),
            ["esconv_0000", "esconv_0000", "esconv_0001", "esconv_0001"],
        )
        meisd = pd.DataFrame({"segment": ["start", "start", "end"]})
        attached_meisd = attach_segment_conversation_ids(
            meisd, dataset="meisd", require_pairs=False
        )
        self.assertEqual(attached_meisd["conversation_id"].nunique(), 2)

    def test_style_patterns_use_train_split_only(self):
        frame = pd.DataFrame(
            [
                {
                    "conversation_id": "esconv_0000",
                    "split": "train",
                    "Utterances": "I keep checking every detail and cannot relax",
                    "sentiment": "negative",
                    "emotion1": "anxiety",
                    "intensity1": 2,
                    "emotion2": "",
                    "intensity2": pd.NA,
                    "emotion3": "",
                    "intensity3": pd.NA,
                },
                {
                    "conversation_id": "esconv_0001",
                    "split": "test",
                    "Utterances": "forbiddenvalidationtoken appears only here",
                    "sentiment": "negative",
                    "emotion1": "anxiety",
                    "intensity1": 2,
                    "emotion2": "",
                    "intensity2": pd.NA,
                    "emotion3": "",
                    "intensity3": pd.NA,
                },
            ]
        )
        patterns = extract_style_patterns(frame)
        serialised = str(patterns)
        self.assertNotIn("forbiddenvalidationtoken", serialised)
        self.assertEqual(patterns["metadata"]["source_split"], "train")

    def test_augmentation_never_touches_validation(self):
        esconv = pd.DataFrame(
            [
                {
                    "conversation_id": "esconv_0000",
                    "split": "train",
                    "Utterances": "I cannot stop thinking about the situation",
                    "sentiment": "negative",
                    "emotion1": "anger",
                    "intensity1": 2,
                    "emotion2": "",
                    "intensity2": pd.NA,
                    "emotion3": "",
                    "intensity3": pd.NA,
                }
                for _ in range(5)
            ]
        )
        patterns = extract_style_patterns(esconv)
        rows = []
        for index in range(7):
            rows.append(
                {
                    "conversation_id": f"meisd_{index:04d}",
                    "split": "train" if index < 6 else "validation",
                    "Utterances": f"source message number {index}",
                    "sentiment": "negative",
                    "emotion1": "anger",
                    "intensity1": 2,
                    "emotion2": "",
                    "intensity2": pd.NA,
                    "emotion3": "",
                    "intensity3": pd.NA,
                }
            )
        augmented, report = augment_source_training(
            pd.DataFrame(rows),
            patterns,
            DeterministicMockGenerator(),
            max_aug_per_group=2,
        )
        validation = augmented[augmented["split"] == "validation"]
        self.assertFalse(validation["augmented"].any())
        self.assertEqual(report["validation_rows"], 1)


if __name__ == "__main__":
    unittest.main()
