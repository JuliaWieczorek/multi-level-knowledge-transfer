import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlkt.transfer import (
    DeterministicMockGenerator,
    LlamaCppGenerator,
    _json_array_schema_from_prompt,
    attach_segment_conversation_ids,
    augment_source_training,
    build_augmentation_plan,
    extract_style_patterns,
    resolve_gguf_model_path,
)


class InterruptingGenerator:
    name = "interrupting-mock"

    def __init__(self, successful_calls: int) -> None:
        self.successful_calls = successful_calls
        self.calls = 0
        self.delegate = DeterministicMockGenerator()

    def generate(self, prompt: str, max_tokens: int, seed: int) -> str:
        if self.calls >= self.successful_calls:
            raise KeyboardInterrupt
        self.calls += 1
        return self.delegate.generate(prompt, max_tokens, seed)


class TransferPipelineTests(unittest.TestCase):
    def test_outcome_prompt_builds_exact_length_json_grammar(self):
        schema = json.loads(
            _json_array_schema_from_prompt(
                "return a valid JSON array of exactly 7 strings and nothing else"
            )
        )
        self.assertEqual(schema["minItems"], 7)
        self.assertEqual(schema["maxItems"], 7)

    def test_local_gguf_resolution_does_not_require_hugging_face(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.touch()
            resolved, metadata = resolve_gguf_model_path(model)
        self.assertEqual(resolved, model.resolve())
        self.assertEqual(metadata["source"], "local")
        self.assertIsNone(metadata["hf_repo"])

    def test_gguf_generator_uses_embedded_chat_template(self):
        class FakeModel:
            def __init__(self) -> None:
                self.arguments = None

            def create_chat_completion(self, **arguments):
                self.arguments = arguments
                return {"choices": [{"message": {"content": '["rewritten"]'}}]}

        fake = FakeModel()
        generator = LlamaCppGenerator.__new__(LlamaCppGenerator)
        generator._model = fake
        generator._temperature = 0.5
        generator._top_p = 0.9
        result = generator.generate("Return JSON.", max_tokens=32, seed=42)
        self.assertEqual(result, '["rewritten"]')
        self.assertEqual(fake.arguments["messages"][1]["content"], "Return JSON.")
        self.assertEqual(fake.arguments["temperature"], 0.5)

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

    def test_plan_is_deterministic_and_resume_has_no_duplicates(self):
        esconv = pd.DataFrame(
            [
                {
                    "conversation_id": f"esconv_{index:04d}",
                    "split": "train",
                    "Utterances": "I cannot stop thinking about this situation",
                    "sentiment": "negative",
                    "emotion1": emotion,
                    "intensity1": 2,
                    "emotion2": "",
                    "intensity2": pd.NA,
                    "emotion3": "",
                    "intensity3": pd.NA,
                }
                for index, emotion in enumerate(("anger", "disgust"))
            ]
        )
        patterns = extract_style_patterns(esconv)
        rows = []
        for index, emotion in enumerate(("anger", "anger", "anger", "disgust")):
            rows.append(
                {
                    "conversation_id": f"meisd_{index:04d}",
                    "split": "train",
                    "Utterances": f"source message number {index}",
                    "sentiment": "negative",
                    "emotion1": emotion,
                    "intensity1": 2,
                    "emotion2": "",
                    "intensity2": pd.NA,
                    "emotion3": "",
                    "intensity3": pd.NA,
                }
            )
        meisd = pd.DataFrame(rows)
        _, _, first_plan, _ = build_augmentation_plan(
            meisd, patterns, min_compatible_samples=1
        )
        _, _, second_plan, _ = build_augmentation_plan(
            meisd, patterns, min_compatible_samples=1
        )
        self.assertEqual(first_plan, second_plan)
        self.assertEqual(len(first_plan), 2)

        with tempfile.TemporaryDirectory() as temporary:
            progress = Path(temporary) / "progress.jsonl"
            with self.assertRaises(KeyboardInterrupt):
                augment_source_training(
                    meisd,
                    patterns,
                    InterruptingGenerator(successful_calls=1),
                    min_compatible_samples=1,
                    progress_path=progress,
                    checkpoint_every=1,
                )
            with progress.open("a", encoding="utf-8") as handle:
                handle.write('{"truncated":')
            resumed, resumed_report = augment_source_training(
                meisd,
                patterns,
                DeterministicMockGenerator(),
                min_compatible_samples=1,
                progress_path=progress,
                checkpoint_every=1,
                resume=True,
            )
            uninterrupted, _ = augment_source_training(
                meisd,
                patterns,
                DeterministicMockGenerator(),
                min_compatible_samples=1,
            )

        resumed_augmented = resumed[resumed["augmented"]].reset_index(drop=True)
        uninterrupted_augmented = uninterrupted[
            uninterrupted["augmented"]
        ].reset_index(drop=True)
        self.assertEqual(len(resumed_augmented), 2)
        self.assertEqual(
            resumed_augmented["generation_seed"].tolist(),
            uninterrupted_augmented["generation_seed"].tolist(),
        )
        self.assertEqual(
            resumed_augmented["source_conversation_id"].tolist(),
            uninterrupted_augmented["source_conversation_id"].tolist(),
        )
        self.assertEqual(resumed_report["resume_count"], 1)


if __name__ == "__main__":
    unittest.main()
