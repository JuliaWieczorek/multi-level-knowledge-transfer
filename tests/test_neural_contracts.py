import importlib.util
import unittest
from unittest.mock import patch


TORCH_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(TORCH_AVAILABLE, "neural optional dependencies not installed")
class NeuralContractTests(unittest.TestCase):
    def test_outcome_finite_guard_reports_stage_batch_and_conversations(self):
        import torch

        from mlkt.training import _assert_finite_tensors

        with self.assertRaisesRegex(
            FloatingPointError,
            r"stage=model_outputs, batch=20, tensor=joint_pair_logits.*conv-a",
        ):
            _assert_finite_tensors(
                (("joint_pair_logits", torch.tensor([[0.0, float("nan")]])),),
                stage="model_outputs",
                batch_index=20,
                conversation_ids=["conv-a"],
            )

    def test_turn_packing_preserves_order_and_boundaries(self):
        from mlkt.neural_data import pack_tokenized_turns

        chunks = pack_tokenized_turns(
            [[1, 2], [3, 4, 5], [6]],
            separator_id=99,
            max_content_tokens=5,
        )
        self.assertEqual(chunks, [[1, 2, 99, 3, 4], [5, 99, 6]])

    def test_token_preparation_supports_transformers_five_api(self):
        from mlkt.neural_data import prepare_token_chunk

        class ModernTokenizer:
            pad_token_id = 0
            padding_side = "right"

            @staticmethod
            def build_inputs_with_special_tokens(tokens):
                return [101, *tokens, 102]

        prepared = prepare_token_chunk(ModernTokenizer(), [7, 8], max_length=6)
        self.assertEqual(prepared["input_ids"], [101, 7, 8, 102, 0, 0])
        self.assertEqual(prepared["attention_mask"], [1, 1, 1, 1, 0, 0])

        class BertFiveTokenizer:
            cls_token_id = 101
            sep_token_id = 102
            bos_token_id = None
            eos_token_id = None
            pad_token_id = 0
            padding_side = "right"

        prepared = prepare_token_chunk(BertFiveTokenizer(), [7, 8], max_length=6)
        self.assertEqual(prepared["input_ids"], [101, 7, 8, 102, 0, 0])

    def test_role_aware_chunks_retain_speaker_and_seeker_phase(self):
        from mlkt.neural_data import tokenize_role_aware_chunks

        class DummyTokenizer:
            sep_token_id = 99
            eos_token_id = None

            @staticmethod
            def num_special_tokens_to_add(pair=False):
                return 2

            @staticmethod
            def encode(text, add_special_tokens=False):
                return [len(text)]

            @staticmethod
            def prepare_for_model(tokens, **kwargs):
                max_length = kwargs["max_length"]
                values = [101, *tokens, 102]
                padding = max_length - len(values)
                return {
                    "input_ids": values + [0] * padding,
                    "attention_mask": [1] * len(values) + [0] * padding,
                }

        chunks = tokenize_role_aware_chunks(
            DummyTokenizer(),
            ["seeker: one", "supporter: two", "seeker: three"],
            max_length=10,
        )
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0]["speaker_features"][0], 2 / 3)
        self.assertAlmostEqual(chunks[0]["speaker_features"][1], 1 / 3)
        self.assertAlmostEqual(chunks[0]["trajectory_features"][0], 1 / 3)
        self.assertAlmostEqual(chunks[0]["trajectory_features"][1], 1 / 3)

    def test_temporal_strategy_model_supports_batch_size_one(self):
        import torch

        from mlkt.models import TemporalMultiModalModel

        model = TemporalMultiModalModel(
            modality="strategy",
            transformer_name="unused",
            transfer_checkpoint=None,
            strategy_vocabulary_size=10,
            strategy_numeric_size=4,
            strategy_hidden_size=32,
            dropout=0.0,
        )
        output = model(
            {
                "strategy_ids": torch.tensor([[1]]),
                "strategy_positions": torch.tensor([[0.0]]),
                "strategy_mask": torch.tensor([[True]]),
                "strategy_numeric": torch.zeros((1, 4)),
            }
        )
        self.assertEqual(output["final_intensity"].shape, (1, 4))
        self.assertEqual(output["drop_magnitude"].shape, (1, 4))

    def test_temporal_model_supports_coarse_three_class_heads(self):
        import torch

        from mlkt.models import TemporalMultiModalModel

        model = TemporalMultiModalModel(
            modality="strategy",
            transformer_name="unused",
            transfer_checkpoint=None,
            strategy_vocabulary_size=10,
            strategy_numeric_size=4,
            strategy_hidden_size=32,
            dropout=0.0,
            num_outcome_classes=3,
        )
        output = model(
            {
                "strategy_ids": torch.tensor([[1]]),
                "strategy_positions": torch.tensor([[0.0]]),
                "strategy_mask": torch.tensor([[True]]),
                "strategy_numeric": torch.zeros((1, 4)),
            }
        )
        self.assertEqual(output["final_intensity"].shape, (1, 3))
        self.assertEqual(output["drop_magnitude"].shape, (1, 3))

    def test_ordinal_encoding_and_decoding_enforce_valid_drop(self):
        import torch

        from mlkt.models import (
            cumulative_ordinal_predictions,
            cumulative_ordinal_targets,
        )

        targets = cumulative_ordinal_targets(torch.tensor([1, 2, 4]))
        self.assertEqual(
            targets.tolist(),
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        )
        predictions = cumulative_ordinal_predictions(
            torch.tensor([[10.0, 10.0, 10.0], [10.0, -10.0, -10.0]]),
            initial_intensity=torch.tensor([3, 5]),
        )
        self.assertEqual(predictions.tolist(), [2, 2])

    def test_outcome_model_conditions_role_text_on_metadata(self):
        import torch
        from torch import nn

        from mlkt.models import EmotionConditionedOutcomeModel

        class DummyTextEncoder(nn.Module):
            hidden_size = 16

            def __init__(self, **kwargs):
                super().__init__()

            def forward(
                self,
                input_ids,
                attention_mask,
                chunk_mask,
                speaker_features=None,
                trajectory_features=None,
                return_trajectory=False,
            ):
                overall = torch.ones((input_ids.shape[0], self.hidden_size))
                if return_trajectory:
                    return {
                        "overall": overall,
                        "early": overall * 0.5,
                        "late": overall * 1.5,
                    }
                return overall

        with patch("mlkt.models.AffectiveTextEncoder", DummyTextEncoder):
            model = EmotionConditionedOutcomeModel(
                transformer_name="unused",
                transfer_checkpoint=None,
                emotion_vocabulary_size=5,
                problem_vocabulary_size=7,
                metadata_size=4,
                dropout=0.0,
            )
        output = model(
            {
                "input_ids": torch.ones((2, 1, 3), dtype=torch.long),
                "attention_mask": torch.ones((2, 1, 3), dtype=torch.long),
                "chunk_mask": torch.ones((2, 1), dtype=torch.bool),
                "emotion_id": torch.tensor([1, 2]),
                "problem_id": torch.tensor([2, 3]),
                "initial_intensity": torch.tensor([4, 5]),
            }
        )
        self.assertEqual(output["final_logits"].shape, (2, 4))
        self.assertEqual(output["drop_logits"].shape, (2, 4))
        self.assertEqual(output["joint_pair_logits"].shape, (2, 10))
        self.assertEqual(output["final_ordinal_logits"].shape, (2, 3))

    def test_full_outcome_model_fuses_trajectory_and_strategies(self):
        import torch
        from torch import nn

        from mlkt.models import EmotionConditionedOutcomeModel

        class DummyTextEncoder(nn.Module):
            hidden_size = 16

            def __init__(self, **kwargs):
                super().__init__()

            def forward(self, input_ids, attention_mask, chunk_mask, **kwargs):
                overall = torch.ones((input_ids.shape[0], self.hidden_size))
                return {
                    "overall": overall,
                    "early": overall * 0.5,
                    "late": overall * 1.5,
                }

        with patch("mlkt.models.AffectiveTextEncoder", DummyTextEncoder):
            model = EmotionConditionedOutcomeModel(
                transformer_name="unused",
                transfer_checkpoint=None,
                emotion_vocabulary_size=5,
                problem_vocabulary_size=7,
                metadata_size=4,
                strategy_vocabulary_size=10,
                strategy_numeric_size=4,
                strategy_hidden_size=32,
                use_speaker_features=True,
                use_trajectory=True,
                use_strategy=True,
                use_auxiliary_regression=True,
                dropout=0.0,
            )
        output = model(
            {
                "input_ids": torch.ones((2, 1, 3), dtype=torch.long),
                "attention_mask": torch.ones((2, 1, 3), dtype=torch.long),
                "chunk_mask": torch.ones((2, 1), dtype=torch.bool),
                "speaker_features": torch.ones((2, 1, 2)),
                "trajectory_features": torch.ones((2, 1, 2)),
                "strategy_ids": torch.tensor([[1, 2], [2, 0]]),
                "strategy_positions": torch.tensor([[0.0, 1.0], [0.5, 0.0]]),
                "strategy_mask": torch.tensor([[True, True], [True, False]]),
                "strategy_numeric": torch.zeros((2, 4)),
                "emotion_id": torch.tensor([1, 2]),
                "problem_id": torch.tensor([2, 3]),
                "initial_intensity": torch.tensor([4, 5]),
            }
        )
        self.assertEqual(output["final_ordinal_logits"].shape, (2, 3))
        self.assertEqual(output["drop_regression"].shape, (2,))

    def test_outcome_epoch_combines_ordinal_and_auxiliary_losses(self):
        import torch
        from torch import nn

        from mlkt.training import _run_outcome_ceiling_epoch

        class DummyOutcomeModel(nn.Module):
            def forward(self, batch):
                size = batch["final_target"].shape[0]
                return {
                    "final_logits": torch.zeros((size, 4)),
                    "drop_logits": torch.zeros((size, 4)),
                    "final_ordinal_logits": torch.zeros((size, 3)),
                    "drop_regression": torch.full((size,), 2.0),
                }

        loader = [
            {
                "conversation_id": ["a", "b"],
                "initial_intensity": torch.tensor([4, 3]),
                "final_target": torch.tensor([2, 1]),
                "drop_target": torch.tensor([2, 2]),
            }
        ]
        metrics, predictions = _run_outcome_ceiling_epoch(
            DummyOutcomeModel(),
            loader,
            torch.device("cpu"),
            torch.ones(3),
            auxiliary_regression_weight=0.2,
            final_class_weights=torch.ones(4),
            drop_class_weights=torch.ones(4),
        )
        self.assertIn("auxiliary_regression_loss", metrics)
        self.assertGreater(metrics["final_classification_loss"], 0.0)
        self.assertGreater(metrics["drop_classification_loss"], 0.0)
        self.assertIn("drop_regression", predictions)
        self.assertEqual(metrics["joint_consistency_rate"], 1.0)

    def test_structured_outcome_decoder_enforces_initial_identity(self):
        import torch

        from mlkt.training import _structured_outcome_predictions

        final_logits = torch.tensor(
            [[0.0, 0.0, 10.0, 0.0], [10.0, 0.0, 0.0, 0.0]]
        )
        drop_logits = torch.tensor(
            [[0.0, 10.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]]
        )
        initial = torch.tensor([5, 3])
        final, drop = _structured_outcome_predictions(
            final_logits, drop_logits, initial
        )
        self.assertEqual(final.tolist(), [3, 1])
        self.assertEqual(drop.tolist(), [2, 2])
        self.assertTrue(torch.equal(final + drop, initial))

    def test_direct_joint_decoder_masks_invalid_pairs_and_marginalises(self):
        import torch

        from mlkt.training import (
            _joint_marginal_probabilities,
            _joint_pair_predictions,
            _outcome_pair_targets,
        )

        targets = _outcome_pair_targets(
            torch.tensor([1, 4, 2]), torch.tensor([4, 1, 2])
        )
        self.assertEqual(targets.tolist(), [4, 10, 6])

        logits = torch.zeros((2, 10))
        logits[0, 9] = 100.0  # (4, 1) is invalid when initial intensity is 3.
        logits[0, 4] = 10.0   # (2, 1) is valid.
        logits[1, 3] = 10.0   # (1, 4) is valid when initial intensity is 5.
        final, drop, probabilities = _joint_pair_predictions(
            logits, torch.tensor([3, 5])
        )
        self.assertEqual(final.tolist(), [2, 1])
        self.assertEqual(drop.tolist(), [1, 4])
        self.assertTrue(torch.equal(final + drop, torch.tensor([3, 5])))
        final_probability, drop_probability = _joint_marginal_probabilities(
            probabilities
        )
        self.assertTrue(
            torch.allclose(final_probability.sum(dim=1), torch.ones(2))
        )
        self.assertTrue(
            torch.allclose(drop_probability.sum(dim=1), torch.ones(2))
        )

    def test_outcome_discriminative_learning_rates_do_not_overlap(self):
        import torch
        from torch import nn

        from mlkt.training import _outcome_parameter_groups

        class DummyOutcomeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_encoder = nn.Linear(3, 3)
                self.head = nn.Linear(3, 1)

        model = DummyOutcomeModel()
        groups = _outcome_parameter_groups(
            model,
            {
                "learning_rate": 2e-5,
                "encoder_learning_rate": 5e-6,
                "head_learning_rate": 2e-5,
            },
        )
        self.assertEqual([group["lr"] for group in groups], [5e-6, 2e-5])
        encoder_ids = {id(parameter) for parameter in groups[0]["params"]}
        head_ids = {id(parameter) for parameter in groups[1]["params"]}
        self.assertFalse(encoder_ids & head_ids)
        self.assertEqual(encoder_ids | head_ids, {id(p) for p in model.parameters()})

    def test_outcome_gradual_unfreeze_schedule(self):
        from mlkt.training import _outcome_trainable_layers

        config = {
            "freeze_text_encoder_epochs": 2,
            "gradual_unfreeze_layers": [4],
        }
        self.assertEqual(_outcome_trainable_layers(1, config), 0)
        self.assertEqual(_outcome_trainable_layers(2, config), 0)
        self.assertEqual(_outcome_trainable_layers(3, config), 4)
        self.assertIsNone(_outcome_trainable_layers(4, config))

        capped_config = {
            **config,
            "max_trainable_text_encoder_layers": 4,
        }
        self.assertEqual(_outcome_trainable_layers(3, capped_config), 4)
        self.assertEqual(_outcome_trainable_layers(4, capped_config), 4)

    def test_joint_outcome_sampler_upweights_rare_pairs(self):
        import pandas as pd

        from mlkt.training import _outcome_sampling_weights

        frame = pd.DataFrame(
            {
                "final_intensity": [1, 1, 1, 1, 4],
                "drop_magnitude": [2, 2, 2, 2, 1],
            }
        )
        weights = _outcome_sampling_weights(
            frame, strategy="joint", power=0.5, max_ratio=4.0
        )
        self.assertIsNotNone(weights)
        self.assertGreater(float(weights[-1]), float(weights[0]))
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertIsNone(_outcome_sampling_weights(frame, strategy="none"))

    def test_affective_encoder_exposes_only_top_layers(self):
        import torch
        from torch import nn

        from mlkt.models import AffectiveTextEncoder

        class DummyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Linear(2, 2)
                self.encoder = nn.Module()
                self.encoder.layer = nn.ModuleList(
                    [nn.Linear(2, 2) for _ in range(6)]
                )

        encoder = AffectiveTextEncoder.__new__(AffectiveTextEncoder)
        nn.Module.__init__(encoder)
        encoder.emotion_encoder = DummyBackbone()
        encoder.intensity_encoder = DummyBackbone()
        encoder.gate = nn.Linear(4, 2)
        encoder.chunk_positions = nn.Embedding(4, 2)
        encoder.speaker_projection = nn.Linear(2, 2)
        encoder.chunk_encoder = nn.Linear(2, 2)
        encoder.set_trainable_layers(2)

        for backbone in (encoder.emotion_encoder, encoder.intensity_encoder):
            self.assertFalse(any(p.requires_grad for p in backbone.embeddings.parameters()))
            self.assertFalse(
                any(p.requires_grad for layer in backbone.encoder.layer[:-2] for p in layer.parameters())
            )
            self.assertTrue(
                all(p.requires_grad for layer in backbone.encoder.layer[-2:] for p in layer.parameters())
            )
        self.assertTrue(all(p.requires_grad for p in encoder.gate.parameters()))


if __name__ == "__main__":
    unittest.main()
