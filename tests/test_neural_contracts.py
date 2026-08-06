import importlib.util
import unittest


TORCH_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(TORCH_AVAILABLE, "neural optional dependencies not installed")
class NeuralContractTests(unittest.TestCase):
    def test_turn_packing_preserves_order_and_boundaries(self):
        from mlkt.neural_data import pack_tokenized_turns

        chunks = pack_tokenized_turns(
            [[1, 2], [3, 4, 5], [6]],
            separator_id=99,
            max_content_tokens=5,
        )
        self.assertEqual(chunks, [[1, 2, 99, 3, 4], [5, 99, 6]])

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


if __name__ == "__main__":
    unittest.main()
