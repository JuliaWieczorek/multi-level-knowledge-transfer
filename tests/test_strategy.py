import unittest

import numpy as np
import pandas as pd

from mlkt.strategy import (
    _independent_columns_without_constant,
    benjamini_hochberg,
    encode_strategy_sequence,
    ensure_strategy_columns,
    parse_strategy_positions,
    parse_strategy_sequence,
    retrospective_ordinal_analysis,
    strategy_feature_columns,
    strategy_vocabulary,
)


class StrategyFeatureTests(unittest.TestCase):
    def test_empty_sequence_uses_explicit_no_strategy_token(self):
        identifiers, positions = encode_strategy_sequence([], [])
        self.assertEqual(identifiers, [strategy_vocabulary()["NO_STRATEGY"]])
        self.assertEqual(positions, [0.0])

    def test_sequence_and_positions_round_trip(self):
        sequence = parse_strategy_sequence(
            "Question > Reflection of feelings"
        )
        positions = parse_strategy_positions("0.25 | 0.75", len(sequence))
        identifiers, encoded_positions = encode_strategy_sequence(
            sequence, positions
        )
        self.assertEqual(len(identifiers), 2)
        self.assertEqual(encoded_positions, [0.25, 0.75])

    def test_all_fixed_features_are_materialised(self):
        frame = ensure_strategy_columns(pd.DataFrame([{}]))
        columns = strategy_feature_columns("quantity_timing_order")
        self.assertTrue(set(columns).issubset(frame.columns))
        self.assertEqual(frame[columns].isna().sum().sum(), 0)

    def test_fdr_adjustment_is_monotonic_in_rank(self):
        raw = np.array([0.01, 0.04, 0.03, 0.20])
        adjusted = benjamini_hochberg(raw)
        order = np.argsort(raw)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= -1e-12))
        self.assertTrue(np.all(adjusted >= raw))

    def test_design_selection_excludes_implicit_constant(self):
        design = pd.DataFrame(
            {
                "group_a": [1.0, 0.0, 1.0, 0.0],
                "group_b": [0.0, 1.0, 0.0, 1.0],
                "trend": [0.0, 1.0, 2.0, 3.0],
            }
        )
        selected = _independent_columns_without_constant(design)
        augmented = np.column_stack(
            [
                np.ones(len(design), dtype=float),
                design[selected].to_numpy(dtype=float),
            ]
        )

        self.assertEqual(np.linalg.matrix_rank(augmented), len(selected) + 1)
        self.assertEqual(len({"group_a", "group_b"}.intersection(selected)), 1)
        self.assertIn("trend", selected)

    def test_retrospective_rejects_negative_bootstrap_count(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            retrospective_ordinal_analysis(
                pd.DataFrame(), "unused", bootstrap_samples=-1
            )


if __name__ == "__main__":
    unittest.main()
