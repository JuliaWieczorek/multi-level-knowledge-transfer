import unittest

import numpy as np
import pandas as pd

from mlkt.reporting import aggregate_seed_metrics, paired_transfer_deltas


class ReportingTests(unittest.TestCase):
    def test_seed_aggregation_and_paired_transfer_delta(self):
        rows = []
        for seed in (42, 52):
            for transfer, score in ((False, 0.40), (True, 0.50)):
                rows.append(
                    {
                        "checkpoint": 0.1,
                        "checkpoint_percent": 10,
                        "split": "test",
                        "target": "final_intensity",
                        "modality": "text",
                        "transfer": transfer,
                        "seed": seed,
                        "f1_macro": score,
                        "accuracy": score,
                        "f1_weighted": score,
                        "mae": 1.0,
                        "quadratic_weighted_kappa": 0.0,
                    }
                )
        frame = pd.DataFrame(rows)
        summary = aggregate_seed_metrics(frame)
        self.assertEqual(set(summary["n_seeds"]), {2})
        deltas = paired_transfer_deltas(frame)
        self.assertTrue(
            np.allclose(
                deltas["f1_macro_delta_transfer_minus_vanilla"], 0.10
            )
        )


if __name__ == "__main__":
    unittest.main()
