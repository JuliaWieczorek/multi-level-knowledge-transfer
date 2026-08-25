import unittest

import pandas as pd

from mlkt.metrics import joint_prediction_diagnostics
from mlkt.outcome_labels import (
    COARSE_3,
    get_outcome_label_scheme,
    map_outcome_values,
    relabel_outcome_frame,
)


class OutcomeLabelSchemeTests(unittest.TestCase):
    def test_coarse3_maps_five_point_values_to_three_ordered_classes(self):
        mapped = map_outcome_values([1, 2, 3, 4, 5], COARSE_3)
        self.assertEqual(mapped.tolist(), [1, 1, 2, 3, 3])
        scheme = get_outcome_label_scheme(COARSE_3)
        self.assertEqual(scheme.labels, (1, 2, 3))
        self.assertEqual(scheme.label_names, ("low_1_2", "middle_3", "high_4_5"))

    def test_relabel_preserves_raw_targets(self):
        frame = pd.DataFrame(
            {"final_intensity": [1, 3, 4], "drop_magnitude": [2, 3, 4]}
        )
        relabelled = relabel_outcome_frame(frame, COARSE_3)
        self.assertEqual(relabelled["final_intensity"].tolist(), [1, 2, 3])
        self.assertEqual(relabelled["drop_magnitude"].tolist(), [1, 2, 3])
        self.assertEqual(relabelled["final_intensity_raw"].tolist(), [1, 3, 4])
        self.assertEqual(set(relabelled["label_scheme"]), {COARSE_3})

    def test_coarse_joint_consistency_is_set_valued_on_raw_scale(self):
        metrics, arrays = joint_prediction_diagnostics(
            initial_intensity=[5, 4, 3],
            final_prediction=[2, 1, 1],
            drop_target=[1, 1, 1],
            drop_prediction=[1, 2, 3],
            label_scheme=COARSE_3,
        )
        self.assertEqual(arrays["joint_consistent"].tolist(), [True, True, False])
        self.assertEqual(arrays["derived_drop_prediction_min"].tolist(), [1, 1, 1])
        self.assertEqual(arrays["derived_drop_prediction_max"].tolist(), [1, 2, 1])
        self.assertAlmostEqual(metrics["joint_consistency_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
