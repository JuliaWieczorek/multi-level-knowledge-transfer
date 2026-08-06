import unittest

from mlkt.experiments import experiment_variants, matrix_manifest


class ExperimentMatrixTests(unittest.TestCase):
    def test_full_matrix_contains_exactly_125_unique_runs(self):
        runs = matrix_manifest(
            checkpoints=(0.10, 0.25, 0.50, 0.75, 1.00),
            seeds=(42, 52, 62, 72, 82),
        )
        self.assertEqual(len(runs), 125)
        identities = {
            (
                run["checkpoint"],
                run["seed"],
                run["name"],
                run["modality"],
                run["transfer"],
            )
            for run in runs
        }
        self.assertEqual(len(identities), 125)

    def test_variants_match_main_ablations_and_vanilla_controls(self):
        variants = {variant["name"]: variant for variant in experiment_variants()}
        self.assertEqual(
            set(variants),
            {
                "transferred_text",
                "strategy",
                "transferred_text_strategy",
                "vanilla_text",
                "vanilla_text_strategy",
            },
        )
        self.assertFalse(variants["strategy"]["transfer"])
        self.assertTrue(variants["transferred_text"]["transfer"])
        self.assertTrue(variants["transferred_text_strategy"]["transfer"])


if __name__ == "__main__":
    unittest.main()
