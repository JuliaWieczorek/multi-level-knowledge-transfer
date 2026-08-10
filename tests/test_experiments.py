import unittest

from mlkt.experiments import experiment_variants, matrix_manifest


class ExperimentMatrixTests(unittest.TestCase):
    def test_full_matrix_contains_exactly_150_unique_runs(self):
        runs = matrix_manifest(
            checkpoints=(0.10, 0.25, 0.50, 0.75, 1.00),
            seeds=(42, 52, 62, 72, 82),
        )
        self.assertEqual(len(runs), 150)
        identities = {
            (
                run["checkpoint"],
                run["seed"],
                run["name"],
                run["modality"],
                run["transfer"],
                run["use_initial_intensity"],
            )
            for run in runs
        }
        self.assertEqual(len(identities), 150)

    def test_variants_match_main_ablations_and_vanilla_controls(self):
        variants = {variant["name"]: variant for variant in experiment_variants()}
        self.assertEqual(
            set(variants),
            {
                "transferred_text",
                "strategy",
                "transferred_text_strategy",
                "transferred_text_strategy_initial",
                "vanilla_text",
                "vanilla_text_strategy",
            },
        )
        self.assertFalse(variants["strategy"]["transfer"])
        self.assertTrue(variants["transferred_text"]["transfer"])
        self.assertTrue(variants["transferred_text_strategy"]["transfer"])
        self.assertTrue(variants["transferred_text_strategy_initial"]["transfer"])
        self.assertTrue(
            variants["transferred_text_strategy_initial"]["use_initial_intensity"]
        )
        self.assertEqual(
            [
                name
                for name, variant in variants.items()
                if variant["use_initial_intensity"]
            ],
            ["transferred_text_strategy_initial"],
        )


if __name__ == "__main__":
    unittest.main()
