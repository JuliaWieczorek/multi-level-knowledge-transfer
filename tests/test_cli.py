import unittest

from mlkt.cli import build_parser


class CliTests(unittest.TestCase):
    def test_strategy_bootstrap_zero_is_preserved_by_parser(self):
        args = build_parser().parse_args(
            ["analyze-strategies", "--retrospective-only", "--bootstrap-samples", "0"]
        )
        self.assertEqual(args.bootstrap_samples, 0)

    def test_matrix_accepts_coarse_three_class_scheme(self):
        args = build_parser().parse_args(
            [
                "run-matrix",
                "--dry-run",
                "--label-scheme",
                "coarse3",
                "--checkpoints",
                "100",
                "--seeds",
                "42",
                "52",
            ]
        )
        self.assertEqual(args.label_scheme, "coarse3")
        self.assertEqual(args.checkpoints, [100.0])
        self.assertEqual(args.seeds, [42, 52])


if __name__ == "__main__":
    unittest.main()
