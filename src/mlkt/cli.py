from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baseline import run_naive_baselines, run_tfidf_baseline
from .data import (
    assert_checkpoint_integrity,
    build_esconv_checkpoints,
    build_meisd_checkpoints,
)
from .splits import assign_conversation_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.json"


def _load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_config_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def preprocess(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    checkpoints = config["checkpoints"]
    output_dir = Path(args.output_dir or PROJECT_ROOT / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    esconv_path = (
        Path(args.esconv)
        if args.esconv
        else _resolve_config_path(config["esconv"]["path"], config_path)
    )
    meisd_path = (
        Path(args.meisd)
        if args.meisd
        else _resolve_config_path(config["meisd"]["path"], config_path)
    )
    split_config = config["split"]

    esconv = build_esconv_checkpoints(esconv_path, checkpoints)
    assert_checkpoint_integrity(esconv, checkpoints)
    esconv = assign_conversation_splits(
        esconv,
        label_column=split_config["stratify_by"],
        train_fraction=split_config["train"],
        validation_fraction=split_config["validation"],
        seed=config["seed"],
    )
    esconv_output = output_dir / "esconv_checkpoints.csv"
    esconv.to_csv(esconv_output, index=False)

    meisd = build_meisd_checkpoints(meisd_path, checkpoints)
    assert_checkpoint_integrity(meisd, checkpoints)
    meisd = assign_conversation_splits(
        meisd,
        label_column=split_config["stratify_by"],
        train_fraction=split_config["train"],
        validation_fraction=split_config["validation"],
        seed=config["seed"],
    )
    meisd_output = output_dir / "meisd_checkpoints.csv"
    meisd.to_csv(meisd_output, index=False)

    print(f"Wrote {len(esconv):,} ESConv checkpoint rows to {esconv_output}")
    print(f"Wrote {len(meisd):,} MEISD checkpoint rows to {meisd_output}")


def describe(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir or PROJECT_ROOT / "data" / "processed")
    for dataset in ("esconv", "meisd"):
        path = input_dir / f"{dataset}_checkpoints.csv"
        frame = pd.read_csv(path)
        conversations = frame[frame["checkpoint"] == frame["checkpoint"].max()]
        print(
            f"\n{dataset.upper()}: "
            f"{conversations['conversation_id'].nunique():,} conversations"
        )
        print("Final intensity:")
        print(conversations["final_intensity"].value_counts().sort_index().to_string())
        print("Change direction:")
        print(conversations["intensity_change"].value_counts().to_string())
        print("Split conversations:")
        print(conversations["split"].value_counts().to_string())


def baseline(args: argparse.Namespace) -> None:
    input_path = Path(
        args.input
        or PROJECT_ROOT
        / "data"
        / "processed"
        / f"{args.dataset}_checkpoints.csv"
    )
    frame = pd.read_csv(input_path)
    if args.model == "naive":
        metrics, predictions = run_naive_baselines(frame, task=args.task)
    else:
        metrics, predictions = run_tfidf_baseline(
            frame, task=args.task, text_column=args.text_column, seed=args.seed
        )
    output_dir = Path(args.output_dir or PROJECT_ROOT / "outputs" / "baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.task}_{args.model}_{args.text_column}"
    metrics_path = output_dir / f"{stem}_metrics.csv"
    predictions_path = output_dir / f"{stem}_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    print(metrics.to_string(index=False))
    print(f"\nWrote metrics to {metrics_path}")
    print(f"Wrote predictions to {predictions_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlkt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    preprocess_parser.add_argument("--esconv")
    preprocess_parser.add_argument("--meisd")
    preprocess_parser.add_argument("--output-dir")
    preprocess_parser.set_defaults(function=preprocess)

    describe_parser = subparsers.add_parser("describe")
    describe_parser.add_argument("--input-dir")
    describe_parser.set_defaults(function=describe)

    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument(
        "--dataset", choices=("esconv", "meisd"), required=True
    )
    baseline_parser.add_argument(
        "--task",
        choices=("final_intensity", "intensity_change"),
        required=True,
    )
    baseline_parser.add_argument(
        "--model", choices=("naive", "tfidf"), default="naive"
    )
    baseline_parser.add_argument("--text-column", default="text")
    baseline_parser.add_argument("--input")
    baseline_parser.add_argument("--output-dir")
    baseline_parser.add_argument("--seed", type=int, default=42)
    baseline_parser.set_defaults(function=baseline)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
