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


def _config_output_path(value: str, config_path: Path) -> Path:
    return _resolve_config_path(value, config_path)


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


def prepare_transfer(args: argparse.Namespace) -> None:
    from .transfer import prepare_transfer_inputs

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    transfer = config["transfer"]
    result = prepare_transfer_inputs(
        esconv_path=args.esconv_da
        or _resolve_config_path(transfer["esconv_da_path"], config_path),
        meisd_path=args.meisd_da
        or _resolve_config_path(transfer["meisd_da_path"], config_path),
        esconv_checkpoints_path=args.checkpoints
        or _resolve_config_path(transfer["esconv_checkpoints_path"], config_path),
        output_dir=args.output_dir
        or _config_output_path(transfer["prepared_dir"], config_path),
        seed=config["seed"],
    )
    print(json.dumps(result.__dict__, indent=2))


def augment_transfer(args: argparse.Namespace) -> None:
    from .transfer import (
        DeterministicMockGenerator,
        LlamaCppGenerator,
        run_augmentation_pipeline,
    )

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    transfer = config["transfer"]
    prepared_dir = _config_output_path(transfer["prepared_dir"], config_path)
    if args.mock_generator:
        generator = DeterministicMockGenerator()
    else:
        if not args.llama_model:
            raise ValueError(
                "--llama-model is required unless --mock-generator is used."
            )
        generator = LlamaCppGenerator(
            args.llama_model,
            context_size=transfer["augmentation"]["context_size"],
            threads=transfer["augmentation"]["threads"],
        )
    manifest = run_augmentation_pipeline(
        prepared_esconv_path=prepared_dir / "esconv_transfer_prepared.csv",
        prepared_meisd_path=prepared_dir / "meisd_transfer_prepared.csv",
        output_dir=args.output_dir
        or _config_output_path(transfer["augmented_dir"], config_path),
        generator=generator,
        seed=config["seed"],
        min_compatible_samples=transfer["augmentation"][
            "min_compatible_samples"
        ],
        max_aug_per_group=transfer["augmentation"]["max_aug_per_group"],
    )
    print(json.dumps(manifest, indent=2))


def pretrain_transfer(args: argparse.Namespace) -> None:
    from .training import pretrain_source_mtl

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    transfer = config["transfer"]
    source_config = config["source_mtl"]
    source_path = (
        Path(args.input)
        if args.input
        else _config_output_path(transfer["augmented_dir"], config_path)
        / "meisd_target_style_onehot.csv"
    )
    output_root = Path(
        args.output_dir
        or _config_output_path(source_config["output_dir"], config_path)
    )
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    summaries = []
    for seed in seeds:
        summaries.append(
            pretrain_source_mtl(
                source_path=source_path,
                output_dir=output_root / f"seed_{seed}",
                seed=seed,
                config=source_config,
            )
        )
    print(json.dumps(summaries, indent=2))


def train_temporal(args: argparse.Namespace) -> None:
    from .training import train_temporal_model

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    temporal = config["temporal"]
    if args.modality == "strategy" and args.transfer:
        raise ValueError(
            "Strategy-only has no text encoder and therefore no transfer variant."
        )
    transfer_checkpoint = args.source_checkpoint
    if args.transfer and not transfer_checkpoint:
        source_root = _config_output_path(
            config["source_mtl"]["output_dir"], config_path
        )
        transfer_checkpoint = (
            source_root / f"seed_{args.seed}" / "source_transfer_checkpoint.pt"
        )
    if not args.transfer:
        transfer_checkpoint = None
    summary = train_temporal_model(
        checkpoints_path=args.input
        or _resolve_config_path(
            config["transfer"]["esconv_checkpoints_path"], config_path
        ),
        output_dir=args.output_dir,
        checkpoint=args.checkpoint / 100.0 if args.checkpoint > 1 else args.checkpoint,
        modality=args.modality,
        seed=args.seed,
        config=temporal,
        transfer_checkpoint_path=transfer_checkpoint,
    )
    print(json.dumps(summary, indent=2))


def run_matrix(args: argparse.Namespace) -> None:
    from .experiments import matrix_manifest, run_experiment_matrix

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    runs = matrix_manifest(config["checkpoints"], config["seeds"])
    if args.dry_run:
        print(json.dumps(runs, indent=2))
        print(f"\nPlanned runs: {len(runs)}")
        return
    result = run_experiment_matrix(
        checkpoints_path=_resolve_config_path(
            config["transfer"]["esconv_checkpoints_path"], config_path
        ),
        source_root=_config_output_path(
            config["source_mtl"]["output_dir"], config_path
        ),
        output_root=args.output_dir
        or _config_output_path(config["temporal"]["output_dir"], config_path),
        config=config["temporal"],
        checkpoints=config["checkpoints"],
        seeds=config["seeds"],
        skip_existing=not args.overwrite,
    )
    print(json.dumps(result, indent=2))


def analyze_strategies(args: argparse.Namespace) -> None:
    from .strategy import (
        retrospective_ordinal_analysis,
        run_nested_strategy_models,
    )

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    frame = pd.read_csv(
        args.input
        or _resolve_config_path(
            config["transfer"]["esconv_checkpoints_path"], config_path
        )
    )
    output = Path(
        args.output_dir
        or _config_output_path(config["strategy_analysis"]["output_dir"], config_path)
    )
    results: dict[str, object] = {}
    if not args.retrospective_only:
        metrics, _ = run_nested_strategy_models(
            frame, output / "prospective", seed=args.seed
        )
        results["prospective_metric_rows"] = len(metrics)
    if not args.prospective_only:
        associations = retrospective_ordinal_analysis(
            frame,
            output / "retrospective",
            bootstrap_samples=args.bootstrap_samples
            or config["strategy_analysis"]["bootstrap_samples"],
            seed=args.seed,
        )
        results["retrospective_associations"] = len(associations)
    print(json.dumps(results, indent=2))


def report(args: argparse.Namespace) -> None:
    from .reporting import build_report

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    manifest = build_report(
        experiments_root=args.experiments
        or _config_output_path(config["temporal"]["output_dir"], config_path),
        output_dir=args.output_dir
        or _config_output_path(config["reporting"]["output_dir"], config_path),
    )
    print(json.dumps(manifest, indent=2))


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
        choices=("final_intensity", "drop_magnitude", "intensity_change"),
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

    transfer_parser = subparsers.add_parser("prepare-transfer")
    transfer_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    transfer_parser.add_argument("--esconv-da")
    transfer_parser.add_argument("--meisd-da")
    transfer_parser.add_argument("--checkpoints")
    transfer_parser.add_argument("--output-dir")
    transfer_parser.set_defaults(function=prepare_transfer)

    augment_parser = subparsers.add_parser("augment-transfer")
    augment_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    augment_parser.add_argument("--llama-model")
    augment_parser.add_argument("--mock-generator", action="store_true")
    augment_parser.add_argument("--output-dir")
    augment_parser.set_defaults(function=augment_transfer)

    source_parser = subparsers.add_parser("pretrain-transfer")
    source_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    source_parser.add_argument("--input")
    source_parser.add_argument("--output-dir")
    source_parser.add_argument("--seed", type=int)
    source_parser.set_defaults(function=pretrain_transfer)

    temporal_parser = subparsers.add_parser("train-temporal")
    temporal_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    temporal_parser.add_argument("--input")
    temporal_parser.add_argument("--output-dir", required=True)
    temporal_parser.add_argument(
        "--checkpoint",
        type=float,
        choices=(0.1, 0.25, 0.5, 0.75, 1.0, 10, 25, 50, 75, 100),
        required=True,
    )
    temporal_parser.add_argument(
        "--modality", choices=("text", "strategy", "text_strategy"), required=True
    )
    temporal_parser.add_argument("--seed", type=int, default=42)
    transfer_group = temporal_parser.add_mutually_exclusive_group()
    transfer_group.add_argument("--transfer", action="store_true")
    transfer_group.add_argument("--vanilla", action="store_true")
    temporal_parser.add_argument("--source-checkpoint")
    temporal_parser.set_defaults(function=train_temporal)

    matrix_parser = subparsers.add_parser("run-matrix")
    matrix_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    matrix_parser.add_argument("--output-dir")
    matrix_parser.add_argument("--dry-run", action="store_true")
    matrix_parser.add_argument("--overwrite", action="store_true")
    matrix_parser.set_defaults(function=run_matrix)

    strategy_parser = subparsers.add_parser("analyze-strategies")
    strategy_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    strategy_parser.add_argument("--input")
    strategy_parser.add_argument("--output-dir")
    strategy_parser.add_argument("--seed", type=int, default=42)
    strategy_parser.add_argument("--bootstrap-samples", type=int)
    strategy_scope = strategy_parser.add_mutually_exclusive_group()
    strategy_scope.add_argument("--prospective-only", action="store_true")
    strategy_scope.add_argument("--retrospective-only", action="store_true")
    strategy_parser.set_defaults(function=analyze_strategies)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    report_parser.add_argument("--experiments")
    report_parser.add_argument("--output-dir")
    report_parser.set_defaults(function=report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
