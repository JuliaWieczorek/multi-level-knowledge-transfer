from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def collect_metric_files(root: str | Path) -> pd.DataFrame:
    files = sorted(Path(root).rglob("metrics.csv"))
    frames = [pd.read_csv(path).assign(metrics_file=str(path)) for path in files]
    if not frames:
        raise ValueError(f"No metrics.csv files found below {root}.")
    return pd.concat(frames, ignore_index=True)


def _ci95(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    return float(1.96 * clean.std(ddof=1) / math.sqrt(len(clean)))


def aggregate_seed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "checkpoint",
        "checkpoint_percent",
        "split",
        "target",
        "modality",
        "transfer",
    ]
    metric_columns = [
        column
        for column in (
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "mae",
            "quadratic_weighted_kappa",
        )
        if column in metrics
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(identifiers, dropna=False, sort=True):
        row = dict(zip(identifiers, keys))
        row["n_seeds"] = group["seed"].nunique()
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_ci95"] = _ci95(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_transfer_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    text_models = metrics[metrics["modality"].isin(["text", "text_strategy"])]
    index = [
        "checkpoint",
        "checkpoint_percent",
        "split",
        "target",
        "modality",
        "seed",
    ]
    pivot = text_models.pivot_table(
        index=index,
        columns="transfer",
        values="f1_macro",
        aggfunc="first",
    )
    if True not in pivot or False not in pivot:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=[True, False]).reset_index()
    pivot["f1_macro_delta_transfer_minus_vanilla"] = pivot[True] - pivot[False]
    return pivot.drop(columns=[True, False])


def confusion_matrix_frame(
    target: Iterable[int], prediction: Iterable[int], labels: Iterable[int] = (1, 2, 3, 4)
) -> pd.DataFrame:
    target_values = np.asarray(list(target))
    predicted_values = np.asarray(list(prediction))
    label_values = list(labels)
    matrix = np.zeros((len(label_values), len(label_values)), dtype=int)
    lookup = {label: index for index, label in enumerate(label_values)}
    for truth, predicted in zip(target_values, predicted_values):
        if truth in lookup and predicted in lookup:
            matrix[lookup[truth], lookup[predicted]] += 1
    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in label_values],
        columns=[f"pred_{label}" for label in label_values],
    )


def _plot_curves(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    created: list[str] = []
    test = summary[summary["split"] == "test"]
    for target in test["target"].unique():
        subset = test[test["target"] == target]
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for (modality, transfer), group in subset.groupby(
            ["modality", "transfer"], sort=True
        ):
            group = group.sort_values("checkpoint_percent")
            label = (
                "strategy-only"
                if modality == "strategy"
                else f"{modality} ({'transfer' if transfer else 'vanilla'})"
            )
            axis.errorbar(
                group["checkpoint_percent"],
                group["f1_macro_mean"],
                yerr=group["f1_macro_ci95"],
                marker="o",
                capsize=3,
                label=label,
            )
        axis.set_xlabel("Observed dialogue (%)")
        axis.set_ylabel("Macro-F1")
        axis.set_title(target.replace("_", " ").title())
        axis.set_xticks([10, 25, 50, 75, 100])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = output_dir / f"{target}_checkpoint_f1.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(str(path))
    return created


def _build_confusion_tables(
    experiments_root: str | Path, output_dir: Path
) -> list[str]:
    files = sorted(Path(experiments_root).rglob("test_predictions.csv"))
    if not files:
        return []
    predictions = pd.concat(
        [pd.read_csv(path).assign(predictions_file=str(path)) for path in files],
        ignore_index=True,
    )
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for keys, group in predictions.groupby(
        ["checkpoint_percent", "modality", "transfer"], sort=True
    ):
        checkpoint, modality, transfer = keys
        variant = "transfer" if transfer else "vanilla"
        if modality == "strategy":
            variant = "strategy"
        for target, prediction in (
            ("final", "final_prediction"),
            ("drop", "drop_prediction"),
        ):
            matrix = confusion_matrix_frame(
                group[f"{target}_target"], group[prediction]
            )
            path = (
                confusion_dir
                / f"{int(checkpoint):03d}_{modality}_{variant}_{target}.csv"
            )
            matrix.to_csv(path)
            created.append(str(path))
    return created


def build_report(
    experiments_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = collect_metric_files(experiments_root)
    metrics.to_csv(output / "all_run_metrics.csv", index=False)
    summary = aggregate_seed_metrics(metrics)
    summary.to_csv(output / "metrics_by_checkpoint.csv", index=False)
    deltas = paired_transfer_deltas(metrics)
    deltas.to_csv(output / "paired_transfer_deltas.csv", index=False)
    figures = _plot_curves(summary, output)
    confusion_tables = _build_confusion_tables(experiments_root, output)
    manifest = {
        "runs": int(
            metrics[
                ["checkpoint", "modality", "transfer", "seed", "metrics_file"]
            ].drop_duplicates().shape[0]
        ),
        "seeds": sorted(int(seed) for seed in metrics["seed"].unique()),
        "checkpoints": sorted(
            float(value) for value in metrics["checkpoint"].unique()
        ),
        "figures": figures,
        "confusion_tables": confusion_tables,
        "primary_metric": "macro-F1",
    }
    with (output / "report_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
