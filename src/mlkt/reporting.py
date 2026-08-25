from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .outcome_labels import ORIGINAL_4, get_outcome_label_scheme


_T_CRITICAL_975 = {
    1: 12.7062,
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
    10: 2.2281,
    11: 2.2010,
    12: 2.1788,
    13: 2.1604,
    14: 2.1448,
    15: 2.1314,
    16: 2.1199,
    17: 2.1098,
    18: 2.1009,
    19: 2.0930,
    20: 2.0860,
    21: 2.0796,
    22: 2.0739,
    23: 2.0687,
    24: 2.0639,
    25: 2.0595,
    26: 2.0555,
    27: 2.0518,
    28: 2.0484,
    29: 2.0452,
    30: 2.0423,
}


def collect_metric_files(root: str | Path) -> pd.DataFrame:
    files = sorted(Path(root).rglob("metrics.csv"))
    frames = [pd.read_csv(path).assign(metrics_file=str(path)) for path in files]
    if not frames:
        raise ValueError(f"No metrics.csv files found below {root}.")
    combined = pd.concat(frames, ignore_index=True)
    if "use_initial_intensity" not in combined:
        combined["use_initial_intensity"] = False
    combined["use_initial_intensity"] = combined["use_initial_intensity"].fillna(False)
    if "label_scheme" not in combined:
        combined["label_scheme"] = ORIGINAL_4
    combined["label_scheme"] = combined["label_scheme"].fillna(ORIGINAL_4)
    return combined


def _ci95(values: pd.Series) -> float:
    """Return a two-sided 95% Student-t half-width for seed means."""
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    degrees_of_freedom = len(clean) - 1
    critical = _T_CRITICAL_975.get(degrees_of_freedom, 1.96)
    return float(critical * clean.std(ddof=1) / math.sqrt(len(clean)))


def aggregate_seed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if "use_initial_intensity" not in metrics:
        metrics = metrics.assign(use_initial_intensity=False)
    if "label_scheme" not in metrics:
        metrics = metrics.assign(label_scheme=ORIGINAL_4)
    identifiers = [
        "checkpoint",
        "checkpoint_percent",
        "split",
        "target",
        "modality",
        "transfer",
        "use_initial_intensity",
        "label_scheme",
    ]
    metric_columns = [
        column
        for column in (
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "mae",
            "quadratic_weighted_kappa",
            "precision_macro",
            "recall_macro",
            "f1_micro",
            "joint_consistency_rate",
            "derived_drop_accuracy",
            "derived_drop_mae",
            "invalid_derived_drop_rate",
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
    if "use_initial_intensity" not in metrics:
        metrics = metrics.assign(use_initial_intensity=False)
    if "label_scheme" not in metrics:
        metrics = metrics.assign(label_scheme=ORIGINAL_4)
    text_models = metrics[
        metrics["modality"].isin(["text", "text_strategy"])
        & ~metrics["use_initial_intensity"].fillna(False)
    ]
    index = [
        "checkpoint",
        "checkpoint_percent",
        "split",
        "target",
        "modality",
        "seed",
        "use_initial_intensity",
        "label_scheme",
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
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    created: list[str] = []
    test = summary[summary["split"] == "test"]
    for target in test["target"].unique():
        subset = test[test["target"] == target]
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for (modality, transfer, initial), group in subset.groupby(
            ["modality", "transfer", "use_initial_intensity"], sort=True
        ):
            group = group.sort_values("checkpoint_percent")
            label = (
                "strategy-only"
                if modality == "strategy"
                else (
                    f"{modality} (transfer + initial)"
                    if initial
                    else f"{modality} ({'transfer' if transfer else 'vanilla'})"
                )
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
        axis.legend(
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0.12, 1, 1))
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
    if "use_initial_intensity" not in predictions:
        predictions["use_initial_intensity"] = False
    if "label_scheme" not in predictions:
        predictions["label_scheme"] = ORIGINAL_4
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    mixed_schemes = predictions["label_scheme"].nunique() > 1
    created: list[str] = []
    for keys, group in predictions.groupby(
        [
            "checkpoint_percent",
            "modality",
            "transfer",
            "use_initial_intensity",
            "label_scheme",
        ],
        sort=True,
    ):
        checkpoint, modality, transfer, initial, label_scheme = keys
        scheme = get_outcome_label_scheme(str(label_scheme))
        variant = "transfer" if transfer else "vanilla"
        if modality == "strategy":
            variant = "strategy"
        elif initial:
            variant = "transfer_initial"
        for target, prediction in (
            ("final", "final_prediction"),
            ("drop", "drop_prediction"),
        ):
            matrix = confusion_matrix_frame(
                group[f"{target}_target"], group[prediction], labels=scheme.labels
            )
            target_dir = (
                confusion_dir / scheme.name if mixed_schemes else confusion_dir
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            path = (
                target_dir
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
                [
                    "checkpoint",
                    "modality",
                    "transfer",
                    "use_initial_intensity",
                    "label_scheme",
                    "seed",
                    "metrics_file",
                ]
            ].drop_duplicates().shape[0]
        ),
        "seeds": sorted(int(seed) for seed in metrics["seed"].unique()),
        "checkpoints": sorted(
            float(value) for value in metrics["checkpoint"].unique()
        ),
        "label_schemes": sorted(str(value) for value in metrics["label_scheme"].unique()),
        "figures": figures,
        "confusion_tables": confusion_tables,
        "primary_metric": "macro-F1",
        "confidence_interval": "two-sided 95% Student-t interval across seeds",
    }
    with (output / "report_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def compare_label_scheme_reports(
    original_report_dir: str | Path,
    coarse_report_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a descriptive four-class versus coarse-three-class comparison.

    The delta is deliberately labelled descriptive because macro-F1 values from
    different target spaces do not estimate performance on the same task.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for report_dir, expected_scheme in (
        (original_report_dir, ORIGINAL_4),
        (coarse_report_dir, "coarse3"),
    ):
        path = Path(report_dir) / "metrics_by_checkpoint.csv"
        frame = pd.read_csv(path)
        if "label_scheme" not in frame:
            frame["label_scheme"] = expected_scheme
        observed = set(frame["label_scheme"].astype(str))
        if observed != {expected_scheme}:
            raise ValueError(
                f"Expected only {expected_scheme!r} in {path}, found {sorted(observed)}."
            )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output / "combined_metrics_by_checkpoint.csv", index=False)
    identifiers = [
        "checkpoint",
        "checkpoint_percent",
        "split",
        "target",
        "modality",
        "transfer",
        "use_initial_intensity",
    ]
    value_columns = [
        column
        for column in ("f1_macro_mean", "accuracy_mean", "mae_mean")
        if column in combined
    ]
    comparison = combined.pivot_table(
        index=identifiers,
        columns="label_scheme",
        values=value_columns,
        aggfunc="first",
    )
    comparison.columns = [f"{metric}_{scheme}" for metric, scheme in comparison.columns]
    comparison = comparison.reset_index()
    if {
        "f1_macro_mean_original4",
        "f1_macro_mean_coarse3",
    }.issubset(comparison.columns):
        comparison["f1_macro_descriptive_delta_coarse3_minus_original4"] = (
            comparison["f1_macro_mean_coarse3"]
            - comparison["f1_macro_mean_original4"]
        )
    comparison.to_csv(output / "label_scheme_comparison.csv", index=False)
    test = combined[combined["split"] == "test"].copy()
    best = test.loc[
        test.groupby(["label_scheme", "target"])["f1_macro_mean"].idxmax()
    ].sort_values(["label_scheme", "target"])
    best.to_csv(output / "best_models_by_label_scheme.csv", index=False)
    manifest = {
        "original_report": str(Path(original_report_dir).resolve()),
        "coarse_report": str(Path(coarse_report_dir).resolve()),
        "combined_rows": int(len(combined)),
        "matched_comparison_rows": int(len(comparison)),
        "warning": (
            "Deltas are descriptive only: original4 and coarse3 use different "
            "target spaces and are not interchangeable estimates of one task."
        ),
    }
    with (output / "comparison_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
