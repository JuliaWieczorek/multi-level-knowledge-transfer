from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import confusion_matrix_records, per_class_metrics


def analyze_outcome_errors(
    predictions_path: str | Path,
    checkpoints_path: str | Path,
    output_dir: str | Path,
    min_group_size: int = 5,
) -> dict[str, Any]:
    """Write class, confusion, support, and subgroup diagnostics.

    The function only consumes saved predictions and metadata. It never fits a
    model and is therefore safe to run after test evaluation without changing
    model selection.
    """
    predictions = pd.read_csv(predictions_path)
    checkpoints = pd.read_csv(checkpoints_path)
    required_predictions = {
        "conversation_id",
        "final_target",
        "final_prediction",
        "drop_target",
        "drop_prediction",
        "initial_intensity",
    }
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"Missing outcome prediction columns: {sorted(missing_predictions)}"
        )
    required_metadata = {
        "conversation_id",
        "checkpoint",
        "split",
        "emotion_family",
        "problem_type",
        "final_intensity",
        "drop_magnitude",
    }
    missing_metadata = required_metadata - set(checkpoints.columns)
    if missing_metadata:
        raise ValueError(f"Missing outcome metadata columns: {sorted(missing_metadata)}")
    if min_group_size < 1:
        raise ValueError("min_group_size must be positive.")

    split_values = predictions.get("split", pd.Series(dtype=str)).dropna().unique()
    if len(split_values) > 1:
        raise ValueError("Analyze one prediction split at a time.")
    split_name = str(split_values[0]) if len(split_values) == 1 else "unknown"
    metadata = checkpoints[np.isclose(checkpoints["checkpoint"].astype(float), 1.0)]
    if split_name != "unknown":
        metadata = metadata[metadata["split"].astype(str) == split_name]
    metadata = metadata[
        ["conversation_id", "emotion_family", "problem_type"]
    ].drop_duplicates("conversation_id")
    frame = predictions.merge(
        metadata, on="conversation_id", how="left", validate="one_to_one"
    )

    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    zero_recall: list[str] = []
    for target, prefix in (
        ("final_intensity", "final"),
        ("drop_magnitude", "drop"),
    ):
        truth = frame[f"{prefix}_target"]
        predicted = frame[f"{prefix}_prediction"]
        rows = per_class_metrics(
            truth, predicted, labels=(1, 2, 3, 4), label_names=("1", "2", "3", "4")
        )
        for row in rows:
            per_class_rows.append({"split": split_name, "target": target, **row})
            if row["support"] > 0 and row["recall"] == 0.0:
                zero_recall.append(f"{target}={row['label']}")
        confusion_rows.extend(
            {"split": split_name, "target": target, **row}
            for row in confusion_matrix_records(
                truth, predicted, labels=(1, 2, 3, 4)
            )
        )

    subgroup_rows: list[dict[str, Any]] = []
    for column in ("initial_intensity", "emotion_family", "problem_type"):
        for value, group in frame.groupby(column, dropna=False):
            if len(group) < min_group_size:
                continue
            subgroup_rows.append(
                {
                    "split": split_name,
                    "grouping": column,
                    "group": str(value),
                    "support": int(len(group)),
                    "final_accuracy": float(
                        (group["final_target"] == group["final_prediction"]).mean()
                    ),
                    "drop_accuracy": float(
                        (group["drop_target"] == group["drop_prediction"]).mean()
                    ),
                    "final_mae": float(
                        (group["final_target"] - group["final_prediction"])
                        .abs()
                        .mean()
                    ),
                    "drop_mae": float(
                        (group["drop_target"] - group["drop_prediction"])
                        .abs()
                        .mean()
                    ),
                }
            )

    training = checkpoints[
        np.isclose(checkpoints["checkpoint"].astype(float), 1.0)
        & checkpoints["split"].astype(str).eq("train")
    ]
    support_rows: list[dict[str, Any]] = []
    for target in ("final_intensity", "drop_magnitude"):
        for label, count in training[target].astype(int).value_counts().sort_index().items():
            support_rows.append(
                {"target": target, "label": str(label), "support": int(count)}
            )
    joint_counts = training.groupby(["final_intensity", "drop_magnitude"]).size()
    for (final_label, drop_label), count in joint_counts.items():
        support_rows.append(
            {
                "target": "joint_pair",
                "label": f"{int(final_label)}+{int(drop_label)}",
                "support": int(count),
            }
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_class_rows).to_csv(output / "per_class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output / "confusion_matrix.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(output / "subgroup_metrics.csv", index=False)
    pd.DataFrame(support_rows).to_csv(output / "training_support.csv", index=False)
    summary = {
        "split": split_name,
        "examples": int(len(frame)),
        "zero_recall_classes": zero_recall,
        "minimum_group_size": min_group_size,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
