from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    labels: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Return comparable single-label metrics using an explicit label space."""
    true = np.asarray(list(y_true))
    pred = np.asarray(list(y_pred))
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("Metric inputs must be non-empty and have equal shapes.")
    label_values = list(labels) if labels is not None else list(
        np.unique(np.concatenate([true, pred]))
    )
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision_macro": float(
            precision_score(true, pred, labels=label_values, average="macro", zero_division=0)
        ),
        "precision_micro": float(
            precision_score(true, pred, labels=label_values, average="micro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(
                true, pred, labels=label_values, average="weighted", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(true, pred, labels=label_values, average="macro", zero_division=0)
        ),
        "recall_micro": float(
            recall_score(true, pred, labels=label_values, average="micro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(
                true, pred, labels=label_values, average="weighted", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(true, pred, labels=label_values, average="macro", zero_division=0)
        ),
        "f1_micro": float(
            f1_score(true, pred, labels=label_values, average="micro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(true, pred, labels=label_values, average="weighted", zero_division=0)
        ),
    }


def per_class_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    labels: Sequence[Any],
    label_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    true = np.asarray(list(y_true))
    pred = np.asarray(list(y_pred))
    label_values = list(labels)
    names = list(label_names) if label_names is not None else [str(x) for x in labels]
    if len(names) != len(label_values):
        raise ValueError("label_names and labels must have equal length.")
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=label_values, zero_division=0
    )
    return [
        {
            "label": label,
            "label_name": name,
            "precision": float(p),
            "recall": float(r),
            "f1": float(score),
            "support": int(count),
        }
        for label, name, p, r, score, count in zip(
            label_values, names, precision, recall, f1, support
        )
    ]


def multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Metrics for an [examples, labels] multilabel prediction matrix."""
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape or true.ndim != 2 or true.size == 0:
        raise ValueError("Multilabel inputs must be non-empty equal 2-D matrices.")
    result = {
        "subset_accuracy": float(accuracy_score(true, pred)),
        "hamming_loss": float(hamming_loss(true, pred)),
    }
    for average in ("micro", "macro", "weighted", "samples"):
        result[f"precision_{average}"] = float(
            precision_score(true, pred, average=average, zero_division=0)
        )
        result[f"recall_{average}"] = float(
            recall_score(true, pred, average=average, zero_division=0)
        )
        result[f"f1_{average}"] = float(
            f1_score(true, pred, average=average, zero_division=0)
        )
        result[f"jaccard_{average}"] = float(
            jaccard_score(true, pred, average=average, zero_division=0)
        )
    if y_score is not None:
        score = np.asarray(y_score, dtype=float)
        if score.shape != true.shape:
            raise ValueError("Multilabel scores must match the target matrix.")
        for average in ("micro", "macro", "weighted"):
            try:
                result[f"average_precision_{average}"] = float(
                    average_precision_score(true, score, average=average)
                )
            except ValueError:
                result[f"average_precision_{average}"] = float("nan")
            try:
                result[f"roc_auc_{average}"] = float(
                    roc_auc_score(true, score, average=average)
                )
            except ValueError:
                result[f"roc_auc_{average}"] = float("nan")
    return result


def multilabel_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str],
) -> list[dict[str, Any]]:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if true.shape != pred.shape or true.shape != score.shape:
        raise ValueError("Per-label multilabel inputs must have equal shapes.")
    if true.ndim != 2 or true.shape[1] != len(label_names):
        raise ValueError("label_names must match the multilabel matrix width.")
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(label_names):
        target = true[:, index]
        predicted = pred[:, index]
        row: dict[str, Any] = {
            "label_name": name,
            "precision": float(precision_score(target, predicted, zero_division=0)),
            "recall": float(recall_score(target, predicted, zero_division=0)),
            "f1": float(f1_score(target, predicted, zero_division=0)),
            "support": int(target.sum()),
        }
        try:
            row["average_precision"] = float(
                average_precision_score(target, score[:, index])
            )
        except ValueError:
            row["average_precision"] = float("nan")
        try:
            row["roc_auc"] = float(roc_auc_score(target, score[:, index]))
        except ValueError:
            row["roc_auc"] = float("nan")
        rows.append(row)
    return rows


def confusion_matrix_records(
    y_true: Iterable[Any], y_pred: Iterable[Any], labels: Sequence[Any]
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(list(y_true), list(y_pred), labels=list(labels))
    return [
        {"true_label": true_label, "predicted_label": predicted_label, "count": int(count)}
        for true_label, row in zip(labels, matrix)
        for predicted_label, count in zip(labels, row)
    ]


def joint_prediction_diagnostics(
    initial_intensity: Iterable[int],
    final_prediction: Iterable[int],
    drop_target: Iterable[int],
    drop_prediction: Iterable[int],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    initial = np.asarray(list(initial_intensity), dtype=int)
    final = np.asarray(list(final_prediction), dtype=int)
    drop_true = np.asarray(list(drop_target), dtype=int)
    drop_pred = np.asarray(list(drop_prediction), dtype=int)
    if not (
        initial.shape == final.shape == drop_true.shape == drop_pred.shape
        and initial.size > 0
    ):
        raise ValueError("Joint diagnostic inputs must be non-empty and equal length.")
    consistent = final + drop_pred == initial
    derived_drop = initial - final
    derived_valid = (derived_drop >= 1) & (derived_drop <= 4)
    metrics = {
        "joint_consistency_rate": float(consistent.mean()),
        "derived_drop_accuracy": float((derived_drop == drop_true).mean()),
        "derived_drop_mae": float(np.mean(np.abs(derived_drop - drop_true))),
        "invalid_derived_drop_rate": float((~derived_valid).mean()),
    }
    arrays = {
        "joint_consistent": consistent,
        "derived_drop_prediction": derived_drop,
        "derived_drop_valid": derived_valid,
    }
    return metrics, arrays


def ordinal_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Sequence[int] | None = None,
) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("Metric inputs must be non-empty and have equal shapes.")

    label_values = (
        np.asarray(list(labels), dtype=int)
        if labels is not None
        else np.arange(min(true.min(), pred.min()), max(true.max(), pred.max()) + 1)
    )
    n_labels = len(label_values)
    if n_labels == 1:
        kappa = 1.0
    else:
        lookup = {label: index for index, label in enumerate(label_values)}
        observed = np.zeros((n_labels, n_labels), dtype=float)
        for left, right in zip(true, pred):
            if left not in lookup or right not in lookup:
                raise ValueError("Ordinal values fall outside the configured labels.")
            observed[lookup[left], lookup[right]] += 1
        expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
        indices = np.arange(n_labels)
        weights = (indices[:, None] - indices[None, :]) ** 2 / ((n_labels - 1) ** 2)
        denominator = float(np.sum(weights * expected))
        kappa = (
            1.0 - float(np.sum(weights * observed)) / denominator
            if denominator
            else 1.0
        )
    return {
        "mae": float(np.mean(np.abs(true - pred))),
        "quadratic_weighted_kappa": kappa,
    }
