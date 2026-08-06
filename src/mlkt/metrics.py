from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def classification_metrics(
    y_true: Iterable[Any], y_pred: Iterable[Any]
) -> dict[str, float]:
    true = np.asarray(list(y_true))
    pred = np.asarray(list(y_pred))
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("Metric inputs must be non-empty and have equal shapes.")

    labels = np.unique(np.concatenate([true, pred]))
    f1_values: list[float] = []
    supports: list[int] = []
    for label in labels:
        tp = int(np.sum((true == label) & (pred == label)))
        fp = int(np.sum((true != label) & (pred == label)))
        fn = int(np.sum((true == label) & (pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        supports.append(int(np.sum(true == label)))

    support_array = np.asarray(supports, dtype=float)
    return {
        "accuracy": float(np.mean(true == pred)),
        "f1_macro": float(np.mean(f1_values)),
        "f1_weighted": float(np.average(f1_values, weights=support_array)),
    }


def ordinal_metrics(
    y_true: Iterable[int], y_pred: Iterable[int]
) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("Metric inputs must be non-empty and have equal shapes.")

    labels = np.arange(min(true.min(), pred.min()), max(true.max(), pred.max()) + 1)
    n_labels = len(labels)
    if n_labels == 1:
        kappa = 1.0
    else:
        label_to_index = {label: index for index, label in enumerate(labels)}
        observed = np.zeros((n_labels, n_labels), dtype=float)
        for left, right in zip(true, pred):
            observed[label_to_index[left], label_to_index[right]] += 1
        expected = (
            np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
        )
        indices = np.arange(n_labels)
        weights = (
            (indices[:, None] - indices[None, :]) ** 2 / ((n_labels - 1) ** 2)
        )
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

