from __future__ import annotations

import random
from collections import defaultdict
from typing import Hashable

import pandas as pd


def assign_conversation_splits(
    frame: pd.DataFrame,
    label_column: str = "final_intensity",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign stratified splits once per conversation, then broadcast to checkpoints."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("A positive test fraction is required.")
    required = {"dataset", "conversation_id", label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    conversations = frame[["dataset", "conversation_id", label_column]].drop_duplicates()
    duplicates = conversations.duplicated(["dataset", "conversation_id"], keep=False)
    if duplicates.any():
        raise ValueError("A conversation has inconsistent labels across checkpoints.")

    assignments: dict[tuple[str, str], str] = {}
    grouped: dict[tuple[str, Hashable], list[str]] = defaultdict(list)
    for row in conversations.itertuples(index=False):
        grouped[(str(row.dataset), getattr(row, label_column))].append(
            str(row.conversation_id)
        )

    rng = random.Random(seed)
    for (dataset, _label), ids in grouped.items():
        rng.shuffle(ids)
        count = len(ids)
        n_train = round(count * train_fraction)
        n_validation = round(count * validation_fraction)
        if count >= 3:
            n_train = min(max(n_train, 1), count - 2)
            n_validation = min(max(n_validation, 1), count - n_train - 1)
        for index, conversation_id in enumerate(ids):
            if index < n_train:
                split = "train"
            elif index < n_train + n_validation:
                split = "validation"
            else:
                split = "test"
            assignments[(dataset, conversation_id)] = split

    result = frame.copy()
    result["split"] = [
        assignments[(str(dataset), str(conversation_id))]
        for dataset, conversation_id in zip(result["dataset"], result["conversation_id"])
    ]
    assert_no_split_leakage(result)
    return result


def assert_no_split_leakage(frame: pd.DataFrame) -> None:
    counts = frame.groupby(["dataset", "conversation_id"])["split"].nunique()
    leaking = counts[counts > 1]
    if not leaking.empty:
        raise ValueError(
            f"Conversation leakage across splits: {leaking.index.tolist()[:5]}"
        )

