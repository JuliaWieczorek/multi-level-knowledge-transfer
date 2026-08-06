from __future__ import annotations

from typing import Any

import pandas as pd

from .metrics import classification_metrics, ordinal_metrics


def run_naive_baselines(
    frame: pd.DataFrame,
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate training-majority and, for intensity, initial-value persistence."""
    target_column = {
        "final_intensity": "final_intensity",
        "intensity_change": "intensity_change",
        "drop_magnitude": "drop_magnitude",
    }.get(task)
    if target_column is None:
        raise ValueError(
            "Task must be 'final_intensity', 'drop_magnitude', or "
            "'intensity_change'."
        )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for checkpoint, checkpoint_frame in frame.groupby("checkpoint", sort=True):
        train = checkpoint_frame[checkpoint_frame["split"] == "train"]
        test = checkpoint_frame[checkpoint_frame["split"] == "test"]
        majority = train[target_column].mode().iloc[0]
        predictors: dict[str, Any] = {
            "majority": [majority] * len(test),
        }
        if task == "final_intensity":
            predictors["initial_persistence"] = test["initial_intensity"].to_numpy()

        for model_name, predicted in predictors.items():
            scores = classification_metrics(test[target_column], predicted)
            if task in {"final_intensity", "drop_magnitude"}:
                scores.update(ordinal_metrics(test[target_column], predicted))
            metric_rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_percent": int(round(float(checkpoint) * 100)),
                    "split": "test",
                    "task": task,
                    "model": model_name,
                    **scores,
                }
            )
            predictions = test[
                ["dataset", "conversation_id", "checkpoint", target_column]
            ].copy()
            predictions = predictions.rename(columns={target_column: "target"})
            predictions["prediction"] = predicted
            predictions["split"] = "test"
            predictions["task"] = task
            predictions["model"] = model_name
            prediction_frames.append(predictions)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def run_tfidf_baseline(
    frame: pd.DataFrame,
    task: str,
    text_column: str = "text",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train an independent TF-IDF model at each checkpoint."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as error:
        raise RuntimeError(
            "The baseline requires scikit-learn. Install with `pip install -e .`."
        ) from error

    target_column = {
        "final_intensity": "final_intensity",
        "intensity_change": "intensity_change",
        "drop_magnitude": "drop_magnitude",
    }.get(task)
    if target_column is None:
        raise ValueError(
            "Task must be 'final_intensity', 'drop_magnitude', or "
            "'intensity_change'."
        )
    required = {
        "checkpoint",
        "split",
        text_column,
        target_column,
        "conversation_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing baseline columns: {sorted(missing)}")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for checkpoint, checkpoint_frame in frame.groupby("checkpoint", sort=True):
        train = checkpoint_frame[checkpoint_frame["split"] == "train"]
        validation = checkpoint_frame[checkpoint_frame["split"] == "validation"]
        test = checkpoint_frame[checkpoint_frame["split"] == "test"]
        if train[target_column].nunique() < 2:
            raise ValueError(
                f"Task {task} has fewer than two training classes at "
                f"checkpoint {checkpoint}."
            )

        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(train[text_column].fillna(""), train[target_column])
        for split_name, split_frame in (("validation", validation), ("test", test)):
            predicted = model.predict(split_frame[text_column].fillna(""))
            scores = classification_metrics(split_frame[target_column], predicted)
            if task in {"final_intensity", "drop_magnitude"}:
                scores.update(ordinal_metrics(split_frame[target_column], predicted))
            metric_rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_percent": int(round(float(checkpoint) * 100)),
                    "split": split_name,
                    "task": task,
                    "model": "tfidf_logistic_regression",
                    **scores,
                }
            )
            predictions = split_frame[
                ["dataset", "conversation_id", "checkpoint", target_column]
            ].copy()
            predictions = predictions.rename(columns={target_column: "target"})
            predictions["prediction"] = predicted
            predictions["split"] = split_name
            predictions["task"] = task
            predictions["model"] = "tfidf_logistic_regression"
            prediction_frames.append(predictions)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)
