from __future__ import annotations

from typing import Any

import pandas as pd

from .metrics import classification_metrics, ordinal_metrics


def run_naive_baselines(
    frame: pd.DataFrame,
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the training-majority baseline."""
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
        predictors: dict[str, Any] = {"majority": [majority] * len(test)}

        for model_name, predicted in predictors.items():
            labels = (1, 2, 3, 4) if task in {"final_intensity", "drop_magnitude"} else None
            scores = classification_metrics(
                test[target_column], predicted, labels=labels
            )
            if task in {"final_intensity", "drop_magnitude"}:
                scores.update(
                    ordinal_metrics(test[target_column], predicted, labels=(1, 2, 3, 4))
                )
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
            labels = (1, 2, 3, 4) if task in {"final_intensity", "drop_magnitude"} else None
            scores = classification_metrics(
                split_frame[target_column], predicted, labels=labels
            )
            if task in {"final_intensity", "drop_magnitude"}:
                scores.update(
                    ordinal_metrics(
                        split_frame[target_column], predicted, labels=(1, 2, 3, 4)
                    )
                )
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


def run_initial_only_baseline(
    frame: pd.DataFrame,
    task: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit checkpoint-specific logistic regression using only initial intensity."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError("The baseline requires scikit-learn.") from error
    target_column = {
        "final_intensity": "final_intensity",
        "drop_magnitude": "drop_magnitude",
    }.get(task)
    if target_column is None:
        raise ValueError("Initial-only baseline supports final_intensity or drop_magnitude.")
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for checkpoint, checkpoint_frame in frame.groupby("checkpoint", sort=True):
        train = checkpoint_frame[checkpoint_frame["split"] == "train"]
        for split_name in ("validation", "test"):
            subset = checkpoint_frame[checkpoint_frame["split"] == split_name]
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced", max_iter=2000, random_state=seed
                        ),
                    ),
                ]
            )
            model.fit(train[["initial_intensity"]], train[target_column])
            predicted = model.predict(subset[["initial_intensity"]])
            scores = classification_metrics(
                subset[target_column], predicted, labels=(1, 2, 3, 4)
            )
            scores.update(
                ordinal_metrics(
                    subset[target_column], predicted, labels=(1, 2, 3, 4)
                )
            )
            metric_rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_percent": int(round(float(checkpoint) * 100)),
                    "split": split_name,
                    "task": task,
                    "model": "initial_only_logistic_regression",
                    **scores,
                }
            )
            predictions = subset[
                ["dataset", "conversation_id", "checkpoint", "initial_intensity", target_column]
            ].copy()
            predictions = predictions.rename(columns={target_column: "target"})
            predictions["prediction"] = predicted
            predictions["split"] = split_name
            predictions["task"] = task
            predictions["model"] = "initial_only_logistic_regression"
            prediction_frames.append(predictions)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def run_outcome_metadata_baselines(
    frame: pd.DataFrame,
    checkpoint: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict final intensity from pre-outcome metadata and derive drop exactly."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as error:
        raise RuntimeError("The metadata baselines require scikit-learn.") from error

    required = {
        "dataset",
        "conversation_id",
        "checkpoint",
        "split",
        "initial_intensity",
        "emotion_family",
        "problem_type",
        "final_intensity",
        "drop_magnitude",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing metadata baseline columns: {sorted(missing)}")
    subset = frame[frame["checkpoint"].astype(float).sub(checkpoint).abs() < 1e-8]
    if subset.empty:
        raise ValueError(f"No rows found for checkpoint {checkpoint}.")
    train = subset[subset["split"] == "train"]
    specifications = {
        "initial": ["initial_intensity"],
        "emotion": ["emotion_family"],
        "initial_emotion": ["initial_intensity", "emotion_family"],
        "initial_emotion_problem": [
            "initial_intensity",
            "emotion_family",
            "problem_type",
        ],
    }
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_name, feature_columns in specifications.items():
        numeric = [name for name in feature_columns if name == "initial_intensity"]
        categorical = [name for name in feature_columns if name not in numeric]
        transformers = []
        if categorical:
            transformers.append(
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
            )
        if numeric:
            transformers.append(("numeric", StandardScaler(), numeric))
        model = Pipeline(
            [
                ("features", ColumnTransformer(transformers)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2_000, random_state=seed
                    ),
                ),
            ]
        )
        model.fit(train[feature_columns], train["final_intensity"])
        for split_name in ("validation", "test"):
            split_frame = subset[subset["split"] == split_name]
            final_prediction = model.predict(split_frame[feature_columns]).astype(int)
            # All supervised rows represent a decrease, so final must be below initial.
            final_prediction = final_prediction.clip(1, 4)
            final_prediction = pd.Series(
                final_prediction, index=split_frame.index
            ).clip(upper=split_frame["initial_intensity"] - 1).astype(int)
            drop_prediction = (
                split_frame["initial_intensity"] - final_prediction
            ).astype(int)
            predictions = split_frame[
                [
                    "dataset",
                    "conversation_id",
                    "checkpoint",
                    "initial_intensity",
                    "emotion_family",
                    "problem_type",
                    "final_intensity",
                    "drop_magnitude",
                ]
            ].copy()
            predictions["final_prediction"] = final_prediction
            predictions["drop_prediction"] = drop_prediction
            predictions["split"] = split_name
            predictions["model"] = model_name
            prediction_frames.append(predictions)
            for task, target, predicted in (
                (
                    "final_intensity",
                    split_frame["final_intensity"],
                    final_prediction,
                ),
                ("drop_magnitude", split_frame["drop_magnitude"], drop_prediction),
            ):
                scores = classification_metrics(target, predicted, labels=(1, 2, 3, 4))
                scores.update(ordinal_metrics(target, predicted, labels=(1, 2, 3, 4)))
                metric_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "checkpoint_percent": int(round(checkpoint * 100)),
                        "split": split_name,
                        "task": task,
                        "model": model_name,
                        **scores,
                    }
                )
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)
