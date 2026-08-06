from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .data import ESCONV_STRATEGIES, _strategy_slug
from .metrics import classification_metrics, ordinal_metrics


def quantity_feature_columns() -> list[str]:
    columns = [
        "n_observed_strategies",
        "n_unique_strategies",
        "n_observed_supporter_turns",
    ]
    for strategy in ESCONV_STRATEGIES:
        slug = _strategy_slug(strategy)
        columns.extend(
            [f"strategy_count__{slug}", f"strategy_rate__{slug}"]
        )
    return columns


def timing_feature_columns() -> list[str]:
    columns = [
        "strategy_first_position",
        "strategy_mean_position",
        "strategy_last_position",
        "strategy_early_share",
        "strategy_late_share",
    ]
    for strategy in ESCONV_STRATEGIES:
        slug = _strategy_slug(strategy)
        columns.extend(
            [
                f"strategy_first__{slug}",
                f"strategy_mean__{slug}",
                f"strategy_last__{slug}",
            ]
        )
    return columns


def order_feature_columns() -> list[str]:
    return [
        f"strategy_transition__{_strategy_slug(left)}__{_strategy_slug(right)}"
        for left in ESCONV_STRATEGIES
        for right in ESCONV_STRATEGIES
    ]


def strategy_feature_columns(mode: str = "quantity_timing_order") -> list[str]:
    choices = {
        "quantity": quantity_feature_columns(),
        "quantity_timing": quantity_feature_columns() + timing_feature_columns(),
        "quantity_timing_order": (
            quantity_feature_columns()
            + timing_feature_columns()
            + order_feature_columns()
        ),
    }
    if mode not in choices:
        raise ValueError(f"Unknown strategy feature mode: {mode}")
    return choices[mode]


def ensure_strategy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialise all fixed-size strategy features required by models."""
    result = frame.copy()
    zero_columns = quantity_feature_columns() + order_feature_columns()
    missing_zero = {
        column: 0.0 for column in zero_columns if column not in result
    }
    missing_timing = {
        column: -1.0
        for column in timing_feature_columns()
        if column not in result
    }
    if missing_zero or missing_timing:
        result = pd.concat(
            [
                result,
                pd.DataFrame(
                    {**missing_zero, **missing_timing}, index=result.index
                ),
            ],
            axis=1,
        )
    for column in zero_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    for column in timing_feature_columns():
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(-1.0)
    return result


def parse_strategy_sequence(value: Any) -> list[str]:
    if value is None or pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(">") if part.strip()]


def parse_strategy_positions(value: Any, count: int) -> list[float]:
    if value is None or pd.isna(value) or not str(value).strip():
        return [0.0] * count
    positions = [
        float(part.strip())
        for part in str(value).split("|")
        if part.strip()
    ]
    if len(positions) != count:
        raise ValueError("Strategy sequence and position counts differ.")
    return positions


def strategy_vocabulary() -> dict[str, int]:
    return {
        "PAD": 0,
        "NO_STRATEGY": 1,
        **{strategy: index + 2 for index, strategy in enumerate(ESCONV_STRATEGIES)},
    }


def encode_strategy_sequence(
    sequence: Sequence[str], positions: Sequence[float]
) -> tuple[list[int], list[float]]:
    vocabulary = strategy_vocabulary()
    if len(sequence) != len(positions):
        raise ValueError("Strategy sequence and positions must have equal length.")
    if not sequence:
        return [vocabulary["NO_STRATEGY"]], [0.0]
    unknown = sorted(set(sequence) - set(vocabulary))
    if unknown:
        raise ValueError(f"Unknown ESConv strategies: {unknown}")
    return [vocabulary[value] for value in sequence], list(positions)


def run_nested_strategy_models(
    frame: pd.DataFrame,
    output_dir: str | Path,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate quantity, timing, and order with lightweight ordinal classifiers."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError(
            "Strategy models require scikit-learn. Install the project dependencies."
        ) from error

    data = ensure_strategy_columns(frame)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for checkpoint, checkpoint_frame in data.groupby("checkpoint", sort=True):
        train = checkpoint_frame[checkpoint_frame["split"] == "train"]
        for mode in ("quantity", "quantity_timing", "quantity_timing_order"):
            features = strategy_feature_columns(mode)
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            for target in ("final_intensity", "drop_magnitude"):
                model.fit(train[features], train[target])
                for split in ("validation", "test"):
                    subset = checkpoint_frame[checkpoint_frame["split"] == split]
                    predicted = model.predict(subset[features])
                    scores = classification_metrics(subset[target], predicted)
                    scores.update(ordinal_metrics(subset[target], predicted))
                    metric_rows.append(
                        {
                            "checkpoint": checkpoint,
                            "checkpoint_percent": int(round(float(checkpoint) * 100)),
                            "split": split,
                            "target": target,
                            "model": f"strategy_{mode}",
                            "seed": seed,
                            **scores,
                        }
                    )
                    predictions = subset[
                        ["conversation_id", "checkpoint", target]
                    ].copy()
                    predictions = predictions.rename(columns={target: "target"})
                    predictions["prediction"] = predicted
                    predictions["split"] = split
                    predictions["target_name"] = target
                    predictions["model"] = f"strategy_{mode}"
                    predictions["seed"] = seed
                    prediction_rows.append(predictions)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "nested_strategy_metrics.csv", index=False)
    predictions.to_csv(output / "nested_strategy_predictions.csv", index=False)
    figures = _plot_nested_strategy_curves(metrics, output)
    with (output / "prospective_analysis_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "seed": seed,
                "models": [
                    "strategy_quantity",
                    "strategy_quantity_timing",
                    "strategy_quantity_timing_order",
                ],
                "figures": figures,
                "causal_interpretation": False,
            },
            handle,
            indent=2,
        )
    return metrics, predictions


def _plot_nested_strategy_curves(
    metrics: pd.DataFrame, output_dir: Path
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    created: list[str] = []
    test = metrics[metrics["split"] == "test"]
    for target, target_frame in test.groupby("target", sort=True):
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for model, group in target_frame.groupby("model", sort=True):
            group = group.sort_values("checkpoint_percent")
            axis.plot(
                group["checkpoint_percent"],
                group["f1_macro"],
                marker="o",
                label=model.removeprefix("strategy_").replace("_", " + "),
            )
        axis.set_xlabel("Observed dialogue (%)")
        axis.set_ylabel("Macro-F1")
        axis.set_title(
            f"Strategy contribution: {str(target).replace('_', ' ').title()}"
        )
        axis.set_xticks([10, 25, 50, 75, 100])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = output_dir / f"{target}_quantity_timing_order_f1.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(str(path))
    return created


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def retrospective_ordinal_analysis(
    frame: pd.DataFrame,
    output_dir: str | Path,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate observational strategy associations at the 100% checkpoint."""
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
    except ImportError as error:
        raise RuntimeError(
            "Retrospective ordinal analysis requires statsmodels."
        ) from error

    complete = ensure_strategy_columns(
        frame[frame["checkpoint"] == frame["checkpoint"].max()]
    )
    controls = [
        "initial_intensity",
        "n_total_turns",
        "n_observed_supporter_turns",
    ]
    categorical_controls = ["emotion", "problem_type"]
    control_frame = pd.get_dummies(
        complete[controls + categorical_controls],
        columns=categorical_controls,
        drop_first=True,
        dtype=float,
    )
    feature_blocks = {
        "quantity": [
            feature
            for feature in quantity_feature_columns()
            if feature not in controls
        ],
        "timing": timing_feature_columns(),
        "order": order_feature_columns(),
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for target in ("final_intensity", "drop_magnitude"):
        endog = complete[target].astype(int)
        for block_name, block_features in feature_blocks.items():
            varying_features = [
                feature
                for feature in block_features
                if complete[feature].nunique() >= 2
            ]
            if not varying_features:
                continue
            design = pd.concat(
                [
                    complete[varying_features].astype(float).reset_index(drop=True),
                    control_frame.reset_index(drop=True),
                ],
                axis=1,
            )
            design = design.loc[:, design.nunique() > 1]
            # Remove exact linear dependencies (for example total counts vs.
            # per-strategy counts) before fitting an unregularised ordinal model.
            independent_columns: list[str] = []
            current_rank = 0
            for column in design.columns:
                candidate = independent_columns + [column]
                rank = np.linalg.matrix_rank(
                    design[candidate].to_numpy(dtype=float)
                )
                if rank > current_rank:
                    independent_columns.append(column)
                    current_rank = rank
            design = design[independent_columns]
            retained_features = [
                feature for feature in varying_features if feature in design
            ]
            if not retained_features:
                continue
            try:
                fitted = OrderedModel(
                    endog.reset_index(drop=True),
                    design,
                    distr="logit",
                ).fit(method="bfgs", disp=False)
            except Exception:
                continue
            bootstrap_coefficients: dict[str, list[float]] = {
                feature: [] for feature in retained_features
            }
            for _ in range(bootstrap_samples):
                indices = rng.integers(0, len(complete), len(complete))
                sampled_y = endog.iloc[indices].reset_index(drop=True)
                sampled_x = design.iloc[indices].reset_index(drop=True)
                if sampled_y.nunique() < 2:
                    continue
                try:
                    sampled_fit = OrderedModel(
                        sampled_y, sampled_x, distr="logit"
                    ).fit(method="bfgs", disp=False)
                    for feature in retained_features:
                        bootstrap_coefficients[feature].append(
                            float(sampled_fit.params[feature])
                        )
                except Exception:
                    continue
            for feature in retained_features:
                values = bootstrap_coefficients[feature]
                low, high = (
                    np.quantile(values, [0.025, 0.975])
                    if values
                    else (np.nan, np.nan)
                )
                coefficient = float(fitted.params[feature])
                rows.append(
                    {
                        "target": target,
                        "feature_block": block_name,
                        "feature": feature,
                        "coefficient": coefficient,
                        "odds_ratio": float(np.exp(coefficient)),
                        "p_value": float(fitted.pvalues[feature]),
                        "ci_2.5": float(low),
                        "ci_97.5": float(high),
                        "bootstrap_successes": len(values),
                        "interpretation": "observational_association",
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_fdr"] = benjamini_hochberg(result["p_value"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "retrospective_ordinal_associations.csv", index=False)
    with (output / "retrospective_analysis_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "checkpoint": 1.0,
                "bootstrap_samples": bootstrap_samples,
                "seed": seed,
                "causal_interpretation": False,
                "controls": controls + categorical_controls,
            },
            handle,
            indent=2,
        )
    return result
