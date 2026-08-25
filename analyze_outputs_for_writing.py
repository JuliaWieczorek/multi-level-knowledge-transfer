"""Reproducible audit of completed experiment outputs for publication writing.

The script never retrains models.  It reads the durable CSV/JSON artifacts in
``outputs`` and writes compact, publication-oriented tables to
``outputs/article_chapter_analysis``.
"""

from __future__ import annotations

import json
import hashlib
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DEST = OUTPUTS / "article_chapter_analysis"
SEEDS = [42, 52, 62, 72, 82]
CHECKPOINTS = [10, 25, 50, 75, 100]
T_CRITICAL_975 = {
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


def variant_label(row: pd.Series) -> str:
    modality = row["modality"]
    transfer = bool(row["transfer"])
    initial = bool(row["use_initial_intensity"])
    if modality == "strategy":
        return "strategy_only"
    if modality == "text" and transfer:
        return "transferred_text"
    if modality == "text" and not transfer:
        return "vanilla_text"
    if modality == "text_strategy" and transfer and initial:
        return "transferred_text_strategy_initial"
    if modality == "text_strategy" and transfer:
        return "transferred_text_strategy"
    if modality == "text_strategy" and not transfer:
        return "vanilla_text_strategy"
    raise ValueError(f"Unknown variant: {row.to_dict()}")


def t_ci_half_width(values: pd.Series, confidence: float = 0.95) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    if confidence != 0.95:
        raise ValueError("The dependency-free audit currently supports 95% CIs only")
    df = len(clean) - 1
    critical = T_CRITICAL_975.get(df, 1.96)
    return float(critical * clean.std(ddof=1) / math.sqrt(len(clean)))


def signflip_p(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(clean) == 0 or np.allclose(clean, 0):
        return 1.0
    observed = abs(float(clean.mean()))
    permuted = [
        abs(float(np.mean(clean * np.asarray(signs))))
        for signs in product((-1.0, 1.0), repeat=len(clean))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def summarise_deltas(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["delta"]
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            n=int(values.notna().sum()),
            delta_mean=float(values.mean()),
            delta_std=float(values.std(ddof=1)),
            delta_t_ci95=t_ci_half_width(values),
            wins=int((values > 0).sum()),
            ties=int(np.isclose(values, 0).sum()),
            losses=int((values < 0).sum()),
            paired_signflip_p=signflip_p(values),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_report(name: str, corpus: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    report_dir = OUTPUTS / name
    runs = pd.read_csv(report_dir / "all_run_metrics.csv")
    aggregates = pd.read_csv(report_dir / "metrics_by_checkpoint.csv")
    for frame in (runs, aggregates):
        frame["variant"] = frame.apply(variant_label, axis=1)
        frame["corpus"] = corpus
    return runs, aggregates


def publication_baseline_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_dir = OUTPUTS / "publication_baselines" / "original"
    baseline_files = sorted(baseline_dir.glob("*_metrics.csv"))
    baseline_metrics = (
        pd.concat(
            [pd.read_csv(path).assign(source_file=str(path)) for path in baseline_files],
            ignore_index=True,
        )
        if baseline_files
        else pd.DataFrame()
    )
    metadata_path = OUTPUTS / "publication_baselines" / "metadata_100" / "metadata_metrics.csv"
    metadata_metrics = (
        pd.read_csv(metadata_path).assign(source_file=str(metadata_path))
        if metadata_path.exists()
        else pd.DataFrame()
    )
    return baseline_metrics, metadata_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publication_claim_checks(
    comparisons_collapsed: pd.DataFrame,
    confusion: pd.DataFrame,
    retrospective_wald: pd.DataFrame,
) -> pd.DataFrame:
    article_path = (
        ROOT.parent
        / "papers_dissertation"
        / "output"
        / "IEEE_Temporal_Emotion_Intensity_Forecasting_in_Emotional_Support_Dialogues_with_Multi-Level_Knowledge_Transfer_and_Support-Strategy_Modeling"
        / "paper_sections.tex"
    )
    thesis_path = (
        ROOT.parent
        / "PhD_Coventry_Overleaf_official"
        / "chapters"
        / "chapter6_temporal_multilevel_transfer.tex"
    )
    article = article_path.read_text(encoding="utf-8") if article_path.exists() else ""
    thesis = thesis_path.read_text(encoding="utf-8") if thesis_path.exists() else ""

    transfer = comparisons_collapsed[
        comparisons_collapsed.corpus.eq("original")
        & comparisons_collapsed.comparison.eq("parameter_transfer_text_strategy")
        & comparisons_collapsed.target.eq("drop_magnitude")
    ].iloc[0]
    rare = confusion[
        confusion.corpus.eq("target_augmented")
        & confusion.target.eq("final_intensity")
        & confusion.variant.eq("transferred_text")
        & confusion["class"].eq(4)
    ].recall.mean() - confusion[
        confusion.corpus.eq("original")
        & confusion.target.eq("final_intensity")
        & confusion.variant.eq("transferred_text")
        & confusion["class"].eq(4)
    ].recall.mean()
    raw_signals = int((retrospective_wald.p_value < 0.05).sum())
    minimum_q = float(retrospective_wald.p_value_fdr.min())
    checks = [
        {
            "claim": "strategy-aware transfer delta for decrease",
            "source_value": float(transfer.delta_mean),
            "article_token": "0.020",
            "thesis_token": "0.0197",
        },
        {
            "claim": "rare final-class recall gain for transferred text",
            "source_value": float(rare),
            "article_token": "0.112",
            "thesis_token": "0.112",
        },
        {
            "claim": "retrospective coefficient count",
            "source_value": len(retrospective_wald),
            "article_token": "216 coefficients",
            "thesis_token": "216 strategy coefficients",
        },
        {
            "claim": "nominal retrospective signals",
            "source_value": raw_signals,
            "article_token": "16 of 216",
            "thesis_token": "Sixteen have nominal",
        },
        {
            "claim": "minimum retrospective FDR q",
            "source_value": minimum_q,
            "article_token": "q=.335",
            "thesis_token": "q=.335",
        },
    ]
    result = pd.DataFrame(checks)
    result["article_present"] = result.article_token.map(lambda token: token in article)
    result["thesis_present"] = result.thesis_token.map(lambda token: token in thesis)
    return result


def audit_completeness(runs: pd.DataFrame) -> pd.DataFrame:
    expected_variants = {
        "strategy_only",
        "transferred_text",
        "transferred_text_strategy",
        "transferred_text_strategy_initial",
        "vanilla_text",
        "vanilla_text_strategy",
    }
    rows = []
    for corpus, group in runs.groupby("corpus"):
        run_keys = group[
            ["checkpoint_percent", "seed", "variant"]
        ].drop_duplicates()
        metric_keys = group[
            ["checkpoint_percent", "seed", "variant", "split", "target"]
        ]
        rows.append(
            {
                "corpus": corpus,
                "metric_rows": len(group),
                "unique_runs": len(run_keys),
                "duplicate_metric_keys": int(metric_keys.duplicated().sum()),
                "missing_metric_values": int(group.select_dtypes("number").isna().sum().sum()),
                "seeds": ",".join(map(str, sorted(group.seed.unique()))),
                "checkpoints": ",".join(map(str, sorted(group.checkpoint_percent.unique()))),
                "variants": ",".join(sorted(group.variant.unique())),
                "expected_variants_present": set(group.variant.unique()) == expected_variants,
            }
        )
    return pd.DataFrame(rows)


def comparison_rows(test_runs: pd.DataFrame) -> pd.DataFrame:
    comparisons = {
        "parameter_transfer_text": ("transferred_text", "vanilla_text"),
        "parameter_transfer_text_strategy": (
            "transferred_text_strategy",
            "vanilla_text_strategy",
        ),
        "add_strategies_to_transferred_text": (
            "transferred_text_strategy",
            "transferred_text",
        ),
        "add_strategies_to_vanilla_text": (
            "vanilla_text_strategy",
            "vanilla_text",
        ),
        "add_initial_intensity": (
            "transferred_text_strategy_initial",
            "transferred_text_strategy",
        ),
    }
    index = ["corpus", "checkpoint_percent", "target", "seed"]
    wide = test_runs.pivot_table(index=index, columns="variant", values="f1_macro")
    parts = []
    for comparison, (treatment, control) in comparisons.items():
        piece = (wide[treatment] - wide[control]).rename("delta").reset_index()
        piece["comparison"] = comparison
        parts.append(piece)
    return pd.concat(parts, ignore_index=True)


def collapsed_delta_summary(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Average checkpoints within seed before uncertainty across five seeds."""
    per_seed = (
        raw.groupby(group_cols + ["seed"], dropna=False, as_index=False)["delta"]
        .mean()
    )
    return summarise_deltas(per_seed, group_cols)


def horizon_delta_summary(raw: pd.DataFrame) -> pd.DataFrame:
    phased = raw.copy()
    phased["horizon"] = pd.cut(
        phased["checkpoint_percent"],
        bins=[0, 25, 50, 100],
        labels=["early_10_25", "middle_50", "late_75_100"],
        include_lowest=True,
    )
    per_seed = (
        phased.groupby(
            ["corpus", "comparison", "target", "horizon", "seed"],
            observed=True,
            as_index=False,
        )["delta"]
        .mean()
    )
    return summarise_deltas(
        per_seed, ["corpus", "comparison", "target", "horizon"]
    )


def augmented_deltas(test_runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = ["checkpoint_percent", "target", "variant", "seed"]
    wide = test_runs.pivot_table(index=index, columns="corpus", values="f1_macro")
    raw = (wide["target_augmented"] - wide["original"]).rename("delta").reset_index()
    raw["comparison"] = "target_augmented_minus_original"
    summary = summarise_deltas(raw, ["target", "variant", "checkpoint_percent"])
    return raw, summary


def temporal_change(test_runs: pd.DataFrame) -> pd.DataFrame:
    index = ["corpus", "target", "variant", "seed"]
    wide = test_runs.pivot_table(index=index, columns="checkpoint_percent", values="f1_macro")
    raw = (wide[100] - wide[10]).rename("delta").reset_index()
    return summarise_deltas(raw, ["corpus", "target", "variant"])


def rank_models(aggregates: pd.DataFrame) -> pd.DataFrame:
    test = aggregates[aggregates.split.eq("test")].copy()
    test["rank_at_checkpoint"] = test.groupby(
        ["corpus", "target", "checkpoint_percent"]
    )["f1_macro_mean"].rank(method="min", ascending=False)
    test["f1_macro_t_ci95"] = (
        test["f1_macro_std"] * T_CRITICAL_975[4] / math.sqrt(5)
    )
    keep = [
        "corpus",
        "target",
        "checkpoint_percent",
        "variant",
        "rank_at_checkpoint",
        "f1_macro_mean",
        "f1_macro_std",
        "f1_macro_ci95",
        "f1_macro_t_ci95",
        "accuracy_mean",
        "mae_mean",
        "quadratic_weighted_kappa_mean",
    ]
    return test[keep].sort_values(
        ["corpus", "target", "checkpoint_percent", "rank_at_checkpoint", "variant"]
    )


def best_models(ranked: pd.DataFrame) -> pd.DataFrame:
    return ranked.loc[ranked.rank_at_checkpoint.eq(1)].reset_index(drop=True)


def joint_diagnostics(test_runs: pd.DataFrame) -> pd.DataFrame:
    # Joint metrics are duplicated once per target in each metrics file.
    deduped = test_runs.drop_duplicates(
        ["corpus", "checkpoint_percent", "variant", "seed"]
    )
    metrics = [
        "joint_consistency_rate",
        "derived_drop_accuracy",
        "derived_drop_mae",
        "invalid_derived_drop_rate",
    ]
    grouped = deduped.groupby(
        ["corpus", "checkpoint_percent", "variant"], sort=True
    )[metrics]
    result = grouped.agg(["mean", "std"]).reset_index()
    result.columns = [
        "_".join(map(str, col)).rstrip("_") if isinstance(col, tuple) else col
        for col in result.columns
    ]
    return result


def confusion_diagnostics(report_name: str, corpus: str) -> pd.DataFrame:
    mapping = {
        "strategy_strategy": "strategy_only",
        "text_vanilla": "vanilla_text",
        "text_transfer": "transferred_text",
        "text_strategy_vanilla": "vanilla_text_strategy",
        "text_strategy_transfer": "transferred_text_strategy",
        "text_strategy_transfer_initial": "transferred_text_strategy_initial",
    }
    rows = []
    directory = OUTPUTS / report_name / "confusion_matrices"
    for path in sorted(directory.glob("*.csv")):
        stem = path.stem
        checkpoint = int(stem[:3])
        target_short = stem.rsplit("_", 1)[-1]
        target = "final_intensity" if target_short == "final" else "drop_magnitude"
        middle = stem[4 : -(len(target_short) + 1)]
        variant = mapping[middle]
        matrix = pd.read_csv(path, index_col=0)
        values = matrix.to_numpy()
        for idx, label in enumerate(range(1, 5)):
            support = int(values[idx, :].sum())
            predicted = int(values[:, idx].sum())
            true_positive = int(values[idx, idx])
            recall = true_positive / support if support else float("nan")
            precision = true_positive / predicted if predicted else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "corpus": corpus,
                    "checkpoint_percent": checkpoint,
                    "target": target,
                    "variant": variant,
                    "class": label,
                    "support_across_5_seeds": support,
                    "predicted_across_5_seeds": predicted,
                    "true_positive_across_5_seeds": true_positive,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return pd.DataFrame(rows)


def prospective_strategy_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = OUTPUTS / "strategy_analysis" / "prospective" / "nested_strategy_metrics.csv"
    metrics = pd.read_csv(path)
    test = metrics[metrics.split.eq("test")].copy()
    index = ["checkpoint_percent", "target", "seed"]
    wide = test.pivot_table(index=index, columns="model", values="f1_macro")
    raw = pd.concat(
        [
            (wide["strategy_quantity_timing"] - wide["strategy_quantity"])
            .rename("delta")
            .reset_index()
            .assign(comparison="add_timing"),
            (wide["strategy_quantity_timing_order"] - wide["strategy_quantity_timing"])
            .rename("delta")
            .reset_index()
            .assign(comparison="add_order"),
            (wide["strategy_quantity_timing_order"] - wide["strategy_quantity"])
            .rename("delta")
            .reset_index()
            .assign(comparison="timing_and_order_vs_quantity"),
        ],
        ignore_index=True,
    )
    return test, raw


def source_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    summary_rows = []
    per_emotion_rows = []
    for seed in SEEDS:
        directory = OUTPUTS / "source_mtl" / f"seed_{seed}"
        metric_rows.append(pd.read_csv(directory / "source_validation_metrics.csv").iloc[0])
        summary = json.loads((directory / "source_summary.json").read_text(encoding="utf-8"))
        report = summary["input_report"]
        summary_rows.append(
            {
                "seed": seed,
                "best_epoch": summary["best_epoch"],
                "selection_score": summary["best_selection_score"],
                "duration_seconds": summary["duration_seconds"],
                "input_rows": report["input_rows"],
                "output_rows": report["output_rows"],
                "invalid_generation_rows": report["invalid_generation_rows_present"],
                "valid_augmented_rows": report["augmentation_artifacts"]["valid_generations"],
                "planned_generation_rows": report["augmentation_artifacts"]["planned_generation_rows"],
                "augmented_train_rows_after_cleaning": report["augmented_train_rows"],
                "exact_duplicates_dropped": report["exact_train_duplicate_rows_dropped"],
                "conflicting_rows_dropped": report["conflicting_train_rows_dropped"],
                "augmentation_quality_mean": report["valid_augmented_quality"]["mean"],
                "augmentation_quality_p05": report["valid_augmented_quality"]["p05"],
            }
        )
        per_class = pd.read_csv(directory / "source_validation_per_class_metrics.csv")
        emotion = per_class[(per_class.task == "emotion") & (per_class.label == "1")].copy()
        emotion.insert(0, "seed", seed)
        per_emotion_rows.append(emotion[["seed", "emotion", "precision", "recall", "f1", "support", "average_precision", "roc_auc"]])
    metrics = pd.DataFrame(metric_rows)
    summaries = pd.DataFrame(summary_rows)
    per_emotion = pd.concat(per_emotion_rows, ignore_index=True)
    emotion_summary = (
        per_emotion.groupby("emotion")
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            support_per_seed=("support", "mean"),
            average_precision_mean=("average_precision", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
        )
        .reset_index()
        .sort_values("f1_mean", ascending=False)
    )
    return metrics, summaries, emotion_summary


def overall_variant_summary(test_runs: pd.DataFrame) -> pd.DataFrame:
    return (
        test_runs.groupby(["corpus", "target", "variant"])
        .agg(
            f1_macro_mean_across_25_runs=("f1_macro", "mean"),
            f1_macro_std_across_25_runs=("f1_macro", "std"),
            accuracy_mean_across_25_runs=("accuracy", "mean"),
            mae_mean_across_25_runs=("mae", "mean"),
            qwk_mean_across_25_runs=("quadratic_weighted_kappa", "mean"),
        )
        .reset_index()
        .sort_values(["corpus", "target", "f1_macro_mean_across_25_runs"], ascending=[True, True, False])
    )


def original_majority_baselines() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "data" / "processed" / "esconv_checkpoints.csv")
    rows = []
    for checkpoint, group in frame.groupby("checkpoint", sort=True):
        train = group[group.split.eq("train")]
        test = group[group.split.eq("test")]
        for target in ("final_intensity", "drop_magnitude"):
            majority = int(train[target].mode().iloc[0])
            truth = test[target].to_numpy(dtype=int)
            predicted = np.full(len(truth), majority, dtype=int)
            support = int(np.sum(truth == majority))
            precision = support / len(truth)
            recall = 1.0
            class_f1 = 2 * precision * recall / (precision + recall)
            rows.append(
                {
                    "checkpoint_percent": int(round(checkpoint * 100)),
                    "target": target,
                    "model": "training_majority",
                    "majority_class": majority,
                    "test_n": len(truth),
                    "accuracy": float(np.mean(truth == predicted)),
                    "f1_macro": float(class_f1 / 4),
                    "mae": float(np.mean(np.abs(truth - predicted))),
                    "quadratic_weighted_kappa": 0.0,
                }
            )
    return pd.DataFrame(rows)


def prediction_integrity_audit() -> pd.DataFrame:
    run_names = [
        "strategy",
        "transferred_text",
        "transferred_text_strategy",
        "transferred_text_strategy_initial",
        "vanilla_text",
        "vanilla_text_strategy",
    ]
    rows = []
    for split in ("validation", "test"):
        ids_equal = []
        targets_equal = []
        duplicate_ids = []
        invalid_labels = []
        probability_errors = []
        sample_counts = []
        for seed in SEEDS:
            for checkpoint in CHECKPOINTS:
                for run_name in run_names:
                    relative = (
                        Path(f"seed_{seed}")
                        / f"checkpoint_{checkpoint:03d}"
                        / run_name
                        / f"{split}_predictions.csv"
                    )
                    original = pd.read_csv(OUTPUTS / "temporal" / relative)
                    augmented = pd.read_csv(OUTPUTS / "temporal_augmented" / relative)
                    ids_equal.append(
                        original.conversation_id.tolist()
                        == augmented.conversation_id.tolist()
                    )
                    targets_equal.append(
                        original[["final_target", "drop_target"]].equals(
                            augmented[["final_target", "drop_target"]]
                        )
                    )
                    duplicate_ids.extend(
                        [
                            int(original.conversation_id.duplicated().sum()),
                            int(augmented.conversation_id.duplicated().sum()),
                        ]
                    )
                    for predictions in (original, augmented):
                        sample_counts.append(len(predictions))
                        invalid_labels.append(
                            int(
                                (~predictions.final_target.isin([1, 2, 3, 4])).sum()
                                + (~predictions.drop_target.isin([1, 2, 3, 4])).sum()
                                + (~predictions.final_prediction.isin([1, 2, 3, 4])).sum()
                                + (~predictions.drop_prediction.isin([1, 2, 3, 4])).sum()
                            )
                        )
                        for prefix in ("final_probability_", "drop_probability_"):
                            probability_columns = [f"{prefix}{label}" for label in range(1, 5)]
                            sums = predictions[probability_columns].sum(axis=1)
                            probability_errors.append(float((sums - 1.0).abs().max()))
        rows.append(
            {
                "split": split,
                "paired_run_files_compared": len(ids_equal),
                "all_original_augmented_ids_equal": all(ids_equal),
                "all_original_augmented_targets_equal": all(targets_equal),
                "sample_count_min": min(sample_counts),
                "sample_count_max": max(sample_counts),
                "duplicate_conversation_ids": sum(duplicate_ids),
                "invalid_target_or_prediction_labels": sum(invalid_labels),
                "maximum_probability_sum_absolute_error": max(probability_errors),
            }
        )
    return pd.DataFrame(rows)


def temporal_manifest_tables(
    test_runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_name_to_variant = {
        "strategy": "strategy_only",
        "transferred_text": "transferred_text",
        "transferred_text_strategy": "transferred_text_strategy",
        "transferred_text_strategy_initial": "transferred_text_strategy_initial",
        "vanilla_text": "vanilla_text",
        "vanilla_text_strategy": "vanilla_text_strategy",
    }
    rows = []
    for corpus, directory_name in (
        ("original", "temporal"),
        ("target_augmented", "temporal_augmented"),
    ):
        for seed in SEEDS:
            for checkpoint in CHECKPOINTS:
                for run_name, variant in run_name_to_variant.items():
                    path = (
                        OUTPUTS
                        / directory_name
                        / f"seed_{seed}"
                        / f"checkpoint_{checkpoint:03d}"
                        / run_name
                        / "run_manifest.json"
                    )
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "corpus": corpus,
                            "seed": seed,
                            "checkpoint_percent": checkpoint,
                            "variant": variant,
                            "best_epoch": manifest["best_epoch"],
                            "best_validation_selection_score": manifest[
                                "best_selection_score"
                            ],
                            "duration_seconds": manifest["duration_seconds"],
                            "device": manifest["device"],
                            "precision": manifest["precision"],
                        }
                    )
    manifests = pd.DataFrame(rows)
    cost = (
        manifests.groupby(["corpus", "variant"], as_index=False)
        .agg(
            runs=("seed", "size"),
            total_gpu_hours=("duration_seconds", lambda x: float(x.sum() / 3600)),
            duration_seconds_mean=("duration_seconds", "mean"),
            duration_seconds_std=("duration_seconds", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            best_epoch_std=("best_epoch", "std"),
            best_epoch_at_10_count=("best_epoch", lambda x: int((x == 10).sum())),
            selection_score_mean=("best_validation_selection_score", "mean"),
        )
        .sort_values(["corpus", "variant"])
    )

    split_scores = (
        pd.concat(
            [
                pd.read_csv(OUTPUTS / "report_tci" / "all_run_metrics.csv").assign(
                    corpus="original"
                ),
                pd.read_csv(
                    OUTPUTS / "report_augmented_tci" / "all_run_metrics.csv"
                ).assign(corpus="target_augmented"),
            ],
            ignore_index=True,
        )
        .assign(variant=lambda x: x.apply(variant_label, axis=1))
        .groupby(
            ["corpus", "checkpoint_percent", "variant", "seed", "split"],
            as_index=False,
        )
        .f1_macro.mean()
        .pivot_table(
            index=["corpus", "checkpoint_percent", "variant", "seed"],
            columns="split",
            values="f1_macro",
        )
        .reset_index()
    )
    alignment_rows = []
    for corpus, group in split_scores.groupby("corpus"):
        alignment_rows.append(
            {
                "corpus": corpus,
                "runs": len(group),
                "validation_test_pearson": float(
                    group.validation.corr(group.test, method="pearson")
                ),
                "validation_test_spearman": float(
                    group.validation.rank(method="average").corr(
                        group.test.rank(method="average"), method="pearson"
                    )
                ),
                "test_minus_validation_mean": float(
                    (group.test - group.validation).mean()
                ),
                "test_minus_validation_std": float(
                    (group.test - group.validation).std(ddof=1)
                ),
            }
        )
    return manifests, cost, pd.DataFrame(alignment_rows)


def save(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(DEST / name, index=False, float_format="%.8f")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    # This directory is owned by this audit; remove stale tables after schema changes.
    for stale in DEST.glob("*.csv"):
        stale.unlink()
    manifest_path = DEST / "analysis_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    original_runs, original_aggregates = load_report("report_tci", "original")
    augmented_runs, augmented_aggregates = load_report(
        "report_augmented_tci", "target_augmented"
    )
    runs = pd.concat([original_runs, augmented_runs], ignore_index=True)
    aggregates = pd.concat([original_aggregates, augmented_aggregates], ignore_index=True)
    test_runs = runs[runs.split.eq("test")].copy()

    completeness = audit_completeness(runs)
    ranked = rank_models(aggregates)
    comparisons_raw = comparison_rows(test_runs)
    comparisons = summarise_deltas(
        comparisons_raw,
        ["corpus", "comparison", "target", "checkpoint_percent"],
    )
    comparisons_collapsed = collapsed_delta_summary(
        comparisons_raw, ["corpus", "comparison", "target"]
    )
    comparisons_horizon = horizon_delta_summary(comparisons_raw)
    augmented_raw, augmented_summary = augmented_deltas(test_runs)
    augmented_collapsed = collapsed_delta_summary(
        augmented_raw, ["comparison", "target", "variant"]
    )
    temporal = temporal_change(test_runs)
    confusion = pd.concat(
        [
            confusion_diagnostics("report_tci", "original"),
            confusion_diagnostics("report_augmented_tci", "target_augmented"),
        ],
        ignore_index=True,
    )
    prospective, prospective_deltas = prospective_strategy_tables()
    retrospective = pd.read_csv(
        OUTPUTS
        / "strategy_analysis"
        / "retrospective"
        / "retrospective_ordinal_associations.csv"
    )
    wald_path = (
        OUTPUTS
        / "strategy_analysis_wald"
        / "retrospective"
        / "retrospective_ordinal_associations.csv"
    )
    retrospective_wald = pd.read_csv(wald_path) if wald_path.exists() else pd.DataFrame()
    baseline_metrics, metadata_metrics = publication_baseline_tables()
    source_metrics, source_summaries, source_emotions = source_tables()
    temporal_manifests, temporal_cost, validation_test = temporal_manifest_tables(
        test_runs
    )

    save(completeness, "00_completeness_audit.csv")
    save(prediction_integrity_audit(), "01_prediction_integrity_audit.csv")
    save(original_majority_baselines(), "02_majority_baselines_original.csv")
    save(overall_variant_summary(test_runs), "03_overall_variant_summary.csv")
    save(ranked, "04_test_performance_ranked.csv")
    save(best_models(ranked), "05_best_model_per_checkpoint.csv")
    save(comparisons_raw, "06_component_deltas_per_seed.csv")
    save(comparisons, "07_component_deltas_summary.csv")
    save(comparisons_collapsed, "08_component_deltas_across_checkpoints.csv")
    save(comparisons_horizon, "09_component_deltas_by_horizon.csv")
    save(augmented_raw, "10_augmentation_deltas_per_seed.csv")
    save(augmented_summary, "11_augmentation_deltas_summary.csv")
    save(augmented_collapsed, "12_augmentation_deltas_across_checkpoints.csv")
    save(temporal, "13_temporal_10_to_100_change.csv")
    save(joint_diagnostics(test_runs), "14_joint_consistency.csv")
    save(confusion, "15_class_diagnostics.csv")
    save(prospective, "16_prospective_strategy_metrics.csv")
    save(prospective_deltas, "17_prospective_strategy_deltas.csv")
    save(retrospective.sort_values("p_value_fdr"), "18_retrospective_associations_ranked.csv")
    save(source_metrics, "19_source_validation_metrics.csv")
    save(source_summaries, "20_source_training_summary.csv")
    save(source_emotions, "21_source_emotion_diagnostics.csv")
    save(temporal_manifests, "22_temporal_run_manifests.csv")
    save(temporal_cost, "23_temporal_training_cost.csv")
    save(validation_test, "24_validation_test_alignment.csv")
    save(baseline_metrics, "25_publication_baselines.csv")
    save(metadata_metrics, "26_metadata_baselines.csv")
    save(
        retrospective_wald.sort_values("p_value_fdr")
        if not retrospective_wald.empty
        else retrospective_wald,
        "27_retrospective_wald_associations.csv",
    )
    claim_checks = publication_claim_checks(
        comparisons_collapsed, confusion, retrospective_wald
    )
    save(claim_checks, "28_publication_claim_checks.csv")

    augmented_dataset = ROOT / "data" / "processed" / "esconv_checkpoints_augmented.csv"
    integrity = {
        "original_temporal_runs": int(completeness.loc[
            completeness.corpus.eq("original"), "unique_runs"
        ].iloc[0]),
        "augmented_temporal_runs": int(completeness.loc[
            completeness.corpus.eq("target_augmented"), "unique_runs"
        ].iloc[0]),
        "publication_baseline_metric_files": len(
            list((OUTPUTS / "publication_baselines" / "original").glob("*_metrics.csv"))
        ),
        "metadata_baseline_rows": len(metadata_metrics),
        "wald_coefficients": len(retrospective_wald),
        "publication_claim_checks_passed": bool(
            claim_checks.article_present.all() and claim_checks.thesis_present.all()
        ),
        "target_augmented_dataset": {
            "path": str(augmented_dataset),
            "present": augmented_dataset.exists(),
            "sha256": sha256_file(augmented_dataset) if augmented_dataset.exists() else None,
            "expected_sha256": "8bf2eb871050b1f03ae205314745b83f84498c60973c5f2b416478098174dfcd",
        },
    }
    (DEST / "publication_integrity_manifest.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )

    manifest = {
        "inputs": {
            "original_metric_rows": len(original_runs),
            "augmented_metric_rows": len(augmented_runs),
            "prospective_metric_rows": len(prospective),
            "retrospective_coefficients": len(retrospective),
            "source_seeds": len(source_metrics),
            "publication_baseline_rows": len(baseline_metrics),
            "metadata_baseline_rows": len(metadata_metrics),
            "wald_coefficients": len(retrospective_wald),
        },
        "critical_notes": [
            "Publication reports use two-sided Student-t intervals for n=5 seeds.",
            "Paired exact sign-flip p-values are descriptive with n=5 and cannot be below 0.0625 in a two-sided test.",
            "Prospective nested strategy models contain one seed (42), so they are exploratory.",
            "Retrospective FDR correction is global across all 216 coefficients; no coefficient has q<0.05.",
            "Analytic Wald intervals are primary; the preserved B=100 percentile bootstrap is a sensitivity analysis.",
            "outputs/outcome_ceiling and outputs/source_aligned_mtl are empty and are not analyzed as completed experiments.",
        ],
    }
    (DEST / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
