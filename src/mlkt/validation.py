from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def _boolean_series(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default)
    normalised = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    invalid = sorted(set(normalised) - set(mapping) - {"nan", "none", ""})
    if invalid:
        raise ValueError(f"Invalid boolean values in {column}: {invalid}")
    return normalised.map(mapping).fillna(default).astype(bool)


def validate_augmentation_artifacts(
    source_path: str | Path, frame: pd.DataFrame
) -> dict[str, Any]:
    source = Path(source_path)
    manifest_path = source.parent / "augmentation_manifest.json"
    state_path = source.parent / "augmentation_state.json"
    if not manifest_path.exists() or not state_path.exists():
        raise ValueError("Augmentation manifest and state must accompany source data.")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with state_path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    generator = str(manifest.get("generator", ""))
    if not generator or "mock" in generator.lower():
        raise ValueError("Source pretraining cannot use mock augmentation output.")
    if state.get("status") != "complete":
        raise ValueError("Augmentation state is not complete.")
    expected_rows = (
        int(manifest.get("original_train_rows", 0))
        + int(manifest.get("generated_train_rows", 0))
        + int(manifest.get("validation_rows", 0))
    )
    if expected_rows != len(frame):
        raise ValueError(
            f"Augmentation manifest expects {expected_rows} rows, found {len(frame)}."
        )
    augmented = _boolean_series(frame, "augmented", False)
    generation_valid = _boolean_series(frame, "generation_valid", True)
    invalid_rows = int((augmented & ~generation_valid).sum())
    if invalid_rows != int(manifest.get("invalid_generations", -1)):
        raise ValueError("Invalid-generation count differs from augmentation manifest.")
    if ((frame["split"] == "validation") & augmented).any():
        raise ValueError("Augmented rows were found in source validation.")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "state_path": str(state_path.resolve()),
        "generator": generator,
        "status": state["status"],
        "planned_generation_rows": int(state["planned_generation_rows"]),
        "completed_generation_rows": int(state["completed_generation_rows"]),
        "invalid_generations": invalid_rows,
        "valid_generations": int(manifest["generated_train_rows"]) - invalid_rows,
    }


def validate_source_training_frame(
    frame: pd.DataFrame,
    emotion_names: Sequence[str],
    exclude_invalid_generations: bool = True,
    drop_exact_train_duplicates: bool = True,
    drop_conflicting_train_texts: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate source labels and remove failed augmentation fallbacks."""
    required = {
        "Utterances",
        "sentiment",
        "conversation_id",
        "split",
        *[f"emotion__{name}" for name in emotion_names],
        *[f"intensity__{name}" for name in emotion_names],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Source data is missing required columns: {missing}")
    if frame["Utterances"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Source data contains empty utterances.")
    invalid_sentiments = sorted(
        set(frame["sentiment"].dropna().astype(str))
        - {"negative", "neutral", "positive"}
    )
    if invalid_sentiments:
        raise ValueError(f"Unsupported sentiment labels: {invalid_sentiments}")
    if frame["split"].isna().any() or not {"train", "validation"}.issubset(
        set(frame["split"])
    ):
        raise ValueError("Source data requires non-empty train and validation splits.")
    for name in emotion_names:
        emotion = pd.to_numeric(frame[f"emotion__{name}"], errors="coerce")
        intensity = pd.to_numeric(frame[f"intensity__{name}"], errors="coerce")
        if emotion.isna().any() or not set(emotion.astype(int).unique()).issubset({0, 1}):
            raise ValueError(f"emotion__{name} must be binary and non-missing.")
        active_intensity = intensity[emotion.astype(int) == 1]
        if active_intensity.isna().any() or not set(
            active_intensity.astype(int).unique()
        ).issubset({1, 2, 3}):
            raise ValueError(f"Active intensity__{name} values must be in 1..3.")

    validated = frame.copy()
    augmented = _boolean_series(validated, "augmented", False)
    generation_valid = _boolean_series(validated, "generation_valid", True)
    failed = (validated["split"] == "train") & augmented & ~generation_valid
    input_rows = len(validated)
    if exclude_invalid_generations:
        validated = validated.loc[~failed].copy()
        augmented = augmented.loc[validated.index]
    duplicate_columns = ["Utterances", "sentiment"] + [
        f"emotion__{name}" for name in emotion_names
    ] + [f"intensity__{name}" for name in emotion_names]
    train_duplicate_mask = (validated["split"] == "train") & validated.duplicated(
        duplicate_columns
    )
    duplicate_train_rows = int(train_duplicate_mask.sum())
    if drop_exact_train_duplicates:
        validated = validated.loc[~train_duplicate_mask].copy()
        augmented = augmented.loc[validated.index]
    label_columns = ["sentiment"] + [
        f"emotion__{name}" for name in emotion_names
    ] + [f"intensity__{name}" for name in emotion_names]
    train_only = validated[validated["split"] == "train"].copy()
    label_signatures = train_only[label_columns].astype(str).agg("|".join, axis=1)
    signatures_per_text = label_signatures.groupby(train_only["Utterances"]).nunique()
    conflicting_texts = set(signatures_per_text[signatures_per_text > 1].index)
    conflicting_mask = (validated["split"] == "train") & validated[
        "Utterances"
    ].isin(conflicting_texts)
    conflicting_rows = int(conflicting_mask.sum())
    if drop_conflicting_train_texts:
        validated = validated.loc[~conflicting_mask].copy()
        augmented = augmented.loc[validated.index]
    train_conversations = set(
        validated.loc[validated["split"] == "train", "conversation_id"]
    )
    validation_conversations = set(
        validated.loc[validated["split"] == "validation", "conversation_id"]
    )
    conversation_overlap = train_conversations & validation_conversations
    if conversation_overlap:
        raise ValueError("Source conversation IDs overlap train and validation.")
    train_texts = set(validated.loc[validated["split"] == "train", "Utterances"])
    validation_texts = set(
        validated.loc[validated["split"] == "validation", "Utterances"]
    )
    text_overlap = train_texts & validation_texts
    if text_overlap:
        raise ValueError("Exact utterance text overlaps source train and validation.")
    report = {
        "input_rows": input_rows,
        "output_rows": len(validated),
        "excluded_invalid_generation_rows": int(failed.sum())
        if exclude_invalid_generations
        else 0,
        "invalid_generation_rows_present": int(failed.sum()),
        "exclude_invalid_generations": exclude_invalid_generations,
        "train_rows": int((validated["split"] == "train").sum()),
        "validation_rows": int((validated["split"] == "validation").sum()),
        "augmented_train_rows": int(
            ((validated["split"] == "train") & augmented).sum()
        ),
        "exact_train_duplicate_rows_detected": duplicate_train_rows,
        "exact_train_duplicate_rows_dropped": duplicate_train_rows
        if drop_exact_train_duplicates
        else 0,
        "drop_exact_train_duplicates": drop_exact_train_duplicates,
        "conflicting_train_texts_detected": len(conflicting_texts),
        "conflicting_train_rows_detected": conflicting_rows,
        "conflicting_train_rows_dropped": conflicting_rows
        if drop_conflicting_train_texts
        else 0,
        "drop_conflicting_train_texts": drop_conflicting_train_texts,
        "conversation_split_overlap": 0,
        "train_validation_text_overlap": 0,
        "unique_utterances": int(validated["Utterances"].nunique()),
    }
    if "quality" in validated:
        valid_augmented_quality = pd.to_numeric(
            validated.loc[
                (validated["split"] == "train") & augmented, "quality"
            ],
            errors="coerce",
        ).dropna()
        if not valid_augmented_quality.empty:
            report["valid_augmented_quality"] = {
                "min": float(valid_augmented_quality.min()),
                "mean": float(valid_augmented_quality.mean()),
                "median": float(valid_augmented_quality.median()),
                "p05": float(valid_augmented_quality.quantile(0.05)),
            }
    report["class_distributions"] = {}
    for split_name in ("train", "validation"):
        split_frame = validated[validated["split"] == split_name]
        report["class_distributions"][split_name] = {
            "sentiment": {
                str(label): int(count)
                for label, count in split_frame["sentiment"].value_counts().items()
            },
            "emotion_positive": {
                name: int(split_frame[f"emotion__{name}"].sum())
                for name in emotion_names
            },
            "intensity_active": {
                name: {
                    str(int(label)): int(count)
                    for label, count in split_frame.loc[
                        split_frame[f"emotion__{name}"] == 1,
                        f"intensity__{name}",
                    ].value_counts().sort_index().items()
                }
                for name in emotion_names
            },
        }
    return validated.reset_index(drop=True), report
