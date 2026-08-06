from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

DEFAULT_CHECKPOINTS = (0.10, 0.25, 0.50, 0.75, 1.00)


def _validate_checkpoints(checkpoints: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in checkpoints)
    if not values or any(value <= 0 or value > 1 for value in values):
        raise ValueError("Checkpoints must be in the interval (0, 1].")
    if tuple(sorted(set(values))) != values:
        raise ValueError("Checkpoints must be unique and sorted.")
    return values


def _prefix_size(total: int, fraction: float) -> int:
    if total < 1:
        raise ValueError("A dialogue must contain at least one turn.")
    return min(total, max(1, math.ceil(total * fraction)))


def _normalise_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _as_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"-?\d+", str(value))
    return int(match.group()) if match else None


def change_label(initial: int, final: int) -> str:
    if final < initial:
        return "decrease"
    if final > initial:
        return "increase"
    return "same"


def _strategy_slug(strategy: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", strategy.lower()).strip("_")
    return slug or "unknown"


def _strategy_features(sequence: Sequence[str]) -> dict[str, Any]:
    counts = Counter(sequence)
    total = len(sequence)
    result: dict[str, Any] = {
        "strategy_sequence": " > ".join(sequence),
        "n_observed_strategies": total,
        "n_unique_strategies": len(counts),
        "first_strategy": sequence[0] if sequence else "",
        "last_strategy": sequence[-1] if sequence else "",
        "strategy_transitions": " | ".join(
            f"{left} -> {right}" for left, right in zip(sequence, sequence[1:])
        ),
    }
    for strategy, count in counts.items():
        slug = _strategy_slug(strategy)
        result[f"strategy_count__{slug}"] = count
        result[f"strategy_rate__{slug}"] = count / total if total else 0.0
    return result


def build_esconv_checkpoints(
    source: str | Path | list[dict[str, Any]],
    checkpoints: Sequence[float] = DEFAULT_CHECKPOINTS,
    require_complete_outcome: bool = True,
) -> pd.DataFrame:
    """Convert ESConv into one leakage-safe row per dialogue and checkpoint."""
    checkpoints = _validate_checkpoints(checkpoints)
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8") as handle:
            conversations = json.load(handle)
    else:
        conversations = source

    rows: list[dict[str, Any]] = []
    for index, conversation in enumerate(conversations):
        seeker_survey = conversation.get("survey_score", {}).get("seeker", {})
        initial = _as_int(seeker_survey.get("initial_emotion_intensity"))
        final = _as_int(seeker_survey.get("final_emotion_intensity"))
        if require_complete_outcome and (initial is None or final is None):
            continue
        if initial is None or final is None:
            continue

        turns = [
            turn
            for turn in conversation.get("dialog", [])
            if _normalise_text(turn.get("content"))
        ]
        if not turns:
            continue

        conversation_id = f"esconv_{index:04d}"
        for checkpoint in checkpoints:
            observed = turns[: _prefix_size(len(turns), checkpoint)]
            all_text = " ".join(_normalise_text(turn.get("content")) for turn in observed)
            seeker_text = " ".join(
                _normalise_text(turn.get("content"))
                for turn in observed
                if turn.get("speaker") == "seeker"
            )
            supporter_text = " ".join(
                _normalise_text(turn.get("content"))
                for turn in observed
                if turn.get("speaker") == "supporter"
            )
            strategies = [
                _normalise_text(turn.get("annotation", {}).get("strategy"))
                for turn in observed
                if turn.get("speaker") == "supporter"
                and _normalise_text(turn.get("annotation", {}).get("strategy"))
            ]
            row = {
                "dataset": "esconv",
                "conversation_id": conversation_id,
                "checkpoint": checkpoint,
                "checkpoint_percent": int(round(checkpoint * 100)),
                "n_total_turns": len(turns),
                "n_observed_turns": len(observed),
                "observed_fraction_actual": len(observed) / len(turns),
                "text": all_text,
                "text_seeker": seeker_text,
                "text_supporter": supporter_text,
                "initial_intensity": initial,
                "final_intensity": final,
                "intensity_delta": final - initial,
                "intensity_change": change_label(initial, final),
                "emotion": _normalise_text(conversation.get("emotion_type")),
                "problem_type": _normalise_text(conversation.get("problem_type")),
            }
            row.update(_strategy_features(strategies))
            rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        strategy_columns = [
            column
            for column in frame.columns
            if column.startswith("strategy_count__") or column.startswith("strategy_rate__")
        ]
        frame[strategy_columns] = frame[strategy_columns].fillna(0)
    return frame


def _clean_emotion(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    emotion = re.sub(r"[^a-z]+", "", str(value).strip().lower())
    corrections = {
        "faer": "fear",
        "fera": "fear",
        "digust": "disgust",
        "sadnes": "sadness",
        "asadness": "sadness",
    }
    return corrections.get(emotion, emotion)


def meisd_turn_intensity(row: pd.Series) -> int:
    """Reduce MEISD multi-emotion annotations to an ordinal scalar in [0, 3]."""
    active: list[int] = []
    for emotion_column, intensity_column in (
        ("emotion", "intensity"),
        ("emotion2", "intensity2"),
        ("emotion3", "intensity3"),
    ):
        emotion = _clean_emotion(row.get(emotion_column))
        intensity = _as_int(row.get(intensity_column))
        if emotion and emotion != "neutral" and intensity in {1, 2, 3}:
            active.append(intensity)
    return max(active, default=0)


def build_meisd_checkpoints(
    source: str | Path | pd.DataFrame,
    checkpoints: Sequence[float] = DEFAULT_CHECKPOINTS,
) -> pd.DataFrame:
    """Create derived temporal targets from MEISD turn-level annotations."""
    checkpoints = _validate_checkpoints(checkpoints)
    frame = pd.read_csv(source) if isinstance(source, (str, Path)) else source.copy()
    required = {"TV Series", "dialog_ids", "uttr_ids", "Utterances"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MEISD is missing required columns: {sorted(missing)}")

    frame["_turn_intensity"] = frame.apply(meisd_turn_intensity, axis=1)
    frame = frame.sort_values(["TV Series", "dialog_ids", "uttr_ids"])

    rows: list[dict[str, Any]] = []
    for (series, dialog_id), dialogue in frame.groupby(
        ["TV Series", "dialog_ids"], sort=True, dropna=False
    ):
        dialogue = dialogue.reset_index(drop=True)
        texts = [_normalise_text(value) for value in dialogue["Utterances"]]
        valid_positions = [index for index, text in enumerate(texts) if text]
        if not valid_positions:
            continue
        dialogue = dialogue.iloc[valid_positions].reset_index(drop=True)
        texts = [texts[index] for index in valid_positions]
        initial = int(dialogue["_turn_intensity"].iloc[0])
        final = int(dialogue["_turn_intensity"].iloc[-1])
        conversation_id = f"meisd_{series}_{dialog_id}"

        for checkpoint in checkpoints:
            observed_count = _prefix_size(len(dialogue), checkpoint)
            observed = dialogue.iloc[:observed_count]
            rows.append(
                {
                    "dataset": "meisd",
                    "conversation_id": conversation_id,
                    "checkpoint": checkpoint,
                    "checkpoint_percent": int(round(checkpoint * 100)),
                    "n_total_turns": len(dialogue),
                    "n_observed_turns": observed_count,
                    "observed_fraction_actual": observed_count / len(dialogue),
                    "text": " ".join(texts[:observed_count]),
                    "initial_intensity": initial,
                    "current_intensity": int(observed["_turn_intensity"].iloc[-1]),
                    "final_intensity": final,
                    "intensity_delta": final - initial,
                    "intensity_change": change_label(initial, final),
                    "tv_series": str(series),
                }
            )
    return pd.DataFrame(rows)


def assert_checkpoint_integrity(
    frame: pd.DataFrame,
    checkpoints: Iterable[float] = DEFAULT_CHECKPOINTS,
) -> None:
    """Fail fast on duplicate, incomplete, non-monotonic, or out-of-range rows."""
    expected = set(float(value) for value in checkpoints)
    key_columns = ["dataset", "conversation_id", "checkpoint"]
    if frame.duplicated(key_columns).any():
        raise ValueError("Duplicate conversation/checkpoint rows detected.")
    for (_, conversation_id), group in frame.groupby(["dataset", "conversation_id"]):
        observed = set(group["checkpoint"].astype(float))
        if observed != expected:
            raise ValueError(
                f"{conversation_id} has checkpoints {sorted(observed)}, "
                f"expected {sorted(expected)}."
            )
        ordered = group.sort_values("checkpoint")
        counts = ordered["n_observed_turns"].tolist()
        if counts != sorted(counts):
            raise ValueError(f"{conversation_id} has non-monotonic observed turn counts.")
        if (ordered["n_observed_turns"] > ordered["n_total_turns"]).any():
            raise ValueError(f"{conversation_id} observes turns beyond the dialogue boundary.")

