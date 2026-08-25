from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import time
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .metrics import (
    classification_metrics,
    confusion_matrix_records,
    joint_prediction_diagnostics,
    multilabel_metrics,
    multilabel_per_label_metrics,
    ordinal_metrics,
    per_class_metrics,
)
from .models import (
    BinaryFocalLoss,
    EmotionConditionedOutcomeModel,
    OUTCOME_PAIRS,
    SoftSharingMTL,
    TemporalMultiModalModel,
    cumulative_ordinal_predictions,
    cumulative_ordinal_targets,
    temporal_multitask_loss,
)
from .neural_data import (
    OutcomeCeilingDataset,
    SourceMTLDataset,
    TemporalDataset,
    categorical_vocabulary,
    outcome_ceiling_collate,
    temporal_collate,
)
from .outcome_augmentation import validate_outcome_augmentation_frame
from .outcome_labels import get_outcome_label_scheme, relabel_outcome_frame
from .strategy import strategy_feature_columns, strategy_vocabulary
from .validation import validate_augmentation_artifacts, validate_source_training_frame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_precision(requested: str, device: torch.device) -> str:
    precision = requested.lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be either 'fp32' or 'bf16'.")
    if precision == "bf16" and device.type != "cuda":
        raise RuntimeError("BF16 training requires a CUDA/HIP GPU device.")
    return precision


def _autocast_context(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _class_weights(
    values: Iterable[int], classes: Sequence[int], device: torch.device
) -> torch.Tensor:
    array = np.asarray(list(values), dtype=int)
    counts = np.asarray([(array == label).sum() for label in classes], dtype=float)
    weights = len(array) / (len(classes) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float, device=device)


def _outcome_sampling_weights(
    frame: pd.DataFrame,
    strategy: str = "none",
    power: float = 0.5,
    max_ratio: float = 4.0,
) -> torch.Tensor | None:
    """Return deterministic sample weights for rare valid outcome pairs.

    The structured decoder couples final intensity and drop magnitude, so the
    balancing unit is their joint pair rather than either marginal label.
    ``power=0.5`` applies square-root inverse frequency and avoids duplicating
    the strongest class-weight correction already present in focal loss.
    """
    if strategy == "none":
        return None
    if strategy != "joint":
        raise ValueError("outcome_sampling_strategy must be 'none' or 'joint'.")
    if not 0.0 < power <= 1.0:
        raise ValueError("joint_sampling_power must be in (0, 1].")
    if max_ratio < 1.0:
        raise ValueError("joint_sampling_max_ratio must be at least 1.")
    required = {"final_intensity", "drop_magnitude"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing outcome sampling columns: {sorted(missing)}")

    pairs = list(
        zip(
            frame["final_intensity"].astype(int),
            frame["drop_magnitude"].astype(int),
        )
    )
    counts = pd.Series(pairs, dtype="object").value_counts().to_dict()
    raw = np.asarray([counts[pair] ** (-power) for pair in pairs], dtype=float)
    raw /= raw.min()
    raw = np.minimum(raw, float(max_ratio))
    raw /= raw.mean()
    return torch.tensor(raw, dtype=torch.double)


def _move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _append_progress(path: Path, event: dict[str, Any]) -> None:
    record = {"timestamp_unix": time.time(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        handle.flush()


def _source_loss(
    model: SoftSharingMTL,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    sentiment_loss: nn.Module,
    emotion_loss: nn.Module,
    task_weights: dict[str, float],
    soft_sharing_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    sent = sentiment_loss(outputs["sentiment"], batch["sentiment"])
    emotion = emotion_loss(outputs["emotion"], batch["emotion"])
    active = batch["intensity"] >= 0
    if active.any():
        intensity = nn.functional.cross_entropy(
            outputs["intensity"][active], batch["intensity"][active]
        )
    else:
        intensity = outputs["intensity"].sum() * 0.0
    sharing = model.soft_sharing_penalty(soft_sharing_lambda)
    total = (
        task_weights["sentiment"] * sent
        + task_weights["emotion"] * emotion
        + task_weights["intensity"] * intensity
        + sharing
    )
    return total, {
        "sentiment_loss": float(sent.detach().cpu()),
        "emotion_loss": float(emotion.detach().cpu()),
        "intensity_loss": float(intensity.detach().cpu()),
        "sharing_loss": float(sharing.detach().cpu()),
    }


def _run_source_epoch(
    model: SoftSharingMTL,
    loader: DataLoader,
    device: torch.device,
    sentiment_loss: nn.Module,
    emotion_loss: nn.Module,
    task_weights: dict[str, float],
    soft_sharing_lambda: float,
    emotion_names: Sequence[str],
    emotion_threshold: float,
    optimizer: AdamW | None = None,
    scheduler: Any | None = None,
    collect_predictions: bool = False,
    progress_description: str | None = None,
    progress_position: int = 0,
    precision: str = "fp32",
) -> tuple[dict[str, float], pd.DataFrame | None, dict[str, np.ndarray] | None]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    component_losses: dict[str, list[float]] = {
        "sentiment_loss": [],
        "emotion_loss": [],
        "intensity_loss": [],
        "sharing_loss": [],
    }
    sentiment_true: list[int] = []
    sentiment_pred: list[int] = []
    sentiment_probabilities: list[list[float]] = []
    emotion_true: list[list[int]] = []
    emotion_pred: list[list[int]] = []
    emotion_probabilities: list[list[float]] = []
    intensity_true_matrix: list[list[int]] = []
    intensity_pred_matrix: list[list[int]] = []
    intensity_probabilities: list[list[list[float]]] = []
    intensity_true: list[int] = []
    intensity_pred: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        iterator = tqdm(
            loader,
            desc=progress_description,
            unit="batch",
            position=progress_position,
            leave=False,
            dynamic_ncols=True,
            disable=progress_description is None,
        )
        for batch_index, raw_batch in enumerate(iterator, start=1):
            if collect_predictions:
                batch_size = len(raw_batch["conversation_id"])
                for index in range(batch_size):
                    metadata_rows.append(
                        {
                            "row_id": int(raw_batch["row_id"][index]),
                            "conversation_id": str(raw_batch["conversation_id"][index]),
                            "source_conversation_id": str(
                                raw_batch["source_conversation_id"][index]
                            ),
                            "augmented": bool(raw_batch["augmented"][index]),
                            "generation_valid": bool(
                                raw_batch["generation_valid"][index]
                            ),
                            "generation_quality": float(
                                raw_batch["generation_quality"][index]
                            ),
                        }
                    )
            batch = _move_tensors(raw_batch, device)
            with _autocast_context(device, precision):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss, loss_parts = _source_loss(
                    model,
                    outputs,
                    batch,
                    sentiment_loss,
                    emotion_loss,
                    task_weights,
                    soft_sharing_lambda,
                )
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            losses.append(float(loss.detach().cpu()))
            for name, value in loss_parts.items():
                component_losses[name].append(value)
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
                iterator.set_postfix(loss=f"{np.mean(losses):.4f}")
            sentiment_probability = torch.softmax(outputs["sentiment"], dim=-1)
            sentiment_true.extend(batch["sentiment"].detach().cpu().tolist())
            sentiment_pred.extend(
                outputs["sentiment"].argmax(dim=-1).detach().cpu().tolist()
            )
            sentiment_probabilities.extend(sentiment_probability.detach().cpu().tolist())
            emotion_probability = torch.sigmoid(outputs["emotion"])
            emotion_prediction = (emotion_probability >= emotion_threshold).to(torch.int)
            emotion_true.extend(batch["emotion"].detach().cpu().to(torch.int).tolist())
            emotion_pred.extend(emotion_prediction.detach().cpu().tolist())
            emotion_probabilities.extend(emotion_probability.detach().cpu().tolist())
            intensity_probability = torch.softmax(outputs["intensity"], dim=-1)
            intensity_prediction = outputs["intensity"].argmax(dim=-1)
            intensity_true_matrix.extend(batch["intensity"].detach().cpu().tolist())
            intensity_pred_matrix.extend(intensity_prediction.detach().cpu().tolist())
            intensity_probabilities.extend(intensity_probability.detach().cpu().tolist())
            active = batch["intensity"] >= 0
            intensity_true.extend(batch["intensity"][active].detach().cpu().tolist())
            intensity_pred.extend(
                intensity_prediction[active].detach().cpu().tolist()
            )
    metrics = {"loss": float(np.mean(losses))}
    metrics.update(
        {name: float(np.mean(values)) for name, values in component_losses.items()}
    )
    metrics.update(
        {
            f"sentiment_{key}": value
            for key, value in classification_metrics(
                sentiment_true, sentiment_pred, labels=(0, 1, 2)
            ).items()
        }
    )
    emotion_true_array = np.asarray(emotion_true, dtype=int)
    emotion_pred_array = np.asarray(emotion_pred, dtype=int)
    emotion_probability_array = np.asarray(emotion_probabilities, dtype=float)
    metrics.update(
        {
            f"emotion_{key}": value
            for key, value in multilabel_metrics(
                emotion_true_array, emotion_pred_array, emotion_probability_array
            ).items()
        }
    )
    if intensity_true:
        metrics.update(
            {
                f"intensity_{key}": value
                for key, value in classification_metrics(
                    intensity_true, intensity_pred, labels=(0, 1, 2)
                ).items()
            }
        )
        metrics.update(
            {
                f"intensity_{key}": value
                for key, value in ordinal_metrics(
                    intensity_true, intensity_pred, labels=(0, 1, 2)
                ).items()
            }
        )
    if not collect_predictions:
        return metrics, None, None

    predictions = pd.DataFrame(metadata_rows)
    predictions["sentiment_target"] = sentiment_true
    predictions["sentiment_prediction"] = sentiment_pred
    sentiment_names = ("negative", "neutral", "positive")
    for index, name in enumerate(sentiment_names):
        predictions[f"sentiment_probability__{name}"] = np.asarray(
            sentiment_probabilities
        )[:, index]
    for index, name in enumerate(emotion_names):
        predictions[f"emotion_target__{name}"] = emotion_true_array[:, index]
        predictions[f"emotion_prediction__{name}"] = emotion_pred_array[:, index]
        predictions[f"emotion_probability__{name}"] = emotion_probability_array[:, index]
        raw_intensity = np.asarray(intensity_true_matrix)[:, index]
        predictions[f"intensity_target__{name}"] = np.where(
            raw_intensity >= 0, raw_intensity + 1, np.nan
        )
        predictions[f"intensity_prediction__{name}"] = (
            np.asarray(intensity_pred_matrix)[:, index] + 1
        )
        for intensity_index in range(3):
            predictions[f"intensity_probability__{name}__{intensity_index + 1}"] = (
                np.asarray(intensity_probabilities)[:, index, intensity_index]
            )
    details = {
        "sentiment_true": np.asarray(sentiment_true),
        "sentiment_pred": np.asarray(sentiment_pred),
        "emotion_true": emotion_true_array,
        "emotion_pred": emotion_pred_array,
        "emotion_probability": emotion_probability_array,
        "intensity_true": np.asarray(intensity_true_matrix),
        "intensity_pred": np.asarray(intensity_pred_matrix),
    }
    return metrics, predictions, details


def _source_diagnostic_tables(
    details: dict[str, np.ndarray], emotion_names: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    sentiment_names = ("negative", "neutral", "positive")
    for row in per_class_metrics(
        details["sentiment_true"],
        details["sentiment_pred"],
        labels=(0, 1, 2),
        label_names=sentiment_names,
    ):
        metric_rows.append({"task": "sentiment", "emotion": "", **row})
    for row in confusion_matrix_records(
        details["sentiment_true"], details["sentiment_pred"], labels=(0, 1, 2)
    ):
        confusion_rows.append({"task": "sentiment", "emotion": "", **row})

    for index, emotion in enumerate(emotion_names):
        true_emotion = details["emotion_true"][:, index]
        pred_emotion = details["emotion_pred"][:, index]
        for row in per_class_metrics(
            true_emotion,
            pred_emotion,
            labels=(0, 1),
            label_names=("absent", "present"),
        ):
            metric_rows.append({"task": "emotion", "emotion": emotion, **row})
        for row in confusion_matrix_records(true_emotion, pred_emotion, labels=(0, 1)):
            confusion_rows.append({"task": "emotion", "emotion": emotion, **row})

        true_intensity = details["intensity_true"][:, index]
        active = true_intensity >= 0
        if active.any():
            pred_intensity = details["intensity_pred"][:, index][active]
            true_intensity_display = true_intensity[active] + 1
            pred_intensity_display = pred_intensity + 1
            intensity_summary = classification_metrics(
                true_intensity_display, pred_intensity_display, labels=(1, 2, 3)
            )
            intensity_summary.update(
                ordinal_metrics(
                    true_intensity_display, pred_intensity_display, labels=(1, 2, 3)
                )
            )
            metric_rows.append(
                {
                    "task": "emotion_intensity_summary",
                    "emotion": emotion,
                    "label": "all",
                    "label_name": "all",
                    "support": int(active.sum()),
                    **intensity_summary,
                }
            )
            for row in per_class_metrics(
                true_intensity_display,
                pred_intensity_display,
                labels=(1, 2, 3),
                label_names=("1", "2", "3"),
            ):
                metric_rows.append(
                    {"task": "emotion_intensity", "emotion": emotion, **row}
                )
            for row in confusion_matrix_records(
                true_intensity_display, pred_intensity_display, labels=(1, 2, 3)
            ):
                confusion_rows.append(
                    {"task": "emotion_intensity", "emotion": emotion, **row}
                )
    emotion_probability_rows = multilabel_per_label_metrics(
        details["emotion_true"],
        details["emotion_pred"],
        details["emotion_probability"],
        emotion_names,
    )
    for row in emotion_probability_rows:
        metric_rows.append(
            {
                "task": "emotion_summary",
                "emotion": row.pop("label_name"),
                "label": 1,
                **row,
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(confusion_rows)


def pretrain_source_mtl(
    source_path: str | Path,
    output_dir: str | Path,
    seed: int,
    config: dict[str, Any],
    progress_position: int = 0,
) -> dict[str, Any]:
    started_at = time.time()
    set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source_path)
    available_emotion_names = sorted(
        column.split("emotion__", 1)[1]
        for column in frame.columns
        if column.startswith("emotion__")
    )
    if not available_emotion_names:
        raise ValueError("Source data has no one-hot emotion columns.")
    configured_emotions = config.get("emotion_names")
    emotion_names = (
        [str(name) for name in configured_emotions]
        if configured_emotions
        else available_emotion_names
    )
    missing_emotions = sorted(set(emotion_names) - set(available_emotion_names))
    if missing_emotions:
        raise ValueError(
            f"Configured source emotions are unavailable: {missing_emotions}"
        )
    augmentation_report = validate_augmentation_artifacts(source_path, frame)
    frame, input_report = validate_source_training_frame(
        frame,
        emotion_names,
        exclude_invalid_generations=config.get(
            "exclude_invalid_generations", True
        ),
        drop_exact_train_duplicates=config.get(
            "drop_exact_train_duplicates", True
        ),
        drop_conflicting_train_texts=config.get(
            "drop_conflicting_train_texts", True
        ),
    )
    synthetic_ratio = config.get("max_synthetic_to_original_ratio")
    if synthetic_ratio is not None:
        synthetic_ratio = float(synthetic_ratio)
        if synthetic_ratio < 0:
            raise ValueError("max_synthetic_to_original_ratio cannot be negative.")
        augmented = frame.get(
            "augmented", pd.Series(False, index=frame.index, dtype=bool)
        )
        if augmented.dtype != bool:
            augmented = (
                augmented.astype(str).str.strip().str.lower().map(
                    {"true": True, "1": True, "false": False, "0": False}
                )
            ).fillna(False)
        train_mask = frame["split"] == "train"
        original_train = frame[train_mask & ~augmented]
        synthetic_train = frame[train_mask & augmented]
        maximum_synthetic = int(len(original_train) * synthetic_ratio)
        retained_synthetic = synthetic_train
        if len(synthetic_train) > maximum_synthetic:
            retained_synthetic = synthetic_train.sample(
                n=maximum_synthetic, random_state=seed
            )
            frame = pd.concat(
                [frame[~(train_mask & augmented)], retained_synthetic],
                ignore_index=True,
            )
        input_report["origin_balancing"] = {
            "max_synthetic_to_original_ratio": synthetic_ratio,
            "original_train_rows": len(original_train),
            "synthetic_train_rows_before": len(synthetic_train),
            "synthetic_train_rows_after": len(retained_synthetic),
            "total_train_rows_after": int((frame["split"] == "train").sum()),
        }
    input_report["augmentation_artifacts"] = augmentation_report
    input_report["available_emotion_names"] = available_emotion_names
    input_report["trained_emotion_names"] = emotion_names
    with (output / "source_input_report.json").open("w", encoding="utf-8") as handle:
        json.dump(input_report, handle, indent=2, sort_keys=True)
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Source data requires train and validation rows.")

    device = resolve_device(config.get("device", "auto"))
    precision = resolve_precision(config.get("precision", "fp32"), device)
    model_name = config["transformer_name"]
    local_files_only = bool(config.get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    train_dataset = SourceMTLDataset(
        train, tokenizer, emotion_names, max_length=config["max_length"]
    )
    validation_dataset = SourceMTLDataset(
        validation, tokenizer, emotion_names, max_length=config["max_length"]
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=loader_generator,
        num_workers=config.get("num_workers", 0),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
    )
    model = SoftSharingMTL(
        transformer_name=model_name,
        num_emotions=len(emotion_names),
        dropout=config["dropout"],
        local_files_only=local_files_only,
    ).to(device)
    sentiment_weights = _class_weights(
        train["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2}),
        (0, 1, 2),
        device,
    )
    emotion_frequency = np.asarray(
        [train[f"emotion__{emotion}"].mean() for emotion in emotion_names],
        dtype=np.float32,
    )
    emotion_alpha = torch.tensor(
        1.0 / np.maximum(emotion_frequency, 1e-6),
        dtype=torch.float,
        device=device,
    )
    emotion_alpha = emotion_alpha / emotion_alpha.mean()
    sentiment_loss = nn.CrossEntropyLoss(weight=sentiment_weights)
    emotion_loss = BinaryFocalLoss(
        gamma=config["focal_gamma"], alpha=emotion_alpha
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    total_steps = max(len(train_loader) * config["epochs"], 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    task_weights = config["task_weights"]
    emotion_threshold = float(config.get("emotion_threshold", 0.4))
    history: list[dict[str, Any]] = []
    selection_metric = config.get(
        "checkpoint_selection", "emotion_intensity_macro_f1"
    )
    if selection_metric not in {"loss", "emotion_intensity_macro_f1"}:
        raise ValueError(
            "source_mtl.checkpoint_selection must be 'loss' or "
            "'emotion_intensity_macro_f1'."
        )
    best_selection = (
        float("inf") if selection_metric == "loss" else -float("inf")
    )
    best_loss = float("inf")
    best_epoch = 0
    checkpoint_path = output / "source_transfer_checkpoint.pt"
    progress_path = output / "source_training_progress.jsonl"
    _append_progress(
        progress_path,
        {"event": "run_started", "run_type": "source_mtl", "seed": seed},
    )
    source_split_hash = _split_hash(frame)
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_type": "source_mtl",
                "seed": seed,
                "source_path": str(Path(source_path).resolve()),
                "source_sha256": _file_hash(source_path),
                "split_hash": source_split_hash,
                "emotion_names": emotion_names,
                "emotion_threshold": emotion_threshold,
                "precision": precision,
                "class_weights": {
                    "sentiment": sentiment_weights.detach().cpu().tolist(),
                    "emotion_alpha": emotion_alpha.detach().cpu().tolist(),
                },
                "input_report": input_report,
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                },
                "config": config,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    epoch_iterator = tqdm(
        range(1, config["epochs"] + 1),
        desc=f"Source seed {seed}",
        unit="epoch",
        position=progress_position,
        leave=True,
        dynamic_ncols=True,
    )
    for epoch in epoch_iterator:
        epoch_started = time.time()
        train_metrics, _, _ = _run_source_epoch(
            model,
            train_loader,
            device,
            sentiment_loss,
            emotion_loss,
            task_weights,
            config["soft_sharing_lambda"],
            emotion_names,
            emotion_threshold,
            optimizer,
            scheduler,
            progress_description=f"seed {seed} epoch {epoch} train",
            progress_position=progress_position + 1,
            precision=precision,
        )
        validation_metrics, validation_predictions, validation_details = _run_source_epoch(
            model,
            validation_loader,
            device,
            sentiment_loss,
            emotion_loss,
            task_weights,
            config["soft_sharing_lambda"],
            emotion_names,
            emotion_threshold,
            collect_predictions=True,
            progress_description=f"seed {seed} epoch {epoch} validation",
            progress_position=progress_position + 1,
            precision=precision,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "duration_seconds": time.time() - epoch_started,
            }
        )
        _append_progress(
            progress_path,
            {
                "event": "epoch_completed",
                "seed": seed,
                **history[-1],
            },
        )
        selection_value = (
            validation_metrics["loss"]
            if selection_metric == "loss"
            else (
                validation_metrics["emotion_f1_macro"]
                + validation_metrics["intensity_f1_macro"]
            )
            / 2.0
        )
        improved = (
            selection_value < best_selection
            if selection_metric == "loss"
            else selection_value > best_selection
        )
        if improved:
            best_selection = selection_value
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    **model.export_transfer_state(),
                    "seed": seed,
                    "emotion_names": emotion_names,
                    "source_config": config,
                    "best_epoch": best_epoch,
                    "best_validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
            assert validation_predictions is not None
            assert validation_details is not None
            validation_predictions["split"] = "validation"
            validation_predictions["seed"] = seed
            validation_predictions["best_epoch"] = best_epoch
            validation_predictions.to_csv(
                output / "source_validation_predictions.csv", index=False
            )
            with (output / "source_validation_metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(validation_metrics, handle, indent=2, sort_keys=True)
            pd.DataFrame(
                [{"seed": seed, "best_epoch": best_epoch, **validation_metrics}]
            ).to_csv(output / "source_validation_metrics.csv", index=False)
            per_class, confusion = _source_diagnostic_tables(
                validation_details, emotion_names
            )
            per_class.to_csv(
                output / "source_validation_per_class_metrics.csv", index=False
            )
            confusion.to_csv(
                output / "source_validation_confusion_matrices.csv", index=False
            )
            tqdm.write(
                f"[source seed {seed}] saved best checkpoint at epoch {best_epoch} "
                f"({selection_metric}={best_selection:.4f}; "
                f"validation loss={best_loss:.4f})"
            )
        epoch_iterator.set_postfix(
            val_loss=f"{validation_metrics['loss']:.4f}",
            emotion_f1=f"{validation_metrics['emotion_f1_macro']:.4f}",
            selection=f"{selection_value:.4f}",
            best_epoch=best_epoch or "-",
        )
        if epoch - best_epoch >= config["early_stopping_patience"]:
            break
    with (output / "source_training_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(history, handle, indent=2)
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "checkpoint_selection": selection_metric,
        "best_selection_score": best_selection,
        "checkpoint": str(checkpoint_path),
        "split_hash": source_split_hash,
        "emotion_names": emotion_names,
        "device": str(device),
        "precision": precision,
        "emotion_threshold": emotion_threshold,
        "input_report": input_report,
        "duration_seconds": time.time() - started_at,
    }
    with (output / "source_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _append_progress(progress_path, {"event": "run_completed", **summary})
    return summary


def _split_hash(frame: pd.DataFrame) -> str:
    content = (
        frame[["conversation_id", "split"]]
        .drop_duplicates()
        .sort_values("conversation_id")
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.sha256(content).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_temporal_epoch(
    model: TemporalMultiModalModel,
    loader: DataLoader,
    device: torch.device,
    final_weights: torch.Tensor,
    drop_weights: torch.Tensor,
    optimizer: AdamW | None = None,
    scheduler: Any | None = None,
    progress_description: str | None = None,
    progress_position: int = 0,
    label_scheme: str = "original4",
    precision: str = "fp32",
) -> tuple[dict[str, float], pd.DataFrame]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    component_losses: dict[str, list[float]] = {"final_loss": [], "drop_loss": []}
    identifiers: list[str] = []
    initial_values: list[int] = []
    final_raw_values: list[int] = []
    drop_raw_values: list[int] = []
    final_true: list[int] = []
    final_pred: list[int] = []
    drop_true: list[int] = []
    drop_pred: list[int] = []
    final_probabilities: list[list[float]] = []
    drop_probabilities: list[list[float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        iterator = tqdm(
            loader,
            desc=progress_description,
            unit="batch",
            position=progress_position,
            leave=False,
            dynamic_ncols=True,
            disable=progress_description is None,
        )
        for batch_index, raw_batch in enumerate(iterator, start=1):
            identifiers.extend(raw_batch["conversation_id"])
            initial_values.extend(raw_batch["initial_intensity"].tolist())
            final_raw_values.extend(raw_batch["final_target_raw"].tolist())
            drop_raw_values.extend(raw_batch["drop_target_raw"].tolist())
            batch = _move_tensors(raw_batch, device)
            with _autocast_context(device, precision):
                outputs = model(batch)
                loss, loss_parts = temporal_multitask_loss(
                    outputs,
                    batch["final_target"],
                    batch["drop_target"],
                    final_weights,
                    drop_weights,
                )
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            losses.append(float(loss.detach().cpu()))
            for name, value in loss_parts.items():
                component_losses[name].append(value)
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
                iterator.set_postfix(loss=f"{np.mean(losses):.4f}")
            final_probability = torch.softmax(outputs["final_intensity"], dim=-1)
            drop_probability = torch.softmax(outputs["drop_magnitude"], dim=-1)
            final_probabilities.extend(final_probability.detach().cpu().tolist())
            drop_probabilities.extend(drop_probability.detach().cpu().tolist())
            final_true.extend((batch["final_target"] + 1).detach().cpu().tolist())
            final_pred.extend(
                (outputs["final_intensity"].argmax(dim=-1) + 1)
                .detach()
                .cpu()
                .tolist()
            )
            drop_true.extend((batch["drop_target"] + 1).detach().cpu().tolist())
            drop_pred.extend(
                (outputs["drop_magnitude"].argmax(dim=-1) + 1)
                .detach()
                .cpu()
                .tolist()
            )
    scheme = get_outcome_label_scheme(label_scheme)
    final_metrics = classification_metrics(final_true, final_pred, labels=scheme.labels)
    final_metrics.update(ordinal_metrics(final_true, final_pred, labels=scheme.labels))
    drop_metrics = classification_metrics(drop_true, drop_pred, labels=scheme.labels)
    drop_metrics.update(ordinal_metrics(drop_true, drop_pred, labels=scheme.labels))
    metrics = {"loss": float(np.mean(losses))}
    metrics.update(
        {name: float(np.mean(values)) for name, values in component_losses.items()}
    )
    metrics.update({f"final_{key}": value for key, value in final_metrics.items()})
    metrics.update({f"drop_{key}": value for key, value in drop_metrics.items()})
    predictions = pd.DataFrame(
        {
            "conversation_id": identifiers,
            "initial_intensity": initial_values,
            "final_target_raw": final_raw_values,
            "drop_target_raw": drop_raw_values,
            "final_target": final_true,
            "final_prediction": final_pred,
            "drop_target": drop_true,
            "drop_prediction": drop_pred,
        }
    )
    for index in range(scheme.num_classes):
        predictions[f"final_probability_{index + 1}"] = np.asarray(
            final_probabilities
        )[:, index]
        predictions[f"drop_probability_{index + 1}"] = np.asarray(
            drop_probabilities
        )[:, index]
    joint_metrics, joint_arrays = joint_prediction_diagnostics(
        predictions["initial_intensity"],
        predictions["final_prediction"],
        predictions["drop_target"],
        predictions["drop_prediction"],
        label_scheme=scheme.name,
    )
    metrics.update(joint_metrics)
    for name, values in joint_arrays.items():
        predictions[name] = values
    return metrics, predictions


def train_temporal_model(
    checkpoints_path: str | Path,
    output_dir: str | Path,
    checkpoint: float,
    modality: str,
    seed: int,
    config: dict[str, Any],
    transfer_checkpoint_path: str | Path | None = None,
    use_initial_intensity: bool = False,
    progress_position: int = 0,
) -> dict[str, Any]:
    started_at = time.time()
    set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(checkpoints_path)
    scheme = get_outcome_label_scheme(config.get("label_scheme"))
    frame = relabel_outcome_frame(frame, scheme.name)
    subset = frame[np.isclose(frame["checkpoint"].astype(float), checkpoint)].copy()
    if subset.empty:
        raise ValueError(f"No rows found for checkpoint {checkpoint}.")
    train = subset[subset["split"] == "train"]
    validation = subset[subset["split"] == "validation"]
    test = subset[subset["split"] == "test"]
    if use_initial_intensity and (
        modality != "text_strategy" or transfer_checkpoint_path is None
    ):
        raise ValueError(
            "Initial-aware ablation requires transferred text_strategy modality."
        )
    initial_mean = float(train["initial_intensity"].mean())
    initial_std = float(train["initial_intensity"].std(ddof=0))
    if use_initial_intensity and initial_std <= 0:
        raise ValueError("Train initial intensity has zero variance.")
    transfer_checkpoint = None
    source_checkpoint_hash = None
    if transfer_checkpoint_path is not None:
        transfer_checkpoint = torch.load(
            transfer_checkpoint_path, map_location="cpu", weights_only=False
        )
        source_checkpoint_hash = _file_hash(transfer_checkpoint_path)
    device = resolve_device(config.get("device", "auto"))
    precision = resolve_precision(config.get("precision", "fp32"), device)
    local_files_only = bool(config.get("local_files_only", False))
    tokenizer = (
        AutoTokenizer.from_pretrained(
            config["transformer_name"], local_files_only=local_files_only
        )
        if "text" in modality
        else None
    )
    dataset_arguments = {
        "tokenizer": tokenizer,
        "modality": modality,
        "strategy_mode": "quantity_timing_order",
        "max_length": config["max_length"],
        "max_chunks": config["max_chunks"],
        "use_initial_intensity": use_initial_intensity,
        "initial_intensity_mean": initial_mean if use_initial_intensity else None,
        "initial_intensity_std": initial_std if use_initial_intensity else None,
        "cache_tokenization": config.get("cache_tokenization", True),
    }
    train_dataset = TemporalDataset(train, **dataset_arguments)
    validation_dataset = TemporalDataset(validation, **dataset_arguments)
    test_dataset = TemporalDataset(test, **dataset_arguments)
    collate = partial(
        temporal_collate,
        pad_token_id=tokenizer.pad_token_id if tokenizer is not None else 0,
    )
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            generator=generator,
            collate_fn=collate,
            num_workers=config.get("num_workers", 0),
        ),
        "validation": DataLoader(
            validation_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate,
            num_workers=config.get("num_workers", 0),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate,
            num_workers=config.get("num_workers", 0),
        ),
    }
    model = TemporalMultiModalModel(
        modality=modality,
        transformer_name=config["transformer_name"],
        transfer_checkpoint=transfer_checkpoint,
        strategy_vocabulary_size=len(strategy_vocabulary()),
        strategy_numeric_size=len(
            strategy_feature_columns("quantity_timing_order")
        ),
        strategy_hidden_size=config["strategy_hidden_size"],
        dropout=config["dropout"],
        max_chunks=config["max_chunks"],
        use_initial_intensity=use_initial_intensity,
        num_outcome_classes=scheme.num_classes,
        local_files_only=local_files_only,
    ).to(device)
    model.set_text_encoder_trainable_layers(
        config.get("max_trainable_text_encoder_layers")
    )
    run_config = {
        "run_type": "temporal",
        "checkpoint": checkpoint,
        "modality": modality,
        "transfer": transfer_checkpoint is not None,
        "seed": seed,
        "use_initial_intensity": use_initial_intensity,
        "label_scheme": scheme.to_manifest(),
        "precision": precision,
        "initial_intensity_standardization": (
            {"mean": initial_mean, "std": initial_std, "source": "train_only"}
            if use_initial_intensity
            else None
        ),
        "checkpoints_path": str(Path(checkpoints_path).resolve()),
        "checkpoints_sha256": _file_hash(checkpoints_path),
        "split_hash": _split_hash(subset),
        "source_checkpoint": (
            str(Path(transfer_checkpoint_path).resolve())
            if transfer_checkpoint_path is not None
            else None
        ),
        "source_checkpoint_sha256": source_checkpoint_hash,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": config,
    }
    final_weights = _class_weights(train["final_intensity"], scheme.labels, device)
    drop_weights = _class_weights(train["drop_magnitude"], scheme.labels, device)
    run_config["class_weights"] = {
        "final_intensity": final_weights.detach().cpu().tolist(),
        "drop_magnitude": drop_weights.detach().cpu().tolist(),
    }
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    total_steps = max(len(loaders["train"]) * config["epochs"], 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    best_score = -float("inf")
    best_epoch = 0
    model_path = output / "best_model.pt"
    progress_path = output / "training_progress.jsonl"
    _append_progress(
        progress_path,
        {
            "event": "run_started",
            "checkpoint": checkpoint,
            "modality": modality,
            "transfer": transfer_checkpoint is not None,
            "use_initial_intensity": use_initial_intensity,
            "seed": seed,
        },
    )
    history: list[dict[str, Any]] = []
    run_label = (
        f"Target {round(checkpoint * 100)}% {modality} seed {seed}"
        + (" +initial" if use_initial_intensity else "")
    )
    epoch_iterator = tqdm(
        range(1, config["epochs"] + 1),
        desc=run_label,
        unit="epoch",
        position=progress_position,
        leave=True,
        dynamic_ncols=True,
    )
    for epoch in epoch_iterator:
        epoch_started = time.time()
        train_metrics, _ = _run_temporal_epoch(
            model,
            loaders["train"],
            device,
            final_weights,
            drop_weights,
            optimizer,
            scheduler,
            progress_description=f"epoch {epoch} train",
            progress_position=progress_position + 1,
            label_scheme=scheme.name,
            precision=precision,
        )
        validation_metrics, _ = _run_temporal_epoch(
            model,
            loaders["validation"],
            device,
            final_weights,
            drop_weights,
            progress_description=f"epoch {epoch} validation",
            progress_position=progress_position + 1,
            label_scheme=scheme.name,
            precision=precision,
        )
        score = (
            validation_metrics["final_f1_macro"]
            + validation_metrics["drop_f1_macro"]
        ) / 2.0
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "selection_score": score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "duration_seconds": time.time() - epoch_started,
            }
        )
        _append_progress(
            progress_path,
            {"event": "epoch_completed", **history[-1]},
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            tqdm.write(
                f"[{run_label}] saved best checkpoint at epoch {best_epoch} "
                f"(selection macro-F1={best_score:.4f})"
            )
        epoch_iterator.set_postfix(
            val_f1=f"{score:.4f}",
            val_loss=f"{validation_metrics['loss']:.4f}",
            best_epoch=best_epoch or "-",
        )
        if epoch - best_epoch >= config["early_stopping_patience"]:
            break
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    metric_rows: list[dict[str, Any]] = []
    evaluate_test = bool(config.get("evaluate_test", True))
    evaluation_splits = ("validation", "test") if evaluate_test else ("validation",)
    for split_name in evaluation_splits:
        metrics, predictions = _run_temporal_epoch(
            model,
            loaders[split_name],
            device,
            final_weights,
            drop_weights,
            progress_description=f"{run_label} final {split_name}",
            progress_position=progress_position + 1,
            label_scheme=scheme.name,
            precision=precision,
        )
        predictions["split"] = split_name
        predictions["checkpoint"] = checkpoint
        predictions["checkpoint_percent"] = round(checkpoint * 100)
        predictions["modality"] = modality
        predictions["transfer"] = transfer_checkpoint is not None
        predictions["use_initial_intensity"] = use_initial_intensity
        predictions["label_scheme"] = scheme.name
        predictions["seed"] = seed
        predictions.to_csv(output / f"{split_name}_predictions.csv", index=False)
        diagnostic_rows: list[dict[str, Any]] = []
        confusion_rows: list[dict[str, Any]] = []
        for target, truth, predicted in (
            ("final_intensity", predictions["final_target"], predictions["final_prediction"]),
            ("drop_magnitude", predictions["drop_target"], predictions["drop_prediction"]),
        ):
            diagnostic_rows.extend(
                {"target": target, **row}
                for row in per_class_metrics(
                    truth,
                    predicted,
                    labels=scheme.labels,
                    label_names=scheme.label_names,
                )
            )
            confusion_rows.extend(
                {"target": target, **row}
                for row in confusion_matrix_records(
                    truth, predicted, labels=scheme.labels
                )
            )
        pd.DataFrame(diagnostic_rows).to_csv(
            output / f"{split_name}_per_class_metrics.csv", index=False
        )
        pd.DataFrame(confusion_rows).to_csv(
            output / f"{split_name}_confusion_matrices.csv", index=False
        )
        for target, prefix in (
            ("final_intensity", "final"),
            ("drop_magnitude", "drop"),
        ):
            metric_rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_percent": round(checkpoint * 100),
                    "split": split_name,
                    "target": target,
                    "modality": modality,
                    "transfer": transfer_checkpoint is not None,
                    "use_initial_intensity": use_initial_intensity,
                    "label_scheme": scheme.name,
                    "seed": seed,
                    **{
                        key.removeprefix(f"{prefix}_"): value
                        for key, value in metrics.items()
                        if key.startswith(f"{prefix}_")
                    },
                    "joint_consistency_rate": metrics["joint_consistency_rate"],
                    "derived_drop_accuracy": metrics["derived_drop_accuracy"],
                    "derived_drop_mae": metrics["derived_drop_mae"],
                    "invalid_derived_drop_rate": metrics[
                        "invalid_derived_drop_rate"
                    ],
                }
            )
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / "metrics.csv", index=False)
    with (output / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    summary = {
        "checkpoint": checkpoint,
        "modality": modality,
        "transfer": transfer_checkpoint is not None,
        "seed": seed,
        "use_initial_intensity": use_initial_intensity,
        "label_scheme": scheme.to_manifest(),
        "initial_intensity_standardization": (
            {"mean": initial_mean, "std": initial_std, "source": "train_only"}
            if use_initial_intensity
            else None
        ),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "split_hash": _split_hash(subset),
        "source_checkpoint": (
            str(Path(transfer_checkpoint_path).resolve())
            if transfer_checkpoint_path is not None
            else None
        ),
        "source_checkpoint_sha256": source_checkpoint_hash,
        "device": str(device),
        "precision": precision,
        "test_evaluated": evaluate_test,
        "duration_seconds": time.time() - started_at,
        "best_model_retained": bool(config.get("retain_best_model", False)),
    }
    if not summary["best_model_retained"]:
        # Predictions, metrics, diagnostics, and history are the durable research
        # outputs. Retaining every target state dict adds roughly 100 GiB across
        # the full experiment matrix without being used by the reporting stage.
        model_path.unlink(missing_ok=True)
    with (output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _append_progress(progress_path, {"event": "run_completed", **summary})
    return summary


def _ordinal_class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits)
    probabilities = torch.cat(
        [
            1.0 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ],
        dim=1,
    )
    return probabilities.clamp_min(0.0) / probabilities.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)


def _multiclass_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    """Class-balanced focal loss for one-indexed outcome labels."""
    zero_indexed = targets.long() - 1
    log_probabilities = nn.functional.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    target_log_probability = log_probabilities.gather(
        1, zero_indexed.unsqueeze(1)
    ).squeeze(1)
    target_probability = probabilities.gather(
        1, zero_indexed.unsqueeze(1)
    ).squeeze(1)
    loss = -(1.0 - target_probability).pow(gamma) * target_log_probability
    if class_weights is not None:
        loss = loss * class_weights[zero_indexed]
    return loss.mean()


def _structured_outcome_predictions(
    final_logits: torch.Tensor,
    drop_logits: torch.Tensor,
    initial_intensity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the highest-scoring valid pair satisfying final + drop = initial."""
    final_scores = nn.functional.log_softmax(final_logits, dim=-1)
    drop_scores = nn.functional.log_softmax(drop_logits, dim=-1)
    joint_scores = final_scores.unsqueeze(2) + drop_scores.unsqueeze(1)
    classes = torch.arange(1, 5, device=joint_scores.device)
    valid = (
        classes.view(1, 4, 1) + classes.view(1, 1, 4)
        == initial_intensity.view(-1, 1, 1)
    )
    joint_scores = joint_scores.masked_fill(~valid, -torch.inf)
    flat = joint_scores.flatten(1).argmax(dim=1)
    predicted_final = flat.div(4, rounding_mode="floor") + 1
    predicted_drop = flat.remainder(4) + 1
    return predicted_final, predicted_drop


def _outcome_pair_targets(
    final_targets: torch.Tensor, drop_targets: torch.Tensor
) -> torch.Tensor:
    """Encode valid one-indexed final/drop pairs as one-indexed joint classes."""
    pairs = torch.tensor(OUTCOME_PAIRS, device=final_targets.device)
    observed = torch.stack([final_targets, drop_targets], dim=1)
    matches = (observed.unsqueeze(1) == pairs.unsqueeze(0)).all(dim=2)
    if not bool(matches.any(dim=1).all()):
        invalid = observed[~matches.any(dim=1)].detach().cpu().tolist()
        raise ValueError(f"Unsupported outcome pairs: {invalid}")
    return matches.to(torch.long).argmax(dim=1) + 1


def _joint_pair_predictions(
    joint_logits: torch.Tensor,
    initial_intensity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the direct joint head after masking pairs invalid for the initial level."""
    pairs = torch.tensor(
        OUTCOME_PAIRS, device=joint_logits.device, dtype=initial_intensity.dtype
    )
    valid = pairs.sum(dim=1).unsqueeze(0) == initial_intensity.unsqueeze(1)
    if not bool(valid.any(dim=1).all()):
        invalid = initial_intensity[~valid.any(dim=1)].detach().cpu().tolist()
        raise ValueError(f"Initial intensities have no valid outcome pair: {invalid}")
    masked_logits = joint_logits.masked_fill(~valid, -torch.inf)
    probabilities = masked_logits.softmax(dim=-1)
    predicted_pair = probabilities.argmax(dim=1)
    return (
        pairs[predicted_pair, 0],
        pairs[predicted_pair, 1],
        probabilities,
    )


def _joint_marginal_probabilities(
    joint_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalise the ten valid pair probabilities into four-class outcomes."""
    pairs = torch.tensor(OUTCOME_PAIRS, device=joint_probabilities.device)
    batch_size = joint_probabilities.shape[0]
    final = joint_probabilities.new_zeros((batch_size, 4))
    drop = joint_probabilities.new_zeros((batch_size, 4))
    final.scatter_add_(
        1,
        (pairs[:, 0] - 1).unsqueeze(0).expand(batch_size, -1),
        joint_probabilities,
    )
    drop.scatter_add_(
        1,
        (pairs[:, 1] - 1).unsqueeze(0).expand(batch_size, -1),
        joint_probabilities,
    )
    return final, drop


def _assert_finite_tensors(
    tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    stage: str,
    batch_index: int,
    conversation_ids: Sequence[str],
) -> None:
    """Fail before non-finite values can poison an outcome training run."""
    for name, tensor in tensors:
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        finite = torch.isfinite(tensor)
        if bool(finite.all()):
            continue
        detached = tensor.detach()
        finite_values = detached[finite]
        finite_min = (
            float(finite_values.min().cpu()) if finite_values.numel() else None
        )
        finite_max = (
            float(finite_values.max().cpu()) if finite_values.numel() else None
        )
        raise FloatingPointError(
            "Non-finite tensor in outcome training: "
            f"stage={stage}, batch={batch_index}, tensor={name}, "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"nan_count={int(torch.isnan(detached).sum().cpu())}, "
            f"inf_count={int(torch.isinf(detached).sum().cpu())}, "
            f"finite_min={finite_min}, finite_max={finite_max}, "
            f"conversation_ids={list(conversation_ids)}"
        )


def _run_outcome_ceiling_epoch(
    model: EmotionConditionedOutcomeModel,
    loader: DataLoader,
    device: torch.device,
    threshold_weights: torch.Tensor,
    auxiliary_regression_weight: float = 0.0,
    optimizer: AdamW | None = None,
    scheduler: Any | None = None,
    progress_description: str | None = None,
    progress_position: int = 0,
    final_class_weights: torch.Tensor | None = None,
    drop_class_weights: torch.Tensor | None = None,
    joint_class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    joint_classification_weight: float = 1.0,
    marginal_auxiliary_weight: float = 0.3,
    direct_classification_weight: float = 1.0,
    ordinal_auxiliary_weight: float = 0.2,
    consistency_weight: float = 0.1,
    precision: str = "fp32",
) -> tuple[dict[str, float], pd.DataFrame]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    identifiers: list[str] = []
    initial_values: list[int] = []
    final_true: list[int] = []
    final_pred: list[int] = []
    drop_true: list[int] = []
    drop_pred: list[int] = []
    final_probabilities: list[list[float]] = []
    drop_regression_values: list[float] = []
    ordinal_losses: list[float] = []
    auxiliary_losses: list[float] = []
    final_classification_losses: list[float] = []
    drop_classification_losses: list[float] = []
    consistency_losses: list[float] = []
    joint_classification_losses: list[float] = []
    drop_probabilities: list[list[float]] = []
    joint_probabilities: list[list[float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context, _autocast_context(device, precision):
        iterator = tqdm(
            loader,
            desc=progress_description,
            unit="batch",
            position=progress_position,
            leave=False,
            dynamic_ncols=True,
            disable=progress_description is None,
        )
        for batch_index, raw_batch in enumerate(iterator, start=1):
            identifiers.extend(raw_batch["conversation_id"])
            batch_conversation_ids = [str(value) for value in raw_batch["conversation_id"]]
            batch = _move_tensors(raw_batch, device)
            _assert_finite_tensors(
                (
                    (name, value)
                    for name, value in batch.items()
                    if isinstance(value, torch.Tensor)
                ),
                stage="batch_inputs",
                batch_index=batch_index,
                conversation_ids=batch_conversation_ids,
            )
            outputs = model(batch)
            _assert_finite_tensors(
                outputs.items(),
                stage="model_outputs",
                batch_index=batch_index,
                conversation_ids=batch_conversation_ids,
            )
            logits = outputs["final_ordinal_logits"]
            targets = cumulative_ordinal_targets(batch["final_target"])
            ordinal_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=threshold_weights
            )
            direct_outputs = "final_logits" in outputs and "drop_logits" in outputs
            final_classification_loss = logits.new_zeros(())
            drop_classification_loss = logits.new_zeros(())
            joint_classification_loss = logits.new_zeros(())
            consistency_loss = logits.new_zeros(())
            if direct_outputs:
                final_classification_loss = _multiclass_focal_loss(
                    outputs["final_logits"],
                    batch["final_target"],
                    final_class_weights,
                    focal_gamma,
                )
                drop_classification_loss = _multiclass_focal_loss(
                    outputs["drop_logits"],
                    batch["drop_target"],
                    drop_class_weights,
                    focal_gamma,
                )
                class_values = torch.arange(
                    1, 5, device=logits.device, dtype=logits.dtype
                )
                expected_final = (
                    outputs["final_logits"].softmax(dim=-1) * class_values
                ).sum(dim=-1)
                expected_drop = (
                    outputs["drop_logits"].softmax(dim=-1) * class_values
                ).sum(dim=-1)
                consistency_loss = nn.functional.smooth_l1_loss(
                    expected_final + expected_drop,
                    batch["initial_intensity"].to(logits.dtype),
                )
                marginal_loss = final_classification_loss + drop_classification_loss
                if "joint_pair_logits" in outputs:
                    joint_targets = _outcome_pair_targets(
                        batch["final_target"], batch["drop_target"]
                    )
                    joint_classification_loss = _multiclass_focal_loss(
                        outputs["joint_pair_logits"],
                        joint_targets,
                        joint_class_weights,
                        focal_gamma,
                    )
                    loss = (
                        joint_classification_weight * joint_classification_loss
                        + marginal_auxiliary_weight * marginal_loss
                    )
                else:
                    loss = direct_classification_weight * marginal_loss
                loss = loss + ordinal_auxiliary_weight * ordinal_loss
                loss = loss + consistency_weight * consistency_loss
            else:
                loss = ordinal_loss
            auxiliary_loss = logits.new_zeros(())
            if auxiliary_regression_weight > 0:
                if "drop_regression" not in outputs:
                    raise ValueError(
                        "Auxiliary regression weight requires a regression head."
                    )
                auxiliary_loss = nn.functional.smooth_l1_loss(
                    outputs["drop_regression"], batch["drop_target"].float()
                )
                loss = loss + auxiliary_regression_weight * auxiliary_loss
            _assert_finite_tensors(
                (
                    ("loss", loss),
                    ("ordinal_loss", ordinal_loss),
                    ("final_classification_loss", final_classification_loss),
                    ("drop_classification_loss", drop_classification_loss),
                    ("joint_classification_loss", joint_classification_loss),
                    ("consistency_loss", consistency_loss),
                    ("auxiliary_regression_loss", auxiliary_loss),
                ),
                stage="loss_components",
                batch_index=batch_index,
                conversation_ids=batch_conversation_ids,
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not bool(torch.isfinite(gradient_norm)):
                    # The aggregate norm is a single inexpensive GPU reduction.
                    # Only scan individual gradients after it detects a failure.
                    _assert_finite_tensors(
                        (
                            (f"{name}.grad", parameter.grad)
                            for name, parameter in model.named_parameters()
                            if parameter.grad is not None
                        ),
                        stage="gradients_after_backward",
                        batch_index=batch_index,
                        conversation_ids=batch_conversation_ids,
                    )
                    _assert_finite_tensors(
                        (("gradient_norm", gradient_norm),),
                        stage="gradient_clipping",
                        batch_index=batch_index,
                        conversation_ids=batch_conversation_ids,
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            losses.append(float(loss.detach().cpu()))
            ordinal_losses.append(float(ordinal_loss.detach().cpu()))
            auxiliary_losses.append(float(auxiliary_loss.detach().cpu()))
            final_classification_losses.append(
                float(final_classification_loss.detach().cpu())
            )
            drop_classification_losses.append(
                float(drop_classification_loss.detach().cpu())
            )
            consistency_losses.append(float(consistency_loss.detach().cpu()))
            joint_classification_losses.append(
                float(joint_classification_loss.detach().cpu())
            )
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
                iterator.set_postfix(loss=f"{np.mean(losses):.4f}")
            if direct_outputs:
                if "joint_pair_logits" in outputs:
                    predicted_final, predicted_drop, pair_probability = (
                        _joint_pair_predictions(
                            outputs["joint_pair_logits"],
                            batch["initial_intensity"],
                        )
                    )
                    final_probability, drop_probability = (
                        _joint_marginal_probabilities(pair_probability)
                    )
                    joint_probabilities.extend(
                        pair_probability.detach().cpu().tolist()
                    )
                    final_probabilities.extend(
                        final_probability.detach().cpu().tolist()
                    )
                    drop_probabilities.extend(
                        drop_probability.detach().cpu().tolist()
                    )
                else:
                    predicted_final, predicted_drop = _structured_outcome_predictions(
                        outputs["final_logits"],
                        outputs["drop_logits"],
                        batch["initial_intensity"],
                    )
                    final_probabilities.extend(
                        outputs["final_logits"].softmax(dim=-1).detach().cpu().tolist()
                    )
                    drop_probabilities.extend(
                        outputs["drop_logits"].softmax(dim=-1).detach().cpu().tolist()
                    )
            else:
                predicted_final = cumulative_ordinal_predictions(
                    logits, batch["initial_intensity"]
                )
                predicted_drop = batch["initial_intensity"] - predicted_final
                final_probabilities.extend(
                    _ordinal_class_probabilities(logits).detach().cpu().tolist()
                )
            initial_values.extend(batch["initial_intensity"].detach().cpu().tolist())
            final_true.extend(batch["final_target"].detach().cpu().tolist())
            final_pred.extend(predicted_final.detach().cpu().tolist())
            drop_true.extend(batch["drop_target"].detach().cpu().tolist())
            drop_pred.extend(predicted_drop.detach().cpu().tolist())
            if "drop_regression" in outputs:
                drop_regression_values.extend(
                    outputs["drop_regression"].detach().cpu().tolist()
                )

    final_metrics = classification_metrics(final_true, final_pred, labels=(1, 2, 3, 4))
    final_metrics.update(ordinal_metrics(final_true, final_pred, labels=(1, 2, 3, 4)))
    drop_metrics = classification_metrics(drop_true, drop_pred, labels=(1, 2, 3, 4))
    drop_metrics.update(ordinal_metrics(drop_true, drop_pred, labels=(1, 2, 3, 4)))
    metrics = {
        "loss": float(np.mean(losses)),
        "ordinal_loss": float(np.mean(ordinal_losses)),
        "auxiliary_regression_loss": float(np.mean(auxiliary_losses)),
        "final_classification_loss": float(np.mean(final_classification_losses)),
        "drop_classification_loss": float(np.mean(drop_classification_losses)),
        "joint_classification_loss": float(np.mean(joint_classification_losses)),
        "consistency_loss": float(np.mean(consistency_losses)),
    }
    metrics.update({f"final_{key}": value for key, value in final_metrics.items()})
    metrics.update({f"drop_{key}": value for key, value in drop_metrics.items()})
    predictions = pd.DataFrame(
        {
            "conversation_id": identifiers,
            "initial_intensity": initial_values,
            "final_target": final_true,
            "final_prediction": final_pred,
            "drop_target": drop_true,
            "drop_prediction": drop_pred,
        }
    )
    for index in range(4):
        predictions[f"final_probability_{index + 1}"] = np.asarray(
            final_probabilities
        )[:, index]
        if drop_probabilities:
            predictions[f"drop_probability_{index + 1}"] = np.asarray(
                drop_probabilities
            )[:, index]
    for index, (final_value, drop_value) in enumerate(OUTCOME_PAIRS):
        if joint_probabilities:
            predictions[
                f"joint_probability_final_{final_value}_drop_{drop_value}"
            ] = np.asarray(joint_probabilities)[:, index]
    if drop_regression_values:
        predictions["drop_regression"] = drop_regression_values
        metrics["drop_regression_mae"] = float(
            np.mean(
                np.abs(
                    predictions["drop_regression"].to_numpy()
                    - predictions["drop_target"].to_numpy()
                )
            )
        )
    metrics["joint_consistency_rate"] = float(
        (
            predictions["final_prediction"] + predictions["drop_prediction"]
            == predictions["initial_intensity"]
        ).mean()
    )
    metrics["invalid_derived_drop_rate"] = float(
        (~predictions["drop_prediction"].between(1, 4)).mean()
    )
    return metrics, predictions


def _outcome_parameter_groups(
    model: EmotionConditionedOutcomeModel,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build non-overlapping encoder/head groups for discriminative tuning."""
    encoder_learning_rate = float(
        config.get("encoder_learning_rate", config["learning_rate"])
    )
    head_learning_rate = float(
        config.get("head_learning_rate", config["learning_rate"])
    )
    if encoder_learning_rate <= 0 or head_learning_rate <= 0:
        raise ValueError("Outcome learning rates must be positive.")
    encoder_parameters = list(model.text_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    return [
        {"params": encoder_parameters, "lr": encoder_learning_rate},
        {"params": head_parameters, "lr": head_learning_rate},
    ]


def _outcome_trainable_layers(
    epoch: int,
    config: dict[str, Any],
) -> int | None:
    """Return 0 (frozen), top-N, or None (fully trainable) for an epoch."""
    freeze_epochs = int(config.get("freeze_text_encoder_epochs", 1))
    if freeze_epochs < 0:
        raise ValueError("freeze_text_encoder_epochs cannot be negative.")
    if epoch <= freeze_epochs:
        return 0
    stages = [int(value) for value in config.get("gradual_unfreeze_layers", [])]
    if any(value <= 0 for value in stages):
        raise ValueError("gradual_unfreeze_layers must contain positive values.")
    stage_index = epoch - freeze_epochs - 1
    configured_layers = stages[stage_index] if stage_index < len(stages) else None
    maximum_layers = config.get("max_trainable_text_encoder_layers")
    if maximum_layers is None:
        return configured_layers
    maximum_layers = int(maximum_layers)
    if maximum_layers <= 0:
        raise ValueError("max_trainable_text_encoder_layers must be positive.")
    if configured_layers is None:
        return maximum_layers
    return min(configured_layers, maximum_layers)


def train_outcome_ceiling_model(
    checkpoints_path: str | Path,
    output_dir: str | Path,
    seed: int,
    config: dict[str, Any],
    transfer_checkpoint_path: str | Path | None = None,
    progress_position: int = 0,
) -> dict[str, Any]:
    """Train the role-aware, metadata-conditioned 100% context ceiling."""
    started_at = time.time()
    set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(checkpoints_path)
    outcome_augmentation_report = validate_outcome_augmentation_frame(frame)
    subset = frame[np.isclose(frame["checkpoint"].astype(float), 1.0)].copy()
    if subset.empty:
        raise ValueError("The outcome ceiling experiment requires checkpoint 1.0.")
    required = {"text_role_turns", "emotion_family", "problem_type"}
    missing = required - set(subset.columns)
    if missing:
        raise ValueError(
            "Preprocess ESConv again before training the outcome ceiling model; "
            f"missing columns: {sorted(missing)}"
        )
    train = subset[subset["split"] == "train"].copy()
    validation = subset[subset["split"] == "validation"].copy()
    test = subset[subset["split"] == "test"].copy()
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("Outcome ceiling training requires train/validation/test rows.")

    emotion_vocabulary = categorical_vocabulary(train["emotion_family"].tolist())
    problem_vocabulary = categorical_vocabulary(train["problem_type"].tolist())
    device = resolve_device(config.get("device", "auto"))
    precision = resolve_precision(config.get("precision", "fp32"), device)
    tokenizer = AutoTokenizer.from_pretrained(config["transformer_name"])
    dataset_arguments = {
        "tokenizer": tokenizer,
        "emotion_vocabulary": emotion_vocabulary,
        "problem_vocabulary": problem_vocabulary,
        "strategy_mode": config.get("strategy_mode", "quantity_timing_order"),
        "max_length": config["max_length"],
        "max_chunks": config["max_chunks"],
        "cache_tokenization": config.get("cache_tokenization", True),
    }
    datasets = {
        "train": OutcomeCeilingDataset(train, **dataset_arguments),
        "validation": OutcomeCeilingDataset(validation, **dataset_arguments),
        "test": OutcomeCeilingDataset(test, **dataset_arguments),
    }
    collate = partial(
        outcome_ceiling_collate, pad_token_id=tokenizer.pad_token_id or 0
    )
    generator = torch.Generator().manual_seed(seed)
    sampling_strategy = str(config.get("outcome_sampling_strategy", "none"))
    sampling_weights = _outcome_sampling_weights(
        train,
        strategy=sampling_strategy,
        power=float(config.get("joint_sampling_power", 0.5)),
        max_ratio=float(config.get("joint_sampling_max_ratio", 4.0)),
    )
    train_sampler = (
        WeightedRandomSampler(
            sampling_weights,
            num_samples=len(sampling_weights),
            replacement=True,
            generator=generator,
        )
        if sampling_weights is not None
        else None
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config["batch_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            generator=generator if train_sampler is None else None,
            collate_fn=collate,
            num_workers=config.get("num_workers", 0),
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate,
            num_workers=config.get("num_workers", 0),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate,
            num_workers=config.get("num_workers", 0),
        ),
    }
    transfer_checkpoint = None
    source_checkpoint_hash = None
    if transfer_checkpoint_path is not None:
        transfer_checkpoint = torch.load(
            transfer_checkpoint_path, map_location="cpu", weights_only=False
        )
        source_checkpoint_hash = _file_hash(transfer_checkpoint_path)
    model = EmotionConditionedOutcomeModel(
        transformer_name=config["transformer_name"],
        transfer_checkpoint=transfer_checkpoint,
        emotion_vocabulary_size=len(emotion_vocabulary),
        problem_vocabulary_size=len(problem_vocabulary),
        dropout=config["dropout"],
        max_chunks=config["max_chunks"],
        metadata_size=config.get("metadata_size", 64),
        strategy_vocabulary_size=len(strategy_vocabulary()),
        strategy_numeric_size=len(
            strategy_feature_columns(
                config.get("strategy_mode", "quantity_timing_order")
            )
        ),
        strategy_hidden_size=config.get("strategy_hidden_size", 256),
        use_speaker_features=config.get("use_speaker_features", False),
        use_trajectory=config.get("use_trajectory", False),
        use_strategy=config.get("use_strategy", False),
        use_auxiliary_regression=config.get("use_auxiliary_regression", True),
        use_joint_pair_head=config.get("use_joint_pair_head", True),
        use_emotion_conditioned_heads=config.get(
            "use_emotion_conditioned_heads", True
        ),
    ).to(device)
    auxiliary_regression_weight = float(
        config.get("auxiliary_regression_weight", 0.0)
    )
    train_targets = cumulative_ordinal_targets(
        torch.tensor(train["final_intensity"].to_numpy(), dtype=torch.long)
    )
    positive = train_targets.sum(dim=0)
    negative = len(train_targets) - positive
    threshold_weights = (negative / positive.clamp_min(1.0)).to(device)
    final_class_weights = _class_weights(
        train["final_intensity"], (1, 2, 3, 4), device
    )
    drop_class_weights = _class_weights(
        train["drop_magnitude"], (1, 2, 3, 4), device
    )
    joint_targets = [
        OUTCOME_PAIRS.index((int(final), int(drop))) + 1
        for final, drop in zip(train["final_intensity"], train["drop_magnitude"])
    ]
    joint_class_weights = _class_weights(
        joint_targets, tuple(range(1, len(OUTCOME_PAIRS) + 1)), device
    )
    epoch_loss_kwargs = {
        "final_class_weights": final_class_weights,
        "drop_class_weights": drop_class_weights,
        "joint_class_weights": joint_class_weights,
        "focal_gamma": float(config.get("outcome_focal_gamma", 2.0)),
        "joint_classification_weight": float(
            config.get("joint_classification_weight", 1.0)
        ),
        "marginal_auxiliary_weight": float(
            config.get("marginal_auxiliary_weight", 0.3)
        ),
        "direct_classification_weight": float(
            config.get("direct_classification_weight", 1.0)
        ),
        "ordinal_auxiliary_weight": float(
            config.get("ordinal_auxiliary_weight", 0.2)
        ),
        "consistency_weight": float(config.get("consistency_weight", 0.1)),
    }
    optimizer = AdamW(
        _outcome_parameter_groups(model, config),
        weight_decay=config["weight_decay"],
    )
    total_steps = max(len(loaders["train"]) * config["epochs"], 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    freeze_epochs = int(config.get("freeze_text_encoder_epochs", 1))
    best_score = -float("inf")
    best_epoch = 0
    model_path = output / "best_model.pt"
    progress_path = output / "training_progress.jsonl"
    history: list[dict[str, Any]] = []
    run_config = {
        "run_type": "outcome_ceiling",
        "checkpoint": 1.0,
        "seed": seed,
        "transfer": transfer_checkpoint is not None,
        "source_checkpoint": (
            str(Path(transfer_checkpoint_path).resolve())
            if transfer_checkpoint_path is not None
            else None
        ),
        "source_checkpoint_sha256": source_checkpoint_hash,
        "split_hash": _split_hash(subset),
        "emotion_vocabulary": emotion_vocabulary,
        "problem_vocabulary": problem_vocabulary,
        "threshold_positive_weights": threshold_weights.detach().cpu().tolist(),
        "final_class_weights": final_class_weights.detach().cpu().tolist(),
        "drop_class_weights": drop_class_weights.detach().cpu().tolist(),
        "joint_pairs": [list(pair) for pair in OUTCOME_PAIRS],
        "joint_class_weights": joint_class_weights.detach().cpu().tolist(),
        "outcome_sampling_strategy": sampling_strategy,
        "sampling_weight_summary": (
            {
                "minimum": float(sampling_weights.min()),
                "maximum": float(sampling_weights.max()),
                "mean": float(sampling_weights.mean()),
            }
            if sampling_weights is not None
            else None
        ),
        "architecture": config.get("architecture", "base"),
        "precision": precision,
        "outcome_augmentation": outcome_augmentation_report,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": getattr(torch.version, "hip", None),
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "flash_sdp_enabled": (
                torch.backends.cuda.flash_sdp_enabled()
                if torch.cuda.is_available()
                else None
            ),
            "memory_efficient_sdp_enabled": (
                torch.backends.cuda.mem_efficient_sdp_enabled()
                if torch.cuda.is_available()
                else None
            ),
            "math_sdp_enabled": (
                torch.backends.cuda.math_sdp_enabled()
                if torch.cuda.is_available()
                else None
            ),
        },
        "config": config,
    }
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)
    _append_progress(progress_path, {"event": "run_started", **run_config})

    epoch_iterator = tqdm(
        range(1, config["epochs"] + 1),
        desc=f"Outcome ceiling seed {seed}",
        unit="epoch",
        position=progress_position,
        leave=True,
        dynamic_ncols=True,
    )
    for epoch in epoch_iterator:
        trainable_layers = _outcome_trainable_layers(epoch, config)
        model.set_text_encoder_trainable_layers(trainable_layers)
        epoch_started = time.time()
        train_metrics, _ = _run_outcome_ceiling_epoch(
            model,
            loaders["train"],
            device,
            threshold_weights,
            auxiliary_regression_weight,
            optimizer,
            scheduler,
            precision=precision,
            progress_description=f"outcome epoch {epoch} train",
            progress_position=progress_position + 1,
            **epoch_loss_kwargs,
        )
        validation_metrics, _ = _run_outcome_ceiling_epoch(
            model,
            loaders["validation"],
            device,
            threshold_weights,
            auxiliary_regression_weight,
            precision=precision,
            progress_description=f"outcome epoch {epoch} validation",
            progress_position=progress_position + 1,
            **epoch_loss_kwargs,
        )
        score = (
            validation_metrics["final_f1_macro"]
            + validation_metrics["drop_f1_macro"]
        ) / 2.0
        history.append(
            {
                "epoch": epoch,
                "text_encoder_frozen": epoch <= freeze_epochs,
                "text_encoder_trainable_layers": trainable_layers,
                "train": train_metrics,
                "validation": validation_metrics,
                "selection_score": score,
                "encoder_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
                "duration_seconds": time.time() - epoch_started,
            }
        )
        _append_progress(progress_path, {"event": "epoch_completed", **history[-1]})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
        epoch_iterator.set_postfix(
            val_f1=f"{score:.4f}", best_epoch=best_epoch or "-"
        )
        if epoch - best_epoch >= config["early_stopping_patience"]:
            break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    metric_rows: list[dict[str, Any]] = []
    evaluation_splits = (
        ("validation", "test")
        if config.get("evaluate_test", True)
        else ("validation",)
    )
    for split_name in evaluation_splits:
        metrics, predictions = _run_outcome_ceiling_epoch(
            model,
            loaders[split_name],
            device,
            threshold_weights,
            auxiliary_regression_weight,
            precision=precision,
            progress_description=f"outcome final {split_name}",
            progress_position=progress_position + 1,
            **epoch_loss_kwargs,
        )
        predictions["split"] = split_name
        predictions["seed"] = seed
        predictions["transfer"] = transfer_checkpoint is not None
        predictions["architecture"] = config.get("architecture", "base")
        predictions.to_csv(output / f"{split_name}_predictions.csv", index=False)
        diagnostic_rows: list[dict[str, Any]] = []
        confusion_rows: list[dict[str, Any]] = []
        for target, prefix in (
            ("final_intensity", "final"),
            ("drop_magnitude", "drop"),
        ):
            truth = predictions[f"{prefix}_target"]
            predicted = predictions[f"{prefix}_prediction"]
            diagnostic_rows.extend(
                {"target": target, **row}
                for row in per_class_metrics(
                    truth, predicted, labels=(1, 2, 3, 4), label_names=("1", "2", "3", "4")
                )
            )
            confusion_rows.extend(
                {"target": target, **row}
                for row in confusion_matrix_records(
                    truth, predicted, labels=(1, 2, 3, 4)
                )
            )
            metric_rows.append(
                {
                    "checkpoint": 1.0,
                    "checkpoint_percent": 100,
                    "split": split_name,
                    "target": target,
                    "model": (
                        f"{'emotion_conditioned' if config.get('use_emotion_conditioned_heads', True) else 'shared'}_"
                        f"{'joint' if config.get('use_joint_pair_head', True) else 'factorized'}_"
                        f"{config.get('architecture', 'base')}"
                    ),
                    "transfer": transfer_checkpoint is not None,
                    "seed": seed,
                    **{
                        key.removeprefix(f"{prefix}_"): value
                        for key, value in metrics.items()
                        if key.startswith(f"{prefix}_")
                    },
                    "joint_consistency_rate": metrics["joint_consistency_rate"],
                    "invalid_derived_drop_rate": metrics[
                        "invalid_derived_drop_rate"
                    ],
                }
            )
        pd.DataFrame(diagnostic_rows).to_csv(
            output / f"{split_name}_per_class_metrics.csv", index=False
        )
        pd.DataFrame(confusion_rows).to_csv(
            output / f"{split_name}_confusion_matrices.csv", index=False
        )
    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    with (output / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    summary = {
        "checkpoint": 1.0,
        "model": (
            f"{'emotion_conditioned' if config.get('use_emotion_conditioned_heads', True) else 'shared'}_"
            f"{'joint' if config.get('use_joint_pair_head', True) else 'factorized'}_"
            f"{config.get('architecture', 'base')}"
        ),
        "transfer": transfer_checkpoint is not None,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "split_hash": _split_hash(subset),
        "source_checkpoint": run_config["source_checkpoint"],
        "source_checkpoint_sha256": source_checkpoint_hash,
        "device": str(device),
        "precision": precision,
        "environment": run_config["environment"],
        "test_evaluated": "test" in evaluation_splits,
        "duration_seconds": time.time() - started_at,
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _append_progress(progress_path, {"event": "run_completed", **summary})
    return summary
