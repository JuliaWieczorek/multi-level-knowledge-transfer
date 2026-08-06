from __future__ import annotations

import hashlib
import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .metrics import classification_metrics, ordinal_metrics
from .models import (
    BinaryFocalLoss,
    SoftSharingMTL,
    TemporalMultiModalModel,
    temporal_multitask_loss,
)
from .neural_data import SourceMTLDataset, TemporalDataset, temporal_collate
from .strategy import strategy_feature_columns, strategy_vocabulary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _class_weights(
    values: Iterable[int], classes: Sequence[int], device: torch.device
) -> torch.Tensor:
    array = np.asarray(list(values), dtype=int)
    counts = np.asarray([(array == label).sum() for label in classes], dtype=float)
    weights = len(array) / (len(classes) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float, device=device)


def _move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


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
    optimizer: AdamW | None = None,
    scheduler: Any | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    sentiment_true: list[int] = []
    sentiment_pred: list[int] = []
    emotion_true: list[int] = []
    emotion_pred: list[int] = []
    intensity_true: list[int] = []
    intensity_pred: list[int] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = _move_tensors(batch, device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss, _ = _source_loss(
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
            sentiment_true.extend(batch["sentiment"].detach().cpu().tolist())
            sentiment_pred.extend(
                outputs["sentiment"].argmax(dim=-1).detach().cpu().tolist()
            )
            emotion_true.extend(
                batch["emotion"].detach().cpu().to(torch.int).flatten().tolist()
            )
            emotion_pred.extend(
                (torch.sigmoid(outputs["emotion"]) >= 0.4)
                .to(torch.int)
                .detach()
                .cpu()
                .flatten()
                .tolist()
            )
            active = batch["intensity"] >= 0
            intensity_true.extend(batch["intensity"][active].detach().cpu().tolist())
            intensity_pred.extend(
                outputs["intensity"][active].argmax(dim=-1).detach().cpu().tolist()
            )
    metrics = {"loss": float(np.mean(losses))}
    metrics.update(
        {
            f"sentiment_{key}": value
            for key, value in classification_metrics(
                sentiment_true, sentiment_pred
            ).items()
        }
    )
    metrics.update(
        {
            f"emotion_{key}": value
            for key, value in classification_metrics(
                emotion_true, emotion_pred
            ).items()
        }
    )
    if intensity_true:
        metrics.update(
            {
                f"intensity_{key}": value
                for key, value in classification_metrics(
                    intensity_true, intensity_pred
                ).items()
            }
        )
    return metrics


def pretrain_source_mtl(
    source_path: str | Path,
    output_dir: str | Path,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source_path)
    emotion_names = sorted(
        column.split("emotion__", 1)[1]
        for column in frame.columns
        if column.startswith("emotion__")
    )
    if not emotion_names:
        raise ValueError("Source data has no one-hot emotion columns.")
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Source data requires train and validation rows.")

    device = resolve_device(config.get("device", "auto"))
    model_name = config["transformer_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    checkpoint_path = output / "source_transfer_checkpoint.pt"
    source_split_hash = _split_hash(frame)
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_type": "source_mtl",
                "seed": seed,
                "source_path": str(Path(source_path).resolve()),
                "source_sha256": _file_hash(source_path),
                "split_hash": source_split_hash,
                "config": config,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = _run_source_epoch(
            model,
            train_loader,
            device,
            sentiment_loss,
            emotion_loss,
            task_weights,
            config["soft_sharing_lambda"],
            optimizer,
            scheduler,
        )
        validation_metrics = _run_source_epoch(
            model,
            validation_loader,
            device,
            sentiment_loss,
            emotion_loss,
            task_weights,
            config["soft_sharing_lambda"],
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        if validation_metrics["loss"] < best_loss:
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
        "checkpoint": str(checkpoint_path),
        "split_hash": source_split_hash,
        "emotion_names": emotion_names,
        "device": str(device),
    }
    with (output / "source_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
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
) -> tuple[dict[str, float], pd.DataFrame]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    identifiers: list[str] = []
    final_true: list[int] = []
    final_pred: list[int] = []
    drop_true: list[int] = []
    drop_pred: list[int] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            identifiers.extend(raw_batch["conversation_id"])
            batch = _move_tensors(raw_batch, device)
            outputs = model(batch)
            loss, _ = temporal_multitask_loss(
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
    final_metrics = classification_metrics(final_true, final_pred)
    final_metrics.update(ordinal_metrics(final_true, final_pred))
    drop_metrics = classification_metrics(drop_true, drop_pred)
    drop_metrics.update(ordinal_metrics(drop_true, drop_pred))
    metrics = {"loss": float(np.mean(losses))}
    metrics.update({f"final_{key}": value for key, value in final_metrics.items()})
    metrics.update({f"drop_{key}": value for key, value in drop_metrics.items()})
    predictions = pd.DataFrame(
        {
            "conversation_id": identifiers,
            "final_target": final_true,
            "final_prediction": final_pred,
            "drop_target": drop_true,
            "drop_prediction": drop_pred,
        }
    )
    return metrics, predictions


def train_temporal_model(
    checkpoints_path: str | Path,
    output_dir: str | Path,
    checkpoint: float,
    modality: str,
    seed: int,
    config: dict[str, Any],
    transfer_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(checkpoints_path)
    subset = frame[np.isclose(frame["checkpoint"].astype(float), checkpoint)].copy()
    if subset.empty:
        raise ValueError(f"No rows found for checkpoint {checkpoint}.")
    train = subset[subset["split"] == "train"]
    validation = subset[subset["split"] == "validation"]
    test = subset[subset["split"] == "test"]
    transfer_checkpoint = None
    source_checkpoint_hash = None
    if transfer_checkpoint_path is not None:
        transfer_checkpoint = torch.load(
            transfer_checkpoint_path, map_location="cpu", weights_only=False
        )
        source_checkpoint_hash = _file_hash(transfer_checkpoint_path)
    device = resolve_device(config.get("device", "auto"))
    tokenizer = (
        AutoTokenizer.from_pretrained(config["transformer_name"])
        if "text" in modality
        else None
    )
    dataset_arguments = {
        "tokenizer": tokenizer,
        "modality": modality,
        "strategy_mode": "quantity_timing_order",
        "max_length": config["max_length"],
        "max_chunks": config["max_chunks"],
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
    ).to(device)
    run_config = {
        "run_type": "temporal",
        "checkpoint": checkpoint,
        "modality": modality,
        "transfer": transfer_checkpoint is not None,
        "seed": seed,
        "checkpoints_path": str(Path(checkpoints_path).resolve()),
        "checkpoints_sha256": _file_hash(checkpoints_path),
        "split_hash": _split_hash(subset),
        "source_checkpoint": (
            str(Path(transfer_checkpoint_path).resolve())
            if transfer_checkpoint_path is not None
            else None
        ),
        "source_checkpoint_sha256": source_checkpoint_hash,
        "config": config,
    }
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)
    final_weights = _class_weights(
        train["final_intensity"], (1, 2, 3, 4), device
    )
    drop_weights = _class_weights(
        train["drop_magnitude"], (1, 2, 3, 4), device
    )
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
    history: list[dict[str, Any]] = []
    for epoch in range(1, config["epochs"] + 1):
        train_metrics, _ = _run_temporal_epoch(
            model,
            loaders["train"],
            device,
            final_weights,
            drop_weights,
            optimizer,
            scheduler,
        )
        validation_metrics, _ = _run_temporal_epoch(
            model,
            loaders["validation"],
            device,
            final_weights,
            drop_weights,
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
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
        if epoch - best_epoch >= config["early_stopping_patience"]:
            break
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    metric_rows: list[dict[str, Any]] = []
    for split_name in ("validation", "test"):
        metrics, predictions = _run_temporal_epoch(
            model,
            loaders[split_name],
            device,
            final_weights,
            drop_weights,
        )
        predictions["split"] = split_name
        predictions["checkpoint"] = checkpoint
        predictions["checkpoint_percent"] = int(round(checkpoint * 100))
        predictions["modality"] = modality
        predictions["transfer"] = transfer_checkpoint is not None
        predictions["seed"] = seed
        predictions.to_csv(output / f"{split_name}_predictions.csv", index=False)
        for target, prefix in (
            ("final_intensity", "final"),
            ("drop_magnitude", "drop"),
        ):
            metric_rows.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_percent": int(round(checkpoint * 100)),
                    "split": split_name,
                    "target": target,
                    "modality": modality,
                    "transfer": transfer_checkpoint is not None,
                    "seed": seed,
                    **{
                        key.removeprefix(f"{prefix}_"): value
                        for key, value in metrics.items()
                        if key.startswith(f"{prefix}_")
                    },
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
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
