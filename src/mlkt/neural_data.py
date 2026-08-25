from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .strategy import (
    encode_strategy_sequence,
    ensure_strategy_columns,
    parse_strategy_positions,
    parse_strategy_sequence,
    strategy_feature_columns,
)

SENTIMENT_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


def pack_tokenized_turns(
    tokenized_turns: Sequence[Sequence[int]],
    separator_id: int,
    max_content_tokens: int,
) -> list[list[int]]:
    """Pack complete turns into chunks, splitting only overlong individual turns."""
    if max_content_tokens < 1:
        raise ValueError("max_content_tokens must be positive.")
    chunks: list[list[int]] = []
    current: list[int] = []
    for turn in tokenized_turns:
        remaining = list(turn)
        if not remaining:
            continue
        while remaining:
            separator_cost = 1 if current else 0
            available = max_content_tokens - len(current) - separator_cost
            if available <= 0:
                chunks.append(current)
                current = []
                continue
            take = remaining[:available]
            if current:
                current.append(separator_id)
            current.extend(take)
            remaining = remaining[len(take) :]
            if remaining or len(current) == max_content_tokens:
                chunks.append(current)
                current = []
    if current:
        chunks.append(current)
    return chunks or [[]]


def prepare_token_chunk(
    tokenizer: Any,
    token_ids: Sequence[int],
    max_length: int,
) -> dict[str, list[int]]:
    """Add special tokens and padding across Transformers 4.x and 5.x."""
    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if cls_token_id is not None and sep_token_id is not None:
        input_ids = [cls_token_id, *token_ids, sep_token_id]
    elif bos_token_id is not None and eos_token_id is not None:
        input_ids = [bos_token_id, *token_ids, eos_token_id]
    else:
        build_inputs = getattr(
            tokenizer, "build_inputs_with_special_tokens", None
        )
        if build_inputs is not None:
            input_ids = list(build_inputs(list(token_ids)))
        else:
            input_ids = []
    if not input_ids:
        return tokenizer.prepare_for_model(
            list(token_ids),
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=False,
            return_attention_mask=True,
        )
    if len(input_ids) > max_length:
        raise ValueError("Prepared token chunk exceeds max_length.")
    padding = max_length - len(input_ids)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0
    if getattr(tokenizer, "padding_side", "right") == "left":
        return {
            "input_ids": [pad_token_id] * padding + input_ids,
            "attention_mask": [0] * padding + [1] * len(input_ids),
        }
    return {
        "input_ids": input_ids + [pad_token_id] * padding,
        "attention_mask": [1] * len(input_ids) + [0] * padding,
    }


def tokenize_turn_chunks(
    tokenizer: Any,
    turns: Sequence[str],
    max_length: int = 128,
    max_chunks: int = 32,
) -> list[dict[str, list[int]]]:
    separator = tokenizer.sep_token_id
    if separator is None:
        separator = tokenizer.eos_token_id
    if separator is None:
        raise ValueError("Tokenizer must expose a separator or EOS token.")
    tokenized = [
        tokenizer.encode(turn, add_special_tokens=False) for turn in turns if turn.strip()
    ]
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    content_limit = max_length - special_count
    packed = pack_tokenized_turns(tokenized, separator, content_limit)
    if len(packed) > max_chunks:
        raise ValueError(
            f"Dialogue needs {len(packed)} chunks, configured maximum is {max_chunks}."
        )
    return [
        prepare_token_chunk(tokenizer, chunk, max_length)
        for chunk in packed
    ]


def tokenize_role_aware_chunks(
    tokenizer: Any,
    turns: Sequence[str],
    max_length: int = 128,
    max_chunks: int = 32,
) -> list[dict[str, Any]]:
    """Tokenize dialogue chunks and retain speaker/trajectory composition.

    Each chunk receives seeker/supporter token shares and early/late seeker
    token shares. This keeps the compact chunk representation while making
    roles and within-dialogue emotional trajectory explicit to the model.
    """
    separator = tokenizer.sep_token_id
    if separator is None:
        separator = tokenizer.eos_token_id
    if separator is None:
        raise ValueError("Tokenizer must expose a separator or EOS token.")
    content_limit = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_limit < 1:
        raise ValueError("max_length leaves no room for content tokens.")

    parsed: list[tuple[str, str]] = []
    for value in turns:
        role, delimiter, _ = str(value).partition(": ")
        parsed.append((role.strip().lower() if delimiter else "unknown", str(value)))
    seeker_count = sum(role == "seeker" for role, _ in parsed)
    seeker_index = 0
    annotated_turns: list[tuple[list[int], np.ndarray, np.ndarray]] = []
    for role, text in parsed:
        speaker = np.asarray(
            [float(role == "seeker"), float(role == "supporter")],
            dtype=np.float32,
        )
        trajectory = np.zeros(2, dtype=np.float32)
        if role == "seeker":
            if seeker_count == 1:
                trajectory[:] = 1.0
            elif seeker_index < (seeker_count + 1) // 2:
                trajectory[0] = 1.0
            else:
                trajectory[1] = 1.0
            seeker_index += 1
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if tokens:
            annotated_turns.append((tokens, speaker, trajectory))

    chunks: list[dict[str, Any]] = []
    current: list[int] = []
    speaker_counts = np.zeros(2, dtype=np.float32)
    trajectory_counts = np.zeros(2, dtype=np.float32)
    content_tokens = 0

    def flush() -> None:
        nonlocal current, speaker_counts, trajectory_counts, content_tokens
        if not current:
            return
        encoded = prepare_token_chunk(tokenizer, current, max_length)
        denominator = float(max(content_tokens, 1))
        encoded["speaker_features"] = (speaker_counts / denominator).tolist()
        encoded["trajectory_features"] = (
            trajectory_counts / denominator
        ).tolist()
        chunks.append(encoded)
        current = []
        speaker_counts = np.zeros(2, dtype=np.float32)
        trajectory_counts = np.zeros(2, dtype=np.float32)
        content_tokens = 0

    for tokens, speaker, trajectory in annotated_turns:
        remaining = list(tokens)
        while remaining:
            separator_cost = 1 if current else 0
            available = content_limit - len(current) - separator_cost
            if available <= 0:
                flush()
                continue
            take = remaining[:available]
            if current:
                current.append(separator)
            current.extend(take)
            token_count = len(take)
            speaker_counts += speaker * token_count
            trajectory_counts += trajectory * token_count
            content_tokens += token_count
            remaining = remaining[token_count:]
            if remaining or len(current) == content_limit:
                flush()
    flush()
    if not chunks:
        raise ValueError("Role-aware tokenization requires at least one text token.")
    if len(chunks) > max_chunks:
        raise ValueError(
            f"Dialogue needs {len(chunks)} chunks, configured maximum is {max_chunks}."
        )
    return chunks


class SourceMTLDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        emotion_names: Sequence[str],
        max_length: int = 128,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.emotion_names = list(emotion_names)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            str(row["Utterances"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        emotions = np.asarray(
            [int(row[f"emotion__{name}"]) for name in self.emotion_names],
            dtype=np.float32,
        )
        intensities = np.asarray(
            [
                int(row[f"intensity__{name}"]) - 1
                if int(row[f"emotion__{name}"]) == 1
                else -100
                for name in self.emotion_names
            ],
            dtype=np.int64,
        )
        return {
            "row_id": int(index),
            "conversation_id": str(row["conversation_id"]),
            "source_conversation_id": str(
                row.get("source_conversation_id", row["conversation_id"])
            ),
            "augmented": bool(row.get("augmented", False)),
            "generation_valid": bool(row.get("generation_valid", True)),
            "generation_quality": float(row.get("quality", 1.0)),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "sentiment": torch.tensor(
                SENTIMENT_TO_ID[str(row["sentiment"])], dtype=torch.long
            ),
            "emotion": torch.tensor(emotions, dtype=torch.float),
            "intensity": torch.tensor(intensities, dtype=torch.long),
        }


class TemporalDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any | None,
        modality: str,
        strategy_mode: str = "quantity_timing_order",
        max_length: int = 128,
        max_chunks: int = 32,
        use_initial_intensity: bool = False,
        initial_intensity_mean: float | None = None,
        initial_intensity_std: float | None = None,
        cache_tokenization: bool = True,
    ) -> None:
        self.frame = ensure_strategy_columns(frame).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.modality = modality
        self.strategy_columns = strategy_feature_columns(strategy_mode)
        self.max_length = max_length
        self.max_chunks = max_chunks
        self.use_initial_intensity = use_initial_intensity
        self.initial_intensity_mean = initial_intensity_mean
        self.initial_intensity_std = initial_intensity_std
        if "text" in modality and tokenizer is None:
            raise ValueError("Text modalities require a tokenizer.")
        if use_initial_intensity and modality != "text_strategy":
            raise ValueError("Initial intensity is supported only for text_strategy.")
        if use_initial_intensity and (
            initial_intensity_mean is None
            or initial_intensity_std is None
            or initial_intensity_std <= 0
        ):
            raise ValueError("Initial-aware datasets require train-derived mean and std.")
        self._tokenized_chunks: list[list[dict[str, torch.Tensor]]] | None = None
        if "text" in modality and cache_tokenization:
            self._tokenized_chunks = [
                tokenize_turn_chunks(
                    self.tokenizer,
                    json.loads(raw_turns),
                    max_length=self.max_length,
                    max_chunks=self.max_chunks,
                )
                for raw_turns in self.frame["text_seeker_turns"]
            ]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        item: dict[str, Any] = {
            "conversation_id": str(row["conversation_id"]),
            "final_target": int(row["final_intensity"]) - 1,
            "drop_target": int(row["drop_magnitude"]) - 1,
            "initial_intensity": int(row["initial_intensity"]),
            "final_target_raw": int(
                row.get("final_intensity_raw", row["final_intensity"])
            ),
            "drop_target_raw": int(
                row.get("drop_magnitude_raw", row["drop_magnitude"])
            ),
        }
        if self.use_initial_intensity:
            item["initial_feature"] = (
                float(row["initial_intensity"]) - float(self.initial_intensity_mean)
            ) / float(self.initial_intensity_std)
        if "text" in self.modality:
            item["chunks"] = (
                self._tokenized_chunks[index]
                if self._tokenized_chunks is not None
                else tokenize_turn_chunks(
                    self.tokenizer,
                    json.loads(row["text_seeker_turns"]),
                    max_length=self.max_length,
                    max_chunks=self.max_chunks,
                )
            )
        if "strategy" in self.modality:
            sequence = parse_strategy_sequence(row["strategy_sequence"])
            positions = parse_strategy_positions(
                row.get("strategy_positions_normalized", ""), len(sequence)
            )
            strategy_ids, strategy_positions = encode_strategy_sequence(
                sequence, positions
            )
            item["strategy_ids"] = strategy_ids
            item["strategy_positions"] = strategy_positions
            item["strategy_numeric"] = row[self.strategy_columns].astype(float).to_numpy(
                dtype=np.float32
            )
        return item


def categorical_vocabulary(values: Sequence[Any]) -> dict[str, int]:
    """Build a train-only categorical vocabulary with an explicit unknown id."""
    normalised = sorted({str(value).strip().lower() for value in values})
    return {"<unknown>": 0, **{value: index + 1 for index, value in enumerate(normalised)}}


class OutcomeCeilingDataset(Dataset):
    """Full-context role-aware ESConv data with pre-outcome metadata."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        emotion_vocabulary: dict[str, int],
        problem_vocabulary: dict[str, int],
        strategy_mode: str = "quantity_timing_order",
        max_length: int = 128,
        max_chunks: int = 32,
        cache_tokenization: bool = True,
    ) -> None:
        required = {
            "conversation_id",
            "text_role_turns",
            "initial_intensity",
            "final_intensity",
            "drop_magnitude",
            "emotion_family",
            "problem_type",
            "strategy_sequence",
            "strategy_positions_normalized",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                "Outcome ceiling data is missing columns: " f"{sorted(missing)}"
            )
        self.frame = ensure_strategy_columns(frame).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.emotion_vocabulary = emotion_vocabulary
        self.problem_vocabulary = problem_vocabulary
        self.strategy_columns = strategy_feature_columns(strategy_mode)
        self.max_length = max_length
        self.max_chunks = max_chunks
        self._tokenized_chunks: list[list[dict[str, torch.Tensor]]] | None = None
        if cache_tokenization:
            self._tokenized_chunks = [
                tokenize_role_aware_chunks(
                    self.tokenizer,
                    [str(turn) for turn in json.loads(raw_turns)],
                    max_length=self.max_length,
                    max_chunks=self.max_chunks,
                )
                for raw_turns in self.frame["text_role_turns"]
            ]

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def _category_id(value: Any, vocabulary: dict[str, int]) -> int:
        return vocabulary.get(str(value).strip().lower(), 0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        chunks = (
            self._tokenized_chunks[index]
            if self._tokenized_chunks is not None
            else tokenize_role_aware_chunks(
                self.tokenizer,
                [str(turn) for turn in json.loads(row["text_role_turns"])],
                max_length=self.max_length,
                max_chunks=self.max_chunks,
            )
        )
        strategy_sequence = parse_strategy_sequence(row["strategy_sequence"])
        strategy_positions = parse_strategy_positions(
            row.get("strategy_positions_normalized", ""),
            len(strategy_sequence),
        )
        strategy_ids, strategy_positions = encode_strategy_sequence(
            strategy_sequence, strategy_positions
        )
        return {
            "conversation_id": str(row["conversation_id"]),
            "chunks": chunks,
            "initial_intensity": int(row["initial_intensity"]),
            "emotion_id": self._category_id(
                row["emotion_family"], self.emotion_vocabulary
            ),
            "problem_id": self._category_id(
                row["problem_type"], self.problem_vocabulary
            ),
            "final_target": int(row["final_intensity"]),
            "drop_target": int(row["drop_magnitude"]),
            "strategy_ids": strategy_ids,
            "strategy_positions": strategy_positions,
            "strategy_numeric": row[self.strategy_columns].astype(float).to_numpy(
                dtype=np.float32
            ),
        }


def outcome_ceiling_collate(
    items: Sequence[dict[str, Any]], pad_token_id: int = 0
) -> dict[str, Any]:
    """Collate role-aware chunks and metadata for the ceiling experiment."""
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    max_chunks = max(len(item["chunks"]) for item in items)
    token_length = len(items[0]["chunks"][0]["input_ids"])
    input_ids = torch.full(
        (len(items), max_chunks, token_length), pad_token_id, dtype=torch.long
    )
    attention = torch.zeros_like(input_ids)
    chunk_mask = torch.zeros((len(items), max_chunks), dtype=torch.bool)
    speaker_features = torch.zeros((len(items), max_chunks, 2), dtype=torch.float)
    trajectory_features = torch.zeros(
        (len(items), max_chunks, 2), dtype=torch.float
    )
    for row_index, item in enumerate(items):
        for chunk_index, chunk in enumerate(item["chunks"]):
            input_ids[row_index, chunk_index] = torch.tensor(
                chunk["input_ids"], dtype=torch.long
            )
            attention[row_index, chunk_index] = torch.tensor(
                chunk["attention_mask"], dtype=torch.long
            )
            chunk_mask[row_index, chunk_index] = True
            speaker_features[row_index, chunk_index] = torch.tensor(
                chunk["speaker_features"], dtype=torch.float
            )
            trajectory_features[row_index, chunk_index] = torch.tensor(
                chunk["trajectory_features"], dtype=torch.float
            )
    max_strategies = max(len(item["strategy_ids"]) for item in items)
    strategy_ids = torch.zeros((len(items), max_strategies), dtype=torch.long)
    strategy_positions = torch.zeros(
        (len(items), max_strategies), dtype=torch.float
    )
    strategy_mask = torch.zeros((len(items), max_strategies), dtype=torch.bool)
    for row_index, item in enumerate(items):
        length = len(item["strategy_ids"])
        strategy_ids[row_index, :length] = torch.tensor(
            item["strategy_ids"], dtype=torch.long
        )
        strategy_positions[row_index, :length] = torch.tensor(
            item["strategy_positions"], dtype=torch.float
        )
        strategy_mask[row_index, :length] = True
    return {
        "conversation_id": [item["conversation_id"] for item in items],
        "input_ids": input_ids,
        "attention_mask": attention,
        "chunk_mask": chunk_mask,
        "speaker_features": speaker_features,
        "trajectory_features": trajectory_features,
        "strategy_ids": strategy_ids,
        "strategy_positions": strategy_positions,
        "strategy_mask": strategy_mask,
        "strategy_numeric": torch.tensor(
            np.stack([item["strategy_numeric"] for item in items]),
            dtype=torch.float,
        ),
        "initial_intensity": torch.tensor(
            [item["initial_intensity"] for item in items], dtype=torch.long
        ),
        "emotion_id": torch.tensor(
            [item["emotion_id"] for item in items], dtype=torch.long
        ),
        "problem_id": torch.tensor(
            [item["problem_id"] for item in items], dtype=torch.long
        ),
        "final_target": torch.tensor(
            [item["final_target"] for item in items], dtype=torch.long
        ),
        "drop_target": torch.tensor(
            [item["drop_target"] for item in items], dtype=torch.long
        ),
    }


def temporal_collate(
    items: Sequence[dict[str, Any]],
    pad_token_id: int = 0,
) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    batch: dict[str, Any] = {
        "conversation_id": [item["conversation_id"] for item in items],
        "final_target": torch.tensor(
            [item["final_target"] for item in items], dtype=torch.long
        ),
        "drop_target": torch.tensor(
            [item["drop_target"] for item in items], dtype=torch.long
        ),
        "initial_intensity": torch.tensor(
            [item["initial_intensity"] for item in items], dtype=torch.long
        ),
        "final_target_raw": torch.tensor(
            [item.get("final_target_raw", item["final_target"] + 1) for item in items],
            dtype=torch.long,
        ),
        "drop_target_raw": torch.tensor(
            [item.get("drop_target_raw", item["drop_target"] + 1) for item in items],
            dtype=torch.long,
        ),
    }
    if "initial_feature" in items[0]:
        batch["initial_feature"] = torch.tensor(
            [[item["initial_feature"]] for item in items], dtype=torch.float
        )
    if "chunks" in items[0]:
        max_chunks = max(len(item["chunks"]) for item in items)
        token_length = len(items[0]["chunks"][0]["input_ids"])
        input_ids = torch.full(
            (len(items), max_chunks, token_length),
            pad_token_id,
            dtype=torch.long,
        )
        attention = torch.zeros_like(input_ids)
        chunk_mask = torch.zeros((len(items), max_chunks), dtype=torch.bool)
        for row_index, item in enumerate(items):
            for chunk_index, chunk in enumerate(item["chunks"]):
                input_ids[row_index, chunk_index] = torch.tensor(
                    chunk["input_ids"], dtype=torch.long
                )
                attention[row_index, chunk_index] = torch.tensor(
                    chunk["attention_mask"], dtype=torch.long
                )
                chunk_mask[row_index, chunk_index] = True
        batch.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention,
                "chunk_mask": chunk_mask,
            }
        )
    if "strategy_ids" in items[0]:
        max_strategies = max(len(item["strategy_ids"]) for item in items)
        strategy_ids = torch.zeros(
            (len(items), max_strategies), dtype=torch.long
        )
        strategy_positions = torch.zeros(
            (len(items), max_strategies), dtype=torch.float
        )
        strategy_mask = torch.zeros(
            (len(items), max_strategies), dtype=torch.bool
        )
        for row_index, item in enumerate(items):
            length = len(item["strategy_ids"])
            strategy_ids[row_index, :length] = torch.tensor(
                item["strategy_ids"], dtype=torch.long
            )
            strategy_positions[row_index, :length] = torch.tensor(
                item["strategy_positions"], dtype=torch.float
            )
            strategy_mask[row_index, :length] = True
        batch.update(
            {
                "strategy_ids": strategy_ids,
                "strategy_positions": strategy_positions,
                "strategy_mask": strategy_mask,
                "strategy_numeric": torch.tensor(
                    np.stack([item["strategy_numeric"] for item in items]),
                    dtype=torch.float,
                ),
            }
        )
    return batch
