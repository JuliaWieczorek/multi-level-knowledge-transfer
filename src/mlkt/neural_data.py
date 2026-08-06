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
        tokenizer.prepare_for_model(
            chunk,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=False,
            return_attention_mask=True,
        )
        for chunk in packed
    ]


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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
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
    ) -> None:
        self.frame = ensure_strategy_columns(frame).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.modality = modality
        self.strategy_columns = strategy_feature_columns(strategy_mode)
        self.max_length = max_length
        self.max_chunks = max_chunks
        if "text" in modality and tokenizer is None:
            raise ValueError("Text modalities require a tokenizer.")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        item: dict[str, Any] = {
            "conversation_id": str(row["conversation_id"]),
            "final_target": int(row["final_intensity"]) - 1,
            "drop_target": int(row["drop_magnitude"]) - 1,
        }
        if "text" in self.modality:
            turns = json.loads(row["text_seeker_turns"])
            item["chunks"] = tokenize_turn_chunks(
                self.tokenizer,
                turns,
                max_length=self.max_length,
                max_chunks=self.max_chunks,
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
    }
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
