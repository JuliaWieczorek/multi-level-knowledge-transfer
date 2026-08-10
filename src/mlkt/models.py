from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from transformers import AutoModel


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dimension: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    numerator = (values * weights).sum(dim=dimension)
    denominator = weights.sum(dim=dimension).clamp_min(1.0)
    return numerator / denominator


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        binary_ce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probability = torch.sigmoid(logits)
        pt = probability * targets + (1.0 - probability) * (1.0 - targets)
        loss = (1.0 - pt).pow(self.gamma) * binary_ce
        if self.alpha is not None:
            positive_weight = self.alpha.view(1, -1)
            alpha_factor = positive_weight * targets + (1.0 - targets)
            loss = loss * alpha_factor
        return loss.mean()


class SoftSharingMTL(nn.Module):
    """Winning three-task architecture reused from the IEEE study."""

    TASKS = ("sentiment", "emotion", "intensity")

    def __init__(
        self,
        transformer_name: str,
        num_emotions: int,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.transformer_name = transformer_name
        self.num_emotions = num_emotions
        self.encoders = nn.ModuleDict(
            {
                task: AutoModel.from_pretrained(transformer_name)
                for task in self.TASKS
            }
        )
        hidden = self.encoders["emotion"].config.hidden_size
        projection = hidden // 2
        self.shared_projection = nn.Linear(hidden, projection)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                "sentiment": nn.Linear(projection, 3),
                "emotion": nn.Linear(projection, num_emotions),
                "intensity": nn.Linear(projection, num_emotions * 3),
            }
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for task, encoder in self.encoders.items():
            pooled = encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0]
            projected = torch.relu(
                self.shared_projection(self.dropout(pooled))
            )
            outputs[task] = self.heads[task](projected)
        outputs["intensity"] = outputs["intensity"].view(
            -1, self.num_emotions, 3
        )
        return outputs

    def soft_sharing_penalty(self, coefficient: float = 1e-4) -> torch.Tensor:
        encoders = list(self.encoders.values())
        loss = next(self.parameters()).new_zeros(())
        for left_index in range(len(encoders)):
            for right_index in range(left_index + 1, len(encoders)):
                for left, right in zip(
                    encoders[left_index].parameters(),
                    encoders[right_index].parameters(),
                ):
                    if left.shape == right.shape:
                        loss = loss + (left - right).pow(2).sum()
        return coefficient * loss

    def export_transfer_state(self) -> dict[str, Any]:
        return {
            "transformer_name": self.transformer_name,
            "num_emotions": self.num_emotions,
            "emotion_encoder": self.encoders["emotion"].state_dict(),
            "intensity_encoder": self.encoders["intensity"].state_dict(),
        }


def load_transferred_encoder_pair(
    transformer_name: str,
    checkpoint: dict[str, Any] | None,
) -> tuple[nn.Module, nn.Module]:
    emotion_encoder = AutoModel.from_pretrained(transformer_name)
    intensity_encoder = AutoModel.from_pretrained(transformer_name)
    if checkpoint is not None:
        emotion_encoder.load_state_dict(checkpoint["emotion_encoder"])
        intensity_encoder.load_state_dict(checkpoint["intensity_encoder"])
    return emotion_encoder, intensity_encoder


class AffectiveTextEncoder(nn.Module):
    def __init__(
        self,
        transformer_name: str,
        transfer_checkpoint: dict[str, Any] | None,
        dropout: float = 0.4,
        chunk_layers: int = 2,
        chunk_heads: int = 8,
        max_chunks: int = 32,
    ) -> None:
        super().__init__()
        (
            self.emotion_encoder,
            self.intensity_encoder,
        ) = load_transferred_encoder_pair(transformer_name, transfer_checkpoint)
        hidden = self.emotion_encoder.config.hidden_size
        self.hidden_size = hidden
        self.max_chunks = max_chunks
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.chunk_positions = nn.Embedding(max_chunks, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=chunk_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.chunk_encoder = nn.TransformerEncoder(layer, num_layers=chunk_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, chunks, tokens = input_ids.shape
        if chunks > self.max_chunks:
            raise ValueError(
                f"Received {chunks} chunks, configured maximum is {self.max_chunks}."
            )
        flat_ids = input_ids.reshape(batch * chunks, tokens)
        flat_attention = attention_mask.reshape(batch * chunks, tokens)
        emotion = self.emotion_encoder(
            input_ids=flat_ids, attention_mask=flat_attention
        ).last_hidden_state[:, 0]
        intensity = self.intensity_encoder(
            input_ids=flat_ids, attention_mask=flat_attention
        ).last_hidden_state[:, 0]
        gate = self.gate(torch.cat([emotion, intensity], dim=-1))
        combined = gate * emotion + (1.0 - gate) * intensity
        combined = combined.reshape(batch, chunks, self.hidden_size)
        positions = torch.arange(chunks, device=input_ids.device).unsqueeze(0)
        combined = self.dropout(combined + self.chunk_positions(positions))
        encoded = self.chunk_encoder(
            combined, src_key_padding_mask=~chunk_mask.bool()
        )
        return masked_mean(encoded, chunk_mask, dimension=1)


class StrategyEncoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        numeric_size: int,
        hidden_size: int,
        dropout: float = 0.4,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("Strategy hidden size must be divisible by heads.")
        self.embedding = nn.Embedding(
            vocabulary_size, hidden_size, padding_idx=0
        )
        self.position_projection = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        strategy_ids: torch.Tensor,
        strategy_positions: torch.Tensor,
        strategy_mask: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        sequence = self.embedding(strategy_ids)
        sequence = sequence + self.position_projection(
            strategy_positions.unsqueeze(-1)
        )
        sequence = self.sequence_encoder(
            sequence, src_key_padding_mask=~strategy_mask.bool()
        )
        sequence = masked_mean(sequence, strategy_mask, dimension=1)
        numeric = self.numeric_projection(numeric_features)
        return self.fusion(torch.cat([sequence, numeric], dim=-1))


class TemporalMultiModalModel(nn.Module):
    MODALITIES = ("text", "strategy", "text_strategy")

    def __init__(
        self,
        modality: str,
        transformer_name: str,
        transfer_checkpoint: dict[str, Any] | None,
        strategy_vocabulary_size: int,
        strategy_numeric_size: int,
        strategy_hidden_size: int = 256,
        dropout: float = 0.4,
        max_chunks: int = 32,
        use_initial_intensity: bool = False,
    ) -> None:
        super().__init__()
        if modality not in self.MODALITIES:
            raise ValueError(f"Unsupported modality: {modality}")
        self.modality = modality
        self.use_initial_intensity = use_initial_intensity
        if use_initial_intensity and modality != "text_strategy":
            raise ValueError("Initial intensity is supported only for text_strategy.")
        self.text_encoder: AffectiveTextEncoder | None = None
        self.strategy_encoder: StrategyEncoder | None = None
        if "text" in modality:
            self.text_encoder = AffectiveTextEncoder(
                transformer_name=transformer_name,
                transfer_checkpoint=transfer_checkpoint,
                dropout=dropout,
                max_chunks=max_chunks,
            )
            output_size = self.text_encoder.hidden_size
        if "strategy" in modality:
            self.strategy_encoder = StrategyEncoder(
                vocabulary_size=strategy_vocabulary_size,
                numeric_size=strategy_numeric_size,
                hidden_size=strategy_hidden_size,
                dropout=dropout,
            )
            output_size = strategy_hidden_size
        if modality == "text_strategy":
            assert self.text_encoder is not None
            self.strategy_to_text = nn.Linear(
                strategy_hidden_size, self.text_encoder.hidden_size
            )
            self.modality_gate = nn.Sequential(
                nn.Linear(self.text_encoder.hidden_size * 2, self.text_encoder.hidden_size),
                nn.Sigmoid(),
            )
            output_size = self.text_encoder.hidden_size
        shared_input_size = output_size + (1 if use_initial_intensity else 0)
        self.shared = nn.Sequential(
            nn.Linear(shared_input_size, output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.final_head = nn.Linear(output_size // 2, 4)
        self.drop_head = nn.Linear(output_size // 2, 4)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text: torch.Tensor | None = None
        strategy: torch.Tensor | None = None
        if self.text_encoder is not None:
            text = self.text_encoder(
                batch["input_ids"],
                batch["attention_mask"],
                batch["chunk_mask"],
            )
        if self.strategy_encoder is not None:
            strategy = self.strategy_encoder(
                batch["strategy_ids"],
                batch["strategy_positions"],
                batch["strategy_mask"],
                batch["strategy_numeric"],
            )
        if self.modality == "text":
            assert text is not None
            representation = text
        elif self.modality == "strategy":
            assert strategy is not None
            representation = strategy
        else:
            assert text is not None and strategy is not None
            projected_strategy = self.strategy_to_text(strategy)
            gate = self.modality_gate(
                torch.cat([text, projected_strategy], dim=-1)
            )
            representation = gate * text + (1.0 - gate) * projected_strategy
        if self.use_initial_intensity:
            if "initial_feature" not in batch:
                raise ValueError("Initial-aware model requires initial_feature.")
            representation = torch.cat([representation, batch["initial_feature"]], dim=-1)
        shared = self.shared(representation)
        return {
            "final_intensity": self.final_head(shared),
            "drop_magnitude": self.drop_head(shared),
        }


def temporal_multitask_loss(
    outputs: dict[str, torch.Tensor],
    final_targets: torch.Tensor,
    drop_targets: torch.Tensor,
    final_weights: torch.Tensor | None = None,
    drop_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    final_loss = nn.functional.cross_entropy(
        outputs["final_intensity"], final_targets, weight=final_weights
    )
    drop_loss = nn.functional.cross_entropy(
        outputs["drop_magnitude"], drop_targets, weight=drop_weights
    )
    total = final_loss + drop_loss
    return total, {
        "final_loss": float(final_loss.detach().cpu()),
        "drop_loss": float(drop_loss.detach().cpu()),
    }
