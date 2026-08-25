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


def weighted_chunk_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Pool chunks with continuous weights, falling back for empty phases."""
    effective = weights.to(values.dtype) * mask.to(values.dtype)
    numerator = (values * effective.unsqueeze(-1)).sum(dim=1)
    denominator = effective.sum(dim=1, keepdim=True)
    pooled = numerator / denominator.clamp_min(1e-8)
    return torch.where(denominator > 0, pooled, fallback)


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
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_name = transformer_name
        self.num_emotions = num_emotions
        self.encoders = nn.ModuleDict(
            {
                task: AutoModel.from_pretrained(
                    transformer_name, local_files_only=local_files_only
                )
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
    local_files_only: bool = False,
) -> tuple[nn.Module, nn.Module]:
    emotion_encoder = AutoModel.from_pretrained(
        transformer_name, local_files_only=local_files_only
    )
    intensity_encoder = AutoModel.from_pretrained(
        transformer_name, local_files_only=local_files_only
    )
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
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        (
            self.emotion_encoder,
            self.intensity_encoder,
        ) = load_transferred_encoder_pair(
            transformer_name, transfer_checkpoint, local_files_only
        )
        hidden = self.emotion_encoder.config.hidden_size
        self.hidden_size = hidden
        self.max_chunks = max_chunks
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.chunk_positions = nn.Embedding(max_chunks, hidden)
        self.speaker_projection = nn.Linear(2, hidden, bias=False)
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

    def set_trainable_layers(self, top_layers: int | None) -> None:
        """Freeze the encoder or expose only its top transformer layers.

        ``None`` unfreezes the complete affective encoder, ``0`` freezes it,
        and a positive value trains that many top layers in both transferred
        BERT backbones together with the target-domain fusion modules.
        """
        if top_layers is not None and top_layers < 0:
            raise ValueError("top_layers cannot be negative.")
        for parameter in self.parameters():
            parameter.requires_grad = top_layers is None
        if top_layers in (None, 0):
            return

        fusion_modules = (
            self.gate,
            self.chunk_positions,
            self.speaker_projection,
            self.chunk_encoder,
        )
        for module in fusion_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

        for backbone in (self.emotion_encoder, self.intensity_encoder):
            encoder = getattr(backbone, "encoder", None)
            layers = getattr(encoder, "layer", None)
            if layers is None:
                for parameter in backbone.parameters():
                    parameter.requires_grad = True
                continue
            selected = list(layers)[-min(top_layers, len(layers)) :]
            for layer in selected:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
        speaker_features: torch.Tensor | None = None,
        trajectory_features: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
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
        combined = combined + self.chunk_positions(positions)
        if speaker_features is not None:
            if speaker_features.shape != (batch, chunks, 2):
                raise ValueError("Speaker features must have shape [batch, chunks, 2].")
            combined = combined + self.speaker_projection(speaker_features)
        combined = self.dropout(combined)
        encoded = self.chunk_encoder(
            combined, src_key_padding_mask=~chunk_mask.bool()
        )
        overall = masked_mean(encoded, chunk_mask, dimension=1)
        if not return_trajectory:
            return overall
        if trajectory_features is None or trajectory_features.shape != (
            batch,
            chunks,
            2,
        ):
            raise ValueError(
                "Trajectory features must have shape [batch, chunks, 2]."
            )
        early = weighted_chunk_mean(
            encoded, trajectory_features[:, :, 0], chunk_mask, overall
        )
        late = weighted_chunk_mean(
            encoded, trajectory_features[:, :, 1], chunk_mask, overall
        )
        return {"overall": overall, "early": early, "late": late}


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
        num_outcome_classes: int = 4,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if modality not in self.MODALITIES:
            raise ValueError(f"Unsupported modality: {modality}")
        self.modality = modality
        self.use_initial_intensity = use_initial_intensity
        if num_outcome_classes < 2:
            raise ValueError("Temporal outcomes require at least two classes.")
        self.num_outcome_classes = num_outcome_classes
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
                local_files_only=local_files_only,
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
        self.final_head = nn.Linear(output_size // 2, num_outcome_classes)
        self.drop_head = nn.Linear(output_size // 2, num_outcome_classes)

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


def cumulative_ordinal_targets(
    labels: torch.Tensor, num_classes: int = 4
) -> torch.Tensor:
    """Encode labels 1..K as cumulative targets [y>1, ..., y>K-1]."""
    if labels.ndim != 1:
        raise ValueError("Ordinal labels must be a one-dimensional tensor.")
    if labels.numel() and (
        int(labels.min()) < 1 or int(labels.max()) > num_classes
    ):
        raise ValueError(f"Ordinal labels must be in 1..{num_classes}.")
    thresholds = torch.arange(1, num_classes, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).to(torch.float)


def cumulative_ordinal_predictions(
    logits: torch.Tensor,
    initial_intensity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode cumulative logits and optionally enforce final < initial."""
    if logits.ndim != 2:
        raise ValueError("Ordinal logits must have shape [batch, thresholds].")
    predicted = 1 + (torch.sigmoid(logits) >= 0.5).sum(dim=1)
    if initial_intensity is not None:
        if initial_intensity.shape != predicted.shape:
            raise ValueError("Initial intensity must match the prediction batch.")
        predicted = torch.minimum(predicted, initial_intensity - 1)
    return predicted.clamp(min=1, max=logits.shape[1] + 1)


class CumulativeOrdinalHead(nn.Module):
    """A proportional-odds head with monotonically ordered thresholds."""

    def __init__(self, input_size: int, num_classes: int = 4) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("An ordinal head requires at least two classes.")
        self.score = nn.Linear(input_size, 1)
        self.first_threshold = nn.Parameter(torch.tensor(-1.0))
        self.threshold_steps = nn.Parameter(torch.zeros(num_classes - 2))

    def thresholds(self) -> torch.Tensor:
        if self.threshold_steps.numel() == 0:
            return self.first_threshold.unsqueeze(0)
        increments = nn.functional.softplus(self.threshold_steps)
        return torch.cat(
            [
                self.first_threshold.unsqueeze(0),
                self.first_threshold + torch.cumsum(increments, dim=0),
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.score(values) - self.thresholds().unsqueeze(0)


OUTCOME_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (4, 1),
)


class EmotionConditionedClassificationHead(nn.Module):
    """Shared classifier plus a small residual expert for each emotion family."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        emotion_vocabulary_size: int,
    ) -> None:
        super().__init__()
        if emotion_vocabulary_size < 1:
            raise ValueError("Emotion-conditioned heads require a vocabulary.")
        self.shared = nn.Linear(input_size, output_size)
        self.expert_weight = nn.Parameter(
            torch.zeros(emotion_vocabulary_size, output_size, input_size)
        )
        self.expert_bias = nn.Parameter(
            torch.zeros(emotion_vocabulary_size, output_size)
        )

    def forward(
        self, values: torch.Tensor, emotion_ids: torch.Tensor
    ) -> torch.Tensor:
        if emotion_ids.ndim != 1 or emotion_ids.shape[0] != values.shape[0]:
            raise ValueError("Emotion ids must have one value per batch item.")
        expert_weight = self.expert_weight[emotion_ids]
        expert_bias = self.expert_bias[emotion_ids]
        residual = torch.einsum("boi,bi->bo", expert_weight, values)
        return self.shared(values) + residual + expert_bias


class EmotionConditionedOutcomeModel(nn.Module):
    """Role-aware full-context final-intensity model with metadata conditioning."""

    def __init__(
        self,
        transformer_name: str,
        transfer_checkpoint: dict[str, Any] | None,
        emotion_vocabulary_size: int,
        problem_vocabulary_size: int,
        dropout: float = 0.4,
        max_chunks: int = 32,
        metadata_size: int = 64,
        strategy_vocabulary_size: int = 0,
        strategy_numeric_size: int = 0,
        strategy_hidden_size: int = 256,
        use_speaker_features: bool = False,
        use_trajectory: bool = False,
        use_strategy: bool = False,
        use_auxiliary_regression: bool = True,
        use_joint_pair_head: bool = True,
        use_emotion_conditioned_heads: bool = True,
    ) -> None:
        super().__init__()
        self.text_encoder = AffectiveTextEncoder(
            transformer_name=transformer_name,
            transfer_checkpoint=transfer_checkpoint,
            dropout=dropout,
            max_chunks=max_chunks,
        )
        hidden = self.text_encoder.hidden_size
        self.use_speaker_features = use_speaker_features
        self.use_trajectory = use_trajectory
        self.use_strategy = use_strategy
        self.use_auxiliary_regression = use_auxiliary_regression
        self.use_joint_pair_head = use_joint_pair_head
        self.use_emotion_conditioned_heads = use_emotion_conditioned_heads
        self.trajectory_projection: nn.Module | None = None
        if use_trajectory:
            self.trajectory_projection = nn.Sequential(
                nn.Linear(hidden * 4, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        self.strategy_encoder: StrategyEncoder | None = None
        if use_strategy:
            if strategy_vocabulary_size < 2 or strategy_numeric_size < 1:
                raise ValueError("Strategy-aware models require strategy dimensions.")
            self.strategy_encoder = StrategyEncoder(
                vocabulary_size=strategy_vocabulary_size,
                numeric_size=strategy_numeric_size,
                hidden_size=strategy_hidden_size,
                dropout=dropout,
            )
            self.strategy_to_text = nn.Linear(strategy_hidden_size, hidden)
            self.strategy_gate = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.Sigmoid(),
            )
        self.emotion_embedding = nn.Embedding(
            emotion_vocabulary_size, metadata_size, padding_idx=0
        )
        self.problem_embedding = nn.Embedding(
            problem_vocabulary_size, metadata_size, padding_idx=0
        )
        self.initial_embedding = nn.Embedding(6, metadata_size, padding_idx=0)
        self.conditioning = nn.Sequential(
            nn.Linear(metadata_size * 3, hidden * 2),
            nn.Tanh(),
        )
        self.normalisation = nn.LayerNorm(hidden)
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Direct class heads optimise the metric used for model selection.  The
        # ordinal and regression heads remain as auxiliary objectives so that
        # the representation still respects distance between intensity levels.
        head_type = (
            EmotionConditionedClassificationHead
            if use_emotion_conditioned_heads
            else None
        )
        self.final_class_head = (
            head_type(hidden // 2, 4, emotion_vocabulary_size)
            if head_type is not None
            else nn.Linear(hidden // 2, 4)
        )
        self.drop_class_head = (
            head_type(hidden // 2, 4, emotion_vocabulary_size)
            if head_type is not None
            else nn.Linear(hidden // 2, 4)
        )
        self.joint_pair_head: nn.Module | None = None
        if use_joint_pair_head:
            self.joint_pair_head = (
                head_type(hidden // 2, len(OUTCOME_PAIRS), emotion_vocabulary_size)
                if head_type is not None
                else nn.Linear(hidden // 2, len(OUTCOME_PAIRS))
            )
        self.final_head = CumulativeOrdinalHead(hidden // 2, num_classes=4)
        self.drop_regression_head = (
            nn.Linear(hidden // 2, 1) if use_auxiliary_regression else None
        )

    def set_text_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = trainable

    def set_text_encoder_trainable_layers(self, top_layers: int | None) -> None:
        self.text_encoder.set_trainable_layers(top_layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text_output = self.text_encoder(
            batch["input_ids"],
            batch["attention_mask"],
            batch["chunk_mask"],
            speaker_features=(
                batch["speaker_features"] if self.use_speaker_features else None
            ),
            trajectory_features=(
                batch["trajectory_features"] if self.use_trajectory else None
            ),
            return_trajectory=self.use_trajectory,
        )
        if self.use_trajectory:
            assert isinstance(text_output, dict)
            assert self.trajectory_projection is not None
            early = text_output["early"]
            late = text_output["late"]
            text = self.trajectory_projection(
                torch.cat([text_output["overall"], early, late, late - early], dim=-1)
            )
        else:
            assert isinstance(text_output, torch.Tensor)
            text = text_output
        if self.strategy_encoder is not None:
            strategy = self.strategy_encoder(
                batch["strategy_ids"],
                batch["strategy_positions"],
                batch["strategy_mask"],
                batch["strategy_numeric"],
            )
            projected_strategy = self.strategy_to_text(strategy)
            gate = self.strategy_gate(torch.cat([text, projected_strategy], dim=-1))
            text = gate * text + (1.0 - gate) * projected_strategy
        metadata = torch.cat(
            [
                self.emotion_embedding(batch["emotion_id"]),
                self.problem_embedding(batch["problem_id"]),
                self.initial_embedding(batch["initial_intensity"]),
            ],
            dim=-1,
        )
        gamma, beta = self.conditioning(metadata).chunk(2, dim=-1)
        conditioned = self.normalisation(text * (1.0 + gamma) + beta)
        shared = self.shared(conditioned)
        emotion_ids = batch["emotion_id"]
        if self.use_emotion_conditioned_heads:
            final_logits = self.final_class_head(shared, emotion_ids)
            drop_logits = self.drop_class_head(shared, emotion_ids)
        else:
            final_logits = self.final_class_head(shared)
            drop_logits = self.drop_class_head(shared)
        outputs = {
            "final_logits": final_logits,
            "drop_logits": drop_logits,
            "final_ordinal_logits": self.final_head(shared),
        }
        if self.joint_pair_head is not None:
            outputs["joint_pair_logits"] = (
                self.joint_pair_head(shared, emotion_ids)
                if self.use_emotion_conditioned_heads
                else self.joint_pair_head(shared)
            )
        if self.drop_regression_head is not None:
            outputs["drop_regression"] = 1.0 + 3.0 * torch.sigmoid(
                self.drop_regression_head(shared).squeeze(-1)
            )
        return outputs


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
