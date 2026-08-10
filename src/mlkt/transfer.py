from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

SENTIMENTS = ("negative", "neutral", "positive")
EMOTION_COLUMNS = ("emotion1", "emotion2", "emotion3")
INTENSITY_COLUMNS = ("intensity1", "intensity2", "intensity3")
INTENSIFIER_VOCABULARY = {
    "absolutely",
    "barely",
    "completely",
    "deeply",
    "especially",
    "extremely",
    "highly",
    "incredibly",
    "particularly",
    "quite",
    "really",
    "seriously",
    "slightly",
    "so",
    "somewhat",
    "terribly",
    "totally",
    "very",
}
EXPLICIT_EMOTION_WORDS = {
    "acceptance",
    "anger",
    "angry",
    "anxiety",
    "anxious",
    "depression",
    "depressed",
    "disgust",
    "disgusted",
    "fear",
    "guilt",
    "happy",
    "jealousy",
    "joy",
    "nervous",
    "nervousness",
    "pain",
    "sad",
    "sadness",
    "shame",
    "surprise",
    "surprised",
}


def _normalise_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _normalise_sentiment(value: Any) -> str:
    sentiment = _normalise_label(value)
    aliases = {
        "neg": "negative",
        "neu": "neutral",
        "pos": "positive",
        "positve": "positive",
    }
    sentiment = aliases.get(sentiment, sentiment)
    if sentiment not in SENTIMENTS:
        raise ValueError(f"Unsupported sentiment label: {value!r}")
    return sentiment


def _as_intensity(value: Any) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        intensity = int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid intensity value: {value!r}") from error
    if intensity not in {1, 2, 3}:
        raise ValueError(f"Intensity must be in {{1, 2, 3}}, got {intensity}.")
    return intensity


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_split_hash(frame: pd.DataFrame) -> str:
    values = (
        frame[["conversation_id", "split"]]
        .drop_duplicates()
        .sort_values("conversation_id")
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.sha256(values).hexdigest()


def attach_segment_conversation_ids(
    frame: pd.DataFrame,
    dataset: str,
    require_pairs: bool,
) -> pd.DataFrame:
    """Recover conversation IDs from ordered start/end segment files."""
    if "segment" not in frame:
        raise ValueError("Segment data must contain a 'segment' column.")
    segments = frame["segment"].astype(str).str.lower().str.strip()
    if not segments.isin({"start", "end"}).all():
        invalid = sorted(segments[~segments.isin({"start", "end"})].unique())
        raise ValueError(f"Unsupported segment labels: {invalid}")
    group_numbers = segments.eq("start").cumsum()
    if (group_numbers == 0).any():
        raise ValueError("The segment file contains rows before the first start row.")

    result = frame.copy()
    result["conversation_id"] = [
        f"{dataset}_{int(group_number) - 1:04d}" for group_number in group_numbers
    ]
    patterns = result.groupby("conversation_id", sort=False)["segment"].agg(
        lambda values: tuple(values.astype(str).str.lower())
    )
    allowed = {("start", "end")} if require_pairs else {("start",), ("start", "end")}
    invalid_patterns = patterns[~patterns.isin(allowed)]
    if not invalid_patterns.empty:
        raise ValueError(
            "Invalid start/end ordering for conversations: "
            f"{invalid_patterns.head().to_dict()}"
        )
    return result


def _clean_da_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Utterances",
        "sentiment",
        "emotion1",
        "intensity1",
        "segment",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DA input is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["Utterances"] = (
        result["Utterances"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    result["sentiment"] = result["sentiment"].map(_normalise_sentiment)
    for emotion_column, intensity_column in zip(
        EMOTION_COLUMNS, INTENSITY_COLUMNS
    ):
        if emotion_column not in result:
            result[emotion_column] = ""
        if intensity_column not in result:
            result[intensity_column] = pd.NA
        result[emotion_column] = result[emotion_column].map(_normalise_label)
        cleaned_intensities: list[int | None] = []
        for emotion, intensity in zip(
            result[emotion_column], result[intensity_column]
        ):
            cleaned_intensities.append(
                _as_intensity(intensity) if emotion else None
            )
        result[intensity_column] = pd.array(cleaned_intensities, dtype="Int64")
    if (result["Utterances"].str.len() < 3).any():
        raise ValueError("DA input contains empty or unusably short utterances.")
    return result


def assign_source_splits(
    frame: pd.DataFrame,
    train_fraction: float = 0.80,
    seed: int = 42,
) -> pd.DataFrame:
    """Split MEISD segments once per recovered conversation."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")
    conversations = (
        frame.groupby("conversation_id", sort=True)
        .agg(sentiment=("sentiment", lambda values: values.mode().iloc[0]))
        .reset_index()
    )
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for _, group in conversations.groupby("sentiment", sort=True):
        identifiers = group["conversation_id"].tolist()
        rng.shuffle(identifiers)
        n_train = max(1, min(len(identifiers) - 1, round(len(identifiers) * train_fraction)))
        for index, conversation_id in enumerate(identifiers):
            assignments[conversation_id] = "train" if index < n_train else "validation"
    result = frame.copy()
    result["split"] = result["conversation_id"].map(assignments)
    if result.groupby("conversation_id")["split"].nunique().max() != 1:
        raise AssertionError("Source conversation leakage detected.")
    return result


@dataclass(frozen=True)
class TransferPreparationResult:
    esconv_rows: int
    meisd_rows: int
    esconv_train_conversations: int
    meisd_train_conversations: int
    meisd_validation_conversations: int
    esconv_split_hash: str
    meisd_split_hash: str
    input_hashes: dict[str, str]


def prepare_transfer_inputs(
    esconv_path: str | Path,
    meisd_path: str | Path,
    esconv_checkpoints_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> TransferPreparationResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    esconv = attach_segment_conversation_ids(
        _clean_da_frame(pd.read_csv(esconv_path)),
        dataset="esconv",
        require_pairs=True,
    )
    meisd = attach_segment_conversation_ids(
        _clean_da_frame(pd.read_csv(meisd_path)),
        dataset="meisd",
        require_pairs=False,
    )

    checkpoint_frame = pd.read_csv(esconv_checkpoints_path)
    split_map = (
        checkpoint_frame[["conversation_id", "split"]]
        .drop_duplicates()
        .set_index("conversation_id")["split"]
    )
    esconv["split"] = esconv["conversation_id"].map(split_map)
    if esconv["split"].isna().any():
        missing = esconv.loc[esconv["split"].isna(), "conversation_id"].unique()
        # The 150 incomplete ESConv outcomes are allowed for style extraction only
        # when they are not part of the supervised checkpoint table.
        esconv.loc[esconv["conversation_id"].isin(missing), "split"] = "excluded"
    meisd = assign_source_splits(meisd, seed=seed)

    esconv_output = output / "esconv_transfer_prepared.csv"
    meisd_output = output / "meisd_transfer_prepared.csv"
    esconv.to_csv(esconv_output, index=False)
    meisd.to_csv(meisd_output, index=False)

    result = TransferPreparationResult(
        esconv_rows=len(esconv),
        meisd_rows=len(meisd),
        esconv_train_conversations=esconv.loc[
            esconv["split"] == "train", "conversation_id"
        ].nunique(),
        meisd_train_conversations=meisd.loc[
            meisd["split"] == "train", "conversation_id"
        ].nunique(),
        meisd_validation_conversations=meisd.loc[
            meisd["split"] == "validation", "conversation_id"
        ].nunique(),
        esconv_split_hash=_stable_split_hash(esconv[esconv["split"] != "excluded"]),
        meisd_split_hash=_stable_split_hash(meisd),
        input_hashes={
            "esconv_da": _file_sha256(esconv_path),
            "meisd_da": _file_sha256(meisd_path),
            "esconv_checkpoints": _file_sha256(esconv_checkpoints_path),
        },
    )
    with (output / "transfer_preparation_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(asdict(result), handle, indent=2, sort_keys=True)
    return result


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _tfidf_keywords(texts: Sequence[str], top_n: int = 30) -> list[str]:
    documents = [_tokens(text) for text in texts]
    if not documents:
        return []
    document_frequency: Counter[str] = Counter()
    term_frequency: Counter[str] = Counter()
    for document in documents:
        term_frequency.update(document)
        document_frequency.update(set(document))
    scores = {
        token: count
        * (math.log((1 + len(documents)) / (1 + document_frequency[token])) + 1)
        for token, count in term_frequency.items()
        if len(token) > 2
    }
    return [
        token
        for token, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            :top_n
        ]
    ]


def _sentence_starters(texts: Sequence[str], top_n: int = 15) -> list[str]:
    starters: Counter[str] = Counter()
    for text in texts:
        words = _tokens(text)
        if words:
            starters[" ".join(words[: min(3, len(words))])] += 1
    return [starter for starter, _ in starters.most_common(top_n)]


def extract_style_patterns(
    prepared_esconv: pd.DataFrame,
) -> dict[str, Any]:
    """Extract target-domain patterns strictly from ESConv training rows."""
    train = prepared_esconv[prepared_esconv["split"] == "train"].copy()
    if train.empty:
        raise ValueError("No ESConv training rows are available for style extraction.")
    records: list[dict[str, Any]] = []
    for row in train.itertuples(index=False):
        for emotion_column, intensity_column in zip(
            EMOTION_COLUMNS, INTENSITY_COLUMNS
        ):
            emotion = getattr(row, emotion_column)
            intensity = getattr(row, intensity_column)
            if emotion and not pd.isna(intensity):
                records.append(
                    {
                        "emotion": emotion,
                        "intensity": int(intensity),
                        "sentiment": row.sentiment,
                        "text": row.Utterances,
                    }
                )
    if not records:
        raise ValueError("ESConv training split has no labelled style records.")
    record_frame = pd.DataFrame(records)
    patterns: dict[str, Any] = {
        "metadata": {
            "source_split": "train",
            "conversation_count": train["conversation_id"].nunique(),
            "row_count": len(train),
            "split_hash": _stable_split_hash(train),
        },
        "groups": {},
    }
    for key, group in record_frame.groupby(
        ["emotion", "intensity", "sentiment"], sort=True
    ):
        texts = group["text"].tolist()
        words = [word for text in texts for word in _tokens(text)]
        intensifiers = [
            token
            for token, _ in Counter(words).most_common()
            if token in INTENSIFIER_VOCABULARY
        ][:15]
        encoded_key = "|".join(map(str, key))
        patterns["groups"][encoded_key] = {
            "emotion": key[0],
            "intensity": int(key[1]),
            "sentiment": key[2],
            "sample_count": len(texts),
            "avg_length": sum(len(_tokens(text)) for text in texts) / len(texts),
            "question_ratio": sum("?" in text for text in texts) / len(texts),
            "exclamation_ratio": sum("!" in text for text in texts) / len(texts),
            "keywords": _tfidf_keywords(texts),
            "sentence_starters": _sentence_starters(texts),
            "intensifiers": intensifiers,
            "examples": texts[:3],
        }
    return patterns


def save_style_patterns(patterns: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(patterns, handle, indent=2, ensure_ascii=False, sort_keys=True)


def emotion_bundle(row: pd.Series) -> tuple[tuple[str, int], ...]:
    values: list[tuple[str, int]] = []
    for emotion_column, intensity_column in zip(
        EMOTION_COLUMNS, INTENSITY_COLUMNS
    ):
        emotion = _normalise_label(row.get(emotion_column))
        intensity = _as_intensity(row.get(intensity_column)) if emotion else None
        if emotion and intensity is not None:
            values.append((emotion, intensity))
    return tuple(values)


def filter_meisd_for_esconv(
    meisd: pd.DataFrame,
    esconv_patterns: dict[str, Any],
    min_samples: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_emotions = {
        group["emotion"] for group in esconv_patterns["groups"].values()
    }
    source_reference = (
        meisd[meisd["split"] == "train"] if "split" in meisd else meisd
    )
    source_counts: Counter[tuple[str, int]] = Counter()
    for _, row in source_reference.iterrows():
        source_counts.update(emotion_bundle(row))
    compatible_pairs = {
        pair
        for pair, count in source_counts.items()
        if pair[0] in target_emotions and count >= min_samples
    }
    keep = meisd.apply(
        lambda row: any(pair in compatible_pairs for pair in emotion_bundle(row)),
        axis=1,
    )
    filtered = meisd.loc[keep].copy()
    report = {
        "original_rows": len(meisd),
        "filtered_rows": len(filtered),
        "removed_rows": int((~keep).sum()),
        "compatibility_reference_split": (
            "train" if "split" in meisd else "all"
        ),
        "target_emotions": sorted(target_emotions),
        "compatible_pairs": [
            {"emotion": emotion, "intensity": intensity}
            for emotion, intensity in sorted(compatible_pairs)
        ],
    }
    return filtered, report


def _merge_bundle_patterns(
    patterns: dict[str, Any],
    bundle: Sequence[tuple[str, int]],
    sentiment: str,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    groups = patterns["groups"]
    for emotion, intensity in bundle:
        exact = groups.get(f"{emotion}|{intensity}|{sentiment}")
        if exact:
            candidates.append(exact)
            continue
        candidates.extend(
            group
            for group in groups.values()
            if group["emotion"] == emotion and group["intensity"] == intensity
        )
    if not candidates:
        candidates = list(groups.values())
    merged: dict[str, Any] = {
        "avg_length": sum(group["avg_length"] for group in candidates)
        / len(candidates),
    }
    for name, limit in (
        ("keywords", 30),
        ("sentence_starters", 15),
        ("intensifiers", 15),
        ("examples", 6),
    ):
        values: list[str] = []
        for group in candidates:
            values.extend(group.get(name, []))
        merged[name] = list(dict.fromkeys(values))[:limit]
    return merged


class TextGenerator(Protocol):
    name: str

    def generate(self, prompt: str, max_tokens: int, seed: int) -> str:
        ...


class LlamaCppGenerator:
    name = "llama-2-7b-chat-gguf"

    def __init__(
        self,
        model_path: str | Path,
        context_size: int = 2048,
        threads: int = 8,
        gpu_layers: int = -1,
        batch_size: int = 1024,
        require_gpu: bool = False,
    ) -> None:
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as error:
            raise RuntimeError(
                "Llama generation requires llama-cpp-python. "
                "Install the 'augmentation' optional dependency."
            ) from error
        model_path = Path(model_path).resolve()
        if not model_path.is_file():
            raise ValueError(f"Model path does not exist: {model_path}")
        system_info = llama_cpp.llama_print_system_info().decode(
            errors="replace"
        )
        upper_info = system_info.upper()
        native_library_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
        native_libraries = (
            {path.name.lower() for path in native_library_dir.iterdir()}
            if native_library_dir.is_dir()
            else set()
        )
        if "VULKAN" in upper_info or any(
            "vulkan" in name for name in native_libraries
        ):
            backend = "vulkan"
        elif (
            "ROCM" in upper_info
            or "HIP" in upper_info
            or any("hip" in name for name in native_libraries)
        ):
            backend = "rocm"
        elif "CUDA" in upper_info or any(
            "cuda" in name for name in native_libraries
        ):
            backend = "cuda"
        else:
            backend = "cpu"
        supports_offload = bool(llama_cpp.llama_supports_gpu_offload())
        if require_gpu and (backend == "cpu" or not supports_offload):
            raise RuntimeError(
                "GPU generation was required, but llama-cpp-python has no "
                "GPU-offload backend. Install the Vulkan wheel and retry. "
                f"System info: {system_info}"
            )
        if require_gpu and gpu_layers == 0:
            raise ValueError("--require-gpu cannot be combined with --gpu-layers 0.")
        print(
            "llama.cpp backend: "
            f"{backend}; GPU offload: {supports_offload}; "
            f"requested GPU layers: {gpu_layers}"
        )
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=context_size,
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=gpu_layers,
            n_batch=batch_size,
            n_ubatch=min(batch_size, 512),
            flash_attn=True,
            offload_kqv=True,
            verbose=False,
        )
        model_layers = int(llama_cpp.llama_model_n_layer(self._model.model))
        gpu_layers_effective = (
            model_layers if gpu_layers < 0 else min(gpu_layers, model_layers)
        )
        if require_gpu and gpu_layers_effective < model_layers:
            raise RuntimeError(
                "Full GPU offload was required, but fewer than all model "
                f"layers were requested ({gpu_layers_effective}/{model_layers})."
            )
        self._metadata = {
            "backend": backend,
            "gpu_offload_supported": supports_offload,
            "gpu_layers_requested": gpu_layers,
            "model_layers": model_layers,
            "gpu_layers_effective": gpu_layers_effective,
            "batch_size": batch_size,
            "context_size": context_size,
            "threads": threads,
            "model_path": str(model_path),
            "system_info": system_info,
            "native_libraries": sorted(native_libraries),
        }

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def generate(self, prompt: str, max_tokens: int, seed: int) -> str:
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.8,
            top_p=0.9,
            seed=seed,
            stop=["Original:", "Rules:", "\n\n\n"],
        )
        return str(output["choices"][0]["text"]).strip()


class DeterministicMockGenerator:
    """Non-semantic generator used only by tests and pipeline smoke checks."""

    name = "deterministic-mock"

    def metadata(self) -> dict[str, Any]:
        return {"backend": "mock", "gpu_offload_supported": False}

    def generate(self, prompt: str, max_tokens: int, seed: int) -> str:
        match = re.search(r'Original: "(.*?)"', prompt, flags=re.DOTALL)
        original = match.group(1) if match else "I need some help with this."
        return f"{original} I keep thinking about what happened."


def _augmentation_prompt(
    text: str,
    bundle: Sequence[tuple[str, int]],
    sentiment: str,
    pattern: dict[str, Any],
) -> str:
    state = ", ".join(f"{emotion} at level {intensity}" for emotion, intensity in bundle)
    examples = "\n".join(f"- {value}" for value in pattern.get("examples", []))
    keywords = ", ".join(pattern.get("keywords", [])[:10])
    intensifiers = ", ".join(pattern.get("intensifiers", [])[:8])
    target_length = round(float(pattern.get("avg_length", 50)))
    return f"""Rewrite this message as someone talking to a therapist.
Show their emotional state through situations, physical reactions, and thoughts.

Original: "{text}"

Latent annotation bundle (do not name it in the output): {state}
Sentiment to preserve: {sentiment}
Target length: approximately {target_length} words
Target-domain lexical cues: {keywords}
Target-domain intensity cues: {intensifiers}
Authentic target-domain examples:
{examples}

Rules:
- preserve all emotions and their relative intensities
- do not explicitly name emotions or intensity values
- do not add labels, explanations, formatting, or stage directions
- preserve {sentiment} sentiment and the original meaning
- return only the rewritten support-seeking message

Message:"""


def validate_generated_text(text: str) -> tuple[bool, str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'")
    if len(cleaned.split()) < 3:
        return False, cleaned
    lowered = cleaned.lower()
    invalid_phrases = (
        "intensity",
        "rewritten",
        "primary emotion",
        "in this message",
        "note that",
        "as an ai",
    )
    if any(phrase in lowered for phrase in invalid_phrases) or "*" in cleaned:
        return False, cleaned
    tokens = set(_tokens(cleaned))
    if tokens & EXPLICIT_EMOTION_WORDS:
        return False, cleaned
    return True, cleaned


def _quality_score(
    original: str, generated: str, pattern: dict[str, Any]
) -> float:
    target = max(float(pattern.get("avg_length", 50)), 1.0)
    length_score = max(0.0, 1.0 - abs(len(_tokens(generated)) - target) / target)
    keywords = pattern.get("keywords", [])[:10]
    keyword_score = (
        sum(keyword in generated.lower() for keyword in keywords) / len(keywords)
        if keywords
        else 0.5
    )
    changed = float(original.strip().lower() != generated.strip().lower())
    return round(0.45 * length_score + 0.35 * keyword_score + 0.20 * changed, 6)


def build_augmentation_plan(
    prepared_meisd: pd.DataFrame,
    patterns: dict[str, Any],
    seed: int = 42,
    min_compatible_samples: int = 5,
    max_aug_per_group: int = 600,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Create the deterministic source-row and seed plan before generation."""
    filtered, filter_report = filter_meisd_for_esconv(
        prepared_meisd,
        patterns,
        min_samples=min_compatible_samples,
    )
    train = filtered[filtered["split"] == "train"].copy()
    validation = filtered[filtered["split"] == "validation"].copy()
    train["emotion_bundle_key"] = train.apply(
        lambda row: "|".join(
            f"{emotion}:{intensity}" for emotion, intensity in emotion_bundle(row)
        ),
        axis=1,
    )
    group_counts = train["emotion_bundle_key"].value_counts()
    if group_counts.empty:
        raise ValueError("No compatible MEISD training rows remain after filtering.")
    target_count = int(group_counts.max())
    rng = random.Random(seed)
    plan: list[dict[str, Any]] = []
    generation_index = 0
    for key, count in group_counts.items():
        needed = min(target_count - int(count), max_aug_per_group)
        if needed <= 0:
            continue
        group = train[train["emotion_bundle_key"] == key]
        source_indices = group.index.tolist()
        for _ in range(needed):
            source_index = int(rng.choice(source_indices))
            source = train.loc[source_index]
            plan.append(
                {
                    "generation_index": generation_index,
                    "generation_seed": seed + generation_index,
                    "emotion_bundle_key": key,
                    "source_index": source_index,
                    "source_conversation_id": str(source["conversation_id"]),
                    "source_text_sha256": hashlib.sha256(
                        str(source["Utterances"]).encode("utf-8")
                    ).hexdigest(),
                }
            )
            generation_index += 1

    report = {
        **filter_report,
        "seed": seed,
        "original_train_rows": len(train),
        "planned_generation_rows": len(plan),
        "validation_rows": len(validation),
        "min_compatible_samples": min_compatible_samples,
        "max_aug_per_group": max_aug_per_group,
        "target_group_count": target_count,
        "style_split_hash": patterns["metadata"]["split_hash"],
    }
    return train, validation, plan, report


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _plan_signature(plan: Sequence[dict[str, Any]], report: dict[str, Any]) -> str:
    identity = {
        "plan": list(plan),
        "seed": report["seed"],
        "style_split_hash": report["style_split_hash"],
        "max_aug_per_group": report["max_aug_per_group"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_progress(
    path: Path, plan_signature: str
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    valid_lines: list[str] = []
    truncated_final_line = False
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A process can be killed in the middle of the final append.
                if line_number == len(lines):
                    truncated_final_line = True
                    break
                raise
            if record.get("plan_signature") != plan_signature:
                raise RuntimeError(
                    "Existing augmentation progress belongs to a different plan. "
                    "Use another output directory or remove the progress file."
                )
            index = int(record["generation_index"])
            if index in completed:
                raise RuntimeError(f"Duplicate generation index in progress: {index}")
            completed[index] = record["row"]
            valid_lines.append(line)
    if truncated_final_line:
        with path.open("w", encoding="utf-8") as handle:
            handle.writelines(valid_lines)
            handle.flush()
            os.fsync(handle.fileno())
    return completed


def _generator_metadata(generator: TextGenerator) -> dict[str, Any]:
    metadata = getattr(generator, "metadata", None)
    return dict(metadata()) if callable(metadata) else {}


def _execute_augmentation_plan(
    train: pd.DataFrame,
    patterns: dict[str, Any],
    generator: TextGenerator,
    plan: Sequence[dict[str, Any]],
    plan_signature: str,
    progress_path: Path | None = None,
    checkpoint_every: int = 25,
    resume: bool = False,
    max_tokens: int = 150,
) -> tuple[list[dict[str, Any]], int, int]:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")
    completed: dict[int, dict[str, Any]] = {}
    resume_count = 0
    if progress_path is not None and progress_path.exists():
        if not resume:
            raise RuntimeError(
                f"Progress file already exists: {progress_path}. "
                "Pass --resume or use another output directory."
            )
        completed = _read_progress(progress_path, plan_signature)
        resume_count = 1 if completed else 0
    invalid_generations = sum(
        not bool(row.get("generation_valid")) for row in completed.values()
    )
    pending = [item for item in plan if item["generation_index"] not in completed]

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(
            pending,
            total=len(plan),
            initial=len(completed),
            unit="generation",
            desc="Augmenting MEISD",
        )
    except ImportError:
        iterator = pending

    progress_handle = None
    try:
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_handle = progress_path.open("a", encoding="utf-8")
        since_flush = 0
        started = time.perf_counter()
        for item in iterator:
            source = train.loc[item["source_index"]]
            bundle = emotion_bundle(source)
            style = _merge_bundle_patterns(patterns, bundle, source["sentiment"])
            prompt = _augmentation_prompt(
                source["Utterances"], bundle, source["sentiment"], style
            )
            raw = generator.generate(
                prompt,
                max_tokens=max_tokens,
                seed=item["generation_seed"],
            )
            valid, cleaned = validate_generated_text(raw)
            if not valid:
                invalid_generations += 1
                cleaned = source["Utterances"]
            row = source.drop(labels=["emotion_bundle_key"]).to_dict()
            row.update(
                {
                    "Utterances": cleaned,
                    "original": source["Utterances"],
                    "source_conversation_id": source["conversation_id"],
                    "augmented": True,
                    "generation_valid": valid,
                    "generation_seed": item["generation_seed"],
                    "generator": generator.name,
                    "quality": _quality_score(source["Utterances"], cleaned, style),
                }
            )
            safe_row = _json_safe(row)
            completed[item["generation_index"]] = safe_row
            if progress_handle is not None:
                record = {
                    "plan_signature": plan_signature,
                    "generation_index": item["generation_index"],
                    "row": safe_row,
                }
                progress_handle.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                since_flush += 1
                if since_flush >= checkpoint_every:
                    progress_handle.flush()
                    os.fsync(progress_handle.fileno())
                    since_flush = 0
            if hasattr(iterator, "set_postfix"):
                elapsed = max(time.perf_counter() - started, 1e-9)
                new_count = len(completed) - (len(plan) - len(pending))
                iterator.set_postfix(
                    invalid=invalid_generations,
                    rate=f"{new_count / elapsed:.2f}/s",
                )
    finally:
        if progress_handle is not None:
            progress_handle.flush()
            os.fsync(progress_handle.fileno())
            progress_handle.close()

    if len(completed) != len(plan):
        raise RuntimeError(
            f"Augmentation incomplete: {len(completed)} of {len(plan)} rows."
        )
    generated_rows = [
        completed[int(item["generation_index"])] for item in plan
    ]
    return generated_rows, invalid_generations, resume_count


def augment_source_training(
    prepared_meisd: pd.DataFrame,
    patterns: dict[str, Any],
    generator: TextGenerator,
    seed: int = 42,
    min_compatible_samples: int = 5,
    max_aug_per_group: int = 600,
    progress_path: str | Path | None = None,
    checkpoint_every: int = 25,
    resume: bool = False,
    max_tokens: int = 150,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Balance source train bundles; never augment source validation."""
    train, validation, plan, report = build_augmentation_plan(
        prepared_meisd,
        patterns,
        seed=seed,
        min_compatible_samples=min_compatible_samples,
        max_aug_per_group=max_aug_per_group,
    )
    signature = _plan_signature(plan, report)
    generated_rows, invalid_generations, resume_count = _execute_augmentation_plan(
        train,
        patterns,
        generator,
        plan,
        signature,
        progress_path=Path(progress_path) if progress_path is not None else None,
        checkpoint_every=checkpoint_every,
        resume=resume,
        max_tokens=max_tokens,
    )

    original_train = train.drop(columns=["emotion_bundle_key"]).copy()
    original_train["original"] = original_train["Utterances"]
    original_train["source_conversation_id"] = original_train["conversation_id"]
    original_train["augmented"] = False
    original_train["generation_valid"] = True
    original_train["generation_seed"] = pd.NA
    original_train["generator"] = "original"
    original_train["quality"] = 1.0
    validation["original"] = validation["Utterances"]
    validation["source_conversation_id"] = validation["conversation_id"]
    validation["augmented"] = False
    validation["generation_valid"] = True
    validation["generation_seed"] = pd.NA
    validation["generator"] = "original"
    validation["quality"] = 1.0

    parts = [original_train]
    if generated_rows:
        parts.append(pd.DataFrame(generated_rows))
    parts.append(validation)
    output = pd.concat(parts, ignore_index=True, sort=False)
    if output.loc[output["split"] == "validation", "augmented"].any():
        raise AssertionError("Validation augmentation leakage detected.")
    report = {
        **report,
        "generator": generator.name,
        "generated_train_rows": len(generated_rows),
        "invalid_generations": invalid_generations,
        "resume_count": resume_count,
        "plan_signature": signature,
        "max_tokens": max_tokens,
        "generator_metadata": _generator_metadata(generator),
    }
    return output, report


def to_one_hot_source(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    emotions = sorted(
        {
            emotion
            for _, row in frame.iterrows()
            for emotion, _ in emotion_bundle(row)
        }
    )
    rows: list[dict[str, Any]] = []
    preserved = [
        "Utterances",
        "sentiment",
        "conversation_id",
        "source_conversation_id",
        "split",
        "augmented",
        "generation_valid",
        "generation_seed",
        "generator",
        "quality",
        "original",
    ]
    for _, row in frame.iterrows():
        bundles = dict(emotion_bundle(row))
        output_row = {column: row.get(column) for column in preserved}
        for emotion in emotions:
            output_row[f"emotion__{emotion}"] = int(emotion in bundles)
            output_row[f"intensity__{emotion}"] = bundles.get(emotion, 0)
        rows.append(output_row)
    return pd.DataFrame(rows), emotions


def run_augmentation_pipeline(
    prepared_esconv_path: str | Path,
    prepared_meisd_path: str | Path,
    output_dir: str | Path,
    generator: TextGenerator,
    seed: int = 42,
    min_compatible_samples: int = 5,
    max_aug_per_group: int = 600,
    checkpoint_every: int = 25,
    resume: bool = False,
    max_tokens: int = 150,
    benchmark: int | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    esconv = pd.read_csv(prepared_esconv_path)
    meisd = pd.read_csv(prepared_meisd_path)
    patterns = extract_style_patterns(esconv)
    train, _, plan, plan_report = build_augmentation_plan(
        meisd,
        patterns,
        seed=seed,
        min_compatible_samples=min_compatible_samples,
        max_aug_per_group=max_aug_per_group,
    )
    signature = _plan_signature(plan, plan_report)
    if benchmark is not None:
        if benchmark < 1:
            raise ValueError("benchmark must be at least 1")
        sample_size = min(benchmark, len(plan))
        positions = (
            [0]
            if sample_size == 1
            else [
                round(index * (len(plan) - 1) / (sample_size - 1))
                for index in range(sample_size)
            ]
        )
        sample = [plan[position] for position in positions]
        started = time.perf_counter()
        _execute_augmentation_plan(
            train,
            patterns,
            generator,
            sample,
            signature,
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - started
        seconds_per_generation = elapsed / sample_size
        return {
            **plan_report,
            "mode": "benchmark",
            "benchmark_rows": sample_size,
            "benchmark_seconds": round(elapsed, 3),
            "seconds_per_generation": round(seconds_per_generation, 3),
            "estimated_total_seconds": round(
                seconds_per_generation * len(plan), 3
            ),
            "estimated_total_hours": round(
                seconds_per_generation * len(plan) / 3600, 3
            ),
            "generator": generator.name,
            "generator_metadata": _generator_metadata(generator),
            "plan_signature": signature,
        }

    output.mkdir(parents=True, exist_ok=True)
    plan_manifest_path = output / "augmentation_plan.json"
    state_path = output / "augmentation_state.json"
    progress_path = output / "augmentation_progress.jsonl"
    previous_state: dict[str, Any] = {}
    if state_path.exists():
        with state_path.open(encoding="utf-8") as handle:
            previous_state = json.load(handle)
        if previous_state.get("plan_signature") != signature:
            raise RuntimeError(
                "Existing augmentation state belongs to a different plan."
            )
    resume_count = int(previous_state.get("resume_count", 0))
    if resume and progress_path.exists():
        resume_count += 1
    _atomic_json(
        plan_manifest_path,
        {
            **plan_report,
            "plan_signature": signature,
            "items": plan,
            "generator": generator.name,
            "generator_metadata": _generator_metadata(generator),
        },
    )
    _atomic_json(
        state_path,
        {
            "status": "running",
            "plan_signature": signature,
            "resume_count": resume_count,
            "planned_generation_rows": len(plan),
        },
    )
    augmented, report = augment_source_training(
        meisd,
        patterns,
        generator,
        seed=seed,
        min_compatible_samples=min_compatible_samples,
        max_aug_per_group=max_aug_per_group,
        progress_path=progress_path,
        checkpoint_every=checkpoint_every,
        resume=resume,
        max_tokens=max_tokens,
    )
    report["resume_count"] = resume_count
    one_hot, emotions = to_one_hot_source(augmented)
    manifest = {
        **report,
        "emotions": emotions,
        "prepared_esconv_sha256": _file_sha256(prepared_esconv_path),
        "prepared_meisd_sha256": _file_sha256(prepared_meisd_path),
    }
    _atomic_csv(output / "meisd_target_style_augmented.csv", augmented)
    _atomic_csv(output / "meisd_target_style_onehot.csv", one_hot)
    _atomic_json(output / "esconv_train_style_patterns.json", patterns)
    _atomic_json(output / "augmentation_manifest.json", manifest)
    _atomic_json(
        state_path,
        {
            "status": "complete",
            "plan_signature": signature,
            "resume_count": resume_count,
            "planned_generation_rows": len(plan),
            "completed_generation_rows": len(plan),
        },
    )
    return manifest
