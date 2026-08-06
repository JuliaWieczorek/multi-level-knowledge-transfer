from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

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
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise RuntimeError(
                "Llama generation requires llama-cpp-python. "
                "Install the 'augmentation' optional dependency."
            ) from error
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=context_size,
            n_threads=threads,
            verbose=False,
        )

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


def augment_source_training(
    prepared_meisd: pd.DataFrame,
    patterns: dict[str, Any],
    generator: TextGenerator,
    seed: int = 42,
    min_compatible_samples: int = 5,
    max_aug_per_group: int = 600,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Balance source train bundles; never augment source validation."""
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
    generated_rows: list[dict[str, Any]] = []
    invalid_generations = 0
    generation_index = 0
    for key, count in group_counts.items():
        needed = min(target_count - int(count), max_aug_per_group)
        if needed <= 0:
            continue
        group = train[train["emotion_bundle_key"] == key]
        source_indices = group.index.tolist()
        for _ in range(needed):
            source_index = rng.choice(source_indices)
            source = train.loc[source_index]
            bundle = emotion_bundle(source)
            style = _merge_bundle_patterns(patterns, bundle, source["sentiment"])
            prompt = _augmentation_prompt(
                source["Utterances"], bundle, source["sentiment"], style
            )
            generation_seed = seed + generation_index
            raw = generator.generate(prompt, max_tokens=150, seed=generation_seed)
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
                    "generation_seed": generation_seed,
                    "generator": generator.name,
                    "quality": _quality_score(source["Utterances"], cleaned, style),
                }
            )
            generated_rows.append(row)
            generation_index += 1

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
        **filter_report,
        "generator": generator.name,
        "seed": seed,
        "original_train_rows": len(original_train),
        "generated_train_rows": len(generated_rows),
        "validation_rows": len(validation),
        "invalid_generations": invalid_generations,
        "min_compatible_samples": min_compatible_samples,
        "max_aug_per_group": max_aug_per_group,
        "style_split_hash": patterns["metadata"]["split_hash"],
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
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    esconv = pd.read_csv(prepared_esconv_path)
    meisd = pd.read_csv(prepared_meisd_path)
    patterns = extract_style_patterns(esconv)
    save_style_patterns(patterns, output / "esconv_train_style_patterns.json")
    augmented, report = augment_source_training(
        meisd,
        patterns,
        generator,
        seed=seed,
        min_compatible_samples=min_compatible_samples,
        max_aug_per_group=max_aug_per_group,
    )
    augmented.to_csv(output / "meisd_target_style_augmented.csv", index=False)
    one_hot, emotions = to_one_hot_source(augmented)
    one_hot.to_csv(output / "meisd_target_style_onehot.csv", index=False)
    manifest = {
        **report,
        "emotions": emotions,
        "prepared_esconv_sha256": _file_sha256(prepared_esconv_path),
        "prepared_meisd_sha256": _file_sha256(prepared_meisd_path),
    }
    with (output / "augmentation_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest
