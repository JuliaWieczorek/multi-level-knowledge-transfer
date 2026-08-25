from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .transfer import TextGenerator


_CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "could", "for",
    "from", "have", "i", "in", "is", "it", "just", "me", "my", "of",
    "on", "one", "or", "so", "that", "the", "this", "to", "was", "well",
    "will", "with", "you",
}

_PROTECTED_CONCEPTS = (
    ("motorcy",),
    ("steal", "stole", "stolen", "theft"),
    ("drink", "drinking", "alcohol"),
    ("murder", "murdered", "kill", "killed"),
    ("die", "died", "dead", "death"),
    ("illness", "disease", "sick"),
    ("poverty",),
)


def _normalise_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _content_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z]+", str(value).lower())
        if len(token) >= 3 and token not in _CONTENT_STOPWORDS
    ]


def _short_turn_keeps_content_anchor(source: str, generated: str) -> bool:
    if len(str(source).split()) > 7:
        return True
    source_tokens = _content_tokens(source)
    generated_tokens = _content_tokens(generated)
    if not source_tokens:
        return True
    return any(
        SequenceMatcher(None, left, right).ratio() >= 0.86
        for left in source_tokens
        for right in generated_tokens
    )


def _preserves_protected_concepts(source: str, generated: str) -> bool:
    source_lower = str(source).lower()
    generated_lower = str(generated).lower()
    for equivalents in _PROTECTED_CONCEPTS:
        if any(term in source_lower for term in equivalents) and not any(
            term in generated_lower for term in equivalents
        ):
            return False
    return True


def _full_train_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "conversation_id",
        "checkpoint",
        "split",
        "text_role_turns",
        "initial_intensity",
        "final_intensity",
        "drop_magnitude",
        "emotion_family",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ESConv outcome data is missing columns: {sorted(missing)}")
    train = frame[
        (frame["split"] == "train")
        & frame["checkpoint"].astype(float).sub(1.0).abs().lt(1e-8)
    ].copy()
    if train.empty:
        raise ValueError("No full-context ESConv training conversations were found.")
    if train["conversation_id"].duplicated().any():
        raise ValueError("Full-context training rows must be unique by conversation.")
    return train


def build_outcome_augmentation_plan(
    frame: pd.DataFrame,
    fraction: float = 0.5,
    seed: int = 42,
    max_conversations: int | None = None,
    focus_pairs: Sequence[tuple[int, int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prioritise the least-supported valid outcome pairs deterministically."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Outcome augmentation fraction must be in (0, 1].")
    if max_conversations is not None and max_conversations < 1:
        raise ValueError("max_conversations must be positive when provided.")
    train = _full_train_rows(frame)
    requested_budget = int(math.ceil(len(train) * fraction))
    if max_conversations is not None:
        requested_budget = min(requested_budget, int(max_conversations))

    groups = {
        pair: group.index.tolist()
        for pair, group in train.groupby(["final_intensity", "drop_magnitude"])
    }
    if focus_pairs:
        requested = {tuple(map(int, pair)) for pair in focus_pairs}
        groups = {
            pair: indices
            for pair, indices in groups.items()
            if tuple(map(int, pair)) in requested
        }
        missing_pairs = requested - {tuple(map(int, pair)) for pair in groups}
        if missing_pairs:
            raise ValueError(
                f"Requested focus pairs are absent from training: {sorted(missing_pairs)}"
            )
    eligible_sources = sum(len(indices) for indices in groups.values())
    budget = min(requested_budget, eligible_sources)
    counts = {tuple(map(int, pair)): len(indices) for pair, indices in groups.items()}
    adjusted_counts = dict(counts)
    rng = random.Random(seed)
    shuffled_sources: dict[tuple[int, int], list[int]] = {}
    source_positions: dict[tuple[int, int], int] = {}
    for raw_pair, indices in groups.items():
        pair = tuple(map(int, raw_pair))
        shuffled = list(indices)
        rng.shuffle(shuffled)
        shuffled_sources[pair] = shuffled
        source_positions[pair] = 0

    plan: list[dict[str, Any]] = []
    for generation_index in range(budget):
        available_pairs = [
            pair
            for pair, sources in shuffled_sources.items()
            if source_positions[pair] < len(sources)
        ]
        pair = min(
            available_pairs, key=lambda value: (adjusted_counts[value], value)
        )
        sources = shuffled_sources[pair]
        position = source_positions[pair]
        source_index = sources[position]
        source_positions[pair] = position + 1
        adjusted_counts[pair] += 1
        source = train.loc[source_index]
        role_turns = str(source["text_role_turns"])
        plan.append(
            {
                "generation_index": generation_index,
                "generation_seed": seed + generation_index,
                "source_index": int(source_index),
                "source_conversation_id": str(source["conversation_id"]),
                "final_intensity": pair[0],
                "drop_magnitude": pair[1],
                "source_text_sha256": hashlib.sha256(
                    role_turns.encode("utf-8")
                ).hexdigest(),
            }
        )
    report = {
        "seed": seed,
        "fraction": fraction,
        "original_train_conversations": len(train),
        "eligible_source_conversations": eligible_sources,
        "requested_conversations": requested_budget,
        "planned_conversations": len(plan),
        "source_reuse": False,
        "max_conversations": max_conversations,
        "focus_pairs": (
            [list(pair) for pair in sorted({tuple(pair) for pair in focus_pairs})]
            if focus_pairs
            else None
        ),
        "pair_counts_before": {
            f"final_{pair[0]}_drop_{pair[1]}": count
            for pair, count in sorted(counts.items())
        },
        "pair_counts_after_planned_augmentation": {
            f"final_{pair[0]}_drop_{pair[1]}": count
            for pair, count in sorted(adjusted_counts.items())
        },
    }
    return plan, report


def _split_seeker_windows(
    seeker_items: Sequence[dict[str, str]],
    max_prompt_words: int,
    max_turns_per_window: int | None = None,
) -> list[list[dict[str, str]]]:
    if max_prompt_words < 20:
        raise ValueError("max_prompt_words must be at least 20.")
    if max_turns_per_window is not None and max_turns_per_window < 1:
        raise ValueError("max_turns_per_window must be at least 1.")
    windows: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_words = 0
    for item in seeker_items:
        words = max(
            len(str(item["seeker"]).split())
            + len(str(item.get("preceding_supporter", "")).split()),
            1,
        )
        if current and (
            current_words + words > max_prompt_words
            or (
                max_turns_per_window is not None
                and len(current) >= max_turns_per_window
            )
        ):
            windows.append(current)
            current = []
            current_words = 0
        current.append(dict(item))
        current_words += words
    if current:
        windows.append(current)
    return windows


def outcome_rewrite_prompt(
    seeker_turns: Sequence[str],
    emotion_family: str,
    initial_intensity: int,
    final_intensity: int,
    preceding_supporter_turns: Sequence[str] | None = None,
) -> str:
    """Build the label-preserving target-side ESConv rewriting prompt."""
    contexts = list(preceding_supporter_turns or [""] * len(seeker_turns))
    if len(contexts) != len(seeker_turns):
        raise ValueError("Each seeker turn requires aligned supporter context.")
    original = json.dumps(
        [
            {
                "context_only_do_not_answer": context,
                "seeker": turn,
                "minimum_rewrite_words": (
                    1
                    if len(turn.split()) < 6
                    else math.ceil(len(turn.split()) * 0.4)
                ),
                "maximum_rewrite_words": (
                    len(turn.split()) + 4
                    if len(turn.split()) < 6
                    else math.floor(len(turn.split()) * 2.5)
                ),
            }
            for context, turn in zip(contexts, seeker_turns)
        ],
        ensure_ascii=False,
    )
    return f"""Create a label-preserving paraphrase for an emotional-support dialogue.

Rewrite only the support seeker's utterances below. Make the smallest wording
changes needed for a natural paraphrase. The supporter text is context only:
never answer it and never replace the seeker's response with another plausible
response. The seeker utterances must remain coherent in the same order.

Original context-and-seeker JSON:
{original}

Latent constraints for preservation only (never state these labels or numbers):
- negative emotion family: {emotion_family}
- initial survey intensity category: {initial_intensity}
- final survey intensity category: {final_intensity}

Rules:
- return a valid JSON array of exactly {len(seeker_turns)} strings and nothing else
- preserve every fact, event, person, object, action, uncertainty, negation,
  number, time relation, sarcasm, and conversational intention
- preserve the emotional severity and trajectory at each point; do not invent,
  remove, strengthen, or weaken improvement or deterioration
- keep explicit references to violence, death, alcohol, theft, illness, poverty,
  and named activities; never replace them with safer or related alternatives
- use natural language typical of a person seeking emotional support, but prefer
  a conservative edit over a creative rewrite
- keep short utterances short: when an original seeker utterance has fewer than
  6 words, add no more than 4 words; it is acceptable to return that short
  utterance unchanged when changing it would alter its meaning
- obey `minimum_rewrite_words` and `maximum_rewrite_words` separately for every
  item; these bounds refer only to the corresponding output string
- do not add diagnoses, advice, supporter replies, annotations, role names, labels,
  intensity numbers, explanations, markdown, or stage directions
- each rewritten item must correspond to the original item at the same position

Rewritten seeker turns JSON:"""


def _parse_rewritten_turns(raw: str, expected_count: int) -> list[str]:
    value = str(raw or "").strip()
    start = value.find("[")
    end = value.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Generator did not return a JSON array.")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, list) or len(parsed) != expected_count:
        raise ValueError(
            f"Expected {expected_count} rewritten turns, received "
            f"{len(parsed) if isinstance(parsed, list) else 'non-list output'}."
        )
    cleaned = [_normalise_space(item) for item in parsed]
    if any(not item for item in cleaned):
        raise ValueError("Generated seeker turns cannot be empty.")
    forbidden = (
        "initial intensity",
        "final intensity",
        "emotion family",
        "survey intensity",
        "as an ai",
    )
    if any(phrase in item.lower() for item in cleaned for phrase in forbidden):
        raise ValueError("Generated text exposed latent labels or model commentary.")
    return cleaned


def _rewrite_seeker_turns(
    seeker_turns: Sequence[str],
    preceding_supporter_turns: Sequence[str],
    row: pd.Series,
    generator: TextGenerator,
    generation_seed: int,
    max_prompt_words: int,
    max_tokens: int,
    max_turns_per_window: int = 1,
) -> list[str]:
    rewritten: list[str] = []
    if len(preceding_supporter_turns) != len(seeker_turns):
        raise ValueError("Seeker turns and supporter contexts must align.")
    items = [
        {"preceding_supporter": context, "seeker": turn}
        for context, turn in zip(preceding_supporter_turns, seeker_turns)
    ]
    windows = _split_seeker_windows(
        items,
        max_prompt_words,
        max_turns_per_window=max_turns_per_window,
    )
    for window_index, window in enumerate(windows):
        prompt = outcome_rewrite_prompt(
            [item["seeker"] for item in window],
            str(row["emotion_family"]),
            int(row["initial_intensity"]),
            int(row["final_intensity"]),
            [item["preceding_supporter"] for item in window],
        )
        raw = generator.generate(
            prompt,
            max_tokens=max_tokens,
            seed=generation_seed + window_index,
        )
        rewritten.extend(_parse_rewritten_turns(raw, len(window)))
    if len(rewritten) != len(seeker_turns):
        raise AssertionError("Rewriting changed the number of seeker turns.")
    changed = 0
    for index, (original, generated) in enumerate(zip(seeker_turns, rewritten)):
        source_numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", str(original))
        generated_numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", generated)
        structural_artefact = generated.strip() in {
            "{", "}", "[", "]", '"', "null"
        }
        unsafe_semantic_rewrite = (
            source_numbers != generated_numbers
            or not _short_turn_keeps_content_anchor(str(original), generated)
            or not _preserves_protected_concepts(str(original), generated)
        )
        if structural_artefact or unsafe_semantic_rewrite:
            # Keeping the observed source turn is safer than introducing a
            # plausible but factually different synthetic response.
            rewritten[index] = str(original)
            generated = str(original)
        if _normalise_space(original).lower() != generated.lower():
            changed += 1
        original_words = max(len(str(original).split()), 1)
        generated_words = len(generated.split())
        length_ratio = generated_words / original_words
        short_turn_too_long = (
            original_words < 6 and generated_words > original_words + 4
        )
        regular_turn_outside_ratio = (
            original_words >= 6 and not 0.4 <= length_ratio <= 2.5
        )
        if short_turn_too_long or regular_turn_outside_ratio:
            raise ValueError(
                "Generated turn length is inconsistent with its source "
                f"(source_words={original_words}, generated_words={generated_words}, "
                f"ratio={length_ratio:.2f}, source={str(original)[:180]!r}, "
                f"generated={generated[:180]!r})."
            )
        if generated.lower().startswith(("seeker:", "supporter:")):
            raise ValueError("Generated text included an explicit role label.")
    if changed == 0:
        raise ValueError("Generator returned an unchanged conversation.")
    return rewritten


def _augmented_row(
    source: pd.Series,
    rewritten_seekers: Sequence[str],
    generation_index: int,
    generation_seed: int,
    generator_name: str,
) -> dict[str, Any]:
    role_turns = json.loads(source["text_role_turns"])
    seeker_index = 0
    rewritten_role_turns: list[str] = []
    all_contents: list[str] = []
    supporter_contents: list[str] = []
    original_seeker_contents: list[str] = []
    for raw_turn in role_turns:
        role, separator, content = str(raw_turn).partition(": ")
        normalised_role = role.strip().lower() if separator else "unknown"
        if normalised_role == "seeker":
            original_seeker_contents.append(_normalise_space(content))
            content = rewritten_seekers[seeker_index]
            seeker_index += 1
        elif normalised_role == "supporter":
            supporter_contents.append(_normalise_space(content))
        content = _normalise_space(content)
        rewritten_role_turns.append(f"{normalised_role}: {content}")
        all_contents.append(content)
    if seeker_index != len(rewritten_seekers):
        raise ValueError("Role-aware dialogue and seeker rewrite counts differ.")

    row = source.to_dict()
    source_id = str(source["conversation_id"])
    row.update(
        {
            "conversation_id": f"{source_id}__outcome_aug_{generation_index:04d}",
            "text": " ".join(all_contents),
            "text_role_turns": json.dumps(rewritten_role_turns, ensure_ascii=False),
            "text_seeker": " ".join(rewritten_seekers),
            "text_seeker_turns": json.dumps(
                list(rewritten_seekers), ensure_ascii=False
            ),
            "text_supporter": " ".join(supporter_contents),
            "source_conversation_id": source_id,
            "augmented": True,
            "generation_valid": True,
            "generation_seed": generation_seed,
            "generator": generator_name,
            "changed_seeker_turn_share": sum(
                _normalise_space(original).lower() != generated.lower()
                for original, generated in zip(
                    original_seeker_contents, rewritten_seekers
                )
            )
            / max(len(rewritten_seekers), 1),
        }
    )
    return row


def _text_duplicate_key(value: Any) -> str:
    return _normalise_space(str(value)).casefold()


def _apply_novelty_gate(
    generated_row: dict[str, Any],
    source: pd.Series,
    known_texts: set[str],
    min_changed_turn_share: float,
    max_source_text_similarity: float,
) -> None:
    """Reject synthetic conversations that add too little textual information."""
    changed_share = float(generated_row["changed_seeker_turn_share"])
    source_text = _text_duplicate_key(source["text_seeker"])
    generated_text = _text_duplicate_key(generated_row["text_seeker"])
    source_similarity = SequenceMatcher(None, source_text, generated_text).ratio()
    generated_row.update(
        {
            "source_text_similarity": source_similarity,
            "min_changed_turn_share_required": min_changed_turn_share,
            "max_source_text_similarity_allowed": max_source_text_similarity,
        }
    )
    if changed_share < min_changed_turn_share:
        raise ValueError(
            "Novelty gate rejected generation: too few seeker turns changed "
            f"(share={changed_share:.3f}, required={min_changed_turn_share:.3f})."
        )
    if source_similarity > max_source_text_similarity:
        raise ValueError(
            "Novelty gate rejected generation: seeker text is too similar to its source "
            f"(similarity={source_similarity:.3f}, maximum={max_source_text_similarity:.3f})."
        )
    if generated_text in known_texts:
        raise ValueError(
            "Novelty gate rejected generation: exact seeker-text duplicate already exists."
        )


class DeterministicOutcomeMockGenerator:
    """JSON-preserving smoke generator; never use its output for training."""

    name = "deterministic-outcome-mock"

    def metadata(self) -> dict[str, Any]:
        return {"backend": "mock"}

    def generate(self, prompt: str, max_tokens: int, seed: int) -> str:
        marker = "Original context-and-seeker JSON:"
        rules = "Latent constraints for preservation only"
        source = prompt.split(marker, 1)[1].split(rules, 1)[0].strip()
        turns = json.loads(source)
        return json.dumps(
            [
                f"{_normalise_space(item['seeker'])} Please understand me."
                for item in turns
            ]
        )


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


def _atomic_json(path: Path, payload: Any) -> None:
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


def _boolean_series(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default)
    return values.fillna(default).astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes"}
    )


def validate_outcome_augmentation_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Reject mock, invalid, leaked, or label-inconsistent target augmentation."""
    augmented = _boolean_series(frame, "augmented", False)
    if not augmented.any():
        return {"augmented_rows": 0, "validated": True}
    generated = frame.loc[augmented]
    if (generated["split"] != "train").any():
        raise ValueError("Outcome augmentation may contain training rows only.")
    if not generated["checkpoint"].astype(float).sub(1.0).abs().lt(1e-8).all():
        raise ValueError("Outcome augmentation may add full-context rows only.")
    generation_valid = _boolean_series(frame, "generation_valid", False)
    if not generation_valid.loc[augmented].all():
        raise ValueError("Invalid generated rows cannot be used for outcome training.")
    generators = generated["generator"].fillna("").astype(str).str.lower()
    if generators.str.contains("mock", regex=False).any():
        raise ValueError("Mock outcome augmentation cannot be used for training.")
    if generated["conversation_id"].duplicated().any():
        raise ValueError("Augmented conversation ids must be unique.")
    generated_text_keys = generated["text_seeker"].map(_text_duplicate_key)
    original_text_keys = set(frame.loc[~augmented, "text_seeker"].map(_text_duplicate_key))
    if generated_text_keys.duplicated().any() or generated_text_keys.isin(
        original_text_keys
    ).any():
        raise ValueError("Augmented seeker texts must not duplicate existing texts.")
    novelty_columns = {
        "changed_seeker_turn_share",
        "source_text_similarity",
        "min_changed_turn_share_required",
        "max_source_text_similarity_allowed",
    }
    missing_novelty = novelty_columns - set(generated.columns)
    if missing_novelty:
        raise ValueError(
            "Outcome augmentation is missing novelty metrics: "
            f"{sorted(missing_novelty)}"
        )
    changed_share = generated["changed_seeker_turn_share"].astype(float)
    required_changed_share = generated["min_changed_turn_share_required"].astype(float)
    source_similarity = generated["source_text_similarity"].astype(float)
    allowed_similarity = generated["max_source_text_similarity_allowed"].astype(float)
    if (changed_share < required_changed_share).any():
        raise ValueError("Augmented rows violate the minimum changed-turn share.")
    if (source_similarity > allowed_similarity).any():
        raise ValueError("Augmented rows are too similar to their source conversations.")
    if (
        generated["final_intensity"].astype(int)
        + generated["drop_magnitude"].astype(int)
        != generated["initial_intensity"].astype(int)
    ).any():
        raise ValueError("Augmented outcome labels violate final + drop = initial.")
    original_ids = set(frame.loc[~augmented, "conversation_id"].astype(str))
    source_ids = set(generated["source_conversation_id"].astype(str))
    missing_sources = source_ids - original_ids
    if missing_sources:
        raise ValueError(
            "Augmented rows reference missing source conversations: "
            f"{sorted(missing_sources)[:5]}"
        )
    return {
        "augmented_rows": int(augmented.sum()),
        "source_conversations": len(source_ids),
        "generators": sorted(set(generated["generator"].astype(str))),
        "validated": True,
    }


def _plan_signature(plan: Sequence[dict[str, Any]], report: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"plan": list(plan), "settings": report},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_outcome_augmentation(
    input_path: str | Path,
    output_dir: str | Path,
    generator: TextGenerator | None,
    fraction: float = 0.5,
    seed: int = 42,
    max_conversations: int | None = None,
    focus_pairs: Sequence[tuple[int, int]] | None = None,
    max_prompt_words: int = 180,
    max_tokens: int = 640,
    checkpoint_every: int = 10,
    max_generation_attempts: int = 3,
    max_turns_per_window: int = 1,
    min_changed_turn_share: float = 0.6,
    max_source_text_similarity: float = 0.92,
    resume: bool = False,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Augment ESConv train conversations only and keep validation/test immutable."""
    source_path = Path(input_path).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source_path)
    plan, report = build_outcome_augmentation_plan(
        frame,
        fraction=fraction,
        seed=seed,
        max_conversations=max_conversations,
        focus_pairs=focus_pairs,
    )
    if not 0.0 < min_changed_turn_share <= 1.0:
        raise ValueError("min_changed_turn_share must be in (0, 1].")
    if not 0.0 <= max_source_text_similarity < 1.0:
        raise ValueError("max_source_text_similarity must be in [0, 1).")
    report = {
        **report,
        "input_path": str(source_path),
        "input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "max_prompt_words": max_prompt_words,
        "max_tokens": max_tokens,
        "max_generation_attempts": max_generation_attempts,
        "max_turns_per_window": max_turns_per_window,
        "min_changed_turn_share": min_changed_turn_share,
        "max_source_text_similarity": max_source_text_similarity,
    }
    signature = _plan_signature(plan, report)
    report["plan_signature"] = signature
    _atomic_json(output / "augmentation_plan.json", plan)
    if plan_only:
        report["status"] = "planned"
        _atomic_json(output / "augmentation_manifest.json", report)
        return report
    if generator is None:
        raise ValueError("A text generator is required unless plan_only=True.")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1.")
    if max_generation_attempts < 1:
        raise ValueError("max_generation_attempts must be at least 1.")
    if max_turns_per_window < 1:
        raise ValueError("max_turns_per_window must be at least 1.")
    train = _full_train_rows(frame)
    known_texts = set(frame["text_seeker"].map(_text_duplicate_key))
    progress_path = output / "augmentation_progress.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    invalid_loaded_for_retry = 0
    if progress_path.exists():
        if not resume:
            raise RuntimeError(
                f"Progress already exists at {progress_path}; pass --resume."
            )
        with progress_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["plan_signature"] != signature:
                    raise RuntimeError("Existing progress belongs to another plan.")
                generation_index = int(record["generation_index"])
                if bool(record.get("generation_valid")):
                    try:
                        source_index = int(plan[generation_index]["source_index"])
                        _apply_novelty_gate(
                            record["row"],
                            train.loc[source_index],
                            known_texts,
                            min_changed_turn_share,
                            max_source_text_similarity,
                        )
                    except (KeyError, TypeError, ValueError):
                        completed.pop(generation_index, None)
                        invalid_loaded_for_retry += 1
                    else:
                        completed[generation_index] = record
                        known_texts.add(_text_duplicate_key(record["row"]["text_seeker"]))
                else:
                    # Invalid generations are safe to retry: they never contribute
                    # a synthetic row to the durable output dataset.
                    completed.pop(generation_index, None)
                    invalid_loaded_for_retry += 1

    pending = [
        item for item in plan if int(item["generation_index"]) not in completed
    ]
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(
            pending,
            total=len(plan),
            initial=len(completed),
            desc="Augmenting ESConv outcomes",
            unit="conversation",
        )
    except ImportError:
        iterator = pending

    invalid = sum(not bool(item.get("generation_valid")) for item in completed.values())
    report["invalid_loaded_for_retry"] = invalid_loaded_for_retry
    with progress_path.open("a", encoding="utf-8") as progress:
        since_flush = 0
        for item in iterator:
            source = train.loc[int(item["source_index"])]
            role_turns = json.loads(source["text_role_turns"])
            seeker_turns: list[str] = []
            preceding_supporter_turns: list[str] = []
            preceding_supporter = ""
            for turn in role_turns:
                role, _, content = str(turn).partition(": ")
                if role.strip().lower() == "supporter":
                    preceding_supporter = content
                elif role.strip().lower() == "seeker":
                    seeker_turns.append(content)
                    preceding_supporter_turns.append(preceding_supporter)
            error: str | None = None
            generated_row: dict[str, Any] | None = None
            attempts_used = 0
            attempt_errors: list[str] = []
            for attempt in range(max_generation_attempts):
                attempts_used = attempt + 1
                attempt_seed = int(item["generation_seed"]) + attempt * 1_000_003
                try:
                    rewritten = _rewrite_seeker_turns(
                        seeker_turns,
                        preceding_supporter_turns,
                        source,
                        generator,
                        attempt_seed,
                        max_prompt_words,
                        max_tokens,
                        max_turns_per_window,
                    )
                    generated_row = _augmented_row(
                        source,
                        rewritten,
                        int(item["generation_index"]),
                        attempt_seed,
                        generator.name,
                    )
                    _apply_novelty_gate(
                        generated_row,
                        source,
                        known_texts,
                        min_changed_turn_share,
                        max_source_text_similarity,
                    )
                    known_texts.add(_text_duplicate_key(generated_row["text_seeker"]))
                    break
                except (ValueError, json.JSONDecodeError) as generation_error:
                    attempt_errors.append(str(generation_error))
            if generated_row is None:
                invalid += 1
                error = attempt_errors[-1]
            record = {
                "plan_signature": signature,
                "generation_index": int(item["generation_index"]),
                "generation_valid": generated_row is not None,
                "error": error,
                "attempts_used": attempts_used,
                "attempt_errors": attempt_errors,
                "row": _json_safe(generated_row),
            }
            completed[int(item["generation_index"])] = record
            progress.write(json.dumps(record, sort_keys=True) + "\n")
            since_flush += 1
            if since_flush >= checkpoint_every:
                progress.flush()
                os.fsync(progress.fileno())
                since_flush = 0
        progress.flush()
        os.fsync(progress.fileno())

    original = frame.copy()
    original["source_conversation_id"] = original["conversation_id"]
    original["augmented"] = False
    original["generation_valid"] = True
    original["generation_seed"] = pd.NA
    original["generator"] = "original"
    generated_rows = [
        record["row"]
        for _, record in sorted(completed.items())
        if record.get("generation_valid") and record.get("row") is not None
    ]
    changed_shares = [float(row["changed_seeker_turn_share"]) for row in generated_rows]
    source_similarities = [float(row["source_text_similarity"]) for row in generated_rows]
    parts = [original]
    if generated_rows:
        parts.append(pd.DataFrame(generated_rows))
    augmented = pd.concat(parts, ignore_index=True, sort=False)
    if augmented.loc[
        augmented["split"].isin(["validation", "test"]), "augmented"
    ].any():
        raise AssertionError("Outcome augmentation leaked into validation/test.")
    output_path = output / "esconv_outcome_augmented.csv"
    _atomic_csv(output_path, augmented)
    generator_metadata = getattr(generator, "metadata", None)
    report.update(
        {
            "status": "complete",
            "generator": generator.name,
            "generator_metadata": (
                dict(generator_metadata()) if callable(generator_metadata) else {}
            ),
            "valid_generated_conversations": len(generated_rows),
            "invalid_generations": invalid,
            "accepted_changed_turn_share_min": (
                min(changed_shares) if changed_shares else None
            ),
            "accepted_changed_turn_share_mean": (
                sum(changed_shares) / len(changed_shares) if changed_shares else None
            ),
            "accepted_source_text_similarity_max": (
                max(source_similarities) if source_similarities else None
            ),
            "accepted_source_text_similarity_mean": (
                sum(source_similarities) / len(source_similarities)
                if source_similarities
                else None
            ),
            "output_path": str(output_path.resolve()),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "output_rows": len(augmented),
        }
    )
    _atomic_json(output / "augmentation_manifest.json", report)
    return report
