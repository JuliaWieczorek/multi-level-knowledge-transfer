from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .transfer import TextGenerator


def _normalise_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


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
    budget = int(math.ceil(len(train) * fraction))
    if max_conversations is not None:
        budget = min(budget, int(max_conversations))

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
        pair = min(adjusted_counts, key=lambda value: (adjusted_counts[value], value))
        sources = shuffled_sources[pair]
        position = source_positions[pair]
        source_index = sources[position % len(sources)]
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
        "planned_conversations": len(plan),
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
    seeker_items: Sequence[dict[str, str]], max_prompt_words: int
) -> list[list[dict[str, str]]]:
    if max_prompt_words < 20:
        raise ValueError("max_prompt_words must be at least 20.")
    windows: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_words = 0
    for item in seeker_items:
        words = max(
            len(str(item["seeker"]).split())
            + len(str(item.get("preceding_supporter", "")).split()),
            1,
        )
        if current and current_words + words > max_prompt_words:
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
            {"preceding_supporter": context, "seeker": turn}
            for context, turn in zip(contexts, seeker_turns)
        ],
        ensure_ascii=False,
    )
    return f"""Create a label-preserving paraphrase for an emotional-support dialogue.

Rewrite only the support seeker's utterances below. They are consecutive seeker
utterances from one conversation and must remain coherent in the same order.

Original context-and-seeker JSON:
{original}

Latent constraints for preservation only (never state these labels or numbers):
- negative emotion family: {emotion_family}
- initial survey intensity category: {initial_intensity}
- final survey intensity category: {final_intensity}

Rules:
- return a valid JSON array of exactly {len(seeker_turns)} strings and nothing else
- preserve every fact, event, person, uncertainty, and conversational intention
- preserve the emotional severity and trajectory at each point; do not invent,
  remove, strengthen, or weaken improvement or deterioration
- use natural, varied language typical of a person seeking emotional support
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
) -> list[str]:
    rewritten: list[str] = []
    if len(preceding_supporter_turns) != len(seeker_turns):
        raise ValueError("Seeker turns and supporter contexts must align.")
    items = [
        {"preceding_supporter": context, "seeker": turn}
        for context, turn in zip(preceding_supporter_turns, seeker_turns)
    ]
    windows = _split_seeker_windows(items, max_prompt_words)
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
    for original, generated in zip(seeker_turns, rewritten):
        if _normalise_space(original).lower() != generated.lower():
            changed += 1
        original_words = max(len(str(original).split()), 1)
        length_ratio = len(generated.split()) / original_words
        if not 0.4 <= length_ratio <= 2.5:
            raise ValueError(
                "Generated turn length is inconsistent with its source "
                f"(ratio={length_ratio:.2f})."
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
    signature = _plan_signature(plan, report)
    report = {
        **report,
        "input_path": str(source_path),
        "input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "plan_signature": signature,
        "max_prompt_words": max_prompt_words,
        "max_tokens": max_tokens,
    }
    _atomic_json(output / "augmentation_plan.json", plan)
    if plan_only:
        report["status"] = "planned"
        _atomic_json(output / "augmentation_manifest.json", report)
        return report
    if generator is None:
        raise ValueError("A text generator is required unless plan_only=True.")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1.")

    train = _full_train_rows(frame)
    progress_path = output / "augmentation_progress.jsonl"
    completed: dict[int, dict[str, Any]] = {}
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
                completed[int(record["generation_index"])] = record

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
            try:
                rewritten = _rewrite_seeker_turns(
                    seeker_turns,
                    preceding_supporter_turns,
                    source,
                    generator,
                    int(item["generation_seed"]),
                    max_prompt_words,
                    max_tokens,
                )
                generated_row = _augmented_row(
                    source,
                    rewritten,
                    int(item["generation_index"]),
                    int(item["generation_seed"]),
                    generator.name,
                )
            except (ValueError, json.JSONDecodeError) as generation_error:
                invalid += 1
                error = str(generation_error)
            record = {
                "plan_signature": signature,
                "generation_index": int(item["generation_index"]),
                "generation_valid": generated_row is not None,
                "error": error,
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
            "output_path": str(output_path.resolve()),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "output_rows": len(augmented),
        }
    )
    _atomic_json(output / "augmentation_manifest.json", report)
    return report
