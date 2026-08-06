from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def experiment_variants() -> tuple[dict[str, Any], ...]:
    return (
        {"name": "transferred_text", "modality": "text", "transfer": True},
        {"name": "strategy", "modality": "strategy", "transfer": False},
        {
            "name": "transferred_text_strategy",
            "modality": "text_strategy",
            "transfer": True,
        },
        {"name": "vanilla_text", "modality": "text", "transfer": False},
        {
            "name": "vanilla_text_strategy",
            "modality": "text_strategy",
            "transfer": False,
        },
    )


def matrix_manifest(
    checkpoints: Sequence[float],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        {
            "checkpoint": float(checkpoint),
            "checkpoint_percent": int(round(float(checkpoint) * 100)),
            "seed": int(seed),
            **variant,
        }
        for seed in seeds
        for checkpoint in checkpoints
        for variant in experiment_variants()
    ]


def run_experiment_matrix(
    checkpoints_path: str | Path,
    source_root: str | Path,
    output_root: str | Path,
    config: dict[str, Any],
    checkpoints: Sequence[float],
    seeds: Sequence[int],
    skip_existing: bool = True,
) -> dict[str, Any]:
    # Import lazily so manifest/dry-run commands work without torch.
    from .training import train_temporal_model

    source_root = Path(source_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = matrix_manifest(checkpoints, seeds)
    with (output_root / "matrix_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(runs, handle, indent=2)
    completed = 0
    skipped = 0
    for run in runs:
        run_dir = (
            output_root
            / f"seed_{run['seed']}"
            / f"checkpoint_{run['checkpoint_percent']:03d}"
            / run["name"]
        )
        manifest_path = run_dir / "run_manifest.json"
        if skip_existing and manifest_path.exists():
            skipped += 1
            continue
        source_checkpoint = None
        if run["transfer"]:
            source_checkpoint = (
                source_root
                / f"seed_{run['seed']}"
                / "source_transfer_checkpoint.pt"
            )
            if not source_checkpoint.exists():
                raise FileNotFoundError(
                    f"Missing source checkpoint for seed {run['seed']}: "
                    f"{source_checkpoint}"
                )
        train_temporal_model(
            checkpoints_path=checkpoints_path,
            output_dir=run_dir,
            checkpoint=run["checkpoint"],
            modality=run["modality"],
            seed=run["seed"],
            config=config,
            transfer_checkpoint_path=source_checkpoint,
        )
        completed += 1
    return {
        "planned_runs": len(runs),
        "completed_runs": completed,
        "skipped_runs": skipped,
        "output_root": str(output_root),
    }
