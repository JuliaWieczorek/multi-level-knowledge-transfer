from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence


def _append_matrix_progress(path: Path, event: dict[str, Any]) -> None:
    record = {"timestamp_unix": time.time(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def experiment_variants() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "transferred_text",
            "modality": "text",
            "transfer": True,
            "use_initial_intensity": False,
        },
        {
            "name": "strategy",
            "modality": "strategy",
            "transfer": False,
            "use_initial_intensity": False,
        },
        {
            "name": "transferred_text_strategy",
            "modality": "text_strategy",
            "transfer": True,
            "use_initial_intensity": False,
        },
        {
            "name": "transferred_text_strategy_initial",
            "modality": "text_strategy",
            "transfer": True,
            "use_initial_intensity": True,
        },
        {
            "name": "vanilla_text",
            "modality": "text",
            "transfer": False,
            "use_initial_intensity": False,
        },
        {
            "name": "vanilla_text_strategy",
            "modality": "text_strategy",
            "transfer": False,
            "use_initial_intensity": False,
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
    from tqdm.auto import tqdm

    source_root = Path(source_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = matrix_manifest(checkpoints, seeds)
    with (output_root / "matrix_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(runs, handle, indent=2)
    progress_path = output_root / "matrix_progress.jsonl"
    completed = 0
    skipped = 0
    run_iterator = tqdm(
        runs,
        desc="Target experiment matrix",
        unit="run",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    for run in run_iterator:
        run_iterator.set_postfix(
            checkpoint=f"{run['checkpoint_percent']}%",
            variant=run["name"],
            seed=run["seed"],
        )
        run_dir = (
            output_root
            / f"seed_{run['seed']}"
            / f"checkpoint_{run['checkpoint_percent']:03d}"
            / run["name"]
        )
        manifest_path = run_dir / "run_manifest.json"
        if skip_existing and manifest_path.exists():
            skipped += 1
            _append_matrix_progress(
                progress_path, {"event": "run_skipped", **run}
            )
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
        _append_matrix_progress(progress_path, {"event": "run_started", **run})
        try:
            train_temporal_model(
                checkpoints_path=checkpoints_path,
                output_dir=run_dir,
                checkpoint=run["checkpoint"],
                modality=run["modality"],
                seed=run["seed"],
                config=config,
                transfer_checkpoint_path=source_checkpoint,
                use_initial_intensity=run.get("use_initial_intensity", False),
                progress_position=1,
            )
        except Exception as exc:
            _append_matrix_progress(
                progress_path,
                {"event": "run_failed", "error": repr(exc), **run},
            )
            raise
        completed += 1
        _append_matrix_progress(
            progress_path, {"event": "run_completed", **run}
        )
    return {
        "planned_runs": len(runs),
        "completed_runs": completed,
        "skipped_runs": skipped,
        "output_root": str(output_root),
    }
