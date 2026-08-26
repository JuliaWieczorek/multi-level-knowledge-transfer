# Reproducing the temporal multi-level transfer study

## Canonical scope

This repository supports **Temporal Emotion Intensity Forecasting in Emotional
Support Dialogues with Multi-Level Knowledge Transfer and Support-Strategy
Modeling** and the corresponding temporal multi-level transfer chapter of the
PhD dissertation.

The repository is already structured as an installable package. Use the CLI and
configuration in `configs/experiment.json`; do not create manuscript-only forks
of the training logic.

## Preservation rule

Before publication reruns, record the current commit and preserve the completed
outputs. A rerun with changed data, labels, dependencies, architecture, or
random seeds is a new experiment even if it writes files with familiar names.

## Installation and validation

For preprocessing and tests:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

For the complete training and analysis environment:

```powershell
python -m pip install -e ".[neural,analysis]"
```

Install the optional `augmentation` dependencies only on the machine that will
run local LLaMA generation.

## Canonical workflow

The end-to-end commands are maintained in `README.md`. Publication closeout,
output-count checks, expected hashes, and CPU-only reporting commands are
maintained in `PUBLICATION_CLOSEOUT_WINDOWS_PL.md`. That document takes
precedence over ad hoc reruns when preparing manuscript tables.

## Required evidence bundle

For every reported experiment family, retain:

- Git commit and dirty/clean status;
- `configs/experiment.json` hash;
- dataset and augmentation-manifest hashes;
- conversation split hash;
- source checkpoint identity;
- seed and complete run configuration;
- predictions, histories, metrics, and report manifests;
- Python, dependency, accelerator, and hardware versions;
- the exact manuscript or thesis revision consuming the result.

## Completion criteria

A reproduction must verify experiment counts, checkpoint coverage, five-seed
aggregation, prediction/label consistency, confidence-interval method, and the
claims consumed by the manuscript. Passing unit tests alone is necessary but
not sufficient.

The publication-closeout audit should be the final authority for expected files
and claim checks.
