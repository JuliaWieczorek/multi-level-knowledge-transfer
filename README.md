# Multi-Level Knowledge Transfer for Temporal Emotion Intensity Forecasting

Research code for the final study of the PhD dissertation *Knowledge Transfer
for Emotion Intensity Prediction in Mental Health Support Dialogues*.

The project reuses the winning transfer pipeline from *Multi-Task Aware
Learning for Joint Emotion, Intensity, and Sentiment Analysis*: target-style
feature extraction from ESConv, multilabel-aware MEISD augmentation, and
three-task soft parameter sharing with BERT. The new study adds temporal
forecasting and explicit support-strategy modelling.

## Research design

For each ESConv conversation, separate models observe 10%, 25%, 50%, 75%, or
100% of the dialogue and predict:

1. final self-reported emotion intensity (classes 1-4);
2. drop magnitude, `initial_intensity - final_intensity` (classes 1-4).

Initial intensity is not a model input. It is used to construct the drop target,
as an initial-persistence baseline, and as a control in retrospective
association models.

The main ablation is intentionally asymmetric:

| Variant | Seeker text | Supporter strategies | MEISD transfer |
|---|---:|---:|---:|
| transferred text-only | yes | no | yes |
| strategy-only | no | yes | not applicable |
| transferred text+strategy | yes | yes | yes |
| transferred text+strategy+initial | yes | yes | yes |
| vanilla text-only control | yes | no | no |
| vanilla text+strategy control | yes | yes | no |

Text contains only observed seeker turns. Strategies contain only annotations
available in the observed prefix. A `NO_STRATEGY` token represents prefixes
that have not yet observed a supporter strategy.

## Reused transfer pipeline

One source transfer model is trained per random seed and shared by all five
temporal checkpoints:

1. **Feature transfer:** extract target-domain length, TF-IDF, starter,
   intensifier, and style patterns from ESConv training conversations only.
2. **Instance transfer:** filter MEISD to compatible emotion-intensity pairs,
   transform training instances with Llama-2-7B Chat, preserve multilabel
   annotations, and balance bundles with at most 600 generated examples per
   group. Source validation is never augmented.
3. **Parameter transfer:** train sentiment, multilabel emotion, and
   emotion-conditioned intensity tasks with separate BERT encoders and an L2
   soft-sharing constraint. The emotion and intensity encoders initialise the
   temporal text branch.

The implementation preserves the published task weights and losses:
sentiment `1.0` with weighted cross-entropy, emotion `2.0` with focal loss, and
conditional intensity `0.7` with cross-entropy.

## Temporal architecture

- Complete seeker turns are packed into 128-token chunks.
- Transferred emotion and intensity encoders process every chunk.
- A learned gate combines both affective representations.
- A small Transformer aggregates chunks without discarding the end of long
  conversations.
- The strategy branch combines an ordered strategy Transformer, actual
  normalised turn positions, and quantity/timing features.
- Gated fusion feeds two four-class heads for final intensity and drop
  magnitude.

All neural runs are repeated for seeds `42, 52, 62, 72, 82`. Macro-F1 is the
primary metric; accuracy, precision, recall, micro/weighted-F1, MAE, quadratic
weighted kappa, confusion matrices, standard deviation, and 95% confidence
intervals are also reported. Source MTL additionally records proper multilabel
emotion metrics, per-emotion diagnostics, probabilities, and per-class results.

## Strategy analysis

Prospective models evaluate nested information at every checkpoint:

1. quantity;
2. quantity + timing;
3. quantity + timing + order.

The 100% retrospective analysis fits ordinal association models with
conversation-level bootstrap confidence intervals and Benjamini-Hochberg FDR
correction. It controls for initial intensity, dialogue length, supporter-turn
count, emotion, and problem type. Results are observational associations, not
causal effects.

## Installation

Python 3.11+ is required. Preprocessing and lightweight tests:

```powershell
python -m pip install -e ".[dev]"
```

Full GPU training and analysis:

```powershell
python -m pip install -e ".[neural,analysis]"
```

Llama-2 GGUF augmentation additionally requires:

```powershell
python -m pip install -e ".[augmentation]"
```

On Windows with an AMD GPU, replace the default CPU wheel with the Vulkan
wheel:

```powershell
python -m pip uninstall -y llama-cpp-python
python -m pip install --only-binary=:all: llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan
```

## End-to-end commands

Paths and all research hyperparameters live in `configs/experiment.json`.

```powershell
# 1. Build leakage-safe checkpoint tables.
python -m mlkt.cli preprocess

# 2. Pair legacy start/end files and attach group-level splits.
python -m mlkt.cli prepare-transfer

# 3. Generate the single target-style augmented MEISD dataset.
python -m mlkt.cli augment-transfer `
  --llama-model D:\models\llama-2-7b-chat.Q5_K_M.gguf `
  --gpu-layers -1 --batch-size 1024 --require-gpu --resume

# Benchmark 50 representative generations without writing final outputs.
python -m mlkt.cli augment-transfer `
  --llama-model D:\models\llama-2-7b-chat.Q5_K_M.gguf `
  --gpu-layers -1 --batch-size 1024 --require-gpu --benchmark 50

# A non-semantic pipeline smoke test:
python -m mlkt.cli augment-transfer --mock-generator `
  --output-dir data\processed\transfer\smoke_augmented

# 4. Train five shared source checkpoints.
python -m mlkt.cli preflight-source
python -m mlkt.cli pretrain-transfer

# 5. Inspect the complete 150-run target matrix without training.
python -m mlkt.cli run-matrix --dry-run

# 6. Run/resume all target experiments on CUDA.
python -m mlkt.cli run-matrix

# 7. Quantity/timing/order and retrospective association analyses.
python -m mlkt.cli analyze-strategies

# 8. Aggregate seeds, transfer deltas, curves, and confusion tables.
python -m mlkt.cli report
```

Full augmentation appends every result to `augmentation_progress.jsonl`.
Repeating the same command with `--resume` continues from that file. Final CSV
and manifest files are replaced atomically only after the plan is complete.

Neural training displays nested progress bars for seeds/runs, epochs, and batches,
including ETA, current loss, validation metrics, and the best epoch. Durable progress
events are appended to `source_training_progress.jsonl` for source pretraining,
`training_progress.jsonl` for each target run, and `matrix_progress.jsonl` for the
complete target matrix.

Single-run example:

```powershell
python -m mlkt.cli train-temporal `
  --checkpoint 25 `
  --modality text_strategy `
  --transfer `
  --seed 42 `
  --output-dir outputs\manual\checkpoint_025
```

Baseline and validation commands:

```powershell
python -m mlkt.cli describe
python -m mlkt.cli baseline --dataset esconv --task final_intensity
python -m mlkt.cli baseline --dataset esconv --task drop_magnitude
python -m mlkt.cli baseline --dataset esconv --task final_intensity --model initial
python -m mlkt.cli baseline --dataset esconv --task drop_magnitude --model initial
python -m unittest discover -s tests -v
```

## Reproducibility and data policy

- Conversation splits are assigned before checkpoint expansion.
- ESConv validation/test rows never inform target-style pattern extraction.
- MEISD start/end segments remain in one source split.
- Only source training rows are augmented.
- Failed augmentation generations are excluded from source training by default.
- Test curves are never used for tuning or early stopping.
- Every run records its split hash, source checkpoint, seed, configuration,
  predictions, history, and metrics.
- Input datasets, generated data, Llama weights, neural checkpoints, and
  outputs are ignored by Git. The repository contains code, configurations,
  manifests, and tests.
