# Multi-Level Knowledge Transfer for Temporal Emotion Intensity Forecasting

## Dissertation relationship

This repository supports Chapter 6, *Multi-Level Knowledge Transfer for Temporal Emotion-Intensity Outcome Forecasting*, of the PhD dissertation *Knowledge Transfer for Emotion Intensity Prediction in Mental Health Support Dialogues*. It contains the temporal outcome-forecasting study described in that chapter.

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

Before extending this matrix, the project now includes a stricter 100% context
ceiling experiment. It consumes the complete role-marked dialogue plus
train-vocabulary encodings of initial intensity, a MEISD-compatible negative
emotion family, and problem type. Its full architecture also adds continuous
speaker composition for every text chunk, separate early and late seeker
representations, an ordered strategy encoder, and auxiliary ordinal and
drop-regression losses used only as training regularisers.

The strengthened ceiling model directly predicts the ten valid
`(final intensity, drop magnitude)` pairs. Invalid pairs are masked using the
initial intensity before decoding, while auxiliary marginal and ordinal heads
regularise training. Each categorical outcome head contains a shared classifier
plus a small residual expert selected by the negative emotion family. This makes
emotion-conditioned outcome prediction explicit rather than relying only on a
metadata-conditioned shared representation.

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

The default experiment configuration uses the already-cached BERT files with
`local_files_only: true`. Training therefore does not contact Hugging Face and
can run without network access. Set this option to `false` only when a configured
transformer still needs to be downloaded.

Source, temporal, and outcome neural training use BF16 automatic mixed precision
by default. Model parameters and saved checkpoints remain FP32, while supported
GPU operations run in BF16. The selected precision is recorded in every run
configuration and summary; set `precision` to `fp32` to disable autocasting.
Temporal and outcome training cache tokenized conversations in memory, use
batches of eight, and limit fine-tuning to the top four layers of each BERT
encoder.

Qwen2.5 GGUF augmentation additionally requires:

```powershell
python -m pip install -e ".[augmentation]"
```

The default generator is `Qwen2.5-7B-Instruct-Q5_K_M`. When no model path is
provided, its single-file GGUF is downloaded once to
`data/models/huggingface` on the project drive and reused from that cache on
later runs. A network connection is needed only for the first download. Use
`--model-cache-dir` to relocate the cache, `--model-path` for an existing local
GGUF, or `--model-repo` and `--model-file` to select another Hugging Face model.
The generator uses the chat template embedded in the GGUF and records the model
name, repository, filename, resolved cache path, and sampling settings in each
manifest.

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
  --gpu-layers -1 --batch-size 1024 --require-gpu --resume

# Benchmark 50 representative generations without writing final outputs.
python -m mlkt.cli augment-transfer `
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

# Optional augmentation experiment: expand each valid synthetic ESConv training
# conversation over the same 10/25/50/75/100% boundaries as its source, then
# keep its 150 runs isolated from the original-data control matrix.
python -m mlkt.cli prepare-augmented-checkpoints
python -m mlkt.cli run-matrix `
  --input data\processed\esconv_checkpoints_augmented.csv `
  --output-dir outputs\temporal_augmented

# 7. Quantity/timing/order and retrospective association analyses.
python -m mlkt.cli analyze-strategies

# 8. Aggregate seeds, transfer deltas, curves, and confusion tables.
python -m mlkt.cli report

# Controlled full-context baselines and ordinal ceiling experiment.
python -m mlkt.cli outcome-baselines
python -m mlkt.cli pretrain-transfer --aligned-negative --seed 42
python -m mlkt.cli train-outcome-ceiling --vanilla --architecture full --seed 42
python -m mlkt.cli train-outcome-ceiling --transfer --architecture full --seed 42

# Diagnose a completed run without refitting or touching model selection.
python -m mlkt.cli analyze-outcome-errors --predictions outputs/outcome_ceiling/seed_42/full/transferred/test_predictions.csv --output-dir outputs/outcome_ceiling/seed_42/full/transferred/error_analysis

# Validation-only ablations: the test split is deliberately not evaluated.
python -m mlkt.cli train-outcome-ceiling --transfer --architecture full --seed 42 --validation-only --run-name ablation_no_focal --outcome-focal-gamma 0
python -m mlkt.cli train-outcome-ceiling --transfer --architecture full --seed 42 --validation-only --run-name ablation_no_ordinal --ordinal-auxiliary-weight 0
python -m mlkt.cli train-outcome-ceiling --transfer --architecture full --seed 42 --validation-only --run-name ablation_no_consistency --consistency-weight 0

# Rare-pair balancing. Start with square-root inverse joint frequency so it
# does not duplicate the full strength of class-balanced focal loss.
python -m mlkt.cli train-outcome-ceiling --transfer --architecture full --seed 42 --validation-only --run-name joint_balanced_p05 --outcome-sampling-strategy joint --joint-sampling-power 0.5

# First isolate the new joint-pair and emotion-conditioned heads on the original
# ESConv training set. This run requires no new text generation and never reads test.
python -m mlkt.cli train-outcome-ceiling --transfer --architecture full --seed 42 --validation-only --run-name joint_emotion_original

# Local AMD preflight (verified on PyTorch 2.9.1 + ROCm 7.2.1). Run offline so
# the cached BERT assets are used without a Hugging Face network probe.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.venv\Scripts\python.exe -m mlkt.cli train-outcome-ceiling --transfer --architecture full --seed 42 --validation-only --run-name joint_emotion_original

# Outcome training fails immediately on any non-finite input, output, loss, or
# aggregate gradient norm and reports the batch and conversation ids. If the
# gradient norm fails, it identifies the first affected parameter. PyTorch
# 2.12.0 + ROCm 7.14 produced infinite StrategyEncoder
# gradients on the tested RX 9070 and must not be used for the final experiment.

# Component ablations on the same untouched training/validation data.
python -m mlkt.cli train-outcome-ceiling --transfer --architecture full --seed 42 --validation-only --run-name joint_only --no-emotion-conditioned-heads
python -m mlkt.cli train-outcome-ceiling --transfer --architecture full --seed 42 --validation-only --run-name emotion_heads_only --no-joint-pair-head
python -m mlkt.cli train-outcome-ceiling --transfer --architecture full --seed 42 --validation-only --run-name legacy_factorized_heads --no-joint-pair-head --no-emotion-conditioned-heads

# Inspect the deterministic 50% target-augmentation plan without loading an LLM.
python -m mlkt.cli augment-outcomes --plan-only `
  --output-dir data\processed\outcome_augmentation\plan_50pct

# Small usable pilot focused on the two zero-recall outcomes: final=4 and drop=3.
# Mock output is only a pipeline smoke test and is rejected by model training.
python -m mlkt.cli augment-outcomes --mock-generator --max-conversations 3 `
  --focus-pair 4,1 --focus-pair 1,3 --focus-pair 2,3 `
  --output-dir data\processed\outcome_augmentation\smoke

python -m mlkt.cli augment-outcomes `
  --gpu-layers -1 --batch-size 1024 --require-gpu `
  --max-conversations 30 `
  --focus-pair 4,1 --focus-pair 1,3 --focus-pair 2,3 `
  --output-dir data\processed\outcome_augmentation\pilot_30

# Classify the pilot with validation-only model selection.
python -m mlkt.cli train-outcome-ceiling --transfer --architecture full --seed 42 `
  --validation-only --run-name joint_emotion_pilot_30 `
  --input data\processed\outcome_augmentation\pilot_30\esconv_outcome_augmented.csv

# Only after the pilot improves validation, generate/resume the full 50% plan.
python -m mlkt.cli augment-outcomes `
  --gpu-layers -1 --batch-size 1024 --require-gpu --resume

# Architecture ablations (repeat with --transfer after the vanilla control).
python -m mlkt.cli train-outcome-ceiling --vanilla --architecture base --seed 42
python -m mlkt.cli train-outcome-ceiling --vanilla --architecture speaker --seed 42
python -m mlkt.cli train-outcome-ceiling --vanilla --architecture trajectory --seed 42
python -m mlkt.cli train-outcome-ceiling --vanilla --architecture strategies --seed 42
```

After augmentation is complete, the remaining study can also be run with one
resumable command:

```powershell
python -m mlkt.cli run-pipeline
```

This performs source pretraining, the 150-run target matrix, strategy analyses,
and final reporting in sequence. It prints a coarse live summary after every
source seed and every five target runs, including elapsed time, completed runs,
and an approximate training ETA. Re-running the same command skips compatible
completed source seeds and target runs. Bootstrap analyses display their own live
progress and ETA.

Full augmentation appends every result to `augmentation_progress.jsonl`.
Repeating the same command with `--resume` continues from that file. Final CSV
and manifest files are replaced atomically only after the plan is complete.

Neural training displays nested progress bars for seeds/runs, epochs, and batches,
including ETA, current loss, validation metrics, and the best epoch. Durable progress
events are appended to `source_training_progress.jsonl` for source pretraining,
`training_progress.jsonl` for each target run, and `matrix_progress.jsonl` for the
complete target matrix.

By default, each target run uses its temporary `best_model.pt` for final
validation/test evaluation and then removes it after predictions, metrics,
diagnostics, and training history have been written. This prevents the complete
150-run matrix from accumulating roughly 100 GiB of target checkpoints. The five
source-transfer checkpoints are retained because subsequent target runs need
them. Set `temporal.retain_best_model` to `true` when the trained target models
themselves are required for later inference.

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

## Secondary three-class outcome analysis

The optional `coarse3` scheme maps the original five-point values for both
outcomes as `1-2 -> 1`, `3 -> 2`, and `4-5 -> 3`. For final intensity these
classes mean low, middle, and high intensity; for drop magnitude they mean
small, middle, and large decrease. Raw targets remain in every predictions
file, and joint consistency is evaluated as feasibility on the original scale.
The default `original4` experiment is unchanged.

First run a full-context pilot (30 runs), then resume with the complete matrix
only if the validation and test diagnostics justify the cost:

```powershell
python -m mlkt.cli run-matrix --label-scheme coarse3 --checkpoints 100 `
  --output-dir outputs\temporal_coarse3

python -m mlkt.cli report --experiments outputs\temporal_coarse3 `
  --output-dir outputs\report_coarse3_tci

# Resume to all 150 runs; completed 100% runs are skipped.
python -m mlkt.cli run-matrix --label-scheme coarse3 `
  --output-dir outputs\temporal_coarse3

python -m mlkt.cli compare-label-schemes `
  --original-report outputs\report_tci `
  --coarse-report outputs\report_coarse3_tci `
  --output-dir outputs\label_scheme_comparison
```

This is a new target-training experiment. Source MTL checkpoints are reused,
but all selected temporal target models must be trained again.

## Reproducibility and data policy

Start with [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the preservation and
evidence checklist. The Windows publication-closeout procedure is documented in
[`PUBLICATION_CLOSEOUT_WINDOWS_PL.md`](PUBLICATION_CLOSEOUT_WINDOWS_PL.md).

- Conversation splits are assigned before checkpoint expansion.
- ESConv validation/test rows never inform target-style pattern extraction.
- MEISD start/end segments remain in one source split.
- Only source training rows are augmented.
- Target-side outcome augmentation adds full-context ESConv training rows only;
  validation and test conversations are copied unchanged.
- The outcome augmenter rewrites seeker turns in bounded windows, keeps all
  supporter turns and strategy features unchanged, and preserves joint labels.
- Each source conversation is augmented at most once. Exact text duplicates,
  near-copies of the source, and generations with too few changed seeker turns
  are rejected before they can enter the training CSV.
- Deterministic mock outcome augmentation is rejected by neural training.
- Failed augmentation generations are excluded from source training by default.
- Test curves are never used for tuning or early stopping.
- Every run records its split hash, source checkpoint, seed, configuration,
  predictions, history, and metrics.
- Input datasets, generated data, Llama weights, neural checkpoints, and
  outputs are ignored by Git. The repository contains code, configurations,
  manifests, and tests.
