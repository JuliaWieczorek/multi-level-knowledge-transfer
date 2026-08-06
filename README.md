# Multi-Level Knowledge Transfer for Temporal Emotion Intensity Forecasting

Research code for the final study of the PhD dissertation *Knowledge Transfer for
Emotion Intensity Prediction in Mental Health Support Dialogues*.

The project asks when a model has observed enough of a dialogue to forecast its
final emotional intensity, whether it can forecast the direction of intensity
change, and how observed support strategies are associated with the outcome.
It extends the earlier binary early/late protocol to cumulative checkpoints:
10%, 25%, 50%, 75%, and 100% of each dialogue.

## Research tasks

1. **Final intensity forecasting**
   - Input: the dialogue prefix available at checkpoint `p`.
   - Target: the final dialogue-level intensity.
   - Primary metric: macro-F1.
   - Ordinal metrics: MAE and quadratic weighted kappa.

2. **Intensity-change forecasting**
   - Target: `decrease`, `same`, or `increase`, computed from final minus
     initial intensity.
   - Important dataset constraint: all 1,150 complete ESConv records decrease,
     so the three-class task is not identifiable on ESConv. MEISD supplies all
     three directions and is the primary dataset for this task.

3. **Support-strategy analysis (ESConv)**
   - Quantity: counts and normalised rates of strategies observed by a
     checkpoint.
   - Timing: first occurrence and early/late concentration.
   - Order: adjacent strategy transitions and compact sequence features.
   - Interpretation: observational associations, not causal effects, unless a
     separate causal identification design is justified.

4. **Cross-dataset validation**
   - ESConv: dialogue-level self-reported initial/final intensity and supporter
     strategy annotations.
   - MEISD: turn-level externally annotated, multi-label intensity. A scalar
     turn intensity is derived as the maximum intensity among active emotions
     (neutral/no active emotion = 0); the first and last turn define the derived
     change target.
   - Scores across the two datasets must not be treated as directly equivalent:
     they differ in domain, scale, unit of annotation, and epistemic perspective.

## Leakage-safe experimental protocol

- Split at **conversation level** before expanding conversations into
  checkpoints.
- Keep every checkpoint from one conversation in the same split.
- Use only turns available at the requested checkpoint as model input.
- Fit text vectorisers, scalers, resampling, and augmentation on the training
  split only.
- Report a separate test curve for every checkpoint; never tune on the test
  curve.
- For support strategies distinguish:
  - **prospective features**: only strategies observed up to the checkpoint;
  - **retrospective analysis**: the complete strategy sequence, used only to
    explain associations after the dialogue has ended.
- Repeat neural experiments over multiple seeds and report confidence
  intervals. The lightweight TF-IDF baseline is a pipeline smoke test, not the
  final model.

## Project layout

```text
configs/                 experiment configuration
data/processed/          generated checkpoint tables (not committed)
outputs/                 metrics, predictions, and figures
src/mlkt/                reusable data, split, metric, and baseline code
tests/                   synthetic-data unit tests
```

The source datasets remain in the neighbouring legacy project during the first
reproducibility milestone:

```text
../ESConv.json
../mtl-emotion-intensity-sentiment/data/MEISD_text.csv
```

Paths can be overridden on the command line.

## Quick start

Use Python 3.11+.

```powershell
cd multi-level-knowledge-transfer
python -m pip install -e ".[dev]"
python -m mlkt.cli preprocess
python -m mlkt.cli describe
python -m mlkt.cli baseline --dataset esconv --task final_intensity
python -m mlkt.cli baseline --dataset meisd --task intensity_change
# after installing scikit-learn:
python -m mlkt.cli baseline --dataset esconv --task final_intensity --model tfidf
python -m unittest discover -s tests -v
```

The preprocessing command writes one long-format row per conversation and
checkpoint to `data/processed/esconv_checkpoints.csv` and
`data/processed/meisd_checkpoints.csv`.

## Planned model matrix

| Level | Knowledge transferred | First comparison |
|---|---|---|
| Linguistic | pretrained contextual representations | TF-IDF vs BERT/RoBERTa |
| Temporal | representations shared across checkpoints | separate vs checkpoint-conditioned model |
| Affective/task | emotion and intensity supervision | single-task vs soft-sharing MTL |
| Strategy | observed support actions | text-only vs strategy-only vs fused |
| Domain | ESConv and MEISD representations | in-domain vs sequential fine-tuning vs feature alignment |

The main ablation compares text-only, strategy-only, and text+strategy models at
every checkpoint. A later transfer stage can add parameter, instance, and
feature transfer without changing the data protocol.

## Expected figures

1. Macro-F1 vs observed dialogue percentage for final intensity.
2. Macro-F1 vs observed dialogue percentage for change direction.
3. MAE and quadratic weighted kappa vs checkpoint.
4. Text-only vs strategy-only vs fused ablation curves.
5. Strategy timing/order association plots with uncertainty intervals.
6. In-domain and cross-domain transfer comparison.
