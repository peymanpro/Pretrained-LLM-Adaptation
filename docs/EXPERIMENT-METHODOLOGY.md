# Experiment Methodology

## Dataset

Use PolyAI BANKING77 and preserve the official test set. The original training set is split into fit/validation subsets with stratification and seed 42.

The source files are downloaded from the original PolyAI dataset repository using the pinned commit in configuration. Raw files are intentionally not committed to Git.

## Baselines

1. Majority-class floor.
2. TF-IDF plus logistic regression.
3. Frozen DeBERTa encoder with a trainable task head.
4. LoRA adaptation of DeBERTa-v3-small.

Full fine-tuning is conditional on available compute.

## Primary metrics

- accuracy;
- macro F1;
- weighted F1;
- per-class precision/recall/F1;
- confusion matrix.

## LoRA configuration

The initial configuration targets query_proj/value_proj attention projections with rank 8, alpha 16, dropout 0.1, and explicitly saved task modules. The exact target module names must be inspected on the instantiated model before the first training run.

PEFT documents modules_to_save as the mechanism for training and saving additional modules alongside adapter weights.

## Leakage controls

No test sample is used for hyperparameter selection. Text overlap is checked after whitespace normalization and case folding. Duplicates within splits are reported.

## Ablation

The default rank ablation is r = 4, 8, and 16. Final conclusions must be based on actual experiment output.

## Evidence rule

A result is measured only when the command completes and the generated artifact can be tied back to a concrete configuration. This repository never substitutes an external benchmark number for a result produced by the local pipeline.

## Controlled benchmark execution

The full benchmark is intentionally not executed on every push. It is started explicitly with the `[run-experiment]` commit marker or through the GitHub Actions workflow dispatcher.

The benchmark produces the classical baselines, frozen-representation baseline, LoRA rank ablation, validation-based rank selection, a full-data final LoRA run, held-out test evaluation, structured error analysis, prediction comparison, model inspection, inference smoke output, and a compiled results summary.
