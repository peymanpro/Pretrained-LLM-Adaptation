# Experiment Methodology

## Benchmark protocol

The final portfolio benchmark is resource-bounded so that the complete pipeline can execute reliably on GitHub-hosted CPU runners.

The protocol uses the full public BANKING77 test split for final evaluation while limiting training runs to deterministic stratified subsets recorded in the configuration and run metadata. Any exact normalized text overlap between the upstream train and test files is removed from the training pool, and repeated normalized texts within the training pool are collapsed deterministically before the validation split.

Training allocation:
- frozen baseline: 600 training examples and 200 validation examples for 2 epochs;
- LoRA rank selection: 600 training examples and 200 validation examples for each of r=4, r=8, and r=16 for 3 epochs;
- final LoRA run: 1,200 training examples and 300 validation examples for 3 epochs;
- final test metrics: full BANKING77 public test split.

The benchmark uses 5e-5 for LoRA parameters and the DeBERTa pooler, plus 2e-4 for the classification layer. A pilot that applied 1e-4 to the whole task head produced NaN gradients/loss, so the higher rate is restricted to the small randomly initialized classifier while the pooler remains at the conservative adaptation rate. The effective learning rates are recorded in run metadata. The sequence length is capped at 48; inspection of the prepared pinned split showed that this retains more than 99% of examples without truncation while reducing CPU work. This remains a resource-bounded engineering benchmark rather than a full-data training study.

This is not presented as a full-data benchmark. Its purpose is to demonstrate a complete, reproducible adaptation workflow under constrained compute.

## Data

PolyAI BANKING77 is downloaded from the pinned upstream revision in configuration. The preparation pipeline validates schema, labels, empty text, duplicates, and train/test leakage, then creates deterministic stratified fit/validation splits.

The public test set is never used for rank selection.

## Baselines

1. Majority-class floor.
2. TF-IDF plus logistic regression.
3. Frozen DeBERTa representation baseline.
4. LoRA adaptation of DeBERTa-v3-small.

Full fine-tuning remains conditional because this benchmark is explicitly CPU-bounded.

## Model

Primary model: microsoft/deberta-v3-small

The exact Hugging Face revision is pinned in benchmark configuration.

## LoRA

The benchmark uses PEFT LoRA with query/value projections and explicitly saved classifier/pooler modules.

Target names are verified against the instantiated DeBERTa architecture before training.

## Rank selection

Validation macro F1 is the selection metric. The public test set is excluded from this decision.

Ranks evaluated:
- r=4;
- r=8;
- r=16.

The final selected rank is retrained on the larger final training allocation and then evaluated once on the held-out test split.

## Evaluation

Primary metrics:
- accuracy;
- macro F1;
- weighted F1;
- per-class metrics;
- confusion matrix.

Additional evidence:
- trainable parameter count;
- trainable percentage;
- training runtime;
- error distribution;
- representative prediction changes;
- inference smoke test.

The evaluation command rejects models whose predictions contain fewer than two distinct intents, preventing a collapsed classifier from being silently accepted as benchmark evidence.

## Error analysis

The final test predictions are analyzed for highest-frequency confusion pairs, error counts by gold intent, error counts by input-length bucket, and high-confidence errors.

## Limitations

The resource-bounded training allocation limits statistical power and may understate or distort performance relative to full-data training. Results should therefore be interpreted as engineering evidence for the adaptation pipeline, not as a claim about the best achievable BANKING77 score.

The project does not claim state-of-the-art performance.

## Evidence rule

A result is accepted only when:
1. the producing workflow step completes successfully;
2. the generated artifact exists;
3. the configuration/model/data provenance is recorded;
4. the test split was not used for model selection.
