# Experiment Methodology

## Benchmark protocol

The portfolio benchmark is deliberately resource-bounded so that the complete workflow can run on GitHub-hosted CPU runners.

The benchmark uses the full public BANKING77 test split for the final held-out evaluation. Training and rank-selection runs use deterministic stratified subsets defined by the CI benchmark configurations.

Before splitting, the preparation pipeline:

1. validates the BANKING77 schema and 77-intent label set;
2. checks for empty or malformed samples;
3. checks exact normalized text overlap between the official train and test files;
4. removes overlapping training texts from the training pool;
5. removes repeated normalized training texts deterministically;
6. creates a deterministic stratified fit/validation split using seed 42.

### Benchmark allocations

The GitHub Actions benchmark currently uses these configurations:

| Stage | Configuration | Train cap | Validation cap | Epochs | Max length |
|---|---|---:|---:|---:|---:|
| Frozen DeBERTa | `configs/ci-frozen.yaml` | 600 | 200 | 2 | 48 |
| LoRA rank 4 | `configs/ci-r4.yaml` | 3,000 | 1,000 | 1 | 48 |
| LoRA rank 8 | `configs/ci-r8.yaml` | 3,000 | 1,000 | 1 | 48 |
| LoRA rank 16 | `configs/ci-r16.yaml` | 3,000 | 1,000 | 1 | 48 |
| Final LoRA | `configs/ci-final.yaml` | 1,200 | 300 | 3 | 48 |
| Final evaluation | — | — | — | — | 48 |

The final evaluation uses the complete 3,080-example public BANKING77 test split.

The rank-ablation configurations use:

- adapter learning rate: `2e-5`;
- per-device training batch size: 32;
- per-device evaluation batch size: 64;
- weight decay: `0.01`;
- warmup ratio: `0.1`;
- Adam epsilon: `1e-6`;
- gradient clipping: `1.0`;
- fp16: disabled;
- bf16: disabled;
- seed: 42.

The final LoRA configuration uses the same optimization settings and retrains the selected rank on its larger final training allocation.

This is a resource-bounded engineering benchmark, not a full-data training study and not a state-of-the-art benchmark.

## Dataset

The task is 77-class intent classification using PolyAI BANKING77.

The source data are downloaded from a pinned revision of the upstream dataset repository. Raw files are intentionally not committed to Git.

The public test set is kept out of rank selection and other model-selection decisions.

## Baselines

The benchmark contains:

1. majority-class floor;
2. TF-IDF plus logistic regression;
3. frozen DeBERTa-v3-small with a trainable task head;
4. LoRA adaptation of DeBERTa-v3-small.

Full fine-tuning remains conditional on whether a controlled comparison is feasible under the available compute.

## Model

Primary model:

`microsoft/deberta-v3-small`

The benchmark pins the Hugging Face model revision:

`a59be8aa63396e73dbb45a1487e4cde4be98bfa4`

## LoRA

The benchmark applies PEFT LoRA to the DeBERTa query and value projections:

```text
query_proj
value_proj
```

The sequence-classification modules are explicitly saved:

```text
classifier
pooler
```

The implementation verifies the expected module names against the instantiated model before applying LoRA.

The rank ablation evaluates:

```text
r = 4
r = 8
r = 16
```

## Rank selection

Validation macro F1 is the rank-selection metric.

The public test set is excluded from this decision.

After the three rank experiments finish, `scripts/select_rank.py` selects the best rank by validation macro F1, using accuracy only as a secondary tie-breaker.

The selected rank is then retrained with `configs/ci-final.yaml` and evaluated once on the full public test split.

## Evaluation

Primary metrics:

- accuracy;
- macro F1;
- weighted F1;
- per-class precision, recall, and F1;
- confusion matrix.

Additional evidence includes:

- total and trainable parameter counts;
- trainable-parameter percentage;
- training runtime;
- prediction distribution;
- error-analysis outputs;
- prediction changes between frozen and final models;
- inference smoke test.

The evaluation path rejects non-finite logits and probabilities.

For final benchmark evidence, a classifier whose predictions collapse to fewer than two unique intents is rejected. The benchmark integrity gate is stricter and requires at least five distinct predicted intents in the final evaluation artifact.

## Error analysis

The final test predictions are analyzed for:

- total error count and error rate;
- errors by gold intent;
- errors by input-length bucket;
- most frequent gold/predicted confusion pairs;
- high-confidence errors.

The prediction-comparison step also checks row alignment and records examples whose predictions changed between the frozen baseline and final adapted model.

## Benchmark publication and integrity

The GitHub Actions benchmark proceeds through:

```text
prepare
→ classical baselines
→ frozen baseline
→ LoRA rank ablation
→ validation rank selection
→ final LoRA training
→ full test evaluation
→ error analysis
→ prediction comparison
→ result compilation
→ benchmark report
→ integrity verification
→ inference smoke test
→ artifact publication
```

The final integrity gate verifies, among other things:

- selected rank matches the final training configuration;
- final evaluation contains enough examples;
- final prediction diversity is sufficient;
- reported metrics are within valid ranges;
- required baseline rows exist;
- adapter and tokenizer artifacts exist.

The publication workflow copies only artifacts produced by the verified benchmark run into the repository.

## Reproducibility rule

A result is accepted only when:

1. the producing workflow step completes successfully;
2. the generated artifact exists;
3. model, dataset, environment, and training configuration are retained;
4. the public test split was not used for model selection.

External benchmark numbers are not substituted for missing measurements.

## Limitations

The benchmark uses constrained training allocations to make the complete workflow feasible on CPU runners. Results may therefore differ from a full-data or differently tuned training regime.

The project studies one English, single-domain classification problem with an encoder-style pretrained model. Conclusions do not automatically generalize to multilingual, multi-domain, generative, or large-scale language-model adaptation.

The project makes no state-of-the-art or production-readiness claim.
