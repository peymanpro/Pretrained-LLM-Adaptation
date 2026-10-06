# Pretrained LLM Adaptation

I built this project to answer a practical question: **how do I take a pretrained language model, adapt it to a real downstream NLP task, and still keep the whole process measurable and reproducible?**

<p align="center">
  <img src="assets/adaptation-cycle.svg" alt="Animated workflow of my pretrained LLM adaptation process" width="900">
</p>

> **My focus:** move beyond simply knowing how LLMs work and demonstrate how I engineer, measure, debug, and reproduce an actual model-adaptation workflow.

## Why I built this

I did not want this repository to be another minimal fine-tuning example where a pretrained model is loaded, a few lines of training code are added, and a final accuracy number is presented without much context.

Instead, I wanted to work through the complete engineering path:

```text
Problem
   ↓
Data engineering + leakage checks
   ↓
Classical baselines
   ↓
Pretrained model
   ↓
LoRA / PEFT adaptation
   ↓
Evaluation + ablation
   ↓
Error analysis
   ↓
Inference
   ↓
Reproducible evidence
   ↺
```

The goal is not just to make a model run. **My goal is to make every important decision traceable to code, configuration, tests, or measured evidence.**

## The task I chose

I use the **BANKING77** intent-classification dataset as the concrete downstream task. It contains 13,083 English customer-service queries across 77 intents, with 10,003 training examples and 3,080 test examples.

I chose this problem because it is small enough to experiment with under constrained compute, but rich enough to expose the real issues that matter in NLP adaptation: closely related intents, class imbalance, leakage, model selection, and meaningful error analysis.

## The model and adaptation method

My primary pretrained model is **microsoft/deberta-v3-small**.

I use **LoRA through Hugging Face PEFT** as the main adaptation method. The central question I am exploring is simple:

> **How much task-specific improvement can I obtain while updating only a small fraction of the pretrained model?**

I explicitly verify the target modules against the instantiated DeBERTa architecture before applying LoRA. For sequence classification, I also preserve and save the classifier and pooler modules so that the adapted model can be reconstructed correctly.

## My baseline ladder

I use a small baseline ladder so that I can separate different sources of performance rather than treating LoRA as a black box.

| ID | Method | What I use it for |
|---|---|---|
| B0 | Majority class | I use this as the evaluation floor and pipeline sanity check. |
| B1 | TF-IDF + Logistic Regression | I use this to measure how far a classical lexical approach can go without a Transformer. |
| B2 | Frozen DeBERTa | I use this to isolate the value of pretrained representations without encoder adaptation. |
| A1 | LoRA | This is my primary parameter-efficient adaptation experiment. |
| A2 | Full fine-tuning | I keep this as a conditional compute-controlled comparison rather than forcing an irresponsible benchmark. |

## What I measure

I care about more than one headline number.

For model quality, I measure:

- accuracy;
- macro F1;
- weighted F1;
- per-class precision, recall, and F1;
- confusion matrices.

Where the runtime allows reliable measurement, I also record:

- total and trainable parameter counts;
- trainable-parameter percentage;
- training runtime;
- memory;
- adapter size;
- inference latency.

For me, the important part is not simply asking **“which model scored higher?”**. I also want to know **why**, where it fails, and what changes when I adapt the model.

## How I keep the experiments reproducible

I keep experiments configuration-driven and record the important provenance with every substantive run.

That includes the dataset and model revisions, seed, split definition, tokenizer, maximum sequence length, training parameters, and LoRA configuration.

I also treat the public test set as a held-out evaluation set. Rank selection is performed on validation data, and the final selected configuration is then retrained before the final test evaluation.

Because the project is designed to run on GitHub-hosted CPU runners, the benchmark is intentionally resource-bounded. I am explicit about that limitation rather than presenting a constrained experiment as a full-data or state-of-the-art study.

Raw datasets and large model weights are not committed to Git.

## Data engineering

Before training, I validate the dataset rather than assuming the upstream files are already suitable for an experiment.

The preparation pipeline checks:

1. schema;
2. label validity;
3. empty or malformed text;
4. duplicate samples;
5. train/test text overlap;
6. deterministic fit/validation splitting;
7. leakage between fit and validation sets.

Exact normalized train/test text overlaps are removed from the training pool and recorded in the generated manifest.

## LoRA implementation

I keep the LoRA configuration explicit:

```text
rank
alpha
dropout
target modules
modules to save
```

For DeBERTa, I target the query and value projections and explicitly verify that the expected module names exist before training.

I also calculate trainable versus frozen parameters so that I can quantify what parameter-efficient adaptation actually means for this model.

## Evaluation and failure handling

I do not want invalid runs to quietly become published results.

The evaluation path checks for non-finite logits and probabilities. It also detects prediction collapse in final benchmark evaluations so that a broken or numerically invalid model is not silently treated as successful evidence.

This is intentional: **a failed experiment is useful information during engineering, but it should not be disguised as a successful benchmark.**

## Error analysis

After final evaluation, I inspect the mistakes rather than stopping at aggregate metrics.

The analysis includes:

- the most frequent confusion pairs;
- error counts by gold intent;
- error distribution by input length;
- high-confidence errors;
- representative prediction changes between models.

I want the error analysis to tell me what the adaptation actually changed, not just produce another table of numbers.

## Inference

I keep a small CLI as the inference surface.

It can load the base model and adapter, classify an input, and expose the resulting intent and confidence. The benchmark also runs an inference smoke test after producing the final adapter artifact.

This gives me a simple path from:

```text
training evidence → saved adapter → reproducible inference
```

## Run it locally

Create a Python 3.12 environment and install the project:

```bash
python -m pip install -e ".[dev]"
```

Download and prepare the pinned source data:

```bash
python scripts/download_data.py --config configs/lora.yaml
python scripts/prepare_data.py --config configs/lora.yaml
```

Inspect the actual DeBERTa module names before LoRA training:

```bash
python scripts/inspect_model.py --config configs/lora.yaml
```

Run the classical baselines:

```bash
python scripts/run_baselines.py --config configs/baseline.yaml
```

Run the frozen representation baseline:

```bash
python scripts/train_frozen.py --config configs/lora.yaml
```

Run the LoRA experiment:

```bash
python scripts/train_lora.py --config configs/lora.yaml
```

Run the rank ablation:

```bash
bash scripts/run_rank_ablation.sh
```

Run the complete portfolio benchmark through GitHub Actions by committing a message containing `[run-experiment]`, or start the Benchmark workflow manually.

Evaluate a saved LoRA adapter:

```bash
python scripts/evaluate_model.py --model-dir artifacts/runs/lora-r8 --adapter
```

Run error analysis:

```bash
python scripts/analyze_errors.py experiments/model-evaluation.json
```

Run the test suite:

```bash
pytest
```

Run static checks:

```bash
ruff check src tests scripts
ruff format --check src tests scripts
mypy src
```

## Repository structure

```text
src/pretrained_llm_adaptation/
├── baselines.py
├── config.py
├── data.py
├── evaluation.py
├── inference.py
├── lora.py
├── modeling.py
├── seed.py
└── training.py

configs/
scripts/
tests/
docs/
experiments/
artifacts/
examples/
```

I keep the repository organized around one narrow lifecycle: **prepare → compare → adapt → evaluate → understand → reproduce**.

## What I intentionally did not build

I deliberately keep this project focused on pretrained-model adaptation.

That means I am **not** adding RAG, agents, vector databases, a frontend, SaaS, Kubernetes, distributed training, or a general MLOps platform just to increase the technology list.

Those are useful technologies, but they answer different questions. I want this repository to communicate one thing clearly: **I can take a pretrained model and engineer a disciplined adaptation workflow around a concrete problem.**

## Current status

The implementation, data pipeline, LoRA workflow, evaluation safeguards, CI checks, and benchmark publication pipeline are in place.

I am deliberately not presenting a final performance number until the current benchmark run produces verified evidence on the current `main` commit. This keeps my release claims tied to measurements rather than planned experiments or stale outputs.

## My evidence standard

I distinguish between **implementation** and **measurement**.

A planned experiment is not a result.

A metric becomes verified only after the command that produces it completes successfully and the corresponding artifact is retained with its provenance.

That rule is important to me because I want this repository to demonstrate engineering judgment as well as model knowledge.

## Scope and limitations

This project studies one English, single-domain intent-classification task using a relatively small dataset and an encoder-style pretrained model.

The benchmark is resource-bounded and therefore should be interpreted as engineering evidence for the adaptation pipeline, not as a claim about the best achievable BANKING77 score.

I also do not claim state-of-the-art performance.

## References

- BANKING77 dataset: https://huggingface.co/datasets/PolyAI/banking77
- DeBERTa-v3-small: https://huggingface.co/microsoft/deberta-v3-small
- Transformers PEFT: https://huggingface.co/docs/transformers/peft
- PEFT LoRA: https://huggingface.co/docs/peft/main/package_reference/lora
- PEFT troubleshooting: https://huggingface.co/docs/peft/main/developer_guides/troubleshooting
- Original BANKING77 repository: https://github.com/PolyAI-LDN/task-specific-datasets

## License

MIT

## Benchmark report

When a benchmark run is successfully verified, the workflow publishes `docs/BENCHMARK-REPORT.md` together with the measured artifacts under `experiments/`.

I deliberately do not carry stale benchmark outputs forward between runs, and I never replace pending values with external benchmark numbers.

Release verification is performed on the current `main` commit before a benchmark result is treated as final evidence.
