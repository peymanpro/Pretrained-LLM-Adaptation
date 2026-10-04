# Pretrained LLM Adaptation

A reproducible study of adapting a pretrained language model to a concrete downstream NLP task using parameter-efficient fine-tuning.

> Portfolio focus: move from understanding LLM internals to engineering, measuring, and reproducing an actual adaptation workflow.

## What this project does

~~~text
BANKING77
   ↓
Data validation + leakage checks
   ↓
Classical baselines
   ↓
DeBERTa-v3-small
   ↓
LoRA / PEFT
   ↓
Evaluation + ablation + error analysis
   ↓
Base model + adapter inference
~~~

The primary task is 77-class banking intent classification. The official dataset contains 13,083 English customer-service queries: 10,003 train and 3,080 test examples.

The primary model is microsoft/deberta-v3-small. Its model card reports 6 Transformer layers, hidden size 768, 44M backbone parameters, and a 128K-token vocabulary whose embedding layer adds 98M parameters.

The primary adaptation method is LoRA through PEFT. The PEFT documentation defines rank, scaling, dropout, target modules, and saved trainable modules as the core LoRA configuration controls.

## Baseline ladder

| ID | Method | Purpose |
|---|---|---|
| B0 | Majority class | Evaluation floor |
| B1 | TF-IDF + Logistic Regression | Classical non-Transformer baseline |
| B2 | Frozen DeBERTa | Value of pretrained representations without encoder adaptation |
| A1 | LoRA | Primary adaptation experiment |
| A2 | Full fine-tuning | Conditional compute-controlled comparison |

## Evaluation

Primary metrics are accuracy, macro F1, weighted F1, per-class metrics, and a confusion matrix. Resource measurements include trainable parameter count and, when reliably available, training time, memory, adapter size, and inference latency.

## Reproducibility

Experiments are configuration-driven. Dataset source revision, model revision, random seed, split rule, tokenizer, sequence length, training parameters, and LoRA settings are recorded with each run.

Raw data and large model weights are not committed to Git.

## Run locally

Create a Python 3.12 environment and install the project:

~~~bash
python -m pip install -e ".[dev]"
~~~

Download the pinned source data:

~~~bash
python scripts/download_data.py --config configs/lora.yaml
python scripts/prepare_data.py --config configs/lora.yaml
~~~

Inspect the actual DeBERTa module names before LoRA training:

~~~bash
python scripts/inspect_model.py --config configs/lora.yaml
~~~

Run the classical baselines:

~~~bash
python scripts/run_baselines.py --config configs/baseline.yaml
~~~

Run the frozen representation baseline:

~~~bash
python scripts/train_frozen.py --config configs/lora.yaml
~~~

Run the LoRA experiment:

~~~bash
python scripts/train_lora.py --config configs/lora.yaml
~~~

Run the rank ablation:

~~~bash
bash scripts/run_rank_ablation.sh
~~~

Evaluate a saved LoRA adapter:

~~~bash
python scripts/evaluate_model.py --model-dir artifacts/runs/lora-r8 --adapter
~~~

Run error analysis:

~~~bash
python scripts/analyze_errors.py experiments/model-evaluation.json
~~~

Run tests:

~~~bash
pytest
~~~

Run static checks:

~~~bash
ruff check src tests scripts
ruff format --check src tests scripts
mypy src
~~~

## Repository structure

~~~text
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
~~~

## Scope boundaries

This repository is about LLM adaptation. It intentionally does not introduce RAG, agents, vector databases, frontend/UI, SaaS, Kubernetes, distributed training, or a general MLOps platform.

## Evidence standard

The repository distinguishes implementation from measurement. A planned experiment is not reported as a result. A metric is considered verified only after its producing command completes and the corresponding artifact is retained.

## References

- BANKING77 dataset: https://huggingface.co/datasets/PolyAI/banking77
- DeBERTa-v3-small: https://huggingface.co/microsoft/deberta-v3-small
- Transformers PEFT: https://huggingface.co/docs/transformers/peft
- PEFT LoRA: https://huggingface.co/docs/peft/main/package_reference/lora
- PEFT troubleshooting: https://huggingface.co/docs/peft/main/developer_guides/troubleshooting
- Original BANKING77 repository: https://github.com/PolyAI-LDN/task-specific-datasets

## License

MIT
