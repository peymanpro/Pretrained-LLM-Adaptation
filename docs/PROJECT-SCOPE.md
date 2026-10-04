# Project Scope — Pretrained LLM Adaptation

## Status

**Planning / Foundation**

No training result or model-performance claim has been established yet.

## Purpose

This repository studies the engineering process of adapting a pretrained language model to a concrete downstream task.

The project is intentionally different from a tutorial that merely demonstrates how to fine-tune a Hugging Face model. The goal is to produce a reproducible artifact covering:

~~~text
Pretrained Model
    ↓
Task Definition
    ↓
Dataset Engineering
    ↓
Baseline
    ↓
PEFT / LoRA Adaptation
    ↓
Evaluation
    ↓
Error Analysis
    ↓
Inference
    ↓
Reproducible Artifact
~~~

## Problem

### Task

Fine-grained intent classification for online banking customer-service queries.

### Dataset

Primary dataset:

**PolyAI BANKING77**

The dataset contains 13,083 English customer-service queries labeled with 77 intents, with 10,003 training examples and 3,080 test examples.

Canonical source:

https://huggingface.co/datasets/PolyAI/banking77

The exact dataset revision used by experiments will be recorded.

## Model

Primary pretrained model:

**microsoft/deberta-v3-small**

The model is intended for English NLU tasks and exposes a sequence-classification architecture through Transformers.

Reference:

https://huggingface.co/microsoft/deberta-v3-small

The exact model revision used by final experiments will be recorded.

## Why this model

The model is small enough to make controlled experiments realistic while still providing a real pretrained Transformer rather than a toy network.

The model card reports 6 Transformer layers, hidden size 768, and approximately 44M backbone parameters, with additional embedding parameters.

These published characteristics make it a practical candidate for studying the trade-off between full adaptation and parameter-efficient adaptation.

## Primary Adaptation Method

**LoRA via Hugging Face PEFT**

LoRA is the primary adaptation method because it directly addresses the central engineering question:

> How much task adaptation can be achieved while updating only a small fraction of the pretrained model?

The implementation will verify the actual DeBERTa module names before selecting LoRA targets.

For sequence classification, the randomly initialized classification head and pooler must be preserved and reloaded correctly. The PEFT documentation specifically identifies classifier and pooler for DeBERTa sequence-classification models.

References:

- https://huggingface.co/docs/transformers/peft
- https://huggingface.co/docs/peft/main/package_reference/lora
- https://huggingface.co/docs/peft/en/developer_guides/troubleshooting

## Baseline Ladder

### B0 — Majority Class

A trivial floor used to validate the evaluation pipeline.

### B1 — TF-IDF + Linear Classifier

A classical non-Transformer baseline.

Purpose: estimate how much of the task can be solved from lexical information without a pretrained language model.

### B2 — Frozen DeBERTa Representation

The pretrained encoder remains frozen while the task head is trained.

Purpose: separate the value of pretrained representations from the value of adapting the encoder.

### A1 — LoRA Adaptation

The primary experiment.

The pretrained encoder is adapted using LoRA while the task head is trained.

### A2 — Full Fine-Tuning

Conditional comparison only.

It will be run if the available compute makes the comparison responsible and reproducible. Otherwise the limitation will be documented instead of simulated.

## Evaluation

Primary metrics:

- accuracy;
- macro F1;
- weighted F1;
- per-class precision, recall, and F1;
- confusion matrix.

Resource measurements, where the runtime permits reliable measurement:

- total parameters;
- trainable parameters;
- trainable-parameter percentage;
- training wall-clock time;
- peak memory;
- saved adapter size;
- inference latency.

No performance target is declared achieved before an experiment produces the evidence.

## Data Engineering

The dataset pipeline will include:

1. acquisition by explicit dataset identifier/revision;
2. schema validation;
3. malformed/empty sample validation;
4. label validation;
5. duplicate detection;
6. train/validation/test contamination checks;
7. deterministic validation split;
8. sequence-length inspection;
9. tokenizer configuration;
10. reproducible dataset preparation.

The public test set remains held out for final evaluation.

## Experiment Reproducibility

Every substantive run will record:

- model identifier and revision;
- dataset identifier and revision;
- seed;
- split definition;
- tokenizer;
- maximum sequence length;
- learning rate;
- batch size;
- gradient accumulation;
- epochs;
- evaluation/checkpoint strategy;
- LoRA rank;
- LoRA alpha;
- LoRA dropout;
- LoRA target modules;
- modules to save;
- runtime precision/options.

Results must be traceable to a concrete configuration.

## Ablation

At least one meaningful ablation is required.

Initial plan:

~~~text
LoRA rank 4
LoRA rank 8
LoRA rank 16
~~~

The final choice must be based on observed evidence.

## Error Analysis

The final study will inspect:

- false positives and false negatives;
- most-confused intent pairs;
- short versus long inputs;
- semantically adjacent intents;
- high-confidence wrong predictions;
- examples that change between baseline and LoRA;
- intents that improve or regress after adaptation.

Failure categories must emerge from observed errors.

## Inference

A small CLI is the preferred inference surface.

It will support:

- loading the base model;
- loading the adapter;
- classifying a text input;
- reporting the predicted intent;
- exposing confidence/probabilities where appropriate;
- identifying the model and adapter used.

Representative baseline-versus-adapted comparisons should be reproducible.

## Explicit Non-Goals

The core project does not include:

- RAG;
- agents;
- LangChain/LangGraph;
- vector databases;
- frontend/UI;
- SaaS;
- unnecessary microservices;
- Kubernetes;
- distributed training;
- a general MLOps platform;
- multi-model orchestration.

These technologies are outside the problem being studied.

## Compute Strategy

The project is designed for constrained, single-person experimentation.

Possible optimizations include:

- PEFT/LoRA;
- gradient accumulation;
- controlled sequence length;
- mixed precision;
- quantized loading when justified;
- small-model selection.

Optimizations must be motivated by a real compute constraint or experimental question.

## QLoRA

QLoRA is not part of the mandatory baseline scope.

It may be added only when it answers a meaningful engineering question under the available hardware. It must not be added solely for technology coverage.

## Evidence Standard

Important claims must be supported by at least one of:

- implementation evidence;
- test evidence;
- experiment evidence;
- literature/vendor documentation.

No unsupported SOTA, production-ready, scalable, robust, or enterprise-grade claims will be made.

## Completion Definition

The project is complete only when the repository contains verified evidence for:

- foundation and configuration;
- dataset engineering;
- baselines;
- LoRA adaptation;
- evaluation;
- ablation;
- error analysis;
- inference;
- reproducibility;
- tests and CI;
- methodology and limitations;
- final portfolio documentation.

A file existing is not sufficient evidence of completion.

## External References

- BANKING77: https://huggingface.co/datasets/PolyAI/banking77
- DeBERTa-v3-small: https://huggingface.co/microsoft/deberta-v3-small
- Transformers PEFT: https://huggingface.co/docs/transformers/peft
- PEFT LoRA: https://huggingface.co/docs/peft/main/package_reference/lora
- PEFT DeBERTa sequence-classification troubleshooting: https://huggingface.co/docs/peft/en/developer_guides/troubleshooting

## Scope Decision

**Approved planning choice:** BANKING77 intent classification with microsoft/deberta-v3-small, using LoRA as the primary adaptation method and a classical + frozen-representation baseline ladder.

This is a planning decision. Actual feasibility and performance remain to be measured.
