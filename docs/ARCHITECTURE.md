# Architecture

The repository is organized around one narrow ML lifecycle: data preparation, baseline evaluation, parameter-efficient adaptation, evaluation, and inference.

~~~text
                  ┌─────────────────────┐
                  │  PolyAI BANKING77   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Data validation +   │
                  │ deterministic split │
                  └──────────┬──────────┘
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
     ┌──────────────────┐         ┌──────────────────┐
     │ Classical        │         │ DeBERTa-v3-small │
     │ baselines        │         │ pretrained model │
     │ majority / TFIDF │         └────────┬─────────┘
     └────────┬─────────┘                  │
              │                            ▼
              │                    ┌─────────────────┐
              │                    │ LoRA / PEFT     │
              │                    └────────┬────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                    ┌──────────────────┐
                    │ Shared evaluation│
                    │ metrics + errors │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Inference CLI    │
                    │ base + adapter   │
                    └──────────────────┘
~~~

## Module boundaries

- config.py contains declarative experiment configuration and parsing.
- data.py contains dataset acquisition, validation, leakage checks, and deterministic splitting.
- baselines.py contains non-Transformer sanity and classical baselines.
- modeling.py contains lazy Hugging Face model/tokenizer loading and architecture inspection.
- lora.py contains the PEFT adapter configuration and parameter accounting.
- training.py contains the reusable LoRA and frozen-encoder training pipelines.
- evaluation.py contains task-level metrics and serializable reports.
- inference.py contains clean base-model plus adapter loading and prediction.

Optional ML dependencies are imported lazily in model/training paths so data-quality and classical unit tests remain cheap and deterministic.
