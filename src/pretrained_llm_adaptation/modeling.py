from __future__ import annotations

from typing import Any


def require_transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required for model operations. Install project dependencies first."
        ) from exc
    return AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding


def load_sequence_classifier(
    model_name: str,
    revision: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> Any:
    _, AutoModelForSequenceClassification, _ = require_transformers()
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=revision,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )


def load_tokenizer(
    model_name: str,
    revision: str,
) -> Any:
    AutoTokenizer, _, _ = require_transformers()
    return AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
    )


def inspect_lora_modules(model: Any) -> dict[str, list[str]]:
    linear_names: list[str] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in {"Linear", "Dense"}:
            linear_names.append(name)
    return {
        "linear_modules": [name for name in linear_names],
        "classifier_candidates": [
            name for name, _ in model.named_modules() if "classifier" in name.lower()
        ],
        "pooler_candidates": [
            name for name, _ in model.named_modules() if "pooler" in name.lower()
        ],
    }
