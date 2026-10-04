from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .data import INTENTS
from .modeling import load_tokenizer, require_transformers

@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float

def load_adapter_model(
    base_model: str,
    base_revision: str,
    adapter_dir: str,
) -> tuple[Any, Any]:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("PEFT is required for adapter inference.") from exc
    _, AutoModelForSequenceClassification, _ = require_transformers()
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        revision=base_revision,
        num_labels=len(INTENTS),
        id2label={i: label for i, label in enumerate(INTENTS)},
        label2id={label: i for i, label in enumerate(INTENTS)},
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    tokenizer = load_tokenizer(base_model, base_revision)
    return model, tokenizer

def predict(model: Any, tokenizer: Any, text: str) -> Prediction:
    import torch
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        output = model(**encoded)
    probabilities = torch.softmax(output.logits, dim=-1)[0]
    index = int(torch.argmax(probabilities).item())
    return Prediction(
        INTENTS[index],
        float(probabilities[index].item()),
    )
