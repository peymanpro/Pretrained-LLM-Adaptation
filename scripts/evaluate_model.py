from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pretrained_llm_adaptation.data import INTENTS, Sample, read_jsonl
from pretrained_llm_adaptation.evaluation import detailed_report, save_json
from pretrained_llm_adaptation.modeling import load_tokenizer


def _predict(
    model: Any,
    tokenizer: Any,
    samples: list[Sample],
    max_length: int,
    batch_size: int,
) -> tuple[list[str], list[float]]:
    import torch

    model.eval()
    predictions: list[str] = []
    confidences: list[float] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        encoded = tokenizer(
            [sample.text for sample in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"Non-finite logits detected during evaluation at batch starting index {start}."
            )
        probabilities = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError(
                "Non-finite probabilities detected during evaluation at "
                f"batch starting index {start}."
            )
        indices = torch.argmax(probabilities, dim=-1)
        predictions.extend(INTENTS[int(index)] for index in indices)
        confidences.extend(
            float(probabilities[row, indices[row]].item()) for row in range(len(batch))
        )
    return predictions, confidences


def _load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers and PEFT are required for evaluation.") from exc

    id2label = {index: label for index, label in enumerate(INTENTS)}
    label2id = {label: index for index, label in id2label.items()}
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        revision=args.revision,
        num_labels=len(INTENTS),
        id2label=id2label,
        label2id=label2id,
    )
    model = (
        PeftModel.from_pretrained(base, args.model_dir)
        if args.adapter
        else AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    )
    tokenizer_path = Path(args.model_dir) / "tokenizer_config.json"
    tokenizer = (
        AutoTokenizer.from_pretrained(args.model_dir)
        if tokenizer_path.exists()
        else load_tokenizer(args.model_name, args.revision)
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument(
        "--model-name",
        default="microsoft/deberta-v3-small",
    )
    parser.add_argument("--revision", default="a59be8aa63396e73dbb45a1487e4cde4be98bfa4")
    parser.add_argument(
        "--data",
        default="data/processed/banking77/test.jsonl",
    )
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="Allow a one-intent prediction collapse for diagnostic baseline/ablation runs.",
    )
    parser.add_argument(
        "--output",
        default="experiments/model-evaluation.json",
    )
    args = parser.parse_args()

    model, tokenizer = _load_model(args)
    samples = read_jsonl(Path(args.data))
    predictions, confidences = _predict(
        model,
        tokenizer,
        samples,
        args.max_length,
        args.batch_size,
    )
    output_name = Path(args.output).name
    diagnostic_output = output_name == "frozen-evaluation.json" or (
        output_name.startswith("ablation-r") and output_name.endswith("-validation.json")
    )
    if len(set(predictions)) < 2 and not (args.allow_degenerate or diagnostic_output):
        raise RuntimeError("Degenerate model output: evaluation predicted only one unique intent.")
    labels = sorted({sample.label for sample in samples})
    report = detailed_report(
        [sample.label for sample in samples],
        predictions,
        labels,
    )
    rows = [
        {
            "text": sample.text,
            "gold": sample.label,
            "predicted": prediction,
            "confidence": confidence,
        }
        for sample, prediction, confidence in zip(
            samples,
            predictions,
            confidences,
            strict=True,
        )
    ]
    save_json(
        {"report": report, "predictions": rows},
        args.output,
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
