from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ProjectConfig, config_to_dict
from .data import INTENTS, Sample
from .evaluation import classification_metrics
from .lora import apply_lora, trainable_parameter_stats
from .modeling import load_sequence_classifier, load_tokenizer
from .seed import set_seed




def _limit_samples(
    samples: list[Sample],
    limit: int | None,
    seed: int,
) -> list[Sample]:
    if limit is None or limit >= len(samples):
        return samples
    if limit < 1:
        raise ValueError("sample limits must be positive")
    labels = [sample.label for sample in samples]
    if limit < len(set(labels)):
        raise ValueError("sample limit must be at least the number of classes")
    from sklearn.model_selection import train_test_split

    selected, _ = train_test_split(
        samples,
        train_size=limit,
        random_state=seed,
        stratify=labels,
    )
    return list(selected)


def _tokenize(tokenizer: Any, samples: list[Sample], max_length: int) -> Any:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for training.") from exc
    dataset = Dataset.from_dict(
        {
            "text": [sample.text for sample in samples],
            "labels": [INTENTS.index(sample.label) for sample in samples],
        }
    )
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def _trainer(
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    val_ds: Any,
    config: ProjectConfig,
    output_dir: Path,
) -> Any:
    try:
        from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Transformers is required for training.") from exc

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        eval_strategy=config.training.eval_strategy,
        save_strategy=config.training.save_strategy,
        logging_steps=config.training.logging_steps,
        seed=config.training.seed,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        load_best_model_at_end=config.training.load_best_model_at_end,
        report_to=[],
    )

    def metrics(eval_prediction: Any) -> dict[str, float]:
        import numpy as np

        logits, labels = eval_prediction
        predicted = np.argmax(logits, axis=-1)
        scores = classification_metrics(
            [INTENTS[int(value)] for value in labels],
            [INTENTS[int(value)] for value in predicted],
        )
        return {
            "accuracy": scores.accuracy,
            "macro_f1": scores.macro_f1,
            "weighted_f1": scores.weighted_f1,
        }

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
    )


def _load_model(config: ProjectConfig) -> tuple[Any, Any]:
    labels = {index: label for index, label in enumerate(INTENTS)}
    inverse = {label: index for index, label in labels.items()}
    model = load_sequence_classifier(
        config.model.name,
        config.model.revision,
        len(INTENTS),
        labels,
        inverse,
    )
    return model, load_tokenizer(config.model.name, config.model.revision)


def _write_metadata(
    output_dir: Path,
    config: ProjectConfig,
    method: str,
    model: Any,
    train_size: int,
    validation_size: int,
    training_result: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "experiment": config.experiment_name,
        "method": method,
        "model": config.model.name,
        "model_revision": config.model.revision,
        "config": config_to_dict(config),
        "trainable_parameters": trainable_parameter_stats(model),
        "train_samples": train_size,
        "validation_samples": validation_size,
        "training_result": training_result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return metadata


def train_lora(
    config: ProjectConfig,
    train: list[Sample],
    validation: list[Sample],
) -> dict[str, Any]:
    set_seed(config.training.seed)
    train = _limit_samples(train, config.training.max_train_samples, config.training.seed)
    validation = _limit_samples(
        validation,
        config.training.max_validation_samples,
        config.training.seed,
    )
    model, tokenizer = _load_model(config)
    model = apply_lora(
        model,
        r=config.lora.r,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        modules_to_save=config.lora.modules_to_save,
    )
    train_ds = _tokenize(tokenizer, train, config.model.max_length)
    val_ds = _tokenize(tokenizer, validation, config.model.max_length)
    output_dir = Path(config.training.output_dir) / config.experiment_name
    trainer = _trainer(model, tokenizer, train_ds, val_ds, config, output_dir)
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    return _write_metadata(
        output_dir,
        config,
        "lora",
        model,
        len(train),
        len(validation),
        result.metrics,
    )


def train_frozen(
    config: ProjectConfig,
    train: list[Sample],
    validation: list[Sample],
) -> dict[str, Any]:
    set_seed(config.training.seed)
    model, tokenizer = _load_model(config)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, module in model.named_modules():
        if name == "classifier" or name.endswith(".classifier"):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if name == "pooler" or name.endswith(".pooler"):
            for parameter in module.parameters():
                parameter.requires_grad = True

    train_ds = _tokenize(tokenizer, train, config.model.max_length)
    val_ds = _tokenize(tokenizer, validation, config.model.max_length)
    output_dir = Path(config.training.output_dir) / f"{config.experiment_name}-frozen"
    trainer = _trainer(model, tokenizer, train_ds, val_ds, config, output_dir)
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    return _write_metadata(
        output_dir,
        config,
        "frozen-encoder",
        model,
        len(train),
        len(validation),
        result.metrics,
    )
