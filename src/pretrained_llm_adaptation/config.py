from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "PolyAI/banking77"
    source_revision: str = "57ec275d8078af65b7731c2a98be812d844a6d6b"
    raw_dir: str = "data/raw/banking77"
    processed_dir: str = "data/processed/banking77"
    validation_fraction: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    name: str = "microsoft/deberta-v3-small"
    revision: str = "a59be8aa63396e73dbb45a1487e4cde4be98bfa4"
    max_length: int = 128
    ignore_mismatched_sizes: bool = False


@dataclass(frozen=True)
class LoRAConfig:
    enabled: bool = True
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query_proj", "value_proj")
    modules_to_save: tuple[str, ...] = ("classifier", "pooler")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2.0e-5
    head_learning_rate: float | None = 1.0e-4
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 3.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 25
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    output_dir: str = "artifacts/runs"
    load_best_model_at_end: bool = True
    max_train_samples: int | None = None
    max_validation_samples: int | None = None


@dataclass(frozen=True)
class ProjectConfig:
    experiment_name: str = "lora-r8"
    task: str = "banking77-intent-classification"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        raise ValueError("Expected a list of module names")
    return tuple(str(item) for item in value)


def load_config(path: str | Path) -> ProjectConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    dataset = DatasetConfig(**raw.get("dataset", {}))
    model = ModelConfig(**raw.get("model", {}))
    lora_raw = dict(raw.get("lora", {}))
    if "target_modules" in lora_raw:
        lora_raw["target_modules"] = _tuple(lora_raw["target_modules"])
    if "modules_to_save" in lora_raw:
        lora_raw["modules_to_save"] = _tuple(lora_raw["modules_to_save"])
    lora = LoRAConfig(**lora_raw)
    training = TrainingConfig(**raw.get("training", {}))
    return ProjectConfig(
        experiment_name=str(raw.get("experiment_name", "unnamed")),
        task=str(raw.get("task", "banking77-intent-classification")),
        dataset=dataset,
        model=model,
        lora=lora,
        training=training,
    )


def config_to_dict(config: ProjectConfig) -> dict[str, Any]:
    return asdict(config)
