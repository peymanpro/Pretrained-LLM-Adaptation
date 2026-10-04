from __future__ import annotations

from typing import Any


def verify_target_modules(model: Any, target_modules: tuple[str, ...]) -> None:
    names = [name for name, _ in model.named_modules()]
    suffixes = {name.rsplit(".", 1)[-1] for name in names}
    missing = [target for target in target_modules if target not in suffixes]
    if missing:
        raise ValueError("LoRA target modules were not found in the model: " + ", ".join(missing))


def apply_lora(
    model: Any,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: tuple[str, ...],
    modules_to_save: tuple[str, ...],
) -> Any:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("PEFT is required for LoRA operations.") from exc

    verify_target_modules(model, target_modules)
    verify_target_modules(model, modules_to_save)
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
        modules_to_save=list(modules_to_save),
    )
    return get_peft_model(model, config)


def trainable_parameter_stats(model: Any) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_percentage": (100.0 * trainable / total) if total else 0.0,
    }
