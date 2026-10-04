from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pretrained_llm_adaptation.config import load_config
from pretrained_llm_adaptation.data import (
    INTENTS,
    Sample,
    duplicate_texts,
    leakage,
    read_categories,
    read_csv,
    validate_categories,
    split_train_validation,
    validate_samples,
    write_manifest,
)


def _write_jsonl(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    {"text": sample.text, "label": sample.label},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    raw = Path(config.dataset.raw_dir)
    train = read_csv(raw / "train.csv")
    test = read_csv(raw / "test.csv")

    categories = read_categories(raw / "categories.json")
    category_errors = validate_categories(categories)
    if category_errors:
        raise ValueError("Category validation failed: " + "; ".join(category_errors))

    errors = validate_samples(train) + validate_samples(test)
    if errors:
        raise ValueError("Dataset validation failed: " + "; ".join(errors[:10]))

    train_test_leakage = leakage(train, test)
    train_labels = {sample.text.strip().casefold(): sample.label for sample in train}
    test_labels = {sample.text.strip().casefold(): sample.label for sample in test}
    conflicting_labels = {
        text
        for text in train_test_leakage
        if train_labels[text] != test_labels[text]
    }
    if conflicting_labels:
        raise ValueError(
            "Conflicting train/test labels for identical text: "
            f"{len(conflicting_labels)} cases"
        )
    if train_test_leakage:
        print(
            "warning: official BANKING77 split contains "
            f"{len(train_test_leakage)} exact train/test text overlaps"
        )

    duplicates = duplicate_texts(train)
    if duplicates:
        print(f"warning: {len(duplicates)} duplicated training texts detected")

    fit, validation = split_train_validation(
        train,
        config.dataset.validation_fraction,
        config.dataset.seed,
    )
    fit_validation_leakage = leakage(fit, validation)
    if fit_validation_leakage:
        raise ValueError(
            f"Fit/validation text leakage detected: {len(fit_validation_leakage)} overlaps"
        )
    output = Path(config.dataset.processed_dir)
    _write_jsonl(output / "train.jsonl", fit)
    _write_jsonl(output / "validation.jsonl", validation)
    _write_jsonl(output / "test.jsonl", test)
    write_manifest(
        output / "manifest.json",
        config.dataset.source_revision,
        {"train": fit, "validation": validation, "test": test},
        config.dataset.validation_fraction,
        config.dataset.seed,
        train_test_overlap_count=len(train_test_leakage),
        fit_validation_overlap_count=len(fit_validation_leakage),
    )
    print(
        "counts:",
        Counter(sample.label for sample in train).most_common(3),
        "fit=",
        len(fit),
        "validation=",
        len(validation),
        "test=",
        len(test),
        "labels=",
        len(INTENTS),
    )


if __name__ == "__main__":
    main()
