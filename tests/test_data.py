from pathlib import Path

from pretrained_llm_adaptation.data import (
    INTENTS,
    Sample,
    duplicate_texts,
    exclude_overlaps,
    leakage,
    read_categories,
    split_train_validation,
    validate_categories,
    validate_samples,
)


def test_official_intent_set_is_complete() -> None:
    categories = read_categories(Path("tests/fixtures/categories.json"))
    assert len(INTENTS) == 77
    assert validate_categories(categories) == []


def test_validation_and_duplicates() -> None:
    samples = [Sample("same row", "age_limit"), Sample("same row", "age_limit")]
    errors = validate_samples(samples)
    assert errors == []
    assert len(duplicate_texts(samples)) == 1


def test_invalid_samples_are_reported() -> None:
    errors = validate_samples(
        [
            Sample("", "age_limit"),
            Sample("hello", "not_an_intent"),
        ]
    )
    assert any("empty text" in error for error in errors)
    assert any("unknown label" in error for error in errors)


def test_exclude_overlaps_removes_normalized_matches() -> None:
    train = [Sample("Hello", "age_limit"), Sample("Keep", "card_arrival")]
    test = [Sample(" hello ", "age_limit")]
    assert exclude_overlaps(train, test) == [Sample("Keep", "card_arrival")]


def test_leakage_is_case_insensitive() -> None:
    assert leakage(
        [Sample("Hello", "age_limit")],
        [Sample(" hello ", "age_limit")],
    ) == {"hello"}


def test_stratified_split_preserves_labels() -> None:
    samples = [
        Sample(
            f"text {i}",
            "age_limit" if i < 4 else "card_arrival",
        )
        for i in range(8)
    ]
    train, validation = split_train_validation(samples, 0.25, 42)
    assert len(train) == 6
    assert len(validation) == 2
    assert {item.label for item in validation} == {"age_limit", "card_arrival"}
