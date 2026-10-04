from pretrained_llm_adaptation.data import (
    Sample,
    duplicate_texts,
    leakage,
    split_train_validation,
    validate_samples,
)

def test_validation_and_duplicates() -> None:
    samples = [
        Sample("Hello", "age_limit"),
        Sample("hello", "age_limit"),
        Sample("", "bad"),
    ]
    errors = validate_samples(samples)
    assert any("empty text" in error for error in errors)
    assert any("unknown label" in error for error in errors)
    assert len(duplicate_texts(samples)) == 1

def test_leakage_is_case_insensitive() -> None:
    assert leakage(
        [Sample("Hello", "age_limit")],
        [Sample(" hello ", "age_limit")],
    ) == {"hello"}

def test_stratified_split_preserves_labels() -> None:
    samples = [
        Sample(f"text {i}", "age_limit" if i < 4 else "card_arrival")
        for i in range(8)
    ]
    train, validation = split_train_validation(samples, 0.25, 42)
    assert len(train) == 6
    assert len(validation) == 2
    assert {item.label for item in validation} == {"age_limit", "card_arrival"}
