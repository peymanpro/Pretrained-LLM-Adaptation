from pretrained_llm_adaptation.baselines import (
    TfidfBaseline,
    majority_predictions,
)
from pretrained_llm_adaptation.data import Sample


def dataset() -> tuple[list[Sample], list[Sample]]:
    train = [
        Sample("card arrived", "card_arrival"),
        Sample("card late", "card_arrival"),
        Sample("age requirement", "age_limit"),
    ]
    test = [
        Sample("card arrival", "card_arrival"),
        Sample("age requirement", "age_limit"),
    ]
    return train, test


def test_majority_baseline() -> None:
    train, test = dataset()
    assert majority_predictions(train, test) == [
        "card_arrival",
        "card_arrival",
    ]


def test_tfidf_baseline() -> None:
    train, test = dataset()
    model = TfidfBaseline(max_features=100)
    model.fit(train)
    predictions = model.predict(test)
    assert len(predictions) == len(test)
    assert set(predictions) <= {"card_arrival", "age_limit"}
