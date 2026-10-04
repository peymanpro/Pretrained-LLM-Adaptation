from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import Sample
from .evaluation import ClassificationMetrics, classification_metrics


@dataclass
class TfidfBaseline:
    max_features: int = 10000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1

    def __post_init__(self) -> None:
        self.pipeline: Any = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=self.ngram_range,
                        min_df=self.min_df,
                        max_features=self.max_features,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def fit(self, samples: list[Sample]) -> None:
        self.pipeline.fit(
            [sample.text for sample in samples],
            [sample.label for sample in samples],
        )

    def predict(self, samples: list[Sample]) -> list[str]:
        return [str(value) for value in self.pipeline.predict([sample.text for sample in samples])]

    def evaluate(self, samples: list[Sample]) -> ClassificationMetrics:
        return classification_metrics(
            [sample.label for sample in samples],
            self.predict(samples),
        )


def majority_predictions(
    train: list[Sample],
    samples: list[Sample],
) -> list[str]:
    if not train:
        raise ValueError("training samples cannot be empty")
    counts: dict[str, int] = {}
    for sample in train:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    majority = min(counts, key=lambda label: (-counts[label], label))
    return [majority for _ in samples]
