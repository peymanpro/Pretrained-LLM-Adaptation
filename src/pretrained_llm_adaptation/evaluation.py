from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )


def detailed_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, object]:
    metrics = classification_metrics(y_true, y_pred)
    return {
        "metrics": asdict(metrics),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(labels),
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(labels),
        ).tolist(),
        "labels": list(labels),
    }


def save_json(payload: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
