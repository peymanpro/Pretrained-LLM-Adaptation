from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretrained_llm_adaptation.baselines import TfidfBaseline, majority_predictions
from pretrained_llm_adaptation.config import load_config
from pretrained_llm_adaptation.data import Sample, read_jsonl
from pretrained_llm_adaptation.evaluation import detailed_report, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default="experiments/baselines.json")
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = Path(args.data_dir or config.dataset.processed_dir)
    train = read_jsonl(data_dir / "train.jsonl")
    test = read_jsonl(data_dir / "test.jsonl")
    labels = sorted({sample.label for sample in train + test})

    majority = majority_predictions(train, test)
    majority_report = detailed_report(
        [sample.label for sample in test],
        majority,
        labels,
    )

    model = TfidfBaseline()
    model.fit(train)
    predictions = model.predict(test)
    tfidf_report = detailed_report(
        [sample.label for sample in test],
        predictions,
        labels,
    )

    payload = {
        "majority": majority_report,
        "tfidf_logistic_regression": tfidf_report,
    }
    save_json(payload, args.output)
    print(
        json.dumps(
            {name: block["metrics"] for name, block in payload.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
