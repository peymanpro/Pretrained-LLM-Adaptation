from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _length_bucket(text: str) -> str:
    words = len(text.split())
    if words <= 5:
        return "1-5"
    if words <= 10:
        return "6-10"
    if words <= 20:
        return "11-20"
    return "21+"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze structured classification errors")
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/error_analysis.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = payload["predictions"] if isinstance(payload, dict) else payload
    errors = [row for row in rows if row["gold"] != row["predicted"]]

    confusion = Counter((row["gold"], row["predicted"]) for row in errors)
    errors_by_gold = Counter(row["gold"] for row in errors)
    errors_by_length = Counter(_length_bucket(str(row["text"])) for row in errors)

    high_confidence_errors = sorted(
        errors,
        key=lambda row: float(row.get("confidence", 0.0)),
        reverse=True,
    )[:20]

    result = {
        "total": len(rows),
        "errors": len(errors),
        "error_rate": len(errors) / len(rows) if rows else None,
        "errors_by_gold_intent": [
            {"intent": intent, "count": count}
            for intent, count in errors_by_gold.most_common()
        ],
        "errors_by_length_bucket": dict(sorted(errors_by_length.items())),
        "top_confusions": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in confusion.most_common(20)
        ],
        "high_confidence_errors": high_confidence_errors,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
