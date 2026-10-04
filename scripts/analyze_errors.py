from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize structured classification errors")
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/error_analysis.json"),
    )
    args = parser.parse_args()
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows["predictions"]
    errors = [row for row in rows if row["gold"] != row["predicted"]]
    confusion = Counter((row["gold"], row["predicted"]) for row in errors)
    payload = {
        "total": len(rows),
        "errors": len(errors),
        "error_rate": len(errors) / len(rows) if rows else None,
        "top_confusions": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in confusion.most_common(20)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
