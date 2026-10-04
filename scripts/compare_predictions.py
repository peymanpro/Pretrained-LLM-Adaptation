from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["predictions"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected prediction rows in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two aligned prediction artifacts.",
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = _read(args.reference)
    candidate = _read(args.candidate)

    if len(reference) != len(candidate):
        raise ValueError("Prediction artifacts contain different numbers of rows")

    changes: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        if left.get("text") != right.get("text") or left.get("gold") != right.get("gold"):
            raise ValueError(f"Prediction alignment mismatch at row {index}")
        if left.get("predicted") != right.get("predicted"):
            changes.append(
                {
                    "index": index,
                    "text": left["text"],
                    "gold": left["gold"],
                    "reference_prediction": left["predicted"],
                    "reference_confidence": left.get("confidence"),
                    "candidate_prediction": right["predicted"],
                    "candidate_confidence": right.get("confidence"),
                }
            )

    result = {
        "total": len(reference),
        "changed_predictions": len(changes),
        "change_rate": len(changes) / len(reference) if reference else None,
        "changes": changes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
