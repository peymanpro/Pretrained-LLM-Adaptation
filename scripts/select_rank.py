from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretrained_llm_adaptation.selection import choose_best_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Select LoRA rank by validation macro F1")
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/selected-rank.json"),
    )
    args = parser.parse_args()

    candidates = []
    for report_path in args.reports:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = payload["report"]["metrics"]
        rank = int(
            report_path.stem.split("ablation-r", 1)[1].split("-validation", 1)[0]
        )
        candidates.append(
            {
                "rank": rank,
                "macro_f1": float(metrics["macro_f1"]),
                "accuracy": float(metrics["accuracy"]),
                "report": str(report_path),
            }
        )

    selected_rank = choose_best_rank(args.reports)
    selected = next(item for item in candidates if item["rank"] == selected_rank)
    result = {
        "selection_metric": "macro_f1",
        "candidates": sorted(candidates, key=lambda item: item["rank"]),
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(selected_rank)


if __name__ == "__main__":
    main()
