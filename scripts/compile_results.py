from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(name: str, payload: dict[str, Any], run_dir: Path | None = None) -> dict[str, Any]:
    metrics = payload["report"]["metrics"] if "report" in payload else payload["metrics"]
    row: dict[str, Any] = {
        "experiment": name,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
    }
    if run_dir is not None:
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.exists():
            metadata = _load(metadata_path)
            stats = metadata.get("trainable_parameters", {})
            row["trainable_parameters"] = stats.get("trainable_parameters")
            row["trainable_percentage"] = stats.get("trainable_percentage")
            row["training_runtime_seconds"] = metadata.get("training_result", {}).get(
                "train_runtime"
            )
    return row


def _fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile measured benchmark artifacts")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    experiments = args.root / "experiments"
    runs = args.root / "artifacts" / "runs"
    rows: list[dict[str, Any]] = []

    baseline_path = experiments / "baselines.json"
    if baseline_path.exists():
        baseline = _load(baseline_path)
        rows.append(_row("B0 Majority", baseline["majority"]))
        rows.append(_row("B1 TF-IDF + LR", baseline["tfidf_logistic_regression"]))

    frozen_path = experiments / "frozen-evaluation.json"
    if frozen_path.exists():
        rows.append(_row("B2 Frozen DeBERTa", _load(frozen_path), runs / "frozen-deberta"))

    for path in sorted(experiments.glob("ablation-r*-evaluation.json")):
        rank = path.stem.split("ablation-r", 1)[1].split("-evaluation", 1)[0]
        rows.append(_row(f"A1 LoRA r={rank}", _load(path), runs / f"ablation-r{rank}"))

    final_paths = sorted(experiments.glob("selected-r*-final-evaluation.json"))
    for path in final_paths:
        rank = path.stem.split("selected-r", 1)[1].split("-final-evaluation", 1)[0]
        rows.append(
            _row(
                f"A1 Final LoRA r={rank}",
                _load(path),
                runs / f"selected-r{rank}-final",
            )
        )

    selected_path = experiments / "selected-rank.json"
    selected = _load(selected_path)["selected"] if selected_path.exists() else None

    result = {
        "selection": selected,
        "experiments": rows,
        "evidence_rule": (
            "Only artifacts produced by this repository's benchmark workflow are included."
        ),
    }

    experiments.mkdir(parents=True, exist_ok=True)
    (experiments / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Experiment Results",
        "",
        "| Experiment | Accuracy | Macro F1 | Weighted F1 | Trainable Params | Trainable % | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["experiment"],
                    _fmt(row["accuracy"]),
                    _fmt(row["macro_f1"]),
                    _fmt(row["weighted_f1"]),
                    _fmt(row.get("trainable_parameters")),
                    _fmt(row.get("trainable_percentage")),
                    _fmt(row.get("training_runtime_seconds")),
                ]
            )
            + " |"
        )
    if selected is not None:
        lines.extend(
            [
                "",
                f"Selected LoRA rank: **{selected['rank']}** by validation macro F1.",
            ]
        )

    (experiments / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
