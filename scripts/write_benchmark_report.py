from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the final benchmark report.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="unknown")
    args = parser.parse_args()

    summary = _load(args.summary)
    selection = _load(args.selection)
    errors = _load(args.errors)
    selected = selection["selected"]
    rows = summary.get("experiments", [])
    frozen = next(
        (row for row in rows if row["experiment"] == "B2 Frozen DeBERTa"),
        None,
    )
    final = next(
        (row for row in rows if "Final LoRA" in row["experiment"]),
        None,
    )

    lines = [
        "# Benchmark Report",
        "",
        f"GitHub Actions run: {args.run_id}",
        "",
        "## Protocol",
        "",
        (
            "Resource-bounded benchmark; rank selection uses validation only; final "
            "metrics use the full public BANKING77 test split."
        ),
        "",
        "## Selected rank",
        "",
        f"- LoRA rank: **{selected['rank']}**",
        f"- Validation macro F1: **{selected['macro_f1']:.4f}**",
        f"- Validation accuracy: **{selected['accuracy']:.4f}**",
        "",
        "## Final held-out result",
        "",
    ]
    if final:
        lines.extend(
            [
                "| Metric | Value |",
                "|---|---:|",
                f"| Accuracy | {final['accuracy']:.4f} |",
                f"| Macro F1 | {final['macro_f1']:.4f} |",
                f"| Weighted F1 | {final['weighted_f1']:.4f} |",
                f"| Trainable parameters | {final.get('trainable_parameters', 'n/a')} |",
                f"| Trainable percentage | {final.get('trainable_percentage', 'n/a')} |",
                f"| Training runtime (s) | {final.get('training_runtime_seconds', 'n/a')} |",
                "",
            ]
        )
    if frozen:
        lines.extend(
            [
                "## Frozen baseline",
                "",
                f"- Accuracy: **{frozen['accuracy']:.4f}**",
                f"- Macro F1: **{frozen['macro_f1']:.4f}**",
                f"- Weighted F1: **{frozen['weighted_f1']:.4f}**",
                "",
            ]
        )
    lines.extend(
        [
            "## Error analysis",
            "",
            f"- Test examples: **{errors.get('total', 0)}**",
            f"- Errors: **{errors.get('errors', 0)}**",
            f"- Error rate: **{errors.get('error_rate', 0.0):.4f}**",
            "",
            "### Top confusion pairs",
            "",
            "| Gold | Predicted | Count |",
            "|---|---|---:|",
        ]
    )
    for row in errors.get("top_confusions", [])[:10]:
        lines.append(f"| {row['gold']} | {row['predicted']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This benchmark is evidence of a reproducible adaptation pipeline, not "
                "a state-of-the-art claim. Training is deliberately resource-bounded."
            ),
            "",
            "## Provenance",
            "",
            (
                "Model/data revisions and training configuration are retained "
                "in run metadata artifacts."
            ),
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
