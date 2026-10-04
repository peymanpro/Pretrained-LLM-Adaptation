import json
import subprocess
import sys
from pathlib import Path


def test_error_analysis_reports_observed_failure_structure(tmp_path: Path) -> None:
    source = {
        "predictions": [
            {
                "text": "short query",
                "gold": "card_arrival",
                "predicted": "card_arrival",
                "confidence": 0.99,
            },
            {
                "text": "A deliberately longer incorrect prediction example for analysis",
                "gold": "card_arrival",
                "predicted": "cash_withdrawal",
                "confidence": 0.91,
            },
        ]
    }
    source_path = tmp_path / "predictions.json"
    output_path = tmp_path / "analysis.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_errors.py",
            str(source_path),
            "--output",
            str(output_path),
        ],
        check=True,
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["total"] == 2
    assert result["errors"] == 1
    assert result["top_confusions"][0]["gold"] == "card_arrival"
    assert result["high_confidence_errors"][0]["predicted"] == "cash_withdrawal"


def test_compare_predictions_rejects_misaligned_inputs(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    output = tmp_path / "changes.json"

    left.write_text(
        json.dumps({"predictions": [{"text": "hello", "gold": "a", "predicted": "a"}]}),
        encoding="utf-8",
    )
    right.write_text(
        json.dumps({"predictions": [{"text": "different", "gold": "a", "predicted": "b"}]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_predictions.py",
            str(left),
            str(right),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
