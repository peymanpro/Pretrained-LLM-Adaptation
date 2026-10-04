from __future__ import annotations

import json
from pathlib import Path


def choose_best_rank(reports: list[str | Path]) -> int:
    candidates: list[tuple[float, float, int]] = []
    for report_path in reports:
        path = Path(report_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["report"]["metrics"]
        rank = int(path.stem.split("ablation-r", 1)[1].split("-validation", 1)[0])
        candidates.append(
            (float(metrics["macro_f1"]), float(metrics["accuracy"]), -rank)
        )
    if not candidates:
        raise ValueError("at least one validation report is required")
    return -max(candidates)[2]
