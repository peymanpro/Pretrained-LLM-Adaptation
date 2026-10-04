import json

from pretrained_llm_adaptation.selection import choose_best_rank


def test_best_rank_uses_validation_macro_f1(tmp_path) -> None:
    reports = []
    for rank, f1 in [(4, 0.70), (8, 0.75), (16, 0.74)]:
        path = tmp_path / f"ablation-r{rank}-validation.json"
        path.write_text(
            json.dumps({"report": {"metrics": {"macro_f1": f1, "accuracy": 0.8}}}),
            encoding="utf-8",
        )
        reports.append(path)

    assert choose_best_rank(reports) == 8
