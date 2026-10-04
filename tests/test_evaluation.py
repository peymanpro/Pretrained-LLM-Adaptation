from pretrained_llm_adaptation.evaluation import classification_metrics

def test_metrics_are_deterministic() -> None:
    metrics = classification_metrics(
        ["a", "a", "b"],
        ["a", "b", "b"],
    )
    assert metrics.accuracy == 2 / 3
    assert round(metrics.macro_f1, 6) == round(
        (2 / 3 + 2 / 3) / 2,
        6,
    )
