from pretrained_llm_adaptation.inference import Prediction


def test_prediction_shape() -> None:
    prediction = Prediction("card_arrival", 0.91)
    assert prediction.intent == "card_arrival"
    assert 0 <= prediction.confidence <= 1
