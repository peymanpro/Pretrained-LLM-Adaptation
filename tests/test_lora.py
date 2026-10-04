from pretrained_llm_adaptation.lora import trainable_parameter_stats

class Parameter:
    def __init__(self, count: int, trainable: bool) -> None:
        self._count = count
        self.requires_grad = trainable
    def numel(self) -> int:
        return self._count

class Model:
    def parameters(self):
        return [Parameter(100, False), Parameter(20, True)]

def test_trainable_parameter_stats() -> None:
    stats = trainable_parameter_stats(Model())
    assert stats == {
        "total_parameters": 120,
        "trainable_parameters": 20,
        "frozen_parameters": 100,
        "trainable_percentage": 100 * 20 / 120,
    }
