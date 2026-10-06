from pretrained_llm_adaptation.training import _is_head_parameter, _is_no_decay_parameter


def test_task_head_parameters_are_separated() -> None:
    assert _is_head_parameter("base_model.pooler.dense.weight")
    assert _is_head_parameter("base_model.classifier.bias")
    assert not _is_head_parameter("base_model.encoder.layer.0.attention.query_proj.lora_A.default.weight")


def test_no_decay_parameter_detection() -> None:
    assert _is_no_decay_parameter("classifier.bias")
    assert _is_no_decay_parameter("encoder.layer.0.attention.LayerNorm.weight")
    assert _is_no_decay_parameter("encoder.layer.0.attention.Layer_Norm.weight")
    assert not _is_no_decay_parameter("pooler.dense.weight")
