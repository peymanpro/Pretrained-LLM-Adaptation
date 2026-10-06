from pretrained_llm_adaptation.training import _is_no_decay_parameter, _parameter_group_kind


def test_task_parameter_groups_are_separated() -> None:
    assert _parameter_group_kind(
        "base_model.classifier.bias",
        head_learning_rate=0.00005,
        classifier_learning_rate=0.0002,
    ) == "classifier"
    assert _parameter_group_kind(
        "base_model.pooler.dense.weight",
        head_learning_rate=0.00005,
        classifier_learning_rate=0.0002,
    ) == "head"
    assert _parameter_group_kind(
        "base_model.encoder.layer.0.attention.query_proj.lora_A.default.weight",
        head_learning_rate=0.00005,
        classifier_learning_rate=0.0002,
    ) == "base"


def test_no_decay_parameter_detection() -> None:
    assert _is_no_decay_parameter("classifier.bias")
    assert _is_no_decay_parameter("encoder.layer.0.attention.LayerNorm.weight")
    assert _is_no_decay_parameter("encoder.layer.0.attention.Layer_Norm.weight")
    assert not _is_no_decay_parameter("pooler.dense.weight")
