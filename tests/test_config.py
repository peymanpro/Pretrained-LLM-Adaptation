from pathlib import Path

from pretrained_llm_adaptation.config import ProjectConfig, load_config


def test_lora_config_loads() -> None:
    config = load_config(Path("configs/lora.yaml"))
    assert config.experiment_name == "lora-r8"
    assert config.lora.r == 8
    assert config.lora.target_modules == ("query_proj", "value_proj")
    assert config.model.ignore_mismatched_sizes is False


def test_default_lora_targets_match_deberta() -> None:
    assert ProjectConfig().lora.target_modules == ("query_proj", "value_proj")


def test_all_config_files_parse() -> None:
    config_paths = sorted(Path("configs").glob("*.yaml"))
    assert config_paths
    for path in config_paths:
        config = load_config(path)
        assert config.task == "banking77-intent-classification"
        assert config.model.max_length > 0
