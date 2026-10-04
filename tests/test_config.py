from pathlib import Path

from pretrained_llm_adaptation.config import load_config


def test_lora_config_loads() -> None:
    config = load_config(Path("configs/lora.yaml"))
    assert config.experiment_name == "lora-r8"
    assert config.lora.r == 8
    assert config.lora.target_modules == ("query_proj", "value_proj")
