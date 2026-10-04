from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pretrained_llm_adaptation.config import config_to_dict, load_config
from pretrained_llm_adaptation.data import read_jsonl
from pretrained_llm_adaptation.training import train_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config = replace(
            config,
            training=replace(config.training, num_train_epochs=args.epochs),
        )
    if args.experiment_name is not None:
        config = replace(config, experiment_name=args.experiment_name)
    data_dir = Path(args.data_dir or config.dataset.processed_dir)
    metadata = train_lora(
        config,
        read_jsonl(data_dir / "train.jsonl"),
        read_jsonl(data_dir / "validation.jsonl"),
    )
    print(
        json.dumps(
            {"config": config_to_dict(config), "metadata": metadata},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
