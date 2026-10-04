from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretrained_llm_adaptation.config import config_to_dict, load_config
from pretrained_llm_adaptation.data import Sample
from pretrained_llm_adaptation.training import train_lora


def _load_jsonl(path: Path) -> list[Sample]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = Path(args.data_dir or config.dataset.processed_dir)
    metadata = train_lora(
        config,
        _load_jsonl(data_dir / "train.jsonl"),
        _load_jsonl(data_dir / "validation.jsonl"),
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
