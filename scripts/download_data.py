from __future__ import annotations

import argparse

from pretrained_llm_adaptation.config import load_config
from pretrained_llm_adaptation.data import download_banking77


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = download_banking77(
        config.dataset.raw_dir,
        config.dataset.source_revision,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
