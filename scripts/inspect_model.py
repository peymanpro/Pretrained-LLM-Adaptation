from __future__ import annotations

import argparse
import json

from pretrained_llm_adaptation.config import load_config
from pretrained_llm_adaptation.data import INTENTS
from pretrained_llm_adaptation.lora import trainable_parameter_stats
from pretrained_llm_adaptation.modeling import inspect_lora_modules, load_sequence_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    id2label = {index: label for index, label in enumerate(INTENTS)}
    label2id = {label: index for index, label in id2label.items()}
    model = load_sequence_classifier(
        config.model.name,
        config.model.revision,
        len(INTENTS),
        id2label,
        label2id,
    )
    payload = {
        "model": config.model.name,
        "revision": config.model.revision,
        "modules": inspect_lora_modules(model),
        "parameters": trainable_parameter_stats(model),
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        from pathlib import Path

        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
