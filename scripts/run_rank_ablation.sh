#!/usr/bin/env bash
set -euo pipefail

for config in configs/lora-r4.yaml configs/lora-r8.yaml configs/lora-r16.yaml; do
  python scripts/train_lora.py --config "$config"
done
