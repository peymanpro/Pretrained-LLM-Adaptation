#!/usr/bin/env bash
set -euo pipefail

for config in configs/ablation-r4.yaml configs/ablation-r8.yaml configs/ablation-r16.yaml; do
  python scripts/train_lora.py --config "$config"
done
