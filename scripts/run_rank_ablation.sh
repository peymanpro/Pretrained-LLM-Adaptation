#!/usr/bin/env bash
set -euo pipefail

for config in configs/ablation-r4.yaml configs/ablation-r8.yaml configs/ablation-r16.yaml; do
  name="$(basename "$config" .yaml)"
  python scripts/train_lora.py --config "$config"
  python scripts/evaluate_model.py     --model-dir "artifacts/runs/$name"     --adapter     --data "data/processed/banking77/validation.jsonl"     --output "experiments/$name-validation.json"
done
