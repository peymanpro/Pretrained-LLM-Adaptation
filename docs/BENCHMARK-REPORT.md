# Benchmark Report

GitHub Actions run: 37411798932

## Protocol

Resource-bounded benchmark; rank selection uses validation only; final metrics use the full public BANKING77 test split.

## Selected rank

- LoRA rank: **8**
- Validation macro F1: **0.0011**
- Validation accuracy: **0.0195**

## Final held-out result

| Metric | Value |
|---|---:|
| Accuracy | 0.0130 |
| Macro F1 | 0.0003 |
| Weighted F1 | 0.0003 |
| Trainable parameters | 797261 |
| Trainable percentage | 0.5584961535855071 |
| Training runtime (s) | 3671.9411 |

## Frozen baseline

- Accuracy: **0.0130**
- Macro F1: **0.0003**
- Weighted F1: **0.0003**

## Error analysis

- Test examples: **3080**
- Errors: **3040**
- Error rate: **0.9870**

### Top confusion pairs

| Gold | Predicted | Count |
|---|---|---:|
| card_arrival | pending_transfer | 40 |
| card_linking | pending_transfer | 40 |
| exchange_rate | pending_transfer | 40 |
| card_payment_wrong_exchange_rate | pending_transfer | 40 |
| extra_charge_on_statement | pending_transfer | 40 |
| pending_cash_withdrawal | pending_transfer | 40 |
| fiat_currency_support | pending_transfer | 40 |
| card_delivery_estimate | pending_transfer | 40 |
| automatic_top_up | pending_transfer | 40 |
| card_not_working | pending_transfer | 40 |

## Interpretation

This benchmark is evidence of a reproducible adaptation pipeline, not a state-of-the-art claim. Training is deliberately resource-bounded.

## Provenance

Model/data revisions and training configuration are retained in run metadata artifacts.
