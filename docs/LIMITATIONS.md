# Limitations

- BANKING77 is English and single-domain; conclusions do not automatically generalize to multilingual or multi-domain intent classification.
- The dataset is relatively small for modern language-model adaptation and contains customer-service style queries.
- DeBERTa-v3-small is an encoder model, so this project studies classification adaptation rather than generative instruction tuning.
- Full fine-tuning may be omitted when available hardware cannot support a controlled comparison.
- QLoRA is conditional and is intentionally excluded unless quantized loading answers a useful compute question.
- The repository does not claim state-of-the-art performance. Claims are limited to measured experiments in this repository.
- Large ML dependencies and model weights are not required for the cheap unit-test layer. End-to-end training must be run in an environment capable of installing the declared ML dependencies.
