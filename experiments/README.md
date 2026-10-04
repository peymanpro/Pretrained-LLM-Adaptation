# Experiment Artifacts

Files under this directory are generated evidence from reproducible benchmark runs.

The benchmark workflow keeps the public test set held out during model selection, records validation-based LoRA rank selection, evaluates the selected adapter on the held-out test set, and produces error-analysis, prediction-comparison, model-inspection, and summary artifacts.

Generated benchmark output should be treated as measurement evidence, not source code.

Latest verified pre-benchmark commit: `666b86b895bac34ea75c8dda31d7dbd40285f56f` (CI + ML smoke passed with the DeBERTa SentencePiece tokenizer path).

Benchmark trigger revision: verified CI and ML smoke on commit `3a946ba53f2490eaee072d72bb0f0eef80f139e2`.

