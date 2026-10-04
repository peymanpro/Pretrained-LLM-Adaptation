# Experiment Artifacts

Files under this directory are generated evidence from reproducible benchmark runs.

The benchmark workflow keeps the public test set held out during model selection, records validation-based LoRA rank selection, evaluates the selected adapter on the held-out test set, and produces error-analysis, prediction-comparison, model-inspection, and summary artifacts.

Generated benchmark output should be treated as measurement evidence, not source code.
