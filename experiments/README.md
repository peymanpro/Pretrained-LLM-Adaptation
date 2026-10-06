# Experiment Artifacts

Files under this directory are generated evidence from reproducible benchmark runs.

The benchmark workflow keeps the public test set held out during model selection, records validation-based LoRA rank selection, evaluates the selected adapter on the held-out test set, and produces error-analysis, prediction-comparison, model-inspection, and summary artifacts.

Generated benchmark output should be treated as measurement evidence, not source code.

Generated benchmark output is deliberately absent from the repository while no current full benchmark has been verified. Historical artifacts from superseded runs are not treated as current evidence.

The current `main` branch contains the release-candidate implementation and the benchmark publication workflow.

