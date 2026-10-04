# Experiment Artifacts

Generated outputs belong under this directory hierarchy, while large model weights and raw datasets are intentionally excluded from Git.

Recommended layout:

~~~text
artifacts/
└── runs/
    └── <experiment-name>/
        ├── run_metadata.json
        ├── adapter_config.json
        └── adapter_model.safetensors
~~~

Evaluation reports and lightweight analysis outputs belong under experiments/ and must reference the configuration used to produce them.
