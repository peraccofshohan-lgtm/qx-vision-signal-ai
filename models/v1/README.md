# V1 model registry

No production model artifact is checked in. `feature_schema.json` is the versioned contract; it is not predictive weights.

A real-data training run writes a challenger under `ensemble/candidate/`. It must contain a calibrated candidate, validation/holdout metadata, and a passing `onnx_parity.json` before `training.promote` may copy the model into `ensemble/`. The Android application intentionally abstains while `ensemble/model.onnx` and its matching checksummed metadata are absent.
