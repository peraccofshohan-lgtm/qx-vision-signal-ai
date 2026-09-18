# Architecture

## Implemented flow

`Photo Picker / ACTION_SEND` → bounded bitmap decode → `ChartVisionEngine` → separate running-candle representation → `FeatureEngine` → optional schema-matched ONNX model → calibration → uncertainty and abstention policy → Room immutable prediction record → Compose result/history UI.

The Python package mirrors this contract so vision and training behavior can be tested without an Android device.

## Temporal contract

The right-most visible bar is `isComplete=false` and `partialObservation=true`. Its geometry is retained as explicitly tagged observation features, but OHLC training windows use completed rows only and the next row is the target. Unknown candle progress is never converted into a close. The app does not derive a higher timeframe from a one-minute screenshot.

## Vision

`ChartVisionEngine` estimates the border-background mode, quantized dominant HSV-like color families, colored column runs, body rows, wicks, and normalized price coordinates. No fixed RGB values are required. A candle confidence is reduced for color ambiguity, narrow/merged geometry, and the right edge. Quality reports contain edge energy, coverage, color visibility, resolution adequacy, and failure reasons. This is a geometric extractor, not OCR and not a cloud classifier.

The Python implementation is deterministic and dependency-light; Pillow is optional for formats not supported by its fallback decoder. Android uses `Bitmap` and applies the same geometry rules.

## Feature pipeline

`FeatureEngine` computes:

- normalized body/wick/range ratios and recent returns;
- adaptive prominence swings and HH/HL/LH/LL labels;
- zone-based support/resistance clusters;
- morphology as features, not independent signals;
- momentum, acceleration, efficiency, exhaustion, volatility class;
- regression trend and smoothness;
- regime classification;
- quality and explicitly tagged running-candle features.

`qxvision.features.v1` is the compatibility boundary. Model files must declare it.

## ML, calibration, and OOD

The training path currently implements a transparent logistic baseline. Its model is fitted on train windows, Platt calibration on validation windows, and robust median/MAD OOD bounds on training windows. The registry loader refuses missing or incompatible artifacts. Android uses ONNX Runtime Mobile only when `models/v1/ensemble/model.onnx` exists and a fitted calibration artifact exists.

The architecture accepts additional validated components (random forest, gradient, compact sequence model) behind the same `ModelPrediction` contract. They are not included merely for appearance. A model failure is visible and becomes `NO_TRADE`.

## Signal policy

The final policy requires a supported screenshot, minimum reliable candle count, minimum quality, calibrated probability, model agreement, bounded uncertainty, non-extreme volatility, and an in-distribution feature vector. A missing model or calibration artifact is a normal abstention reason. No technical evidence is converted into fake probability.

## Persistence

Room stores immutable prediction rows and allows one separate outcome label update while refusing to overwrite an existing label. Screenshots are not stored indefinitely. Candidate training data is an external/user-managed dataset; prediction history is never silently fed back into training.

## Performance and diagnostics

The Android ViewModel reuses one `LocalModelRunner` and measures vision, feature, and total stages with `System.nanoTime`. The Python pipeline returns stage timings. The UI receives measured timing values and does not synthesize them.

## CI and reproducibility

Python unit tests run without external libraries. GitHub Actions runs Python tests and attempts Android tests/debug APK build on a standard Android runner. Training records feature schema, seed, chronological split sizes, baseline comparison, and candidate status.
