# Testing and evidence policy

## Local checks

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q qxvision training
python3 -m training.validate_dataset data/market.csv --metadata data/market.metadata.json
```

The Python tests are deterministic and dependency-light. Synthetic chart renders validate geometric reconstruction only; they are not predictive market validation.

## Android checks

With JDK 17, Android SDK 35, and Gradle available:

```bash
./gradlew clean
./gradlew test
./gradlew assembleDebug
```

The Android unit test verifies that missing artifacts produce `NO_TRADE`. CI also verifies the APK package ID and version.

## Model evidence

A real-data run must include:

- dataset metadata and SHA-256 checksum;
- label schema and date range;
- chronological train/validation/final-holdout boundaries;
- purge gaps and fold metadata;
- logistic, gradient-stump, and random-stump-forest candidate reports;
- random, majority, previous-direction, and simple-momentum baselines;
- validation-only calibration and threshold selection;
- risk-coverage curve and reliability curve;
- regime, asset, and time-bucket breakdowns;
- one-time final-holdout report;
- ONNX numerical parity report before promotion.

The final holdout is not used to choose a model, threshold, calibration, or feature set. Numbers without this provenance must not be displayed as model accuracy.

## Safety regression checks

The tests assert that the right-most visible candle is marked `PARTIAL_OBSERVATION`, that its eventual color is not used as a future target, and that a missing or incompatible model artifact cannot yield an invented probability.
