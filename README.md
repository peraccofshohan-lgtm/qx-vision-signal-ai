# QX Vision Signal AI

**V1.0.0 · local-first screenshot analysis for Android**

QX Vision Signal AI reconstructs normalized candle geometry from a user-supplied chart screenshot, extracts market-structure features, and applies a validated model artifact only when one is present. `NO TRADE` is a normal, first-class result. The application does not execute trades, click platform controls, collect credentials, upload screenshots, or claim a win rate.

> **Model honesty:** this checkout intentionally contains no trained production weights and no real labelled market dataset. The Android app therefore reports `MODEL_ARTIFACT_UNAVAILABLE` and abstains instead of showing fabricated probabilities. The Python training gate can create a candidate artifact from a user-supplied chronological OHLC dataset, but it promotes nothing unless leakage checks and a holdout baseline gate pass.

## Repository layout

- `app/` — Kotlin Android application, Jetpack Compose UI, Room history, on-device OpenCV-free geometric vision, ONNX Runtime Mobile artifact runner.
- `qxvision/` — dependency-light reference pipeline used for reproducible research, CV tests, feature extraction, calibration, OOD, abstention, and metrics.
- `training/train.py` — chronological OHLC ingestion and candidate logistic model training.
- `tests/` — deterministic vision, temporal framing, feature, model, metrics, and leakage tests.
- `models/v1/` — model registry location. No weights are committed.
- `ARCHITECTURE.md`, `MODEL_CARD.md`, `DATASET_GUIDE.md` — implementation and limitation documents.

## Build Android APK

Requirements: JDK 17, Android SDK 35, and a network-enabled Gradle environment for the first dependency resolution.

```bash
./gradlew clean
./gradlew test
./gradlew assembleDebug
```

Expected artifact when the Android toolchain is installed:

```text
app/build/outputs/apk/debug/app-debug.apk
```

Install/smoke test on a connected device:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n ai.qxvision.signal/.MainActivity
```

This repository snapshot was created in an environment without Java, Android SDK, or Gradle, so APK generation could not be executed here. No APK is claimed until those commands succeed.

## Python checks

The research core has no mandatory third-party runtime dependency:

```bash
python3 -m unittest discover -s tests -v
python3 -m qxvision.cli analyze /path/to/chart.ppm --json
```

Pillow is optional and enables JPEG/WEBP decoding in the Python utility:

```bash
python3 -m pip install Pillow
```

## Training and evaluation

Supply a real OHLC CSV with:

```text
timestamp,open,high,low,close,volume,asset,timeframe
```

Then run:

```bash
python3 -m training.train --ohlc data/your-real-data.csv --output models/v1/ensemble
```

The pipeline sorts chronologically, purges split boundaries, excludes the future target from features, fits OOD statistics on training only, calibrates on validation only, and evaluates on a later holdout. A candidate is not promoted when it fails the previous-candle baseline gate. Synthetic screenshots are for reconstruction testing and are never counted as real predictive validation.

A production artifact intended for Android must additionally be exported as `models/v1/ensemble/model.onnx` with a matching `feature_schema_version`, calibration file, and metadata. Until that happens the Android result remains `NO TRADE`.

## Privacy and safety

All core inference is local. Full screenshots are not persisted by default; only metadata, a hash, features, and immutable prediction records are stored in Room. The application does not include an advertising SDK, tracking SDK, paid API, account login, credentials, martingale logic, or trade execution.

## Known V1 limitations

- A single screenshot cannot prove whether its right-most candle is closed; it is always represented as `PARTIAL_OBSERVATION`.
- Exact prices, asset identity, payout, and higher-timeframe context are not invented when absent.
- The dependency-light Python fallback supports PPM and non-interlaced PNG without Pillow; Android handles platform image decoding for PNG/JPEG/WEBP.
- No trained weights or real dataset are included. Therefore accuracy, confidence, calibration, and coverage are `UNKNOWN` in the shipped app.
- Android build and APK smoke verification require JDK/SDK tooling not available in the authoring environment.
