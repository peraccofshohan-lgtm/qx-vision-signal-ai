# Build and release

## Android

Required: JDK 17, Android SDK platform 35, build tools 35.0.0, and Gradle 8.10.2.

```bash
./gradlew clean
./gradlew test
./gradlew assembleDebug
```

The debug APK is created at `app/build/outputs/apk/debug/app-debug.apk` only after a successful build. Package ID is `ai.qxvision.signal`; version is `1.0.0`.

The current Agent sandbox has no Java, Android SDK, or Gradle. Therefore no APK path is asserted as existing here. CI provisions the toolchain and uploads the debug APK.

## Validated artifact deployment

Training writes a challenger under `models/v1/ensemble/candidate/`, not directly to the Android production path. Export and parity-check it:

```bash
python3 -m training.export_onnx \
  --registry models/v1/ensemble/candidate \
  --output models/v1/ensemble/candidate/model.onnx
```

Promotion requires a passing `onnx_parity.json`, metadata, schema, calibration, and validation gates:

```bash
python3 -m training.promote \
  --candidate models/v1/ensemble/candidate \
  --champion models/v1/ensemble
```

Only a promoted artifact should be copied into `app/src/main/assets/models/v1/ensemble/`. The Android loader validates presence, metadata, checksum/shape contract, schema, output bounds, and calibration before inference. Missing or invalid assets produce `NO_TRADE`.
