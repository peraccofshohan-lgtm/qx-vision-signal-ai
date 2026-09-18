# Dependency policy

The Python core uses only the Python standard library by default. Pillow is optional for additional image formats. The Android build uses Apache-2.0/BSD-style open-source AndroidX, Kotlin, Jetpack Compose, Room, and ONNX Runtime Mobile dependencies resolved from Google Maven/Maven Central. No paid SDK, analytics, advertising, cloud inference API, or secret is required. Exact versions are pinned in `app/build.gradle.kts`; license notices for transitive dependencies are supplied by their upstream distributions and should be included in any release bundle.
