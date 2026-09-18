#!/usr/bin/env sh
# Source checkout launcher. A standard Gradle wrapper distribution is generated
# in CI with `gradle wrapper`; local builds need JDK 17 + Gradle or the wrapper.
set -eu
if command -v gradle >/dev/null 2>&1; then
  exec gradle "$@"
fi
printf '%s\n' 'Gradle is unavailable. Install JDK 17, Android SDK 35, and Gradle 8.10.2, or run in CI.' >&2
exit 1
