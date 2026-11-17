#!/bin/bash
# Build phone-store-run (macOS, Release) with xcodebuild: SwiftPM alone cannot compile MLX's Metal shaders into
# default.metallib. Output: native/.xcbuild/Build/Products/Release/phone-store-run (+ mlx-swift_Cmlx.bundle beside it).
# Needs the Metal toolchain component (xcodebuild -downloadComponent MetalToolchain, once).
set -uo pipefail
cd "$(dirname "$0")"
if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
  echo "error: the Metal toolchain is not installed. Run once: xcodebuild -downloadComponent MetalToolchain" >&2; exit 1
fi
CONF=${CONF:-Release}
LOG=.xcbuild/last-build.log
mkdir -p .xcbuild
xcodebuild -scheme phone-store-run -configuration "$CONF" -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .xcbuild -skipPackagePluginValidation -skipMacroValidation COMPILER_INDEX_STORE_ENABLE=NO build > $LOG 2>&1
if grep -q "BUILD SUCCEEDED" $LOG; then
  ./metallib.sh macosx .xcbuild/Build/Products/$CONF || exit 1
  echo "BUILD SUCCEEDED: native/.xcbuild/Build/Products/$CONF/phone-store-run (+ mlx.metallib, -fno-fast-math)"
else
  grep -E "error:" $LOG | sort -u | head -40
  echo "BUILD FAILED (full log: native/$LOG)"; exit 1
fi
