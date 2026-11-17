#!/bin/bash
# Build mlx.metallib from the vendored mlx v0.32.3 kernels exactly as MLX's own CMake does (MLX_METAL_JIT=ON base set,
# -fno-fast-math). Usage: metallib.sh macosx|iphoneos OUT_DIR
#
# Why: Xcode compiles mlx-swift's bundled .metal files (default.metallib) with the Metal compiler's default fast math, so
# rms_norm / rope / sdpa / ... round differently from the Python wheel's kernels (logits after a 41-token prompt differed by
# up to 0.10; 437 of 9,640 released top-8 sets differed in 200 tokens). MLX loads a colocated mlx.metallib before the
# bundle, so this file next to the binary (or in the .app) makes the native runtime bit-identical to mlx_run.py.
set -euo pipefail
cd "$(dirname "$0")"
if ! xcrun -sdk "${1:-macosx}" metal --version >/dev/null 2>&1; then
  echo "error: the Metal toolchain is not installed (Xcode 26 ships it as a separate download)." >&2
  echo "error: run once:  xcodebuild -downloadComponent MetalToolchain   then build again." >&2
  exit 1
fi
SDK=${1:-macosx}; OUT=${2:-.xcbuild/Build/Products/Release}
SRC=vendor/mlx-swift/Source/Cmlx/mlx
K=$SRC/mlx/backend/metal/kernels
if [ "$SDK" = macosx ]; then MIN="-mmacosx-version-min=26.0"; else MIN="-mios-version-min=26.0"; fi
TMP=.xcbuild/metallib-$SDK; mkdir -p $TMP "$OUT"
FLAGS="-x metal -Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions -Wmetal-addr-spaces $MIN"
AIRS=()
for k in arg_reduce conv dot gather_mm_offsets layer_norm random rms_norm rope scaled_dot_product_attention gated_delta_update fence; do
  xcrun -sdk $SDK metal $FLAGS -c $K/$k.metal -I$SRC -o $TMP/$k.air 2> $TMP/$k.log || { cat $TMP/$k.log; exit 1; }
  AIRS+=($TMP/$k.air)
done
xcrun -sdk $SDK metal $MIN "${AIRS[@]}" -o "$OUT/mlx.metallib"
echo "mlx.metallib ($SDK): $OUT/mlx.metallib ($(stat -f %z "$OUT/mlx.metallib") bytes)"
