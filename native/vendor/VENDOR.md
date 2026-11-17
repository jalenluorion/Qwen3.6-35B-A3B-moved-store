# Vendored packages

## mlx-swift (vendor/mlx-swift)

- Base: github.com/ml-explore/mlx-swift tag **0.32.3** (commit 19601207e9a0de51e03ee6ec0c3c5f3784275075).
- MLX core submodule moved from v0.32.2 (what mlx-swift 0.32.3 pins, 1f8e74e3) to **v0.32.3** (64ea011cb65f14d9ce2737e60db9a4ae91ed7441),
  the version of the Python runner's `mlx` (0.32.3, with mlx-lm 0.32.0). v0.32.2 -> v0.32.3 changes numerics the runner
  depends on (e.g. "Use precise::exp in Sigmoid so compiled and eager sigmoid agree", gather_qmm kernels).
- mlx-c submodule: ebc88f10caa1b625e6b581437a8dea6df8a70085 (as pinned by mlx-swift 0.32.3).
- Generated sources regenerated for v0.32.3 with mlx-swift's own `tools/update-mlx.sh` (Source/Cmlx/mlx-generated,
  include-framework, include).

Local changes (all small, none changes what a kernel computes):
1. `Package.swift`: `swift-tools-version: 6.2` (was `6.3;(experimentalCGen)`; this Mac has Xcode 26.3 / Swift 6.2.4, and
   the manifest needs nothing from 6.3); test and example targets removed (their sources are not vendored); `Cmlx`
   exported as a product (the runtime calls `_mlx_array_wait` / `mlx_array_data_*` from the name thread); MLX_VERSION
   string 0.32.3; two new CUDA host files of v0.32.3 excluded (`backend/cuda/cusolver_utils.cpp`, `backend/cuda/wddm.cpp`).
2. `Source/Cmlx/mlx-c/mlx/c/ops.cpp`: `mlx_gather_qmm` passes `std::nullopt` for the `global_scale` argument that core
   v0.32.3 added to `gather_qmm` (used only by the fp quantization modes).
3. `Source/Cmlx/mlx/mlx/backend/metal/device.cpp`: instrumentation only. `mlx_phone_gpu_stats()` (GPU busy time as the
   union of every committed command buffer's [GPUStartTime, GPUEndTime], and the buffer count) and
   `mlx_phone_dispatches()` (kernels encoded).
3b. `device.h` / `device.cpp` (2026-11-03, docs/G4-phone-gpu.md): GPU profiler "phone_prof", instrumentation only and
   off unless `mlx_phone_prof_set(1)` (the app sets it only with env `PHONE_GPU_PROF`): per dispatch (kernel, tag), per
   committed buffer its commit / GPU / kernel start-end times; kernel names registered in `get_kernel_`;
   `mlx_phone_prof_{set,tag,mark,clear,dump,now}`. `set(2)` also waits for every committed buffer (serial mode).
4. Removed to save space: mlx's python/docs/benchmarks/tests/examples, mlx-swift's Tests, Examples, xcode project, skills.

## Kernels: mlx.metallib, not the bundle's default.metallib

Xcode compiles the bundled `.metal` files into `mlx-swift_Cmlx.bundle/default.metallib` with the Metal compiler's
default fast math. MLX's own CMake compiles them with `-fno-fast-math`, as the Python wheel does. With Xcode's metallib
the native runtime's logits after a 41-token prompt differed from mlx_run.py by up to 0.10 and 437 of 9,640 released
top-8 sets differed over 200 tokens (the generated tokens happened to stay equal). `native/metallib.sh` builds
`mlx.metallib` from `vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels` with MLX's flags (the MLX_METAL_JIT=ON
base set: arg_reduce conv dot gather_mm_offsets layer_norm random rms_norm rope scaled_dot_product_attention
gated_delta_update fence); MLX loads a colocated `mlx.metallib` before the bundle. With it the runtime is bit-identical to
mlx_run.py (docs/G4-runtime.md). `build.sh` runs it; the CLI names the metallib it found in the startup line and JSON.
Every other kernel is JIT-compiled at run time by MLX (MTLCompileOptions safe math), as in mlx-swift.

## Other packages (SwiftPM, pinned in Package.resolved)

- swift-transformers **1.4.0** (exact): Tokenizers (tokenizer.json, chat template via swift-jinja).
- Transitive: swift-numerics (mlx-swift), swift-jinja, swift-huggingface, swift-collections, swift-crypto, yyjson: see
  `native/Package.resolved` for the resolved versions.
