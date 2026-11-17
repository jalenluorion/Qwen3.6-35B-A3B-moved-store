// Fused kernels for the decode step (2026-11-03, docs/G4-phone-gpu.md "Kernel fusion").
//
// Every kernel here replaces a run of MLX kernels with one launch and computes the same arithmetic in the same order:
// - the quantized matvecs replicate MLX's affine qmv_fast_impl / qmv_impl row by row (same lane layout per simdgroup,
//   same per-block load_vector / qdot accumulation, same simd_sum), so several matrices that read the same input run in
//   one launch and their epilogues (bias add, gelu, swiglu, sigmoid gate, negation) follow MLX's eager / compiled
//   elementwise ops with the same intermediate dtypes;
// - reductions (rms_norm, softmax, the row / column sums) replicate MLX's kernels' thread layout and reduction order;
// - argpartition is MLX's stable block merge sort: the permutation is the stable ascending order, reproduced by ranks.
// Each fusion is a switch (env PHONE_FUSE_<NAME>=0|1, PHONE_FUSE=0 turns all off); `fuseSelfTest` checks every fused
// kernel bit for bit against the MLX ops it replaces on random inputs of the model's shapes.
import Foundation
import MLX
import MLXFast
import MLXNN

/// Fusion switches. Read once from the environment (PHONE_FUSE=0: all off; PHONE_FUSE_<NAME>=0|1 per fusion).
public enum Fuse {
    static let env = ProcessInfo.processInfo.environment
    static func flag(_ name: String, _ def: Bool) -> Bool {
        if let v = env["PHONE_FUSE_" + name] { return v != "0" }
        if let v = env["PHONE_FUSE"] { return v != "0" }
        return def
    }
    /// lag-4 router: [lin | mlp0] in one int8 matvec over the concatenated input (no concat copies) with bias and gelu
    /// epilogues, then mlp2 with the sum, bias and negation epilogue (15 kernels -> 3 with the sort).
    public static var router = flag("ROUTER", true)
    /// GatedDeltaNet: in_proj qkv/z/b/a in one launch; conv1d + silu + q/k rms norms + beta + g + the gated-delta step +
    /// the gated rms norm + precise swiglu in one kernel; out_proj (18 kernels -> 3).
    public static var gdn = flag("GDN", true)
    /// Shared expert: gate/up/shared_expert_gate in one launch with the swiglu epilogue; down_proj with the sigmoid-gate
    /// epilogue (7 kernels -> 2).
    public static var shared = flag("SHARED", true)
    /// Attention: q/k/v projections in one launch (3 -> 1).
    public static var attnQKV = flag("ATTN", true)
    /// Attention: q/k rms norms + partial RoPE (and the copies around them) in one launch (~7 -> 1).
    public static var attnRope = flag("ROPE", true)
    /// Engine hooks (Fused.* called from Engine.step): masked top-k + released gate + scores + hit counter in one kernel.
    public static var select = flag("SELECT", true)
    /// Engine hook: the routed expert step (gate/up gather + swiglu; down gather + scores + expert sum + residual adds).
    public static var routed = flag("ROUTED", true)
    /// Engine hook: residual add + post-attention rms norm in one kernel.
    public static var addNorm = flag("ADDNORM", true)
    /// Engine hook (prefetch, not a fusion of arithmetic): the lag router's prediction rides with the layer's next submit
    /// instead of its own command buffer (one buffer per layer instead of two; the names reach the store later).
    public static var mergeSubmit = flag("MERGESUBMIT", false)

    public static var summary: String {
        "router=\(router ? 1 : 0) gdn=\(gdn ? 1 : 0) shared=\(shared ? 1 : 0) attn=\(attnQKV ? 1 : 0) rope=\(attnRope ? 1 : 0) select=\(select ? 1 : 0) routed=\(routed ? 1 : 0) addnorm=\(addNorm ? 1 : 0) mergesubmit=\(mergeSubmit ? 1 : 0)"
    }
}

// MARK: - Metal sources

/// Shared by every fused kernel: MLX's elementwise ops (unary_ops.h / binary_ops.h, erf.h, expm1f.h) and the quantized
/// matvec row routine of quantized.h (qmv_fast_impl / qmv_impl with group 64), verbatim where it computes.
let fuseHeader = #"""
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// ---- mlx/backend/metal/kernels/expm1f.h, erf.h (renamed) ----
inline float fz_expm1f_scaled_unchecked(float a, float b) {
  float f, j, r, s, t, u, v, x, y;
  int i;
  j = fma(1.442695f, a, 12582912.f);
  j = j - 12582912.0f;
  i = (int)j;
  f = fma(j, -6.93145752e-1f, a);
  s = f * f;
  if (a == 0.0f)
    s = a;
  r = 1.97350979e-4f;
  r = fma(r, f, 1.39309070e-3f);
  r = fma(r, f, 8.33343994e-3f);
  r = fma(r, f, 4.16668020e-2f);
  r = fma(r, f, 1.66666716e-1f);
  r = fma(r, f, 4.99999970e-1f);
  u = (j == 1) ? (f + 0.5f) : f;
  v = fma(r, s, u);
  s = 0.5f * b;
  t = ldexp(s, i);
  y = t - s;
  x = (t - y) - s;
  r = fma(v, t, x) + y;
  r = r + r;
  if (j == 0)
    r = v;
  if (j == 1)
    r = v + v;
  return r;
}
inline float fz_expm1f(float a) {
  float r;
  r = fz_expm1f_scaled_unchecked(a, 1.0f);
  if (abs(a - 1.0f) > 88.0f) {
    r = pow(2, a);
    r = fma(r, r, -1.0f);
  }
  return r;
}
inline float fz_erf(float a) {
  float r, s, t, u;
  t = metal::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
    u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f);
    r = metal::fma(r, t, -6.34846687e-1f);
    r = metal::fma(r, t, -1.28717512e-1f);
    r = metal::fma(r, t, -t);
    r = -fz_expm1f(r);
    r = metal::copysign(r, a);
  } else {
    r = -5.96761703e-4f;
    r = metal::fma(r, s, 4.99119423e-3f);
    r = metal::fma(r, s, -2.67681349e-2f);
    r = metal::fma(r, s, 1.12819925e-1f);
    r = metal::fma(r, s, -3.76125336e-1f);
    r = metal::fma(r, s, 1.28379166e-1f);
    r = metal::fma(r, a, a);
  }
  return r;
}

// ---- unary_ops.h / binary_ops.h ----
struct FzSigmoid {
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct FzLogAddExp {
  template <typename T>
  T operator()(T x, T y) thread {
    if (metal::isnan(x) || metal::isnan(y)) {
      return metal::numeric_limits<T>::quiet_NaN();
    }
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : (maxval + log1p(metal::exp(minval - maxval)));
  };
};
// MLXNN silu (compiled x * sigmoid(x)) on half
inline half fz_silu(half x) { half s = FzSigmoid()(x); return x * s; }
// MLXNN gelu (compiled x * (1 + erf(x / sqrt(2))) / 2) on half
inline half fz_gelu(half x) {
  half a = x / half(1.41421356237309504880f);
  half e = static_cast<half>(fz_erf(static_cast<float>(a)));
  half b = half(1) + e;
  half m = x * b;
  return m / half(2);
}

// ---- quantized.h (group 64, bits 4 and 8) ----
template <int bits, int values_per_thread, typename XP>
inline float fz_load_vector(XP x, thread float* x_thread) {
  float sum = 0;
  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}
template <int bits, int values_per_thread, typename XP>
inline float fz_load_vector_safe(XP x, thread float* x_thread, int N) {
  float sum = 0;
  if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }
  return sum;
}
template <int bits, int values_per_thread>
inline float fz_qdot(const device uint8_t* w, const thread float* x_thread, float scale, float bias, float sum) {
  float accum = 0;
  if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}
template <int bits, int values_per_thread>
inline float fz_qdot_safe(const device uint8_t* w, const thread float* x_thread, float scale, float bias, float sum, int N) {
  float accum = 0;
  if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}

/// qmv_fast_impl for R consecutive rows starting at out_row, one simdgroup; result[r] = the row's simd_sum (float).
/// `x` points at the input vector (half, any address space); `K` = in_vec_size (a multiple of the block size).
template <int bits, int R, typename XP>
inline void fz_qmv_fast_rows(
    const device uint32_t* w, const device half* scales, const device half* biases, XP x, int K, int out_row,
    uint simd_lid, thread float* result) {
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = 32 / bits;
  constexpr int bytes_per_pack = 4;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int group_size = 64;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x_thread[values_per_thread];
  for (int r = 0; r < R; r++) result[r] = 0;
  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;
  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  for (int k = 0; k < K; k += block_size) {
    float sum = fz_load_vector<bits, values_per_thread>(x, x_thread);
    for (int row = 0; row < R; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device half* sl = scales + row * in_vec_size_g;
      const device half* bl = biases + row * in_vec_size_g;
      float s = sl[0];
      float b = bl[0];
      result[row] += fz_qdot<bits, values_per_thread>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }
  for (int row = 0; row < R; row++) {
    result[row] = simd_sum(result[row]);
  }
}

/// qmv_impl's small-N branch (out_vec_size < 8) for one row, packs_per_thread 1: MLX's kernel for shared_expert_gate.
template <int bits, typename XP>
inline float fz_qmv_small_row(
    const device uint32_t* w, const device half* scales, const device half* biases, XP x, int K, int out_row,
    uint simd_lid) {
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = 32 / bits;
  constexpr int bytes_per_pack = 4;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int group_size = 64;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x_thread[values_per_thread];
  float result = 0;
  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;
  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  int k = 0;
  for (; k < K - block_size; k += block_size) {
    float sum = fz_load_vector<bits, values_per_thread>(x, x_thread);
    float s = scales[0];
    float b = biases[0];
    result += fz_qdot<bits, values_per_thread>(ws, x_thread, s, b, sum);
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }
  const int remaining = clamp(static_cast<int>(K - k - simd_lid * values_per_thread), 0, values_per_thread);
  if (remaining > 0) {
    float sum = fz_load_vector_safe<bits, values_per_thread>(x, x_thread, remaining);
    float s = scales[0];
    float b = biases[0];
    result += fz_qdot_safe<bits, values_per_thread>(ws, x_thread, s, b, sum, remaining);
  }
  return simd_sum(result);
}

// stable ascending order (MLX sort.h LessThan + stable merge sort): a before b
inline bool fz_less(half a, half b) {
  bool an = metal::isnan(a);
  bool bn = metal::isnan(b);
  if (an | bn) {
    return (!an) & bn;
  }
  return a < b;
}
"""#

// MARK: - kernel cache

final class KernelCache: @unchecked Sendable {
    static let shared = KernelCache()
    private var k: [String: MLXFast.MLXFastKernel] = [:]
    private let lock = NSLock()
    func get(_ key: String, _ make: () -> MLXFast.MLXFastKernel) -> MLXFast.MLXFastKernel {
        lock.lock(); defer { lock.unlock() }
        if let v = k[key] { return v }
        let v = make()
        k[key] = v
        return v
    }
}

/// A quantized Lin's arrays for a kernel input list (weight, scales, biases).
@inline(__always) func qa(_ l: Lin) -> [MLXArray] { [l.w, l.s!, l.b!] }

public enum Fused {
    // MARK: plain multi-matrix matvec (same x, same bits, same K): one launch, one output per matrix

    /// y_i = lins[i](x) for every i, exactly as `quantizedMatmul` computes each (qmv_fast rows). Every row count must be
    /// a multiple of 8, K a multiple of 512 (4 bit) / 256 (8 bit).
    public static func qmvMulti(_ x: MLXArray, _ lins: [Lin]) -> [MLXArray] {
        let n = lins.count
        let bits = lins[0].bits
        let K = x.dim(-1)
        let rows = lins.map { $0.w.dim(0) }
        let name = "fz_qmv_multi_\(n)_b\(bits)"
        let kern = KernelCache.shared.get(name) {
            var names = ["x"]
            for i in 0..<n { names += ["w\(i)", "s\(i)", "b\(i)"] }
            var src = """
                uint simd_lid = thread_index_in_simdgroup;
                uint sg = threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup;
                int row = int(sg) * 4;
                float res[4];

                """
            var off = 0
            _ = off
            for i in 0..<n {
                src += """
                    if (row < R\(i)) {
                      fz_qmv_fast_rows<\(bits), 4>(w\(i), s\(i), b\(i), x, K, row, simd_lid, res);
                      if (simd_lid == 0) { for (int r = 0; r < 4; r++) y\(i)[row + r] = static_cast<half>(res[r]); }
                      return;
                    }
                    row -= R\(i);

                    """
                off += 1
            }
            return MLXFast.metalKernel(name: name, inputNames: names, outputNames: (0..<n).map { "y\($0)" }, source: src,
                                       header: fuseHeader)
        }
        var inputs = [x]
        for l in lins { inputs += qa(l) }
        var tmpl: [(String, any KernelTemplateArg)] = [("K", K)]
        for i in 0..<n { tmpl.append(("R\(i)", rows[i])) }
        let total = rows.reduce(0, +)
        let outs = kern(inputs, template: tmpl, grid: (32, total / 4, 1), threadGroup: (32, 2, 1),
                        outputShapes: rows.map { [$0] }, outputDTypes: Array(repeating: .float16, count: n))
        return outs
    }

    // MARK: shared expert

    /// sigmoid(seGate(x)) * down(swiglu(gate(x), up(x))) as Forward.shared computes it, in two launches.
    public static func shared(_ w: LayerW, _ x: MLXArray) -> MLXArray {
        let K = x.dim(-1)
        let I = w.seGateProj.w.dim(0)
        let up = KernelCache.shared.get("fz_shared_up") {
            MLXFast.metalKernel(
                name: "fz_shared_up", inputNames: ["x", "gw", "gs", "gb", "uw", "us", "ub", "ew", "es", "eb"],
                outputNames: ["act", "seg"],
                source: """
                    uint simd_lid = thread_index_in_simdgroup;
                    uint sg = threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup;
                    int row = int(sg) * 4;
                    if (row < I) {
                      float g[4], u[4];
                      fz_qmv_fast_rows<4, 4>(gw, gs, gb, x, K, row, simd_lid, g);
                      fz_qmv_fast_rows<4, 4>(uw, us, ub, x, K, row, simd_lid, u);
                      if (simd_lid == 0) {
                        for (int r = 0; r < 4; r++) {
                          half gh = static_cast<half>(g[r]);
                          half uh = static_cast<half>(u[r]);
                          half sg_ = FzSigmoid()(gh);
                          half sl = gh * sg_;
                          act[row + r] = sl * uh;
                        }
                      }
                    } else if (row == I) {
                      float e = fz_qmv_small_row<8>(ew, es, eb, x, K, 0, simd_lid);
                      if (simd_lid == 0) seg[0] = static_cast<half>(e);
                    }
                    """,
                header: fuseHeader)
        }
        let down = KernelCache.shared.get("fz_shared_down") {
            MLXFast.metalKernel(
                name: "fz_shared_down", inputNames: ["act", "seg", "dw", "ds", "db"], outputNames: ["y"],
                source: """
                    uint simd_lid = thread_index_in_simdgroup;
                    uint sg = threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup;
                    int row = int(sg) * 4;
                    float d[4];
                    fz_qmv_fast_rows<4, 4>(dw, ds, db, act, I, row, simd_lid, d);
                    if (simd_lid == 0) {
                      half sig = FzSigmoid()(seg[0]);
                      for (int r = 0; r < 4; r++) y[row + r] = sig * static_cast<half>(d[r]);
                    }
                    """,
                header: fuseHeader)
        }
        let xf = x.reshaped([K])
        // gate/up rows: I / 4 simdgroups; one more simdgroup for shared_expert_gate -> (I/4 + 1) simdgroups, padded to 2
        let sgs = I / 4 + 2
        let a = up([xf] + qa(w.seGateProj) + qa(w.seUpProj) + qa(w.seGate), template: [("K", K), ("I", I)],
                   grid: (32, sgs, 1), threadGroup: (32, 2, 1), outputShapes: [[I], [1]], outputDTypes: [.float16, .float16])
        let D = w.seDownProj.w.dim(0)
        let y = down([a[0], a[1]] + qa(w.seDownProj), template: [("I", I)], grid: (32, D / 4, 1), threadGroup: (32, 2, 1),
                     outputShapes: [[D]], outputDTypes: [.float16])[0]
        return y.reshaped(x.shape)
    }

    // MARK: lag-4 router

    /// Forward.routeLogits(r, embN, hPrev, hLag) negated (the argpartition key), in two launches.
    public static func routerNegLogits(_ r: Router, embN: MLXArray, hPrev: MLXArray, hLag: MLXArray?) -> MLXArray {
        let H = embN.dim(-1)
        let lagged = r.lagged
        let K = (lagged ? 3 : 2) * H
        let NO = r.lin.w.dim(0)  // 256
        let HD = r.mlp0.w.dim(0)  // 1024
        let name1 = "fz_router1_\(lagged ? 3 : 2)"
        let k1 = KernelCache.shared.get(name1) {
            // x = [embN * emb_norm, hPrev, hLag]: built in threadgroup memory (the concatenation's values), then the
            // [lin | mlp0] rows as qmv_fast (8 bit) over it.
            MLXFast.metalKernel(
                name: name1,
                inputNames: ["en", "enw", "hp"] + (lagged ? ["hl"] : []) + ["lw", "ls", "lb", "lbias", "mw", "ms", "mb", "mbias"],
                outputNames: ["z1", "hh"],
                source: """
                    threadgroup half xs[KX];
                    uint tid = thread_index_in_threadgroup;
                    uint nt = threads_per_threadgroup.x * threads_per_threadgroup.y;
                    for (uint i = tid; i < uint(H); i += nt) {
                      xs[i] = en[i] * enw[i];
                      xs[H + i] = hp[i];
                    \(lagged ? "  xs[2 * H + i] = hl[i];" : "")
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint simd_lid = thread_index_in_simdgroup;
                    uint sg = threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup;
                    int row = int(sg) * 4;
                    float res[4];
                    if (row < NO) {
                      fz_qmv_fast_rows<8, 4>(lw, ls, lb, (const threadgroup half*)xs, KX, row, simd_lid, res);
                      if (simd_lid == 0) for (int r = 0; r < 4; r++) z1[row + r] = static_cast<half>(res[r]) + lbias[row + r];
                    } else {
                      row -= NO;
                      fz_qmv_fast_rows<8, 4>(mw, ms, mb, (const threadgroup half*)xs, KX, row, simd_lid, res);
                      if (simd_lid == 0) for (int r = 0; r < 4; r++) {
                        half v = static_cast<half>(res[r]) + mbias[row + r];
                        hh[row + r] = fz_gelu(v);
                      }
                    }
                    """,
                header: fuseHeader)
        }
        let k2 = KernelCache.shared.get("fz_router2") {
            MLXFast.metalKernel(
                name: "fz_router2", inputNames: ["hh", "z1", "w", "s", "b", "bias"], outputNames: ["neg"],
                source: """
                    uint simd_lid = thread_index_in_simdgroup;
                    uint sg = threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup;
                    int row = int(sg) * 4;
                    float res[4];
                    fz_qmv_fast_rows<8, 4>(w, s, b, hh, HD, row, simd_lid, res);
                    if (simd_lid == 0) for (int r = 0; r < 4; r++) {
                      half z = z1[row + r] + static_cast<half>(res[r]);
                      z = z + bias[row + r];
                      neg[row + r] = -z;
                    }
                    """,
                header: fuseHeader)
        }
        var in1 = [embN.reshaped([H]), r.embNorm, hPrev.reshaped([H])]
        if lagged { in1.append(hLag!.reshaped([H])) }
        in1 += qa(r.lin) + [r.linBias] + qa(r.mlp0) + [r.mlp0Bias]
        let o1 = k1(in1, template: [("H", H), ("KX", K), ("NO", NO)], grid: (32, (NO + HD) / 4, 1), threadGroup: (32, 2, 1),
                    outputShapes: [[NO], [HD]], outputDTypes: [.float16, .float16])
        let neg = k2([o1[1], o1[0]] + qa(r.mlp2) + [r.mlp2Bias], template: [("HD", HD)], grid: (32, NO / 4, 1),
                     threadGroup: (32, 2, 1), outputShapes: [[NO]], outputDTypes: [.float16])[0]
        return neg
    }
}

// MARK: - GatedDeltaNet mega kernel

/// One threadgroup per value head (512 threads = 16 simdgroups x 8 rows of 4 lanes, the packed kernel's row layout):
/// A. the head's q/k/v channels: depthwise conv1d over [conv state; qkv] (MLX depthwise_conv_1d), silu (compiled);
///    q, k: rms_norm without weight (MLX rms_single_row, 32 lanes x 4) times the scale (half), beta = sigmoid(b),
///    g = compute_g (compiled); the new conv state (last 3 rows) for the head's channels;
/// B. gated_delta_step_packed_btree, the same per-row code (state in/out float);
/// C. the head's y: rms_norm with weight (MLX rms_single_row), precise_swiglu(z) (compiled), written for out_proj.
let gdnMegaSource = """
  constexpr int lanes_per_row = 4;
  constexpr int rows_per_simdgroup = 32 / lanes_per_row;
  constexpr int values_per_lane = Dk / lanes_per_row;
  constexpr int partials_per_lane = values_per_lane / 4;

  threadgroup half sq[Dk];
  threadgroup half sk[Dk];
  threadgroup half sv[Dv];
  threadgroup half sqn[Dk];
  threadgroup half skn[Dk];
  threadgroup half sy[Dv];
  threadgroup float sg[1];
  threadgroup half sb[1];

  const uint hv_idx = threadgroup_position_in_grid.z;
  const uint hk_idx = hv_idx / (Hv / Hk);
  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint simd = simdgroup_index_in_threadgroup;

  // ---- A. conv1d + silu for the head's 3 x 128 channels; conv state shift ----
  if (tid < uint(2 * Dk + Dv)) {
    int part = tid / Dk;           // 0 q, 1 k, 2 v
    int j = tid % Dk;
    int c = part == 0 ? int(hk_idx) * Dk + j : (part == 1 ? KD + int(hk_idx) * Dk + j : 2 * KD + int(hv_idx) * Dv + j);
    float acc = 0.0;
    for (int i = 0; i < CK; ++i) {
      half inv_ = (i < CK - 1) ? cs[i * C + c] : qkv[c];
      acc += static_cast<float>(inv_) * cw[c * CK + i];
    }
    half co = static_cast<half>(acc);
    half o = fz_silu(co);
    if (part == 0) sq[j] = o; else if (part == 1) sk[j] = o; else sv[j] = o;
    bool writer = part == 2 || (hv_idx % (Hv / Hk)) == 0;
    if (writer) {
      for (int r = 0; r < CK - 1; ++r) {
        cs_out[r * C + c] = (r + 1 < CK - 1) ? cs[(r + 1) * C + c] : qkv[c];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // q (simdgroup 0) and k (simdgroup 1): rms_single_row, 32 lanes x 4, no weight; times the scale
  if (simd < 2) {
    threadgroup half* src = simd == 0 ? sq : sk;
    threadgroup half* dst = simd == 0 ? sqn : skn;
    float acc = 0;
    float thread_x[4];
    for (int i = 0; i < 4; i++) {
      thread_x[i] = src[lane * 4 + i];
      acc += thread_x[i] * thread_x[i];
    }
    acc = simd_sum(acc);
    float tot = simd_sum(lane == 0 ? acc : 0.0f);
    float inv_mean = metal::precise::rsqrt(tot / uint(Dk) + eps_qk);
    half scale = simd == 0 ? scale_q : scale_k;
    for (int i = 0; i < 4; i++) {
      half nrm = half(1) * static_cast<half>(thread_x[i] * inv_mean);
      dst[lane * 4 + i] = scale * nrm;
    }
  } else if (simd == 2 && lane == 0) {
    sb[0] = FzSigmoid()(bb[hv_idx]);
    // compute_g: exp(-exp(A_log) * softplus(a + dt_bias)), softplus = logaddexp(x, 0) in half
    float ea = metal::precise::exp(alog[hv_idx]);
    float na = -ea;
    half s = aa[hv_idx] + dtb[hv_idx];
    half sp = FzLogAddExp()(s, half(0));
    float spf = static_cast<float>(sp);
    float m = na * spf;
    sg[0] = metal::precise::exp(m);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- B. gated delta step (packed btree), T = 1 ----
  {
    auto row_in_simdgroup = lane / lanes_per_row;
    auto lane_in_row = lane & (lanes_per_row - 1);
    auto dv_idx = simd * rows_per_simdgroup + row_in_simdgroup;
    const threadgroup half* q_ = sqn + lane_in_row * values_per_lane;
    const threadgroup half* k_ = skn + lane_in_row * values_per_lane;
    auto i_state = state_in + (hv_idx * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
    auto o_state = state_out + (hv_idx * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
    float state[values_per_lane];
    for (int i = 0; i < values_per_lane; ++i) {
      state[i] = static_cast<float>(i_state[i]);
    }
    float gt = static_cast<float>(sg[0]);
    float part[partials_per_lane];
    for (int pb = 0; pb < partials_per_lane; ++pb) {
      float acc = 0.0f;
      for (int i = 0; i < 4; ++i) {
        int e = pb * 4 + i;
        state[e] = state[e] * gt;
        acc += state[e] * static_cast<float>(k_[e]);
      }
      part[pb] = acc;
    }
    float kv_mem =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    kv_mem += simd_shuffle_xor(kv_mem, 1);
    kv_mem += simd_shuffle_xor(kv_mem, 2);
    auto delta =
        (static_cast<float>(sv[dv_idx]) - kv_mem) *
        static_cast<float>(sb[0]);
    for (int pb = 0; pb < partials_per_lane; ++pb) {
      float acc = 0.0f;
      for (int i = 0; i < 4; ++i) {
        int e = pb * 4 + i;
        state[e] = state[e] + static_cast<float>(k_[e]) * delta;
        acc += state[e] * static_cast<float>(q_[e]);
      }
      part[pb] = acc;
    }
    float out =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    out += simd_shuffle_xor(out, 1);
    out += simd_shuffle_xor(out, 2);
    if (lane_in_row == 0) {
      sy[dv_idx] = static_cast<half>(out);
    }
    for (int i = 0; i < values_per_lane; ++i) {
      o_state[i] = static_cast<float>(state[i]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- C. rms_norm(y, norm weight) and precise_swiglu(z) ----
  if (simd == 0) {
    float acc = 0;
    float thread_x[4];
    for (int i = 0; i < 4; i++) {
      thread_x[i] = sy[lane * 4 + i];
      acc += thread_x[i] * thread_x[i];
    }
    acc = simd_sum(acc);
    float tot = simd_sum(lane == 0 ? acc : 0.0f);
    float inv_mean = metal::precise::rsqrt(tot / uint(Dv) + eps_n);
    for (int i = 0; i < 4; i++) {
      int d = lane * 4 + i;
      half nrm = nw[d] * static_cast<half>(thread_x[i] * inv_mean);
      float gf = static_cast<float>(zz[hv_idx * Dv + d]);
      float sgm = FzSigmoid()(gf);
      float sl = gf * sgm;
      float xf = static_cast<float>(nrm);
      o[hv_idx * Dv + d] = static_cast<half>(sl * xf);
    }
  }
"""

extension Fused {
    /// Scalars the Python/Swift code multiplies with (a Float times a float16 array): the same float16 rounding.
    static func halfScalar(_ v: Float) -> MLXArray { (v * MLXArray.ones([1], dtype: .float16)).reshaped([]) }

    /// Forward.gdn for B = S = 1 in three launches: in_proj (qkv, z, b, a), the mega kernel, out_proj.
    public static func gdn(_ fwd: Forward, _ w: GDNW, _ x: MLXArray, _ c: GDNCache) -> MLXArray {
        let cfg = fwd.cfg
        let H = cfg.hidden
        let nv = cfg.linVHeads, dv = cfg.linVDim, nk = cfg.linKHeads, dk = cfg.linKDim
        let C = cfg.convDim, CK = cfg.convK
        let p = qmvMulti(x.reshaped([H]), [w.inQKV, w.inZ, w.inB, w.inA])
        let convState = c.conv ?? MLXArray.zeros([1, CK - 1, C], dtype: .float16)
        let state = c.state ?? MLXArray.zeros([1, nv, dv, dk], dtype: .float32)
        let kern = KernelCache.shared.get("fz_gdn_mega") {
            MLXFast.metalKernel(
                name: "fz_gdn_mega",
                inputNames: ["qkv", "cs", "cw", "bb", "aa", "alog", "dtb", "state_in", "zz", "nw", "scale_q", "scale_k", "eps_qk", "eps_n"],
                outputNames: ["o", "cs_out", "state_out"], source: gdnMegaSource, header: fuseHeader)
        }
        let consts = gdnConsts(fwd)
        let outs = kern(
            [p[0], convState, w.conv, p[2], p[3], w.aLog, w.dtBias, state, p[1], w.norm, consts.0, consts.1, consts.2, consts.3],
            template: [("Dk", dk), ("Dv", dv), ("Hk", nk), ("Hv", nv), ("C", C), ("CK", CK), ("KD", cfg.keyDim)],
            grid: (32, 16, nv), threadGroup: (32, 16, 1),
            outputShapes: [[nv * dv], [1, CK - 1, C], [1, nv, dv, dk]], outputDTypes: [.float16, .float16, .float32])
        c.conv = outs[1]
        c.state = outs[2]
        return w.outProj(outs[0].reshaped([1, 1, nv * dv]))
    }

    nonisolated(unsafe) static var gdnConstCache: (MLXArray, MLXArray, MLXArray, MLXArray)? = nil
    static func gdnConsts(_ fwd: Forward) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        if let c = gdnConstCache { return c }
        let c = (halfScalar(Float(fwd.invScale * fwd.invScale)), halfScalar(Float(fwd.invScale)),
                 MLXArray(fwd.rmsEps), MLXArray(fwd.cfg.eps))
        eval(c.0, c.1, c.2, c.3)
        gdnConstCache = c
        return c
    }
}

// MARK: - Engine hooks: the released gate's selection, the routed expert step, residual + rms norm

/// softmax(precise) of the gate logits, the masked (prefetch) or plain (exact) top-K in MLX's argpartition order, the
/// slots, the normalized scores, and (masked) the hit counter of the released top-K. One threadgroup of NE threads.
let selectSource = """
  threadgroup half g[NE];
  threadgroup half m[NE];
  threadgroup int sel[KK];
  threadgroup float local_max[32];
  threadgroup float local_normalizer[32];
  threadgroup int hit_part[NE / 32];
  const uint t = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint simd = simdgroup_index_in_threadgroup;
  // ---- softmax_single_row<half, float, 4>: threads 0..NE/4-1 (MLX's threadgroup), same lanes ----
  constexpr int SM = NE / 4;  // MLX threadgroup size for this axis (a multiple of 32)
  float ld[4];
  if (simd == 0) {
    local_max[lane] = Limits<float>::min;
    local_normalizer[lane] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxval = Limits<float>::finite_min;
  if (t < uint(SM)) {
    for (int i = 0; i < 4; i++) ld[i] = float(logits[t * 4 + i]);
    for (int i = 0; i < 4; i++) maxval = (maxval < ld[i]) ? ld[i] : maxval;
    maxval = simd_max(maxval);
    if (lane == 0) local_max[simd] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    maxval = simd_max(local_max[lane]);
    if (lane == 0) local_max[0] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];
  float normalizer = 0;
  if (t < uint(SM)) {
    for (int i = 0; i < 4; i++) {
      float exp_x = fast::exp(ld[i] - maxval);
      ld[i] = exp_x;
      normalizer += exp_x;
    }
    normalizer = simd_sum(normalizer);
    if (lane == 0) local_normalizer[simd] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    normalizer = simd_sum(local_normalizer[lane]);
    if (lane == 0) local_normalizer[0] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / local_normalizer[0];
  if (t < uint(SM)) {
    for (int i = 0; i < 4; i++) g[t * 4 + i] = half(ld[i] * normalizer);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // ---- ranks in the stable ascending order (MLX block merge sort) ----
  half gt = g[t];
  int hit = 0;
#if MASKED
  {
    int tb = tab[t];
    // released top-K of the gates: argpartition(gates, NE-K)[NE-K:], i.e. ascending ranks >= NE-K
    int r = 0;
    for (int j = 0; j < NE; j++) {
      half gj = g[j];
      r += (fz_less(gj, gt) || (j < int(t) && !fz_less(gt, gj))) ? 1 : 0;
    }
    hit = (r >= NE - KK && tb >= 0) ? 1 : 0;
    m[t] = tb >= 0 ? gt : half(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // masked top-K: argpartition(-masked, K-1)[:K], ascending ranks < K of -masked
    half nt = -m[t];
    int r2 = 0;
    for (int j = 0; j < NE; j++) {
      half nj = -m[j];
      r2 += (fz_less(nj, nt) || (j < int(t) && !fz_less(nt, nj))) ? 1 : 0;
    }
    if (r2 < KK) sel[r2] = int(t);
    int hs = simd_sum(hit);
    if (lane == 0) hit_part[simd] = hs;
  }
#else
  {
    m[t] = gt;
    int r = 0;
    for (int j = 0; j < NE; j++) {
      half gj = g[j];
      r += (fz_less(gj, gt) || (j < int(t) && !fz_less(gt, gj))) ? 1 : 0;
    }
    if (r >= NE - KK) sel[r - (NE - KK)] = int(t);
  }
#endif
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    half total = 0;
    for (int e = 0; e < KK; e++) total = m[sel[e]] + total;  // row_reduce_small: total = v + total
    half den = metal::isnan(total) ? total : (total > half(EPSH) ? total : half(EPSH));  // Maximum(sum, 1e-9 as half)
    for (int e = 0; e < KK; e++) {
      int i = sel[e];
      inds[e] = uint(i);
      scores[e] = m[i] / den;
#if MASKED
      slots[e] = uint(metal::max(tab[i], 0));
#endif
    }
#if MASKED
    {
      int h = 0;
      for (int s = 0; s < NE / 32; s++) h += hit_part[s];
      hits_out[0] = hits_in + h;
    }
#endif
  }
"""

/// The routed experts' gate and up rows (gather_qmv_fast arithmetic) of the 8 slots, with silu(gate) * up.
let routedUpSource = """
  uint simd_lid = thread_index_in_simdgroup;
  uint sg = threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup;
  int e = int(sg) / (I / 4);
  int row = (int(sg) % (I / 4)) * 4;
  uint slot = slots[e];
  const int wpr = K / 8;   // uint32 per row (4 bit)
  const int gpr = K / 64;  // groups per row
  float gr[4], ur[4];
  fz_qmv_fast_rows<4, 4>(gw + slot * I * wpr, gs + slot * I * gpr, gb + slot * I * gpr, x, K, row, simd_lid, gr);
  fz_qmv_fast_rows<4, 4>(uw + slot * I * wpr, us + slot * I * gpr, ub + slot * I * gpr, x, K, row, simd_lid, ur);
  if (simd_lid == 0) {
    for (int r = 0; r < 4; r++) {
      half gh = static_cast<half>(gr[r]);
      half uh = static_cast<half>(ur[r]);
      half sl = fz_silu(gh);
      act[e * I + row + r] = sl * uh;
    }
  }
"""

/// The down rows of all 8 slots for 4 output rows per simdgroup, times the scores, summed over the experts in
/// col_reduce_small's order, plus the shared expert's output.
let routedDownSource = """
  uint simd_lid = thread_index_in_simdgroup;
  uint sg = threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup;
  int row = int(sg) * 4;
  const int wpr = I / 8;
  const int gpr = I / 64;
  half total[4] = {0, 0, 0, 0};
  for (int e = 0; e < KK; e++) {
    uint slot = slots[e];
    float dr[4];
    fz_qmv_fast_rows<4, 4>(dw + slot * D * wpr, ds + slot * D * gpr, db + slot * D * gpr, act + e * I, I, row, simd_lid, dr);
    half sc = scores[e];
    for (int r = 0; r < 4; r++) {
      half v = static_cast<half>(dr[r]);
      half p = v * sc;
      total[r] = p + total[r];
    }
  }
  if (simd_lid == 0) {
    for (int r = 0; r < 4; r++) y[row + r] = total[r] + sh[row + r];
  }
"""

/// h + r, then rms_single_row (MLX, NT = D/4 threads x 4) of the sum with weight w.
let addNormSource = """
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  const uint lid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint simd = simdgroup_index_in_threadgroup;
  float acc = 0;
  float thread_x[4];
  for (int i = 0; i < 4; i++) {
    half s = h[lid * 4 + i] + r[lid * 4 + i];
    h_out[lid * 4 + i] = s;
    thread_x[i] = s;
    acc += thread_x[i] * thread_x[i];
  }
  acc = simd_sum(acc);
  if (simd == 0) local_sums[lane] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) local_sums[simd] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    acc = simd_sum(local_sums[lane]);
    if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / uint(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 4; i++) {
    x_out[lid * 4 + i] = w[lid * 4 + i] * static_cast<half>(thread_x[i] * local_inv_mean[0]);
  }
"""

extension Fused {
    /// Prefetch mode: from the gate logits [.., NE] and the slot table, MLX's
    ///   gates = softmax(logits, precise); inds = argpartition(-where(tab >= 0, gates, 0), K-1)[:K]
    ///   slots = max(tab[inds], 0) as uint32; scores = masked[inds] / max(sum, 1e-9)
    ///   hits += sum(tab[argpartition(gates, NE-K)[NE-K:]] >= 0)
    /// Exact mode (tab nil): inds = argpartition(gates, NE-K)[NE-K:], scores = gates[inds] / max(sum, 1e-9).
    /// Returns (slots or nil, scores [1, K], inds [1, K], hits or nil).
    public static func select(_ logits: MLXArray, tab: MLXArray?, hits: MLXArray?, k: Int) -> (slots: MLXArray?, scores: MLXArray, inds: MLXArray, hits: MLXArray?) {
        let NE = logits.dim(-1)
        let masked = tab != nil
        let name = masked ? "fz_select_masked" : "fz_select_plain"
        let kern = KernelCache.shared.get(name) {
            MLXFast.metalKernel(
                name: name, inputNames: masked ? ["logits", "tab", "hits_in"] : ["logits"],
                outputNames: masked ? ["slots", "scores", "inds", "hits_out"] : ["scores", "inds"],
                source: "#define MASKED \(masked ? 1 : 0)\nconstexpr float EPSH = 1e-9f;\n" + selectSource, header: fuseHeader)
        }
        let lg = logits.reshaped([NE])
        if masked {
            let o = kern([lg, tab!, hits!], template: [("NE", NE), ("KK", k)], grid: (NE, 1, 1), threadGroup: (NE, 1, 1),
                         outputShapes: [[1, k], [1, k], [1, k], []], outputDTypes: [.uint32, .float16, .uint32, .int32])
            return (o[0], o[1], o[2], o[3])
        }
        let o = kern([lg], template: [("NE", NE), ("KK", k)], grid: (NE, 1, 1), threadGroup: (NE, 1, 1),
                     outputShapes: [[1, k], [1, k]], outputDTypes: [.float16, .uint32])
        return (nil, o[0], o[1], nil)
    }

    /// Engine.routed(xh, L, slots, scores) + sh: (silu(gather gate) * gather up) -> gather down, times the scores, summed
    /// over the K experts, plus the shared expert's output. Two launches. Returns [1, 1, D] float16.
    public static func routedPlusShared(_ x: MLXArray, W: [MLXArray], S: [MLXArray], B: [MLXArray], slots: MLXArray, scores: MLXArray,
                                        sh: MLXArray) -> MLXArray {
        let D = x.dim(-1)  // hidden
        let I = W[0].dim(1)  // moe intermediate (rows of gate/up)
        let k = slots.size
        let up = KernelCache.shared.get("fz_routed_up") {
            MLXFast.metalKernel(name: "fz_routed_up", inputNames: ["x", "slots", "gw", "gs", "gb", "uw", "us", "ub"], outputNames: ["act"],
                                source: routedUpSource, header: fuseHeader)
        }
        let down = KernelCache.shared.get("fz_routed_down") {
            MLXFast.metalKernel(name: "fz_routed_down", inputNames: ["act", "slots", "scores", "sh", "dw", "ds", "db"], outputNames: ["y"],
                                source: routedDownSource, header: fuseHeader)
        }
        let sl = slots.reshaped([k])
        let a = up([x.reshaped([D]), sl, W[0], S[0], B[0], W[1], S[1], B[1]], template: [("K", D), ("I", I)],
                   grid: (32, k * I / 4, 1), threadGroup: (32, 2, 1), outputShapes: [[k * I]], outputDTypes: [.float16])[0]
        let y = down([a, sl, scores.reshaped([k]), sh.reshaped([D]), W[2], S[2], B[2]], template: [("I", I), ("D", D), ("KK", k)],
                     grid: (32, D / 4, 1), threadGroup: (32, 2, 1), outputShapes: [[D]], outputDTypes: [.float16])[0]
        return y.reshaped([1, 1, D])
    }

    /// (h + r, rmsNorm(h + r, weight: w, eps)) in one launch (D <= 4096, D % 128 == 0).
    public static func addRMSNorm(_ h: MLXArray, _ r: MLXArray, weight w: MLXArray, eps: Float) -> (MLXArray, MLXArray) {
        let D = h.dim(-1)
        let kern = KernelCache.shared.get("fz_add_rms") {
            MLXFast.metalKernel(name: "fz_add_rms", inputNames: ["h", "r", "w", "eps"], outputNames: ["h_out", "x_out"],
                                source: addNormSource, header: fuseHeader)
        }
        let o = kern([h.reshaped([D]), r.reshaped([D]), w, MLXArray(eps)], template: [("D", D)], grid: (D / 4, 1, 1),
                     threadGroup: (D / 4, 1, 1), outputShapes: [h.shape, h.shape], outputDTypes: [.float16, .float16])
        return (o[0], o[1])
    }
}

// MARK: - attention: q/k rms norm + partial RoPE

/// One threadgroup (D/4 threads, MLX rms_single_row's layout) per q head and per k head: rms_norm with the head's
/// weight, then RoPE (rope_single, non-traditional) on the first DIMS dims at `offset`; the other dims are the normed
/// values (MLX copies them before rotating in place).
let qkRopeSource = """
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  threadgroup half tn[D];
  const uint lid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint simd = simdgroup_index_in_threadgroup;
  const int h = threadgroup_position_in_grid.z;
  const bool isq = h < NQ;
  const int hh = isq ? h : h - NQ;
  float acc = 0;
  float thread_x[4];
  for (int i = 0; i < 4; i++) {
    thread_x[i] = isq ? q[hh * 2 * D + lid * 4 + i] : k[hh * D + lid * 4 + i];
    acc += thread_x[i] * thread_x[i];
  }
  acc = simd_sum(acc);
  if (simd == 0) local_sums[lane] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) local_sums[simd] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    acc = simd_sum(local_sums[lane]);
    if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / uint(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 4; i++) {
    int j = lid * 4 + i;
    half wv = isq ? qw[j] : kw[j];
    tn[j] = wv * static_cast<half>(thread_x[i] * local_inv_mean[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device half* out = isq ? qo + hh * D : ko + hh * D;
  for (int i = 0; i < 4; i++) {
    int j = lid * 4 + i;
    if (j >= DIMS) out[j] = tn[j];
  }
  if (lid < uint(DIMS / 2)) {
    float d = static_cast<float>(lid) / static_cast<float>(DIMS / 2);
    float inv_freq = metal::exp2(-d * base);
    float L = scale * static_cast<float>(offset);
    float theta = L * inv_freq;
    float costheta = metal::fast::cos(theta);
    float sintheta = metal::fast::sin(theta);
    float x1 = static_cast<float>(tn[lid]);
    float x2 = static_cast<float>(tn[lid + DIMS / 2]);
    float rx1 = x1 * costheta - x2 * sintheta;
    float rx2 = x1 * sintheta + x2 * costheta;
    out[lid] = static_cast<half>(rx1);
    out[lid + DIMS / 2] = static_cast<half>(rx2);
  }
"""

extension Fused {
    /// From the q projection (heads x [q | gate]) and the k projection: (queries, keys) as Forward.attention builds them
    /// before the cache update: rms_norm(q_norm / k_norm), transposed to [1, heads, 1, D], RoPE(dims) at `offset`.
    public static func qkNormRope(_ qOut: MLXArray, _ kOut: MLXArray, qNorm: MLXArray, kNorm: MLXArray, heads: Int, kvHeads: Int,
                                  headDim D: Int, dims: Int, eps: Float, theta: Float, offset: Int) -> (MLXArray, MLXArray) {
        let kern = KernelCache.shared.get("fz_qk_rope") {
            MLXFast.metalKernel(name: "fz_qk_rope", inputNames: ["q", "k", "qw", "kw", "eps", "base", "scale", "offset"],
                                outputNames: ["qo", "ko"], source: qkRopeSource, header: fuseHeader)
        }
        let o = kern([qOut.reshaped([-1]), kOut.reshaped([-1]), qNorm, kNorm, MLXArray(eps), MLXArray(log2f(theta)), MLXArray(Float(1.0)),
                      MLXArray(Int32(offset))],
                     template: [("D", D), ("NQ", heads), ("DIMS", dims)], grid: (D / 4, 1, heads + kvHeads), threadGroup: (D / 4, 1, 1),
                     outputShapes: [[1, heads, 1, D], [1, kvHeads, 1, D]], outputDTypes: [.float16, .float16])
        return (o[0], o[1])
    }
}

// MARK: - self-test (bit identity against the MLX ops each fusion replaces)

/// Bits of a float16 / float32 / integer array, for exact comparison.
func fzBits(_ a: MLXArray) -> [UInt32] {
    let f = a.reshaped([-1])
    switch a.dtype {
    case .float16, .bfloat16: return f.view(dtype: .uint16).asArray(UInt16.self).map { UInt32($0) }
    case .float32: return f.view(dtype: .uint32).asArray(UInt32.self)
    case .int32: return f.asArray(Int32.self).map { UInt32(bitPattern: $0) }
    case .uint32: return f.asArray(UInt32.self)
    default: return f.asType(.float32).view(dtype: .uint32).asArray(UInt32.self)
    }
}

/// (mismatching elements, max |difference|) between two arrays of the same shape.
func fzDiff(_ a: MLXArray, _ b: MLXArray) -> (Int, Float) {
    precondition(a.size == b.size, "size \(a.shape) vs \(b.shape)")
    let x = fzBits(a), y = fzBits(b)
    let n = zip(x, y).filter { $0 != $1 }.count
    let d = n == 0 ? Float(0) : abs(a.reshaped([-1]).asType(.float32) - b.reshaped([-1]).asType(.float32)).max().item(Float.self)
    return (n, d)
}

func fzRandLin(_ rows: Int, _ cols: Int, bits: Int, scale: Float = 0.05) -> Lin {
    let w = (MLXRandom.normal([rows, cols]) * scale).asType(.float16)
    let (wq, s, b) = quantized(w, groupSize: G, bits: bits)
    eval(wq, s, b!)
    return Lin(w: wq, s: s, b: b!, bits: bits)
}

/// Runs every fused kernel against the MLX ops it replaces on random inputs of the model's shapes (several seeds);
/// returns one line per check ("ok" = bit-identical) and whether all are identical.
public func fuseSelfTest(cfg: TextConfig, seeds: Int = 3, log: (String) -> Void) -> Bool {
    let H = cfg.hidden
    let fwd = Forward(cfg: cfg)
    var allOK = true
    // references: the unfused MLX ops (Forward with every switch off); fused: Fused.* directly
    let saved = (Fuse.router, Fuse.gdn, Fuse.shared, Fuse.attnQKV, Fuse.attnRope)
    Fuse.router = false; Fuse.gdn = false; Fuse.shared = false; Fuse.attnQKV = false; Fuse.attnRope = false
    defer { (Fuse.router, Fuse.gdn, Fuse.shared, Fuse.attnQKV, Fuse.attnRope) = saved }
    func check(_ what: String, _ a: MLXArray, _ b: MLXArray) {
        eval(a, b)
        let (n, d) = fzDiff(a, b)
        if n != 0 { allOK = false }
        log(String(format: "  %@ %@ (%d of %d differ, max |d| %.3g)", n == 0 ? "ok  " : "DIFF", what, n, a.size, d))
    }
    for seed in 0..<seeds {
        MLXRandom.seed(UInt64(1000 + seed))
        log("fuse self-test seed \(seed)")
        let x = (MLXRandom.normal([1, 1, H]) * Float(1 + seed)).asType(.float16)
        // qmv multi: the GDN in_proj shapes (4 bit) and the attention q/k/v shapes
        let gl = [fzRandLin(cfg.convDim, H, bits: 4), fzRandLin(cfg.valueDim, H, bits: 4), fzRandLin(cfg.linVHeads, H, bits: 4),
                  fzRandLin(cfg.linVHeads, H, bits: 4)]
        let ys = Fused.qmvMulti(x.reshaped([H]), gl)
        for (i, l) in gl.enumerated() { check("qmvMulti gdn[\(i)]", ys[i], l(x).reshaped([-1])) }
        // GatedDeltaNet layer (two steps: from zero state, then from the state it produced)
        do {
            let C = cfg.convDim
            let gw = GDNW(inQKV: gl[0], inZ: gl[1], inB: gl[2], inA: gl[3], outProj: fzRandLin(H, cfg.valueDim, bits: 4),
                          conv: (MLXRandom.normal([C, cfg.convK, 1]) * 0.3).asType(.float16),
                          aLog: MLXRandom.normal([cfg.linVHeads]) * 0.5, dtBias: (MLXRandom.normal([cfg.linVHeads]) * 0.5).asType(.float16),
                          norm: (MLXRandom.uniform(low: 0.5, high: 1.5, [cfg.linKDim])).asType(.float16))
            let c1 = GDNCache(), c2 = GDNCache()
            for step in 0..<3 {
                let xs = (MLXRandom.normal([1, 1, H]) * Float(1 + step)).asType(.float16)
                let a = fwd.gdn(gw, xs, c1)
                let b = Fused.gdn(fwd, gw, xs, c2)
                check("gdn step \(step) out", b, a)
                check("gdn step \(step) conv state", c2.conv!, c1.conv!)
                check("gdn step \(step) recurrent state", c2.state!, c1.state!)
            }
        }
        // attention (q/k/v in one launch), three positions
        do {
            let aw = AttnW(q: fzRandLin(cfg.heads * cfg.headDim * 2, H, bits: 4), k: fzRandLin(cfg.kvHeads * cfg.headDim, H, bits: 4),
                           v: fzRandLin(cfg.kvHeads * cfg.headDim, H, bits: 4), o: fzRandLin(H, cfg.heads * cfg.headDim, bits: 4),
                           qNorm: (MLXRandom.uniform(low: 0.5, high: 1.5, [cfg.headDim])).asType(.float16),
                           kNorm: (MLXRandom.uniform(low: 0.5, high: 1.5, [cfg.headDim])).asType(.float16))
            let k1 = KVCache(), k2 = KVCache()
            for step in 0..<3 {
                let xs = (MLXRandom.normal([1, 1, H]) * Float(1 + step)).asType(.float16)
                Fuse.attnQKV = false; Fuse.attnRope = false
                let a = fwd.attention(aw, xs, k1)
                Fuse.attnQKV = true; Fuse.attnRope = true
                let b = fwd.attention(aw, xs, k2)
                Fuse.attnQKV = false; Fuse.attnRope = false
                check("attention step \(step)", b, a)
            }
        }
        // q/k norm + RoPE alone at large offsets
        for off in [37, 777, 4093] {
            let D = cfg.headDim, dims = Int(Double(cfg.headDim) * cfg.partialRotary)
            let qo = (MLXRandom.normal([1, 1, cfg.heads * D * 2]) * 2).asType(.float16)
            let ko = (MLXRandom.normal([1, 1, cfg.kvHeads * D]) * 2).asType(.float16)
            let qn = (MLXRandom.uniform(low: 0.5, high: 1.5, [D])).asType(.float16), kn = (MLXRandom.uniform(low: 0.5, high: 1.5, [D])).asType(.float16)
            let qs = qo.reshaped([1, 1, cfg.heads, -1]).split(parts: 2, axis: -1)
            var qR = MLXFast.rmsNorm(qs[0], weight: qn, eps: cfg.eps).transposed(0, 2, 1, 3)
            var kR = MLXFast.rmsNorm(ko.reshaped([1, 1, cfg.kvHeads, -1]), weight: kn, eps: cfg.eps).transposed(0, 2, 1, 3)
            qR = MLXFast.RoPE(qR, dimensions: dims, traditional: false, base: cfg.ropeTheta, scale: 1.0, offset: off)
            kR = MLXFast.RoPE(kR, dimensions: dims, traditional: false, base: cfg.ropeTheta, scale: 1.0, offset: off)
            let (qF, kF) = Fused.qkNormRope(qo, ko, qNorm: qn, kNorm: kn, heads: cfg.heads, kvHeads: cfg.kvHeads, headDim: D, dims: dims,
                                            eps: cfg.eps, theta: cfg.ropeTheta, offset: off)
            check("q norm+rope offset \(off)", qF, qR)
            check("k norm+rope offset \(off)", kF, kR)
        }
        // shared expert
        let lw = LayerW(inNorm: MLXArray.ones([H], dtype: .float16), postNorm: MLXArray.ones([H], dtype: .float16), attn: nil, gdn: nil,
                        gate: fzRandLin(cfg.experts, H, bits: 8), seGateProj: fzRandLin(cfg.sharedInter, H, bits: 4),
                        seUpProj: fzRandLin(cfg.sharedInter, H, bits: 4), seDownProj: fzRandLin(H, cfg.sharedInter, bits: 4),
                        seGate: fzRandLin(1, H, bits: 8))
        check("shared", Fused.shared(lw, x), fwd.shared(lw, x))
        // routers, early (2H) and lagged (3H)
        let emb = (MLXRandom.normal([H]) * 0.02).asType(.float16)
        let embN = fwd.routerEmbNorm(emb)
        let hp = (MLXRandom.normal([H]) * Float(2 + seed)).asType(.float16)
        let hl = (MLXRandom.normal([H]) * Float(2 + seed)).asType(.float16)
        for lagged in [false, true] {
            let K = (lagged ? 3 : 2) * H
            let r = Router(embNorm: (MLXRandom.uniform(low: 0.5, high: 1.5, [H])).asType(.float16),
                           lin: fzRandLin(cfg.experts, K, bits: 8, scale: 0.03), linBias: (MLXRandom.normal([cfg.experts]) * 0.1).asType(.float16),
                           mlp0: fzRandLin(1024, K, bits: 8, scale: 0.03), mlp0Bias: (MLXRandom.normal([1024]) * 0.1).asType(.float16),
                           mlp2: fzRandLin(cfg.experts, 1024, bits: 8, scale: 0.05), mlp2Bias: (MLXRandom.normal([cfg.experts]) * 0.1).asType(.float16),
                           lagged: lagged)
            let ref = -fwd.routeLogits(r, embN: embN, hPrev: hp, hLag: lagged ? hl : nil)
            check("router \(lagged ? "lagged" : "early")", Fused.routerNegLogits(r, embN: embN, hPrev: hp, hLag: lagged ? hl : nil), ref)
        }
        // the released gate's selection (prefetch: masked + hit counter; exact: plain), with ties
        let NE = cfg.experts, K = cfg.topK
        for variant in 0..<4 {
            var lg = (MLXRandom.normal([1, NE]) * Float(2 + seed)).asType(.float16)
            if variant >= 2 { lg = (lg * 2).round() / 2 }  // coarse logits: tied gates
            let nHeld = [48, 10, 4, 256][variant]
            let perm = MLXRandom.permutation(NE).asArray(Int32.self)
            var t = [Int32](repeating: -1, count: NE)
            for i in 0..<nHeld { t[Int(perm[i])] = Int32(i) }
            let tab = MLXArray(t, [NE])
            let hits0 = MLXArray(Int32(7 * seed))
            let gates = softmax(lg, axis: -1, precise: true)
            let rel = argPartition(gates, kth: NE - K, axis: -1)[.ellipsis, (NE - K)...]
            let hitsR = hits0 + (tab[rel] .>= 0).sum().asType(.int32)
            let masked = which(tab .>= 0, gates, MLXArray.zeros(like: gates))
            let inds = argPartition(-masked, kth: K - 1, axis: -1)[.ellipsis, ..<K]
            let slotsR = maximum(tab[inds], 0).asType(.uint32)
            let sR = takeAlong(masked, inds, axis: -1)
            let scoresR = sR / maximum(sR.sum(axis: -1, keepDims: true), Float(1e-9))
            let f = Fused.select(lg, tab: tab, hits: hits0, k: K)
            check("select masked v\(variant) inds", f.inds, inds)
            check("select masked v\(variant) slots", f.slots!, slotsR)
            check("select masked v\(variant) scores", f.scores, scoresR)
            check("select masked v\(variant) hits", f.hits!, hitsR)
            let sP = takeAlong(gates, rel, axis: -1)
            let scoresP = sP / maximum(sP.sum(axis: -1, keepDims: true), Float(1e-9))
            let fp = Fused.select(lg, tab: nil, hits: nil, k: K)
            check("select exact v\(variant) inds", fp.inds, rel)
            check("select exact v\(variant) scores", fp.scores, scoresP)
        }
        // the routed expert step + shared output
        do {
            let cap = 12, I = cfg.moeInter
            func slab(_ rows: Int, _ cols: Int) -> (MLXArray, MLXArray, MLXArray) {
                let l = fzRandLin(cap * rows, cols, bits: 4, scale: 0.05)
                return (l.w.reshaped([cap, rows, -1]), l.s!.reshaped([cap, rows, -1]), l.b!.reshaped([cap, rows, -1]))
            }
            let g = slab(I, H), u = slab(I, H), d = slab(H, I)
            let W = [g.0, u.0, d.0], S = [g.1, u.1, d.1], B = [g.2, u.2, d.2]
            let slots = MLXArray((0..<K).map { UInt32(($0 * 5 + seed) % cap) }, [1, K])
            let sc = softmax(MLXRandom.normal([1, K]), axis: -1).asType(.float16)
            let xh = expandedDimensions(x.reshaped([1, H]).asType(.float16), axes: [-2, -3])
            let sh = (MLXRandom.normal([1, 1, H]) * 0.1).asType(.float16)
            let lhs0 = MLXArray.zeros([1, K], dtype: .uint32)
            let lhs1 = MLXArray(Array(0..<Int32(K))).asType(.uint32).reshaped([1, K])
            let gt = gatherQuantizedMM(xh, W[0], scales: S[0], biases: B[0], lhsIndices: lhs0, rhsIndices: slots, transpose: true, groupSize: G, bits: 4)
            let up = gatherQuantizedMM(xh, W[1], scales: S[1], biases: B[1], lhsIndices: lhs0, rhsIndices: slots, transpose: true, groupSize: G, bits: 4)
            let y0 = gatherQuantizedMM(silu(gt) * up, W[2], scales: S[2], biases: B[2], lhsIndices: lhs1, rhsIndices: slots, transpose: true, groupSize: G, bits: 4)
                .squeezed(axis: -2)
            let yR = (y0 * expandedDimensions(sc, axis: -1)).sum(axis: -2).reshaped(x.shape).asType(x.dtype) + sh
            check("routed + shared", Fused.routedPlusShared(x, W: W, S: S, B: B, slots: slots, scores: sc, sh: sh), yR)
        }
        // residual add + rms norm
        do {
            let r = (MLXRandom.normal([1, 1, H]) * 3).asType(.float16)
            let w = (MLXRandom.uniform(low: 0.5, high: 1.5, [H])).asType(.float16)
            let hR = x + r
            let xR = MLXFast.rmsNorm(hR, weight: w, eps: cfg.eps)
            let (hF, xF) = Fused.addRMSNorm(x, r, weight: w, eps: cfg.eps)
            check("add + rms_norm h", hF, hR)
            check("add + rms_norm x", xF, xR)
        }
    }
    log(allOK ? "fuse self-test: all bit-identical" : "fuse self-test: DIFFERENCES")
    return allOK
}
