// The resident math of qwen3_5_moe for one decode token, a port of mlx-lm 0.32.0 (qwen3_5.py, qwen3_next.py,
// gated_delta.py, cache.py) as mlx_run.py runs it: the same MLX ops in the same order with the same dtypes and scalar
// constants, and mx.compile exactly where mlx-lm and mlx.nn use it (swiglu, precise_swiglu, compute_g, silu, gelu), so the
// kernels and their rounding are the Python runner's. Batch 1, one position per call (the bench feeds the prompt token by
// token, as mlx_run.py --bench does).
import Foundation
import MLX
import MLXFast
import MLXNN

// mlx_lm/models/activations.py: @partial(mx.compile, shapeless=True)
let swiglu: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) { gate, x in
    silu(gate) * x
}

let preciseSwiglu: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) { h, gate, x in
    let g = silu(gate.asType(.float32))
    let xx = x.asType(.float32)
    return (g * xx).asType(h.dtype)
}

// mlx_lm/models/gated_delta.py: compute_g, @partial(mx.compile, shapeless=True)
let computeG: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) { aLog, a, dtBias in
    exp(-exp(aLog.asType(.float32)) * logAddExp(a + dtBias, 0))  // nn.softplus(x) = logaddexp(x, 0)
}

/// gated_delta.py's _make_gated_delta_packed_kernel (scalar gate, Dk = 128, float32 g and state): mlx-lm's choice for this
/// model's shapes on the GPU (packed_eligible), the same source byte for byte.
let gatedDeltaPackedSource = """

        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);

          // Partials mirror the generic kernel: each 4-element chain is one
          // original lane's sequential accumulation.
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
          // Butterfly levels xor 1,2,4 stay inside this lane (commutative
          // pairwise tree); levels xor 8,16 become the row-group shuffles.
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

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
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }

"""

let gatedDeltaPacked = MLXFast.metalKernel(
    name: "gated_delta_step_packed_btree", inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
    outputNames: ["y", "state_out"], source: gatedDeltaPackedSource)

/// mlx-lm KVCache (step 256), one position at a time.
public final class KVCache {
    var keys: MLXArray? = nil
    var values: MLXArray? = nil
    public var offset = 0
    let step = 256

    func updateAndFetch(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset
        let n = k.dim(2)
        if keys == nil || prev + n > keys!.dim(2) {
            let nSteps = (step + n - 1) / step
            let newK = MLXArray.zeros([k.dim(0), k.dim(1), nSteps * step, k.dim(3)], dtype: k.dtype)
            let newV = MLXArray.zeros([v.dim(0), v.dim(1), nSteps * step, v.dim(3)], dtype: v.dtype)
            if var kk = keys, var vv = values {
                if prev % step != 0 {
                    kk = kk[.ellipsis, ..<prev, 0...]
                    vv = vv[.ellipsis, ..<prev, 0...]
                }
                keys = concatenated([kk, newK], axis: 2)
                values = concatenated([vv, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
        }
        offset += n
        keys![.ellipsis, prev..<offset, 0...] = k
        values![.ellipsis, prev..<offset, 0...] = v
        if offset < keys!.dim(2) {
            return (keys![.ellipsis, ..<offset, 0...], values![.ellipsis, ..<offset, 0...])
        }
        return (keys!, values!)
    }
}

/// mlx-lm ArraysCache(size=2) of a GatedDeltaNet layer: [conv state, recurrent state].
public final class GDNCache {
    var conv: MLXArray? = nil
    var state: MLXArray? = nil
}

public struct Forward {
    public let cfg: TextConfig
    let invScale: Double
    let rmsEps: Float

    public init(cfg: TextConfig) {
        self.cfg = cfg
        invScale = pow(Double(cfg.linKDim), -0.5)
        rmsEps = Float(1e-6 * invScale * invScale)
    }

    /// qwen3_5.GatedDeltaNet.__call__ for B = 1, S = 1, no mask.
    public func gdn(_ w: GDNW, _ x: MLXArray, _ c: GDNCache) -> MLXArray {
        if Fuse.gdn { return Fused.gdn(self, w, x, c) }  // the same arithmetic in 3 launches (Kernels.swift)
        let B = 1, S = 1
        let nv = cfg.linVHeads, dv = cfg.linVDim, nk = cfg.linKHeads, dk = cfg.linKDim
        let qkv = w.inQKV(x)
        let z = w.inZ(x).reshaped([B, S, nv, dv])
        let b = w.inB(x)
        let a = w.inA(x)
        let convState = c.conv ?? MLXArray.zeros([B, cfg.convK - 1, cfg.convDim], dtype: x.dtype)
        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = cfg.convK - 1
        c.conv = contiguous(convInput[0..., (convInput.dim(1) - nKeep)..., 0...])
        let convOut = silu(conv1d(convInput, w.conv, stride: 1, padding: 0, dilation: 1, groups: cfg.convDim))
        let parts = convOut.split(indices: [cfg.keyDim, 2 * cfg.keyDim], axis: -1)
        var q = parts[0].reshaped([B, S, nk, dk])
        var k = parts[1].reshaped([B, S, nk, dk])
        let v = parts[2].reshaped([B, S, nv, dv])
        // normalize_qk(q, k, inv_scale=head_k_dim**-0.5, eps=1e-6)
        q = Float(invScale * invScale) * MLXFast.rmsNorm(q, weight: .mlxNone, eps: rmsEps)
        k = Float(invScale) * MLXFast.rmsNorm(k, weight: .mlxNone, eps: rmsEps)
        // gated_delta_update(..., use_kernel=True)
        let beta = sigmoid(b)
        let g = computeG(w.aLog, a, w.dtBias)
        let state = c.state ?? MLXArray.zeros([B, nv, dv, dk], dtype: .float32)
        let outs = gatedDeltaPacked(
            [q, k, v, g, beta, state, MLXArray(Int32(S))],
            template: [("InT", q.dtype), ("StT", state.dtype), ("Dk", dk), ("Dv", dv), ("Hk", nk), ("Hv", nv)],
            grid: (32, dv / 8, B * nv), threadGroup: (32, 2, 1),
            outputShapes: [[B, S, nv, dv], state.shape], outputDTypes: [q.dtype, state.dtype])
        c.state = outs[1]
        // Qwen3NextRMSNormGated(out, z): rms_norm then precise_swiglu(hidden_states, gate, x)
        let y = outs[0]
        let normed = MLXFast.rmsNorm(y, weight: w.norm, eps: cfg.eps)
        let o = preciseSwiglu(y, z, normed)
        return w.outProj(o.reshaped([B, S, -1]))
    }

    /// qwen3_next.Qwen3NextAttention.__call__ (gated attention, q/k norms, partial RoPE), one position.
    public func attention(_ w: AttnW, _ x: MLXArray, _ c: KVCache) -> MLXArray {
        let B = 1, L = 1
        var qOut: MLXArray, kOut: MLXArray, vOut: MLXArray
        if Fuse.attnQKV {  // q/k/v projections in one launch (Kernels.swift), the same rows
            let p = Fused.qmvMulti(x.reshaped([-1]), [w.q, w.k, w.v])
            qOut = p[0].reshaped([B, L, -1]); kOut = p[1].reshaped([B, L, -1]); vOut = p[2].reshaped([B, L, -1])
        } else {
            qOut = w.q(x); kOut = w.k(x); vOut = w.v(x)
        }
        let qs = qOut.reshaped([B, L, cfg.heads, -1]).split(parts: 2, axis: -1)
        var queries = qs[0]
        let gate = qs[1].reshaped([B, L, -1])
        var keys = kOut
        var values = vOut
        let dims = Int(Double(cfg.headDim) * cfg.partialRotary)
        if Fuse.attnRope {  // q/k norms + RoPE in one launch (Kernels.swift), the same arithmetic
            (queries, keys) = Fused.qkNormRope(qOut, kOut, qNorm: w.qNorm, kNorm: w.kNorm, heads: cfg.heads, kvHeads: cfg.kvHeads,
                                               headDim: cfg.headDim, dims: dims, eps: cfg.eps, theta: cfg.ropeTheta, offset: c.offset)
        } else {
            queries = MLXFast.rmsNorm(queries, weight: w.qNorm, eps: cfg.eps).transposed(0, 2, 1, 3)
            keys = MLXFast.rmsNorm(keys.reshaped([B, L, cfg.kvHeads, -1]), weight: w.kNorm, eps: cfg.eps).transposed(0, 2, 1, 3)
            queries = MLXFast.RoPE(queries, dimensions: dims, traditional: false, base: cfg.ropeTheta, scale: 1.0, offset: c.offset)
            keys = MLXFast.RoPE(keys, dimensions: dims, traditional: false, base: cfg.ropeTheta, scale: 1.0, offset: c.offset)
        }
        values = values.reshaped([B, L, cfg.kvHeads, -1]).transposed(0, 2, 1, 3)
        (keys, values) = c.updateAndFetch(keys, values)
        let scale = Float(pow(Double(cfg.headDim), -0.5))
        var out = MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: nil)
        out = out.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return w.o(out * sigmoid(gate))
    }

    // MARK: batched prefill (Engine.prefill): S positions at once, the unfused ops of mlx-lm with S > 1

    /// qwen3_5.GatedDeltaNet.__call__ for B = 1 and S = x.dim(1) positions, no mask (create_ssm_mask is None without
    /// padding): the unfused decode path's ops with S in the shapes; the packed gated-delta kernel loops over T = S itself
    /// (mlx-lm's choice for these shapes at any T). The fused decode kernels (Kernels.swift) are single-position.
    public func gdnSeq(_ w: GDNW, _ x: MLXArray, _ c: GDNCache) -> MLXArray {
        let B = 1, S = x.dim(1)
        let nv = cfg.linVHeads, dv = cfg.linVDim, nk = cfg.linKHeads, dk = cfg.linKDim
        let qkv = w.inQKV(x)
        let z = w.inZ(x).reshaped([B, S, nv, dv])
        let b = w.inB(x)
        let a = w.inA(x)
        let convState = c.conv ?? MLXArray.zeros([B, cfg.convK - 1, cfg.convDim], dtype: x.dtype)
        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = cfg.convK - 1
        c.conv = contiguous(convInput[0..., (convInput.dim(1) - nKeep)..., 0...])
        let convOut = silu(conv1d(convInput, w.conv, stride: 1, padding: 0, dilation: 1, groups: cfg.convDim))
        let parts = convOut.split(indices: [cfg.keyDim, 2 * cfg.keyDim], axis: -1)
        var q = parts[0].reshaped([B, S, nk, dk])
        var k = parts[1].reshaped([B, S, nk, dk])
        let v = parts[2].reshaped([B, S, nv, dv])
        q = Float(invScale * invScale) * MLXFast.rmsNorm(q, weight: .mlxNone, eps: rmsEps)
        k = Float(invScale) * MLXFast.rmsNorm(k, weight: .mlxNone, eps: rmsEps)
        let beta = sigmoid(b)
        let g = computeG(w.aLog, a, w.dtBias)
        let state = c.state ?? MLXArray.zeros([B, nv, dv, dk], dtype: .float32)
        let outs = gatedDeltaPacked(
            [q, k, v, g, beta, state, MLXArray(Int32(S))],
            template: [("InT", q.dtype), ("StT", state.dtype), ("Dk", dk), ("Dv", dv), ("Hk", nk), ("Hv", nv)],
            grid: (32, dv / 8, B * nv), threadGroup: (32, 2, 1),
            outputShapes: [[B, S, nv, dv], state.shape], outputDTypes: [q.dtype, state.dtype])
        c.state = outs[1]
        let y = outs[0]
        let normed = MLXFast.rmsNorm(y, weight: w.norm, eps: cfg.eps)
        let o = preciseSwiglu(y, z, normed)
        return w.outProj(o.reshaped([B, S, -1]))
    }

    /// qwen3_next.Qwen3NextAttention.__call__ for L = x.dim(1) positions after the cache's: RoPE at the cache offset,
    /// SDPA with mlx-lm's "causal" mask (bottom-right aligned against the cached keys) when L > 1.
    public func attentionSeq(_ w: AttnW, _ x: MLXArray, _ c: KVCache) -> MLXArray {
        let B = 1, L = x.dim(1)
        let qOut = w.q(x), kOut = w.k(x), vOut = w.v(x)
        let qs = qOut.reshaped([B, L, cfg.heads, -1]).split(parts: 2, axis: -1)
        let gate = qs[1].reshaped([B, L, -1])
        let dims = Int(Double(cfg.headDim) * cfg.partialRotary)
        var queries = MLXFast.rmsNorm(qs[0], weight: w.qNorm, eps: cfg.eps).transposed(0, 2, 1, 3)
        var keys = MLXFast.rmsNorm(kOut.reshaped([B, L, cfg.kvHeads, -1]), weight: w.kNorm, eps: cfg.eps).transposed(0, 2, 1, 3)
        queries = MLXFast.RoPE(queries, dimensions: dims, traditional: false, base: cfg.ropeTheta, scale: 1.0, offset: c.offset)
        keys = MLXFast.RoPE(keys, dimensions: dims, traditional: false, base: cfg.ropeTheta, scale: 1.0, offset: c.offset)
        var values = vOut.reshaped([B, L, cfg.kvHeads, -1]).transposed(0, 2, 1, 3)
        (keys, values) = c.updateAndFetch(keys, values)
        let scale = Float(pow(Double(cfg.headDim), -0.5))
        var out = MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale,
                                                    mask: L > 1 ? .causal : .none)
        out = out.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return w.o(out * sigmoid(gate))
    }

    /// The shared expert for any number of positions (the unfused ops; Fused.shared is single-position).
    public func sharedSeq(_ w: LayerW, _ x: MLXArray) -> MLXArray {
        sigmoid(w.seGate(x)) * w.seDownProj(swiglu(w.seGateProj(x), w.seUpProj(x)))
    }

    /// The shared expert and its gate: sigmoid(shared_expert_gate(x)) * Qwen3NextMLP(x).
    public func shared(_ w: LayerW, _ x: MLXArray) -> MLXArray {
        if Fuse.shared { return Fused.shared(w, x) }  // 2 launches (Kernels.swift)
        return sigmoid(w.seGate(x)) * w.seDownProj(swiglu(w.seGateProj(x), w.seUpProj(x)))
    }

    /// jobs/router_train.py LagRouter.forward as mlx_run.py route_logits: x = cat[RMSNorm(emb) (eps = fp32 machine epsilon,
    /// in fp32, learned weight), h_prev, h_lag]; logits = lin(x) + mlp(x), mlp = Linear, erf GELU, Linear.
    /// The router input's normalized embedding before the learned weight: the same for every layer's router, so it is
    /// computed once per token (the Python runner computes the identical ops 40 times; the values are the same).
    public func routerEmbNorm(_ emb: MLXArray) -> MLXArray {
        let e32 = emb.asType(.float32)
        return (e32 * rsqrt(mean(e32 * e32, axis: -1, keepDims: true) + Float(1.1920929e-07))).asType(.float16)
    }

    /// `embN` = routerEmbNorm(emb).
    public func routeLogits(_ r: Router, embN: MLXArray, hPrev: MLXArray, hLag: MLXArray?) -> MLXArray {
        let e = embN * r.embNorm
        var parts = [e, hPrev.asType(.float16)]
        if r.lagged { parts.append(hLag!.asType(.float16)) }
        let x = concatenated(parts, axis: -1)
        var z = r.lin(x) + r.linBias
        let h = gelu(r.mlp0(x) + r.mlp0Bias)
        z = z + r.mlp2(h) + r.mlp2Bias
        return z
    }

    public func route(_ r: Router, m: Int, embN: MLXArray, hPrev: MLXArray, hLag: MLXArray?) -> MLXArray {
        if Fuse.router {  // -routeLogits in 2 launches (Kernels.swift), then the same argpartition
            return argPartition(Fused.routerNegLogits(r, embN: embN, hPrev: hPrev, hLag: hLag), kth: m - 1, axis: -1)[.ellipsis, ..<m]
        }
        return argPartition(-routeLogits(r, embN: embN, hPrev: hPrev, hLag: hLag), kth: m - 1, axis: -1)[.ellipsis, ..<m]
    }
}
