// The resident weights, the routers and the input embeddings, loaded exactly as mlx_run.py loads them.
//
// mlx_run.py (section 4): resident.safetensors (bf16) -> 1-D tensors to float32, the zero-centred norms +1 (mlx-lm's
// qwen3_5 sanitize; linear_attn.norm is not shifted), conv1d moved to mlx layout [C, K, 1], everything cast to float16
// except A_log (cast_predicate), then nn.quantize with group 64: the router gates (mlp.gate, shared_expert_gate) at 8 bits,
// every other Linear at --resident-bits. The embedding table is never loaded (--embed disk): one bf16 row per token is
// read from the file. The routers (routers_k4.npz, fp32) -> float16 -> mx.quantize(bits 8, group 64) for the weight
// matrices (--router-bits 8), biases and emb_norm in float16.
//
// The same ops run here on the same MLX core, so the quantized bytes are the ones the Python runner builds at load. The
// result is cached once in native/cache/*.safetensors (the phone loads the cache, never the bf16 file).
//
// Two model-folder layouts (ModelLayout): "full" (resident.safetensors + routers_k4.npz; the cache is built from them
// when missing) and "slim" (no resident.safetensors: the converted cache must be present, and the embedding table is
// read from embeddings.safetensors, the same tensor's bytes alone; see make_embeddings.py). When resident.safetensors
// is present the layout is "full" and everything is read as before.
import Foundation
import Cmlx
import MLX
import Metal

public let G = 64  // quantization group

/// Which model-folder layout the engine runs from. "full": resident.safetensors is present (embedding rows are read from
/// it; the cache is built from it and routers_k4.npz when missing). "slim": no resident.safetensors; the embedding table
/// comes from embeddings.safetensors and the resident model and routers only from the converted cache (cacheDir), so
/// neither resident.safetensors nor routers_k4.npz is needed.
public enum ModelLayout: String {
    case full, slim

    public static let residentFile = "resident.safetensors"
    public static let embeddingsFile = "embeddings.safetensors"

    public static func detect(dir: String) throws -> ModelLayout {
        let fm = FileManager.default
        if fm.fileExists(atPath: dir + "/" + residentFile) { return .full }
        if fm.fileExists(atPath: dir + "/" + embeddingsFile) { return .slim }
        throw PSError("\(dir): neither \(residentFile) (full layout) nor \(embeddingsFile) (slim layout)")
    }
    /// The file the embedding rows are read from.
    public var embeddingsFile: String { self == .full ? Self.residentFile : Self.embeddingsFile }
}

/// A Linear layer: quantized (affine, group 64) or dense float16.
public struct Lin {
    public let w: MLXArray
    public let s: MLXArray?
    public let b: MLXArray?
    public let bits: Int  // 0: dense

    @inline(__always) public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if bits == 0 { return matmul(x, w.T) }
        return quantizedMatmul(x, w, scales: s!, biases: b, transpose: true, groupSize: G, bits: bits)
    }

    public var arrays: [MLXArray] { [w] + (s.map { [$0] } ?? []) + (b.map { [$0] } ?? []) }
    public var nbytes: Int { arrays.reduce(0) { $0 + $1.nbytes } }
}

public final class AttnW {
    public let q, k, v, o: Lin
    public let qNorm, kNorm: MLXArray
    init(q: Lin, k: Lin, v: Lin, o: Lin, qNorm: MLXArray, kNorm: MLXArray) {
        self.q = q; self.k = k; self.v = v; self.o = o; self.qNorm = qNorm; self.kNorm = kNorm
    }
}

public final class GDNW {
    public let inQKV, inZ, inB, inA, outProj: Lin
    public let conv: MLXArray  // [conv_dim, K, 1] float16
    public let aLog: MLXArray  // float32 (cast_predicate: A_log keeps the loaded precision)
    public let dtBias: MLXArray
    public let norm: MLXArray
    init(inQKV: Lin, inZ: Lin, inB: Lin, inA: Lin, outProj: Lin, conv: MLXArray, aLog: MLXArray, dtBias: MLXArray, norm: MLXArray) {
        self.inQKV = inQKV; self.inZ = inZ; self.inB = inB; self.inA = inA; self.outProj = outProj
        self.conv = conv; self.aLog = aLog; self.dtBias = dtBias; self.norm = norm
    }
}

public final class LayerW {
    public let inNorm, postNorm: MLXArray
    public let attn: AttnW?
    public let gdn: GDNW?
    public let gate: Lin  // released router, 8-bit
    public let seGateProj, seUpProj, seDownProj: Lin  // shared expert
    public let seGate: Lin  // shared_expert_gate, 8-bit
    init(inNorm: MLXArray, postNorm: MLXArray, attn: AttnW?, gdn: GDNW?, gate: Lin, seGateProj: Lin, seUpProj: Lin, seDownProj: Lin, seGate: Lin) {
        self.inNorm = inNorm; self.postNorm = postNorm; self.attn = attn; self.gdn = gdn; self.gate = gate
        self.seGateProj = seGateProj; self.seUpProj = seUpProj; self.seDownProj = seDownProj; self.seGate = seGate
    }
}

public struct TextConfig {
    public let hidden: Int, layers: Int, vocab: Int
    public let heads: Int, kvHeads: Int, headDim: Int
    public let linVHeads: Int, linKHeads: Int, linKDim: Int, linVDim: Int, convK: Int
    public let fullInterval: Int
    public let eps: Float
    public let ropeTheta: Float, partialRotary: Double
    public let experts: Int, topK: Int, moeInter: Int, sharedInter: Int

    public init(dir: String) throws {
        let d = try Data(contentsOf: URL(fileURLWithPath: dir + "/config.json"))
        let top = try JSONSerialization.jsonObject(with: d) as! [String: Any]
        let t = (top["text_config"] as? [String: Any]) ?? top
        func i(_ k: String) -> Int { (t[k] as! NSNumber).intValue }
        hidden = i("hidden_size"); layers = i("num_hidden_layers"); vocab = i("vocab_size")
        heads = i("num_attention_heads"); kvHeads = i("num_key_value_heads"); headDim = i("head_dim")
        linVHeads = i("linear_num_value_heads"); linKHeads = i("linear_num_key_heads")
        linKDim = i("linear_key_head_dim"); linVDim = i("linear_value_head_dim"); convK = i("linear_conv_kernel_dim")
        fullInterval = i("full_attention_interval")
        eps = (t["rms_norm_eps"] as! NSNumber).floatValue
        let rp = (t["rope_parameters"] as? [String: Any]) ?? [:]
        ropeTheta = ((rp["rope_theta"] ?? t["rope_theta"]) as! NSNumber).floatValue
        partialRotary = ((rp["partial_rotary_factor"] ?? t["partial_rotary_factor"] ?? 0.25) as! NSNumber).doubleValue
        experts = i("num_experts"); topK = i("num_experts_per_tok"); moeInter = i("moe_intermediate_size")
        sharedInter = i("shared_expert_intermediate_size")
        let rtype = (rp["rope_type"] ?? rp["type"] ?? "default") as! String
        precondition(rtype == "default", "only the default RoPE is wired (config has \(rtype))")
    }
    public func isLinear(_ L: Int) -> Bool { (L + 1) % fullInterval != 0 }
    public var keyDim: Int { linKDim * linKHeads }
    public var valueDim: Int { linVDim * linVHeads }
    public var convDim: Int { keyDim * 2 + valueDim }
}

public final class Resident {
    public let cfg: TextConfig
    public let layers: [LayerW]
    public let norm: MLXArray
    public let lmHead: Lin
    public let bits: Int
    public let nbytes: Int
    public let headBytes: Int
    public let nParams: Int
    /// Memory-mapped (MappedWeights): the file and how many tensors are backed by it; nil: loaded into allocated memory.
    public let mapInfo: MappedWeights.Info?

    init(cfg: TextConfig, layers: [LayerW], norm: MLXArray, lmHead: Lin, bits: Int, nbytes: Int, headBytes: Int, nParams: Int,
         mapInfo: MappedWeights.Info?) {
        self.cfg = cfg; self.layers = layers; self.norm = norm; self.lmHead = lmHead; self.bits = bits
        self.nbytes = nbytes; self.headBytes = headBytes; self.nParams = nParams; self.mapInfo = mapInfo
    }

    static let prefix = "model.language_model."
    static let normSuffixes = [".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight"]

    /// Build (or load from the cache) the resident model at `bits` (0: float16, 4 or 8). `mapped`: the cache's page-aligned
    /// copy (resident_q<bits>.mmap.safetensors, written once) memory-mapped instead of read into allocated memory.
    public static func load(dir: String, bits: Int, cacheDir: String, mapped: Bool = false, log: (String) -> Void) throws -> Resident {
        let cfg = try TextConfig(dir: dir)
        let cachePath = cacheDir + "/resident_q\(bits).safetensors"
        var arrays: [String: MLXArray]
        var mapInfo: MappedWeights.Info? = nil
        if mapped && (FileManager.default.fileExists(atPath: cachePath) || MappedWeights.valid(dst: MappedWeights.path(for: cachePath), src: nil)) {
            let r = try MappedWeights.arrays(cacheFile: cachePath, log: log)
            arrays = r.arrays
            mapInfo = r.info
        } else if FileManager.default.fileExists(atPath: cachePath) {
            arrays = try loadArrays(url: URL(fileURLWithPath: cachePath), stream: .cpu)
        } else {
            guard FileManager.default.fileExists(atPath: dir + "/resident.safetensors") else {
                throw PSError("slim layout: \(cachePath) is missing and there is no resident.safetensors to build it from; copy native/cache/resident_q\(bits).safetensors there")
            }
            log("building \(cachePath) from resident.safetensors (once)")
            arrays = try build(dir: dir, cfg: cfg, bits: bits)
            try FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
            try save(arrays: arrays, metadata: ["format": "mlx", "resident_bits": "\(bits)", "source": "phone-store native: mlx_run.py --resident-bits \(bits) recipe"],
                     url: URL(fileURLWithPath: cachePath))
            if mapped {
                let r = try MappedWeights.arrays(cacheFile: cachePath, log: log)
                arrays = r.arrays
                mapInfo = r.info
            } else {
                arrays = try loadArrays(url: URL(fileURLWithPath: cachePath), stream: .cpu)
            }
        }
        // Materialize on the GPU's memory (unified): evaluate every array once.
        var all: [MLXArray] = []
        func a(_ k: String) -> MLXArray {
            guard let v = arrays[k] else { fatalError("resident cache misses \(k)") }
            all.append(v)
            return v
        }
        func lin(_ k: String) -> Lin {
            if let s = arrays[k + ".scales"] {
                let w = a(k + ".weight")
                all.append(s)
                let b = arrays[k + ".biases"]
                if let b { all.append(b) }
                let bitsHere = w.shape.last! * 32 / (s.shape.last! * G)
                return Lin(w: w, s: s, b: b, bits: bitsHere)
            }
            return Lin(w: a(k + ".weight"), s: nil, b: nil, bits: 0)
        }
        var layers: [LayerW] = []
        for L in 0..<cfg.layers {
            let p = "layers.\(L)."
            var attn: AttnW? = nil
            var gdn: GDNW? = nil
            if cfg.isLinear(L) {
                let q = p + "linear_attn."
                gdn = GDNW(inQKV: lin(q + "in_proj_qkv"), inZ: lin(q + "in_proj_z"), inB: lin(q + "in_proj_b"), inA: lin(q + "in_proj_a"),
                           outProj: lin(q + "out_proj"), conv: a(q + "conv1d.weight"), aLog: a(q + "A_log"), dtBias: a(q + "dt_bias"), norm: a(q + "norm.weight"))
            } else {
                let q = p + "self_attn."
                attn = AttnW(q: lin(q + "q_proj"), k: lin(q + "k_proj"), v: lin(q + "v_proj"), o: lin(q + "o_proj"),
                             qNorm: a(q + "q_norm.weight"), kNorm: a(q + "k_norm.weight"))
            }
            let m = p + "mlp."
            layers.append(LayerW(inNorm: a(p + "input_layernorm.weight"), postNorm: a(p + "post_attention_layernorm.weight"), attn: attn, gdn: gdn,
                                 gate: lin(m + "gate"), seGateProj: lin(m + "shared_expert.gate_proj"), seUpProj: lin(m + "shared_expert.up_proj"),
                                 seDownProj: lin(m + "shared_expert.down_proj"), seGate: lin(m + "shared_expert_gate")))
        }
        let norm = a("norm.weight")
        let head = lin("lm_head")
        // the loaded arrays live on the CPU stream's load primitive until evaluated: copy them into GPU-visible buffers now
        for chunk in stride(from: 0, to: all.count, by: 64) {
            eval(Array(all[chunk..<min(chunk + 64, all.count)]))
        }
        let nbytes = all.reduce(0) { $0 + $1.nbytes }
        let nParams = Int(arrays["__n_params__"]?.item(Int64.self) ?? 0)
        return Resident(cfg: cfg, layers: layers, norm: norm, lmHead: head, bits: bits, nbytes: nbytes, headBytes: head.nbytes, nParams: nParams,
                        mapInfo: mapInfo)
    }

    /// resident.safetensors -> the runner's arrays (names without the model.language_model. prefix).
    static func build(dir: String, cfg: TextConfig, bits: Int) throws -> [String: MLXArray] {
        let raw = try loadArrays(url: URL(fileURLWithPath: dir + "/resident.safetensors"), stream: .cpu)
        var out: [String: MLXArray] = [:]
        var nParams = 0
        let keys = raw.keys.sorted()
        var pending: [MLXArray] = []
        for key in keys {
            if key.hasSuffix("embed_tokens.weight") || key.hasPrefix("mtp.") { continue }  // --embed disk; no MTP
            var v = raw[key]!
            nParams += v.size
            let name = key.hasPrefix(prefix) ? String(key.dropFirst(prefix.count)) : key  // lm_head.weight stays
            if v.ndim == 1 { v = v.asType(.float32) }
            if name.contains("conv1d.weight") && v.shape.last! != 1 { v = v.movedAxis(source: 2, destination: 1) }
            if v.ndim == 1 && (normSuffixes.contains { name.hasSuffix($0) } || name == "norm.weight") { v = v + 1.0 }
            if !name.hasSuffix("A_log") { v = v.asType(.float16) }
            let isLinear = v.ndim == 2 && name.hasSuffix(".weight") && !name.contains("conv1d")
            if isLinear && bits > 0 && v.shape[1] % G == 0 {
                let b = (name.hasSuffix("mlp.gate.weight") || name.hasSuffix("shared_expert_gate.weight")) ? 8 : bits
                let (wq, s, bi) = quantized(v, groupSize: G, bits: b)
                let base = String(name.dropLast(".weight".count))
                out[base + ".weight"] = wq
                out[base + ".scales"] = s
                out[base + ".biases"] = bi!
                pending += [wq, s, bi!]
            } else {
                out[name] = v
                pending.append(v)
            }
            if pending.count >= 24 {
                eval(pending)
                pending.removeAll()
            }
        }
        eval(pending)
        out["__n_params__"] = MLXArray(int64: [nParams])
        return out
    }
}

// MARK: - routers

public final class Router {
    public let embNorm: MLXArray
    public let lin: Lin, linBias: MLXArray
    public let mlp0: Lin, mlp0Bias: MLXArray
    public let mlp2: Lin, mlp2Bias: MLXArray
    public let lagged: Bool
    init(embNorm: MLXArray, lin: Lin, linBias: MLXArray, mlp0: Lin, mlp0Bias: MLXArray, mlp2: Lin, mlp2Bias: MLXArray, lagged: Bool) {
        self.embNorm = embNorm; self.lin = lin; self.linBias = linBias; self.mlp0 = mlp0; self.mlp0Bias = mlp0Bias
        self.mlp2 = mlp2; self.mlp2Bias = mlp2Bias; self.lagged = lagged
    }
    public var arrays: [MLXArray] { [embNorm, linBias, mlp0Bias, mlp2Bias] + lin.arrays + mlp0.arrays + mlp2.arrays }
}

public struct Routers {
    public let layers: [Router]
    public let lag: Int
    public let nbytes: Int
    public let bits: Int
    /// Memory-mapped (MappedWeights; routers_k4_r<bits>.mmap.safetensors), or nil.
    public let mapInfo: MappedWeights.Info?

    /// routers_k4.npz (fp32, the runner's conversion of the .pt) -> float16 -> int8 matrices (mx.quantize group 64) when
    /// bits == 8. Cached in cacheDir.
    public static func load(npz: String, bits: Int, hidden H: Int, layers NL: Int, cacheDir: String, mapped: Bool = false,
                            log: (String) -> Void) throws -> Routers {
        let base = URL(fileURLWithPath: npz).deletingPathExtension().lastPathComponent
        let cachePath = cacheDir + "/\(base)_r\(bits).safetensors"
        var arrays: [String: MLXArray]
        var mapInfo: MappedWeights.Info? = nil
        if mapped && (FileManager.default.fileExists(atPath: cachePath) || MappedWeights.valid(dst: MappedWeights.path(for: cachePath), src: nil)) {
            let r = try MappedWeights.arrays(cacheFile: cachePath, log: log)
            arrays = r.arrays
            mapInfo = r.info
        } else if FileManager.default.fileExists(atPath: cachePath) {
            arrays = try loadArrays(url: URL(fileURLWithPath: cachePath), stream: .cpu)
        } else {
            guard FileManager.default.fileExists(atPath: npz) else {
                throw PSError("\(cachePath) is missing and there is no \(npz) to build it from; copy native/cache/\(base)_r\(bits).safetensors there")
            }
            log("building \(cachePath) from \(npz) (once)")
            let z = try readNpz(npz)
            arrays = [:]
            for (k, v) in z {
                if k == "__lag__" {
                    let lag: Int
                    switch v.descr {
                    case "<i8": lag = Int(v.data.withUnsafeBytes { $0.load(as: Int64.self) })
                    case "<i4": lag = Int(v.data.withUnsafeBytes { $0.load(as: Int32.self) })
                    default: throw PSError("__lag__ dtype \(v.descr)")
                    }
                    arrays["__lag__"] = MLXArray(Int32(lag))
                    continue
                }
                guard v.descr == "<f4" else { throw PSError("router \(k): dtype \(v.descr)") }
                let a32 = MLXArray(v.data, v.shape, type: Float.self)
                let a16 = a32.asType(.float16)  // --router-dtype float16
                if bits > 0 && k.hasSuffix(".weight") && !k.hasSuffix("emb_norm.weight") {
                    let (wq, s, b) = quantized(a16, groupSize: G, bits: bits)
                    let stem = String(k.dropLast(".weight".count))
                    arrays[stem + ".weight"] = wq
                    arrays[stem + ".scales"] = s
                    arrays[stem + ".biases"] = b!
                    eval(wq, s, b!)
                } else {
                    arrays[k] = a16
                    eval(a16)
                }
            }
            try FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
            try save(arrays: arrays, metadata: ["format": "mlx", "router_bits": "\(bits)", "source": npz], url: URL(fileURLWithPath: cachePath))
            if mapped {
                let r = try MappedWeights.arrays(cacheFile: cachePath, log: log)
                arrays = r.arrays
                mapInfo = r.info
            } else {
                arrays = try loadArrays(url: URL(fileURLWithPath: cachePath), stream: .cpu)
            }
        }
        let lag = Int(arrays["__lag__"]!.item(Int32.self))
        func lin(_ k: String) -> Lin {
            if let s = arrays[k + ".scales"] {
                let w = arrays[k + ".weight"]!
                return Lin(w: w, s: s, b: arrays[k + ".biases"], bits: w.shape.last! * 32 / (s.shape.last! * G))
            }
            return Lin(w: arrays[k + ".weight"]!, s: nil, b: nil, bits: 0)
        }
        var rs: [Router] = []
        for L in 0..<NL {
            let p = "\(L)."
            let l = lin(p + "lin")
            let inDim = l.bits == 0 ? l.w.shape[1] : l.w.shape[1] * 32 / l.bits
            rs.append(Router(embNorm: arrays[p + "emb_norm.weight"]!, lin: l, linBias: arrays[p + "lin.bias"]!, mlp0: lin(p + "mlp.0"),
                             mlp0Bias: arrays[p + "mlp.0.bias"]!, mlp2: lin(p + "mlp.2"), mlp2Bias: arrays[p + "mlp.2.bias"]!, lagged: inDim == 3 * H))
        }
        let all = rs.flatMap { $0.arrays }
        eval(all)
        return Routers(layers: rs, lag: lag, nbytes: all.reduce(0) { $0 + $1.nbytes }, bits: bits, mapInfo: mapInfo)
    }
}

// MARK: - input embeddings, one row read from disk per token (--embed disk)

/// One row per token read with pread from `layout.embeddingsFile`: resident.safetensors (full layout) or
/// embeddings.safetensors (slim layout; the same tensor name, dtype and bytes, only the data offset differs).
public final class DiskEmbedding {
    let fd: Int32
    let base: Int
    let rowBytes: Int
    let dtype: DType
    let H: Int
    public let path: String
    public init(dir: String, hidden: Int, layout: ModelLayout) throws {
        path = dir + "/" + layout.embeddingsFile
        let hdr = try safetensorsHeader(path)
        guard let (_, e) = hdr.first(where: { $0.key.hasSuffix("embed_tokens.weight") && !$0.key.hasPrefix("mtp.") }) else { throw PSError("no embedding") }
        switch e.dtype {
        case "BF16": dtype = .bfloat16
        case "F16": dtype = .float16
        case "F32": dtype = .float32
        default: throw PSError("embedding dtype \(e.dtype)")
        }
        guard e.shape.count == 2, e.shape[1] == hidden, e.end - e.begin == e.shape[0] * hidden * dtype.size else {
            throw PSError("\(path): embedding shape \(e.shape) does not match hidden \(hidden)")
        }
        H = hidden
        rowBytes = dtype.size * hidden
        base = e.begin
        fd = open(path, O_RDONLY)
        if fd < 0 { throw PSError("open \(path)") }
    }
    /// The row of `token` as float16 [H] (Python: mx.array(rows).view(EMB_DT).astype(DT)).
    public func row(_ token: Int) -> MLXArray {
        var buf = [UInt8](repeating: 0, count: rowBytes)
        let n = buf.withUnsafeMutableBytes { pread(fd, $0.baseAddress, rowBytes, off_t(base + token * rowBytes)) }
        precondition(n == rowBytes, "short embedding read")
        let raw = MLXArray(buf, [rowBytes])
        return raw.view(dtype: dtype).asType(.float16).reshaped([H])
    }
}

// MARK: - memory-mapped weights (2026-11-04)

/// The resident model and the routers backed by their file instead of allocated memory.
///
/// `<stem>.mmap.safetensors` beside a cache file holds the same tensors with the same bytes, each one starting on a 16 KB
/// page (the gaps are `__pad__.N` U8 tensors, so it is still a valid safetensors file; the header is padded so the data
/// region starts on a page). It is written once from the cache file (`ensure`). `load` maps the whole file read-only
/// (MAP_SHARED) and wraps each tensor's pages in a no-copy Metal buffer (MLXArray(rawPointer:), i.e. MLX's
/// allocator::make_buffer -> newBufferWithBytesNoCopy), sliced to the tensor's bytes, viewed as its dtype and shaped:
/// no byte is copied and the pages are clean and file-backed, so iOS does not charge them to the app's phys_footprint
/// (the GPU reads them every token; under memory pressure the kernel may drop them and fault them back in from flash).
/// Every array's data pointer is checked against the mapping after evaluation; `copied` counts the ones MLX had to copy
/// (it copies when Metal refuses the pointer), which the engine reports and refuses in strict mode.
///
/// iPhone 15 Plus (2026-11-06, `-mapTest` and autotests; docs/G4-runtime.md "Mapped weights on the phone"): zero-copy
/// works (1395/1395 tensors) and the pages stay out of phys_footprint even after CPU and GPU use, but they are not free
/// RAM: MLX's residency set keeps them resident for the GPU, the budget cap they were meant to allow (37) was killed by
/// jetsam (vm-pageshortage) during the load, and at cap 19 the kernel dropped and re-read ~0.5 GB of them mid-reply
/// (1.81 tok/s vs 2.03 loaded). Off by default in the app; the copies go to Application Support/phone-store-mmap when
/// the cache folder is not writable (path(for:)).
public enum MappedWeights {
    public static let page = 16384
    public static let formatTag = "phone-store mmap 1"

    public struct Info: Sendable {
        public let path: String
        public let fileBytes: Int
        public let tensors: Int
        public let zeroCopy: Int
        public let copied: Int
        public let built: Bool  // written during this load
        public let seconds: Double
        public var locked: String? = nil  // -mlockWeights: "ok" or the mlock error
    }

    /// Where the page-aligned copy of `<stem>.safetensors` lives: `<stem>.mmap.safetensors` beside it, unless that folder
    /// is not writable by this process and no copy is there yet: then `copyDir` (Application Support/phone-store-mmap).
    /// 2026-11-06, iPhone 15 Plus: Documents/model/cache, made by `devicectl device copy to` (ios/copy_store_to_phone.sh),
    /// refuses the app's open(O_CREAT) with EACCES, so every mapped load failed there before writing a byte (and the
    /// app fell back to the weights in memory). A folder the app made itself (the downloader's) is writable.
    public static func path(for src: String) -> String {
        let beside = (src as NSString).deletingPathExtension + ".mmap.safetensors"
        let fm = FileManager.default
        if fm.fileExists(atPath: beside) { return beside }
        let other = copyDir + "/" + (beside as NSString).lastPathComponent
        if fm.fileExists(atPath: other) { return other }
        return access((src as NSString).deletingLastPathComponent, W_OK) == 0 ? beside : other
    }

    /// The fallback folder for the page-aligned copies (see path(for:)), excluded from backups.
    public static var copyDir: String {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        var u = base.appendingPathComponent("phone-store-mmap", isDirectory: true)
        if !FileManager.default.fileExists(atPath: u.path) {
            try? FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
            var rv = URLResourceValues()
            rv.isExcludedFromBackup = true
            try? u.setResourceValues(rv)
        }
        return u.path
    }

    @inline(__always) static func roundUp(_ n: Int, _ p: Int = page) -> Int { (n + p - 1) / p * p }

    static func header(fd: Int32, path: String) throws -> (meta: [String: String], entries: [String: (String, [Int], Int, Int)], dataStart: Int) {
        var n: UInt64 = 0
        guard pread(fd, &n, 8, 0) == 8, n > 0, n < 1 << 28 else { throw PSError("\(path): bad safetensors header") }
        var h = Data(count: Int(n))
        let got = h.withUnsafeMutableBytes { pread(fd, $0.baseAddress, Int(n), 8) }
        guard got == Int(n), let obj = try JSONSerialization.jsonObject(with: h) as? [String: Any] else { throw PSError("\(path): bad safetensors header") }
        var es: [String: (String, [Int], Int, Int)] = [:]
        for (k, v) in obj where k != "__metadata__" {
            guard let d = v as? [String: Any], let dt = d["dtype"] as? String, let sh = d["shape"] as? [Int], let o = d["data_offsets"] as? [Int]
            else { throw PSError("\(path): bad entry \(k)") }
            es[k] = (dt, sh, o[0], o[1])
        }
        return ((obj["__metadata__"] as? [String: String]) ?? [:], es, 8 + Int(n))
    }

    static func fileSize(_ path: String) -> Int? {
        (try? FileManager.default.attributesOfItem(atPath: path))?[.size].flatMap { ($0 as? NSNumber)?.intValue }
    }

    /// Is `dst` a complete page-aligned copy (of `src`, when `src` is present: same source size)?
    public static func valid(dst: String, src: String?) -> Bool {
        guard let size = fileSize(dst), size % page == 0 else { return false }
        let fd = open(dst, O_RDONLY)
        if fd < 0 { return false }
        defer { close(fd) }
        guard let h = try? header(fd: fd, path: dst), h.meta["phone_store_mmap"] == formatTag, h.dataStart % page == 0,
              let end = h.entries.values.map({ $0.3 }).max(), h.dataStart + end == size else { return false }
        if let src, let ss = fileSize(src), h.meta["source_bytes"] != "\(ss)" { return false }
        return true
    }

    /// Write the page-aligned copy of `src` to `dst` (via dst.tmp and a rename) unless a valid one is there. Refuses when
    /// the volume would keep less than 0.3 GB free. Returns true when it wrote the file.
    @discardableResult
    public static func ensure(src: String, dst: String, log: (String) -> Void) throws -> Bool {
        if valid(dst: dst, src: FileManager.default.fileExists(atPath: src) ? src : nil) { return false }
        guard let srcSize = fileSize(src) else { throw PSError("\(dst) is missing and there is no \(src) to build it from") }
        let fd = open(src, O_RDONLY)
        if fd < 0 { throw PSError("cannot open \(src)") }
        defer { close(fd) }
        let h = try header(fd: fd, path: src)
        // layout: the source's tensors in their file order, each on a page; pads fill the gaps
        let order = h.entries.sorted { $0.value.2 < $1.value.2 }
        var hdr: [String: Any] = [:]
        var meta = h.meta
        meta["phone_store_mmap"] = formatTag
        meta["page"] = "\(page)"
        meta["source"] = (src as NSString).lastPathComponent
        meta["source_bytes"] = "\(srcSize)"
        hdr["__metadata__"] = meta
        var off = 0, pads = 0
        var plan: [(name: String, from: Int, n: Int, to: Int)] = []
        func pad(to target: Int) {
            if target > off {
                hdr["__pad__.\(pads)"] = ["dtype": "U8", "shape": [target - off], "data_offsets": [off, target]] as [String: Any]
                pads += 1
                off = target
            }
        }
        for (name, e) in order {
            pad(to: roundUp(off))
            let n = e.3 - e.2
            hdr[name] = ["dtype": e.0, "shape": e.1, "data_offsets": [off, off + n]] as [String: Any]
            plan.append((name, h.dataStart + e.2, n, off))
            off += n
        }
        pad(to: roundUp(off))
        var hj = try JSONSerialization.data(withJSONObject: hdr, options: [.sortedKeys])
        let dataStart = roundUp(8 + hj.count)
        hj.append(contentsOf: [UInt8](repeating: 0x20, count: dataStart - 8 - hj.count))
        let total = dataStart + off
        if let free = try? URL(fileURLWithPath: (dst as NSString).deletingLastPathComponent)
            .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage,
           Int(free) < total + 300_000_000 {
            throw PSError(String(format: "not enough free space to write %@ (%.2f GB, %.2f GB free)", (dst as NSString).lastPathComponent, Double(total) / 1e9, Double(free) / 1e9))
        }
        let t0 = now()
        log(String(format: "writing %@ (page-aligned copy of %@, %.2f GB, once)", (dst as NSString).lastPathComponent, (src as NSString).lastPathComponent, Double(total) / 1e9))
        let tmp = dst + ".tmp"
        // POSIX open/write/fsync, not FileManager.createFile + FileHandle: on the iPhone (2026-11-06) createFile(atPath:
        // contents: nil) returned false (errno 2) in Documents/model/cache, so FileHandle(forWritingAtPath:) was nil and the
        // mapped load failed before writing a byte. Every error names the step and the byte offset.
        let out = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        if out < 0 { throw PSError("cannot create \(tmp): errno \(errno) (\(String(cString: strerror(errno))))") }
        var ok = false, closed = false
        defer { if !ok { if !closed { close(out) }; unlink(tmp) } }
        var written = 0
        func put(_ d: Data, _ what: String) throws {
            try d.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
                var done = 0
                while done < b.count {
                    let r = write(out, b.baseAddress! + done, b.count - done)
                    if r < 0 {
                        if errno == EINTR { continue }
                        throw PSError(String(format: "write %@ at byte %ld of %ld failed: errno %d (%@)", what, written + done, total, errno,
                                             String(cString: strerror(errno))))
                    }
                    done += r
                }
            }
            written += d.count
        }
        var len = UInt64(hj.count).littleEndian
        try put(withUnsafeBytes(of: &len) { Data($0) }, "header length")
        try put(hj, "header")
        var at = 0
        let chunk = 8 << 20
        var buf = [UInt8](repeating: 0, count: chunk)
        for p in plan {
            if p.to > at { try put(Data(count: p.to - at), "padding before \(p.name)"); at = p.to }
            var done = 0
            while done < p.n {
                let k = min(chunk, p.n - done)
                let r = buf.withUnsafeMutableBytes { pread(fd, $0.baseAddress, k, off_t(p.from + done)) }
                guard r == k else { throw PSError("\(src): short read of \(p.name) (\(r) of \(k) B at \(p.from + done), errno \(errno))") }
                try put(Data(buf[0..<k]), p.name)
                done += k
            }
            at += p.n
        }
        if off > at { try put(Data(count: off - at), "final padding") }
        if fsync(out) != 0 { throw PSError("fsync \(tmp) failed: errno \(errno) (\(String(cString: strerror(errno))))") }
        closed = true
        if close(out) != 0 { throw PSError("close \(tmp) failed: errno \(errno) (\(String(cString: strerror(errno))))") }
        guard fileSize(tmp) == total else { throw PSError("\(tmp): wrote \(fileSize(tmp) ?? -1) B, expected \(total)") }
        if rename(tmp, dst) != 0 { throw PSError("rename \(tmp) -> \(dst) failed: errno \(errno) (\(String(cString: strerror(errno))))") }
        ok = true
        log(String(format: "wrote %@ in %.1f s (%d tensors, %d pads)", (dst as NSString).lastPathComponent, now() - t0, plan.count, pads))
        return true
    }

    /// The whole file, read-only and shared; unmapped when the last array over it is freed.
    final class Mapping: @unchecked Sendable {
        let base: UnsafeMutableRawPointer
        let size: Int
        init(path: String, roundUpSize: Bool = false) throws {
            let fd = open(path, O_RDONLY)
            if fd < 0 { throw PSError("cannot open \(path)") }
            defer { close(fd) }
            var st = stat()
            fstat(fd, &st)
            size = roundUpSize ? MappedWeights.roundUp(Int(st.st_size)) : Int(st.st_size)
            guard size % MappedWeights.page == 0 else { throw PSError("\(path): size \(size) is not a multiple of 16 KB") }
            guard let p = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0), p != MAP_FAILED else { throw PSError("mmap \(path) failed: errno \(errno)") }
            base = p
        }
        private var isLocked = false
        /// mlock the whole mapping: "ok" or the error.
        func lock() -> String {
            if isLocked { return "ok" }
            if mlock(base, size) == 0 { isLocked = true; return "ok" }
            return "errno \(errno) (\(String(cString: strerror(errno))))"
        }
        func unlock() { if isLocked { munlock(base, size); isLocked = false } }
        deinit { unlock(); munmap(base, size) }
    }

    /// Map `path` (a file `ensure` wrote) and return its tensors as MLX arrays over the mapping (no copy), evaluated.
    public static func load(path: String, built: Bool = false) throws -> (arrays: [String: MLXArray], info: Info) {
        let t0 = now()
        let m = try Mapping(path: path)
        let fd = open(path, O_RDONLY)
        if fd < 0 { throw PSError("cannot open \(path)") }
        let h: (meta: [String: String], entries: [String: (String, [Int], Int, Int)], dataStart: Int)
        do { h = try header(fd: fd, path: path) } catch { close(fd); throw error }
        close(fd)
        guard h.meta["phone_store_mmap"] == formatTag, h.dataStart % page == 0 else { throw PSError("\(path): not a page-aligned weights file") }
        var out: [String: MLXArray] = [:]
        var where_: [String: UnsafeMutableRawPointer] = [:]
        for (name, e) in h.entries where !name.hasPrefix("__pad__.") {
            let off = h.dataStart + e.2, n = e.3 - e.2
            guard off % page == 0, off + roundUp(max(n, 1)) <= m.size else { throw PSError("\(path): \(name) is not page-aligned") }
            let dt = try mlxDType(e.0)
            let p = m.base + off
            let L = roundUp(max(n, 1))
            // the tensor's pages as one no-copy buffer (a page multiple: what Metal accepts), then its own bytes, dtype, shape
            let raw = MLXArray(rawPointer: p, [L], dtype: .uint8) { [m] in _ = m }
            var a = n == L ? raw : raw[0 ..< n]
            if dt != .uint8 { a = a.view(dtype: dt) }
            a = a.reshaped(e.1)
            out[name] = a
            where_[name] = p
        }
        let all = Array(out.values)
        for c in stride(from: 0, to: all.count, by: 128) { eval(Array(all[c ..< min(c + 128, all.count)])) }
        var zero = 0, copied = 0
        for (name, a) in out {
            if let d = mlx_array_data_uint8(a.ctx), UnsafeMutableRawPointer(mutating: d) == where_[name]! { zero += 1 } else { copied += 1 }
        }
        var locked: String? = nil
        if lockPages { locked = m.lock() }
        return (out, Info(path: path, fileBytes: m.size, tensors: out.count, zeroCopy: zero, copied: copied, built: built, seconds: now() - t0,
                          locked: locked))
    }

    /// The cache file `src` as mapped arrays: ensure the page-aligned copy, then map it.
    public static func arrays(cacheFile src: String, log: (String) -> Void) throws -> (arrays: [String: MLXArray], info: Info) {
        let dst = path(for: src)
        let built: Bool
        do { built = try ensure(src: src, dst: dst, log: log) } catch {
            lastFailure = "writing the page-aligned copy \((dst as NSString).lastPathComponent): \(error.localizedDescription)"
            throw PSError("mapped weights: " + lastFailure!)
        }
        let r: (arrays: [String: MLXArray], info: Info)
        do { r = try load(path: dst, built: built) } catch {
            lastFailure = "mapping \((dst as NSString).lastPathComponent): \(error.localizedDescription)"
            throw PSError("mapped weights: " + lastFailure!)
        }
        log("mapped \(dst) (\(r.info.tensors) tensors, \(r.info.zeroCopy) zero-copy" + (lockPages ? ", mlock: \(r.info.locked ?? "?")" : "") + ")")
        return r
    }

    /// `-mlockWeights` (launch argument, 2026-11-06 experiment): mlock the whole mapping after mmap, so the kernel cannot
    /// drop the clean weight pages under pressure (on the 15 Plus the mapped run paged ~0.5 GB of them back in during
    /// one reply). Whether wired file-backed pages stay outside phys_footprint is what the experiment measures.
    public static var lockPages: Bool {
        CommandLine.arguments.contains("-mlockWeights") || ProcessInfo.processInfo.environment["PS_MLOCK_WEIGHTS"] == "1"
    }

    /// Why the last mapped load of this process failed (the step and the error), or nil.
    nonisolated(unsafe) public static var lastFailure: String? = nil
}

// MARK: - mapped-weights self-test (-mapTest, 2026-11-06)

/// One step of MappedWeights.selfTest: what the process's memory looked like after it, and whether it worked.
public struct MapTestStep: Codable, Sendable {
    public var step: String
    public var ok: Bool
    public var error: String?
    public var seconds: Double
    public var footprint: Int        // task_vm_info.phys_footprint (what jetsam charges)
    public var external: Int         // task_vm_info.external: file-backed resident pages (the MemProbe's "file-backed")
    public var resident: Int
    public var pageins: Int
    public var mlxActive: Int
    public var detail: String?

    public var line: String {
        String(format: "MAPTEST %@ %@: footprint=%.3f file-backed=%.3f resident=%.3f GB pageins=%ld mlx_active=%.3f GB (%.2f s)%@%@",
               ok ? "ok" : "no", step, Double(footprint) / 1e9, Double(external) / 1e9, Double(resident) / 1e9, pageins,
               Double(mlxActive) / 1e9, seconds, detail.map { " | " + $0 } ?? "", error.map { " | error: " + $0 } ?? "")
    }
}

extension MappedWeights {
    /// The mapped path step by step on this device, the memory after each step (the question of 2026-11-06: does a
    /// file-backed no-copy buffer stay out of phys_footprint on iOS once the GPU uses it?).
    ///  1 write the page-aligned copy of `src` (ensure); on failure the steps below map `src` itself (whole file)
    ///  2 mmap the file (PROT_READ, MAP_SHARED)
    ///  3 one MTLBuffer makeBuffer(bytesNoCopy:) over the whole mapping (our own Metal device, no MLX)
    ///  4 that buffer added to an MTLResidencySet with requestResidency (what MLX does with every buffer on iOS 18)
    ///  5 a GPU blit reading every byte of the buffer into a 64 MB scratch buffer (raw Metal), timed
    ///  6 buffer, set and scratch released
    ///  7 the MLX arrays over the mapping as MappedWeights.load builds them (MLXArray(rawPointer:) per tensor, the
    ///    engine's wired limit set first); zero-copy count from each array's data pointer
    ///  8 the CPU reads one byte of every 16 KB page
    ///  9 one GPU op over every array (an MLX sum of each tensor's bytes), then again, timed
    /// 10 arrays and mapping released
    /// 11 the same file read into MLX memory (loadArrays) and the same GPU op, timed: the speed of allocated memory
    /// 12 released
    public static func selfTest(cacheFile src: String, log: (String) -> Void) -> [MapTestStep] {
        var steps: [MapTestStep] = []
        func rec(_ name: String, _ t0: Double, ok: Bool = true, error: String? = nil, _ detail: String? = nil) {
            let m = ProcMemory.read()
            let s = MapTestStep(step: name, ok: ok, error: error, seconds: now() - t0, footprint: m.footprint, external: m.external,
                                resident: m.resident, pageins: m.pageins, mlxActive: Memory.activeMemory, detail: detail)
            steps.append(s)
            log(s.line)
        }
        PhoneStoreEngine.useMetallib()
        var prev = 0
        _ = mlx_set_cache_limit(&prev, 0)
        if let ws = MTLCreateSystemDefaultDeviceShim() { _ = mlx_set_wired_limit(&prev, ws) }  // as Engine.init (opt.wire)
        rec("baseline", now(), String(format: "src %@ (%.3f GB)", (src as NSString).lastPathComponent, Double(fileSize(src) ?? 0) / 1e9))

        let dst = path(for: src)
        var mapPath = dst, aligned = true
        var t = now()
        do {
            let built = try ensure(src: src, dst: dst, log: log)
            rec("1 page-aligned copy", t, built ? "written now" : "present and valid")
        } catch {
            rec("1 page-aligned copy", t, ok: false, error: error.localizedDescription, "steps 2-10 map the unaligned cache file itself")
            mapPath = src; aligned = false
        }

        t = now()
        var map: Mapping? = nil
        do { map = try Mapping(path: mapPath, roundUpSize: !aligned) } catch {
            rec("2 mmap", t, ok: false, error: error.localizedDescription)
            return steps
        }
        let m = map!
        let mappedSize = m.size
        rec("2 mmap", t, String(format: "%@, %.3f GB at %p", (mapPath as NSString).lastPathComponent, Double(m.size) / 1e9, Int(bitPattern: m.base)))

        // raw Metal
        autoreleasepool {
            t = now()
            guard let dev = MTLCreateSystemDefaultDevice() else { rec("3 MTLBuffer bytesNoCopy", t, ok: false, error: "no Metal device"); return }
            var buf: MTLBuffer? = dev.makeBuffer(bytesNoCopy: m.base, length: m.size, options: [.storageModeShared, .hazardTrackingModeUntracked],
                                                 deallocator: nil)
            guard let b = buf else {
                rec("3 MTLBuffer bytesNoCopy", t, ok: false, error: "makeBuffer(bytesNoCopy:) returned nil")
                return
            }
            rec("3 MTLBuffer bytesNoCopy", t, String(format: "contents==mapping %@, allocatedSize %.3f GB", b.contents() == m.base ? "yes" : "NO",
                                                     Double(b.allocatedSize) / 1e9))
            t = now()
            var rset: AnyObject? = nil
            if #available(iOS 18.0, macOS 15.0, *) {
                let d = MTLResidencySetDescriptor()
                do {
                    let s = try dev.makeResidencySet(descriptor: d)
                    s.addAllocation(b)
                    s.commit()
                    s.requestResidency()
                    rset = s
                    rec("4 residency set + requestResidency", t, String(format: "set allocatedSize %.3f GB", Double(s.allocatedSize) / 1e9))
                } catch {
                    rec("4 residency set + requestResidency", t, ok: false, error: error.localizedDescription)
                }
            } else {
                rec("4 residency set + requestResidency", t, ok: false, error: "MTLResidencySet needs iOS 18")
            }
            t = now()
            let chunk = 64 << 20
            if let q = dev.makeCommandQueue(), let scratch = dev.makeBuffer(length: chunk, options: .storageModePrivate),
               let cb = q.makeCommandBuffer(), let bl = cb.makeBlitCommandEncoder() {
                var o = 0
                while o < m.size {
                    let n = min(chunk, m.size - o)
                    bl.copy(from: b, sourceOffset: o, to: scratch, destinationOffset: 0, size: n)
                    o += n
                }
                bl.endEncoding()
                if let s = rset, #available(iOS 18.0, macOS 15.0, *) { cb.useResidencySet(s as! MTLResidencySet) }
                cb.commit()
                cb.waitUntilCompleted()
                let gpu = cb.gpuEndTime - cb.gpuStartTime
                rec("5 GPU blit over the buffer", t, ok: cb.status == .completed, error: cb.error?.localizedDescription,
                    String(format: "gpu %.3f s, %.2f GB/s", gpu, gpu > 0 ? Double(m.size) / gpu / 1e9 : 0))
            } else {
                rec("5 GPU blit over the buffer", t, ok: false, error: "could not make the queue / scratch / blit encoder")
            }
            t = now()
            if #available(iOS 18.0, macOS 15.0, *), let s = rset as? MTLResidencySet { s.endResidency(); s.removeAllAllocations(); s.commit() }
            rset = nil
            buf = nil
            _ = buf
            rec("6 raw Metal released", t)
        }

        // MLX, as the engine builds the arrays
        t = now()
        var arrays: [String: MLXArray] = [:]
        var zero = 0, copied = 0
        if aligned {
            do {
                let r = try load(path: mapPath)
                arrays = r.arrays; zero = r.info.zeroCopy; copied = r.info.copied
                rec("7 MLX arrays over the mapping (MappedWeights.load)", t, ok: copied == 0, error: copied == 0 ? nil : "\(copied) arrays copied by MLX",
                    "\(r.info.tensors) tensors, \(zero) zero-copy, \(copied) copied")
            } catch {
                rec("7 MLX arrays over the mapping (MappedWeights.load)", t, ok: false, error: error.localizedDescription)
            }
        } else {
            let raw = MLXArray(rawPointer: m.base, [m.size], dtype: .uint8) { [m] in _ = m }
            eval(raw)
            let same = mlx_array_data_uint8(raw.ctx).map { UnsafeMutableRawPointer(mutating: $0) == m.base } ?? false
            arrays = ["whole file": raw]; zero = same ? 1 : 0; copied = same ? 0 : 1
            rec("7 MLX array over the whole mapping (MLXArray(rawPointer:))", t, ok: same, error: same ? nil : "MLX copied the bytes",
                same ? "zero-copy" : "copied")
        }

        t = now()
        var sum: UInt64 = 0
        let p8 = m.base.assumingMemoryBound(to: UInt8.self)
        var o = 0
        while o < m.size { sum &+= UInt64(p8[o]); o += page }
        rec("8 CPU touch of every page", t, "\(m.size / page) pages, checksum \(sum)")
        t = now()
        let lk = m.lock()
        rec("8b mlock of the whole mapping", t, ok: lk == "ok", error: lk == "ok" ? nil : lk)

        func gpuPass() -> Double {
            let t0 = now()
            let all = Array(arrays.values)
            for c in stride(from: 0, to: all.count, by: 64) {
                let part = all[c ..< min(c + 64, all.count)].map { a -> MLXArray in
                    let flat = a.reshaped([-1])
                    return (a.dtype == .uint8 ? flat : flat.view(dtype: .uint8)).sum()
                }
                eval(part)
            }
            return now() - t0
        }
        if !arrays.isEmpty {
            t = now()
            let g1 = gpuPass()
            rec("9a GPU op over every array (first)", t, String(format: "%.3f s, %.2f GB/s", g1, Double(mappedSize) / g1 / 1e9))
            t = now()
            let g2 = gpuPass()
            rec("9b GPU op over every array (again)", t, String(format: "%.3f s, %.2f GB/s", g2, Double(mappedSize) / g2 / 1e9))
        }
        t = now()
        m.unlock()
        arrays = [:]
        map = nil
        Memory.clearCache()
        rec("10 MLX arrays released (the mapping goes with the last reference)", t)

        // the same bytes read into memory, for the speed and footprint of the loaded path
        t = now()
        do {
            arrays = try loadArrays(url: URL(fileURLWithPath: src), stream: .cpu)
            eval(Array(arrays.values))
            rec("11a cache file read into MLX memory (loadArrays)", t, "\(arrays.count) tensors")
            t = now()
            let g1 = gpuPass()
            rec("11b GPU op over every array (first)", t, String(format: "%.3f s, %.2f GB/s", g1, Double(mappedSize) / g1 / 1e9))
            t = now()
            let g2 = gpuPass()
            rec("11c GPU op over every array (again)", t, String(format: "%.3f s, %.2f GB/s", g2, Double(mappedSize) / g2 / 1e9))
        } catch {
            rec("11a cache file read into MLX memory (loadArrays)", t, ok: false, error: error.localizedDescription)
        }
        t = now()
        arrays = [:]
        Memory.clearCache()
        rec("12 released", t)
        _ = mlx_set_cache_limit(&prev, 64 << 20)
        return steps
    }
}
