// One token through the store pipeline (mlx_run.py step() + StoreMoE.__call__), in two modes:
//
// exact    the released top-8 every layer; its misses are read on the spot (synchronously, 8 in flight), so the generated
//          tokens are deterministic. One host<->GPU round trip per layer (the released choice) plus the sampler's.
// prefetch the lag-4 routers name each layer's experts four layers early (layers 0-3 at the token's start). The names go
//          from the GPU to a name thread that waits on the router output's event itself (no command-buffer completion wait
//          on the encode thread), applies the cache policy and queues the reads; the reader threads read records straight
//          into the slots and mark them landed in the slot table. The encode thread, at layer L, waits only until layer L's
//          names were applied and its reads landed (normally long done), uploads the slot table, and the released router
//          picks its top-8 among the experts in memory on the GPU. One round trip per token (the sampler).
//          Read overlap (2026-11-03, docs/G4-runtime.md "Changelog"): late wait (the attention, gate, shared expert and the
//          masked top-k are submitted before the encoder waits; it waits for the layer's reads only right before the routed
//          expert step), split reads (StoreOptions.chunks), and the token-start head start (the sampler's argmax is
//          submitted with the logits; the read-back is followed at once by the embedding row and the layer 0-3 routers).
//
// Both modes compute the same arithmetic as the Python runner (same MLX ops on the same core version, Model.swift).
import Foundation
import MLX
import MLXFast
import MLXNN
import Cmlx

@_silgen_name("mlx_phone_gpu_stats") func mlx_phone_gpu_stats(_ busy: UnsafeMutablePointer<Double>, _ buffers: UnsafeMutablePointer<Int>)

@_silgen_name("mlx_phone_dispatches") func mlx_phone_dispatches() -> Int

// phone-store GPU profiler (vendored mlx device.cpp "phone_prof"; docs/G4-phone-gpu.md). Off unless prof_set(1).
@_silgen_name("mlx_phone_prof_set") public func mlx_phone_prof_set(_ on: Int32)
@_silgen_name("mlx_phone_prof_tag") public func mlx_phone_prof_tag(_ tag: Int32)
@_silgen_name("mlx_phone_prof_mark") public func mlx_phone_prof_mark(_ kind: Int32, _ v: Int)
@_silgen_name("mlx_phone_prof_clear") public func mlx_phone_prof_clear()
@_silgen_name("mlx_phone_prof_dump") public func mlx_phone_prof_dump(_ path: UnsafePointer<CChar>) -> Int

/// Profiler region tags (the `tag` of every kernel dispatched while it is set).
public enum ProfTag: Int32 {
    case router03 = 1, inNorm = 2, attn = 3, gdn = 4, post = 5, routerLag = 6, gate = 7, shared = 8, topk = 9, routed = 10,
         resid = 11, head = 12, cast = 14
}

public func gpuStats() -> (busy: Double, buffers: Int, kernels: Int) {
    var b = 0.0
    var n = 0
    mlx_phone_gpu_stats(&b, &n)
    return (b, n, mlx_phone_dispatches())
}

public enum Mode: String { case exact, prefetch }

public struct EngineOptions {
    public var dir: String
    public var store = "store4"
    public var mode: Mode = .prefetch
    public var routers: String? = nil  // nil: default file, "none": no routers
    public var routerBits = 8
    public var m = 0
    public var keep = 31
    public var evict: EvictKind = .hazard
    public var noDrop = false
    public var maxSlots = -1
    public var memCap = 0.0
    public var residentBits = 4
    public var readers = 8
    public var simReads = false
    public var noCache = true
    public var bwCap = 0.0
    public var cacheDir: String
    public var cacheLimitMB = 64
    public var wire = true
    public var lateWait = true  // prefetch: wait for a layer's reads only right before its routed expert step
    public var readChunks = 4  // reads per record (16 KB aligned byte ranges), spread over the readers; 1 = one preadv per record
    public var headStart = true  // the sampler's argmax is submitted with the logits (step); off: encoded by sampleGreedy
    /// Per-layer slot caps (--layer-caps; LayerCaps.parse). nil: maxSlots (or the --mem-cap choice) in every layer.
    public var layerCaps: [Int]? = nil
    /// How the caps were given, for the startup line / JSON / settings ("uniform", "auto:1920", "list", a file name).
    public var layerCapsSpec = "uniform"
    /// --store-codec: auto = layers with layer_XX.psc + .idx are read compressed and decoded on the GPU into the slot.
    public var storeCodec: StoreCodec = .auto
    public var codecStaging = 24
    public var codecCheck = false
    /// Memory-map the int4 resident model and the int8 routers (Weights.swift MappedWeights: their page-aligned
    /// `.mmap.safetensors` copies, written once from the cache files, wrapped in no-copy Metal buffers) instead of reading
    /// them into allocated memory: the same bytes and kernels, but the 1.44 GB are clean file-backed pages outside the
    /// process's phys_footprint. Off: the previous load (loadArrays into MLX buffers). The load refuses (throws) when a
    /// mapped tensor could not be wrapped without a copy, so a footprint planned for mapped weights is never exceeded.
    /// Off by default since 2026-11-06 (docs/G4-runtime.md "Mapped weights on the phone": on the 6 GB iPhone the GPU keeps
    /// the mapped pages in RAM, so no slots are freed, and the cap the mapped budget allowed was jetsammed). Off, the
    /// engine deletes stale page-aligned copies at load (Engine.removeStaleMappedCopies).
    public var mappedWeights = false
    /// Batched prefill (docs/G4-runtime.md "Batched prefill"): the prompt is fed in chunks of this many positions
    /// (Engine.prefill); 0: one token at a time through step(), the path every reference was made with.
    public var prefillChunk = 128  // 2026-11-06 measurement: the whole of a 41- or 120-token prompt in one chunk reads 39-67% fewer bytes than chunks of 8-16
    /// Auto chunk (2026-11-07, docs/G4-runtime.md "Batched prefill", "Chunks as large as memory allows"): > 0 = the chunk
    /// activations' budget in bytes; each chunk is then the largest whose activation memory (PrefillMemory, Prefill.swift)
    /// fits it at its cache offset, capped at what is left to feed (split evenly). prefillChunk stays the kernel warm-up's
    /// first shape. 0: fixed chunks of prefillChunk. Needs prefillChunk > 1 (batched).
    public var prefillAutoBytes = 0.0
    /// Reader threads while a chunk's reads are queued (the extra ones beyond `readers` sleep during decode). 8 since the
    /// 15 Plus measurement (2026-11-06: 8 in flight beat 16 on the phone's drive, as docs/FLASH-iPhone15Plus.md found;
    /// 8 / 16 / 32 equal on the Mac), i.e. the decode's own 8 readers.
    public var prefillReaders = 8
    /// Reads per record during a chunk (0: `readChunks`).
    public var prefillReadChunks = 0
    /// Kernel warm-up at load (Engine.warmBatchedKernels, Prefill.swift): dry batched chunks on a throwaway state so the
    /// first message's prefill does not pay the pipelines' first-use compile. nil: on when prefillChunk > 1.
    public var kernelWarmup: Bool? = nil
    public var kernelWarmupOn: Bool { kernelWarmup ?? (prefillChunk > 1) }
    /// The warm-up also runs one dry decode step (step(dry:): the decode path's kernels, the sampler's, on the throwaway
    /// state; no store call, no namer), so the reply's first token does not pay their first-use compile (2026-11-07).
    public var decodeWarmup = true
    /// Lazy re-land (2026-11-07, docs/G4-runtime.md "Time to first token"): a prefix restore puts the held set back without
    /// reading or moving any record; the next batched chunk reads only those it uses or keeps; step() lands the rest
    /// first. Same computation as the eager re-land (the identity tests). Off: every held record is landed at the restore.
    public var relandLazy = true
    /// Read-ahead during a batched prefill chunk (2026-11-07): per layer, up to this many records of the experts the
    /// lag-4 routers name for the chunk's positions are read at low priority while earlier layers compute, into empty
    /// slots (readAheadOverwrite). -1: up to the layer's cap; 0: off. Needs the routers (prefetch mode). Changes only
    /// which records are in the slab, not what the chunk computes or what is held after it.
    public var prefillReadAhead = -1
    /// The routers' h_prev for chunk positions after the first (not computed yet when the names are made): false (stale) =
    /// the last token before the chunk's, as mlx_run.py --spec-hprev stale; true (shift) = the lag layer's MoE input at the
    /// previous position.
    /// Read-ahead targets: 0 empty slots only; 1 also stale records (not held) the routers did not name; 2 also held
    /// experts the routers did not name (Store.issueReadAhead).
    public var readAheadOverwrite = 0
    public var readAheadShift = true  // 2026-11-07 Mac: recall over the chunk's union 74-86% (shift) vs 68-80% (stale)
    public init(dir: String, cacheDir: String) { self.dir = dir; self.cacheDir = cacheDir }
}

/// The routers' names on their way from the GPU to the store (prefetch mode).
final class Namer: @unchecked Sendable {
    struct Job { let layers: [Int]; let t: Int; let pred: MLXArray; let m: Int }
    let store: Store
    let mon = Monitor()
    var jobs: [Job] = []
    var head = 0
    var done: [MLXArray] = []  // released on the encode thread (MLX objects are freed where they were made)
    var gpuWaitSec = 0.0
    var jobsDone = 0
    var pushed = 0

    init(store: Store) {
        self.store = store
        startThread(name: "namer") { [unowned self] in self.loop() }
    }

    func push(_ j: Job) {
        mon.withLock {
            jobs.append(j)
            pushed += 1
            mon.broadcast()
        }
    }

    /// Wait until every pushed job's names were applied (a batched prefill chunk then owns the store).
    func drain() { mon.withLock { while jobsDone < pushed { mon.wait() } } }

    func loop() {
        while true {
            mon.lock()
            while head >= jobs.count { mon.wait() }
            let j = jobs[head]
            head += 1
            mon.unlock()
            let t0 = now()
            _ = _mlx_array_wait(j.pred.ctx)  // the router output's event: GPU -> this thread, nothing else waits
            let t1 = now()
            store.event("g", j.layers[0], j.t, t0, t1)
            let n = j.layers.count * j.m
            let p = mlx_array_data_uint32(j.pred.ctx)!
            let flat = (0..<n).map { Int(p[$0]) }
            for (i, L) in j.layers.enumerated() {
                store.name(L, j.t, Array(flat[(i * j.m)..<((i + 1) * j.m)]))
            }
            mon.withLock {
                gpuWaitSec += t1 - t0
                jobsDone += 1
                done.append(j.pred)
                if head == jobs.count && head > 1024 {
                    jobs.removeAll(keepingCapacity: true)
                    head = 0
                }
                mon.broadcast()
            }
        }
    }

    func releaseDone() { mon.withLock { done.removeAll(keepingCapacity: true) } }
}

public struct Timers {
    public var encode = 0.0  // encode thread: building graphs and submitting them (everything not blocked)
    public var submit = 0.0  // of which inside asyncEval (MLX encoding the kernels into command buffers)
    public var syncWait = 0.0  // blocked on GPU -> host read-backs (released choice, sampler)
    public var readWait = 0.0  // blocked on the store (layer ready / on-the-spot reads), names + landing
    public var nameWait = 0.0  // of which waiting for a layer's names to be applied (prefetch)
    public var syncs = 0
    public var asyncEvals = 0
}

public final class Engine {
    public let opt: EngineOptions
    public let res: Resident
    public let routers: Routers?
    public let store: Store
    let emb: DiskEmbedding
    /// The model-folder layout in use ("full" | "slim") and where each part came from (startup line, JSON, probe.json).
    public let layout: ModelLayout
    public var embeddingsPath: String { emb.path }
    public let residentSource: String
    public let routerSource: String
    /// phys_footprint added by loading the resident model and the routers (MappedWeights: ~0, the pages are file-backed).
    public private(set) var weightsFootprint = 0
    public var weightsMapped: Bool { res.mapInfo != nil }
    let fwd: Forward
    let namer: Namer?
    /// Slots per layer. `cap` is the uniform cap, or floor(sum / layers) when the caps differ per layer (capsUniform).
    public let cap: Int
    public let caps: [Int]
    public var capsUniform: Bool { caps.allSatisfy { $0 == caps[0] } }
    public let plan: [String: Int]
    let NL: Int, NE: Int, K: Int, H: Int, M: Int, lag: Int

    var kv: [KVCache?] = []
    var gdnc: [GDNCache?] = []
    var hPrev: [MLXArray] = []
    public internal(set) var t = -1  // set by the prefix restore (PrefixCache.swift)
    let lhs0: MLXArray, lhs1: MLXArray
    var tabDev: [([Int32], MLXArray)?]
    var nextArg: (logits: MLXArray, arg: MLXArray, by: ObjectIdentifier?)? = nil  // by: the Sampler that built it (nil: argmax)
    /// Sampling (Sampling.swift) for the tokens after the prompt: installed by the caller after the prefill, removed after
    /// the reply. While installed, step() feeds each token to it (penalties, key index) and the head start submits its
    /// draw with the logits; nil: the argmax path (sampleGreedy), unchanged.
    public var sampler: Sampler? = nil
    var hitsDev = MLXArray(Int32(0))
    public var asked = 0
    public var heldSum = 0, heldN = 0
    public var tm = Timers()
    public var waitByLayer: [Double] = []  // seconds blocked on the store, per layer
    public var traceTokens: Range<Int>? = nil  // --trace: the token clock values whose timeline is recorded
    var tracing = false
    public var stepStarts: [Int: Double] = [:]  // --trace: token clock -> step start (all tokens)
    public var logRel = false
    public var relLog: [[Int]] = []  // exact mode, --dump: the released top-k of every (token, layer), in order
    var tEnter = 0.0  // start of the current host-active stretch
    /// GPU profiler (docs/G4-phone-gpu.md): 0 off; 1 tags the kernels of each existing submit (the run is unchanged);
    /// 2 also evaluates each region on its own (extra asyncEval points, same kernels and arithmetic) so every kernel's
    /// tag is its region. Only with mlx_phone_prof_set(1).
    public var prof = 0
    @inline(__always) func ptag(_ t: ProfTag) { if prof > 0 { mlx_phone_prof_tag(t.rawValue) } }
    @inline(__always) func region(_ t: ProfTag, _ a: [MLXArray]) {
        if prof >= 2 { mlx_phone_prof_tag(t.rawValue); asyncEval(a) }
    }
    /// Teacher forcing: keep, per step and layer, the released top-k expert ids (the experts actually computed), the ids
    /// the routers named (Store.nameLog) and the experts held when the released choice was made (prefetch: the table
    /// the masked top-k used, landed or in flight; exact: before the misses were admitted), until takeRouting().
    public var collectReleased = false {
        didSet {
            store.snapshot { $0.nameLog = collectReleased ? [:] : nil }
            releasedDev.removeAll(); heldLog.removeAll(); collectT.removeAll()
        }
    }
    /// The last logits submitted (step or prefill chunk): a chunk waits for them first, so no earlier GPU work is in flight.
    var lastLogits: MLXArray? = nil
    /// Batched prefill stats since resetStats: chunks, passes (expert groups computed, one GPU wait each beyond the
    /// first), union experts, records read, demotions, seconds waiting for reads / the GPU (inds read-back and passes).
    public var pf = PrefillCounters()
    var releasedDev: [MLXArray] = []
    var heldLog: [[Int]] = []  // per (step, layer), in order
    var collectT: [Int] = []  // the token clock of each collected step

    public init(opt: EngineOptions, log: (String) -> Void) throws {
        self.opt = opt
        let t00 = now()
        var prevLimit = 0
        _ = mlx_set_cache_limit(&prevLimit, opt.cacheLimitMB << 20)
        if opt.wire, let ws = MTLCreateSystemDefaultDeviceShim() {
            _ = mlx_set_wired_limit(&prevLimit, ws)
        }
        layout = try ModelLayout.detect(dir: opt.dir)
        if !opt.mappedWeights { Engine.removeStaleMappedCopies(cacheDir: opt.cacheDir, log: log) }
        let resCache = opt.cacheDir + "/resident_q\(opt.residentBits).safetensors"
        let mem0 = ProcMemory.read()
        var rsrc = FileManager.default.fileExists(atPath: resCache) ? resCache : "built from resident.safetensors into \(resCache)"
        res = try Resident.load(dir: opt.dir, bits: opt.residentBits, cacheDir: opt.cacheDir, mapped: opt.mappedWeights, log: log)
        if let mi = res.mapInfo {
            guard mi.copied == 0 else { throw PSError("mapped weights: \(mi.copied) of \(mi.tensors) resident tensors were copied by MLX (no-copy buffer refused); run with mapped weights off") }
            rsrc = "mapped \(mi.path) (\(mi.tensors) tensors, no copy\(mi.built ? ", written now" : ""))"
        }
        residentSource = rsrc
        let cfg = res.cfg
        NL = cfg.layers; NE = cfg.experts; K = cfg.topK; H = cfg.hidden
        M = opt.m > 0 ? opt.m : K
        fwd = Forward(cfg: cfg)
        emb = try DiskEmbedding(dir: opt.dir, hidden: H, layout: layout)
        var rpath = opt.routers
        if rpath == nil {
            let def = opt.dir + "/routers_k4.npz"
            rpath = FileManager.default.fileExists(atPath: def) ? def : (opt.mode == .exact ? "none" : def)
        }
        if rpath == "none" {
            precondition(opt.mode == .exact, "prefetch mode needs the lag-4 routers")
            routers = nil
            lag = 0
            routerSource = "none"
        } else {
            var p = rpath!
            if p.hasSuffix(".pt") { p = String(p.dropLast(3)) + ".npz" }  // the runner's one-time conversion of the .pt
            let rc = opt.cacheDir + "/" + URL(fileURLWithPath: p).deletingPathExtension().lastPathComponent + "_r\(opt.routerBits).safetensors"
            let rcExisted = FileManager.default.fileExists(atPath: rc)
            let r = try Routers.load(npz: p, bits: opt.routerBits, hidden: H, layers: NL, cacheDir: opt.cacheDir, mapped: opt.mappedWeights, log: log)
            if let mi = r.mapInfo {
                guard mi.copied == 0 else { throw PSError("mapped weights: \(mi.copied) of \(mi.tensors) router tensors were copied by MLX (no-copy buffer refused); run with mapped weights off") }
                routerSource = "mapped \(mi.path) (\(mi.tensors) tensors, no copy\(mi.built ? ", written now" : ""))"
            } else {
                routerSource = rcExisted ? rc : "built from \(p) into \(rc)"
            }
            routers = r
            lag = r.lag
        }
        let mem1 = ProcMemory.read()
        let wfp = mem1.footprint - mem0.footprint
        weightsFootprint = wfp
        log(String(format: "weights %@: resident %.2f GB + routers %.2f GB; phys_footprint %.3f -> %.3f GB (+%.3f), resident pages +%.3f GB (file-backed +%.3f)",
                   opt.mappedWeights ? "memory-mapped" : "loaded", Double(res.nbytes) / 1e9, Double(routers?.nbytes ?? 0) / 1e9,
                   Double(mem0.footprint) / 1e9, Double(mem1.footprint) / 1e9, Double(wfp) / 1e9,
                   Double(mem1.resident - mem0.resident) / 1e9, Double(mem1.external - mem0.external) / 1e9))
        let meta = try StoreMeta(path: "\(opt.dir)/\(opt.store)/meta.json")
        let slotBytes = meta.recordBytes
        let resBytes = res.nbytes, routBytes = routers?.nbytes ?? 0, nl = cfg.layers
        let planFor: (Int) -> [String: Int] = { slots in
            ["resident": resBytes, "routers": routBytes, "slabs": nl * slots * slotBytes]
        }
        var c = opt.maxSlots
        if c < 0 {
            if opt.memCap > 0 {
                let fits = (1...cfg.experts).filter { planFor($0).values.reduce(0, +) <= Int(opt.memCap * 1e9) }
                guard let f = fits.last, f >= cfg.topK else { throw PSError("--mem-cap \(opt.memCap) GB leaves fewer than top-\(cfg.topK) slots") }
                c = f
            } else {
                c = 2 * cfg.topK + 4
            }
        }
        let nExp = cfg.experts
        let cs: [Int]
        if let lc = opt.layerCaps {
            guard lc.count == cfg.layers else { throw PSError("--layer-caps: \(lc.count) caps for \(cfg.layers) layers") }
            cs = lc.map { min(nExp, $0) }
        } else {
            cs = Array(repeating: min(nExp, c), count: cfg.layers)
        }
        caps = cs
        cap = cs.allSatisfy { $0 == cs[0] } ? cs[0] : cs.reduce(0, +) / cfg.layers
        let capMin = cs.min()!
        // Exact mode with routers: at a token, a layer holds the routers' M names and the released top-k as this token's
        // (neither may be evicted), so it needs M + K slots; with fewer, take_slot finds no victim (the precondition is
        // compiled out under -Ounchecked: memory corruption, 2026-11-03). Exact mode is normally run with routers none.
        if opt.mode == .exact && routers != nil && capMin < M + cfg.topK {
            throw PSError("exact mode with routers needs at least \(M + cfg.topK) slots per layer (the routers' \(M) names + the released top-\(cfg.topK) at the same token); cap \(capMin): use --routers none")
        }
        let slotsTotal = cs.reduce(0, +)
        let plan0 = ["resident": resBytes, "routers": routBytes, "slabs": slotsTotal * slotBytes]  // the sum, not layers x max
        if opt.memCap > 0 && plan0.values.reduce(0, +) > Int(opt.memCap * 1e9) {
            throw PSError(String(format: "refusing to start: the plan needs %.2f GB, above --mem-cap %.2f GB", Double(plan0.values.reduce(0, +)) / 1e9, opt.memCap))
        }
        // PHONE_SIM_READS=1 (env, GPU profiling only, docs/G4-phone-gpu.md): records are not read (the slots keep what they
        // hold), so the GPU never waits for the drive; the tokens are not the model's.
        var so = StoreOptions(cap: cap, keep: opt.keep, evict: opt.evict, noDrop: opt.noDrop, readers: opt.readers, simReads: opt.simReads || ProcessInfo.processInfo.environment["PHONE_SIM_READS"] == "1",
                              noCache: opt.noCache, bwCap: opt.bwCap, chunks: opt.readChunks, layerCaps: cs,
                              codec: opt.storeCodec, codecStaging: opt.codecStaging, codecCheck: opt.codecCheck)
        so.prefillReaders = opt.prefillChunk > 0 ? opt.prefillReaders : 0
        so.prefillReadChunks = opt.prefillReadChunks
        let st = try Store(dir: opt.dir, store: opt.store, opt: so)
        store = st
        var p = plan0
        if st.codecStagingBytes > 0 { p["codec_staging"] = st.codecStagingBytes }
        plan = p
        namer = (opt.mode == .prefetch) ? Namer(store: store) : nil
        lhs0 = MLXArray.zeros([1, K], dtype: .uint32)
        lhs1 = MLXArray(Array(0..<Int32(K))).asType(.uint32).reshaped([1, K])
        eval(lhs0, lhs1)
        tabDev = Array(repeating: nil, count: NL)
        log(String(format: "loaded in %.1fs", now() - t00))
        reset()
    }

    /// Mapped weights off: the page-aligned copies an earlier mapped load wrote (`<cache>/*.mmap.safetensors` and
    /// Application Support/phone-store-mmap/, Weights.swift MappedWeights.path) are 1.45 GB of dead files; deleted here,
    /// with one log line when there were any. A copy that cannot be deleted is named in the line and left.
    static func removeStaleMappedCopies(cacheDir: String, log: (String) -> Void) {
        let fm = FileManager.default
        var paths: [String] = []
        if let names = try? fm.contentsOfDirectory(atPath: cacheDir) {
            paths += names.filter { $0.hasSuffix(".mmap.safetensors") }.map { cacheDir + "/" + $0 }
        }
        if let base = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let d = base.appendingPathComponent("phone-store-mmap", isDirectory: true).path
            if let names = try? fm.contentsOfDirectory(atPath: d) { paths += names.map { d + "/" + $0 } }
        }
        guard !paths.isEmpty else { return }
        var bytes = 0, gone = 0
        var failed: [String] = []
        for p in paths {
            let n = (try? fm.attributesOfItem(atPath: p)[.size] as? Int) ?? 0
            do {
                try fm.removeItem(atPath: p)
                bytes += n
                gone += 1
            } catch {
                failed.append("\((p as NSString).lastPathComponent): \(error.localizedDescription)")
            }
        }
        log(String(format: "mapped weights off: deleted %d stale page-aligned cop%@ (%.2f GB)%@", gone, gone == 1 ? "y" : "ies",
                   Double(bytes) / 1e9, failed.isEmpty ? "" : "; could not delete " + failed.joined(separator: ", ")))
    }

    /// A new session: caches, router inputs, token clock (mlx_run.py Session.feed cold + reset_slots).
    public func reset() {
        kv = (0..<NL).map { res.cfg.isLinear($0) ? nil : KVCache() }
        gdnc = (0..<NL).map { res.cfg.isLinear($0) ? GDNCache() : nil }
        hPrev = (0..<NL).map { _ in MLXArray.zeros([H], dtype: .float16) }
        t = -1
    }

    /// A new conversation, as mlx_run.py Session.feed does when the new prompt does not extend what was fed: the model
    /// state (reset) and the held experts (Store.forgetSlots, mlx_run.py reset_slots) are dropped.
    public func forget() {
        reset()
        store.forgetSlots()
        tabDev = Array(repeating: nil, count: NL)
        nextArg = nil
        lastLogits = nil
    }

    public func resetStats() {
        store.resetStats()
        hitsDev = MLXArray(Int32(0))
        asked = 0; heldSum = 0; heldN = 0
        tm = Timers()
        pf = PrefillCounters()
        waitByLayer = Array(repeating: 0, count: NL)
        if let namer { namer.mon.withLock { namer.gpuWaitSec = 0 } }
    }

    // MARK: timing helpers (encode thread)

    @inline(__always) func blockedBegin() -> Double {
        let n = now()
        tm.encode += n - tEnter
        return n
    }
    @inline(__always) func blockedEnd() { tEnter = now() }

    func submit(_ arrays: [MLXArray]) {
        let t0 = now()
        asyncEval(arrays)
        tm.submit += now() - t0
        tm.asyncEvals += 1
    }

    func readInts(_ a: MLXArray) -> [Int] {
        let t0 = blockedBegin()
        let v = a.asArray(UInt32.self).map { Int($0) }
        tm.syncWait += now() - t0
        tm.syncs += 1
        blockedEnd()
        return v
    }

    public func sampleGreedy(_ logits: MLXArray) -> Int {
        tEnter = now()
        var a: MLXArray
        if let na = nextArg, na.logits === logits, na.by == nil { a = na.arg } else { a = argMax(logits) }
        nextArg = nil
        let t0 = blockedBegin()
        let v = Int(a.item(UInt32.self))
        if tracing { store.event("s", -1, t, t0, now()) }
        tm.syncWait += now() - t0
        tm.syncs += 1
        blockedEnd()
        tm.encode += now() - tEnter
        return v
    }

    /// The next token: the installed sampler's draw (its head-start graph when step() built one for these logits), else
    /// sampleGreedy. Same clocks as sampleGreedy.
    public func sample(_ logits: MLXArray) -> Int {
        guard let s = sampler else { return sampleGreedy(logits) }
        tEnter = now()
        var a: MLXArray
        if let na = nextArg, na.logits === logits, na.by == ObjectIdentifier(s) { a = na.arg } else { a = s.token(logits) }
        nextArg = nil
        let t0 = blockedBegin()
        let v = Int(a.item(UInt32.self))
        if tracing { store.event("s", -1, t, t0, now()) }
        tm.syncWait += now() - t0
        tm.syncs += 1
        blockedEnd()
        tm.encode += now() - tEnter
        return v
    }

    // MARK: the expert step

    func releasedScores(_ gates: MLXArray, _ inds: MLXArray) -> MLXArray {
        let s = takeAlong(gates, inds, axis: -1)
        return s / maximum(s.sum(axis: -1, keepDims: true), Float(1e-9))  // norm_topk_prob (qwen3_5_moe: always)
    }

    /// mlx_run.py routed()/moe_qmm (int4): three gather_qmm over the layer's slab, summed with the scores.
    func routed(_ xh: MLXArray, _ L: Int, _ slots: MLXArray, _ scores: MLXArray) -> MLXArray {
        let W = store.W[L], S = store.S[L], B = store.Bi[L]
        let gt = gatherQuantizedMM(xh, W[0], scales: S[0], biases: B[0], lhsIndices: lhs0, rhsIndices: slots, transpose: true, groupSize: G, bits: 4)
        let up = gatherQuantizedMM(xh, W[1], scales: S[1], biases: B[1], lhsIndices: lhs0, rhsIndices: slots, transpose: true, groupSize: G, bits: 4)
        let y = gatherQuantizedMM(silu(gt) * up, W[2], scales: S[2], biases: B[2], lhsIndices: lhs1, rhsIndices: slots, transpose: true, groupSize: G, bits: 4)
            .squeezed(axis: -2)
        return (y * expandedDimensions(scores, axis: -1)).sum(axis: -2)
    }

    // MARK: one token

    /// Seconds step() spent landing a lazy restore's pending records (Store.landPending) since resetStats, and what it did.
    public var pendingLandSec = 0.0
    public var pendingLanded = (relanded: 0, kept: 0, moved: 0, bytes: 0)

    /// Land a lazy restore's pending records now (a token step, a save, or idle time after a restore with nothing to
    /// prefill). No-op when none.
    public func landPending() {
        guard store.hasPending else { return }
        let t0 = now()
        let r = store.landPending()
        pendingLandSec += now() - t0
        pendingLanded = (pendingLanded.relanded + r.relanded, pendingLanded.kept + r.kept, pendingLanded.moved + r.moved,
                         pendingLanded.bytes + r.bytes)
    }

    /// Feed one token; returns the next-token logits, submitted but not waited for.
    public func step(_ token: Int) -> MLXArray { step(token, dry: nil) }

    /// `dry` (the warm-up, warmBatchedKernels): the same op graph on the warm-up's throwaway caches with no store call
    /// (no naming, plan, read or wait; every expert counts as held, slot = expert mod the layer's slots, the slab read as
    /// it is), no namer job, and none of the engine's state changed (clock, h_prev, slot tables, hit counter, sampler,
    /// head start, last logits).
    func step(_ token: Int, dry: DryChunkState?, warmSampler: Sampler? = nil) -> MLXArray {
        if dry == nil && store.hasPending {
            let tl = now()
            landPending()
            tm.readWait += now() - tl
        }
        tEnter = now()
        let isDry = dry != nil
        if !isDry {
            t += 1
            sampler?.append(token)
        }
        tracing = isDry ? false : (traceTokens?.contains(t) ?? false)
        if prof > 0 && !isDry { mlx_phone_prof_mark(10, t) }
        if traceTokens != nil && !isDry { stepStarts[t] = tEnter }
        if traceTokens != nil && !isDry { store.mon.withLock { store.traceOn = tracing } }
        let tTok = tEnter
        let cfg = res.cfg
        if collectReleased && !isDry { collectT.append(t) }
        let kvs = dry?.kv ?? kv, gdns = dry?.gdnc ?? gdnc
        var dryHits = MLXArray(Int32(0))
        let e = emb.row(token)
        var hPrevNext = hPrev
        let exact = opt.mode == .exact
        let embN = routers == nil ? e : fwd.routerEmbNorm(e)
        // layers whose router needs nothing from this pass: named now, in one submission
        if let routers {
            let early = (0..<NL).filter { !routers.layers[$0].lagged }
            if !early.isEmpty {
                let preds = stacked(early.map { fwd.route(routers.layers[$0], m: M, embN: embN, hPrev: hPrev[$0], hLag: nil) })
                if isDry {
                    submit([preds])
                } else if exact {
                    let v = readInts(preds)
                    for (i, L) in early.enumerated() { store.name(L, t, Array(v[(i * M)..<((i + 1) * M)])) }
                } else {
                    ptag(.router03)
                    submit([preds])
                    namer!.push(.init(layers: early, t: t, pred: preds, m: M))
                }
            }
        }
        var h = e.reshaped([1, 1, H])
        var nextIn: MLXArray? = nil  // Fuse.addNorm: this layer's input norm, computed with the previous residual add
        var pendingName: Namer.Job? = nil  // Fuse.mergeSubmit: a lag-router prediction waiting for the layer's next submit
        func submitP(_ a: [MLXArray]) {
            if let j = pendingName {
                submit(a + [j.pred])
                namer!.push(j)
                pendingName = nil
            } else {
                submit(a)
            }
        }
        for L in 0..<NL {
            let lw = res.layers[L]
            let xin = nextIn ?? MLXFast.rmsNorm(h, weight: lw.inNorm, eps: cfg.eps)
            region(.inNorm, [xin])
            let r = cfg.isLinear(L) ? fwd.gdn(lw.gdn!, xin, gdns[L]!) : fwd.attention(lw.attn!, xin, kvs[L]!)
            region(cfg.isLinear(L) ? .gdn : .attn, [r])
            let x: MLXArray  // the MoE input
            if Fuse.addNorm {
                let hx = Fused.addRMSNorm(h, r, weight: lw.postNorm, eps: cfg.eps)
                h = hx.0
                x = hx.1
            } else {
                h = h + r
                x = MLXFast.rmsNorm(h, weight: lw.postNorm, eps: cfg.eps)
            }
            region(.post, [x])
            let xf = x.reshaped([-1, H])
            let x0 = Fuse.select ? x.reshaped([H]) : xf[0]  // the same values; a view instead of a gather
            hPrevNext[L] = x0
            // predict + name: layer L + lag from this layer's MoE input
            if let routers, L + lag < NL, routers.layers[L + lag].lagged {
                let pred = fwd.route(routers.layers[L + lag], m: M, embN: embN, hPrev: hPrev[L + lag], hLag: x0)
                if isDry {
                    submit([pred])
                } else if exact {
                    store.name(L + lag, t, readInts(pred))
                } else {
                    ptag(.routerLag)
                    if Fuse.mergeSubmit {
                        pendingName = .init(layers: [L + lag], t: t, pred: pred, m: M)
                    } else {
                        submit([pred])  // this layer's attention and prediction go to the GPU now
                        namer!.push(.init(layers: [L + lag], t: t, pred: pred, m: M))
                    }
                }
            }
            // the released gate (Fuse.select: softmax, top-k, scores and the hit counter in one kernel, Fused.select)
            let gl = lw.gate(xf)
            let fsel = Fuse.select
            let gates = fsel ? gl : softmax(gl, axis: -1, precise: true)
            var inds = fsel ? gl : argPartition(gates, kth: NE - K, axis: -1)[.ellipsis, (NE - K)...]
            region(.gate, [gates, inds])
            let xh = expandedDimensions(xf.asType(.float16), axes: [-2, -3])
            region(.cast, [xh])
            var scores: MLXArray
            var sh: MLXArray? = nil
            var slots: MLXArray
            if exact && isDry {  // the exact path's kernels; slots = expert mod the slab, no read-back, no store
                if fsel {
                    let f = Fused.select(gl, tab: nil, hits: nil, k: K)
                    inds = f.inds
                    scores = f.scores
                } else {
                    scores = releasedScores(gates, inds)
                }
                sh = fwd.shared(lw, x)
                submit([inds, scores, xh, sh!])
                slots = MLXArray((0..<K).map { UInt32($0 % caps[L]) }, [1, K])
            } else if exact {
                if fsel {
                    let f = Fused.select(gl, tab: nil, hits: nil, k: K)
                    inds = f.inds
                    scores = f.scores
                } else {
                    scores = releasedScores(gates, inds)
                }
                sh = fwd.shared(lw, x)
                submit([inds, scores, xh, sh!])
                store.exactPrepare(L, t)
                if collectReleased { heldLog.append(store.heldIds(L)) }  // teacher forcing: held before the misses
                let rel = readInts(inds)  // exact mode's one round trip per layer
                if logRel { relLog.append(rel) }
                let t0 = blockedBegin()
                var w = 0.0
                let (sl, held) = store.exactChoose(L, t, rel, waited: &w)
                tm.readWait += w
                if !waitByLayer.isEmpty { waitByLayer[L] += w }
                _ = t0
                blockedEnd()
                heldSum += held
                asked += K
                slots = MLXArray(sl, [1, K])
            } else if isDry {  // prefetch path's kernels: every expert held (slot = expert mod the slab), no store
                if opt.lateWait {
                    sh = fwd.shared(lw, x)
                    submit(fsel ? [gl, xh, sh!] : [gates, inds, xh, sh!])
                }
                let tab = MLXArray((0..<NE).map { Int32($0 % caps[L]) }, [NE])
                if fsel {
                    let f = Fused.select(gl, tab: tab, hits: dryHits, k: K)
                    dryHits = f.hits!
                    inds = f.inds
                    slots = f.slots!
                    scores = f.scores
                } else {
                    dryHits = dryHits + (tab[inds] .>= 0).sum().asType(.int32)
                    let masked = which(tab .>= 0, gates, MLXArray.zeros(like: gates))
                    inds = argPartition(-masked, kth: K - 1, axis: -1)[.ellipsis, ..<K]
                    slots = maximum(tab[inds], 0).asType(.uint32)
                    scores = releasedScores(masked, inds)
                }
                if opt.lateWait { submit([slots, scores, xh, sh!]) }
            } else {
                // Late wait: what needs neither the names nor the reads goes to the GPU before the encoder blocks.
                if opt.lateWait {
                    sh = fwd.shared(lw, x)
                    region(.shared, [sh!])
                    ptag(.gate)
                    if !store.isNamed(L, t) { submitP(fsel ? [gl, xh, sh!] : [gates, inds, xh, sh!]) }
                }
                var w = 0.0
                _ = blockedBegin()
                let tw0 = now()
                let named = store.layerNamed(L, t, waited: &w)
                if tracing { store.event("wn", L, t, tw0, now()) }
                tm.nameWait += w
                if !opt.lateWait {
                    let tw1 = now()
                    store.layerLanded(L, t, waited: &w)
                    if tracing { store.event("wl", L, t, tw1, now()) }
                }  // the old order: everything waits for the reads
                tm.readWait += w
                if !waitByLayer.isEmpty { waitByLayer[L] += w }
                blockedEnd()
                heldSum += named.held
                if collectReleased { heldLog.append(named.tab.indices.filter { named.tab[$0] >= 0 }) }  // the masked top-k's set
                if tabDev[L] == nil || tabDev[L]!.0 != named.tab {
                    tabDev[L] = (named.tab, MLXArray(named.tab, [NE]))
                }
                let tab = tabDev[L]!.1
                asked += K
                if fsel {
                    let f = Fused.select(gl, tab: tab, hits: hitsDev, k: K)
                    hitsDev = f.hits!
                    inds = f.inds
                    slots = f.slots!
                    scores = f.scores
                } else {
                    hitsDev = hitsDev + (tab[inds] .>= 0).sum().asType(.int32)
                    let masked = which(tab .>= 0, gates, MLXArray.zeros(like: gates))
                    inds = argPartition(-masked, kth: K - 1, axis: -1)[.ellipsis, ..<K]
                    slots = maximum(tab[inds], 0).asType(.uint32)
                    scores = releasedScores(masked, inds)
                }
                region(.topk, [slots, scores, hitsDev])
                if opt.lateWait {
                    // the routed expert step is the only op that reads the slab: wait for the layer's reads only now
                    ptag(.topk)
                    if !store.isLanded(L) { submitP([slots, scores, xh, sh!]) }
                    var w2 = 0.0
                    _ = blockedBegin()
                    let tw1 = now()
                    store.layerLanded(L, t, waited: &w2)
                    if tracing { store.event("wl", L, t, tw1, now()) }
                    tm.readWait += w2
                    if !waitByLayer.isEmpty { waitByLayer[L] += w2 }
                    blockedEnd()
                }
            }
            if collectReleased && !isDry { releasedDev.append(inds) }  // teacher forcing only (TeacherForce.swift)
            heldN += L == 0 && !isDry ? 1 : 0
            var y: MLXArray
            if Fuse.routed {  // the routed step, the expert sum and + shared in two launches (Fused.routedPlusShared)
                y = Fused.routedPlusShared(x, W: store.W[L], S: store.S[L], B: store.Bi[L], slots: slots, scores: scores,
                                           sh: sh ?? fwd.shared(lw, x))
                region(.routed, [y])
            } else {
                y = routed(xh, L, slots, scores).reshaped(x.shape).asType(x.dtype)
                region(.routed, [y])
                y = y + (sh ?? fwd.shared(lw, x))
                region(.resid, [y])
            }
            ptag(.routed)
            submitP([y])  // the GPU runs the experts while the host builds the next layer
            if tracing { store.event("y", L, t, now(), 0) }
            if Fuse.addNorm {  // h + y and the next layer's input norm (the final norm after the last layer) in one kernel
                (h, nextIn) = Fused.addRMSNorm(h, y, weight: L + 1 < NL ? res.layers[L + 1].inNorm : res.norm, eps: cfg.eps)
            } else {
                h = h + y
            }
            region(.resid, [h])
        }
        ptag(.head)
        let logits = res.lmHead(nextIn ?? MLXFast.rmsNorm(h, weight: res.norm, eps: cfg.eps))[0, -1]
        if isDry {  // the head start's argmax and (the app) the reply sampler's graph, on a throwaway sampler
            var outs = [logits, argMax(logits), dryHits]
            if let ws = warmSampler { outs.append(ws.token(logits)) }
            submit(outs)
            tm.encode += now() - tEnter
            namer?.releaseDone()
            return logits
        }
        if opt.headStart {  // the sampler's argmax rides with the logits: its read-back is the token's only wait on the GPU
            let a: MLXArray
            if let s = sampler {
                a = s.token(logits)
                nextArg = (logits, a, ObjectIdentifier(s))
            } else {
                a = argMax(logits)
                nextArg = (logits, a, nil)
            }
            submit([logits, a, hitsDev])
        } else {
            submit([logits, hitsDev])
        }
        hPrev = hPrevNext
        lastLogits = logits
        if tracing { store.event("tok", -1, t, tTok, now()) }
        tm.encode += now() - tEnter
        namer?.releaseDone()
        return logits
    }

    /// Every step since the last call (collectReleased), in order, one row per layer each: the released top-k ids
    /// (sorted), the ids the routers named at that step ([] when the layer was not named), the experts held when the
    /// released choice was made (sorted). Call after the steps' logits were evaluated (their names are then applied).
    public func takeRouting() -> (released: [[[Int]]], named: [[[Int]]], held: [[[Int]]]) {
        eval(releasedDev)
        let r = releasedDev.map { $0.asArray(UInt32.self).map { Int($0) }.sorted() }
        precondition(r.count == collectT.count * NL && heldLog.count == r.count, "routing log out of step")
        let names = store.snapshot { s -> [[[Int]]] in
            let rows = collectT.map { s.nameLog?[$0] ?? [[Int]](repeating: [], count: NL) }
            for t in collectT { s.nameLog?[t] = nil }
            return rows
        }
        var rel: [[[Int]]] = [], held: [[[Int]]] = []
        for i in 0..<collectT.count {
            rel.append(Array(r[(i * NL)..<((i + 1) * NL)]))
            held.append(Array(heldLog[(i * NL)..<((i + 1) * NL)]))
        }
        releasedDev.removeAll(); heldLog.removeAll(); collectT.removeAll()
        return (rel, names, held)
    }

    public var inMemoryRate: Double {
        if opt.mode == .exact { return 1 - Double(store.snapshot { $0.syncReads }) / Double(max(1, asked)) }
        return Double(hitsDev.item(Int32.self)) / Double(max(1, asked))
    }

    public var namerGPUWait: Double { namer?.mon.withLock { namer!.gpuWaitSec } ?? 0 }
}

// Metal's recommended working set, for the wired limit (as mlx_run.py: mx.set_wired_limit(max_recommended_working_set_size)).
import Metal
func MTLCreateSystemDefaultDeviceShim() -> Int? {
    guard let d = MTLCreateSystemDefaultDevice() else { return nil }
    return Int(d.recommendedMaxWorkingSetSize)
}
