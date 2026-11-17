// The engine seam an app calls (docs/G4-runtime.md, Phase B): one object owns the Engine, the tokenizer and the
// conversation state, and runs every MLX call on one dedicated user-interactive thread (never the main thread), as the
// CLI runs them on its one thread. generate() applies the chat template, prefills token by token (the deterministic
// path, mlx_run.py --chunk 0), samples (greedy by default; SamplingParams per reply, Sampling.swift) and streams one
// event per generated token with the incremental detokenized piece, its thinking phase (by token id, ChatTemplate.swift)
// and that token's stats. Cancellation is checked between tokens. A new conversation (a prompt that
// does not extend what was fed) forgets the model state and the held experts, as mlx_run.py Session.feed does.
import Foundation
import Cmlx
import MLX
import Tokenizers

public struct TokenEvent: Sendable {
    public let index: Int  // 0-based generated-token index
    public let id: Int
    public let text: String  // incremental detokenized piece ("" while a multi-byte character is incomplete)
    public let ms: Double  // wall time of this token: sample + step + its GPU work
    public let bytesRead: Int  // expert bytes that landed from flash during this token
    public let inMemory: Double  // released top-8 already in memory: running rate over this reply (prefetch: masked top-k hits)
    public let held: Int  // experts held over all layers at this token (<= cap x layers)
    /// Seconds from the first sample of the reply to the end of this token (the engine's decode clock: prefill and the
    /// prefix restore excluded), so (index + 1) / elapsed is the reply's decode rate so far, the rate GenerationResult
    /// reports at the end (tokPerSec = tokens / seconds).
    public let elapsed: Double
    /// Expert bytes landed since the prefill ended (the reply's bytes so far); / (index + 1) = bytes per token so far.
    public let bytesTotal: Int
    /// The token's place in the think block, by token id (ChatTemplate.swift ThinkingTracker): open, thinking, close, answer.
    public let phase: ThinkPhase
    /// Fed by the thinking budget (the template's forced close), not sampled.
    public let forced: Bool
}

public enum StopKind: String, Sendable { case eos, maxNew = "max_new", cancelled }

/// The prompt prefill in progress (2026-11-04): one event before the first prefilled token (after the prefix restore or
/// the live continuation was decided, done = 0) and one after each prefilled token, so a UI can show the real progress.
public struct PrefillEvent: Sendable {
    public let kind: String  // "cold" | "live prefix" | "restored prefix"
    public let promptTokens: Int  // the whole prompt the template gave
    public let reusedTokens: Int  // restored from the prefix cache or continued from the live state (not run again)
    public let total: Int  // tokens this prefill runs (promptTokens - reusedTokens)
    public let done: Int  // of which run so far
    public let bytesRead: Int  // expert bytes read from flash by the prefill so far (the restore's re-land not included)
    public let relandBytes: Int  // bytes the prefix restore read to re-land its experts (0 without a restore)
    public let elapsed: Double  // seconds since generate started: template + restore + prefill so far (+ any state saves)
    public let prefillElapsed: Double  // seconds of prefill so far (the restore and state saves excluded)
}

public struct GenerationResult: Sendable {
    public let reason: StopKind
    public let promptIds: [Int]
    public let ids: [Int]
    public let prefill: String  // "cold" (model state and held experts forgotten), "live prefix" or "restored prefix"
    public let prefillTokens: Int
    public let prefillSeconds: Double
    public let seconds: Double  // first sample -> last token's logits evaluated (the CLI's dt)
    public let tokPerSec: Double
    public let mibPerToken: Double  // 1 MiB = 1048576 B, as the bench JSON
    public let inMemory: Double
    public let heldMean: Double
    public let syncReads: Int
    public let text: String
    /// The reply's per-token split, with the CLI's bench-JSON keys, rounding and definitions (phone-store-run: stats
    /// reset after the prefill, clock from the first sample to the last token's logits): host_encode_ms, expert_wait_ms
    /// (blocked on reads = name_wait_ms + land_wait_ms), sync_wait_ms, gpu_ms_per_token, gpu_busy_frac, drive_gb_per_s,
    /// drive_gb_per_s_busy, mb_read_per_token (MiB), released_topk_in_memory, sync_reads, reads_per_token (records), ...
    public let split: [String: Double]
    /// The prompt prefill that precedes the first token (stats reset before it): prefill_tokens, prefill_s,
    /// prefill_mb_read (1e6 B), prefill_reads (records), prefill_sync_reads, prefill_expert_wait_ms (total),
    /// prefill_gpu_ms (total), prefill_drive_gb_per_s, prefill_drive_gb_per_s_busy, template_ms, and first_token_ms (the
    /// first generated token: sample + step, the engine's own clock), ttft_ms (template + prefix restore + prefill + prefix
    /// save + first token, the engine's clock). With the prefix cache on (PrefixCache.swift): prefix_hit (0/1),
    /// prefix_tokens_restored, prefix_state_mb, prefix_load_ms, prefix_reland_ms, prefix_relanded / prefix_kept /
    /// prefix_moved (records), prefix_reland_mb (read from the store), prefix_held, prefix_restore_ms (all of it),
    /// prefix_saved (states written), prefix_save_ms, prefix_save_mb, prefix_cache_entries, prefix_cache_mb.
    public let prefillStats: [String: Double]
    /// The sampling this reply used (greedy: SamplingParams.greedy).
    public let sampling: SamplingParams
    /// The thinking segment by token id: the prompt ended inside a think block (Qwen3.6's generation prompt with thinking
    /// on), tokens inside it, whether </think> came, whether the budget forced it, and the two texts decoded apart.
    public let startedInThinking: Bool
    public let thinkingTokens: Int
    public let thinkingClosed: Bool
    public let thinkingForced: Bool
    public let thinkingText: String
    public let answerText: String
}

public enum GenerationEvent: Sendable {
    /// The prompt prefill's progress (before the first token).
    case prefill(PrefillEvent)
    case token(TokenEvent)
    case done(GenerationResult)
}

/// A Bool behind a lock (static state read from the app's main thread).
final class LockedFlag: @unchecked Sendable {
    private let l = NSLock()
    private var v = false
    func withLock<R>(_ f: (inout Bool) -> R) -> R { l.lock(); defer { l.unlock() }; return f(&v) }
}

/// A thread that runs submitted closures in order (MLX work stays on one thread).
final class Worker: @unchecked Sendable {
    let mon = Monitor()
    var jobs: [() -> Void] = []

    init(name: String) {
        startThread(name: name) { [unowned self] in
            while true {
                mon.lock()
                while jobs.isEmpty { mon.wait() }
                let j = jobs.removeFirst()
                mon.unlock()
                j()
            }
        }
    }

    func async(_ f: @escaping () -> Void) { mon.withLock { jobs.append(f); mon.broadcast() } }

    func run<T>(_ f: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<T, Error>) in
            self.async {
                do { c.resume(returning: try f()) } catch { c.resume(throwing: error) }
            }
        }
    }
}

public final class PhoneStoreEngine: @unchecked Sendable {
    /// The chunk setting as the settings sheet shows it: "128 positions" or the auto rule with its budget and what it
    /// gives for an empty context (Engine.autoChunk; the chunk shrinks as the context grows: the attention scores grow with the keys).
    public static func chunkSummary(_ o: EngineOptions) -> String {
        guard o.prefillAutoBytes > 0 else { return "\(o.prefillChunk) positions" }
        let m = PrefillMemory.qwen36(readAhead: o.prefillReadAhead != 0 && o.mode == .prefetch)
        let c0 = m.largestChunk(budget: o.prefillAutoBytes, offset: 0, limit: 1 << 20)
        let c4 = m.largestChunk(budget: o.prefillAutoBytes, offset: 4096, limit: 1 << 20)
        return String(format: "auto: the largest whose activations fit %.0f MB (up to %d positions on an empty context, %d after 4096 tokens)",
                      o.prefillAutoBytes / 1e6, c0, c4)
    }

    public let options: EngineOptions
    /// Every setting in force, for the app's settings sheet and probe (the CLI's startup line).
    public let settings: [(String, String)]
    public let metallib: String
    let worker: Worker
    let engine: Engine
    let tokenizer: Tokenizer
    let stopIds: Set<Int>
    var fed: [Int] = []  // tokens whose state is in the caches (worker thread only)
    var logits: MLXArray? = nil
    let lock = NSLock()
    var cancelled = false
    var profDone = false  // GPU profiler (env PHONE_GPU_PROF=tags|regions): the first reply only
    /// The prefix cache (PrefixCache.swift): full-state snapshots at message boundaries (the end of each prompt, and the end
    /// of a leading system message), restored when a new conversation's ids start with a cached prefix. nil: none.
    public private(set) var prefixCache: PrefixCacheDir? = nil
    private var prefixOn = false
    private var saveAtHeader = false
    private var saveAtEnd = false
    private var endHeader = true
    private var saveAtPromptEnd = true
    var prefixLog: (@Sendable (String) -> Void)? = nil
    /// The load's kernel warm-up (Engine.warmBatchedKernels: dry batched chunks, state discarded), or nil when it was off
    /// (EngineOptions.kernelWarmupOn: the batched prefill off, or -kernelWarmup 0) or skipped (kernelWarmupSkipped).
    public let kernelWarmup: KernelWarmup?
    /// The warm-up was not needed: this install's kernels were compiled at an earlier launch (kernelWarmupMarker).
    public let kernelWarmupSkipped: Bool

    /// The mlx.metallib this process uses (built by native/metallib.sh with -fno-fast-math, bit-identical to the Python
    /// runner), or nil: then MLX falls back to the bundle's default.metallib (Xcode's fast-math build). Searched next to
    /// the executable (where MLX itself looks: the CLI, the iOS .app root), in Resources/ beside it, and in the app
    /// bundle's resources (a Mac Catalyst app: Contents/Resources, where codesign wants non-code files).
    public static func metallibPath() -> String? {
        let exe = (Bundle.main.executablePath.map { URL(fileURLWithPath: $0) } ?? URL(fileURLWithPath: CommandLine.arguments[0]))
            .resolvingSymlinksInPath().deletingLastPathComponent()
        var cands = [exe.appendingPathComponent("mlx.metallib"), exe.appendingPathComponent("Resources/mlx.metallib")]
        if let r = Bundle.main.resourceURL { cands.append(r.appendingPathComponent("mlx.metallib")) }
        return cands.map(\.path).first { FileManager.default.fileExists(atPath: $0) }
    }

    public static func metallibInUse() -> String {
        guard let p = metallibPath() else { return "bundle default.metallib (FAST MATH: not bit-identical to mlx_run.py)" }
        return "mlx.metallib (-fno-fast-math): \(p)"
    }

    /// Point MLX at mlx.metallib explicitly (before the first MLX op), so a bundle-resources copy is used too.
    static func useMetallib() {
        if let p = metallibPath() { _ = mlx_metal_set_metallib_path(p) }
        mlxStarted.withLock { $0 = true }
    }

    /// Set once load() has pointed MLX at its kernels: reading MLX's allocator before that would create the Metal device
    /// (and load the bundle's fast-math default.metallib) too early, so mlxMemory() answers nil until then.
    static let mlxStarted = LockedFlag()

    init(options: EngineOptions, worker: Worker, engine: Engine, tokenizer: Tokenizer, stopIds: Set<Int>, metallib: String,
         kernelWarmup: KernelWarmup?, kernelWarmupSkipped: Bool) {
        self.options = options; self.worker = worker; self.engine = engine; self.tokenizer = tokenizer; self.stopIds = stopIds
        self.metallib = metallib
        self.kernelWarmup = kernelWarmup
        self.kernelWarmupSkipped = kernelWarmupSkipped
        let o = options
        let planGB = engine.plan.mapValues { Double($0) / 1e9 }
        settings = [
            ("mode", o.mode.rawValue),
            ("layout", engine.layout.rawValue),
            ("embeddings", "bf16 rows read per token from \(engine.embeddingsPath)"),
            ("routers", engine.routers == nil ? "none" : "lag-\(engine.routers!.lag), \(o.m > 0 ? o.m : engine.res.cfg.topK) named, int\(o.routerBits) matrices (\(engine.routerSource))"),
            ("keep", "\(o.keep)"),
            ("evict", o.evict.rawValue),
            ("drop", o.noDrop ? "no drop" : "drop after keep"),
            ("resident", "int\(o.residentBits), " + String(format: "%.2f GB", Double(engine.res.nbytes) / 1e9) + " (\(engine.residentSource))"),
            ("cap (slots / layer)", engine.capsUniform ? "\(engine.cap)" : "per layer (\(o.layerCapsSpec)), mean \(engine.cap)"),
            ("slots per layer", engine.caps.map(String.init).joined(separator: " ")),
            ("slabs", String(format: "%d slots (%@) x %.2f MB = %.2f GB", engine.caps.reduce(0, +), engine.capsUniform ? "\(engine.store.NL) x \(engine.cap)" : "sum over layers",
                             Double(engine.store.REC) / 1e6, planGB["slabs"] ?? 0)),
            ("memory plan", String(format: "%.2f GB (resident %.2f + routers %.2f + slabs %.2f + codec staging %.2f)", planGB.values.reduce(0, +), planGB["resident"] ?? 0,
                                   planGB["routers"] ?? 0, planGB["slabs"] ?? 0, planGB["codec_staging"] ?? 0)),
            ("store codec", PhoneStoreEngine.codecSummary(engine)),
            ("readers", "\(o.readers), \(o.readChunks) chunks / record"),
            ("page cache", o.noCache ? "F_NOCACHE + no readahead" : "ON"),
            ("read overlap", "late wait \(o.lateWait ? "on" : "off"), head start \(o.headStart ? "on" : "off")"),
            ("sampling", "per reply (default greedy; the app: generation_config.json unless gen.greedy)"),
            ("prefill", o.prefillChunk > 1 ? "batched, chunks of \(PhoneStoreEngine.chunkSummary(o)), \(engine.store.readerThreads) readers in a chunk" : "token by token"),
            ("prefill read-ahead", o.prefillChunk <= 1 || engine.routers == nil || o.prefillReadAhead == 0 ? "off"
                : "lag-4 names of later layers, up to \(o.prefillReadAhead > 0 ? "\(o.prefillReadAhead)" : "the cap") records per layer into \(["empty slots", "empty slots + stale unnamed records", "empty slots + stale and held unnamed records"][min(2, max(0, o.readAheadOverwrite))]), low priority (h_prev \(o.readAheadShift ? "shift" : "stale"))"),
            ("prefix re-land", o.relandLazy ? "lazy: the next chunk reads what it uses or keeps" : "eager: every held record landed at the restore"),
            ("kernel warm-up", kernelWarmup.map { String(format: "batched prefill kernels compiled at load in %.2f s (dry chunks %@, state discarded)", $0.seconds, $0.chunks.map(String.init).joined(separator: "/")) }
                ?? (kernelWarmupSkipped ? "skipped: compiled at an earlier launch of this install (the OS shader cache has them)"
                    : "off (\(o.kernelWarmup == false ? "-kernelWarmup 0" : "token-by-token prefill"))")),
            ("MLX cache limit", "\(o.cacheLimitMB) MB, wired limit \(o.wire ? "on" : "off")"),
            ("kernels", metallib),
            ("MLX core", "0.32.3"),
            ("native cache", o.cacheDir),
            ("model dir", o.dir),
        ]
    }

    /// Load the tokenizer, then (on the engine thread) the resident model, routers and slabs, and start the readers.
    public static func load(options: EngineOptions, log: @escaping @Sendable (String) -> Void) async throws -> PhoneStoreEngine {
        let t0 = now()
        log("tokenizer + chat template from \(URL(fileURLWithPath: options.dir).lastPathComponent)/")
        let tk = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: options.dir))
        var stop = Set<Int>()
        if let d = try? Data(contentsOf: URL(fileURLWithPath: options.dir + "/generation_config.json")),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            if let a = j["eos_token_id"] as? [NSNumber] { a.forEach { stop.insert($0.intValue) } }
            if let a = j["eos_token_id"] as? NSNumber { stop.insert(a.intValue) }
        }
        if let e = tk.eosTokenId { stop.insert(e) }
        useMetallib()
        let ml = metallibInUse()
        log("kernels: \(ml)")
        let worker = Worker(name: "phonestore-engine")
        let eng = try await worker.run { try Engine(opt: options, log: log) }
        // the batched prefill's Metal pipelines are built now, not inside the first message's prefill (Prefill.swift
        // warmBatchedKernels: dry chunks on a throwaway state; the store, the engine's state and the prefix cache untouched)
        var warm: KernelWarmup? = nil
        var skipped = false
        let marker = kernelWarmupMarker(options)
        if options.kernelWarmupOn {
            let have = marker.flatMap { try? String(contentsOfFile: $0.path, encoding: .utf8) }
            log("warm-up marker: " + (marker.map { "\($0.path) (" + (have == nil ? "absent" : (have == $0.key ? "matches this install" : "another key: \(have!)")) + ")" } ?? "none (no cache folder)"))
        }
        if options.kernelWarmupOn && options.kernelWarmup == nil, let m = marker,
           (try? String(contentsOfFile: m.path, encoding: .utf8)) == m.key {
            log("warm-up: skipped, this install's batched kernels were compiled at an earlier launch (\((m.path as NSString).lastPathComponent); -kernelWarmup 1 forces it)")
            skipped = true
        } else if options.kernelWarmupOn {
            log("warm-up: compiling the batched prefill kernels")
            let samp = SamplingParams.generationConfig(dir: options.dir)  // the app's default reply sampler: its graph too
            let w = try await worker.run { eng.warmBatchedKernels(sampling: samp) }
            log(w.line)
            warm = w
            if let m = marker { try? m.key.write(toFile: m.path, atomically: true, encoding: .utf8) }
        }
        let s = PhoneStoreEngine(options: options, worker: worker, engine: eng, tokenizer: tk, stopIds: stop, metallib: ml,
                                 kernelWarmup: warm, kernelWarmupSkipped: skipped)
        log(String(format: "ready in %.1fs: %@, cap %d, %.2f GB planned, layout %@", now() - t0, options.mode.rawValue, eng.cap,
                   Double(eng.plan.values.reduce(0, +)) / 1e9, eng.layout.rawValue))
        log("layout \(eng.layout.rawValue): embeddings \(eng.embeddingsPath); resident \(eng.residentSource); routers \(eng.routerSource)")
        return s
    }

    public var cap: Int { engine.cap }
    public var caps: [Int] { engine.caps }

    /// The kernel warm-up runs once per install: the Metal pipelines it builds are kept by the OS shader cache, so later
    /// launches would only pay its dry chunks' GPU time (Mac: 0.8 s). The marker beside that cache (the user cache dir's
    /// <bundle id>/ folder, where Metal keeps the app's com.apple.metal/; removed with it) holds a key of this install:
    /// the bundle's path (a new container UUID at every iOS install), the executable's size and modification time, the
    /// metallib in use and the chunk size. A different key, or no marker, runs the warm-up. nil: no cache folder.
    static func kernelWarmupMarker(_ o: EngineOptions) -> (path: String, key: String)? {
        var buf = [CChar](repeating: 0, count: Int(PATH_MAX))
        let bid = Bundle.main.bundleIdentifier ?? "phone-store"
        var dirs: [String] = []
        if confstr(_CS_DARWIN_USER_CACHE_DIR, &buf, buf.count) > 0 { dirs.append(String(cString: buf) + "/" + bid) }
        // fallback (2026-11-07): the app's own Library/Caches, should the user cache dir not be writable
        if let c = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first { dirs.append(c.path + "/" + bid) }
        guard let dir = dirs.first(where: { (try? FileManager.default.createDirectory(atPath: $0, withIntermediateDirectories: true)) != nil }) else { return nil }
        let exe = Bundle.main.executablePath ?? CommandLine.arguments[0]
        func stamp(_ p: String?) -> String {
            guard let p, let a = try? FileManager.default.attributesOfItem(atPath: p) else { return "-" }
            return "\((a[.size] as? Int) ?? 0)@\((a[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)"
        }
        let key = "kernel warm-up 2 (+ decode step \(o.decodeWarmup)) | \(Bundle.main.bundlePath) | exe \(stamp(exe)) | metallib \(stamp(metallibPath())) | chunk \(o.prefillChunk)"
        return (dir + "/phone-store-kernel-warmup.txt", key)
    }

    /// --store-codec and which layers are read compressed ("psc", decoded on the GPU into the slot) or raw ("bin").
    public static func codecSummary(_ engine: Engine) -> String {
        let p = engine.store.layerPath
        let c = p.indices.filter { p[$0] == "psc" }
        if engine.opt.storeCodec == .off { return "off (all layers raw .bin)" }
        if c.isEmpty { return "auto: no .psc layers (all raw .bin)" }
        return "auto: \(c.count) layers compressed (\(c.map(String.init).joined(separator: ","))), \(p.count - c.count) raw; staging \(engine.opt.codecStaging) x " + String(format: "%.2f MB", Double(engine.store.codecStagingBytes) / Double(max(1, engine.opt.codecStaging)) / 1e6)
    }
    public var layerPaths: [String] { engine.store.layerPath }

    /// MLX's allocator (bytes): active (live arrays), peak (high-water mark since launch) and cache (freed buffers kept
    /// for reuse, bounded by the cache limit). Plain counter reads; safe from any thread. nil before the first load()
    /// (see mlxStarted).
    public static func mlxMemory() -> (active: Int, peak: Int, cache: Int)? {
        guard mlxStarted.withLock({ $0 }) else { return nil }
        return (Memory.activeMemory, Memory.peakMemory, Memory.cacheMemory)
    }

    /// phone-store-run's per-token split (its bench JSON keys, definitions and rounding), over `n` tokens and `dt` s.
    /// GPU profiler calibration (tags 901-903): one tiny kernel per command buffer, then chains of 16 dependent tiny
    /// kernels and of 16 dependent 2048-wide fp16 kernels in one buffer each: the A16's per-buffer and per-kernel costs.
    static func gpuCalibrate(_ engine: Engine) {
        mlx_phone_prof_mark(5, 0)
        let a = MLXArray([Float(1)])
        eval(a)
        for i in 0..<100 { mlx_phone_prof_tag(901); let b = a + Float(i); eval(b) }
        for i in 0..<50 {
            mlx_phone_prof_tag(902)
            var x = a + Float(i)
            for _ in 0..<15 { x = x + Float(1) }
            eval(x)
        }
        let v = MLXArray.ones([2048], dtype: .float16)
        eval(v)
        for i in 0..<50 {
            mlx_phone_prof_tag(903)
            var x = v + MLXArray(Float16(Float(i)))
            for _ in 0..<15 { x = x * MLXArray(Float16(1.0001)) }
            eval(x)
        }
        // 904-907: the A16's read bandwidth and MLX's batch-1 matvec kernels at the model's widths, on fresh weights
        // (each call streams more than the system cache): 904 sum over 128 MB fp16; 905 int4 qmv 32768x2048 (37.7 MB
        // with scales and biases); 906 int8 qmv 16384x2048 (36 MB); 907 int4 gather_qmv, 8 of 64 experts of 512x2048
        // per call (4.7 MB per call, the routed gate/up shape; 35 MB of experts in all).
        do {
            let big = MLXArray.ones([64 << 20], dtype: .float16)
            eval(big)
            for _ in 0..<10 { mlx_phone_prof_tag(904); eval(big.sum()) }
        }
        do {
            let x = MLXArray.ones([1, 2048], dtype: .float16)
            let w4 = MLX.randInt(0 ..< Int32.max, [32768, 256]).asType(.uint32)
            let s4 = MLXArray.ones([32768, 32], dtype: .float16) * Float16(0.01)
            let b4 = MLXArray.zeros([32768, 32], dtype: .float16)
            eval(x, w4, s4, b4)
            for _ in 0..<10 { mlx_phone_prof_tag(905); eval(quantizedMatmul(x, w4, scales: s4, biases: b4, transpose: true, groupSize: 64, bits: 4)) }
        }
        do {
            let x = MLXArray.ones([1, 2048], dtype: .float16)
            let w8 = MLX.randInt(0 ..< Int32.max, [16384, 512]).asType(.uint32)
            let s8 = MLXArray.ones([16384, 32], dtype: .float16) * Float16(0.01)
            let b8 = MLXArray.zeros([16384, 32], dtype: .float16)
            eval(x, w8, s8, b8)
            for _ in 0..<10 { mlx_phone_prof_tag(906); eval(quantizedMatmul(x, w8, scales: s8, biases: b8, transpose: true, groupSize: 64, bits: 8)) }
        }
        do {
            let xh = MLXArray.ones([1, 1, 1, 2048], dtype: .float16)
            let gw = MLX.randInt(0 ..< Int32.max, [64, 512, 256]).asType(.uint32)
            let gs = MLXArray.ones([64, 512, 32], dtype: .float16) * Float16(0.01)
            let gb = MLXArray.zeros([64, 512, 32], dtype: .float16)
            let lhs = MLXArray.zeros([1, 8], dtype: .uint32)
            eval(xh, gw, gs, gb, lhs)
            for i in 0..<16 {
                let idx = MLXArray((0..<8).map { UInt32((i * 8 + $0) % 64) }, [1, 8])
                eval(idx)
                mlx_phone_prof_tag(907)
                eval(gatherQuantizedMM(xh, gw, scales: gs, biases: gb, lhsIndices: lhs, rhsIndices: idx, transpose: true, groupSize: 64, bits: 4))
            }
        }
        // 910-913: the engine's own weights, one kernel per buffer: 910 lm_head (286 MB), 911 layer 0's GDN in_proj_qkv
        // (8.4 MB + scales), 912 a routed gate matrix of layer 0's slab, slots 0-7 (4.7 MB), 913 layer 4's lag router
        // lin (int8). 914: lm_head 4 times in one buffer (the GPU kept busy).
        do {
            func lin(_ l: Lin, _ tag: Int32, _ n: Int) {
                let x = MLXArray.ones([1, 1, l.w.dim(1) * 32 / l.bits], dtype: .float16)
                eval(x)
                for _ in 0..<n { mlx_phone_prof_tag(tag); eval(l(x)) }
            }
            lin(engine.res.lmHead, 910, 6)
            if let g = engine.res.layers[0].gdn { lin(g.inQKV, 911, 10) }
            if engine.store.W.count > 0 {
                let xh = MLXArray.ones([1, 1, 1, engine.res.cfg.hidden], dtype: .float16)
                let lhs = MLXArray.zeros([1, 8], dtype: .uint32)
                let idx = MLXArray((0..<8).map { UInt32($0 % engine.caps[0]) }, [1, 8])
                eval(xh, lhs, idx)
                for _ in 0..<10 {
                    mlx_phone_prof_tag(912)
                    eval(gatherQuantizedMM(xh, engine.store.W[0][0], scales: engine.store.S[0][0], biases: engine.store.Bi[0][0],
                                           lhsIndices: lhs, rhsIndices: idx, transpose: true, groupSize: G, bits: 4))
                }
            }
            if let r = engine.routers, r.layers.count > 4 { lin(r.layers[4].lin, 913, 10) }
            let l = engine.res.lmHead
            let x = MLXArray.ones([1, 1, l.w.dim(1) * 32 / l.bits], dtype: .float16)
            eval(x)
            for _ in 0..<3 {
                mlx_phone_prof_tag(914)
                eval((0..<4).map { i in l(x + Float16(Float(i))) })
            }
        }
        mlx_phone_prof_tag(0)
        mlx_phone_prof_mark(6, 0)
    }

    static func split(_ engine: Engine, tokens: Int, dt: Double, g0: (busy: Double, buffers: Int, kernels: Int),
                      g1: (busy: Double, buffers: Int, kernels: Int)) -> [String: Double] {
        func r(_ x: Double, _ d: Int = 1) -> Double { (x * pow(10, Double(d))).rounded() / pow(10, Double(d)) }
        let n = Double(max(1, tokens))
        let st = engine.store.snapshot { s in
            (bytes: s.bytes, reads: s.reads, sync: s.syncReads, evicted: s.evicted, early: s.evictedEarly, readSec: s.readSec,
             queueSec: s.queueSec, policySec: s.policySec, span: s.lastReadEnd - s.firstReadStart, chunkReads: s.chunkReads,
             recLat: s.recordLatSec, busy: s.driveBusySec)
        }
        let pol = engine.store.policySeconds
        let tm = engine.tm
        return [
            "tokens": Double(tokens), "seconds": r(dt, 3), "slab": Double(engine.cap), "readers": Double(engine.opt.readers),
            "read_chunks": Double(engine.opt.readChunks),
            "tok_per_s": r(Double(tokens) / dt, 3), "ms_per_token": r(1000 * dt / n),
            "host_encode_ms": r(1000 * tm.encode / n), "host_submit_ms": r(1000 * tm.submit / n),
            "host_graph_ms": r(1000 * (tm.encode - tm.submit) / n), "async_evals_per_token": r(Double(tm.asyncEvals) / n, 1),
            "host_ms": r(1000 * (dt - tm.syncWait - tm.readWait) / n),
            "expert_wait_ms": r(1000 * tm.readWait / n), "name_wait_ms": r(1000 * tm.nameWait / n),
            "land_wait_ms": r(1000 * (tm.readWait - tm.nameWait) / n),
            "sync_wait_ms": r(1000 * tm.syncWait / n), "syncs_per_token": r(Double(tm.syncs) / n, 2),
            "gpu_ms_per_token": r(1000 * (g1.busy - g0.busy) / n), "gpu_busy_frac": r((g1.busy - g0.busy) / dt, 3),
            "gpu_command_buffers_per_token": r(Double(g1.buffers - g0.buffers) / n, 1),
            "gpu_kernels_per_token": r(Double(g1.kernels - g0.kernels) / n, 1),
            "drive_gb_per_s": r(Double(st.bytes) / dt / 1e9, 3), "drive_busy_frac": r(st.busy / dt, 3),
            "drive_gb_per_s_busy": r(st.busy > 0 ? Double(st.bytes) / st.busy / 1e9 : 0, 3),
            "mb_read_per_token": r(Double(st.bytes) / n / 1048576.0), "kb_read_per_token": r(Double(st.bytes) / n / 1e3),
            "reads_per_token": r(Double(st.reads) / n, 2), "chunk_reads_per_token": r(Double(st.chunkReads) / n, 1),
            "chunk_read_ms_mean": r(1000 * st.readSec / Double(max(1, st.chunkReads)), 2),
            "record_latency_ms_mean": r(1000 * st.recLat / Double(max(1, st.reads)), 2),
            "reader_read_ms_mean": r(1000 * st.readSec / Double(max(1, st.reads)), 2),
            "reader_queue_ms_mean": r(1000 * st.queueSec / Double(max(1, st.reads)), 2),
            "reader_busy_ms_per_token": r(1000 * st.readSec / n), "reads_in_flight_mean": r(st.span > 0 ? st.readSec / dt : 0, 2),
            "sync_reads": Double(st.sync), "sync_reads_per_token": r(Double(st.sync) / n, 2),
            "released_topk_in_memory": r(engine.inMemoryRate, 4),
            "held_experts": r(Double(engine.heldSum) / Double(max(1, engine.heldN))),
            "evicted_per_token": r(Double(st.evicted) / n, 2), "evicted_in_window_per_token": r(Double(st.early) / n),
            "policy_ms": r(1000 * st.policySec / n, 3), "evict_host_ms": r(1000 * pol.critical / n, 3),
            "evict_prep_ms": r(1000 * pol.prepare / n, 3), "namer_gpu_wait_ms": r(1000 * engine.namerGPUWait / n),
            "peak_gb": r(Double(Memory.peakMemory) / 1073741824.0, 2), "active_gb": r(Double(Memory.activeMemory) / 1073741824.0, 2),
        ]
    }
    /// "full" | "slim" (ModelLayout): where the embeddings, resident model and routers were read from.
    public var layout: String { engine.layout.rawValue }

    /// The prompt ids the chat template gives for `messages` (enable_thinking = thinking).
    public func promptIds(messages: [[String: String]], thinking: Bool) throws -> [Int] {
        try ChatPrompt.ids(tokenizer, messages: messages, thinking: thinking)
    }

    /// The release's generation_config.json sampling (temperature 1.0, top_k 20, top_p 0.95 for Qwen3.6-35B-A3B).
    public var generationDefaults: SamplingParams { SamplingParams.generationConfig(dir: options.dir) }

    /// Use `dir` as the prefix cache (created if absent), LRU-capped at `capGB` (0: never delete). Call before the first
    /// generate; `prefixCacheEnabled` switches it on and off at any time (default on once set).
    public func setPrefixCache(dir: String, capGB: Double, log: (@Sendable (String) -> Void)? = nil) throws {
        let c = try PrefixCacheDir(dir: dir, capGB: capGB)
        lock.withLock { prefixCache = c; prefixOn = true; prefixLog = log }
    }

    public var prefixCacheEnabled: Bool {
        get { lock.withLock { prefixOn && prefixCache != nil } }
        set { lock.withLock { prefixOn = newValue } }
    }

    /// Also save the state at the assistant header of a prompt that ends inside an open think block (thinking on): the ids
    /// up to "<|im_start|>assistant\n", before the generation prompt's "<think>\n". Qwen's template later renders this
    /// turn as "<|im_start|>assistant\n" + answer (the reasoning stripped), so that state is a prefix of the next prompt
    /// and a follow-up re-prefills only the answer and the new turn instead of the whole conversation. Off by default (the
    /// CLI and the app's test flows save as before); the app turns it on in its "live" history mode (2026-11-04).
    public var prefixSaveAtAssistantHeader: Bool {
        get { lock.withLock { saveAtHeader } }
        set { lock.withLock { saveAtHeader = newValue } }
    }

    /// Also save the state at the end of a reply (2026-11-04): the prompt, the generated tokens and the template's end of
    /// the turn ("<|im_end|>\n", and since 2026-11-07 the next turn's "<|im_start|>user\n": prefixReplyEndHeader; fed
    /// after the reply as one batched chunk), when the next prompt renders the reply as generated, i.e. the app's
    /// "live" history (a thinking-off reply given back after the generation prompt's empty think block, with
    /// reasoning_content ""; checked by rendering the conversation with the reply and a placeholder turn, so a reply the
    /// template would rewrite, e.g. one ending in whitespace, is skipped). Continuing that conversation after a relaunch
    /// or a switch then restores this state and reads only the new turn instead of re-reading the last answer. It runs on
    /// the engine thread after the reply's stream has finished (the UI is not kept waiting; a new generate queues behind
    /// it), and the live state then includes the end of the turn too. Off by default (the CLI and the test flows); the app
    /// turns it on in its "live" history mode. Needs the prefix cache.
    public var prefixSaveAtReplyEnd: Bool {
        get { lock.withLock { saveAtEnd } }
        set { lock.withLock { saveAtEnd = newValue } }
    }

    /// The reply-end save also feeds the next turn's header "<|im_start|>user\n" (2026-11-07; on by default): a follow-up
    /// then prefills its words and the generation prompt only. Off: the end of the turn alone, as before.
    public var prefixReplyEndHeader: Bool {
        get { lock.withLock { endHeader } }
        set { lock.withLock { endHeader = newValue } }
    }

    /// Save the state at the end of each prompt (default on). Off (tests: `-noPromptEndSave`): the same message can be sent
    /// again in a fresh chat and is prefilled again from the system state instead of restored whole.
    public var prefixSaveAtPromptEnd: Bool {
        get { lock.withLock { saveAtPromptEnd } }
        set { lock.withLock { saveAtPromptEnd = newValue } }
    }

    /// The end of the last "<|im_start|>assistant\n" in `ids` when the prompt continues past it (thinking on), else nil.
    func assistantHeader(ids: [Int], thinking: Bool) -> Int? {
        guard thinking else { return nil }
        let hdr = tokenizer.encode(text: "<|im_start|>assistant\n", addSpecialTokens: false)
        guard let first = hdr.first, let i = ids.lastIndex(of: first), i + hdr.count < ids.count,
              Array(ids[i..<(i + hdr.count)]) == hdr else { return nil }
        return i + hdr.count
    }

    /// Delete every cached state (on the engine thread, after the reply in progress). `keepKept`: all but the app's kept
    /// system-prompt states (the test flows: they need no state of theirs cached, and leave the app's setup alone).
    public func clearPrefixCache(keepKept: Bool = false) async throws {
        try await worker.run { [self] in prefixCache?.clear(keepKept: keepKept) }
    }

    /// The first `n` ids of `ids` that are a leading system message and the next message's header (the point where the
    /// ids stop matching a render with the next message's content changed, as mlx_run.py finds a head the template
    /// refuses to render alone), or nil when the conversation does not start with a system message.
    func systemBoundary(messages: [[String: String]], thinking: Bool, ids: [Int]) -> Int? {
        guard messages.count >= 2, messages[0]["role"] == "system" else { return nil }
        guard let a = try? promptIds(messages: [messages[0], ["role": "user", "content": "a"]], thinking: thinking),
              let b = try? promptIds(messages: [messages[0], ["role": "user", "content": "b"]], thinking: thinking) else { return nil }
        var n = 0
        while n < min(a.count, b.count, ids.count) && a[n] == b[n] && a[n] == ids[n] { n += 1 }
        return n > 0 && n < ids.count ? n : nil
    }

    /// Returns when the engine thread has finished everything queued so far (a reply-end save included).
    public func waitIdle() async {
        _ = try? await worker.run { () }
    }

    /// Stop the reply in progress after the token in flight. Safe from any thread, and when idle.
    public func cancel() { lock.withLock { cancelled = true } }
    var isCancelled: Bool { lock.withLock { cancelled } }

    /// Drop the conversation: model state and held experts (the next generate starts cold).
    public func forget() {
        worker.async { [self] in
            engine.forget()
            fed = []
            logits = nil
        }
    }

    /// Teacher forcing (TeacherForce.swift) on the engine thread: a fresh session fed `promptIds` + `ids` one token at a
    /// time through the normal pipeline; per-position argmax, margin, top-K and (with `ref`) the max-abs difference
    /// against another run's top-K and its released experts. cancel() stops it between positions. The conversation is forgotten afterwards.
    public func teacherForce(promptIds: [Int], ids: [Int], k: Int = 32, ref: TFReference?,
                             progress: @escaping @Sendable (TFPosition) -> Void) async throws -> TFRun {
        lock.withLock { cancelled = false }
        return try await worker.run { [self] in
            try ensureSlabs()
            let r = PhoneStore.teacherForce(engine: engine, promptIds: promptIds, ids: ids, k: k, ref: ref,
                                            isCancelled: { self.isCancelled }, progress: progress)
            engine.forget()
            fed = []
            logits = nil
            return r
        }
    }

    /// One reply. `sampling` defaults to greedy (the identity-tested path); `thinkingBudget`: tokens inside the think block
    /// after which the template's close is forced (nil: none).
    public func generate(messages: [[String: String]], thinking: Bool, maxNew: Int, sampling: SamplingParams = .greedy,
                         thinkingBudget: Int? = nil) -> AsyncThrowingStream<GenerationEvent, Error> {
        lock.withLock { cancelled = false }
        return AsyncThrowingStream { cont in
            worker.async { [self] in
                do {
                    let r = try run(messages: messages, thinking: thinking, maxNew: maxNew, sampling: sampling,
                                    thinkingBudget: thinkingBudget, prefillProgress: { cont.yield(.prefill($0)) }) { cont.yield(.token($0)) }
                    cont.yield(.done(r))
                    cont.finish()
                    saveReplyEnd(messages: messages, thinking: thinking, result: r)
                } catch {
                    cont.finish(throwing: error)
                }
            }
            cont.onTermination = { [weak self] t in
                if case .cancelled = t { self?.cancel() }
            }
        }
    }

    // MARK: one-time setup, launch restore and background release (2026-11-07, docs/G4-runtime.md "Launch: once per install")

    /// What prepareSystemState did.
    public struct SystemState: Sendable {
        public let tokens: Int         // the state's ids: the system message and the next user header
        public let path: String        // its file in the prefix cache
        public let how: String         // "live" (already the engine's state), "restored" (from the cache), "prefilled" (cold, saved)
        public let seconds: Double     // all of it
        public let prefillSeconds: Double
        public let mbRead: Double      // expert bytes read (prefill or landing), 1e6 B
        public let landed: Int         // records read to land the held set (restore with land)
        public let landSeconds: Double
        public let savedMB: Double
        public var line: String {
            String(format: "%@ %d tokens in %.2f s (prefill %.2f s, %.0f MB read; landed %d records in %.2f s%@) -> %@", how, tokens, seconds,
                   prefillSeconds, mbRead, landed, landSeconds, savedMB > 0 ? String(format: "; saved %.1f MB", savedMB) : "",
                   (path as NSString).lastPathComponent)
        }
    }

    /// The ids of the system-prompt state for `system`: the system message and the next user turn's header, the save point
    /// generate() uses (systemBoundary; the same ids with thinking on and off: the flag changes only the generation prompt).
    public func systemStateIds(system: String) throws -> [Int] {
        let sm = ["role": "system", "content": system]
        let a = try promptIds(messages: [sm, ["role": "user", "content": "a"]], thinking: false)
        let b = try promptIds(messages: [sm, ["role": "user", "content": "b"]], thinking: false)
        var n = 0
        while n < min(a.count, b.count) && a[n] == b[n] { n += 1 }
        guard n > 0 else { throw PSError("system state: no common prefix") }
        return Array(a[..<n])
    }

    /// The system-prompt state's file in the prefix cache (nil without a cache).
    public func systemStatePath(system: String) throws -> String? {
        guard let pc = lock.withLock({ prefixCache }) else { return nil }
        return pc.path(for: try systemStateIds(system: system))
    }

    /// Make the system-prompt state of `system` the engine's live state, so the first message of a chat with it continues
    /// it ("live prefix": only the user turn is prefilled): restored from the prefix cache when there (with `land`, every
    /// held record is landed now, in idle time, instead of by the first message's chunk), else prefilled cold and saved.
    /// The one-time setup (prefilled) and every later launch and new chat (restored) call this; nothing else is fed.
    /// `progress(done, total)`: the cold prefill's tokens.
    public func prepareSystemState(system: String, land: Bool, progress: @escaping @Sendable (Int, Int) -> Void = { _, _ in })
        async throws -> SystemState {
        let ids = try systemStateIds(system: system)
        return try await worker.run { [self] in
            let t0 = now()
            guard let pc = lock.withLock({ prefixCache }) else { throw PSError("system state: no prefix cache") }
            try ensureSlabs()
            let path = pc.path(for: ids)
            let b0 = engine.store.snapshot { $0.bytes }
            func landNow() -> (Int, Double) {
                guard land, engine.store.hasPending else { return (0, 0) }
                let tl = now()
                let r = engine.store.landPending()
                return (r.relanded, now() - tl)
            }
            if let lg = logits, fed == ids {
                var mb = 0.0
                if FileManager.default.fileExists(atPath: path) {
                    pc.touch(path)
                } else {  // live but not on disk (the cache was cleared): saved again
                    mb = Double(try engine.savePrefix(path: path, ids: fed, logits: lg).bytes) / 1e6
                    pc.enforceCap()
                }
                let (n, ls) = landNow()
                return SystemState(tokens: ids.count, path: path, how: "live", seconds: now() - t0, prefillSeconds: 0,
                                   mbRead: Double(engine.store.snapshot { $0.bytes } - b0) / 1e6, landed: n, landSeconds: ls, savedMB: mb)
            }
            engine.forget()
            fed = []
            logits = nil
            if FileManager.default.fileExists(atPath: path) {
                do {
                    let r = try engine.loadPrefix(path: path, ids: ids, reland: true, lazy: true)
                    fed = r.ids
                    logits = r.logits
                    let (n, ls) = landNow()
                    pc.touch(path)
                    return SystemState(tokens: ids.count, path: path, how: "restored", seconds: now() - t0, prefillSeconds: 0,
                                       mbRead: Double(engine.store.snapshot { $0.bytes } - b0) / 1e6, landed: n, landSeconds: ls, savedMB: 0)
                } catch {
                    // another configuration or a damaged file: prefilled again below (and overwritten)
                    lock.withLock { prefixLog }?("[prefix-cache] system state \((path as NSString).lastPathComponent) refused (\(error)); prefilling it again")
                    try? FileManager.default.removeItem(atPath: path)
                    engine.forget()
                    fed = []
                    logits = nil
                }
            }
            let tp = now()
            let total = ids.count
            progress(0, total)
            logits = engine.prefill(ids[...], layer: { f in progress(min(Int(f), total - 1), total) }) { n in
                self.fed = Array(ids[..<n])
                progress(n, total)
            }
            eval(logits!)
            fed = ids
            let ps = now() - tp
            let s = try engine.savePrefix(path: path, ids: fed, logits: logits!)
            pc.enforceCap()
            return SystemState(tokens: ids.count, path: path, how: "prefilled", seconds: now() - t0, prefillSeconds: ps,
                               mbRead: Double(engine.store.snapshot { $0.bytes } - b0) / 1e6, landed: 0, landSeconds: 0,
                               savedMB: Double(s.bytes) / 1e6)
        }
    }

    /// The one-time setup's drive calibration (Store.calibrateReaders) on the engine thread, right after load: read
    /// GB/s per reader count. The best count is then the session's (limitReaders); a later launch builds its engine
    /// with it (EngineOptions.readers / prefillReaders). Nothing held is touched.
    public func calibrateReaders(_ candidates: [Int], seconds: Double = 2) async -> [(readers: Int, gbps: Double)] {
        (try? await worker.run { [self] in engine.store.calibrateReaders(candidates, seconds: seconds) }) ?? []
    }

    public func limitReaders(_ n: Int) {
        engine.store.limitReaders(n)
    }

    /// Reader threads of the store (the most a calibration can measure).
    public var readerThreads: Int { engine.store.readerThreads }

    /// The slabs back if releaseMemory freed them (engine thread).
    func ensureSlabs() throws {
        if engine.store.slabsReleased { try engine.store.reallocSlabs() }
    }

    /// Free what the app does not need while it is in the background (on the engine thread, after the work queued so
    /// far): the expert slabs (and the codec decoder's aliases of them) and MLX's buffer cache. The weights, the model
    /// state and the held set stay (every held record becomes pending, as after a lazy restore). Returns the bytes freed
    /// from the slabs.
    public func releaseMemory() async -> Int {
        (try? await worker.run { [self] in
            let b = engine.store.releaseSlabs()
            Memory.clearCache()
            return b
        }) ?? 0
    }

    /// After releaseMemory: the slabs allocated again and (with `land`) the held records read back in now, in idle time.
    /// Returns the records read and the seconds. A generate before this does the same itself (ensureSlabs).
    public func restoreMemory(land: Bool) async throws -> (records: Int, seconds: Double) {
        try await worker.run { [self] in
            let t0 = now()
            try ensureSlabs()
            var n = 0
            if land && engine.store.hasPending { n = engine.store.landPending().relanded }
            return (n, now() - t0)
        }
    }

    /// The slabs are freed (releaseMemory) and not allocated again yet.
    public var memoryReleased: Bool { engine.store.slabsReleased }

    /// The conversation as the app's live history mode gives it in the next prompt (2026-11-07; docs/TOOLS.md "History"):
    /// an assistant turn after the last user message whose content starts with the generation prompt's empty think block
    /// and has no `reasoning_content` (the app prepends the block to a tool-call turn of the current round, so the template
    /// splits it off and writes it back) is given as the app gives every earlier thinking-off turn once a new user message
    /// follows (PhoneStoreChatEngine.templateMessages, live): the same content verbatim with `reasoning_content` "", which
    /// the template writes as is. The same bytes as the current render (the template writes "<think>\n" + "" +
    /// "\n</think>\n\n" + the rest for a turn after the last user message); the difference is that the next prompt, where
    /// a new user message makes this an earlier turn, keeps the block instead of dropping it.
    static func liveHistory(_ messages: [[String: String]]) -> [[String: String]] {
        let block = "<think>\n\n</think>\n\n"
        return messages.map { m in
            guard m["role"] == "assistant", m["reasoning_content"] == nil, let c = m["content"], c.hasPrefix(block) else { return m }
            var d = m
            d["reasoning_content"] = ""
            return d
        }
    }

    /// The reply-end save (prefixSaveAtReplyEnd), on the engine thread right after a reply: when the next prompt renders
    /// this reply as generated (checked against the template, with the history as the app's live mode gives it next:
    /// liveHistory), feed the end of the turn ("<|im_end|>\n") and save the state.
    func saveReplyEnd(messages: [[String: String]], thinking: Bool, result r: GenerationResult) {
        let (pcache, pon, plog, on, withHeader) = lock.withLock { (prefixCache, prefixOn, prefixLog, saveAtEnd, endHeader) }
        guard on, pon, let pc = pcache, !r.ids.isEmpty, logits != nil else { return }
        let want = r.promptIds + r.ids
        guard fed == want else { return }
        // a reply inside or with a think block is rewritten by the template (the header save covers thinking replies)
        let tt = ThinkTokens(tokenizer)
        guard !r.startedInThinking, r.thinkingTokens == 0, !r.ids.contains(where: { $0 == tt.open || $0 == tt.close }) else { return }
        let t0 = now()
        let reply: [String: String] = ["role": "assistant", "content": "<think>\n\n</think>\n\n" + r.text, "reasoning_content": ""]
        // the end of the turn and (2026-11-07) the next turn's header "<|im_start|>user\n" (3 tokens every follow-up would
        // otherwise prefill; a tool result is rendered in a user turn too), when the template renders them so; else the
        // end alone
        let ends = (withHeader ? [tokenizer.encode(text: "<|im_end|>\n<|im_start|>user\n", addSpecialTokens: false)] : [])
            + [tokenizer.encode(text: "<|im_end|>\n", addSpecialTokens: false)]
        // a reply after a tool round (2026-11-07): the round's call turn renders with its empty think block only while it
        // comes after the last user message; the next prompt gives it as the app's live history does (liveHistory), so the
        // check renders that, and a searched chat continues after a relaunch from the end of its answer
        guard let next = try? promptIds(messages: Self.liveHistory(messages) + [reply, ["role": "user", "content": "x"]], thinking: thinking),
              next.count > want.count, Array(next[..<want.count]) == want,
              let end = ends.first(where: { next.count > want.count + $0.count && Array(next[want.count..<(want.count + $0.count)]) == $0 }) else {
            plog?("[prefix-cache] reply end: not saved (the next prompt does not render the reply's \(r.ids.count) tokens as generated)")
            return
        }
        let path = pc.path(for: want + end)
        if FileManager.default.fileExists(atPath: path) {
            pc.touch(path)
            plog?("[prefix-cache] reply end: \(want.count + end.count) tokens already cached (\((path as NSString).lastPathComponent))")
            return
        }
        let tf = now()
        logits = engine.prefill(end[...])  // a batched chunk (EngineOptions.prefillChunk), or step() per token at 0
        fed += end
        eval(logits!)
        let feedSec = now() - tf
        do {
            let s = try engine.savePrefix(path: path, ids: fed, logits: logits!)
            let gone = pc.enforceCap()
            let u = pc.usage
            plog?(String(format: "[prefix-cache] reply end: fed %d end tokens in %.3f s, saved %d tokens -> %@ (%.1f MB, %.3f s)%@ | cache: %d entries, %.2f GB | %.3f s in all",
                         end.count, feedSec, fed.count, (path as NSString).lastPathComponent, Double(s.bytes) / 1e6, s.seconds,
                         gone.isEmpty ? "" : ", \(gone.count) least recently used deleted", u.entries, Double(u.bytes) / 1e9, now() - t0))
        } catch {
            plog?("[prefix-cache] reply end: save failed: \(error)")
        }
    }

    /// One reply, on the engine thread. The loop is the CLI's (sample, stop check, step), plus the per-token stats.
    func run(messages: [[String: String]], thinking: Bool, maxNew: Int, sampling: SamplingParams = .greedy, thinkingBudget: Int? = nil,
             prefillProgress: (PrefillEvent) -> Void = { _ in }, emit: (TokenEvent) -> Void) throws -> GenerationResult {
        let tTpl = now()
        try ensureSlabs()  // freed in the background (releaseMemory): allocated again, the held records pending
        let ids = try promptIds(messages: messages, thinking: thinking)
        let templateSec = now() - tTpl
        var how = "cold"
        var start = 0
        let (pcache, pon, plog) = lock.withLock { (prefixCache, prefixOn, prefixLog) }
        let pc = pon ? pcache : nil
        var prefixStats: [String: Double] = [:]
        let tRestore = now()
        // the live state continues this prompt, unless the cache has a longer prefix of it (2026-11-07: the live state is
        // often the system prompt's alone, launch and new chat, and a chat continued after a relaunch has its reply-end
        // state cached)
        let liveOK = logits != nil && !fed.isEmpty && ids.count > fed.count && Array(ids[..<fed.count]) == fed
        let longer = liveOK ? pc.flatMap { c in c.longest(ids).flatMap { $0.n > fed.count ? $0 : nil } } : nil
        if liveOK && longer == nil {
            how = "live prefix"
            start = fed.count
        } else {
            engine.forget()  // mlx_run.py Session.feed: reset() + reset_slots()
            fed = []
            logits = nil
            if let pc {
                if let hit = pc.longest(ids) {
                    do {
                        // lazy re-land (EngineOptions.relandLazy): nothing read here; the prompt's chunk reads what it uses
                        let r = try engine.loadPrefix(path: hit.path, ids: ids, reland: true, lazy: engine.opt.relandLazy)
                        fed = r.ids
                        logits = r.logits
                        start = r.ids.count
                        how = "restored prefix"
                        var info = r.info
                        var landNote = ""
                        if start >= ids.count || engine.opt.prefillChunk <= 1 {
                            // nothing left for a chunk to plan (or token-by-token prefill): land the held set now, as before
                            let tl = now()
                            let l = engine.store.landPending()
                            info = info.landedAtOnce(l, seconds: now() - tl)
                            if r.info.lazy { landNote = String(format: " (lazy restore landed at once: %@)", start >= ids.count ? "nothing to prefill" : "token-by-token prefill") }
                        }
                        prefixStats = info.dict
                        prefixStats["prefix_hit"] = 1
                        prefixStats["prefix_restore_ms"] = 1000 * (now() - tRestore)
                        pc.touch(hit.path)
                        plog?(String(format: "[prefix-cache] hit: %d of %d tokens from %@ | state %.1f MB loaded in %.3f s | re-landed %d records (%.1f MB) in %.3f s, %d kept in place, %d moved, %d deferred to the prompt's chunk, %d held%@",
                                     info.n, ids.count, (hit.path as NSString).lastPathComponent, Double(info.fileBytes) / 1e6, info.loadSeconds,
                                     info.relanded, Double(info.relandBytes) / 1e6, info.relandSeconds, info.kept, info.moved, info.deferred, info.held, landNote))
                    } catch {
                        plog?("[prefix-cache] skipped \((hit.path as NSString).lastPathComponent): \(error)")
                        engine.forget()
                        fed = []
                        logits = nil
                        prefixStats["prefix_hit"] = 0
                        prefixStats["prefix_error"] = 1
                    }
                } else {
                    prefixStats["prefix_hit"] = 0
                    let u = pc.usage
                    plog?(String(format: "[prefix-cache] miss: no cached prefix of the %d tokens | cache: %d entries, %.2f GB", ids.count, u.entries, Double(u.bytes) / 1e9))
                }
            }
        }
        // states to write during this prefill: the end of a leading system message (+ the next header) and the end of the
        // prompt, unless already covered by what was restored or continued, or already on disk
        var saveAt: [Int] = []
        if let pc {
            let hdr = lock.withLock { saveAtHeader } ? assistantHeader(ids: ids, thinking: thinking) : nil
            let sys = systemBoundary(messages: messages, thinking: thinking, ids: ids)
            // the LRU never deletes the system-prompt state of the conversation in progress (restored or about to be saved)
            pc.pinned = sys.map { [pc.path(for: ids[..<$0])] } ?? []
            let cands = [sys, hdr, lock.withLock { saveAtPromptEnd } ? ids.count : nil].compactMap { $0 }
            saveAt = cands.filter { $0 > start && !FileManager.default.fileExists(atPath: pc.path(for: ids[..<$0])) }
        }
        var saveSec = 0.0, saveBytes = 0, saved = 0
        func snapshot() {
            guard let pc, let lg = logits else { return }
            eval(lg)
            do {
                let p = pc.path(for: fed)
                let r = try engine.savePrefix(path: p, ids: fed, logits: lg)
                saveSec += r.seconds
                saveBytes += r.bytes
                saved += 1
                let gone = pc.enforceCap()
                let u = pc.usage
                plog?(String(format: "[prefix-cache] saved %d tokens -> %@ (%.1f MB, %.3f s)%@ | cache: %d entries, %.2f GB", fed.count,
                             (p as NSString).lastPathComponent, Double(r.bytes) / 1e6, r.seconds,
                             gone.isEmpty ? "" : ", \(gone.count) least recently used deleted", u.entries, Double(u.bytes) / 1e9))
            } catch {
                plog?("[prefix-cache] save failed: \(error)")
            }
        }
        // GPU profiler (docs/G4-phone-gpu.md): env PHONE_GPU_PROF=tags|regions|serial profiles this session's first reply
        // and writes Documents/gpuprof-<mode>.txt (vendored mlx device.cpp "phone_prof"); off otherwise. tags: the run as
        // it is, each submit's kernels tagged; regions: each region evaluated on its own; serial: regions, and every
        // command buffer waited for (with MLX_MAX_OPS_PER_BUFFER=0: one kernel on the GPU at a time).
        let profMode = ProcessInfo.processInfo.environment["PHONE_GPU_PROF"]
        let profiling = profMode != nil && !profDone
        if profiling {
            profDone = true
            mlx_phone_prof_clear()
            mlx_phone_prof_set(1)
            Self.gpuCalibrate(engine)
            engine.prof = profMode == "tags" ? 1 : 2
            if profMode == "serial" { mlx_phone_prof_set(2) }
            mlx_phone_prof_mark(1, 0)
        }
        // the prefill's own stats (reset before it; the reply's stats are reset again after it, as before)
        engine.resetStats()
        let gp0 = gpuStats()
        let tp = now()
        let relandBytes = Int((prefixStats["prefix_reland_mb"] ?? 0) * 1e6)
        var lastTick = -1
        func progress(_ done: Int) {
            if done == lastTick { return }
            lastTick = done
            let t = now()
            prefillProgress(PrefillEvent(kind: how, promptTokens: ids.count, reusedTokens: start, total: ids.count - start, done: done,
                                         bytesRead: engine.store.snapshot { $0.bytes }, relandBytes: relandBytes, elapsed: t - tTpl,
                                         prefillElapsed: t - tp - saveSec))
        }
        progress(0)
        // segments end at the state saves inside the prompt; each is fed by Engine.prefill (chunks of
        // options.prefillChunk positions, or token by token at 0), with a progress event per chunk (per token at 0)
        var segStart = start
        for end in (saveAt.filter { $0 > start && $0 < ids.count }.sorted() + [ids.count]) where end > segStart {
            let seg = ids[segStart..<end]
            let base = fed.count
            logits = engine.prefill(seg, layer: { f in
                progress(min(base + Int(f), base + seg.count - 1) - start)  // inside a chunk: the layers done, as tokens
            }) { n in
                self.fed = Array(ids[..<(base + n)])
                progress(self.fed.count - start)
            }
            segStart = end
            if end < ids.count { snapshot() }
        }
        eval(logits!)
        if saveAt.contains(ids.count) { snapshot() }
        let prefillSec = now() - tp - saveSec
        let gp1 = gpuStats()
        let pst = engine.store.snapshot { s in (s.bytes, s.reads, s.syncReads, s.driveBusySec) }
        let nPre = Double(max(1, ids.count - start))
        var prefillStats: [String: Double] = [
            "prefill_tokens": Double(ids.count - start), "prefill_s": prefillSec, "template_ms": 1000 * templateSec,
            "prefill_ms_per_token": ids.count > start ? 1000 * prefillSec / nPre : 0,
            "prefill_drive_busy_frac": prefillSec > 0 ? pst.3 / prefillSec : 0,
            "prefill_read_ahead": Double(engine.opt.prefillReadAhead), "prefix_reland_lazy": engine.opt.relandLazy ? 1 : 0,
            "prefill_mb_read": Double(pst.0) / 1e6, "prefill_reads": Double(pst.1), "prefill_sync_reads": Double(pst.2),
            "prefill_expert_wait_ms": 1000 * engine.tm.readWait, "prefill_sync_wait_ms": 1000 * engine.tm.syncWait,
            "prefill_encode_ms": 1000 * engine.tm.encode, "prefill_gpu_ms": 1000 * (gp1.busy - gp0.busy),
            "prefill_drive_gb_per_s": prefillSec > 0 ? Double(pst.0) / prefillSec / 1e9 : 0,
            "prefill_drive_gb_per_s_busy": pst.3 > 0 ? Double(pst.0) / pst.3 / 1e9 : 0,
            "prefill_chunk": Double(engine.opt.prefillChunk), "prefill_readers": Double(engine.store.readerThreads),
            "prefill_auto_mb": engine.opt.prefillAutoBytes / 1e6,
        ]
        for (k, v) in engine.pf.dict { prefillStats[k] = v }
        if pc != nil {
            for (k, v) in prefixStats { prefillStats[k] = v }
            prefillStats["prefix_saved"] = Double(saved)
            prefillStats["prefix_save_ms"] = 1000 * saveSec
            prefillStats["prefix_save_mb"] = Double(saveBytes) / 1e6
            let u = pc!.usage
            prefillStats["prefix_cache_entries"] = Double(u.entries)
            prefillStats["prefix_cache_mb"] = Double(u.bytes) / 1e6
        }
        engine.resetStats()
        if profiling { mlx_phone_prof_mark(2, 0) }
        let g0 = gpuStats()

        var new: [Int] = []
        var reason = StopKind.maxNew
        // incremental detokenization (mlx_run.py Session.generate): decode only the tokens since the last emitted piece
        var prev = 0, done = 0
        var held = ""
        let t0 = now()
        var tLast = t0
        var bytesLast = 0
        var heldSumLast = 0, heldNLast = 0
        // the thinking segment by token id and the budget's forced close; sampling only after the prefill (greedy: no
        // Sampler, the engine's argmax path exactly as before)
        var tracker = ThinkingTracker(promptIds: ids, tokens: ThinkTokens(tokenizer), budget: thinkingBudget)
        var thinkIds: [Int] = [], answerIds: [Int] = []
        engine.sampler = sampling.isPlainArgmax ? nil : Sampler(sampling)
        defer { engine.sampler = nil }
        for _ in 0..<maxNew {
            if isCancelled { reason = .cancelled; break }
            let forcedTok = tracker.forced(lastText: { tokenizer.decode(tokens: new) })
            let tk = forcedTok ?? engine.sample(logits!)
            if stopIds.contains(tk) { reason = .eos; break }
            new.append(tk)
            let phase = tracker.observe(tk)
            if phase == .thinking { thinkIds.append(tk) } else if phase == .answer { answerIds.append(tk) }
            logits = engine.step(tk)
            fed.append(tk)
            var piece = ""
            let text = tokenizer.decode(tokens: Array(new[prev...]))
            let ts = Array(text.unicodeScalars), hs = held.unicodeScalars.count
            if ts.last != "\u{FFFD}" && ts.count > hs {
                piece = String(String.UnicodeScalarView(ts[hs...]))
                prev = done
                done = new.count
                held = piece
            }
            let inMem = engine.inMemoryRate  // waits for this token's GPU work (the next sample would wait for it anyway)
            let tNow = now()
            let bytes = engine.store.snapshot { $0.bytes }
            let hs2 = engine.heldSum, hn2 = engine.heldN
            if new.count == 1 {
                prefillStats["first_token_ms"] = 1000 * (tNow - t0)
                prefillStats["ttft_ms"] = 1000 * (tNow - tTpl)
            }
            emit(TokenEvent(index: new.count - 1, id: tk, text: piece, ms: 1000 * (tNow - tLast), bytesRead: bytes - bytesLast,
                            inMemory: inMem, held: hn2 > heldNLast ? (hs2 - heldSumLast) / (hn2 - heldNLast) : 0,
                            elapsed: tNow - t0, bytesTotal: bytes, phase: phase, forced: forcedTok != nil))
            tLast = tNow
            bytesLast = bytes
            heldSumLast = hs2
            heldNLast = hn2
        }
        eval(logits!)
        let dt = now() - t0
        let g1 = gpuStats()
        if profiling {
            mlx_phone_prof_mark(3, new.count)
            engine.prof = 0
            usleep(300_000)  // completion handlers of the last buffers
            mlx_phone_prof_set(0)
            let path = NSHomeDirectory() + "/Documents/gpuprof-\(profMode!).txt"
            let n = mlx_phone_prof_dump(path)
            print("GPUPROF mode=\(profMode!) buffers=\(n) tokens=\(new.count) file=\(path)")
            mlx_phone_prof_clear()
        }
        let split = Self.split(engine, tokens: new.count, dt: dt, g0: g0, g1: g1)
        let n = max(1, new.count)
        let st = engine.store.snapshot { s in (s.bytes, s.syncReads) }
        return GenerationResult(
            reason: reason, promptIds: ids, ids: new, prefill: how, prefillTokens: ids.count - start, prefillSeconds: prefillSec,
            seconds: dt, tokPerSec: Double(new.count) / dt, mibPerToken: Double(st.0) / Double(n) / 1048576.0,
            inMemory: engine.inMemoryRate, heldMean: Double(engine.heldSum) / Double(max(1, engine.heldN)), syncReads: st.1,
            text: tokenizer.decode(tokens: new), split: split, prefillStats: prefillStats, sampling: sampling,
            startedInThinking: tracker.startedInside, thinkingTokens: tracker.thinkingTokens, thinkingClosed: tracker.closed,
            thinkingForced: tracker.forcedClose, thinkingText: tokenizer.decode(tokens: thinkIds), answerText: tokenizer.decode(tokens: answerIds))
    }
}
