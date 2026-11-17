import Foundation
import PhoneStore

/// `-mode exact|prefetch` (diagnostics only; default prefetch). Exact mode (native/Sources/PhoneStore/Engine.swift, the
/// CLI's `--mode exact`): the released top-8 of every layer is the true top-8 of the gate, its misses are read on the
/// spot, so the outputs do not depend on what is in memory (nor on the cap): the phone-vs-Mac numerical difference
/// without prefetch mode's path dependence. No routers in exact mode (the CLI's `--mode exact --routers none`, as the Mac
/// references docs/g4_runs/{ref_exact_slim,tf_mac_exact_slim}.json were made; with routers a layer would need cap >= 16:
/// the 8 names and the released 8 at the same token); keep, evict, cap and readers are unchanged.
enum EngineMode {
    static var arg: String? { LaunchArgs.value("-mode") }
    /// nil: an unknown -mode value (ChatModel.load refuses it).
    static var mode: Mode? {
        guard let a = arg else { return .prefetch }
        return Mode(rawValue: a)
    }
    static var isExact: Bool { mode == .exact }
    static var label: String {
        guard let m = mode else { return "INVALID -mode \(arg ?? "")" }
        return m == .exact ? "exact (-mode exact: diagnostics; true top-8 every layer, misses read on the spot)" : "prefetch"
    }
}

/// The native runtime (native/, library `PhoneStore`) behind the app's ChatEngine protocol.
///
/// load(): `PhoneStoreEngine` in prefetch mode (`-mode exact`: exact mode, diagnostics) with the G4 speed settings: int8 lag-4 routers (the cache's
/// routers_k4_r8.safetensors; routers_k4.npz only when that is absent), keep 31,
/// hazard eviction, no drop, int4 resident, 8 readers x 4 chunks, late wait and head start on, and the cap the app chose.
/// generate(): chat template with enable_thinking = `thinking`, sampling per `GenSettings` (below: the release's
/// generation_config.json unless the settings say greedy; greedy in the test paths), the prompt prefilled in batched chunks (EngineOptions.prefillChunk; -prefillChunk 0: token
/// by token), one GenEvent per token. A prompt that does not extend what was fed starts cold (model state and held experts forgotten), as
/// mlx_run.py does.
final class PhoneStoreChatEngine: ChatEngine, @unchecked Sendable {
    private(set) var info: EngineInfo
    private let lock = NSLock()
    private var _thinking = false
    private var eng: PhoneStoreEngine?
    private var _last: ReplyDetail?
    private let cacheDir: URL
    private var _prefixOn = true

    /// The prefix cache's directory (Documents/prefix) and LRU cap (`-prefixCacheGB G`, default 1.5 GB since 2026-11-04,
    /// about 20 states of ~70 MB: with a save at each prompt end and each reply end, 0.5 GB evicted the warm system-prompt
    /// state after a few chats; 0: never delete). The system-prompt state of the conversation in progress is never evicted.
    static var prefixDir: URL { docsURL.appendingPathComponent("prefix", isDirectory: true) }
    static var prefixCapGB: Double { LaunchArgs.value("-prefixCacheGB").flatMap { Double($0) } ?? 1.5 }
    /// The reply-end save (PhoneStoreEngine.prefixSaveAtReplyEnd) in live history mode; `-noReplyEndSave` turns it off
    /// (the before/after measurement).
    static var replyEndSave: Bool { !LaunchArgs.has("-noReplyEndSave") }
    /// The reply-end save also feeds the next turn's "<|im_start|>user\n" (2026-11-07); `-replyEndHeader 0` feeds the end
    /// of the turn alone (the before/after measurement).
    static var replyEndHeader: Bool { LaunchArgs.value("-replyEndHeader").map { $0 != "0" && $0 != "off" } ?? true }

    var prefixCacheEnabled: Bool {
        get { lock.withLock { _prefixOn } }
        set {
            let e = lock.withLock { () -> PhoneStoreEngine? in _prefixOn = newValue; return eng }
            e?.prefixCacheEnabled = newValue
        }
    }

    var prefixCacheStatus: String {
        let on = prefixCacheEnabled
        guard let c = lock.withLock({ eng })?.prefixCache else {
            return (on ? "on" : "off") + String(format: ", %@/, cap %.1f GB (not loaded)", Self.prefixDir.lastPathComponent, Self.prefixCapGB)
        }
        let u = c.usage
        return (on ? "on" : "off") + String(format: ": %d states, %.0f MB in %@/, LRU cap %.1f GB", u.entries, Double(u.bytes) / 1e6,
                                            Self.prefixDir.lastPathComponent, Double(c.capBytes) / 1e9)
    }

    func clearPrefixCache() async { await clearPrefixCache(keepSetup: false) }

    /// `keepSetup`: the one-time setup's kept system states stay (the test flows' emptying: their prompts have no system
    /// message, so those states can never be hit by them).
    func clearPrefixCache(keepSetup: Bool) async {
        if let e = lock.withLock({ eng }) {
            try? await e.clearPrefixCache(keepKept: keepSetup)
        } else {
            try? FileManager.default.removeItem(at: Self.prefixDir)
        }
    }

    var thinking: Bool {
        get { lock.withLock { _thinking } }
        set { lock.withLock { _thinking = newValue } }
    }

    init(cacheDir: URL) {
        self.cacheDir = cacheDir
        info = EngineInfo(name: "PhoneStore", isStub: false, requiresStore: true,
                          detail: "native runtime (mlx-swift 0.32.3, MLX core 0.32.3); kernels: "
                              + PhoneStoreEngine.metallibInUse() + "; not loaded")
    }

    var settings: [(String, String)] {
        if let e = lock.withLock({ eng }) {
            let g = GenSettings.current(defaults: e.generationDefaults)
            return e.settings.map { $0.0 == "sampling" ? ("sampling", g.label) : $0 }
                + [("thinking budget", g.thinkingBudget.map { "\($0) tokens, then </think> forced" } ?? "none"),
                   ("history", HistoryMode.current.label), ("prefix cache", prefixCacheStatus)]
        }
        // not loaded (or refused): the settings load() will use
        let o = Self.options(modelDir: modelDirURL, cap: -1, cacheDir: cacheDir)
        let st = StoreStatus.check(dir: modelDirURL, cacheDir: cacheDir)
        return [("state", "not loaded; settings below are what load() uses"), ("mode", EngineMode.label),
                ("layout", st.layout == "slim" ? "slim: embeddings.safetensors + the converted cache, no resident.safetensors / routers_k4.npz"
                    : "full: resident.safetensors (embeddings; cache built if absent) + routers_k4.npz"),
                ("routers", o.mode == .exact ? "none (exact mode)" : "lag-4, 8 named, int\(o.routerBits) matrices (\(cacheDir.lastPathComponent)/routers_k4_r\(o.routerBits).safetensors, built from routers_k4.npz if absent)"), ("keep", "\(o.keep)"),
                ("evict", o.evict.rawValue), ("drop", o.noDrop ? "no drop" : "drop after keep"),
                ("resident", "int\(o.residentBits), " + (o.mappedWeights ? "memory-mapped (cache/*.mmap.safetensors, written once from the cache files)" : "read into memory (-mappedWeights 0)")),
                ("cap (slots / layer)", "the probe's cap, fixed for the session"),
                ("layer caps", (LaunchArgs.value("-layerCaps") ?? (LaunchArgs.value("-cap") != nil ? "uniform" : "auto")) + " (the probe's total, per layer)"),
                ("store codec", "\(o.storeCodec.rawValue): " + (st.codecLayers.isEmpty ? "no .psc layers" : "\(st.codecLayers.count) layers compressed (\(st.codecLayers.map(String.init).joined(separator: ",")))")),
                ("readers", "\(o.readers), \(o.readChunks) chunks / record"),
                ("page cache", o.noCache ? "F_NOCACHE + no readahead" : "ON"),
                ("read overlap", "late wait \(o.lateWait ? "on" : "off"), head start \(o.headStart ? "on" : "off")"),
                ("sampling", GenSettings.current(defaults: SamplingParams.generationConfig(dir: modelDirURL.path)).label),
                ("prefill", o.prefillChunk > 1 ? "batched, chunks of \(PhoneStoreEngine.chunkSummary(o)), \(max(o.readers, o.prefillReaders)) readers in a chunk" : "token by token"),
                ("kernel warm-up", o.kernelWarmupOn ? "at load: dry batched chunks compile the prefill kernels (state discarded)" : "off"),
                ("MLX cache limit", "\(o.cacheLimitMB) MB, wired limit \(o.wire ? "on" : "off")"),
                ("kernels", PhoneStoreEngine.metallibInUse()), ("native cache", o.cacheDir), ("model dir", o.dir),
                ("history", HistoryMode.current.label), ("prefix cache", prefixCacheStatus)]
    }

    /// The G4 speed settings (docs/G4-runtime.md): the CLI's `--mode prefetch --routers routers_k4 --router-bits 8
    /// --evict hazard --no-drop --keep 31 --resident-bits 4 --max-slots cap`; readers 8, read chunks 4, late wait, head
    /// start, F_NOCACHE, MLX cache limit 64 MB and the wired limit are EngineOptions' defaults.
    /// Test overrides (launch arguments, README "Diagnostics"): `-mlxCacheLimit MB` (the CLI's --cache-mb),
    /// `-readers N` (--readers), `-readChunks N` (--read-chunks), `-noWire` (--no-wire).
    static func options(modelDir: URL, cap: Int, cacheDir: URL) -> EngineOptions {
        var o = EngineOptions(dir: modelDir.path, cacheDir: cacheDir.path)
        o.store = "store4"
        o.mode = EngineMode.mode ?? .prefetch  // -mode exact: diagnostics only (ChatModel.load refuses an unknown value)
        // names the cache file routers_k4_r8.safetensors; the npz itself is read only when the cache file is absent.
        // Exact mode: no routers (see EngineMode).
        o.routers = o.mode == .exact ? "none" : modelDir.appendingPathComponent("routers_k4.npz").path
        o.routerBits = 8
        o.keep = 31
        o.evict = .hazard
        o.noDrop = true
        o.maxSlots = cap
        // per-layer caps (Probe: the auto allocation of 40 x cap slots unless -cap N / -layerCaps uniform)
        let lc = Budget.sessionLayerCaps(cap: cap)
        o.layerCaps = lc.caps
        o.layerCapsSpec = lc.label
        // -storeCodec auto|off: layers with layer_XX.psc + .idx are read compressed (decoded on the GPU into the slot)
        o.storeCodec = StoreCodec(rawValue: StoreStatusCodec.mode) ?? .auto
        if LaunchArgs.has("-codecCheck") { o.codecCheck = true }
        o.residentBits = 4
        if let v = LaunchArgs.value("-mlxCacheLimit").flatMap({ Int($0) }), v >= 0 { o.cacheLimitMB = v }
        // readers: the one-time setup's drive calibration (2026-11-07; docs/G4-runtime.md "Launch: once per install"): the
        // setup's own launch builds the store with the most candidates' threads to measure them; later launches use the
        // count it kept, for decode and prefill; -readers / -prefillReaders and the test flows keep the defaults (8)
        if OneTimeSetup.calibrating { o.prefillReaders = max(o.prefillReaders, OneTimeSetup.readerCandidates.max() ?? 8) }
        else if let r = OneTimeSetup.calibratedReaders { o.readers = r; o.prefillReaders = r }
        if let v = LaunchArgs.value("-readers").flatMap({ Int($0) }), v >= 1 { o.readers = v }
        if let v = LaunchArgs.value("-readChunks").flatMap({ Int($0) }), v >= 1 { o.readChunks = v }
        if LaunchArgs.has("-noWire") { o.wire = false }
        // batched prefill (docs/G4-runtime.md "Batched prefill"): -prefillChunk C (0: token by token), -prefillReaders N,
        // -prefillReadChunks N; defaults are EngineOptions'
        // -autotest / -teacherForce compare with the bundled references, which were made token by token: chunk 0 there
        // unless -prefillChunk says otherwise
        // auto chunk by default (2026-11-07, docs/G4-runtime.md "Chunks as large as memory allows"): each chunk the largest
        // whose activations fit the room under iOS's first memory warning (Budget.prefillAutoBytes), so a long input is
        // one pass over the store instead of one per 128 tokens; -prefillChunk N fixes the chunk, -prefillChunk auto:MB
        // sets the budget
        o.prefillAutoBytes = Budget.prefillAutoBytes(cap: cap, recordBytes: Budget.defaultRecordBytes)
        if LaunchArgs.has("-autotest") || LaunchArgs.has("-teacherForce") { o.prefillChunk = 0 }
        if let a = LaunchArgs.value("-prefillChunk") {
            if let v = Int(a), v >= 0 {
                o.prefillChunk = v
                o.prefillAutoBytes = 0
            } else if a.hasPrefix("auto:"), let mb = Double(a.dropFirst(5)) {
                o.prefillChunk = max(o.prefillChunk, 128)
                o.prefillAutoBytes = mb * 1e6
            } else if a == "auto" {
                o.prefillChunk = max(o.prefillChunk, 128)
            }
        }
        if let v = LaunchArgs.value("-prefillReaders").flatMap({ Int($0) }), v >= 1 { o.prefillReaders = v }
        if let v = LaunchArgs.value("-prefillReadChunks").flatMap({ Int($0) }), v >= 0 { o.prefillReadChunks = v }
        // kernel warm-up at load (Prefill.swift warmBatchedKernels): on with the batched prefill; -kernelWarmup 0|1 forces it
        if let v = LaunchArgs.value("-kernelWarmup") { o.kernelWarmup = v != "0" && v != "off" }
        // time to first token (2026-11-07, docs/G4-runtime.md "Time to first token"): -decodeWarmup 0 drops the warm-up's dry
        // decode step; -relandLazy 0 lands every restored record at the restore (eager); -prefillReadAhead N records per
        // layer read ahead from the lag-4 names during a chunk (-1 up to the cap, 0 off); -readAheadHPrev stale|shift;
        // -readAheadOverwrite 0|1|2 (targets: empty slots; + stale unnamed records; + held unnamed experts)
        if let v = LaunchArgs.value("-decodeWarmup") { o.decodeWarmup = v != "0" && v != "off" }
        if let v = LaunchArgs.value("-relandLazy") { o.relandLazy = v != "0" && v != "off" }
        if let v = LaunchArgs.value("-prefillReadAhead").flatMap({ Int($0) }) { o.prefillReadAhead = v }
        if let v = LaunchArgs.value("-readAheadHPrev") { o.readAheadShift = v == "shift" }
        if let v = LaunchArgs.value("-readAheadOverwrite").flatMap({ Int($0) }) { o.readAheadOverwrite = v }
        o.mappedWeights = MappedWeightsSetting.on
        return o
    }

    var lastDetail: ReplyDetail? { lock.withLock { _last } }

    /// The native engine once load() has set it (nil before, or after a failed load): its tokenizer and template
    /// (`promptIds`) for the tool loop's token counts and the prefix-cache housekeeping (ChatEngine.swift).
    var nativeEngine: PhoneStoreEngine? { lock.withLock { eng } }

    func load(modelDir: URL, cap: Int, progress: @escaping (String) -> Void) async throws {
        let o = Self.options(modelDir: modelDir, cap: cap, cacheDir: cacheDir)
        let send = progress
        let e = try await PhoneStoreEngine.load(options: o) { s in send(s) }
        try e.setPrefixCache(dir: Self.prefixDir.path, capGB: Self.prefixCapGB) { s in log(s) }
        e.prefixCacheEnabled = prefixCacheEnabled
        lock.withLock {
            eng = e
            info.detail = "native runtime (mlx-swift 0.32.3, MLX core 0.32.3); \(e.options.mode == .exact ? "EXACT mode (diagnostics), no routers" : "prefetch, int8 lag-4 routers"), cap \(e.cap) (\(o.layerCapsSpec), \(e.caps.reduce(0, +)) slots); layout \(e.layout); codec: \(e.layerPaths.filter { $0 == "psc" }.count) layers .psc; kernels: \(e.metallib)"
        }
    }

    func generate(messages: [ChatMessage], maxNew: Int) -> AsyncThrowingStream<GenEvent, Error> {
        generate(messages: messages, maxNew: maxNew, systemStateOnly: false)
    }

    /// `systemStateOnly`: of this prompt's states only the system boundary is saved (the system-prompt warm-up,
    /// ChatModel.runWarmUp: its prompt ends with an empty user turn no conversation starts with, so the prompt-end state
    /// and, thinking on, the assistant-header state would only be deleted again). Callers run one generate at a time (the
    /// app's replies wait for the warm-up), so the setting cannot leak into another prompt.
    func generate(messages: [ChatMessage], maxNew: Int, systemStateOnly: Bool) -> AsyncThrowingStream<GenEvent, Error> {
        let (e, think) = lock.withLock { (eng, _thinking) }
        let mode = HistoryMode.current
        let msgs = Self.templateMessages(messages, mode: mode)
        e?.prefixSaveAtAssistantHeader = mode == .live && !systemStateOnly
        e?.prefixSaveAtReplyEnd = mode == .live && Self.replyEndSave
        e?.prefixReplyEndHeader = Self.replyEndHeader
        e?.prefixSaveAtPromptEnd = !systemStateOnly && !LaunchArgs.has("-noPromptEndSave")  // tests: the same message again is prefilled again
        let gen = e.map { GenSettings.current(defaults: $0.generationDefaults) }
        return AsyncThrowingStream { cont in
            guard let e, let gen else { cont.finish(throwing: EngineError.notLoaded); return }
            let t = Task.detached(priority: .userInitiated) { [weak self] in
                var first = true
                var openShown = false  // the reply text already carries a "<think>"
                do {
                    for try await ev in e.generate(messages: msgs, thinking: think, maxNew: maxNew, sampling: gen.sampling,
                                                   thinkingBudget: think ? gen.thinkingBudget : nil) {
                        switch ev {
                        case .prefill(let p):
                            cont.yield(.prefill(PrefillTick(kind: p.kind, promptTokens: p.promptTokens, reusedTokens: p.reusedTokens,
                                                            total: p.total, done: p.done, mbRead: Double(p.bytesRead) / 1e6,
                                                            relandMB: Double(p.relandBytes) / 1e6, elapsed: p.elapsed,
                                                            prefillElapsed: p.prefillElapsed)))
                        case .token(let tk):
                            // The thinking segment by token id (PhoneStore ThinkingTracker). With thinking on, Qwen3.6's
                            // generation prompt already ends with "<think>\n", so the reply starts inside the block and
                            // its first token is thought: the text gets the opening tag (the UI splits on the tags; the
                            // token ids are untouched). A model-emitted <think> is kept once, a redundant one dropped; the
                            // close (sampled or forced by the budget) is in the text as "</think>".
                            var piece = tk.text
                            if first && (tk.phase == .thinking || tk.phase == .close) { piece = "<think>\n" + piece; openShown = true }
                            if tk.phase == .open { if openShown { piece = "" } else { openShown = true } }
                            first = false
                            let mb = Double(tk.bytesRead) / 1e6
                            cont.yield(.text(piece, TokenStats(index: tk.index, ms: tk.ms, tokPerSec: tk.ms > 0 ? 1000 / tk.ms : 0,
                                                               mbRead: mb, inMemory: tk.inMemory, expertsInMemory: tk.held,
                                                               decodeSeconds: tk.elapsed, mbTotal: Double(tk.bytesTotal) / 1e6,
                                                               phase: tk.phase.rawValue)))
                        case .done(let r):
                            // the reply's sampling and thinking segment ride in `split` (gen_* keys) into autotest/ui JSON
                            var split = r.split
                            let sp = r.sampling
                            split["gen_greedy"] = sp.isPlainArgmax ? 1 : 0
                            split["gen_temperature"] = Double(sp.temperature)
                            split["gen_top_k"] = Double(sp.topK)
                            split["gen_top_p"] = Double(sp.topP)
                            split["gen_min_p"] = Double(sp.minP)
                            split["gen_presence_penalty"] = Double(sp.presencePenalty)
                            split["gen_repetition_penalty"] = Double(sp.repetitionPenalty)
                            split["gen_seed"] = Double(sp.seed)
                            split["gen_thinking_budget"] = Double(gen.thinkingBudget ?? -1)
                            split["think_started_inside"] = r.startedInThinking ? 1 : 0
                            split["think_tokens"] = Double(r.thinkingTokens)
                            split["think_closed"] = r.thinkingClosed ? 1 : 0
                            split["think_forced_close"] = r.thinkingForced ? 1 : 0
                            let d = ReplyDetail(promptIds: r.promptIds, ids: r.ids, prefill: r.prefill, prefillTokens: r.prefillTokens,
                                                prefillSeconds: r.prefillSeconds, seconds: r.seconds, tokPerSec: r.tokPerSec,
                                                mibPerToken: r.mibPerToken, releasedTop8InMemory: r.inMemory,
                                                heldExpertsMean: r.heldMean, syncReads: r.syncReads, split: split,
                                                prefillStats: r.prefillStats)
                            self?.lock.withLock { self?._last = d }
                            let reason: StopReason = r.reason == .eos ? .eos : (r.reason == .cancelled ? .cancelled : .maxNew)
                            cont.yield(.finished(GenSummary(reason: reason, tokens: r.ids.count, seconds: r.seconds,
                                                            meanTokPerSec: r.tokPerSec, meanMBPerToken: r.mibPerToken * 1048576 / 1e6)))
                        }
                    }
                    cont.finish()
                } catch {
                    cont.finish(throwing: error)
                }
            }
            cont.onTermination = { reason in
                if case .cancelled = reason { e.cancel(); t.cancel() }
            }
        }
    }

    /// The chat-template input. An earlier assistant reply that was cut inside its think block (max tokens or stop) gets the
    /// template's close ("\n</think>\n\n") so the template drops its reasoning like any earlier turn's (the release's
    /// template strips reasoning only when `</think>` is present; left unclosed, the whole truncated trace would be fed back
    /// as the turn's answer). The test paths (-autotest, -prefixTest, -teacherForce; not -chatOnce) keep the history
    /// verbatim, as their references were made.
    /// History mode `live` (HistoryMode, ChatModel.swift): a thinking-off reply (no think tags in its text) is given back
    /// as the model generated it, after the generation prompt's empty block "<think>\n\n</think>\n\n", with
    /// `reasoning_content` set so the template neither splits nor strips it: the rendered history is then the engine's own
    /// fed tokens and the next prompt continues the live state. `strict`: the template's bytes.
    static func templateMessages(_ messages: [ChatMessage], mode: HistoryMode = .strict) -> [[String: String]] {
        let verbatim = TestFlow.active
        return messages.filter { !($0.role == .assistant && $0.content.isEmpty) }.map { m in
            var d = m.templateDict
            guard m.role == .assistant else { return d }
            if !verbatim, let o = m.content.range(of: "<think>"), m.content[o.upperBound...].range(of: "</think>") == nil {
                d["content"] = m.content + (m.content.hasSuffix("\n") ? "</think>\n\n" : "\n</think>\n\n")
            } else if mode == .live, !m.content.contains("<think>"), !m.content.contains("</think>") {
                d["content"] = "<think>\n\n</think>\n\n" + m.content
                d["reasoning_content"] = ""
            }
            return d
        }
    }

    /// Teacher forcing (`-teacherForce`, native/Sources/PhoneStore/TeacherForce.swift): feed `promptIds` + `ids` through
    /// the loaded engine one token at a time (the normal pipeline of the loaded mode) and record every position.
    func teacherForce(promptIds: [Int], ids: [Int], ref: TFReference?,
                      progress: @escaping @Sendable (TFPosition) -> Void) async throws -> TFRun {
        guard let e = lock.withLock({ eng }) else { throw EngineError.notLoaded }
        return try await e.teacherForce(promptIds: promptIds, ids: ids, k: 32, ref: ref, progress: progress)
    }

    func cancel() { lock.withLock { eng }?.cancel() }

    func waitIdle() async { await lock.withLock { eng }?.waitIdle() }

    func forget() { lock.withLock { eng }?.forget() }
}

/// Sampling and the thinking budget for the app's replies (docs/G4-runtime.md "Prompting and sampling").
///
/// UserDefaults keys (the settings sheet writes them; absent = the model's default from generation_config.json):
///   gen.greedy (Bool)                 true: argmax (the identity-test path); default false
/// Qwen3.6 card presets (SamplingParams.qwenThinking / qwenThinkingCoding / qwenNonThinking): thinking general 1.0 / 0.95 /
/// 20 / min_p 0 / presence 1.5; thinking coding 0.6 / 0.95 / 20 / presence 0; non-thinking 0.7 / 0.8 / 20 / presence 1.5.
///   gen.temperature (Double)          default 1.0
///   gen.topP (Double)                 default 0.95
///   gen.topK (Int)                    default 20
///   gen.minP (Double)                 default 0
///   gen.repetitionPenalty (Double)    default 1.0 (off)
///   gen.presencePenalty (Double)      default 0 (off; the card's thinking/non-thinking presets use 1.5)
///   gen.seed (Int)                    absent: a fresh random seed per reply (recorded in the reply's split, gen_seed)
///   gen.thinkingBudget (Int)          0 / absent: none; N: after N tokens inside the think block the template's close
///                                     "\n</think>\n\n" is forced (thinking on only)
/// Launch arguments: `-greedy` forces argmax; `-seed N`, `-thinkingBudget N` override the keys. The test paths (`-autotest`,
/// `-prefixTest`, `-teacherForce`) are always greedy: their identity checks compare ids with greedy references.
struct GenSettings {
    var sampling: SamplingParams
    var thinkingBudget: Int?
    var greedyBecause: String?

    static var testPathGreedy: String? {
        for a in ["-autotest", "-prefixTest", "-teacherForce", "-greedy"] where LaunchArgs.has(a) { return a }
        return nil
    }

    static func current(defaults: SamplingParams, store: UserDefaults = .standard) -> GenSettings {
        func num(_ k: String) -> NSNumber? { store.object(forKey: k) as? NSNumber }
        let budget = LaunchArgs.value("-thinkingBudget").flatMap { Int($0) } ?? num("gen.thinkingBudget")?.intValue
        let tb = (budget ?? 0) > 0 ? budget : nil
        if let why = testPathGreedy { return GenSettings(sampling: .greedy, thinkingBudget: tb, greedyBecause: "launch argument \(why)") }
        if store.bool(forKey: "gen.greedy") { return GenSettings(sampling: .greedy, thinkingBudget: tb, greedyBecause: "gen.greedy") }
        var p = defaults
        if let v = num("gen.temperature") { p.temperature = v.floatValue }
        if let v = num("gen.topP") { p.topP = v.floatValue }
        if let v = num("gen.topK") { p.topK = v.intValue }
        if let v = num("gen.minP") { p.minP = v.floatValue }
        if let v = num("gen.repetitionPenalty") { p.repetitionPenalty = v.floatValue }
        if let v = num("gen.presencePenalty") { p.presencePenalty = v.floatValue }
        if let v = LaunchArgs.value("-seed").flatMap({ UInt64($0) }) ?? num("gen.seed").map({ UInt64(truncatingIfNeeded: $0.int64Value) }) {
            p.seed = v
        } else {
            p.seed = UInt64.random(in: 0..<(1 << 32))
        }
        return GenSettings(sampling: p, thinkingBudget: tb, greedyBecause: nil)
    }

    var label: String {
        if let why = greedyBecause { return "greedy (\(why))" }
        let seedFixed = LaunchArgs.value("-seed") != nil || UserDefaults.standard.object(forKey: "gen.seed") != nil
        let s = sampling.summary
        return seedFixed ? s : s.replacingOccurrences(of: ", seed \(sampling.seed)", with: ", seed random per reply")
    }
}
