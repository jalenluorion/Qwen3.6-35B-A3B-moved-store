import Foundation
import SwiftUI
import UIKit
import PhoneStore
import CryptoKit

enum EngineState: Equatable {
    case idle
    case loading
    case ready
    case refused(String)
    case failed(String)

    var label: String {
        switch self {
        case .idle: return "idle"
        case .loading: return "loading"
        case .ready: return "ready"
        case .refused(let s): return "refused: " + s
        case .failed(let s): return "failed: " + s
        }
    }
}

/// One finished reply, as the autotest records it.
struct ReplyRecord: Codable {
    var prompt: String
    var thinking: Bool
    var tokens: Int
    var reason: String
    var seconds: Double
    var meanTokPerSec: Double
    var meanMBPerToken: Double
    var firstTokenMs: Double?
    var cancelRequestedAtToken: Int?
    var text: String
    var error: String?
    var maxNew: Int = 0
    /// The generated token ids (store-backed engines), for offline identity checks.
    var ids: [Int]?
    var promptTokens: Int?
    /// The engine's per-token split for this reply, phone-store-run's bench-JSON keys (ReplyDetail.split).
    var split: [String: Double]?
    /// The prompt prefill before the first token, and the first token itself (ReplyDetail.prefillStats).
    var prefill: [String: Double]?
    /// Memory warnings received while this reply was in progress (prefill included).
    var memoryWarnings: Int = 0
    /// "cold", "live prefix" or "restored prefix" (ReplyDetail.prefill).
    var prefillKind: String?
    /// What the UI showed at the end of the reply: the live decode rate (tokens / the engine's decode clock, prefill
    /// excluded), the time to the first token (send -> first piece, wall), MB per token (1e6 B) since the prefill.
    var uiDecodeTokPerSec: Double?
    var uiTtftMs: Double?
    var uiMBPerToken: Double?
    /// The engine's split rate (GenerationResult.tokPerSec) and the UI's difference from it, in percent.
    var engineTokPerSec: Double?
    var uiRateDiffPct: Double?
    /// The round's wall time, generate call to the stream's end (tool rounds: docs/TOOLS.md).
    var wallMs: Double?
}

/// What the app does on a memory warning (`-memWarn stop|log`). Default `log` everywhere since 2026-11-04: on the 15 Plus
/// the warnings come at a measured, survivable footprint (DESIGN.md "Phone, 2026-11-04"), so they are advisory and a reply
/// is never stopped by one. `-memWarn stop` (tests) restores the G4 plan's stop after the token in flight.
enum MemWarnMode: String {
    case stop, log
    static var current: MemWarnMode {
        if let v = LaunchArgs.value("-memWarn"), let m = MemWarnMode(rawValue: v) { return m }
        return .log
    }
}

/// The app's own settings (UserDefaults). The sampling keys `gen.*` are written here and read by the engine wrapper.
enum AppSettings {
    static let systemPromptKey = "chat.systemPrompt"
    static let defaultThinkingKey = "chat.defaultThinking"
    static let maxTokensKey = "chat.maxTokens"
    static let prefillSecPerTokenKey = "chat.prefillSecPerToken"
    static let restoreSecKey = "chat.restoreSec"
    static let liveContinuationKey = "chat.liveContinuationWorks"

    /// The default system prompt for new conversations: short (it is prefilled once at launch and restored after), and it
    /// asks for brevity because replies stream at ~3 tokens per second on the 15 Plus.
    static let defaultSystemPrompt = "You are a helpful assistant running entirely on this iPhone. Replies stream at about three words per second, so be concise: lead with the answer, use short paragraphs or lists, and skip filler."

    static var d: UserDefaults { .standard }
    /// `-system TEXT` (tests) wins; else the setting; else the default above.
    static var systemPrompt: String {
        if let s = LaunchArgs.value("-system") { return s }
        return d.string(forKey: systemPromptKey) ?? defaultSystemPrompt
    }
    static var defaultThinking: Bool { d.object(forKey: defaultThinkingKey) as? Bool ?? false }
    static var maxTokens: Int { (d.object(forKey: maxTokensKey) as? Int).map { max(16, $0) } ?? 1024 }
    /// Measured seconds per cold prompt token (learned from the replies' prefill stats; the phone's first figure ~0.27).
    static var prefillSecPerToken: Double { d.object(forKey: prefillSecPerTokenKey) as? Double ?? 0.27 }
    static var restoreSec: Double { d.object(forKey: restoreSecKey) as? Double ?? 2.0 }
    /// Whether the last follow-up in a live conversation continued the engine's state ("live prefix"); see README.
    static var liveContinuationWorks: Bool { d.object(forKey: liveContinuationKey) as? Bool ?? false }

    // Sampling (written by the settings sheet, read by PhoneStoreChatEngine; absent = the model's defaults below)
    static let temperatureKey = "gen.temperature", topPKey = "gen.topP", topKKey = "gen.topK", minPKey = "gen.minP"
    static let repetitionPenaltyKey = "gen.repetitionPenalty", greedyKey = "gen.greedy"
    static let presencePenaltyKey = "gen.presencePenalty", thinkingBudgetKey = "gen.thinkingBudget"
    static let modelTemperature = 1.0, modelTopP = 0.95, modelTopK = 20, modelMinP = 0.0, modelRepetitionPenalty = 1.0
    static let modelPresencePenalty = 0.0
    static let samplingKeys = [temperatureKey, topPKey, topKKey, minPKey, repetitionPenaltyKey, greedyKey, presencePenaltyKey]
}

/// How earlier assistant turns go into the chat template (Settings > Generation; README "History mode").
/// - live (the app's default): a thinking-off reply keeps the empty `<think>\n\n</think>\n\n` block it was generated
///   after, so the history is the model's own tokens and a follow-up continues the engine's live state (only the new turn
///   is prefilled); a thinking reply's reasoning is dropped as the template does, and the engine also saves the
///   prefix-cache state at the assistant header (before `<think>`), so the follow-up re-prefills only the answer.
/// - strict: exactly transformers' template bytes (the reasoning and the empty block stripped from earlier turns), so
///   every follow-up re-prefills the previous answer (~0.27 s per token on the 15 Plus).
/// `-historyMode live|strict` wins; the test flows (-autotest, -prefixTest, -teacherForce) are strict otherwise, since
/// their references were made with the template's bytes.
enum HistoryMode: String, CaseIterable {
    case live, strict
    static let key = "chat.historyMode"
    static var current: HistoryMode {
        if let v = LaunchArgs.value("-historyMode"), let m = HistoryMode(rawValue: v) { return m }
        if TestFlow.active { return .strict }
        return HistoryMode(rawValue: UserDefaults.standard.string(forKey: key) ?? "") ?? .live
    }
    var label: String {
        switch self {
        case .live: return "live: thinking-off turns keep their empty think block; thinking turns cached at the assistant header"
        case .strict: return "strict template: transformers' bytes; follow-ups re-prefill the previous answer"
        }
    }
}

/// The launch's test flow, if any (the order of ChatModel.startOnce): tests drive the model directly, never persist
/// conversations and never pre-warm. `-chatOnce` is not one: it runs the app's own path (run_on_phone.sh adds `-autotest`
/// to every launch, and -chatOnce takes precedence over it).
enum TestFlow {
    static var active: Bool {
        if ["-memLimitTest", "-teacherForce", "-prefixTest"].contains(where: { LaunchArgs.has($0) }) { return true }
        if LaunchArgs.value("-chatOnce") != nil || LaunchArgs.value("-chatContinue") != nil { return false }
        return LaunchArgs.has("-autotest") || LaunchArgs.has("-probe")
    }
}

/// The app's start-up as the chat screen shows it.
enum StartupPhase: Equatable {
    case checking        // probing memory and the model files
    case loading         // engine load (resident weights, routers, slabs)
    case warming         // a system-prompt state prefilled into the prefix cache (cold: once per prompt; the status line says so)
    case ready
    case unavailable     // refused or failed (engineState says why)
}

/// The prefill of the reply in progress: an estimate from the text until the engine's first progress event, then the
/// engine's own figures (`tick`: tokens run of the tokens to run, MB read, seconds; PhoneStore PrefillEvent).
struct PrefillProgress: Equatable {
    var start: Date
    var expectedTokens: Int
    var expectedSeconds: Double
    var restoring: Bool       // a cached prefix is expected to be restored first
    var waitingForWarmup: Bool
    /// What a message sent before the engine was ready waits for ("the model to load", "the one-time setup", "the system
    /// prompt"); nil when it does not wait.
    var waitingFor: String? = nil
    /// The engine's latest prefill event (nil until the first arrives, and for the stub).
    var tick: PrefillTick? = nil
    /// When `tick` arrived (the card extrapolates the elapsed time from it).
    var tickAt: Date? = nil
}

@MainActor
final class ChatModel: ObservableObject {
    /// The open conversation (a draft until its first message, then also in `conversations` and on disk).
    @Published var conv = Conversation()
    /// Saved conversations, newest first (Documents/conversations/).
    @Published var conversations: [Conversation] = []
    /// The open conversation's messages: what the engine is given (a leading system message included). The test flows
    /// use this directly as before.
    var messages: [ChatMessage] {
        get { conv.messages }
        set { conv.messages = newValue }
    }
    @Published var isGenerating = false
    @Published var engineState: EngineState = .idle
    @Published var startup: StartupPhase = .checking
    @Published var loadLog: [String] = []
    @Published var probe: Probe?
    @Published var thermal = thermalName(ProcessInfo.processInfo.thermalState)
    /// Thinking for the next reply (the chat template's enable_thinking); per conversation in the UI.
    @Published var thinking = false {
        didSet {
            engine.thinking = thinking
            if conv.thinking != thinking { updateConv(conv.id) { $0.thinking = thinking } }
        }
    }
    // the reply in progress, for the views
    /// The conversation and assistant message being generated (nil when idle).
    @Published var generatingConvID: UUID?
    @Published var streamingID: UUID?
    /// Cold-prefill estimate while waiting for the first token (nil once it arrives).
    @Published var prefill: PrefillProgress?
    /// When the reply's thinking block started and how long it took (live; saved in the message's meta at the end).
    @Published var thinkStart: Date?
    @Published var liveThinkingSeconds: Double?
    @Published var liveTokens = 0
    /// The latest memory reading for the status dots (taken every few seconds without logging) and the last warning's time.
    @Published var liveMem: MemSample?
    @Published var lastMemoryWarning: Date?
    /// The system contents (the tools block included when on) whose state is in the prefix cache from this launch's
    /// warm-ups: with tools on and off the system turn differs, so each mode has its own cached state.
    private(set) var warmedSystems: Set<String> = []
    private var warmTask: Task<Void, Never>?
    /// The launch (load, then the one-time setup or the system state's restore), awaited by a message sent before it ends.
    private var launchTask: Task<Void, Never>?
    @Published private(set) var launchPending = false
    /// The one-time setup in progress (its card), nil otherwise (OneTimeSetup).
    @Published var setupRun: SetupRun?
    /// The one-time setup just finished: the chat shows what to expect on this phone (until "Got it").
    @Published var expectationsPending = false
    /// Whether the app's slabs were released for the background (phaseChanged) and when it went there.
    private var backgroundSince: Date?
    private var releaseTask: Task<Void, Never>?
    /// The system content whose state the engine's slab holds right now: set by a warm-up (its records landed), cleared
    /// as soon as any reply runs (its prefill and decode replace them). A new chat whose system content is cached but not
    /// in the slab is warmed again in the background (newConversation): a restore of the cached state, no new tokens.
    private var slabSystem: String?
    /// The conversation whose state the engine holds live (its last reply), for the prefill estimate.
    private var liveConvID: UUID?
    /// The prefix cache toggle (settings; default on, remembered; `-prefixCache 0|1` overrides at launch).
    @Published var prefixCache = true {
        didSet {
            engine.prefixCacheEnabled = prefixCache
            UserDefaults.standard.set(prefixCache, forKey: "prefixCache")
            prefixStatus = engine.prefixCacheStatus
        }
    }
    @Published var prefixStatus = ""

    // live stats. replyTokPerSec: the reply's decode rate so far = generated tokens / the engine's decode clock (from the
    // first sample to this token; prefill, template and prefix restore excluded), the engine's own split rate. Before
    // 2026-11-03 it was tokens / (wall time since send), which counted the 12-18 s prefill and showed 1-1.5 tok/s for
    // replies the engine decoded at 2.6-2.9 tok/s.
    @Published var lastTokPerSec: Double = 0
    @Published var replyTokPerSec: Double = 0
    @Published var mbPerToken: Double = 0
    /// Time to the first token of the reply in progress (send -> first piece, wall ms); nil until it arrives.
    @Published var replyTtftMs: Double?
    /// The last finished reply: time to first token, prefill kind, prefix restore (re-land MB / ms), MB per token.
    @Published var lastTtftMs: Double?
    @Published var lastPrefillKind: String?
    @Published var inMemory: Double = 0
    @Published var expertsInMemory: Int = 0
    @Published var lastSummary: GenSummary?
    // diagnostics
    @Published var latestMem: MemSample?
    @Published var lastSplit: [String: Double]?
    @Published var lastPrefill: [String: Double]?
    @Published var memoryWarnings = 0
    private(set) var memSamples: [MemSample] = []
    let memWarnMode = MemWarnMode.current
    private var replyNo = 0          // replies started (1-based number of the one in progress)
    private var replyTokens = 0      // tokens of the reply in progress
    private var inReply = false
    // the tool loop (docs/TOOLS.md): the call being run (its assistant message) and its query, for the card and status line
    @Published var toolRunningID: UUID?
    @Published var toolRunningQuery: String?
    private var userStopped = false
    private var toolTask: Task<(text: String, meta: ToolMeta), Never>?
    private(set) var lastReply: ReplyRecord?

    let engine: ChatEngine
    let engineChoice: String
    let maxNew = 512
    private var genTask: Task<Void, Never>?
    private var started = false
    private var events: [String] = []

    let interactive = !TestFlow.active

    init() {
        (engine, engineChoice) = makeEngine()
        engine.thinking = false
        let pref: Bool = {
            if let v = LaunchArgs.value("-prefixCache") { return v != "0" && v != "off" }
            return (UserDefaults.standard.object(forKey: "prefixCache") as? Bool) ?? true
        }()
        prefixCache = pref
        engine.prefixCacheEnabled = pref
        if interactive {
            // the app: saved conversations, and a new draft with the settings' system prompt and thinking default
            conversations = ConversationStore.loadAll()
            conv = Conversation(system: AppSettings.systemPrompt, thinking: AppSettings.defaultThinking)
            thinking = conv.thinking  // (no didSet in init)
            engine.thinking = thinking
        } else if let s = LaunchArgs.value("-system") {
            messages = [ChatMessage(role: .system, content: s)]  // a fixed system prompt (tests)
        }
        _ = appStart
        Lifecycle.launched()
        MemProbe.resetLog()
        log("ENGINE \(engine.info.name) (\(engineChoice)) mode=\(EngineMode.label)")
        log("DIAG memWarn=\(memWarnMode.rawValue)" + diagnostics.filter { $0.key != "memWarn" }.sorted { $0.key < $1.key }
            .map { " \($0.key)=\($0.value)" }.joined())
        memSample("launch")
        let nc = NotificationCenter.default
        nc.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main) { [weak self] _ in
            let t = thermalName(ProcessInfo.processInfo.thermalState)
            Task { @MainActor in self?.thermal = t; self?.event("thermal -> \(t)") }
        }
        nc.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { [weak self] _ in
            // G4 plan: on a memory warning, finish the token and stop. On the Mac the warning reflects the whole
            // system's memory pressure (no per-app jetsam limit), so it is logged and generation goes on.
            // -memWarn log: record it (with the memory figures) and keep generating, accepting the jetsam risk (a test).
            Task { @MainActor in self?.memoryWarning() }
        }
    }

    var storeComplete: Bool { probe?.store.complete ?? false }
    var cap: Int { probe?.cap ?? 0 }

    func event(_ s: String) {
        events.append(s)
        log("EVENT " + s)
    }

    /// The diagnostic launch settings (probe.json `diagnostics`, autotest.json `diagnostics`).
    var diagnostics: [String: String] {
        var d = ["memWarn": memWarnMode.rawValue, "mode": EngineMode.arg ?? "prefetch", "prefixCache": prefixCache ? "on" : "off",
                 "historyMode": HistoryMode.current.rawValue]
        for k in ["-mlxCacheLimit", "-readers", "-readChunks", "-maxNew", "-prompt", "-cap", "-ref", "-tfRef", "-prefixCacheGB", "-chatOnce", "-chatThen", "-chatContinue", "-chatNew", "-chatNewDelay", "-newChatWarm", "-system", "-layerCaps", "-storeCodec", "-mappedWeights", "-prefillChunk", "-kernelWarmup"] {
            if let v = LaunchArgs.value(k) { d[String(k.dropFirst())] = v }
        }
        if LaunchArgs.has("-noWire") { d["noWire"] = "1" }
        if LaunchArgs.has("-noReplyEndSave") { d["noReplyEndSave"] = "1" }
        d["mappedWeightsInUse"] = MappedWeightsSetting.on ? "1" : "0"
        if LaunchArgs.has("-codecCheck") { d["codecCheck"] = "1" }
        if LaunchArgs.has("-teacherForce") { d["teacherForce"] = "1" }
        if LaunchArgs.has("-prefixTest") { d["prefixTest"] = "1" }
        if LaunchArgs.has("-memLimitTest") { d["memLimitTest"] = "1" }
        return d
    }

    /// Take a MemProbe sample: console line, mem.jsonl (survives a jetsam kill), the settings sheet, the JSONs.
    @discardableResult
    func memSample(_ label: String) -> MemSample {
        let m = MemProbe.read(label, reply: inReply ? replyNo : nil, token: inReply ? replyTokens : nil)
        memSamples.append(m)
        latestMem = m
        log(m.line)
        MemProbe.append(m)
        return m
    }

    func memoryWarning() {
        memoryWarnings += 1
        lastMemoryWarning = Date()
        let wh = inReply ? "reply \(replyNo), token \(replyTokens)" : (engineState == .loading ? "during load" : "idle")
        let m = memSample("memory warning #\(memoryWarnings) (\(wh))")
        let fig = String(format: "footprint %.3f GB, available %.3f GB, MLX active %@ GB", Double(m.physFootprint) / 1e9,
                         Double(m.osProcAvailable) / 1e9, m.mlxActive.map { String(format: "%.3f", Double($0) / 1e9) } ?? "n/a")
        switch memWarnMode {
        case .stop:
            event("memory warning (\(wh); \(fig)); stopping generation")
            stop()
        case .log:
            #if targetEnvironment(macCatalyst)
            event("memory warning (\(wh); \(fig)); Mac: system memory pressure; not stopped (-memWarn log)")
            #else
            event("memory warning (\(wh); \(fig)); not stopped (-memWarn log)")
            #endif
        }
        writeProbe()
    }

    // MARK: launch

    func startOnce() {
        if started { return }
        started = true
        let args = CommandLine.arguments
        Task {
            if args.contains("-memLimitTest") {  // before any engine load: the app's own memory limit (MemLimitTest.swift)
                reprobe()
                await runMemLimitTest()
                return
            }
            if interactive && LaunchArgs.value("-chatOnce") == nil {
                // the app as the user opens it (also -chatContinue and -interactive): the chat at once, the model loaded in
                // the background; the one-time setup's card only when its key changed (OneTimeSetup)
                await launchApp()
            } else {
                await reprobeAndLoad()
                if interactive { deleteStaleWarmupStates() }  // the warm-up's throwaway end states of earlier launches
            }
            if let text = LaunchArgs.value("-chatContinue") {  // the latest saved conversation, continued after a relaunch
                await chatOnce(text, continueLatest: true)
                return
            }
            if args.contains("-teacherForce") {  // takes precedence over -autotest (run_on_phone.sh always passes -autotest)
                await runTeacherForce()
            } else if args.contains("-prefixTest") {  // takes precedence over -autotest (run_on_phone.sh always passes it)
                await runPrefixTest()
            } else if let text = LaunchArgs.value("-chatOnce") {  // one normal reply through send(), the UI's own path
                await chatOnce(text)
            } else if args.contains("-autotest") {
                await runAutotest()
            } else if args.contains("-probe") {
                log("DONE")
            }
        }
    }

    // MARK: launch: once per install (docs/G4-runtime.md "Launch: once per install")

    /// The app's launch: probe, then (in the background of the chat, which is shown at once) the engine load and either the
    /// one-time setup (OneTimeSetup: its key changed, i.e. a new install, store or cache format; a card with its steps) or
    /// the system-prompt state's restore (no card). A message sent meanwhile waits for it (reply awaits launchTask).
    func launchApp() async {
        startup = .checking
        reprobe()
        let key = OneTimeSetup.key(probe: probe)
        let rec = OneTimeSetup.load()
        let canRun = !engine.info.isStub && storeComplete && prefixCache
        let demo = DownloadConfig.demoFlow && engine.info.isStub   // -demoFlow: the stub plays the setup card (simulator)
        let setup = demo || (canRun && (OneTimeSetup.forced ?? (rec?.key != key)))
        Lifecycle.note("LAUNCH setup \(setup ? "needed" : (canRun ? "done" : "not possible (\(engine.info.isStub ? "stub" : (prefixCache ? "store incomplete" : "prefix cache off")))"))"
            + (setup ? (rec == nil ? ": no setup record" : ": key changed (was \(rec!.key), now \(key))") : (rec.map { ": since \($0.date)" } ?? "")))
        OneTimeSetup.calibrating = setup && LaunchArgs.value("-readers") == nil
        if setup {
            setupRun = SetupRun(steps: [
                SetupStep(id: "kernels", title: "Compiling", state: .running, detail: "Loading the model"),
                SetupStep(id: "drive", title: "Measuring storage speed", state: .waiting, detail: nil),
                SetupStep(id: "assistant", title: "Preparing the assistant", state: .waiting, detail: nil),
                SetupStep(id: "tools", title: "Preparing tools", state: .waiting, detail: nil),
            ])
        }
        launchPending = true
        let t0 = Date()
        let t = Task { @MainActor in
            await self.load()
            let tl = Date()
            await self.afterLoad(setup: setup, key: key)
            Lifecycle.note(String(format: "LAUNCH ready in %.2f s (load %.2f s, then %@ %.2f s): engine %@", Date().timeIntervalSince(t0),
                                  tl.timeIntervalSince(t0), setup ? "one-time setup" : "system state", Date().timeIntervalSince(tl), self.engineState.label))
        }
        launchTask = t
        await t.value
        launchPending = false
    }

    /// After the load: the one-time setup's steps (setup) or the open draft's system state made live in the background.
    private func afterLoad(setup: Bool, key: String) async {
        if DownloadConfig.demoFlow && engine.info.isStub && setupRun != nil {   // -demoFlow: the card's steps, no engine work
            for id in ["kernels", "drive", "assistant", "tools"] {
                setStep(id, .running, nil)
                try? await Task.sleep(nanoseconds: 2_500_000_000)
                setStep(id, .done, nil)
            }
            var r = OneTimeSetup.Record(key: "demo", date: ISO8601DateFormatter().string(from: Date()), seconds: 10, steps: [:], cap: 0, caps: [])
            r.estimate = Expectations.estimate(cap: Expectations.refCap, gbps: Expectations.refGBps)
            OneTimeSetup.save(r)
            log("SETUP demo done (-demoFlow, stub engine)")
            setupRun = nil
            expectationsPending = true
            return
        }
        guard engineState == .ready, !engine.info.isStub, engine.nativeEngine != nil else {
            if setupRun != nil { setupRun?.failed = "the model did not load (\(engineState.label))" }
            return
        }
        deleteStaleWarmupStates()  // the old warm-up's throwaway end states of earlier launches (none since 2026-11-07)
        let ks = engine.settings.first { $0.0 == "kernel warm-up" }?.1 ?? ""
        log("SETUP kernels: " + ks)
        var readers: Int? = nil, drive: [String: Double]? = nil
        if setup {
            setStep("kernels", .done, ks.hasPrefix("skipped") ? "Already compiled" : (ks.hasPrefix("off") ? "Not needed" : "Compiled"))
            if OneTimeSetup.calibrating, let e = engine.nativeEngine {
                setStep("drive", .running, "Reading with \(OneTimeSetup.readerCandidates.map(String.init).joined(separator: " / ")) readers, 2 s each")
                let t0 = Date()
                let r = await e.calibrateReaders(OneTimeSetup.readerCandidates)
                readers = OneTimeSetup.pickReaders(r)
                drive = Dictionary(uniqueKeysWithValues: r.map { ("\($0.readers)", $0.gbps) })
                if let n = readers { e.limitReaders(n) }
                let line = r.map { String(format: "%d: %.3f GB/s", $0.readers, $0.gbps) }.joined(separator: ", ")
                Lifecycle.note(String(format: "SETUP drive: %@ -> %@ readers (%.1f s; %d reader threads)", line, readers.map(String.init) ?? "default",
                                      Date().timeIntervalSince(t0), e.readerThreads))
                setStep("drive", readers == nil ? .failed : .done, readers.map { n in String(format: "%d readers, %.2f GB/s", n, r.first { $0.readers == n }?.gbps ?? 0) } ?? "Not measured: defaults kept")
            } else {
                setStep("drive", .done, "Readers fixed by a launch argument")
            }
            OneTimeSetup.calibrating = false
        }
        let ok = await ensureSystemStates(setup: setup)
        if setup {
            if ok {
                let secs = Date().timeIntervalSince(setupRun?.started ?? Date())
                var r = OneTimeSetup.Record(key: key, date: ISO8601DateFormatter().string(from: Date()), seconds: secs,
                                            steps: Dictionary(uniqueKeysWithValues: (setupRun?.steps ?? []).map { ($0.id, $0.detail ?? "") }),
                                            cap: probe?.cap ?? 0, caps: probe?.layerCaps ?? [])
                r.states = OneTimeSetup.statePaths
                r.readers = readers
                r.drive = drive
                if let n = readers, let g = drive?["\(n)"] {
                    r.estimate = Expectations.estimate(cap: probe?.cap ?? 0, gbps: g)
                    log("SETUP estimate: " + (r.estimate.map { String(format: "%.1f tokens/s, %.1f words/s at cap %d, %.2f GB/s", $0.tokensPerSec, $0.wordsPerSec, $0.cap, $0.driveGBps) } ?? "none"))
                }
                OneTimeSetup.save(r)
                expectationsPending = r.estimate != nil
                Lifecycle.note(String(format: "SETUP done in %.1f s: marker written (%@)", secs, OneTimeSetup.recordURL.lastPathComponent))
                setupRun = nil
            } else {
                setupRun?.failed = "a step failed; it runs again at the next launch"
                Lifecycle.note("SETUP failed: no marker written; the next launch runs it again")
            }
        }
    }

    /// The system-prompt states new chats need, for the settings' system prompt: tools off ("the assistant") and on
    /// ("tools"). `setup`: every step runs (a cached state is restored, a missing one prefilled), shown on the card;
    /// else only missing ones are prefilled (once: e.g. after the system prompt or the tools were changed in settings)
    /// and the others left. Both are kept in the prefix cache across launches (PrefixCacheDir.kept). Then the open draft's
    /// state is made the live one, its records landed (the first message continues it and prefills only its own turn).
    /// Returns false when a state could not be prepared.
    @discardableResult
    func ensureSystemStates(setup: Bool) async -> Bool {
        guard let e = engine.nativeEngine, prefixCache, !engine.info.isStub else { return false }
        let base = AppSettings.systemPrompt
        let modes: [(id: String, sys: String)] = [("assistant", ToolSettings.systemContent(base, tools: false)),
                                                  ("tools", ToolSettings.systemContent(base, tools: true))]
        let paths = modes.compactMap { m in m.sys.isEmpty ? nil : (try? e.systemStatePath(system: m.sys)).flatMap { $0 } }
        OneTimeSetup.statePaths = paths
        e.prefixCache?.kept = Set(paths)
        // the open draft's mode last, so its state is the live one at the end
        let draft = conv.hasUserMessage ? nil : effectiveSystem(conv).trimmingCharacters(in: .whitespacesAndNewlines)
        let order = modes.filter { $0.sys != draft } + modes.filter { $0.sys == draft }
        var ok = true
        for m in order {
            if m.sys.isEmpty { if setup { setStep(m.id, .done, "No system prompt") }; continue }
            let path = (try? e.systemStatePath(system: m.sys)).flatMap { $0 } ?? ""
            let cached = FileManager.default.fileExists(atPath: path)
            let isDraft = m.sys == draft
            if !setup && cached && !isDraft { continue }
            if !setup && isDraft && Self.launchLand == "old" { await warmUp(); continue }  // the warm-up before 2026-11-07
            if setup { setStep(m.id, .running, cached ? "Found in the cache" : "Reading the prompt") }
            let r = await prepareSystem(m.sys, land: isDraft ? Self.launchLand != "lazy" : false, step: setup ? m.id : nil,
                                        why: setup ? "one-time setup" : (cached ? "launch: restore" : "missing: prefilled once"))
            if let r {
                if setup { setStep(m.id, .done, r.how == "prefilled" ? String(format: "Read %d tokens in %.0f s", r.tokens, r.seconds) : "Already prepared") }
            } else {
                ok = false
                if setup { setStep(m.id, .failed, "Failed: see the log") }
            }
        }
        // a draft with a system content of its own (e.g. -system): its state too
        if let d = draft, !d.isEmpty, !modes.contains(where: { $0.sys == d }), slabSystem != d {
            _ = await prepareSystem(d, land: Self.launchLand != "lazy", step: nil, why: "the open chat's system prompt")
        }
        prefixStatus = engine.prefixCacheStatus
        return ok
    }

    /// `-launchLand eager|lazy|old` (the launch's system state; default eager, docs/G4-runtime.md "Launch: once per
    /// install"): eager restores it and lands its held records in idle time; lazy restores it only (the first message's
    /// chunk reads what it uses); old is the warm-up before 2026-11-07 (a maxNew-0 generate of [system, empty user]).
    static var launchLand: String {
        if let v = LaunchArgs.value("-launchLand"), ["eager", "lazy", "old"].contains(v) { return v }
        return "eager"
    }

    func setStep(_ id: String, _ st: SetupStep.State, _ detail: String?) {
        guard let i = setupRun?.steps.firstIndex(where: { $0.id == id }) else { return }
        setupRun?.steps[i].state = st
        setupRun?.steps[i].detail = detail
    }

    /// One system-prompt state made the engine's live state (PhoneStoreEngine.prepareSystemState: restored, or
    /// prefilled cold and saved). nil on failure.
    private func prepareSystem(_ system: String, land: Bool, step: String?, why: String) async -> PhoneStoreEngine.SystemState? {
        guard let e = engine.nativeEngine else { return nil }
        let path = (try? e.systemStatePath(system: system)).flatMap { $0 } ?? ""
        let cold = !FileManager.default.fileExists(atPath: path)
        if cold && setupRun == nil { startup = .warming }
        defer { if startup == .warming { startup = engineState == .ready ? .ready : .unavailable } }
        log("WARM start (\(why)): system prompt, \(system.count) characters, tools \(system.hasPrefix(ToolPrompt.head) ? "on" : "off"), "
            + (cold ? "not cached: prefill" : "cached: restore") + (land ? " + land" : ""))
        let stepID = step
        do {
            let r = try await e.prepareSystemState(system: system, land: land) { done, total in
                guard let stepID else { return }
                Task { @MainActor [weak self] in
                    guard let self, self.setupRun?.steps.first(where: { $0.id == stepID })?.state == .running else { return }
                    self.setStep(stepID, .running, "Reading the prompt · \(done) of \(total) tokens")
                }
            }
            log("WARM done (\(why)): " + r.line)
            warmedSystems.insert(system)
            slabSystem = system
            liveConvID = nil
            prefixStatus = engine.prefixCacheStatus
            return r
        } catch {
            log("WARM failed (\(why)): \(error.localizedDescription)")
            return nil
        }
    }

    /// Settings closed: a changed system prompt or tools definition gets its states (prefilled once, in the background).
    func settingsClosed() {
        guard interactive, engineState == .ready, warmTask == nil, !isGenerating, !launchPending else { return }
        let t = Task { _ = await self.ensureSystemStates(setup: false) }
        warmTask = t
        Task { await t.value; self.warmTask = nil }
    }

    /// Whether a message can be sent now: the engine is ready, or still loading at launch (the message then waits).
    var acceptsMessages: Bool {
        if isGenerating { return false }
        switch engineState {
        case .ready: return true
        case .idle, .loading: return launchPending
        default: return false
        }
    }

    func reprobe() {
        var p = Probe.run()
        p.engine = engineReport
        p.memory = memSamples
        p.diagnostics = diagnostics
        p.write()
        probe = p
        log(p.summaryLine)
        if !p.store.complete {
            log("STORE incomplete: " + p.store.problems.prefix(8).joined(separator: "; ")
                + (p.store.problems.count > 8 ? "; ... (\(p.store.problems.count) total)" : ""))
        }
    }

    /// Every engine setting the settings sheet shows.
    var engineReport: EngineReport {
        EngineReport(name: engine.info.name, isStub: engine.info.isStub, requiresStore: engine.info.requiresStore,
                     choice: engineChoice, detail: engine.info.detail, state: engineState.label, thinking: thinking,
                     settings: Dictionary(engine.settings, uniquingKeysWith: { a, _ in a }), loadLog: loadLog)
    }

    /// probe.json again with the engine section as it is now (after a load or a refusal).
    func writeProbe() {
        guard var p = probe else { return }
        p.engine = engineReport
        p.memory = memSamples
        p.diagnostics = diagnostics
        p.write()
        probe = p
    }

    func reprobeAndLoad() async {
        startup = .checking
        reprobe()
        await load()
    }

    func load() async {
        guard let p = probe else { return }
        if engineState == .loading { return }
        loadLog = []
        defer { startup = engineState == .ready ? .ready : (engineState == .loading ? .loading : .unavailable) }
        if engine.info.requiresStore && !p.store.complete {
            engineState = .refused("store incomplete, \(p.store.problems.count) problem(s); see settings")
            log("LOAD refused: " + EngineError.storeIncomplete(p.store.problems).localizedDescription)
            writeProbe()
            return
        }
        if engine.info.requiresStore && p.cap < Budget.minCap {
            engineState = .refused(EngineError.capTooSmall(p.cap, Budget.minCap).localizedDescription)
            log("LOAD refused: cap \(p.cap) < \(Budget.minCap)")
            writeProbe()
            return
        }
        if !engine.info.isStub && EngineMode.mode == nil {
            engineState = .refused("unknown -mode \(EngineMode.arg ?? "") (exact|prefetch)")
            log("LOAD refused: unknown -mode \(EngineMode.arg ?? "") (exact|prefetch)")
            writeProbe()
            return
        }
        engineState = .loading
        startup = .loading
        if !engine.info.isStub {  // one engine per device (App.swift EngineLock): wait up to 60 s for another instance's to go
            var l = EngineLock.tryAcquire()
            if !l.ok {
                log("LOAD waiting: another instance of the app (pid \(l.holder)) has the engine loaded")
                loadLog.append("waiting for another instance of the app (pid \(l.holder)) to close")
                for _ in 0..<60 where !l.ok {
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                    l = EngineLock.tryAcquire()
                }
                if !l.ok {
                    engineState = .refused("another instance of the app (pid \(l.holder)) has the model loaded")
                    log("LOAD refused: another instance (pid \(l.holder)) still holds the engine after 60 s")
                    writeProbe()
                    return
                }
                log("LOAD the other instance is gone: loading")
            }
        }
        do {
            try await engine.load(modelDir: modelDirURL, cap: p.cap) { [weak self] s in
                log("LOAD " + s)
                Task { @MainActor in self?.loadLog.append(s) }
            }
            engineState = .ready
            log("LOAD ready: " + engine.info.detail)
            engine.prefixCacheEnabled = prefixCache
            prefixStatus = engine.prefixCacheStatus
            for (k, v) in engine.settings { log("SETTING \(k) = \(v)") }
            memSample("after load")
        } catch {
            if MappedWeightsSetting.on && !engine.info.isStub {
                // the cap was planned for memory-mapped weights: re-plan with them read into memory and load again
                MappedWeightsSetting.failedThisLaunch = true
                log("LOAD failed with memory-mapped weights (\(error.localizedDescription)); re-planning with the weights read into memory")
                engineState = .idle
                reprobe()
                await load()
                return
            }
            engineState = .failed(error.localizedDescription)
            log("LOAD failed: \(error.localizedDescription)")
        }
        writeProbe()
    }

    // MARK: chat

    /// The send button. `maxNew`: the reply's token limit (default: the settings' max tokens in the app, 512 in the tests).
    func send(_ text: String, maxNew: Int? = nil) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, acceptsMessages else { return }
        let mx = maxNew ?? (interactive ? AppSettings.maxTokens : self.maxNew)
        genTask = Task { _ = await self.reply(to: t, maxNew: mx) }
    }

    func stop() {
        userStopped = true
        toolTask?.cancel()
        engine.cancel()
    }

    /// Tools on or off for the open chat (the globe button): the chat's override; the system state of the new mode is
    /// warmed in the background so the next message restores it.
    func setTools(_ on: Bool) {
        guard !isGenerating else { return }
        updateConv(conv.id) { $0.tools = on == ToolSettings.enabled ? nil : on }
        persist(conv.id)
        Task { await warmUp() }
    }

    /// The tests' clear button: a new, unsaved conversation with only `-system`; the engine forgets its state (the next
    /// reply starts cold, or restored from the prefix cache), as mlx_run.py does.
    func clear() {
        stop()
        engine.forget()
        liveConvID = nil
        slabSystem = nil
        conv = Conversation(system: LaunchArgs.value("-system") ?? "", thinking: thinking)
        lastSummary = nil
    }

    func clearPrefixCache() {
        Task {
            await engine.clearPrefixCache()
            prefixStatus = engine.prefixCacheStatus
            warmedSystems.removeAll()
            slabSystem = nil
            event("prefix cache cleared")
            // keep the promise that a first message only prefills the user turn: the system states again (background)
            if interactive { await ensureSystemStates(setup: false) }
        }
    }

    // MARK: conversations

    /// A new draft with the settings' system prompt and thinking default. The engine keeps its state: the next reply
    /// decides (a prompt that does not continue it restores the longest cached prefix, i.e. the warmed system prompt).
    /// If another chat ran since the system state was warmed (its records are no longer in the slab), the new chat's system
    /// state is warmed again in the background while the user types (`warmUp(reland: true)`: the cached state restored,
    /// the empty user turn read; nothing is saved). A message sent before it finishes waits for it (reply awaits
    /// `warmTask`), as at launch. `-newChatWarm 0` turns this off (before/after runs).
    func newConversation() {
        if !conv.hasUserMessage && conv.systemPrompt == AppSettings.systemPrompt { prepareNewChat(); return }  // already a fresh draft
        conv = Conversation(system: AppSettings.systemPrompt, thinking: AppSettings.defaultThinking)
        thinking = conv.thinking
        prepareNewChat()
    }

    static var newChatWarm: Bool { LaunchArgs.value("-newChatWarm").map { $0 != "0" && $0 != "off" } ?? true }

    /// The open draft's system state into the slab in the background (see newConversation); no-op when it is there already,
    /// when a reply is running (reply() calls this again when it ends) or a warm-up is in progress.
    func prepareNewChat() {
        guard interactive, Self.newChatWarm, !conv.hasUserMessage else { return }
        Task { await warmUp(reland: true) }
    }

    func open(_ id: UUID) {
        guard conv.id != id, let c = conversations.first(where: { $0.id == id }) else { return }
        conv = c
        thinking = c.thinking
    }

    func delete(_ id: UUID) {
        if generatingConvID == id { stop() }
        conversations.removeAll { $0.id == id }
        ConversationStore.delete(id)
        if liveConvID == id { liveConvID = nil }
        if conv.id == id {
            conv = Conversation(system: AppSettings.systemPrompt, thinking: AppSettings.defaultThinking)
            thinking = conv.thinking
        }
    }

    /// An empty title goes back to the one taken from the first message.
    func rename(_ id: UUID, to title: String) {
        let t = title.trimmingCharacters(in: .whitespacesAndNewlines)
        updateConv(id) { c in
            if t.isEmpty {
                c.titleIsCustom = false
                c.title = Conversation.title(from: c.messages.first { $0.role == .user }?.content ?? "")
            } else {
                c.title = t
                c.titleIsCustom = true
            }
        }
        persist(id)
    }

    /// Drop the last reply (and the message it answered) and send that message again.
    func regenerate() {
        guard !isGenerating, engineState == .ready,
              let a = conv.messages.lastIndex(where: { $0.role == .assistant }),
              let u = conv.messages[..<a].lastIndex(where: { $0.role == .user }) else { return }
        let text = conv.messages[u].content
        let gone = conv.messages[u...].map(\.id)
        updateConv(conv.id) { c in
            c.messages.removeSubrange(u...)
            for id in gone { c.meta[id] = nil }
        }
        send(text)
    }

    /// Edit a sent message: the chat is cut back to before it (like regenerate), then the new text is sent. The engine
    /// restores the longest cached state of the shortened prompt (the previous reply's end, or the system prompt), so
    /// only the new message is read.
    func editAndResend(_ id: UUID, text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, !isGenerating, acceptsMessages,
              let u = conv.messages.firstIndex(where: { $0.id == id && $0.role == .user }) else { return }
        let gone = conv.messages[u...].map(\.id)
        log("EDIT message \(u) of \(conv.messages.count): the chat is cut back and the new text sent")
        updateConv(conv.id) { c in
            c.messages.removeSubrange(u...)
            for g in gone { c.meta[g] = nil }
            if !c.titleIsCustom && !c.hasUserMessage { c.title = "" }   // the first message: the title follows the edit
        }
        send(t)
    }

    /// Apply `f` to conversation `id` wherever it is: the open one and the saved list (kept in step).
    func updateConv(_ id: UUID, _ f: (inout Conversation) -> Void) {
        if conv.id == id { f(&conv) }
        if let i = conversations.firstIndex(where: { $0.id == id }) {
            if conv.id == id { conversations[i] = conv } else { f(&conversations[i]) }
        }
    }

    func conversation(_ id: UUID) -> Conversation? { conv.id == id ? conv : conversations.first { $0.id == id } }

    /// Write conversation `id` to Documents/conversations/ (the app only; the test flows never persist).
    func persist(_ id: UUID) {
        guard interactive, let c = conversation(id), c.hasUserMessage else { return }
        ConversationStore.save(c)
    }

    // MARK: warm-up: the system prompt into the prefix cache at launch

    /// Prefill the system prompt alone so its state is in the prefix cache, and a first message restores it and prefills
    /// only the user turn. The engine has no "prefill these ids" call, so this asks for 0 tokens after [system, empty user]:
    /// the engine saves the state at the system boundary (Session.swift `systemBoundary`: the system message and the next
    /// message's header) and nothing else (`systemStateOnly`: the throwaway prompt's end and assistant-header states are
    /// not written; earlier launches' are deleted by deleteStaleWarmupStates). The prefill costs about 0.3-0.5 s per
    /// system-prompt token once; later warm-ups restore the boundary state (47 tokens for the default prompt, tools off)
    /// and read the empty user turn (9 tokens thinking off, 7 on).
    /// The open chat's system content is warmed (a new chat: the settings' system prompt, plus the tools block when tools
    /// are on in new chats; the globe button warms the other mode when it is switched).
    /// `reland`: also when the state is cached already but not in the slab (`slabSystem`), i.e. a new chat after another
    /// one ran: a background restore that lands the system state's records while the user types (newConversation).
    func warmUp(reland: Bool = false) async {
        let system = effectiveSystem(conv).trimmingCharacters(in: .whitespacesAndNewlines)
        guard engineState == .ready, !engine.info.isStub, prefixCache, !system.isEmpty, !isGenerating, warmTask == nil else { return }
        if Self.launchLand == "old" {  // the warm-up before 2026-11-07 (measurements)
            let again = warmedSystems.contains(system)
            if again && !(reland && slabSystem != system) { return }
            let t = Task { await self.runWarmUp(system, background: again) }
            warmTask = t
            await t.value
            warmTask = nil
            return
        }
        // since 2026-11-07: the state made live (restored + landed, or prefilled once), unless it is live already. In a chat
        // with messages (the globe button) only a missing state is prepared: the conversation's own live state stays.
        if slabSystem == system { return }
        if conv.hasUserMessage, let p = (try? engine.nativeEngine?.systemStatePath(system: system)).flatMap({ $0 }),
           FileManager.default.fileExists(atPath: p) { return }
        let t = Task { _ = await self.prepareSystem(system, land: Self.launchLand != "lazy", step: nil,
                                                    why: reland ? "new chat" : (conv.hasUserMessage ? "tools switched: missing" : "open chat")) }
        warmTask = t
        await t.value
        warmTask = nil
    }

    /// `background`: the state is cached (a new chat's re-warm): the start-up card is not shown for it.
    private func runWarmUp(_ system: String, background: Bool = false) async {
        if !background { startup = .warming }
        defer { if !background { startup = engineState == .ready ? .ready : .unavailable } }
        let t0 = Date()
        log("WARM start: system prompt, \(system.count) characters, tools \(system.hasPrefix(ToolPrompt.head) ? "on" : "off")"
            + (background ? ", new chat: cached, not in the slab (background re-land)" : ""))
        let msgs = [ChatMessage(role: .system, content: system), ChatMessage(role: .user, content: "")]
        // no state at the end of this throwaway prompt (no conversation starts with an empty user turn); the system
        // boundary state (system turn + the user header) is what the warm-up caches
        let stream = (engine as? PhoneStoreChatEngine)?.generate(messages: msgs, maxNew: 0, systemStateOnly: true)
            ?? engine.generate(messages: msgs, maxNew: 0)
        do {
            for try await _ in stream {}
        } catch {
            log("WARM failed: \(error.localizedDescription)")
            return
        }
        if let d = engine.lastDetail {
            // the throwaway prompt's end state, should an engine have saved it (the engine's file name: n%07d-<key>)
            let end = PrefixCacheUsage.stateURL(ids: d.promptIds)
            let deleted = (try? FileManager.default.removeItem(at: end)) != nil
            let p = d.prefillStats ?? [:]
            log(String(format: "WARM done in %.2f s: %@, %.0f tokens prefilled, %.0f restored, %.0f states saved, prefill %.0f MB / %.0f records, re-land %.0f read / %.0f kept / %.0f deferred (prompt %d tokens; end state %@)",
                       Date().timeIntervalSince(t0), d.prefill, p["prefill_tokens"] ?? 0, p["prefix_tokens_restored"] ?? 0,
                       p["prefix_saved"] ?? 0, p["prefill_mb_read"] ?? 0, p["prefill_reads"] ?? 0, p["prefix_relanded"] ?? 0,
                       p["prefix_kept"] ?? 0, p["prefix_deferred"] ?? 0, d.promptIds.count,
                       deleted ? "deleted: \(end.lastPathComponent)" : "not saved"))
            learnTiming(d)
        }
        warmedSystems.insert(system)
        slabSystem = system
        liveConvID = nil
        prefixStatus = engine.prefixCacheStatus
    }

    /// Delete the warm-up's throwaway states earlier launches left in the prefix cache (PrefixCacheUsage
    /// .deleteWarmupThrowaways): the end of its prompt (thinking off and on) and, thinking on, its assistant header. Their
    /// ids end with the empty user turn and the generation prompt (or its header), which the engine's template gives as
    /// the common end of two such prompts with different system contents.
    func deleteStaleWarmupStates() {
        guard engineState == .ready, let e = engine.nativeEngine else { return }
        var tails: [[Int]] = []
        for think in [false, true] {
            func ids(_ sys: String) -> [Int]? {
                try? e.promptIds(messages: [["role": "system", "content": sys], ["role": "user", "content": ""]], thinking: think)
            }
            guard let a = ids("x"), let b = ids("yy z") else { continue }
            var k = 0
            while k < min(a.count, b.count) && a[a.count - 1 - k] == b[b.count - 1 - k] { k += 1 }
            tails.append(Array(a.suffix(k)))
        }
        // thinking on, the warm-up also saved the state at that prompt's assistant header (prefixSaveAtAssistantHeader,
        // before the generation prompt's "<think>"): the two tails' common part without its last token ("<think>")
        if tails.count == 2 {
            var c = 0
            while c < min(tails[0].count, tails[1].count) && tails[0][c] == tails[1][c] { c += 1 }
            if c > 1 { tails.append(Array(tails[0][..<(c - 1)])) }
        }
        let gone = PrefixCacheUsage.deleteWarmupThrowaways(tails: tails)
        let names: String = gone.isEmpty ? "" : " (" + gone.joined(separator: ", ") + ")"
        let tl: String = tails.map { "\($0.count) tokens" }.joined(separator: " / ")
        log("PREFIXCACHE stale warm-up end states deleted: \(gone.count)\(names) | tails \(tl)")
        if !gone.isEmpty { prefixStatus = engine.prefixCacheStatus }
    }

    // MARK: prefill estimate (no prefill progress events from the engine yet)

    /// What the next reply's prefill should cost: the tokens the engine will have to run (an estimate from the text) at
    /// the measured seconds per prompt token, plus a prefix-cache restore when one is expected.
    func expectedPrefill(history: [ChatMessage], convID: UUID, waiting: Bool, waitingFor: String? = nil) -> PrefillProgress {
        let sys = history.first?.role == .system ? history[0].content : ""
        let rest = history.filter { $0.role != .system }
        var tokens = 0
        var restoring = false
        if !engine.info.isStub {
            if liveConvID == convID && AppSettings.liveContinuationWorks, let u = rest.last {
                tokens = estimateTokens(u.content) + 6          // the new turn only: the engine continues its live state
            } else {
                tokens = rest.map { estimateTokens($0.content) }.reduce(0, +) + 4
                if !sys.isEmpty {
                    if prefixCache && warmedSystems.contains(sys) { restoring = slabSystem != sys } else { tokens += estimateTokens(sys) }
                }
            }
        }
        let secs = Double(tokens) * AppSettings.prefillSecPerToken + (restoring ? AppSettings.restoreSec : 0) + 0.5
        return PrefillProgress(start: Date(), expectedTokens: tokens, expectedSeconds: secs, restoring: restoring,
                               waitingForWarmup: waiting, waitingFor: waiting ? (waitingFor ?? "the system prompt") : nil)
    }

    /// Keep the estimate's two figures measured: seconds per prefilled prompt token and the restore time.
    private func learnTiming(_ d: ReplyDetail) {
        guard let p = d.prefillStats else { return }
        if let n = p["prefill_tokens"], let s = p["prefill_s"], n >= 8, s > 0 {
            let v = 0.5 * AppSettings.prefillSecPerToken + 0.5 * (s / n)
            UserDefaults.standard.set(v, forKey: AppSettings.prefillSecPerTokenKey)
        }
        if let r = p["prefix_restore_ms"], (p["prefix_hit"] ?? 0) > 0 {
            UserDefaults.standard.set(0.5 * AppSettings.restoreSec + 0.5 * r / 1000, forKey: AppSettings.restoreSecKey)
        }
    }

    /// The status dots' memory reading (no console line, no mem.jsonl entry).
    func refreshLiveMemory() { liveMem = MemProbe.read("status") }

    // MARK: status line (under the composer)

    enum Level: String { case good, fair, serious, critical, unknown }

    /// Memory headroom before iOS may close the app: os_proc_available_memory (iOS only; the Mac has no per-app limit).
    /// good >= 0.7 GB, fair >= 0.3 GB or a warning in the last 2 minutes, critical below. Warnings are advisory.
    var memoryLevel: Level {
        guard let m = liveMem ?? latestMem, m.osProcAvailable > 0 else { return .unknown }
        let recentWarning = lastMemoryWarning.map { Date().timeIntervalSince($0) < 120 } ?? false
        if m.osProcAvailable < 300_000_000 { return .critical }
        if m.osProcAvailable < 700_000_000 || recentWarning { return .fair }
        return .good
    }

    var thermalLevel: Level {
        switch thermal {
        case "nominal": return .good
        case "fair": return .fair
        case "serious": return .serious
        case "critical": return .critical
        default: return .unknown
        }
    }

    /// One line: the last reply's decode rate and time to first token; the reply in progress while generating.
    var statusLine: String {
        if isGenerating {
            if let q = toolRunningQuery { return "Searching: " + q }
            if replyTtftMs == nil {
                if prefill?.waitingForWarmup == true { return "Waiting for " + (prefill?.waitingFor ?? "the system prompt") }
                if let t = prefill?.tick, t.total > 0 { return "Reading the prompt · \(t.done) of \(t.total) tokens" }
                return "Reading the prompt"
            }
            return String(format: "%d tokens · %.1f tok/s", liveTokens, replyTokPerSec)
        }
        if let s = lastSummary, s.tokens > 0 {
            var parts = [String(format: "%.1f tok/s", s.meanTokPerSec)]
            if let t = lastTtftMs { parts.append(String(format: "first token %.1f s", t / 1000)) }
            return parts.joined(separator: " · ")
        }
        if let r = setupRun, r.failed == nil { return "Setting up once…" }
        switch startup {
        case .checking: return "Loading model…"
        case .loading:  // the load's last step: the batched prefill kernels' warm-up (PhoneStoreEngine.load)
            return loadLog.last?.hasPrefix("warm-up: compiling") == true ? "Preparing the model's kernels…" : "Loading model…"
        case .warming: return "Preparing the system prompt (once)…"
        case .ready: return engine.info.isStub ? "Demo engine (no model)" : "Ready"
        case .unavailable: return "Model unavailable"
        }
    }

    /// What the status line shows (docs/APP-COPY.md). `statusLine` above is the technical line the test flows log
    /// (`SHOWN status_line`); this is the same state in plain words.
    var statusText: String {
        if isGenerating {
            if let q = toolRunningQuery { return q.hasPrefix("http") ? "Reading a web page…" : "Searching the web…" }
            if replyTtftMs == nil {
                if prefill?.waitingForWarmup == true {
                    return prefill?.waitingFor == "the one-time setup" ? "Sending when setup finishes" : "Sending when the model is ready"
                }
                return "Reading your message…"
            }
            return replyTokPerSec > 0 ? String(format: "Replying · %.1f tokens/s", replyTokPerSec) : "Replying…"
        }
        if let s = lastSummary, s.tokens > 0 {
            var parts = [String(format: "%.1f tokens/s", s.meanTokPerSec)]
            if let t = lastTtftMs { parts.append(String(format: "started in %.1f s", t / 1000)) }
            return parts.joined(separator: " · ")
        }
        if let r = setupRun { return r.failed == nil ? "Getting ready…" : "Setup didn't finish" }
        switch startup {
        case .checking, .loading: return "Loading model…"
        case .warming: return "Updating instructions…"
        case .ready: return engine.info.isStub ? "Demo mode" : "Ready"
        case .unavailable: return lowMemory ? "Not enough memory" : "Model unavailable"
        }
    }

    /// The model was refused because this iPhone gives the app too little memory (cap below Budget.minCap).
    var lowMemory: Bool {
        guard engine.info.requiresStore, let p = probe else { return false }
        if case .refused = engineState { return p.cap < Budget.minCap }
        return false
    }

    /// Sends `prompt` in the open conversation and streams the reply into it. If `cancelAfter` is set, calls stop() after
    /// that many tokens. The reply keeps writing to its own conversation if another one is opened meanwhile.
    func reply(to prompt: String, cancelAfter: Int? = nil, maxNew: Int? = nil) async -> ReplyRecord {
        let convID = conv.id
        let assistant = ChatMessage(role: .assistant, content: "")
        var aid = assistant.id
        updateConv(convID) { c in
            c.messages.append(ChatMessage(role: .user, content: prompt))
            if !c.titleIsCustom && c.title.isEmpty { c.title = Conversation.title(from: prompt) }
        }
        let history = conv.messages
        updateConv(convID) { c in
            c.messages.append(assistant)
            c.updated = Date()
        }
        if interactive {
            if let i = conversations.firstIndex(where: { $0.id == convID }) { conversations.remove(at: i) }
            conversations.insert(conv, at: 0)
            persist(convID)
        }
        isGenerating = true
        generatingConvID = convID
        streamingID = aid
        liveTokens = 0
        thinkStart = nil
        liveThinkingSeconds = nil
        userStopped = false
        let continuing = liveConvID == convID
        defer {
            isGenerating = false
            inReply = false
            generatingConvID = nil
            streamingID = nil
            prefill = nil
            thinkStart = nil
            liveThinkingSeconds = nil
            toolRunningID = nil
            toolRunningQuery = nil
            // a new chat opened while this reply ran: its system state into the slab now (newConversation)
            if conv.id != convID { prepareNewChat() }
        }
        // tools (docs/TOOLS.md): the tools block in the system turn when this chat has tools on
        let tools = conversation(convID)?.toolsOn ?? false
        var engineHistory = engineMessages(history, tools: tools)
        if launchPending, let lt = launchTask {  // sent while the model loads at launch (or the one-time setup runs)
            prefill = expectedPrefill(history: engineMessages(history, tools: tools), convID: convID, waiting: true,
                                      waitingFor: setupRun != nil ? "the one-time setup" : "the model to load")
            log("QUEUED reply=\(replyNo + 1): sent while the model loads at launch; starts when it is ready")
            await lt.value
        }
        if engineState != .ready || userStopped {
            let why = userStopped ? "stopped before the model was ready" : "the model is not available (\(engineState.label))"
            let shownText = userStopped ? "Stopped." : (lowMemory ? Copy.lowMemory : "The model isn't available right now.")
            updateConv(convID) { c in
                if let i = c.messages.firstIndex(where: { $0.id == aid }) { c.messages[i].content = shownText }
            }
            persist(convID)
            log("REPLY not run: " + why)
            let r = ReplyRecord(prompt: prompt, thinking: thinking, tokens: 0, reason: userStopped ? "cancelled" : "error", seconds: 0,
                                meanTokPerSec: 0, meanMBPerToken: 0, firstTokenMs: nil, cancelRequestedAtToken: nil, text: "", error: why)
            lastReply = r
            return r
        }
        if let w = warmTask {  // the system prompt's warm-up is still running: the reply starts after it
            prefill = expectedPrefill(history: engineHistory, convID: convID, waiting: true)
            log("QUEUED reply=\(replyNo + 1): sent during the system prompt's warm-up; starts when it ends")
            await w.value
        }
        slabSystem = nil  // this reply's prefill and decode replace the warmed records
        let mx = maxNew ?? self.maxNew
        let think = engine.thinking
        let tSend = Date()
        // round 1; then, while the answer ends in a complete tool call: run it, append the call and the result, generate
        // again from the engine's live state (only the result is prefilled)
        var rounds: [ReplyRecord] = []
        var searches: [ToolMeta] = []
        var calls = 0
        // open_url's `result` names a result of the last search (docs/TOOLS.md "open_url by result number"); the last one
        // may be in an earlier message ("give me more detail" follows the searched answer), so it comes from the history
        var (lastResults, lastQuery) = lastSearch(convID)
        var freeRetryUsed = false
        while true {
            let rec = await runRound(prompt: prompt, convID: convID, aid: aid, history: engineHistory, think: think, mx: mx,
                                     cancelAfter: cancelAfter, continuing: continuing || !rounds.isEmpty, tools: tools,
                                     toolsAllowed: tools && calls < ToolSettings.maxCalls + 1)
            rounds.append(rec)
            guard tools, !userStopped, rec.error == nil, calls <= ToolSettings.maxCalls,
                  let (call, kept) = completedCall(convID: convID, aid: aid, think: think) else { break }
            calls += 1
            // the call turn: its text ends at </tool_call> (anything after it is dropped, as the template would)
            updateConv(convID) { c in
                if let i = c.messages.firstIndex(where: { $0.id == aid }) { c.messages[i].content = kept }
                c.meta[aid]?.toolCall = call
                c.meta[aid]?.reason = "tool_call"
            }
            toolRunningID = aid
            toolRunningQuery = call.query
            log("TOOL reply=\(replyNo) call \(call.name) \(call.arguments) (\(rec.tokens) tokens generated, reason \(rec.reason))")
            let limited = calls > ToolSettings.maxCalls
            let eng = engine
            // open_url ranks the page's sentences by the user's message and this message's searches
            let msgs = conversation(convID)?.messages ?? []
            let lastUser = msgs.lastIndex(where: { $0.role == .user }) ?? 0
            let queries = searches.filter { $0.name == "web_search" }.map(\.query)
            let context = ([msgs.indices.contains(lastUser) ? msgs[lastUser].content : ""]
                + (queries.isEmpty ? [lastQuery].compactMap { $0 } : queries)).joined(separator: " ")
            let last = lastResults
            let task = Task { @MainActor () -> (text: String, meta: ToolMeta) in
                if limited {
                    let t = "Error: tool limit reached. Answer with what you have."
                    return (t, ToolMeta(name: call.name, query: call.query, backend: "-", resultTokens: eng.toolResultTokens(t), error: "limit"))
                }
                return await WebSearch.run(call, context: context, last: last, count: { eng.toolResultTokens($0) })
            }
            toolTask = task
            let (text, meta) = await task.value
            toolTask = nil
            searches.append(meta)
            if meta.name == "web_search", meta.error == nil, !meta.results.isEmpty { (lastResults, lastQuery) = (meta.results, meta.query) }
            // a failed open_url of a URL the model wrote (not a search result) does not use up a call, once per message:
            // its error lists the results by number, so the next call can open one (ToolSettings.maxCalls)
            if call.isPageRead, !limited, meta.error != nil, meta.resultIndex == nil, !lastResults.isEmpty, !freeRetryUsed {
                freeRetryUsed = true
                calls -= 1
                log("TOOL reply=\(replyNo) open_url of an unlisted URL failed (\(meta.error ?? "")): not counted (\(calls) of \(ToolSettings.maxCalls) calls used)")
            }
            log(String(format: "TOOL reply=%d result: %d results, %d pages read (%.0f ms), %@ / %@ tokens, %.0f ms, backend %@%@\nTOOLRESULT %@", replyNo, meta.results.count,
                       meta.pages ?? 0, meta.pagesMs ?? 0, meta.resultTokens.map(String.init) ?? "n/a", meta.budget.map(String.init) ?? "n/a",
                       meta.ms, meta.backend, meta.error.map { ", error " + $0 } ?? "",
                       text.replacingOccurrences(of: "\n", with: " | ")))
            let toolMsg = ChatMessage(role: .tool, content: text)
            let next = ChatMessage(role: .assistant, content: "")
            var tm = MessageMeta()
            tm.tool = meta
            tm.reason = meta.error == nil ? "ok" : "error"
            updateConv(convID) { c in
                c.messages.append(toolMsg)
                c.meta[toolMsg.id] = tm
                c.messages.append(next)
                c.updated = Date()
            }
            persist(convID)
            toolRunningID = nil
            toolRunningQuery = nil
            if userStopped { break }
            aid = next.id
            streamingID = aid
            engineHistory = engineMessages(conversation(convID)?.messages ?? [], tools: tools)
        }
        // the answer after tool rounds must have text (docs/TOOLS.md "Empty final answer"): an answer turn that is empty
        // or only another (unrun) tool call gets one more generation, without tools, after "Answer now with what you
        // have."; if that is empty too, the message says what the searches found
        if tools, !searches.isEmpty, !userStopped, rounds.last?.error == nil, answerIsEmpty(convID, aid, think: think) {
            log("TOOL reply=\(replyNo) the answer after \(searches.count) tool rounds is empty (\(rounds.last?.tokens ?? 0) tokens, reason \(rounds.last?.reason ?? "-")): re-prompting once without tools")
            let unrun = completedCall(convID: convID, aid: aid, think: think)
            let nudge = "Error: no more tools. Answer now with what you have."
            let toolMsg = ChatMessage(role: .tool, content: nudge)
            let next = ChatMessage(role: .assistant, content: "")
            var tm = MessageMeta()
            tm.tool = ToolMeta(name: unrun?.0.name ?? "web_search", query: unrun?.0.query ?? "", backend: "-",
                               resultTokens: engine.toolResultTokens(nudge), error: "limit")
            tm.reason = "error"
            updateConv(convID) { c in
                if let i = c.messages.firstIndex(where: { $0.id == aid }) {
                    if let (call, kept) = unrun {  // the unrun call stays as the call turn it was generated as
                        c.messages[i].content = kept
                        c.meta[aid]?.toolCall = call
                        c.meta[aid]?.reason = "tool_call"
                    } else {  // no answer text (empty, or a think block only): the nudge follows the tool result directly
                        c.messages.remove(at: i); c.meta[aid] = nil
                    }
                }
                c.messages.append(toolMsg)
                c.meta[toolMsg.id] = tm
                c.messages.append(next)
                c.updated = Date()
            }
            persist(convID)
            aid = next.id
            streamingID = aid
            engineHistory = engineMessages(conversation(convID)?.messages ?? [], tools: tools)
            let rec = await runRound(prompt: prompt, convID: convID, aid: aid, history: engineHistory, think: think, mx: mx,
                                     cancelAfter: cancelAfter, continuing: true, tools: tools, toolsAllowed: false)
            rounds.append(rec)
            if !userStopped, answerIsEmpty(convID, aid, think: think) {
                let found = searches.filter { $0.error == nil }.flatMap(\.results)
                let text = found.isEmpty ? "I couldn't find an answer: the searches gave nothing usable."
                    : "I couldn't write an answer. What the search found:\n\n" + found.prefix(4).map {
                        "- **\($0.title)** (\($0.host)): \(Passages.cut($0.snippet, words: 40))"
                    }.joined(separator: "\n")
                updateConv(convID) { c in
                    if let i = c.messages.firstIndex(where: { $0.id == aid }) { c.messages[i].content = text }
                }
                persist(convID)
                log("TOOL reply=\(replyNo) the re-prompted answer is empty too: showing the search results (\(found.count) sources)")
            } else {
                log("TOOL reply=\(replyNo) re-prompted answer: \(rec.tokens) tokens, reason \(rec.reason)")
            }
        }
        // an empty last assistant message (stopped during a search) is removed
        updateConv(convID) { c in
            if let i = c.messages.lastIndex(where: { $0.id == aid }), c.messages[i].content.isEmpty, c.messages[i].role == .assistant,
               i > 0, c.messages[i - 1].role == .tool { c.messages.remove(at: i); c.meta[aid] = nil }
        }
        if !searches.isEmpty { logToolTrip(rounds: rounds, searches: searches, total: Date().timeIntervalSince(tSend)); persist(convID) }
        return rounds.last!
    }

    /// The engine's input for a round (docs/TOOLS.md "History"): empty assistant messages dropped; with tools, the system
    /// message carries the template's tools block (ToolSettings.systemContent; one is added when the chat has no system
    /// prompt); a tool-call turn of the current round (after the last user message) gets the generation prompt's empty think
    /// block it was generated after when it has none (thinking off), so the template renders it as the engine fed it and
    /// the next round continues the live state. Earlier rounds' turns are left to the history mode
    /// (PhoneStoreChatEngine.templateMessages).
    func engineMessages(_ msgs: [ChatMessage], tools: Bool) -> [ChatMessage] {
        var out = msgs.filter { !($0.role == .assistant && $0.content.isEmpty) }
        if tools {
            if out.first?.role == .system {
                out[0].content = ToolSettings.systemContent(out[0].content, tools: true)
            } else {
                out.insert(ChatMessage(role: .system, content: ToolSettings.systemContent("", tools: true)), at: 0)
            }
            // the search pre-check's hint on user turns about live facts (deterministic: the same on every render)
            for i in out.indices where out[i].role == .user { out[i].content = SearchHint.engineContent(out[i].content) }
        }
        if let u = out.lastIndex(where: { $0.role == .user }) {
            for i in out.indices where i > u && out[i].role == .assistant && !out[i].content.contains("</think>") {
                out[i].content = "<think>\n\n</think>\n\n" + out[i].content
            }
        }
        return out
    }

    /// The effective system content of a chat (what the warm-up caches): the tools block when tools are on.
    func effectiveSystem(_ c: Conversation) -> String {
        ToolSettings.systemContent(c.systemPrompt, tools: c.toolsOn)
    }

    /// The last successful web_search of the conversation: its results as the model was shown them ([1] first) and its query.
    private func lastSearch(_ convID: UUID) -> ([SearchResult], String?) {
        guard let c = conversation(convID) else { return ([], nil) }
        for m in c.messages.reversed() where m.role == .tool {
            if let t = c.meta[m.id]?.tool, t.name == "web_search", t.error == nil, !t.results.isEmpty { return (t.results, t.query) }
        }
        return ([], nil)
    }

    /// Message `aid` has no answer text: empty, only a think block, or only a tool call (one the round did not run).
    private func answerIsEmpty(_ convID: UUID, _ aid: UUID, think: Bool) -> Bool {
        guard let content = conversation(convID)?.messages.first(where: { $0.id == aid })?.content else { return true }
        var answer = splitThinking(content, thinkingOn: think).1
        if let sp = ToolCall.split(answer) { answer = sp.before }
        return answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// A complete tool call at the end of message `aid`'s answer: the call and the message text cut after `</tool_call>`.
    private func completedCall(convID: UUID, aid: UUID, think: Bool) -> (ToolCall, String)? {
        guard let content = conversation(convID)?.messages.first(where: { $0.id == aid })?.content else { return nil }
        let answer = splitThinking(content, thinkingOn: think).1
        guard let sp = ToolCall.split(answer), sp.complete, let call = ToolCall.parse(sp.call),
              let close = content.range(of: ToolPrompt.callClose, options: .backwards) else { return nil }
        return (call, String(content[..<close.upperBound]))
    }

    /// The round trip's cost split (console `TOOLTRIP`, docs/TOOLS.md): the tool-call round (prefill, first token, the call's
    /// tokens), the search, the result prefill of the continuation (live prefix: only the result turn is new) and its first
    /// token and decode.
    private func logToolTrip(rounds: [ReplyRecord], searches: [ToolMeta], total: Double) {
        func f(_ v: Double?, _ k: Double = 1000) -> String { v.map { String(format: "%.2f", $0 / k) } ?? "n/a" }
        for (i, r) in rounds.enumerated() {
            let p = r.prefill ?? [:]
            let decode = r.seconds
            log("TOOLTRIP reply=\(replyNo) round=\(i + 1)/\(rounds.count) kind=\(r.prefillKind ?? "n/a") prompt_tokens=\(r.promptTokens ?? 0) reused=\((r.promptTokens ?? 0) - Int(p["prefill_tokens"] ?? 0)) prefilled=\(Int(p["prefill_tokens"] ?? 0)) prefill_s=\(f(p["prefill_s"], 1)) prefill_mb=\(f(p["prefill_mb_read"], 1)) phone_read_s=\(f(p["prefill_mb_read"], 750)) restore_s=\(f(p["prefix_restore_ms"])) first_token_s=\(f(r.firstTokenMs)) generated=\(r.tokens) decode_s=\(String(format: "%.2f", decode)) wall_s=\(f(r.wallMs)) reason=\(r.reason)")
        }
        for (i, m) in searches.enumerated() {
            log("TOOLTRIP reply=\(replyNo) search=\(i + 1) tool=\(m.name) query=\"\(m.query)\" backend=\(m.backend) results=\(m.results.count) pages=\(m.pages ?? 0) pages_s=\(f(m.pagesMs)) result_tokens=\(m.resultTokens.map(String.init) ?? "n/a") budget=\(m.budget.map(String.init) ?? "n/a") search_s=\(f(m.ms))\(m.error.map { " error=" + $0 } ?? "")")
        }
        log(String(format: "TOOLTRIP reply=%d total_s=%.2f (send to the last token)", replyNo, total))
    }

    /// One generation round of a reply: stream into assistant message `aid`, then the caption, persistence and the logs.
    /// With `toolsAllowed`, a complete `<tool_call>…</tool_call>` in the answer stops the round (the model normally ends its
    /// turn right after it; the stop is the safeguard).
    private func runRound(prompt: String, convID: UUID, aid: UUID, history: [ChatMessage], think: Bool, mx: Int,
                          cancelAfter: Int?, continuing: Bool, tools: Bool, toolsAllowed: Bool) async -> ReplyRecord {
        replyNo += 1
        replyTokens = 0
        inReply = true
        let warnings0 = memoryWarnings
        defer { inReply = false }
        liveTokens = 0
        thinkStart = nil
        liveThinkingSeconds = nil
        var toolStopSent = false
        prefill = expectedPrefill(history: history, convID: convID, waiting: false)
        if interactive { log("INDICATOR reply=\(replyNo) dots shown" + (think ? " as the thinking block's header" : " alone") + "; no prefill card") }

        replyTokPerSec = 0
        mbPerToken = 0
        replyTtftMs = nil
        var rec = ReplyRecord(prompt: prompt, thinking: think, tokens: 0, reason: StopReason.cancelled.rawValue,
                              seconds: 0, meanTokPerSec: 0, meanMBPerToken: 0, firstTokenMs: nil,
                              cancelRequestedAtToken: nil, text: "", error: nil, maxNew: mx)
        let t0 = Date()
        var n = 0, mbSum = 0.0
        var finished = false
        var lastTick: PrefillTick? = nil
        do {
            for try await ev in engine.generate(messages: history, maxNew: mx) {
                switch ev {
                case .prefill(let p):
                    // the engine's own prefill progress replaces the estimate (the card and the status line read it)
                    if var pp = prefill {
                        pp.tick = p
                        pp.tickAt = Date()
                        prefill = pp
                    }
                    lastTick = p
                    if p.done == 0 || p.done == p.total || p.done % 10 == 0 {
                        log(String(format: "PROGRESS reply=%d %@ done=%d/%d reused=%d read=%.1f MB reland=%.1f MB prefill=%.2f s elapsed=%.2f s",
                                   replyNo, p.kind, p.done, p.total, p.reusedTokens, p.mbRead, p.relandMB, p.prefillElapsed, p.elapsed))
                    }
                case .text(let piece, let st):
                    if rec.firstTokenMs == nil {
                        rec.firstTokenMs = Date().timeIntervalSince(t0) * 1000
                        replyTtftMs = rec.firstTokenMs
                        if interactive {
                            log(String(format: "INDICATOR reply=%d first token at %.1f s: %@", replyNo, (rec.firstTokenMs ?? 0) / 1000,
                                       think ? "\"Thinking… N s\" label beside the dots" : "dots replaced by the streamed text"))
                        }
                        prefill = nil
                        if think { thinkStart = Date() }
                    }
                    var now = ""
                    updateConv(convID) { c in
                        if let i = c.messages.lastIndex(where: { $0.id == aid }) { c.messages[i].content += piece; now = c.messages[i].content }
                    }
                    // a complete tool call in the answer ends the round (by string: </tool_call> is one token, 248059)
                    if toolsAllowed, !toolStopSent, piece.contains("tool_call>"), st.phase == nil || st.phase == "answer",
                       ToolCall.split(splitThinking(now, thinkingOn: think).1)?.complete == true {
                        toolStopSent = true
                        log("TOOL reply=\(replyNo) </tool_call> at token \(n + 1): stopping the round")
                        engine.cancel()
                    }
                    if st.phase == "close" || st.phase == "answer" || (st.phase == nil && piece.contains("</think>")),
                       let s = thinkStart, liveThinkingSeconds == nil {
                        liveThinkingSeconds = Date().timeIntervalSince(s)
                    }
                    n += 1
                    replyTokens = n
                    liveTokens = n
                    if n == 1 || n % 10 == 0 { memSample(n == 1 ? "reply \(replyNo) first token" : "reply \(replyNo) token \(n)") }
                    mbSum += st.mbRead
                    lastTokPerSec = st.tokPerSec
                    // decode rate: tokens over the engine's decode clock (prefill excluded), as the engine's split rate
                    replyTokPerSec = st.decodeSeconds > 0 ? Double(st.index + 1) / st.decodeSeconds : 0
                    mbPerToken = st.mbTotal > 0 ? st.mbTotal / Double(st.index + 1) : mbSum / Double(n)
                    inMemory = st.inMemory
                    expertsInMemory = st.expertsInMemory
                    if let c = cancelAfter, n == c, rec.cancelRequestedAtToken == nil {
                        rec.cancelRequestedAtToken = n
                        stop()
                    }
                case .finished(let s):
                    finished = true
                    lastSummary = s
                    rec.reason = s.reason.rawValue
                    rec.tokens = s.tokens
                    rec.seconds = s.seconds
                    rec.meanTokPerSec = s.meanTokPerSec
                    rec.meanMBPerToken = s.meanMBPerToken
                }
            }
        } catch {
            rec.error = error.localizedDescription
            event("generate error: \(error.localizedDescription)")
        }
        if !finished {
            rec.tokens = n
            rec.seconds = Date().timeIntervalSince(t0)
            rec.meanTokPerSec = rec.seconds > 0 ? Double(n) / rec.seconds : 0
            rec.meanMBPerToken = n > 0 ? mbSum / Double(n) : 0
        }
        rec.text = conversation(convID)?.messages.last(where: { $0.id == aid })?.content ?? ""
        rec.wallMs = Date().timeIntervalSince(t0) * 1000
        rec.memoryWarnings = memoryWarnings - warnings0
        defer { lastReply = rec }
        let detail = finished ? engine.lastDetail : nil
        if let d = detail {
            rec.ids = d.ids
            rec.promptTokens = d.promptIds.count
            rec.split = d.split
            rec.prefill = d.prefillStats
            rec.prefillKind = d.prefill
            lastSplit = d.split
            lastPrefill = d.prefillStats
            lastPrefillKind = d.prefill
            rec.engineTokPerSec = d.tokPerSec
            // the progress events against the engine's final prefill figures (what the caption shows)
            if let t = lastTick, let ps = d.prefillStats {
                let mb = ps["prefill_mb_read"] ?? -1
                log(String(format: "PROGRESS check reply=%d: last event %d/%d tokens, %.1f MB, %.2f s; engine prefilled %d tokens (%@), %.1f MB, %.2f s; tokens %@, kind %@",
                           replyNo, t.done, t.total, t.mbRead, t.prefillElapsed, d.prefillTokens, d.prefill, mb, ps["prefill_s"] ?? -1,
                           t.done == d.prefillTokens && t.total == d.prefillTokens ? "equal" : "DIFFER", t.kind == d.prefill ? "equal" : "DIFFER"))
            }
        }
        // what the UI shows at the end of the reply (the stats bar, the settings sheet), logged next to the engine's
        lastTtftMs = rec.firstTokenMs
        if n > 0 {
            rec.uiDecodeTokPerSec = replyTokPerSec
            rec.uiMBPerToken = mbPerToken
            rec.uiTtftMs = rec.firstTokenMs
            if let e = rec.engineTokPerSec, e > 0 { rec.uiRateDiffPct = 100 * (replyTokPerSec - e) / e }
        }
        // the message's caption and the conversation's token count, saved with it
        var meta = MessageMeta(tokens: rec.tokens, reason: rec.error != nil ? "error" : rec.reason)
        meta.decodeTokPerSec = n > 0 ? replyTokPerSec : nil
        meta.ttftMs = rec.firstTokenMs
        meta.mbPerToken = n > 0 ? mbPerToken : nil
        meta.error = rec.error
        meta.thinking = think
        if think, let s = thinkStart { meta.thinkingSeconds = liveThinkingSeconds ?? Date().timeIntervalSince(s) }
        if let d = detail {
            meta.prefillKind = d.prefill
            meta.promptTokens = d.promptIds.count
            meta.prefilledTokens = d.prefillTokens
            meta.reusedTokens = d.promptIds.count - d.prefillTokens
            meta.restoreMs = d.prefillStats?["prefix_restore_ms"]
        }
        updateConv(convID) { c in
            c.meta[aid] = meta
            c.updated = Date()
            if let d = detail { c.contextTokens = d.promptIds.count + d.ids.count }
            else { c.contextTokens = c.messages.map { estimateTokens($0.content) }.reduce(0, +) }  // the stub: an estimate
        }
        persist(convID)
        if let d = detail, !engine.info.isStub {
            if continuing { UserDefaults.standard.set(d.prefill == "live prefix", forKey: AppSettings.liveContinuationKey) }
            liveConvID = convID
            learnTiming(d)
        }
        if interactive {
            log("TEXT reply=\(replyNo) " + rec.text.debugDescription)
            log("CAPTION reply=\(replyNo) " + meta.caption + (continuing ? " (continued conversation)" : " (first reply of this launch in it)"))
        }
        prefixStatus = engine.prefixCacheStatus
        log(String(format: "REPLY tokens=%d reason=%@ %.2f tok/s %.1f MB/token", rec.tokens, rec.reason,
                   rec.meanTokPerSec, rec.meanMBPerToken))
        log(String(format: "UI reply=%d decode_tok_s=%.3f engine_tok_s=%.3f diff=%@ ttft_ms=%.0f (engine %@) mb_per_token=%.1f (engine %.1f) prefill=%@",
                   replyNo, rec.uiDecodeTokPerSec ?? 0, rec.engineTokPerSec ?? 0,
                   rec.uiRateDiffPct.map { String(format: "%.2f%%", $0) } ?? "n/a", rec.uiTtftMs ?? 0,
                   rec.prefill?["ttft_ms"].map { String(format: "%.0f", $0) } ?? "n/a", rec.uiMBPerToken ?? 0, rec.meanMBPerToken,
                   rec.prefillKind ?? "n/a"))
        if let p = rec.prefill, p["prefix_hit"] != nil { log("PREFIX reply=\(replyNo) " + Self.prefixLine(p)) }
        if let p = rec.prefill { log("PREFILL reply=\(replyNo) " + Self.prefillLine(p)) }
        if let s = rec.split { log("SPLIT reply=\(replyNo) " + Self.splitLine(s)) }
        memSample("end of reply \(replyNo) (\(rec.tokens) tokens, \(rec.reason))")
        return rec
    }

    /// The CLI's per-token split on one line (phone-store-run bench-JSON keys).
    static func splitLine(_ s: [String: Double]) -> String {
        func v(_ k: String, _ f: String = "%.1f") -> String { s[k].map { String(format: f, $0) } ?? "n/a" }
        return "tokens=\(v("tokens", "%.0f")) tok/s=\(v("tok_per_s", "%.3f")) ms/token=\(v("ms_per_token"))"
            + " encode=\(v("host_encode_ms")) blocked_on_reads=\(v("expert_wait_ms")) (name \(v("name_wait_ms")) land \(v("land_wait_ms")))"
            + " sync_wait=\(v("sync_wait_ms")) gpu=\(v("gpu_ms_per_token")) ms gpu_busy=\(s["gpu_busy_frac"].map { String(format: "%.1f%%", 100 * $0) } ?? "n/a")"
            + " drive=\(v("drive_gb_per_s", "%.3f")) GB/s busy=\(v("drive_gb_per_s_busy", "%.3f")) GB/s (\(s["drive_busy_frac"].map { String(format: "%.0f%%", 100 * $0) } ?? "n/a") of the time)"
            + " MiB/token=\(v("mb_read_per_token")) in_memory=\(v("released_topk_in_memory", "%.4f")) sync_reads=\(v("sync_reads", "%.0f"))"
            + " records/token=\(v("reads_per_token", "%.2f")) reads_in_flight=\(v("reads_in_flight_mean", "%.2f"))"
    }

    static func prefixLine(_ p: [String: Double]) -> String {
        func v(_ k: String, _ f: String = "%.1f") -> String { p[k].map { String(format: f, $0) } ?? "n/a" }
        return "hit=\(v("prefix_hit", "%.0f")) restored_tokens=\(v("prefix_tokens_restored", "%.0f")) restore=\(v("prefix_restore_ms", "%.0f")) ms"
            + " (load \(v("prefix_load_ms", "%.0f")) ms, state \(v("prefix_state_mb")) MB; re-land \(v("prefix_reland_ms", "%.0f")) ms,"
            + " \(v("prefix_reland_mb")) MB read, \(v("prefix_relanded", "%.0f")) read / \(v("prefix_kept", "%.0f")) kept / \(v("prefix_moved", "%.0f")) moved)"
            + " saved=\(v("prefix_saved", "%.0f")) save=\(v("prefix_save_ms", "%.0f")) ms (\(v("prefix_save_mb")) MB)"
            + " cache=\(v("prefix_cache_entries", "%.0f")) states \(v("prefix_cache_mb", "%.0f")) MB"
    }

    static func prefillLine(_ p: [String: Double]) -> String {
        func v(_ k: String, _ f: String = "%.1f") -> String { p[k].map { String(format: f, $0) } ?? "n/a" }
        return "tokens=\(v("prefill_tokens", "%.0f")) seconds=\(v("prefill_s", "%.2f")) read=\(v("prefill_mb_read")) MB"
            + " records=\(v("prefill_reads", "%.0f")) drive=\(v("prefill_drive_gb_per_s", "%.3f")) GB/s (busy \(v("prefill_drive_gb_per_s_busy", "%.3f")))"
            + " blocked_on_reads=\(v("prefill_expert_wait_ms", "%.0f")) ms gpu=\(v("prefill_gpu_ms", "%.0f")) ms encode=\(v("prefill_encode_ms", "%.0f")) ms"
            + " template=\(v("template_ms")) ms first_token=\(v("first_token_ms")) ms"
    }

    // MARK: scene phase

    func phaseChanged(_ p: ScenePhase) {
        Lifecycle.phase(p, detail: "engine \(engineState.label), slabs \(engine.nativeEngine?.memoryReleased == true ? "released" : "allocated")"
                        + (p == .active ? backgroundSince.map { String(format: ", back after %.1f s in the background", Date().timeIntervalSince($0)) } ?? "" : ""))
        #if targetEnvironment(macCatalyst)
        // On the Mac the GPU keeps running in the background; the stop is an iOS rule.
        if p == .background && isGenerating { event("scene -> background during generation (Mac: not stopped)") }
        #else
        if p == .background && isGenerating {
            // iOS suspends GPU work in the background (G4 plan): stop after the current token.
            event("scene -> background during generation; stopping")
            stop()
        }
        #endif
        if p == .background {
            backgroundSince = Date()
            if Self.backgroundRelease { releaseForBackground() }
        } else if p == .active, backgroundSince != nil {
            backgroundSince = nil
            restoreAfterBackground()
        }
    }

    /// `-backgroundRelease 0` keeps the slabs in the background (before/after runs); default on.
    static var backgroundRelease: Bool { LaunchArgs.value("-backgroundRelease").map { $0 != "0" && $0 != "off" } ?? true }

    /// In the background: free the expert slabs and MLX's buffer cache (PhoneStoreEngine.releaseMemory, after the reply
    /// that was stopped and its reply-end save, which is skipped: no GPU work in the background), so iOS keeps the
    /// process (~1.5 GB instead of ~3 GB) and a return finds the weights and the conversation's state in memory. A short
    /// UIKit background task covers it (CPU only: waiting for the engine thread, freeing buffers).
    func releaseForBackground() {
        guard engineState == .ready, !engine.info.isStub, let e = engine.nativeEngine, !e.memoryReleased else { return }
        e.prefixSaveAtReplyEnd = false  // the stopped reply's end save would run GPU work in the background
        #if !targetEnvironment(macCatalyst)
        var bg = UIBackgroundTaskIdentifier.invalid
        bg = UIApplication.shared.beginBackgroundTask(withName: "release slabs") {
            UIApplication.shared.endBackgroundTask(bg)
            bg = .invalid
        }
        #endif
        let before = memSample("background: before release")
        let prev = releaseTask
        releaseTask = Task { @MainActor in
            await prev?.value
            await self.genTask?.value
            let t0 = Date()
            let b = await e.releaseMemory()
            let after = self.memSample("background: after release")
            Lifecycle.note(String(format: "BACKGROUND released %.2f GB of slabs in %.2f s: footprint %.3f -> %.3f GB, available %.3f -> %.3f GB",
                                  Double(b) / 1e9, Date().timeIntervalSince(t0), Double(before.physFootprint) / 1e9,
                                  Double(after.physFootprint) / 1e9, Double(before.osProcAvailable) / 1e9, Double(after.osProcAvailable) / 1e9))
            #if !targetEnvironment(macCatalyst)
            if bg != .invalid { UIApplication.shared.endBackgroundTask(bg); bg = .invalid }
            #endif
        }
    }

    /// Back in the foreground: the slabs allocated again and the held records read back in, in the background of the chat
    /// (a message sent meanwhile queues behind it on the engine thread; its chunk would read them anyway).
    func restoreAfterBackground() {
        guard engineState == .ready, let e = engine.nativeEngine else { return }
        let prev = releaseTask
        releaseTask = Task { @MainActor in
            await prev?.value
            guard e.memoryReleased else { return }
            let t0 = Date()
            do {
                let r = try await e.restoreMemory(land: Self.launchLand != "lazy")
                let m = self.memSample("foreground: slabs back")
                Lifecycle.note(String(format: "FOREGROUND slabs allocated again, %d records re-landed in %.2f s (%.2f s after the return): footprint %.3f GB",
                                      r.records, r.seconds, Date().timeIntervalSince(t0), Double(m.physFootprint) / 1e9))
            } catch {
                Lifecycle.note("FOREGROUND slabs: \(error.localizedDescription)")
            }
        }
    }

    /// The test flows' emptying of the prefix cache: all but the one-time setup's kept system states (no test prompt has a
    /// system message, so they cannot be hit; the app stays ready after a test launch).
    func clearForTest() async {
        if let pe = engine as? PhoneStoreChatEngine { await pe.clearPrefixCache(keepSetup: true) } else { await engine.clearPrefixCache() }
    }

    // MARK: autotest (-autotest): no screen interaction needed

    /// The G4 chat prompt (docs/g4_runs, native/ref): 41 tokens with the template's default (thinking on).
    static let g4Prompt = "Explain how a refrigerator keeps food cold, step by step, for a curious ten-year-old. Then list three ways to make it use less electricity."

    /// Identity against a reference run: a JSON with `ids` (and `slab`, the cap it ran at). `-ref PATH` names it (an
    /// absolute path, e.g. docs/g4_runs/b_py_pf48_r1.json on the Mac; or a file name looked up in Documents/ and then in
    /// the app bundle); `-ref none` skips the check. Without `-ref` the app uses the bundled reference for the session's
    /// cap, `ref_pf<cap>_slim.json` (the CLI's slim-layout runs: ref_pf14_slim.json at cap 14, the phone's budget cap, and
    /// ref_pf10_slim.json at cap 10), since the greedy ids depend on the cap (cap 14 and cap 48 diverge at token 95,
    /// cap 10 and cap 14 at token 101). Any other cap without `-ref`: no check. Only with the G4 prompt (no `-prompt`).
    /// Prompt ids: `-refPromptIds` JSON with `prompt_ids` (e.g. b_sw_pf48_r1.json), else the reference's own `prompt_ids`.
    static let bundledRefCaps = [10, 14]
    /// Per-layer caps (Probe: the auto allocation, 2026-11-04): the reference is keyed by the allocation, `auto<total>`
    /// (ref_auto560_slim.json: the CLI's `--layer-caps auto:560` run, the phone's budget at cap 14), else `pf<cap>`.
    static let bundledRefKeys = ["pf10", "pf14", "auto560"]
    var refKey: String {
        guard let p = probe, p.layerCapsSpec.hasPrefix("auto") else { return "pf\(cap)" }
        return "auto\(p.slotsTotal)"
    }
    static func bundledRef(key: String) -> String { "ref_\(key)_slim.json" }
    static func bundledRef(cap: Int) -> String { "ref_pf\(cap)_slim.json" }
    /// Exact mode (`-mode exact`): the CLI's exact-mode greedy run (slim layout, 300 tokens of the G4 prompt). Exact-mode
    /// ids do not depend on the cap (the Mac's cap-10 and cap-14 exact runs are identical), so it applies at any cap.
    static let bundledExactRef = "ref_exact_slim.json"
    /// Exact-mode teacher forcing feeds these ids (the ones tf_mac_exact_slim.json was made with), at any cap.
    static let exactTFFedRef = "ref_pf10_slim.json"

    /// A JSON by name: an absolute path, else Documents/NAME, else the app bundle's NAME; nil (with the places looked) if none.
    func locateJSON(_ name: String) -> (path: String, json: [String: Any])? {
        var candidates: [String] = []
        if name.hasPrefix("/") {
            candidates.append(name)
        } else {
            candidates.append(docsURL.appendingPathComponent(name).path)
            if let b = Bundle.main.url(forResource: name, withExtension: nil) { candidates.append(b.path) }
        }
        if let hit = candidates.lazy.compactMap({ c in Self.readJSON(c).map { (c, $0) } }).first { return hit }
        log("reference \(name) not found (looked in: \(candidates.joined(separator: ", ")))")
        return nil
    }

    struct Identity: Encodable {
        var ref: String
        var refCap: Int?
        var refPromptIds: String?
        var promptIdsEqual: Bool?
        var refTokens: Int
        var tokens: Int
        var maxNew: Int
        var comparedTokens: Int        // min(tokens, refTokens)
        var idsEqual: Bool             // every reference token reproduced (tokens == refTokens, no divergence)
        var prefixEqual: Bool          // no divergence over the compared tokens (a reply cut short, or -maxNew < 300)
        var firstDivergence: Int?      // index of the first generated token that differs (nil: none in the common length)
        var refTokAtDivergence: Int?
        var appTokAtDivergence: Int?
    }

    static func readJSON(_ path: String) -> [String: Any]? {
        guard let d = FileManager.default.contents(atPath: path) else { return nil }
        return (try? JSONSerialization.jsonObject(with: d)) as? [String: Any]
    }

    /// The identity reference as the doc comment on `bundledRef` describes; nil (with a log line saying why) when none applies.
    func identityReference() -> (path: String, json: [String: Any])? {
        let arg = LaunchArgs.value("-ref")
        if arg == "none" { return nil }
        if EngineMode.isExact {  // exact mode: the exact-mode reference at any cap (or -ref)
            guard let (path, json) = locateJSON(arg ?? Self.bundledExactRef) else { return nil }
            if let m = json["mode"] as? String, m != "exact" {
                log("IDENTITY note: the reference ran in \(m) mode, this session runs in exact mode; the ids may legitimately differ")
            }
            return (path, json)
        }
        if arg == nil && !Self.bundledRefKeys.contains(refKey) {
            log("IDENTITY skipped: no bundled reference for \(refKey) (bundled: \(Self.bundledRefKeys.joined(separator: ", ")); -ref FILE compares anyway)")
            return nil
        }
        let name = arg ?? Self.bundledRef(key: refKey)
        var candidates: [String] = []
        if name.hasPrefix("/") {
            candidates.append(name)
        } else {
            candidates.append(docsURL.appendingPathComponent(name).path)
            if let b = Bundle.main.url(forResource: name, withExtension: nil) { candidates.append(b.path) }
        }
        guard let (path, json) = candidates.lazy.compactMap({ c in Self.readJSON(c).map { (c, $0) } }).first else {
            log("IDENTITY reference \(name) not found (looked in: \(candidates.joined(separator: ", ")))")
            return nil
        }
        let refCaps = (json["slab_list"] as? [Int]) ?? (json["slab"] as? Int).map { Array(repeating: $0, count: 40) }
        if let rc = refCaps, let mine = probe?.layerCaps, rc != mine {
            if arg == nil {
                log("IDENTITY skipped: the bundled reference's slots per layer differ from this session's (\(probe?.layerCapsSpec ?? "?"))")
                return nil
            }
            log("IDENTITY note: the reference ran with other slots per layer than this session (\(probe?.layerCapsSpec ?? "?")); the ids may legitimately differ")
        } else if let slab = json["slab"] as? Int, slab != cap {
            if arg == nil {
                log("IDENTITY skipped: the bundled reference ran at cap \(slab), this session's cap is \(cap) (-ref \(name) compares anyway)")
                return nil
            }
            log("IDENTITY note: the reference ran at cap \(slab), this session's cap is \(cap); the ids may legitimately differ")
        }
        return (path, json)
    }

    struct AutotestOut: Encodable {
        var time: String
        var complete: Bool                // false: written after a reply, the autotest was still running
        var engine: String
        var engineChoice: String
        var engineDetail: String
        var engineState: String
        var engineSettings: [String: String]
        var loadLog: [String]
        var mode: String                  // -mode (exact: diagnostics), default prefetch
        var cap: Int
        var capSource: String
        var osProcAvailableMemoryBytes: Int
        var storeComplete: Bool
        var storeProblemCount: Int
        var realEngineLoadWouldBeRefused: String?
        var prompt: String
        var maxNew: Int
        var diagnostics: [String: String]
        var memWarnMode: String
        var memoryWarnings: Int
        var replies: [ReplyRecord]
        var details: [ReplyDetail?]
        var identity: Identity?
        var checks: [String: Bool]
        var pass: Bool
        var seconds: Double
        var events: [String]
        var memory: [MemSample]
    }

    func writeAutotest(complete: Bool, prompt: String, maxNew: Int, replies: [ReplyRecord], details: [ReplyDetail?],
                       identity: Identity?, checks: [String: Bool], pass: Bool, started: Date) {
        // What a store-backed engine would do with this store right now.
        let wouldRefuse: String? = {
            guard let p = probe else { return "no probe" }
            if !p.store.complete { return EngineError.storeIncomplete(p.store.problems).localizedDescription }
            if p.cap < Budget.minCap { return EngineError.capTooSmall(p.cap, Budget.minCap).localizedDescription }
            return nil
        }()
        let out = AutotestOut(time: ISO8601DateFormatter().string(from: Date()), complete: complete, engine: engine.info.name,
                              engineChoice: engineChoice, engineDetail: engine.info.detail, engineState: engineState.label,
                              engineSettings: Dictionary(engine.settings, uniquingKeysWith: { a, _ in a }), loadLog: loadLog,
                              mode: engine.info.isStub ? "stub" : (EngineMode.arg ?? "prefetch"), cap: cap,
                              capSource: probe?.capSource ?? "?",
                              osProcAvailableMemoryBytes: probe?.osProcAvailableMemoryBytes ?? -1,
                              storeComplete: storeComplete, storeProblemCount: probe?.store.problems.count ?? -1,
                              realEngineLoadWouldBeRefused: wouldRefuse, prompt: prompt, maxNew: maxNew, diagnostics: diagnostics,
                              memWarnMode: memWarnMode.rawValue, memoryWarnings: memoryWarnings, replies: replies, details: details,
                              identity: identity, checks: checks, pass: pass, seconds: Date().timeIntervalSince(started),
                              events: events, memory: memSamples)
        do {
            try jsonEncoder().encode(out).write(to: autotestURL, options: .atomic)
        } catch {
            log("ERROR writing autotest.json: \(error)")
        }
    }

    // MARK: teacher forcing (-teacherForce): numerical agreement with the Mac

    /// The Mac's teacher-forced run of the same reference (`tf_mac_pf<cap>_slim.json`, phone-store-run --teacher-force:
    /// top-32 (id, logit) per position and the released experts per step), bundled; `-tfRef NAME|PATH|none` overrides.
    /// Exact mode: `tf_mac_exact_slim.json` (the CLI's exact-mode teacher-forced run of ref_pf10_slim.json's ids; identical
    /// at caps 10 and 14), at any cap.
    static func bundledTFRef(key: String) -> String { EngineMode.isExact ? "tf_mac_exact_slim.json" : "tf_mac_\(key)_slim.json" }
    /// Documents/tf-<cap>.json (prefetch; tf-auto<total>.json with per-layer caps) or Documents/tf-exact-<cap>.json (-mode exact).
    func tfURL() -> URL {
        docsURL.appendingPathComponent(EngineMode.isExact ? "tf-exact-\(cap).json" : (refKey.hasPrefix("auto") ? "tf-\(refKey).json" : "tf-\(cap).json"))
    }

    func tfReferenceLogits(ids: [Int]) -> (path: String, ref: TFReference)? {
        let arg = LaunchArgs.value("-tfRef")
        if arg == "none" { log("TF reference logits: none (-tfRef none)"); return nil }
        let name = arg ?? Self.bundledTFRef(key: refKey)
        var candidates: [String] = []
        if name.hasPrefix("/") {
            candidates.append(name)
        } else {
            candidates.append(docsURL.appendingPathComponent(name).path)
            if let b = Bundle.main.url(forResource: name, withExtension: nil) { candidates.append(b.path) }
        }
        guard let (path, json) = candidates.lazy.compactMap({ c in Self.readJSON(c).map { (c, $0) } }).first else {
            log("TF reference logits \(name) not found (looked in: \(candidates.joined(separator: ", "))); no comparison")
            return nil
        }
        guard (json["ids"] as? [Int]) == ids, let r = tfReference(json) else {
            log("TF reference logits \(path): ids differ from the fed reference or no positions[].top; no comparison")
            return nil
        }
        return (path, r)
    }

    /// `-teacherForce` (with `-cap N`): load at the cap, feed the reference's prompt ids + 300 generated ids (the bundled
    /// ref_pf<cap>_slim.json, or `-ref`) one at a time through the normal prefetch pipeline, record per position the
    /// argmax, whether it equals the fed id, the top-1 minus top-2 margin, the top-32, and against the Mac's top-32
    /// (tf_mac_pf<cap>_slim.json) the max-abs logit difference over those 32 ids and the Mac's margin; per step the
    /// released experts, compared with the Mac's. Writes Documents/tf-<cap>.json; console: `TF ...` summary,
    /// `TF ROUTING ...`, `TF DIFF ...` per position where the argmax differs (from the fed id or the Mac's), `DONE`.
    func runTeacherForce() async {
        log("TF start (cap \(cap), mode \(EngineMode.label), engine \(engineState.label))")
        defer {
            log("DONE")
            if LaunchArgs.has("-exitWhenDone") { exit(0) }
        }
        guard engineState == .ready, let pe = engine as? PhoneStoreChatEngine else {
            log("TF FAIL: engine not ready (\(engineState.label))")
            return
        }
        // the fed ids: prefetch: the reference for the cap (identityReference); exact: ref_pf10_slim.json at any cap (the ids
        // tf_mac_exact_slim.json was made with); -ref FILE replaces either
        let fed: (path: String, json: [String: Any])? = EngineMode.isExact
            ? (LaunchArgs.value("-ref") == "none" ? nil : locateJSON(LaunchArgs.value("-ref") ?? Self.exactTFFedRef))
            : identityReference()
        guard let (refPath, ref) = fed, let pids = ref["prompt_ids"] as? [Int], let rids = ref["ids"] as? [Int] else {
            log("TF FAIL: no reference ids for cap \(cap), mode \(EngineMode.arg ?? "prefetch") (bundled: caps \(Self.bundledRefCaps), exact: \(Self.exactTFFedRef); -ref FILE)")
            return
        }
        let tfRef = tfReferenceLogits(ids: rids)
        log("TF feeding \(pids.count) prompt + \(rids.count) reference ids from \((refPath as NSString).lastPathComponent); reference logits: \(tfRef.map { ($0.path as NSString).lastPathComponent } ?? "none")")
        memSample("tf start")
        let warnings0 = memoryWarnings
        do {
            let r = try await pe.teacherForce(promptIds: pids, ids: rids, ref: tfRef?.ref) { p in
                if p.i % 50 == 0 || !p.match {
                    log(String(format: "TF pos %d fed %d argmax %d margin %.4f%@", p.i, p.refNext, p.argmax, p.margin,
                               p.refMaxAbs.map { String(format: " max_abs(top32) %.4f", $0) } ?? ""))
                }
            }
            memSample("tf end")
            let meta: [String: Any] = [
                "engine": "PhoneStoreApp", "device": machineName(), "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "slab": cap, "slab_list": probe?.layerCaps ?? [], "layer_caps": probe?.layerCapsSpec ?? "?", "cap_source": probe?.capSource ?? "?", "teacher_force_ref": refPath, "ref_cap": ref["slab"] ?? NSNull(),
                "tf_ref": tfRef.map { $0.path as Any } ?? NSNull(), "engine_settings": Dictionary(engine.settings, uniquingKeysWith: { a, _ in a }),
                "diagnostics": diagnostics, "memory_warnings": memoryWarnings - warnings0,
                "time": ISO8601DateFormatter().string(from: Date()),
            ]
            let rep = tfReport(r, meta: meta)
            let url = tfURL()
            try JSONSerialization.data(withJSONObject: rep, options: [.sortedKeys]).write(to: url, options: .atomic)
            for l in tfConsoleLines(r, label: "cap=\(cap)") { log(l) }
            log("TF wrote \(url.path)")
        } catch {
            log("TF FAIL: \(error.localizedDescription)")
        }
        writeProbe()
    }

    /// The UI's decode rate equals the engine's split rate within 2% for every finished reply of >= 20 tokens.
    static func uiRateOK(_ rs: [ReplyRecord]) -> Bool {
        let checked = rs.filter { $0.tokens >= 20 && $0.engineTokPerSec != nil }
        return !checked.isEmpty && checked.allSatisfy { abs($0.uiRateDiffPct ?? 100) <= 2 }
    }

    // MARK: -chatOnce TEXT: one ordinary reply through send() (the send button's path), the UI's values logged

    struct UIOut: Encodable {
        var time: String
        var prompt: String
        var reply: ReplyRecord?
        var shown: [String: String]   // the stats bar and settings rows as the views format them
        var engineSettings: [String: String]
        var diagnostics: [String: String]
        /// With -chatThen: every reply of the conversation, its caption, and the saved conversation file.
        var replies: [ReplyRecord]?
        var captions: [String]?
        var conversationFile: String?
        /// With -chatDiscard: the conversation itself (its file is deleted, so a test leaves no chat in the list).
        var conversation: ConversationFile? = nil
    }

    /// One conversation through the app's own path: a new conversation (persisted in Documents/conversations/, system
    /// prompt only with `-system TEXT`, which is then warmed into the prefix cache first, as the app does at launch),
    /// `send(TEXT)`, and with `-chatThen TEXT2` a follow-up in the same conversation. Logs `SHOWN <field> = <string>` for
    /// what the views display (the status line and each reply's caption included) and writes Documents/ui.json.
    /// `-chatContinue TEXT` (continueLatest): the same, but TEXT is sent in the most recently updated saved conversation,
    /// after the launch's own warm-up, as a user continuing a chat after a relaunch does (README "Relaunch and continue").
    /// `-chatDiscard`: at the end the conversation is written into ui.json and deleted (no test chat left in the list).
    func chatOnce(_ text: String, continueLatest: Bool = false) async {
        log((continueLatest ? "CHATCONTINUE " : "CHATONCE ") + text)
        guard engineState == .ready else { log("CHATONCE FAIL: engine \(engineState.label)"); log("DONE"); return }
        if continueLatest {
            guard let c = conversations.sorted(by: { $0.updated > $1.updated }).first(where: { $0.hasUserMessage }) else {
                log("CHATCONTINUE FAIL: no saved conversation"); log("DONE"); return
            }
            conv = c
            thinking = c.thinking
            log("CHATCONTINUE conversation \(c.id.uuidString) \"\(c.displayTitle)\": \(c.messages.count) messages, \(c.contextTokens) tokens, thinking \(c.thinking ? "on" : "off")")
        } else {
            thinking = LaunchArgs.value("-thinking").map { $0 != "0" } ?? thinking
            conv = Conversation(system: LaunchArgs.value("-system") ?? "", thinking: thinking)  // as before: no system prompt unless -system
            conv.tools = ToolSettings.forced  // a forced -tools is the chat's own mode (a later -chatContinue renders it the same)
            if LaunchArgs.value("-system") != nil || conv.toolsOn { await warmUp() }  // the system turn (tools block) into the cache
        }
        let mx = Int(LaunchArgs.value("-maxNew") ?? "").flatMap { $0 > 0 ? $0 : nil } ?? maxNew
        var replies: [ReplyRecord] = []
        var captions: [String] = []
        var shown: [String: String] = [:]
        // -chatNew TEXT3: then a new chat (the compose button, newConversation) and TEXT3 sent in it at once, i.e. while
        // its background re-warm runs (queued behind it), or after `-chatNewDelay S` seconds
        let turns = [(text, false)] + (LaunchArgs.value("-chatThen").map { [($0, false)] } ?? [])
            + (LaunchArgs.value("-chatNew").map { [($0, true)] } ?? [])
        for (i, (t, fresh)) in turns.enumerated() {
            if fresh {
                log("CHATONCE new chat, then \(t)")
                newConversation()
                var n = 0
                while warmTask == nil && n < 200 { await Task.yield(); n += 1 }  // the re-warm (if any) has started
                if let d = LaunchArgs.value("-chatNewDelay").flatMap({ Double($0) }), d > 0 {
                    try? await Task.sleep(nanoseconds: UInt64(d * 1e9))
                }
                log("CHATONCE sending in the new chat: re-warm \(warmTask != nil ? "in progress (the message waits for it)" : "not running")")
            } else if i > 0 { log("CHATONCE then \(t)") }
            send(t, maxNew: mx)
            await genTask?.value
            if let r = lastReply { replies.append(r) }
            let cap = conv.messages.last(where: { $0.role == .assistant }).flatMap { conv.meta[$0.id] }?.caption ?? ""
            captions.append(cap)
            shown["message_caption_\(i + 1)"] = cap
            // the tool card of this message (ToolCard's title and caption), if the model searched
            if let u = conv.messages.lastIndex(where: { $0.role == .user }) {
                for (k, tm) in conv.messages[u...].filter({ $0.role == .tool }).compactMap({ conv.meta[$0.id]?.tool }).enumerated() {
                    let card = ToolCard(call: nil, complete: true, result: tm, running: false, streaming: false)
                    shown["tool_card_\(i + 1)_\(k + 1)"] = card.title + " | " + card.caption(tm) + " | "
                        + tm.results.map { $0.title + " <" + $0.url + ">" }.joined(separator: "; ")
                }
            }
            shown["tools_\(i + 1)"] = conv.toolsOn ? "on" : "off"
            shown["status_line_\(i + 1)"] = statusLine
        }
        // the values the views bind to, formatted as the status line / SettingsView show them
        shown.merge([
            "stats_bar_rate": String(format: "%.2f tok/s", lastSummary?.meanTokPerSec ?? 0),
            "stats_bar_ttft": lastTtftMs.map { String(format: "TTFT %.1f s", $0 / 1000) } ?? "",
            "stats_bar_mb": String(format: "%.0f MB/tok", mbPerToken),
            "settings_live_rate": String(format: "%.2f", replyTokPerSec),
            "settings_last_reply": lastSummary.map { String(format: "%d tokens, %@, %.2f tok/s", $0.tokens, $0.reason.rawValue, $0.meanTokPerSec) } ?? "",
            "prefill": lastPrefillKind ?? "",
            "status_line": statusLine,
            "memory_dot": memoryLevel.rawValue,
            "thermal_dot": thermalLevel.rawValue,
            "conversation_title": conv.displayTitle,
            "conversation_tokens": "\(conv.contextTokens)",
        ]) { a, _ in a }
        for (k, v) in shown.sorted(by: { $0.key < $1.key }) { log("SHOWN \(k) = \(v)") }
        let file = ConversationStore.url(conv.id)
        log("CONVERSATION \(FileManager.default.fileExists(atPath: file.path) ? "saved" : "NOT saved") \(file.path)")
        let discard = LaunchArgs.has("-chatDiscard") && !continueLatest
        var out = UIOut(time: ISO8601DateFormatter().string(from: Date()), prompt: text, reply: lastReply, shown: shown,
                        engineSettings: Dictionary(engine.settings, uniquingKeysWith: { a, _ in a }), diagnostics: diagnostics,
                        replies: replies, captions: captions, conversationFile: file.path)
        if discard { out.conversation = ConversationFile(conversation(conv.id) ?? conv) }
        try? jsonEncoder().encode(out).write(to: docsURL.appendingPathComponent("ui.json"), options: .atomic)
        await engine.waitIdle()  // the last reply's reply-end save (a relaunch continues from it)
        if discard {
            let id = conv.id
            delete(id)
            log("CONVERSATION discarded (-chatDiscard): \(FileManager.default.fileExists(atPath: file.path) ? "STILL THERE" : "deleted") \(id.uuidString); kept in ui.json")
        }
        prefixStatus = engine.prefixCacheStatus
        log("PREFIXCACHE " + prefixStatus)
        writeProbe()
        log("DONE")
        if LaunchArgs.has("-exitWhenDone") { exit(0) }
    }

    // MARK: -memLimitTest: the app's memory limit (increased-memory-limit entitlement check)

    struct MemLimitOut: Encodable {
        var time: String
        var entitlementIncreasedMemoryLimit: Bool?   // what the signed binary carries (nil: not readable here)
        var launch: MemSample
        var steps: [MemSample]
        var warning: MemSample?                      // the first memory warning (footprint and available at it)
        var stoppedAt: MemSample?
        var stopReason: String
        var budgetCapAtLaunch: Int                   // the cap formula on os_proc_available_memory at launch
        var budgetCapIfLimitWereWarning: Int?        // the cap formula on (warning footprint - launch footprint)
    }

    /// Allocate and dirty 64 MB at a time (one block per 100 ms) until the first memory warning, or until
    /// os_proc_available_memory falls below 300 MB (never into jetsam), sampling footprint and available memory at each
    /// step; then free everything. No engine is loaded. Writes Documents/memlimit.json; console `MEMLIMIT ...`.
    func runMemLimitTest() async {
        let launch = memSample("memLimitTest launch")
        log(String(format: "MEMLIMIT launch: footprint %.3f GB, os_proc_available_memory %.3f GB, limit ~%.3f GB", Double(launch.physFootprint) / 1e9,
                   Double(launch.osProcAvailable) / 1e9, Double(launch.limitEstimate ?? 0) / 1e9))
        let w0 = memoryWarnings
        var blocks: [UnsafeMutableRawPointer] = []
        var steps: [MemSample] = []
        var warning: MemSample? = nil
        var reason = "max blocks"
        let block = 64 << 20
        for i in 0..<256 {
            let p = UnsafeMutableRawPointer.allocate(byteCount: block, alignment: 16384)
            memset(p, 0x5A, block)  // dirty: counted in phys_footprint
            blocks.append(p)
            try? await Task.sleep(nanoseconds: 100_000_000)  // the main run loop delivers a warning here
            let m = MemProbe.read("memLimitTest +\(blocks.count * 64) MB")
            steps.append(m)
            if i % 4 == 0 { log(m.line) }
            if memoryWarnings > w0 {
                warning = memSamples.last { $0.label.hasPrefix("memory warning") }
                reason = "memory warning"
                break
            }
            if m.osProcAvailable > 0 && m.osProcAvailable < 300_000_000 { reason = "available < 300 MB"; break }
            if m.osProcAvailable == 0 && blocks.count >= 64 { reason = "no os_proc_available_memory (Mac): 4 GB allocated"; break }
        }
        let stop = steps.last
        for p in blocks { p.deallocate() }
        let rb = probe?.store.recordBytes ?? Budget.defaultRecordBytes
        let capWarn = warning.map { Budget.cap(availableBytes: Double($0.physFootprint - launch.physFootprint), recordBytes: rb) }
        // the signed binary's entitlements are read on the Mac (codesign -d --entitlements); iOS has no public API for it.
        // The build passes the setting in as Info.plist key PSIncreasedMemoryLimit (see the project's entitlements note).
        let ent: Bool? = (Bundle.main.object(forInfoDictionaryKey: "PSIncreasedMemoryLimit") as? String).map { $0 == "YES" }
        let out = MemLimitOut(time: ISO8601DateFormatter().string(from: Date()), entitlementIncreasedMemoryLimit: ent, launch: launch,
                              steps: steps, warning: warning, stoppedAt: stop, stopReason: reason,
                              budgetCapAtLaunch: Budget.cap(availableBytes: Double(launch.osProcAvailable), recordBytes: rb),
                              budgetCapIfLimitWereWarning: capWarn)
        try? jsonEncoder().encode(out).write(to: docsURL.appendingPathComponent("memlimit.json"), options: .atomic)
        log("MEMLIMIT entitlement increased-memory-limit: \(ent.map { "\($0)" } ?? "n/a")")
        if let w = warning {
            log(String(format: "MEMLIMIT first warning at footprint %.3f GB (available then %.3f GB; %.3f GB above the launch footprint)",
                       Double(w.physFootprint) / 1e9, Double(w.osProcAvailable) / 1e9, Double(w.physFootprint - launch.physFootprint) / 1e9))
        }
        if let s = stop {
            log(String(format: "MEMLIMIT stopped (%@) at footprint %.3f GB, available %.3f GB", reason, Double(s.physFootprint) / 1e9, Double(s.osProcAvailable) / 1e9))
        }
        log("MEMLIMIT budget cap at launch \(out.budgetCapAtLaunch)" + (capWarn.map { ", cap if the warning point were the budget: \($0)" } ?? ""))
        writeProbe()
        log("DONE")
        if LaunchArgs.has("-exitWhenDone") { exit(0) }
    }

    // MARK: -prefixTest: cold vs restored first-token time for the G4 prompt

    /// Empties the prefix cache (unless `-keepPrefixCache`), then: the G4 prompt with thinking on (cold: prefilled, its
    /// state saved), the clear button, the same prompt again (restored from the cache: re-land, no prefill). `-maxNew N`
    /// (default 40) tokens each. Checks: reply 1 cold and saved, reply 2 restored with all prompt tokens, the same ids,
    /// the UI rate within 2% of the engine's. Console `PREFIXTEST ...`; autotest.json as the autotest writes it.
    func runPrefixTest() async {
        log("PREFIXTEST start")
        let started = Date()
        var checks: [String: Bool] = ["engine_ready": engineState == .ready]
        let n = Int(LaunchArgs.value("-maxNew") ?? "").flatMap { $0 > 0 ? $0 : nil } ?? 40
        var replies: [ReplyRecord] = []
        var details: [ReplyDetail?] = []
        if engineState == .ready && !engine.info.isStub {
            prefixCache = true
            if !LaunchArgs.has("-keepPrefixCache") { await clearForTest(); log("PREFIXTEST prefix cache emptied") }
            thinking = true
            clear()
            let r1 = await reply(to: Self.g4Prompt, maxNew: n)
            replies.append(r1); details.append(engine.lastDetail)
            clear()
            let r2 = await reply(to: Self.g4Prompt, maxNew: n)
            replies.append(r2); details.append(engine.lastDetail)
            let p1 = r1.prefill ?? [:], p2 = r2.prefill ?? [:]
            checks["reply1_cold"] = r1.prefillKind == "cold"
            checks["reply1_state_saved"] = (p1["prefix_saved"] ?? 0) >= 1
            checks["reply2_restored"] = r2.prefillKind == "restored prefix"
            checks["reply2_all_prompt_tokens_restored"] = p2["prefix_tokens_restored"].map { Int($0) } == r2.promptTokens
            checks["ids_equal_cold_vs_restored"] = r1.ids != nil && r1.ids == r2.ids
            checks["ui_rate_matches_engine_2pct"] = Self.uiRateOK(replies)
            func mbBefore(_ r: ReplyRecord) -> Double {  // drive bytes before the first sample: re-land + prefill
                (r.prefill?["prefix_reland_mb"] ?? 0) + (r.prefill?["prefill_mb_read"] ?? 0)
            }
            log(String(format: "PREFIXTEST cold: first token %.0f ms (engine ttft %.0f ms: prefill %.0f tokens %.2f s, save %.0f ms %.1f MB), %.0f MB read before it",
                       r1.firstTokenMs ?? 0, p1["ttft_ms"] ?? 0, p1["prefill_tokens"] ?? 0, p1["prefill_s"] ?? 0, p1["prefix_save_ms"] ?? 0,
                       p1["prefix_save_mb"] ?? 0, mbBefore(r1)))
            log(String(format: "PREFIXTEST restored: first token %.0f ms (engine ttft %.0f ms: restore %.0f ms = load %.0f + re-land %.0f ms, %.0f MB read, %.0f read / %.0f kept / %.0f moved; prefill %.0f tokens), %.0f MB read before it",
                       r2.firstTokenMs ?? 0, p2["ttft_ms"] ?? 0, p2["prefix_restore_ms"] ?? 0, p2["prefix_load_ms"] ?? 0, p2["prefix_reland_ms"] ?? 0,
                       p2["prefix_reland_mb"] ?? 0, p2["prefix_relanded"] ?? 0, p2["prefix_kept"] ?? 0, p2["prefix_moved"] ?? 0,
                       p2["prefill_tokens"] ?? 0, mbBefore(r2)))
        }
        let pass = checks.values.allSatisfy { $0 }
        writeAutotest(complete: true, prompt: Self.g4Prompt, maxNew: n, replies: replies, details: details, identity: nil,
                      checks: checks, pass: pass, started: started)
        writeProbe()
        for (k, v) in checks.sorted(by: { $0.key < $1.key }) { log("CHECK \(k)=\(v)") }
        log("PREFIXTEST \(pass ? "PASS" : "FAIL")")
        log("AUTOTEST \(pass ? "PASS" : "FAIL") (prefixTest)")
        log("DONE")
        if LaunchArgs.has("-exitWhenDone") { exit(pass ? 0 : 1) }
    }

    func runAutotest() async {
        log("AUTOTEST start (\(engine.info.isStub ? "stub" : "native") engine)")
        let started = Date()
        var checks: [String: Bool] = [:]
        checks["probe_written"] = FileManager.default.fileExists(atPath: probeURL.path)
        checks["engine_ready"] = engineState == .ready

        // -prompt TEXT replaces the G4 prompt of reply 1 (identity is then not checked); -maxNew N its length (300).
        let prompt1 = LaunchArgs.value("-prompt") ?? Self.g4Prompt
        let n1 = Int(LaunchArgs.value("-maxNew") ?? "").flatMap { $0 > 0 ? $0 : nil } ?? 300
        var replies: [ReplyRecord] = []
        var details: [ReplyDetail?] = []
        var identity: Identity? = nil
        func partial() {
            writeAutotest(complete: false, prompt: prompt1, maxNew: n1, replies: replies, details: details, identity: identity,
                          checks: checks, pass: false, started: started)
            writeProbe()
        }
        partial()
        // reply 1 must start cold (reply1_cold): since the prefix cache's cap is 1.5 GB (2026-11-04) the G4 prompt's state
        // from an earlier run stays cached, so the cache is emptied first, as -prefixTest does (-keepPrefixCache keeps it)
        if engineState == .ready && !engine.info.isStub && !LaunchArgs.has("-keepPrefixCache") {
            await clearForTest()
            log("AUTOTEST prefix cache emptied (reply 1 must start cold; -keepPrefixCache keeps it)")
        }
        if engineState == .ready && engine.info.isStub {
            thinking = false
            let r1 = await reply(to: "Hello! In one paragraph, what is a mixture-of-experts model?")
            replies.append(r1)
            checks["reply1_finished_eos"] = r1.reason == StopReason.eos.rawValue && r1.tokens > 20
            checks["reply1_rate_8_to_11_tok_s"] = r1.meanTokPerSec >= 8 && r1.meanTokPerSec <= 11

            thinking = true
            let cut = 15
            let r2 = await reply(to: "Now explain how the experts are read from flash, step by step.", cancelAfter: cut)
            replies.append(r2)
            thinking = false
            checks["reply2_cancelled"] = r2.reason == StopReason.cancelled.rawValue
            checks["reply2_partial"] = r2.tokens >= cut && r2.tokens <= cut + 1
            checks["reply2_thinking_text"] = r2.text.hasPrefix("<think>")
            checks["not_generating_after"] = !isGenerating
        } else if engineState == .ready {
            // 1. identity: the G4 prompt, thinking on (the template default the CLI and mlx_run.py used), greedy, 300 tokens
            thinking = true
            let r1 = await reply(to: prompt1, maxNew: n1)
            replies.append(r1)
            let d1 = engine.lastDetail
            details.append(d1)
            checks["reply1_ran_to_max_new"] = r1.reason == StopReason.maxNew.rawValue && r1.tokens == n1
            checks["reply1_cold"] = d1?.prefill == "cold"
            if prompt1 != Self.g4Prompt {
                log("IDENTITY skipped: -prompt replaces the reference prompt")
            } else if let (refPath, ref) = identityReference(), let rids = ref["ids"] as? [Int], let d1 {
                let m = min(d1.ids.count, rids.count)
                var id = Identity(ref: refPath, refCap: ref["slab"] as? Int, refPromptIds: nil, promptIdsEqual: nil,
                                  refTokens: rids.count, tokens: d1.ids.count, maxNew: n1, comparedTokens: m,
                                  idsEqual: d1.ids == rids, prefixEqual: true, firstDivergence: nil)
                if let i = zip(d1.ids, rids).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset {
                    id.firstDivergence = i
                    id.refTokAtDivergence = rids[i]
                    id.appTokAtDivergence = d1.ids[i]
                    id.prefixEqual = false
                }
                if let pp = LaunchArgs.value("-refPromptIds"), let pj = Self.readJSON(pp), let pids = pj["prompt_ids"] as? [Int] {
                    id.refPromptIds = pp
                    id.promptIdsEqual = pids == d1.promptIds
                    checks["identity_prompt_ids_equal"] = id.promptIdsEqual
                } else if let pids = ref["prompt_ids"] as? [Int] {
                    id.refPromptIds = refPath
                    id.promptIdsEqual = pids == d1.promptIds
                    checks["identity_prompt_ids_equal"] = id.promptIdsEqual
                }
                // with -maxNew below the reference's length, the requested tokens must all match
                checks["identity_ids_equal"] = id.prefixEqual && d1.ids.count == min(n1, rids.count)
                identity = id
                log("IDENTITY ref=\((refPath as NSString).lastPathComponent) cap=\(id.refCap.map(String.init) ?? "?") ids_equal=\(id.idsEqual) prefix_equal=\(id.prefixEqual) compared=\(m) tokens=\(id.tokens)/\(id.refTokens) first_divergence=\(id.firstDivergence.map(String.init) ?? "none") prompt_ids_equal=\(id.promptIdsEqual.map { "\($0)" } ?? "n/a")")
            }
            if let d1 {
                log(String(format: "RUN1 %.3f tok/s %.1f MiB/token in_memory %.4f held %.0f sync_reads %d prefill %d tokens %.1fs (%@)",
                           d1.tokPerSec, d1.mibPerToken, d1.releasedTop8InMemory, d1.heldExpertsMean, d1.syncReads,
                           d1.prefillTokens, d1.prefillSeconds, d1.prefill))
            }
            partial()
            // 2. interactive-style: a follow-up with thinking on, stopped after `cut` tokens
            let cut = Int(LaunchArgs.value("-cancelAfter") ?? "") ?? 40
            let r2 = await reply(to: "Now explain how the experts are read from flash, step by step.", cancelAfter: cut)
            replies.append(r2)
            details.append(engine.lastDetail)
            thinking = false
            checks["reply2_cancelled"] = r2.reason == StopReason.cancelled.rawValue
            checks["reply2_partial"] = r2.tokens >= cut && r2.tokens <= cut + 1
            checks["reply2_thinking_text"] = r2.text.hasPrefix("<think>")
            partial()
            // 3. a new conversation (the clear button): forgets the model state and held experts; thinking off
            clear()
            let r3 = await reply(to: "In two sentences, why can a phone not hold a 35B model in memory?", maxNew: 80)
            replies.append(r3)
            details.append(engine.lastDetail)
            checks["reply3_cold_after_clear"] = engine.lastDetail?.prefill == "cold"
                || (engine.lastDetail?.prefill == "restored prefix" && prefixCache)  // the cache may hold this prompt from an earlier launch
            checks["reply3_no_thinking"] = !r3.text.contains("<think>") && r3.tokens > 0
            checks["not_generating_after"] = !isGenerating
            checks["ui_rate_matches_engine_2pct"] = Self.uiRateOK(replies)
        }
        let pass = checks.values.allSatisfy { $0 }
        writeAutotest(complete: true, prompt: prompt1, maxNew: n1, replies: replies, details: details, identity: identity,
                      checks: checks, pass: pass, started: started)
        writeProbe()
        for (k, v) in checks.sorted(by: { $0.key < $1.key }) { log("CHECK \(k)=\(v)") }
        log("AUTOTEST \(pass ? "PASS" : "FAIL")" + (engineState == .ready ? "" : " (engine \(engineState.label))")
            + (memoryWarnings > 0 ? " [\(memoryWarnings) memory warning(s), -memWarn \(memWarnMode.rawValue)]" : ""))
        log("DONE")
        if LaunchArgs.has("-exitWhenDone") { exit(pass ? 0 : 1) }
    }
}

// MARK: - once per install (2026-11-07, docs/G4-runtime.md "Launch: once per install")

/// A step of the one-time setup's card.
struct SetupStep: Identifiable, Equatable {
    enum State: Equatable { case waiting, running, done, failed }
    var id: String
    var title: String
    var state: State
    var detail: String?
}

/// The one-time setup in progress (ChatModel.setupRun; StartupCard shows it).
struct SetupRun: Equatable {
    var steps: [SetupStep]
    var started = Date()
    var failed: String? = nil
}

/// What runs once per install, right after the store is complete (the first launch that finds it: after the download, or
/// a store copied in), and never again unless the key changes: the kernel warm-up (the engine's own marker,
/// PhoneStoreEngine.kernelWarmupMarker), and the system-prompt states of new chats with tools off and on, prefilled and
/// kept in the prefix cache (PrefixCacheDir.kept: never evicted). The record (Documents/setup.json) is written only when
/// every step succeeded. Key: the app build (the executable's size and time), the store (its meta.json and every file's
/// name and size), the prefix-state format, the slot caps (a state is refused under other caps) and the prompts' code
/// (the default system prompt and the tools block's head). A system prompt or tools definition changed in settings is
/// not part of the key: its states are prefilled once in the background (ChatModel.ensureSystemStates).
/// `-setup 1` forces the setup (its card), `-setup 0` skips it.
enum OneTimeSetup {
    static let version = "one-time setup 1"
    static var recordURL: URL { docsURL.appendingPathComponent("setup.json") }
    /// The states the last ensureSystemStates kept (written into the record).
    static var statePaths: [String] = []

    struct Record: Codable {
        var key: String
        var date: String
        var seconds: Double
        var steps: [String: String]
        var cap: Int
        var caps: [Int]
        var states: [String]? = nil
        var exe: String? = exeStamp
        /// The drive calibration (2026-11-07): reader threads chosen, and GB/s per count measured.
        var readers: Int? = nil
        var drive: [String: Double]? = nil
        /// What to expect on this phone (2026-11-07, Expectations): from the calibration and the cap.
        var estimate: Expectations.Estimate? = nil
    }

    /// The reader counts the setup's drive test tries (~2 s each); the smallest within 5% of the best is kept (more
    /// threads past the drive's saturation only add queueing). The 15 Plus saturates at 8 (docs/G4-runtime.md "Drive
    /// throughput on the phone").
    static let readerCandidates = [4, 8, 16, 32]
    /// This launch runs the setup: the engine is built with readerCandidates.max() reader threads to measure them.
    nonisolated(unsafe) static var calibrating = false
    /// The engine's readers from the record (later launches); nil: the defaults (8), or a launch argument, or a test.
    static var calibratedReaders: Int? {
        if calibrating || TestFlow.active || LaunchArgs.value("-readers") != nil { return nil }
        return load()?.readers.flatMap { $0 >= 1 ? $0 : nil }
    }

    /// The smallest count within 5% of the best GB/s.
    static func pickReaders(_ r: [(readers: Int, gbps: Double)]) -> Int? {
        guard let best = r.map(\.gbps).max(), best > 0 else { return nil }
        return r.filter { $0.gbps >= 0.95 * best }.map(\.readers).min()
    }

    static var forced: Bool? { LaunchArgs.value("-setup").map { $0 != "0" && $0 != "off" } }

    static func load() -> Record? {
        guard let d = try? Data(contentsOf: recordURL) else { return nil }
        return try? JSONDecoder().decode(Record.self, from: d)
    }

    static func save(_ r: Record) {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? e.encode(r).write(to: recordURL, options: .atomic)
    }

    static var exeStamp: String {
        guard let p = Bundle.main.executablePath, let a = try? FileManager.default.attributesOfItem(atPath: p) else { return "-" }
        return "\((a[.size] as? Int) ?? 0)@\(Int((a[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0))"
    }

    /// The store's meta.json and the name and size of every file of the store and the converted cache (the mapped
    /// weights' copies excepted: they come and go with -mappedWeights).
    static var storeHash: String {
        var h = SHA256()
        let st = modelDirURL.appendingPathComponent("store4", isDirectory: true)
        if let d = try? Data(contentsOf: st.appendingPathComponent("meta.json")) { h.update(data: d) }
        let fm = FileManager.default
        for dir in [st, nativeCacheURL] {
            for n in ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? []).sorted() where !n.hasPrefix(".") && !n.hasSuffix(".mmap.safetensors") {
                let sz = ((try? fm.attributesOfItem(atPath: dir.appendingPathComponent(n).path))?[.size] as? Int) ?? 0
                h.update(data: Data("\(dir.lastPathComponent)/\(n) \(sz)\n".utf8))
            }
        }
        return h.finalize().prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    static func key(probe: Probe?) -> String {
        let prompts = SHA256.hash(data: Data((AppSettings.defaultSystemPrompt + "\u{0}" + ToolPrompt.head).utf8)).prefix(6)
            .map { String(format: "%02x", $0) }.joined()
        let caps = probe.map { $0.layerCaps.map(String.init).joined(separator: ",") } ?? "-"
        return "\(version) | exe \(exeStamp) | store \(storeHash) | \(prefixFormat) | caps \(caps) | prompts \(prompts)"
    }
}

/// The process's life as the user sees it (2026-11-07): a line at every launch saying it is a fresh process and how the
/// previous one ended (its last scene phase: "background" means iOS ended it there, or it was terminated), and one at
/// every scene phase change (same process, uptime, the engine's state). In the console and appended to
/// Documents/lifecycle.log, which survives the process (pulled by phone_launch_1107.sh).
enum Lifecycle {
    static var url: URL { docsURL.appendingPathComponent("lifecycle.log") }
    static var d: UserDefaults { .standard }
    nonisolated(unsafe) static var started = false

    static func note(_ s: String) {
        log(s)
        let line = ISO8601DateFormatter().string(from: Date()) + String(format: " +%.2fs pid %d ", Date().timeIntervalSince(appStart), getpid()) + s + "\n"
        let fm = FileManager.default
        if let a = try? fm.attributesOfItem(atPath: url.path), ((a[.size] as? Int) ?? 0) > 512_000,
           let t = try? String(contentsOf: url, encoding: .utf8) {
            try? String(t.suffix(256_000)).write(to: url, atomically: true, encoding: .utf8)
        }
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile()
            h.write(Data(line.utf8))
            try? h.close()
        } else {
            try? Data(line.utf8).write(to: url, options: .atomic)
        }
    }

    /// Once per process (ChatModel.init).
    static func launched() {
        guard !started else { return }
        started = true
        let pp = d.integer(forKey: "life.pid"), ph = d.string(forKey: "life.phase") ?? "-", at = d.double(forKey: "life.at")
        let ago = at > 0 ? String(format: "%.0f s ago", Date().timeIntervalSince1970 - at) : "-"
        let how = pp == 0 ? "no earlier process recorded" : "previous process pid \(pp): last scene phase \(ph), \(ago)"
            + (ph == "background" ? " (it ended in the background: iOS ended it, or it was terminated there)" : "")
        note("LAUNCH fresh process (args: \(CommandLine.arguments.dropFirst().joined(separator: " ").prefix(120))); " + how)
        save("launching")
    }

    static func phase(_ p: ScenePhase, detail: String) {
        let n: String
        switch p {
        case .active: n = "active"
        case .inactive: n = "inactive"
        case .background: n = "background"
        @unknown default: n = "unknown"
        }
        note(String(format: "SCENE %@: same process, up %.1f s; %@", n, Date().timeIntervalSince(appStart), detail))
        save(n)
    }

    static func save(_ phase: String) {
        d.set(Int(getpid()), forKey: "life.pid")
        d.set(phase, forKey: "life.phase")
        d.set(Date().timeIntervalSince1970, forKey: "life.at")
    }
}
