import Foundation
import PhoneStore

// The contract between the chat UI and a model engine.
//
// It is shaped after the Phase B plan in docs/G4-runtime.md (`PhoneStoreEngine`, `TokenEvent`, `EngineStats`):
// one event per generated token carrying the incremental detokenized piece and that token's stats, cancellation
// between tokens, and the slab cap chosen by the app from os_proc_available_memory at launch (never hard-coded).
// The real engine from native/ (library `PhoneStore`) is wired in EngineSeam.swift; nothing else in the app changes.

struct ChatMessage: Identifiable, Codable, Equatable, Sendable {
    /// `tool`: a tool's result (Tools.swift), rendered by the template as `<|im_start|>user\n<tool_response>\n…\n</tool_response>`.
    enum Role: String, Codable, Sendable { case system, user, assistant, tool }
    let id: UUID
    var role: Role
    var content: String

    init(id: UUID = UUID(), role: Role, content: String) {
        self.id = id
        self.role = role
        self.content = content
    }

    /// The shape `PhoneStoreEngine.generate(messages:)` takes in the G4 plan (chat-template input).
    var templateDict: [String: String] { ["role": role.rawValue, "content": content] }
}

/// Stats for one generated token. Field meanings follow the native runtime's bench JSON.
struct TokenStats: Codable, Sendable, Equatable {
    var index: Int              // 0-based generated-token index
    var ms: Double              // wall time of this token
    var tokPerSec: Double       // 1000 / ms
    var mbRead: Double          // bytes read from flash for this token, in MB (1e6 B)
    var inMemory: Double        // fraction of the released top-8 experts already in memory (0...1), G4 `inMemory`
    var expertsInMemory: Int    // expert records resident in the slabs right now (<= cap x 40)
    /// The engine's decode clock: seconds from the reply's first sample to the end of this token (prefill and the prefix
    /// restore excluded). (index + 1) / decodeSeconds is the reply's decode rate so far, the engine's tokens / seconds.
    var decodeSeconds: Double = 0
    /// MB (1e6 B) read from flash since the prefill ended (this reply's tokens so far).
    var mbTotal: Double = 0
    /// The token's place in the think block, by token id (PhoneStore ThinkPhase: "open", "thinking", "close", "answer");
    /// nil from the stub.
    var phase: String? = nil
}

enum StopReason: String, Codable, Sendable { case eos, maxNew = "max_new", cancelled }

struct GenSummary: Codable, Sendable, Equatable {
    var reason: StopReason
    var tokens: Int
    var seconds: Double
    var meanTokPerSec: Double
    var meanMBPerToken: Double
}

/// The prompt prefill in progress (PhoneStore PrefillEvent): one before the first prefilled token and one after each.
struct PrefillTick: Sendable, Equatable {
    var kind: String          // "cold" | "live prefix" | "restored prefix"
    var promptTokens: Int     // the whole prompt
    var reusedTokens: Int     // restored or continued (not run)
    var total: Int            // tokens this prefill runs
    var done: Int             // of which run so far
    var mbRead: Double        // MB (1e6 B) read from flash by the prefill so far (the restore's re-land excluded)
    var relandMB: Double      // MB the prefix restore read (0 without one)
    var elapsed: Double       // seconds since the engine started the reply (template + restore + prefill so far)
    var prefillElapsed: Double // seconds of prefill so far (restore and state saves excluded)
}

enum GenEvent: Sendable {
    /// The prompt prefill's progress, before the first token (store-backed engines; the stub sends none).
    case prefill(PrefillTick)
    /// One token: the incremental detokenized text and its stats.
    case text(String, TokenStats)
    /// Always the last event of a stream, also after cancel().
    case finished(GenSummary)
}

struct EngineInfo: Sendable {
    var name: String
    var isStub: Bool
    /// true: load() needs a complete store in modelDir; the app refuses to load otherwise.
    var requiresStore: Bool
    var detail: String
}

/// What a store-backed engine knows about the last reply beyond GenSummary (the autotest's identity check uses it).
struct ReplyDetail: Codable, Sendable {
    var promptIds: [Int]
    var ids: [Int]
    var prefill: String               // "cold" (model state and held experts forgotten) or "live prefix"
    var prefillTokens: Int
    var prefillSeconds: Double
    var seconds: Double
    var tokPerSec: Double
    var mibPerToken: Double           // MiB (1048576 B), as the CLI's bench JSON
    var releasedTop8InMemory: Double
    var heldExpertsMean: Double
    var syncReads: Int
    /// The engine's per-token split for the reply, with phone-store-run's bench-JSON keys and definitions
    /// (PhoneStoreEngine GenerationResult.split): host_encode_ms, expert_wait_ms, gpu_ms_per_token, drive_gb_per_s, ...
    var split: [String: Double]?
    /// The prompt prefill before the first token (GenerationResult.prefillStats): prefill_s, prefill_mb_read, ...
    var prefillStats: [String: Double]?
}

protocol ChatEngine: AnyObject {
    var info: EngineInfo { get }
    /// Every setting in force, in display order (shown in the settings sheet, written to probe.json).
    var settings: [(String, String)] { get }
    /// The last finished reply's detail (store-backed engines; nil for the stub).
    var lastDetail: ReplyDetail? { get }
    /// Maps to the chat template's `enable_thinking`. Default off.
    var thinking: Bool { get set }

    /// Load resident weights + routers, allocate `cap` slab slots per layer, start the readers.
    func load(modelDir: URL, cap: Int, progress: @escaping (String) -> Void) async throws
    /// Stream one assistant reply. The stream always ends with `.finished`.
    func generate(messages: [ChatMessage], maxNew: Int) -> AsyncThrowingStream<GenEvent, Error>
    /// Stop the current reply after the token in flight. Safe to call from any thread, and when idle.
    func cancel()
    /// A new conversation: drop the model state and the held experts (the next reply starts cold, or restored from the
    /// prefix cache when its ids start with a cached prefix).
    func forget()
    /// The prefix cache (native/Sources/PhoneStore/PrefixCache.swift): the model state after each prompt, on disk, restored
    /// when a new conversation starts with the same tokens. Settings toggle; default on.
    var prefixCacheEnabled: Bool { get set }
    /// One line for the settings sheet: directory, entries, size, cap.
    var prefixCacheStatus: String { get }
    /// Delete every cached state.
    func clearPrefixCache() async
    /// Returns when the engine has finished its queued work (the native engine's reply-end save runs after a reply's
    /// stream has ended); the stub returns at once.
    func waitIdle() async
    /// The native engine behind this ChatEngine once loaded (PhoneStoreChatEngine); nil for the stub (the default below).
    var nativeEngine: PhoneStoreEngine? { get }
}

enum EngineError: LocalizedError {
    case storeIncomplete([String])
    case capTooSmall(Int, Int)
    case notLoaded

    var errorDescription: String? {
        switch self {
        case .storeIncomplete(let m): return "store incomplete: \(m.count) problem(s): " + m.prefix(5).joined(separator: "; ")
        case .capTooSmall(let c, let minCap): return "cap \(c) is below the minimum \(minCap) (top-k); not enough memory"
        case .notLoaded: return "engine not loaded"
        }
    }
}

// MARK: token counts for the tool loop (additive; docs/TOOLS.md)

extension ChatEngine {
    /// No native engine (the stub). PhoneStoreChatEngine returns its loaded engine (`nativeEngine`, under its lock).
    var nativeEngine: PhoneStoreEngine? { nil }

    /// Prompt tokens of `messages` (string messages, generation prompt added) by the engine's own tokenizer and template;
    /// nil for the stub or before load.
    func promptTokenCount(_ messages: [[String: String]], thinking: Bool) -> Int? {
        guard let e = nativeEngine else { return nil }
        return try? e.promptIds(messages: messages, thinking: thinking).count
    }

    /// Tokens of a tool result as the model reads it (the content between `<tool_response>\n` and `\n</tool_response>`):
    /// the render with the result minus the render with an empty one. nil without a native engine.
    func toolResultTokens(_ result: String) -> Int? {
        let q = ["role": "user", "content": "q"]
        guard let a = promptTokenCount([q, ["role": "tool", "content": result]], thinking: false),
              let b = promptTokenCount([q, ["role": "tool", "content": ""]], thinking: false) else { return nil }
        return a - b
    }
}
