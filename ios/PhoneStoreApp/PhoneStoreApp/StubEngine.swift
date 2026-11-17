import Foundation

/// Streams canned text at 10 tok/s with synthetic per-token stats, so the UI and the autotest work without a model.
final class StubEngine: ChatEngine, @unchecked Sendable {
    let info: EngineInfo
    var thinking: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _thinking }
        set { lock.lock(); _thinking = newValue; lock.unlock() }
    }

    static let tokensPerSecond: Double = 10

    private let lock = NSLock()
    private var _thinking = false
    private var cap = 0
    private var loaded = false
    private var task: Task<Void, Never>?

    init(requiresStore: Bool = false) {
        info = EngineInfo(name: "StubEngine", isStub: true, requiresStore: requiresStore,
                          detail: "canned text at 10 tok/s; MB/token and in-memory figures are synthetic")
    }

    func load(modelDir: URL, cap: Int, progress: @escaping (String) -> Void) async throws {
        progress("stub: no weights are read (model dir \(modelDir.lastPathComponent)/)")
        try await Task.sleep(nanoseconds: 300_000_000)
        progress("stub: would allocate \(cap) slots x 40 layers = \(cap * 40) expert records")
        lock.withLock { self.cap = cap; self.loaded = true }
        progress("stub: ready")
    }

    func generate(messages: [ChatMessage], maxNew: Int) -> AsyncThrowingStream<GenEvent, Error> {
        lock.lock()
        let isLoaded = loaded, cap = self.cap, think = _thinking
        task?.cancel()
        lock.unlock()

        let prompt = messages.last(where: { $0.role == .user })?.content ?? ""
        let pieces = Self.pieces(Self.toolReply(messages)
                                 ?? Self.reply(to: prompt, thinking: think, turn: messages.filter { $0.role == .user }.count))

        return AsyncThrowingStream { cont in
            guard isLoaded else { cont.finish(throwing: EngineError.notLoaded); return }
            let t = Task.detached(priority: .userInitiated) {
                let period = 1.0 / StubEngine.tokensPerSecond
                let start = Date()
                var last = start
                var n = 0
                var mbSum = 0.0
                var reason = StopReason.eos
                var rng = SplitMix(seed: UInt64(truncatingIfNeeded: prompt.hashValue))
                for (i, p) in pieces.enumerated() {
                    if i >= maxNew { reason = .maxNew; break }
                    // Pace on a fixed schedule so the mean rate is exactly 10 tok/s.
                    let due = start.addingTimeInterval(Double(i + 1) * period)
                    let wait = due.timeIntervalSinceNow
                    if wait > 0 {
                        do { try await Task.sleep(nanoseconds: UInt64(wait * 1e9)) } catch { reason = .cancelled; break }
                    }
                    if Task.isCancelled { reason = .cancelled; break }
                    let now = Date()
                    let ms = now.timeIntervalSince(last) * 1000
                    last = now
                    // Synthetic numbers in the range the FlashBench projection gives (about 90-130 MB/token).
                    let mb = 90 + 40 * rng.unit()
                    mbSum += mb
                    let stats = TokenStats(index: i, ms: ms, tokPerSec: ms > 0 ? 1000 / ms : 0, mbRead: mb,
                                           inMemory: 0.80 + 0.12 * rng.unit(),
                                           expertsInMemory: min(cap * 40, (i + 1) * 8 * 40),
                                           decodeSeconds: now.timeIntervalSince(start), mbTotal: mbSum)
                    cont.yield(.text(p, stats))
                    n += 1
                }
                let secs = Date().timeIntervalSince(start)
                cont.yield(.finished(GenSummary(reason: reason, tokens: n, seconds: secs,
                                                meanTokPerSec: secs > 0 ? Double(n) / secs : 0,
                                                meanMBPerToken: n > 0 ? mbSum / Double(n) : 0)))
                cont.finish()
            }
            self.lock.lock(); self.task = t; self.lock.unlock()
            cont.onTermination = { _ in t.cancel() }
        }
    }

    func cancel() {
        lock.lock(); let t = task; lock.unlock()
        t?.cancel()
    }

    func forget() {}

    var prefixCacheEnabled: Bool = false
    var prefixCacheStatus: String { "stub: no prefix cache" }
    func clearPrefixCache() async {}
    func waitIdle() async {}

    var lastDetail: ReplyDetail? { nil }

    var settings: [(String, String)] {
        [("engine", "stub (no model)"), ("rate", "\(Int(Self.tokensPerSecond)) tok/s, fixed schedule"),
         ("stats", "synthetic MB/token and in-memory figures"), ("cap (slots / layer)", "\(lock.withLock { cap })")]
    }

    // MARK: canned text

    static func reply(to prompt: String, thinking: Bool, turn: Int) -> String {
        let shown = prompt.count > 60 ? String(prompt.prefix(57)) + "..." : prompt
        var s = ""
        if thinking {
            s += "<think>\nYou wrote \"\(shown)\". This is demo mode, so there is no real reasoning here.\n</think>\n\n"
        }
        s += "This is a placeholder reply \(turn), because the model isn't on this device yet. You said: \"\(shown)\". "
        s += "Once the model is downloaded, real answers appear here, written a few words at a time, just like this one. "
        s += "Everything runs on your iPhone, so your messages never leave it unless you turn on web search. "
        s += "Tap stop at any time to end a reply early."
        return s
    }

    /// Web search in demo mode (the citations in the simulator): with the tools block in the system turn, a question about
    /// current events gets a web_search call (the app runs it: `-searchFixture FILE` or the real backend), and the turn
    /// after the result gets a short answer, so the app's tool loop, card and sources run as with the model.
    static func toolReply(_ messages: [ChatMessage]) -> String? {
        guard messages.first?.role == .system, messages[0].content.contains("web_search"), let last = messages.last else { return nil }
        if last.role == .tool {
            return "Here is a short summary of what the sources say. In demo mode this text is a placeholder, "
                + "but the sources below are the real results of the search."
        }
        let q = last.content.lowercased()
        guard last.role == .user, ["news", "today", "latest", "weather", "right now"].contains(where: { q.contains($0) }) else { return nil }
        let query = last.content.components(separatedBy: "\n").first ?? last.content   // without the engine-side search hint
        return "<tool_call>\n<function=web_search>\n<parameter=query>\n\(query)\n</parameter>\n</function>\n</tool_call>"
    }

    /// Split into word-sized pieces that keep their leading whitespace, like a tokenizer's detokenized stream.
    static func pieces(_ s: String) -> [String] {
        var out: [String] = []
        var cur = ""
        for ch in s {
            if ch.isWhitespace && !cur.isEmpty && !cur.allSatisfy({ $0.isWhitespace }) {
                out.append(cur); cur = ""
            }
            cur.append(ch)
        }
        if !cur.isEmpty { out.append(cur) }
        return out
    }
}

struct SplitMix {
    var s: UInt64
    init(seed: UInt64) { s = seed }
    mutating func next() -> UInt64 {
        s &+= 0x9E37_79B9_7F4A_7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func unit() -> Double { Double(next() >> 11) / Double(1 << 53) }
}
