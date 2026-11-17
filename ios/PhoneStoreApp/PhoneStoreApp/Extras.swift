import SwiftUI
import UIKit
import SafariServices

// Chat-app features around the chat (2026-11-07, docs/APP-COPY.md "Features"): share and export, and the helpers the
// other features use.

/// The system share sheet for text or files.
struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

/// Something to share, for `.sheet(item:)`.
struct ShareItem: Identifiable {
    let id = UUID()
    let items: [Any]
}

// MARK: export

extension Conversation {
    /// The conversation as Markdown: title, then each turn; thinking left out, a search shown as one line.
    var markdown: String {
        var out = "# \(displayTitle)\n\n"
        out += "_\(created.formatted(date: .abbreviated, time: .shortened))_\n\n"
        for msg in visible {
            switch msg.role {
            case .user:
                out += "**You:** \(msg.content)\n\n"
            case .assistant:
                let answer = splitThinking(msg.content).1
                if let c = ToolCall.split(answer) {
                    if !c.before.isEmpty { out += "**Assistant:** \(c.before)\n\n" }
                    let q = ToolCall.parse(c.call)?.query ?? ""
                    out += "_Searched the web: \(q)_\n\n"
                } else if !answer.isEmpty {
                    out += "**Assistant:** \(answer)\n\n"
                }
            case .tool:
                if let t = meta[msg.id]?.tool, !t.results.isEmpty {
                    out += t.results.map { "- [\($0.title)](\($0.url))" }.joined(separator: "\n") + "\n\n"
                }
            case .system:
                break
            }
        }
        return out
    }
}

enum ChatExport {
    /// Every saved chat in one file in the temporary folder: Markdown (all chats, newest first) or JSON (the app's own
    /// conversation format, one object per chat).
    static func file(_ chats: [Conversation], json: Bool) -> URL? {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        let stamp = f.string(from: Date())
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("Chats \(stamp).\(json ? "json" : "md")")
        let data: Data?
        if json {
            data = try? ConversationStore.encoder().encode(chats.map(ConversationFile.init))
        } else {
            data = chats.map(\.markdown).joined(separator: "\n---\n\n").data(using: .utf8)
        }
        guard let data, (try? data.write(to: url, options: .atomic)) != nil else { return nil }
        return url
    }
}

// MARK: what to expect (2026-11-07)

/// What to expect on this phone, from the one-time setup's drive calibration and the memory cap. One reference point,
/// the iPhone 15 Plus (cap 18, 0.78 GB/s, ~3.9 tokens/s); the speed scales with the drive's GB/s and with the bytes read
/// per token, which fall as the cap grows (more experts stay in memory; modelled as 1/sqrt(cap)). An estimate, rounded.
enum Expectations {
    static let refCap = 18, refGBps = 0.78, refTokensPerSec = 3.9
    static let wordsPerToken = 0.75

    struct Estimate: Codable, Equatable {
        var cap: Int
        var driveGBps: Double
        var tokensPerSec: Double
        var wordsPerSec: Double
        var memoryGB: Double
    }

    static func estimate(cap: Int, gbps: Double) -> Estimate {
        let c = Double(max(cap, Budget.minCap))
        let tps = min(15, max(0.3, refTokensPerSec * (gbps / refGBps) * (c / Double(refCap)).squareRoot()))
        return Estimate(cap: cap, driveGBps: gbps, tokensPerSec: tps, wordsPerSec: tps * wordsPerToken,
                        memoryGB: Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824)
    }

    /// "about 3 words per second" (one significant digit).
    static func speedText(_ e: Estimate) -> String {
        let w = e.wordsPerSec
        let shown = w >= 1 ? String(Int(w.rounded())) : String(format: "%.1f", w)
        return "About \(shown) word\(shown == "1" ? "" : "s") per second"
    }

    static func memoryText(_ gb: Double) -> String { "\(Int(gb.rounded())) GB" }

    static var recommended: Bool { Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824 >= 11 }
    static var stored: Estimate? { OneTimeSetup.load()?.estimate }
}

/// The card (chat, after the one-time setup) and the low-memory screen (Set up) that say what to expect on this phone.
struct ExpectationsView: View {
    var estimate: Expectations.Estimate?
    var lowMemory = false
    var grouped = false   // on a grouped list (the Set up screen)
    var onDone: (() -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What to expect").font(.headline)
            if lowMemory {
                Text(Copy.lowMemory).font(.subheadline)
            } else if let e = estimate {
                VStack(spacing: 8) {
                    row("Memory", Expectations.memoryText(e.memoryGB))
                    row("Storage speed", String(format: "%.1f GB/s", e.driveGBps))
                    row("Reply speed", Expectations.speedText(e))
                }
                Text("The reply speed is an estimate.").font(.caption).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 6) {
                Label("This is an experimental research app.", systemImage: "flask")
                if !lowMemory { Label("Replies take a few seconds to start.", systemImage: "clock") }
                if lowMemory || !Expectations.recommended {
                    Label("For the best experience, use an iPhone with 12 GB of memory, like iPhone 17 Pro.", systemImage: "iphone")
                }
            }
            .font(.subheadline).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            if let onDone {
                Button("Got it", action: onDone).buttonStyle(.borderedProminent).frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(grouped ? .secondarySystemGroupedBackground : .secondarySystemBackground),
                    in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    func row(_ k: String, _ v: String) -> some View {
        HStack { Text(k).foregroundStyle(.secondary); Spacer(); Text(v).multilineTextAlignment(.trailing) }.font(.subheadline)
    }
}

/// The empty chat: what the app is, what this phone does, and three prompts that show it (one offline, one long paste,
/// one current question that searches the web).
struct EmptyChatView: View {
    var onPick: (String, Bool) -> Void = { _, _ in }   // (text, send now)

    struct Suggestion: Identifiable {
        let id = UUID()
        let icon: String, title: String, prompt: String, send: Bool
    }
    static let suggestions = [
        Suggestion(icon: "airplane", title: "Works in airplane mode", prompt: "Explain how a refrigerator works, in simple terms.", send: true),
        Suggestion(icon: "doc.text", title: "Summarize something long", prompt: "Summarize this in three bullet points:\n\n", send: false),
        Suggestion(icon: "globe", title: "What's in the news today?", prompt: "What are today's top news headlines?", send: true),
    ]

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "iphone.gen3").font(.largeTitle).foregroundStyle(.tertiary).accessibilityHidden(true)
            Text("On this iPhone · works offline").font(.headline)
            if let e = Expectations.stored {
                Text(Expectations.speedText(e) + " on this iPhone").font(.subheadline).foregroundStyle(.secondary)
            }
            VStack(spacing: 8) {
                ForEach(Self.suggestions) { s in
                    Button { onPick(s.prompt, s.send) } label: {
                        HStack(spacing: 10) {
                            Image(systemName: s.icon).frame(width: 20).foregroundStyle(Color.accentColor)
                            Text(s.title).foregroundStyle(.primary).multilineTextAlignment(.leading)
                            Spacer(minLength: 0)
                        }
                        .font(.subheadline)
                        .padding(.horizontal, 14).padding(.vertical, 11)
                        .background(Color(.secondarySystemBackground), in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.top, 6)
        }
        .multilineTextAlignment(.center)
        .frame(maxWidth: 420)
        .frame(maxWidth: .infinity)
        .padding(.top, 40)
        .padding(.horizontal, 8)
    }
}

// MARK: citations (2026-11-07)


/// A web page inside the app.
struct SafariView: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> SFSafariViewController { SFSafariViewController(url: url) }
    func updateUIViewController(_ vc: SFSafariViewController, context: Context) {}
}

struct WebPage: Identifiable { let url: URL; var id: String { url.absoluteString } }

/// The sources a reply drew on, as tappable chips under it (title and site); a tap opens the page in the app.
struct CitationChips: View {
    let sources: [SearchResult]
    @State private var open: WebPage?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Sources").font(.caption).foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(Array(sources.prefix(6).enumerated()), id: \.offset) { i, s in
                        Button { open = URL(string: s.url).map(WebPage.init) } label: {
                            HStack(spacing: 6) {
                                Text("\(i + 1)").font(.caption2.weight(.semibold)).foregroundStyle(.white)
                                    .frame(width: 18, height: 18).background(Color.accentColor, in: Circle())
                                VStack(alignment: .leading, spacing: 0) {
                                    Text(s.title).font(.caption.weight(.medium)).lineLimit(1)
                                    Text(s.host).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                                }
                                .frame(maxWidth: 180, alignment: .leading)
                            }
                            .padding(.horizontal, 10).padding(.vertical, 7)
                            .background(Color(.secondarySystemBackground), in: Capsule())
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Source \(i + 1): \(s.title)")
                    }
                }
            }
        }
        .sheet(item: $open) { SafariView(url: $0.url).ignoresSafeArea() }
    }
}

// MARK: feedback (2026-11-07)

/// Settings > Send feedback: the app's logs and reports from Documents/ in one zip for the share sheet. Never the model,
/// the saved progress (prefix cache) or the search keys (Keychain); chats only when the user ticks the box.
enum Feedback {
    static let skip: Set<String> = ["model", "prefix", "model_dl", ".engine.lock"]
    static let maxFile: Int64 = 20_000_000

    static func bundle(includeChats: Bool) -> URL? {
        let fm = FileManager.default
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HHmm"
        let name = "PhoneStore feedback \(f.string(from: Date()))"
        let dir = fm.temporaryDirectory.appendingPathComponent(name, isDirectory: true)
        try? fm.removeItem(at: dir)
        guard (try? fm.createDirectory(at: dir, withIntermediateDirectories: true)) != nil else { return nil }
        for item in (try? fm.contentsOfDirectory(at: docsURL, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey])) ?? [] {
            let n = item.lastPathComponent
            if skip.contains(n) { continue }
            if n == "conversations" || n == "ui.json" {   // ui.json (-chatOnce) holds a chat's text too
                if includeChats { try? fm.copyItem(at: item, to: dir.appendingPathComponent(n)) }
                continue
            }
            let v = try? item.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard v?.isRegularFile == true, Int64(v?.fileSize ?? 0) <= maxFile else { continue }
            try? fm.copyItem(at: item, to: dir.appendingPathComponent(n))
        }
        let info = Bundle.main.infoDictionary
        let about = """
            PhoneStore \(info?["CFBundleShortVersionString"] ?? "?") (\(info?["CFBundleVersion"] ?? "?"))
            \(machineName()), \(ProcessInfo.processInfo.operatingSystemVersionString)
            memory \(ProcessInfo.processInfo.physicalMemory) B, chats included: \(includeChats ? "yes" : "no")
            """
        try? about.write(to: dir.appendingPathComponent("about.txt"), atomically: true, encoding: .utf8)
        var out: URL?
        var err: NSError?
        NSFileCoordinator().coordinate(readingItemAt: dir, options: .forUploading, error: &err) { zip in
            let dst = fm.temporaryDirectory.appendingPathComponent(name + ".zip")
            try? fm.removeItem(at: dst)
            if (try? fm.copyItem(at: zip, to: dst)) != nil { out = dst }
        }
        try? fm.removeItem(at: dir)
        log("FEEDBACK " + (out.map { "zip \($0.lastPathComponent)" } ?? "failed: \(err?.localizedDescription ?? "?")") + ", chats \(includeChats)")
        return out
    }
}

struct FeedbackView: View {
    @State private var includeChats = false
    @State private var share: ShareItem?
    @State private var failed = false

    var body: some View {
        Form {
            Section {
                Toggle("Include my chats", isOn: $includeChats)
            } footer: {
                Text("Feedback includes the app's logs and settings. It never includes the model or your search keys. Your chats are only included if you turn this on.")
            }
            Section {
                Button("Prepare and share") {
                    if let u = Feedback.bundle(includeChats: includeChats) { share = ShareItem(items: [u]) } else { failed = true }
                }
            } footer: {
                if failed { Text("Couldn't prepare the file. Try again.") }
            }
        }
        .navigationTitle("Send feedback")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $share) { ShareSheet(items: $0.items).presentationDetents([.medium, .large]) }
    }
}
