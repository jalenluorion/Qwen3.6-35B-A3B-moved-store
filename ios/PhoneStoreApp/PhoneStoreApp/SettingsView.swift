import SwiftUI

/// Settings: Model, Generation, Prefix cache, and everything the sheet showed before under Diagnostics.
struct SettingsView: View {
    @EnvironmentObject var m: ChatModel
    @Environment(\.dismiss) private var dismiss
    @AppStorage(AppSettings.defaultThinkingKey) private var defaultThinking = false
    @AppStorage(AppSettings.maxTokensKey) private var maxTokens = 1024
    @AppStorage(AppSettings.greedyKey) private var greedy = false
    @AppStorage(AppSettings.temperatureKey) private var temperature = AppSettings.modelTemperature
    @AppStorage(AppSettings.topPKey) private var topP = AppSettings.modelTopP
    @AppStorage(AppSettings.topKKey) private var topK = AppSettings.modelTopK
    @AppStorage(AppSettings.minPKey) private var minP = AppSettings.modelMinP
    @AppStorage(AppSettings.repetitionPenaltyKey) private var repetitionPenalty = AppSettings.modelRepetitionPenalty
    @AppStorage(AppSettings.presencePenaltyKey) private var presencePenalty = AppSettings.modelPresencePenalty
    @AppStorage(AppSettings.thinkingBudgetKey) private var thinkingBudget = 0
    @AppStorage(HistoryMode.key) private var historyMode = HistoryMode.live.rawValue
    @AppStorage(ToolSettings.enabledKey) private var toolsOn = true
    @AppStorage(ToolSettings.backendKey) private var backend = ToolBackend.duckduckgo.rawValue
    @AppStorage(ToolSettings.pageReadsKey) private var pageReads = true
    @State private var usage = PrefixCacheUsage.read()
    @State private var confirmClear = false
    @State private var share: ShareItem?
    @State private var confirmDeleteModel = false

    var body: some View {
        NavigationStack {
            Form {
                modelSection
                generationSection
                toolsSection
                samplingSection
                prefixSection
                Section {
                    Button("Export as Markdown") { share = ChatExport.file(m.conversations, json: false).map { ShareItem(items: [$0]) } }
                    Button("Export as JSON") { share = ChatExport.file(m.conversations, json: true).map { ShareItem(items: [$0]) } }
                } header: {
                    Text("Your chats")
                } footer: {
                    Text(m.conversations.isEmpty ? "No chats yet." : (m.conversations.count == 1 ? "Your chat in one file." : "All \(m.conversations.count) chats in one file."))
                }
                .disabled(m.conversations.isEmpty)
                Section {
                    NavigationLink { FeedbackView() } label: { Label("Send feedback", systemImage: "envelope") }
                }
                Section {
                    NavigationLink { DiagnosticsView().environmentObject(m) } label: {
                        Label("Diagnostics", systemImage: "stethoscope")
                    }
                } footer: {
                    Text("Speed, memory, engine settings and model details.")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .onAppear { usage = PrefixCacheUsage.read() }
            .onDisappear { Task { await m.warmUp() } }  // a changed system prompt is warmed into the cache
            .sheet(item: $share) { ShareSheet(items: $0.items).presentationDetents([.medium, .large]) }
        }
    }

    // MARK: Model

    var stateColor: Color {
        switch m.engineState {
        case .ready: return m.engine.info.isStub ? .orange : .green
        case .loading, .idle: return .secondary
        case .refused, .failed: return .red
        }
    }

    var stateWord: String {
        switch m.engineState {
        case .ready: return m.engine.info.isStub ? "Demo mode" : "Ready"
        case .idle, .loading: return "Loading…"
        case .refused, .failed: return m.lowMemory ? "Not enough memory" : "Not available"
        }
    }

    /// The compressed-files row: on, off, or the conversion's progress.
    var compressedLine: String {
        let d = ModelDownloader.shared
        if let s = d.swap, s.phase == .running { return "Compressing · \(s.doneCount) of \(s.layers.count)" }
        let n = m.probe?.store.codecLayers.count ?? 0, all = m.probe?.store.layers ?? 40
        if n == all { return "On" }
        return n == 0 ? "Off" : "\(n) of \(all) parts"
    }

    @ViewBuilder var modelSection: some View {
        Section {
            LabeledContent("Status") { Text(stateWord).foregroundStyle(stateColor) }
            LabeledContent("Model", value: m.engine.info.isStub ? "None (demo mode)" : Copy.modelName)
            if let p = m.probe, p.store.presentBytes > 0 {
                LabeledContent("Storage", value: "\(gbs(Int64(p.store.presentBytes))) used · \(gbs(Int64(max(0, p.diskAvailableImportantBytes)))) free")
            }
            NavigationLink {
                SetupView(mode: .manage, beforeDelete: { m.stop() }).environmentObject(ModelDownloader.shared)
            } label: {
                Text("Model files")
            }
            NavigationLink {
                CodecSwapView().environmentObject(ModelDownloader.shared)
            } label: {
                LabeledContent("Compressed files", value: compressedLine)
            }
            Button("Delete model", role: .destructive) { confirmDeleteModel = true }
                .disabled(m.probe.map { $0.store.presentBytes == 0 } ?? true)
                .confirmationDialog("Delete the model?", isPresented: $confirmDeleteModel, titleVisibility: .visible) {
                    Button("Delete model", role: .destructive) {
                        m.stop()
                        Task {
                            await ModelDownloader.shared.deleteModel()
                            SetupGate.shared.modelDeleted()
                        }
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("This frees \(gbs(Int64(m.probe?.store.presentBytes ?? 0))). Your chats are kept. To use the app again, you'll need to download the model again.")
                }
        } header: {
            Text("Model")
        }
    }

    // MARK: Generation

    var systemPromptPreview: String {
        let s = UserDefaults.standard.string(forKey: AppSettings.systemPromptKey) ?? AppSettings.defaultSystemPrompt
        return s.isEmpty ? "None" : s
    }

    @ViewBuilder var generationSection: some View {
        Section {
            Toggle("Thinking in new chats", isOn: $defaultThinking)
            Stepper(value: $thinkingBudget, in: 0...4096, step: 128) {
                LabeledContent("Thinking limit", value: thinkingBudget == 0 ? "None" : "\(thinkingBudget) tokens")
            }
            Stepper(value: $maxTokens, in: 128...4096, step: 128) {
                LabeledContent("Longest reply", value: "\(maxTokens) tokens")
            }
            NavigationLink {
                CustomInstructionsEditor()
            } label: {
                LabeledContent("Custom instructions", value: CustomInstructions.isSet ? "On" : "Off")
            }
        } header: {
            Text("Chat")
        } footer: {
            Text("The brain button turns thinking on or off for each chat.")
        }
    }

    // MARK: Tools

    @ViewBuilder var toolsSection: some View {
        Section {
            Toggle("Web search", isOn: $toolsOn)
            if toolsOn {
                Picker("Search with", selection: $backend) {
                    ForEach(ToolBackend.allCases) { Text($0.short).tag($0.rawValue) }
                }
                if let b = ToolBackend(rawValue: backend), b.needsKey {
                    NavigationLink { SearchKeyEditor(backend: b) } label: {
                        LabeledContent("\(b.short) API key", value: SearchKeychain.has(b) ? "Saved" : "Not set")
                    }
                }
                Toggle("Read result pages", isOn: $pageReads)
            }
        } header: {
            Text("Web search")
        } footer: {
            Text("When a question needs current facts, the model can search the web. The globe button turns search on or off for each chat.")
        }
    }

    @ViewBuilder var samplingSection: some View {
        Section {
            Picker("Chat history", selection: $historyMode) {
                Text("Fast").tag(HistoryMode.live.rawValue)
                Text("Strict").tag(HistoryMode.strict.rawValue)
            }
            Toggle("Always pick the likeliest word", isOn: $greedy)
            if !greedy {
                Stepper(value: $temperature, in: 0...2, step: 0.05) {
                    LabeledContent("Temperature", value: String(format: "%.2f", temperature))
                }
                Stepper(value: $topP, in: 0.05...1, step: 0.05) {
                    LabeledContent("Top-p", value: String(format: "%.2f", topP))
                }
                Stepper(value: $topK, in: 0...200, step: 1) {
                    LabeledContent("Top-k", value: topK == 0 ? "Off" : "\(topK)")
                }
                Stepper(value: $minP, in: 0...0.5, step: 0.01) {
                    LabeledContent("Min-p", value: minP == 0 ? "Off" : String(format: "%.2f", minP))
                }
                Stepper(value: $repetitionPenalty, in: 1...2, step: 0.05) {
                    LabeledContent("Repetition penalty", value: repetitionPenalty == 1 ? "Off" : String(format: "%.2f", repetitionPenalty))
                }
                Stepper(value: $presencePenalty, in: 0...2, step: 0.1) {
                    LabeledContent("Presence penalty", value: presencePenalty == 0 ? "Off" : String(format: "%.1f", presencePenalty))
                }
            }
            Button("Reset to defaults") {
                for k in AppSettings.samplingKeys { UserDefaults.standard.removeObject(forKey: k) }
                greedy = false
                temperature = AppSettings.modelTemperature
                topP = AppSettings.modelTopP
                topK = AppSettings.modelTopK
                minP = AppSettings.modelMinP
                repetitionPenalty = AppSettings.modelRepetitionPenalty
                presencePenalty = AppSettings.modelPresencePenalty
                for k in AppSettings.samplingKeys { UserDefaults.standard.removeObject(forKey: k) }
            }
        } header: {
            Text("Advanced")
        } footer: {
            Text(historyMode == HistoryMode.strict.rawValue
                 ? "Strict history re-reads the last reply with each message, so follow-ups start slower."
                 : "Fast history picks up where the last reply ended, so follow-ups start sooner.")
        }
    }

    // MARK: Prefix cache

    @ViewBuilder var prefixSection: some View {
        Section {
            Toggle("Save chat progress", isOn: $m.prefixCache)
                .disabled(m.isGenerating)
            LabeledContent("Storage used", value: String(format: "%.0f MB of %.1f GB", Double(usage.bytes) / 1e6, usage.capBytes / 1e9))
            Button("Clear saved progress", role: .destructive) { confirmClear = true }
                .disabled(m.isGenerating || usage.entries == 0)
                .confirmationDialog("Clear saved progress?", isPresented: $confirmClear, titleVisibility: .visible) {
                    Button("Clear", role: .destructive) {
                        m.clearPrefixCache()
                        Task { try? await Task.sleep(nanoseconds: 500_000_000); usage = PrefixCacheUsage.read() }
                    }
                } message: {
                    Text("Your chats are kept, but the next message in each one starts slower.")
                }
        } header: {
            Text("Speed")
        } footer: {
            Text("Saving progress lets chats continue quickly instead of starting over.")
        }
    }
}

/// Settings > Custom instructions: two plain fields rendered into the system prompt for new chats
/// (CustomInstructions.render); both empty = the default prompt exactly, so its cached state still applies.
struct CustomInstructionsEditor: View {
    @AppStorage(CustomInstructions.aboutKey) private var about = ""
    @AppStorage(CustomInstructions.howKey) private var how = ""

    var body: some View {
        Form {
            Section {
                TextField("For example: I'm a researcher in Pittsburgh.", text: $about, axis: .vertical).lineLimit(3...8)
            } header: {
                Text("About you")
            }
            Section {
                TextField("For example: Keep answers short and use bullet points.", text: $how, axis: .vertical).lineLimit(3...8)
            } header: {
                Text("How to answer")
            } footer: {
                Text("Used for new chats. The first message after a change starts a little slower.")
            }
            if !about.isEmpty || !how.isEmpty {
                Section { Button("Clear", role: .destructive) { about = ""; how = "" } }
            }
        }
        .navigationTitle("Custom instructions")
        .navigationBarTitleDisplayMode(.inline)
        .onChange(of: about) { _, _ in CustomInstructions.apply() }
        .onChange(of: how) { _, _ in CustomInstructions.apply() }
    }
}

enum CustomInstructions {
    static let aboutKey = "chat.aboutYou", howKey = "chat.howToAnswer"
    static var about: String { (UserDefaults.standard.string(forKey: aboutKey) ?? "").trimmingCharacters(in: .whitespacesAndNewlines) }
    static var how: String { (UserDefaults.standard.string(forKey: howKey) ?? "").trimmingCharacters(in: .whitespacesAndNewlines) }
    static var isSet: Bool { !about.isEmpty || !how.isEmpty }

    static func render(about: String, how: String) -> String {
        var s = AppSettings.defaultSystemPrompt
        if !about.isEmpty { s += "\n\nAbout the user: " + about }
        if !how.isEmpty { s += "\n\nHow to answer: " + how }
        return s
    }

    /// Write the rendered prompt as the system prompt for new chats.
    static func apply() {
        UserDefaults.standard.set(render(about: about, how: how), forKey: AppSettings.systemPromptKey)
    }
}

/// The raw system prompt for new chats (Diagnostics; existing chats keep the one they started with).
struct SystemPromptEditor: View {
    @AppStorage(AppSettings.systemPromptKey) private var prompt = AppSettings.defaultSystemPrompt

    var body: some View {
        Form {
            Section {
                TextEditor(text: $prompt)
                    .frame(minHeight: 160)
            } footer: {
                Text("Used for new chats. Each chat keeps the prompt it started with.")
            }
            Section {
                Button("Restore default") { prompt = AppSettings.defaultSystemPrompt }
                Button("Clear", role: .destructive) { prompt = "" }
            }
        }
        .navigationTitle("System prompt")
        .navigationBarTitleDisplayMode(.inline)
    }
}

/// Everything the settings sheet showed before 2026-11-04: live and last-reply figures, MemProbe, the per-token split,
/// the engine's settings and load log, the memory budget and cap, the model folder and the device.
struct DiagnosticsView: View {
    @EnvironmentObject var m: ChatModel
    @State private var showFiles = false

    var body: some View {
        Form {
            Section("Last reply") {
                Text(m.prefixStatus.isEmpty ? m.engine.prefixCacheStatus : m.prefixStatus).font(.caption).foregroundStyle(.secondary)
                row("Decode tok/s", String(format: "%.2f (last token %.2f)", m.replyTokPerSec, m.lastTokPerSec))
                row("Time to first token", (m.isGenerating ? m.replyTtftMs : m.lastTtftMs).map { String(format: "%.2f s", $0 / 1000) } ?? "-")
                row("MB / token", String(format: "%.1f", m.mbPerToken))
                row("Top-8 in memory", String(format: "%.1f %%", m.inMemory * 100))
                row("Experts in memory", "\(m.expertsInMemory) / \(m.probe?.slotsTotal ?? m.cap * 40)")
                if let s = m.lastSummary {
                    row("Last reply", String(format: "%d tokens, %@, %.2f tok/s, %.1f MB/tok", s.tokens, s.reason.rawValue,
                                             s.meanTokPerSec, s.meanMBPerToken))
                }
                if let k = m.lastPrefillKind, let p = m.lastPrefill {
                    row("Prefill", String(format: "%@: %.0f tokens, %.2f s", k, p["prefill_tokens"] ?? 0, p["prefill_s"] ?? 0))
                    if (p["prefix_hit"] ?? 0) > 0 {
                        row("Prefix restore", String(format: "%.0f tokens in %.2f s", p["prefix_tokens_restored"] ?? 0, (p["prefix_restore_ms"] ?? 0) / 1000))
                        row("Re-land", String(format: "%.0f MB in %.2f s (%.0f read, %.0f kept, %.0f moved)", p["prefix_reland_mb"] ?? 0,
                                              (p["prefix_reland_ms"] ?? 0) / 1000, p["prefix_relanded"] ?? 0, p["prefix_kept"] ?? 0, p["prefix_moved"] ?? 0))
                    }
                    if (p["prefix_saved"] ?? 0) > 0 {
                        row("Prefix saved", String(format: "%.0f MB in %.2f s", p["prefix_save_mb"] ?? 0, (p["prefix_save_ms"] ?? 0) / 1000))
                    }
                }
            }
            Section("Memory (MemProbe)") {
                if let m = m.latestMem {
                    row("Sample", m.label + String(format: " (t %.1f s)", m.t)).font(.caption)
                    row("Footprint (phys_footprint)", gb(m.physFootprint))
                    row("Footprint peak", gb(m.physFootprintPeak))
                    row("os_proc_available_memory", gb(m.osProcAvailable))
                    if let l = m.limitEstimate { row("Limit (footprint + available)", gb(l)) }
                    row("MLX active / peak / cache", [m.mlxActive, m.mlxPeak, m.mlxCache]
                        .map { $0.map { String(format: "%.2f", Double($0) / 1e9) } ?? "n/a" }.joined(separator: " / ") + " GB")
                    row("Resident / compressed", gb(m.residentSize) + " / " + gb(m.compressed)).font(.caption)
                }
                row("Memory warnings", "\(m.memoryWarnings) (-memWarn \(m.memWarnMode.rawValue))")
                Button("Sample memory now") { m.memSample("settings sheet") }
            }
            if let s = m.lastSplit {
                Section("Last reply: per-token split") {
                    row("Encode (host)", ms(s["host_encode_ms"]))
                    row("Blocked on reads", ms(s["expert_wait_ms"]))
                    row("GPU", ms(s["gpu_ms_per_token"]) + (s["gpu_busy_frac"].map { String(format: ", busy %.0f %%", 100 * $0) } ?? ""))
                    row("Drive", String(format: "%.2f GB/s, %.2f while busy", s["drive_gb_per_s"] ?? 0, s["drive_gb_per_s_busy"] ?? 0))
                    row("MiB / token", String(format: "%.1f (%.1f records)", s["mb_read_per_token"] ?? 0, s["reads_per_token"] ?? 0))
                    row("Top-8 in memory", String(format: "%.1f %%", 100 * (s["released_topk_in_memory"] ?? 0)))
                    row("Sync reads", String(format: "%.0f", s["sync_reads"] ?? 0))
                    if let p = m.lastPrefill {
                        row("Prefill", String(format: "%.0f tokens, %.1f s, %.0f MB read", p["prefill_tokens"] ?? 0, p["prefill_s"] ?? 0,
                                              p["prefill_mb_read"] ?? 0))
                    }
                }
            }
            Section("Engine") {
                row("Engine", m.engine.info.name + (m.engine.info.isStub ? " (stub)" : ""))
                Text(m.engineChoice).font(.caption).foregroundStyle(m.engine.info.isStub ? .orange : .secondary)
                row("State", m.engineState.label)
                if let r = OneTimeSetup.load() {  // the one-time setup's calibration (docs/G4-runtime.md "Launch: once per install")
                    row("One-time setup", r.date)
                    if let e = r.estimate {
                        row("Estimate", String(format: "%.1f tokens/s, %.1f words/s (cap %d, %.2f GB/s, %.0f GB)", e.tokensPerSec, e.wordsPerSec, e.cap, e.driveGBps, e.memoryGB))
                    }
                    row("Readers (calibrated)", r.readers.map { n in "\(n)" + (r.drive.map { d in " · " + d.sorted { Int($0.key) ?? 0 < Int($1.key) ?? 0 }.map { "\($0.key): \(String(format: "%.2f", $0.value)) GB/s" }.joined(separator: ", ") } ?? "") } ?? "default (not measured)")
                }
                if !m.engine.info.isStub {
                    row("Mode", EngineMode.label).foregroundStyle(EngineMode.isExact ? .orange : .primary)
                }
                Text(m.engine.info.detail).font(.caption).foregroundStyle(.secondary)
                ForEach(Array(m.engine.settings.enumerated()), id: \.offset) { _, kv in
                    row(kv.0, kv.1).font(.caption)
                }
                ForEach(Array(m.loadLog.enumerated()), id: \.offset) { _, l in
                    Text(l).font(.caption.monospaced())
                }
                Button("Reload engine") { Task { await m.load() } }
                    .disabled(m.isGenerating || m.engineState == .loading)
            }
            if let p = m.probe {
                Section("Memory and cap") {
                    row("os_proc_available_memory", gb(p.osProcAvailableMemoryBytes))
                    row("Physical memory", gb(p.physicalMemoryBytes))
                    row("Cap chosen (slots / layer)", "\(p.cap) (\(p.capSource))")
                    if p.capSource != "budget" { row("Cap from the budget", "\(p.capFromBudget)") }
                    row("Layer caps", "\(p.layerCapsSpec), \(p.slotsTotal) slots")
                    Text("slots per layer: " + p.layerCaps.map(String.init).joined(separator: " ")).font(.caption.monospaced())
                    row("Plan at this cap", gb(p.budget.planBytes) + " + " + gb(p.budget.marginBytes) + " margin")
                    if !p.budget.enoughForMinCap {
                        Text("Below cap \(p.budget.minCap): not enough memory to load.").foregroundStyle(.red)
                    } else if !p.budget.meetsG4Floor {
                        Text("Below the G4 floor of \(p.budget.g4FloorCap) slots (2 x top-8): expect more misses.")
                            .font(.caption).foregroundStyle(.orange)
                    }
                    Text("resident 1.10 + routers 0.34 + state 0.15 GB + cap x (40 + 3) x \(p.budget.recordBytes) B")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Model directory") {
                    Text(p.store.modelDir).font(.caption.monospaced()).textSelection(.enabled)
                    if p.modelDirIsLink { row("Model folder", "link (used as it is)") }
                    row("Layout", p.layout == "slim" ? "slim (store4 + cache + embeddings)" : "full (resident + routers npz)")
                    row("Native cache", p.nativeCacheDir).font(.caption)
                    row("Status", p.store.complete ? "complete" : "incomplete (\(p.store.problems.count))")
                    row("Store codec", "\(p.storeCodec): " + (p.store.codecLayers.isEmpty ? "all layers raw .bin"
                        : "\(p.store.codecLayers.count) layers .psc (\(p.store.codecLayers.map(String.init).joined(separator: ","))), \(p.store.layers - p.store.codecLayers.count) .bin"))
                    row("Present / expected", gb(p.store.presentBytes) + " / " + gb(p.store.expectedBytes))
                    row("Disk free (important use)", gb(p.diskAvailableImportantBytes))
                    row("Excluded from backup", p.modelDirExcludedFromBackup ? "yes" : "no")
                    if !p.store.complete {
                        DisclosureGroup("Missing or short (\(p.store.problems.count))", isExpanded: $showFiles) {
                            ForEach(p.store.problems, id: \.self) { Text($0).font(.caption.monospaced()) }
                        }
                    }
                }
                Section("Device") {
                    row("Machine", p.machine)
                    row("OS", p.os)
                    row("Thermal", m.thermal)
                    row("Low power mode", p.lowPowerMode ? "on" : "off")
                }
            }
            Section("System prompt") {
                NavigationLink { SystemPromptEditor() } label: {
                    Text(UserDefaults.standard.string(forKey: AppSettings.systemPromptKey) ?? AppSettings.defaultSystemPrompt)
                        .font(.caption).foregroundStyle(.secondary).lineLimit(3)
                }
            }
            Section {
                Button("Probe again") { m.reprobe() }
            }
        
        }
        .navigationTitle("Diagnostics")
        .navigationBarTitleDisplayMode(.inline)
    }

    func ms(_ v: Double?) -> String { v.map { String(format: "%.1f ms", $0) } ?? "n/a" }

    func row(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k)
            Spacer()
            Text(v).foregroundStyle(.secondary).multilineTextAlignment(.trailing)
        }
    }
}

/// Paste or clear a search API key; it goes to the Keychain only (SearchKeychain), never to settings or logs.
struct SearchKeyEditor: View {
    let backend: ToolBackend
    @State private var key = ""
    @State private var status = ""
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Form {
            Section {
                SecureField("API key", text: $key)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Button("Save") {
                    let e = SearchKeychain.set(backend, key)
                    status = e ?? (key.isEmpty ? "Key removed." : "Key saved.")
                    key = ""
                }
                if SearchKeychain.has(backend) {
                    Button("Remove key", role: .destructive) {
                        SearchKeychain.set(backend, "")
                        status = "Key removed."
                    }
                }
            } footer: {
                Text(status.isEmpty ? (backend == .brave ? "Get a free key at api-dashboard.search.brave.com." : "Get a key at app.tavily.com.") : status)
            }
        }
        .navigationTitle("\(backend.short) key")
    }
}
