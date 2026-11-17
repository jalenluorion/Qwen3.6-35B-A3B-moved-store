import SwiftUI
import UIKit

/// Whether the app shows the first-run setup (download) screen instead of the chat. Decided at launch, and again when a
/// download completes or the model is deleted.
@MainActor
final class SetupGate: ObservableObject {
    static let shared = SetupGate()
    @Published var showSetup: Bool
    /// The chat (and with it an engine) was created in this process: after Delete + a new download the app asks for a
    /// restart instead of loading a second engine next to the first.
    static var engineCreated = false
    @Published var needsRestart = false

    private init() { showSetup = Self.needsSetup() }

    static func needsSetup() -> Bool {
        if LaunchArgs.value("-codecSwap") != nil { return true }   // the conversion test
        if LaunchArgs.has("-stub") || TestFlow.active { return false }
        if LaunchArgs.value("-chatOnce") != nil || LaunchArgs.value("-chatContinue") != nil { return false }
        if DownloadConfig.autoStart || LaunchArgs.has("-deleteModel") { return true }   // download tests
        if isLink(modelDirURL) { return false }               // Mac runs: a linked qwen35/
        // an unfinished compressed-store conversion continues before the model is opened, so every replaced file's
        // space is freed at once (an open model keeps its files until the app closes)
        if SwapTarget.wanted != nil { return true }
        return !StoreStatus.check(dir: modelDirURL).complete
    }

    /// The download finished while the app was not in front (the usual case: the phone locked for the 20-30 minutes):
    /// the chat, and with it the engine load and the one-time setup (GPU work iOS refuses in the background), waits
    /// until the app is opened.
    private var pendingOpen = false

    func downloadFinished() {
        guard DownloadConfig.only == nil else { return }      // a partial test download never opens the chat
        guard DownloadConfig.demoFlow || StoreStatus.check(dir: modelDirURL).complete else { return }
        // A model deleted from Settings in this session is still open in this process (the engine keeps its files and
        // its reader threads; native/ has no unload), so a second engine cannot be loaded beside it: the one case that
        // needs a relaunch. A first run never gets here with an engine.
        if Self.engineCreated { needsRestart = true; return }
        #if !targetEnvironment(macCatalyst)
        if UIApplication.shared.applicationState != .active {
            pendingOpen = true
            log("SETUP store complete in the background: the chat opens when the app is opened")
            return
        }
        #endif
        log("SETUP store complete: opening the chat")
        showSetup = false
    }

    /// The app came to the front: open the chat a background completion left waiting.
    func appBecameActive() {
        guard pendingOpen else { return }
        pendingOpen = false
        downloadFinished()
    }

    func modelDeleted() { showSetup = true }
}

/// Strings shared by several screens (docs/APP-COPY.md).
enum Copy {
    static let lowMemory = "This iPhone doesn't have enough memory to run the model."
    static let modelName = "Qwen3.6 35B"
}

/// The setup screen: first launch (full screen) and Settings > Model files (`manage`: also Check files and Delete).
struct SetupView: View {
    enum Mode { case firstRun, manage }
    var mode: Mode
    /// Called before the model is deleted (the chat stops a reply in progress).
    var beforeDelete: (() -> Void)? = nil
    @EnvironmentObject var d: ModelDownloader
    @ObservedObject var gate = SetupGate.shared
    @State private var confirmDelete = false
    @State private var confirmCancel = false
    @State private var showFiles = false

    var body: some View {
        Group {
            if mode == .firstRun {
                NavigationStack { form.navigationTitle("").navigationBarTitleDisplayMode(.inline) }
            } else {
                form.navigationTitle("Model files").navigationBarTitleDisplayMode(.inline)
            }
        }
        .task { await d.refreshFree(); d.refreshLayerFiles() }
    }

    var converting: Bool { d.swap != nil || SwapTarget.wanted != nil }

    var form: some View {
        Form {
            if mode == .firstRun && !converting { intro }
            if mode == .firstRun && converting { CodecSwapSections() }
            if mode == .firstRun && lowMemory {
                Section {
                    ExpectationsView(estimate: nil, lowMemory: true, grouped: true)
                }
                .listRowBackground(Color.clear)
                .listRowInsets(EdgeInsets())
            } else {
                if mode == .manage || started || spaceShort { progressSection }
                if mode == .firstRun && !converting { primarySection }
            }
            if mode == .manage { manageSection; filesSection }
            aboutSection
        }
    }

    var lowMemory: Bool { DownloadConfig.memoryTooLow }

    /// Not enough free space: refused, or (before the first tap) the free space already short of the download.
    var spaceShort: Bool {
        if case .refused = d.phase { return true }
        guard d.phase == .idle, let f = d.freeBytes else { return false }
        return f < d.remainingBytes + DownloadConfig.headroomBytes
    }

    /// Anything happened yet (the first screen shows only the intro and the button until then).
    var started: Bool {
        if case .idle = d.phase { return d.doneCount > 0 }
        return true
    }

    // MARK: sections

    var intro: some View {
        Section {
            VStack(alignment: .leading, spacing: 14) {
                Image(systemName: "iphone.gen3.radiowaves.left.and.right")
                    .font(.system(.largeTitle)).imageScale(.large).foregroundStyle(Color.accentColor)
                    .accessibilityHidden(true)
                Text("An AI assistant that lives on your iPhone").font(.title2.bold())
                if !lowMemory { VStack(alignment: .leading, spacing: 8) {
                    Text("Download the model once, \(gbs(d.totalBytes)) over Wi-Fi.")
                    Text("Keep your iPhone plugged in. You can leave the app while it downloads.")
                    Text("After that, it works without a connection.")
                }
                .font(.body).foregroundStyle(.secondary) }
            }
            .padding(.vertical, 8)
            .listRowBackground(Color.clear)
        }
    }

    @ViewBuilder var progressSection: some View {
        Section {
            switch spaceShort ? .refused("") : d.phase {
            case .refused:
                Label { VStack(alignment: .leading, spacing: 4) {
                    Text("Not enough storage").font(.headline)
                    Text(spaceLine)
                    Text("Free up space in Settings › General › iPhone Storage, then come back.").foregroundStyle(.secondary)
                } } icon: { Image(systemName: "externaldrive.badge.xmark").foregroundStyle(.orange) }
            case .failed:
                Label { VStack(alignment: .leading, spacing: 4) {
                    Text("Download interrupted").font(.headline)
                    Text("Check your connection and try again.").foregroundStyle(.secondary)
                } } icon: { Image(systemName: "wifi.exclamationmark").foregroundStyle(.orange) }
            case .offline:
                Label { VStack(alignment: .leading, spacing: 4) {
                    Text("Waiting for a connection").font(.headline)
                    Text("The download starts by itself when you're online.").foregroundStyle(.secondary)
                } } icon: { Image(systemName: "wifi.slash").foregroundStyle(.secondary) }
            case .complete:
                Label(gate.needsRestart ? "Downloaded. Close the app and open it again to finish."
                                        : (DownloadConfig.only != nil ? "The selected files are downloaded."
                                           : (mode == .manage ? "Model downloaded" : "Downloaded. Getting ready…")),
                      systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            default:
                EmptyView()
            }
            if d.doneBytes > 0 || d.active || d.phase == .paused || mode == .manage {
                VStack(alignment: .leading, spacing: 8) {
                    ProgressView(value: d.fraction)
                    Text(progressLine).font(.callout).monospacedDigit()
                    if !statusWord.isEmpty { Text(statusWord).font(.footnote).foregroundStyle(.secondary) }
                }
                .padding(.vertical, 4)
            }
            if mode == .manage, let f = d.freeBytes { LabeledContent("Available", value: gbs(f)) }
        }
    }

    /// "4.2 of 18 GB · 23 MB/s · about 12 min left"
    var progressLine: String {
        var parts = [ofText(d.doneBytes, d.totalBytes)]
        if d.phase == .downloading && d.bytesPerSecond >= 1e5 && !reconnecting {
            parts.append(String(format: "%.0f MB/s", d.bytesPerSecond / 1e6))
            if let e = d.etaSeconds { parts.append(etaText(e)) }
        }
        return parts.joined(separator: " · ")
    }

    /// A file waits for its retry (the connection dropped).
    var reconnecting: Bool { d.files.contains { if case .waiting = $0.state { return true }; return false } }

    var statusWord: String {
        switch d.phase {
        case .idle: return d.doneCount > 0 ? "Paused" : ""
        case .fetchingManifest: return "Starting…"
        case .checking: return "Checking files…"
        case .downloading:
            if d.network == .none { return "Waiting for a connection" }
            if d.network == .cellularOnly && !d.cellular { return "Waiting for Wi-Fi" }
            if reconnecting { return "Reconnecting…" }
            return d.bytesPerSecond < 1e5 && d.doneBytes == 0 ? "Starting…" : "Downloading"
        case .paused: return "Paused"
        case .refused, .failed, .offline: return ""
        case .complete: return d.files.isEmpty ? "" : "All files checked"
        }
    }

    var spaceLine: String {
        let need = d.remainingBytes + DownloadConfig.headroomBytes
        return "The model needs \(gbs(need)). This iPhone has \(gbs(d.freeBytes ?? 0)) free."
    }

    var primarySection: some View {
        Section {
            switch d.phase {
            case .downloading, .fetchingManifest, .checking:
                Button { Task { await d.pause() } } label: { Text("Pause").frame(maxWidth: .infinity) }
                    .buttonStyle(.bordered).controlSize(.large)
                    .disabled(d.phase != .downloading)
            case .paused:
                Button { Task { await d.start() } } label: { Text("Resume").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).controlSize(.large)
            case .failed:
                Button { Task { await d.retryFailed() } } label: { Text("Try again").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).controlSize(.large)
            case .complete:
                EmptyView()
            case .idle, .refused, .offline:
                Button { Task { await d.start() } } label: {
                    Text(spaceShort ? "Check again" : (d.doneCount > 0 ? "Resume" : (d.phase == .idle ? "Download" : "Try again"))).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(gate.needsRestart)
            }
        } footer: {
            Toggle("Use cellular data", isOn: $d.cellular)
                .font(.footnote).foregroundStyle(.secondary)
                .padding(.top, 10)
        }
        .listRowBackground(Color.clear)
        .listRowInsets(EdgeInsets())
    }

    var manageSection: some View {
        Section {
            switch d.phase {
            case .downloading, .fetchingManifest, .checking:
                Button("Pause") { Task { await d.pause() } }.disabled(d.phase != .downloading)
                Button("Cancel download", role: .destructive) { confirmCancel = true }
            case .paused, .failed, .idle, .refused, .offline:
                if d.doneCount < d.files.count || d.files.isEmpty {
                    Button(d.doneCount > 0 ? "Resume download" : "Download") { Task { await d.start() } }
                        .disabled(gate.needsRestart)
                }
            case .complete:
                EmptyView()
            }
            Button("Check files") { Task { await d.verifyAll() } }
                .disabled(d.verifying || d.active)
            if let r = d.verifyResult { Text(r).font(.footnote).foregroundStyle(.secondary) }
            Toggle("Use cellular data", isOn: $d.cellular)
            Button("Delete model", role: .destructive) { confirmDelete = true }
                .disabled(d.verifying)
        } footer: {
            Text("Check files makes sure nothing is damaged. It takes about a minute.")
        }
        .confirmationDialog("Cancel the download?", isPresented: $confirmCancel, titleVisibility: .visible) {
            Button("Cancel download", role: .destructive) { Task { await d.cancel() } }
            Button("Keep downloading", role: .cancel) {}
        } message: {
            Text("Finished files are kept.")
        }
        .confirmationDialog("Delete the model?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete model", role: .destructive) {
                beforeDelete?()
                Task {
                    await d.deleteModel()
                    gate.modelDeleted()
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This frees about \(gbs(d.totalBytes)). Your chats are kept. To use the app again, you'll need to download the model again.")
        }
    }

    var filesSection: some View {
        Section {
            DisclosureGroup(isExpanded: $showFiles) {
                ForEach(d.files) { f in FileRow(f: f) }
            } label: {
                Text(d.files.isEmpty ? "Files" : "Files · \(d.doneCount) of \(d.files.count)")
            }
        }
    }

    var aboutSection: some View {
        Section {
            if mode == .manage {
                Link(destination: DownloadConfig.repoPageURL) { LabeledContent("Source", value: "Hugging Face") }
                LabeledContent("License", value: d.manifest?.license ?? "Apache-2.0")
                if DownloadConfig.manifestURL != DownloadConfig.defaultManifestURL {
                    LabeledContent("Test source", value: DownloadConfig.manifestURL.absoluteString).font(.caption)
                }
            }
        } footer: {
            Text("Qwen3.6-35B-A3B by the Qwen team, Apache 2.0.")
        }
    }
}

/// "4.2 of 18 GB", "25 of 90 MB"
func ofText(_ done: Int64, _ total: Int64) -> String {
    if total >= 1_000_000_000 {
        let t = Double(total) / 1e9
        return String(format: "%.1f of %@ GB", Double(done) / 1e9, t >= 10 ? String(format: "%.0f", t) : String(format: "%.1f", t))
    }
    return String(format: "%.0f of %.0f MB", Double(done) / 1e6, Double(total) / 1e6)
}

/// "about 12 min left"
func etaText(_ s: Double) -> String {
    if s < 60 { return "less than a minute left" }
    let m = Int((s / 60).rounded())
    if m < 90 { return "about \(m) min left" }
    return "about \(m / 60) h \(m % 60) min left"
}

struct FileRow: View {
    let f: FileProgress
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(f.file.local).font(.caption.monospaced()).lineLimit(1)
                Spacer()
                Text(gbs(f.file.size)).font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            HStack(spacing: 6) {
                Image(systemName: icon).foregroundStyle(color).font(.caption)
                Text(word).font(.caption2).foregroundStyle(color).lineLimit(1)
                if f.state == .downloading || f.state == .verifying {
                    ProgressView(value: f.fraction).frame(maxWidth: 120)
                    Text(String(format: "%.0f%%", 100 * f.fraction)).font(.caption2).monospacedDigit()
                }
            }
        }
    }
    /// The state in one word (FileState.label stays the download.json form).
    var word: String {
        switch f.state {
        case .queued: return "Waiting"
        case .downloading: return "Downloading"
        case .verifying: return "Checking"
        case .done: return "Done"
        case .paused: return "Paused"
        case .waiting: return "Retrying soon"
        case .failed: return "Couldn't download"
        }
    }
    var icon: String {
        switch f.state {
        case .done: return "checkmark.circle.fill"
        case .downloading: return "arrow.down.circle"
        case .verifying: return "checkmark.shield"
        case .failed: return "exclamationmark.triangle.fill"
        case .waiting: return "clock.arrow.circlepath"
        case .paused: return "pause.circle"
        case .queued: return "circle.dashed"
        }
    }
    var color: Color {
        switch f.state {
        case .done: return .green
        case .failed: return .red
        case .waiting: return .orange
        default: return .secondary
        }
    }
}

func gbs(_ b: Int64) -> String {
    b >= 1_000_000_000 ? String(format: "%.1f GB", Double(b) / 1e9) : String(format: "%.0f MB", Double(b) / 1e6)
}

// MARK: compressed store (2026-11-07)

/// Settings > Model > Compressed files: convert the installed model files layer by layer (or back), with the progress
/// of every part. ModelDownloader.startSwap does the work (docs/CODEC-STORE.md "What the app needs").
struct CodecSwapView: View {
    @EnvironmentObject var d: ModelDownloader
    var body: some View {
        Form { CodecSwapSections(showControls: true) }
            .navigationTitle("Compressed files")
            .navigationBarTitleDisplayMode(.inline)
            .task { await d.refreshFree(); d.refreshLayerFiles() }
    }
}

struct CodecSwapSections: View {
    var showControls = true
    @EnvironmentObject var d: ModelDownloader
    @State private var confirmRestore = false
    @State private var showLayers = false

    var compressedNow: Int { d.layerFiles.filter { $0 == "psc" }.count }

    var body: some View {
        Section {
            if let s = d.swap {
                switch s.phase {
                case .refused:
                    Label("Not enough storage to continue. Free up some space and try again.", systemImage: "externaldrive.badge.xmark")
                        .foregroundStyle(.orange)
                case .failed:
                    Label("Something went wrong. Nothing was lost; try again.", systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                case .complete:
                    Label(s.target == .codec ? "All files are compressed." : "All files are uncompressed.",
                          systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                    if SetupGate.engineCreated && s.heldBytes > 0 {
                        Text("Close the app and open it again to free up \(gbs(s.heldBytes)).")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                case .paused: Label("Paused", systemImage: "pause.circle")
                case .running: EmptyView()
                }
                VStack(alignment: .leading, spacing: 6) {
                    ProgressView(value: s.totalBytes > 0 ? Double(s.doneBytes + currentBytes(s)) / Double(s.totalBytes) : 0)
                    HStack {
                        Text("\(s.doneCount) of \(s.layers.count) parts").monospacedDigit()
                        Spacer()
                        Text(speedLine).monospacedDigit().foregroundStyle(.secondary)
                    }
                    .font(.callout)
                }
            } else {
                LabeledContent("Compressed", value: compressedNow == 0 ? "Off"
                    : (compressedNow >= max(d.layerFiles.count, 40) ? "On" : "\(compressedNow) of \(max(d.layerFiles.count, 40)) parts"))
            }
            if let f = d.freeBytes { LabeledContent("Available", value: gbs(f)) }
        } footer: {
            Text("Compressed files take 2.4 GB less space and make replies a little faster. Answers stay exactly the same. You can keep chatting while this runs.")
        }
        if showControls {
            Section {
                let running = d.swap?.phase == .running && d.swapActive
                if running {
                    Button("Pause") { Task { await d.pauseSwap() } }
                } else {
                    Button(compressedNow > 0 && compressedNow < d.layerFiles.count ? "Continue compressing" : "Compress files") {
                        d.startSwap(.codec)
                    }
                    .disabled(d.active || compressedNow == d.layerFiles.count && !d.layerFiles.isEmpty)
                    Button("Restore uncompressed files", role: .destructive) { confirmRestore = true }
                        .disabled(d.active || compressedNow == 0)
                }
            }
            .confirmationDialog("Restore uncompressed files?", isPresented: $confirmRestore, titleVisibility: .visible) {
                Button("Restore", role: .destructive) { d.startSwap(.raw) }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This downloads \(gbs(Int64(compressedNow) * 452_984_832)) and uses up to 2.4 GB more space.")
            }
        }
        if let s = d.swap, !s.layers.isEmpty {
            Section {
                DisclosureGroup(isExpanded: $showLayers) {
                    ForEach(s.layers) { l in
                        HStack {
                            Text(String(format: "Part %d", l.layer + 1)).font(.caption)
                            Spacer()
                            Text(word(l.state)).font(.caption2)
                                .foregroundStyle(l.state == .failed ? .red : (l.state == .done ? .green : .secondary))
                        }
                    }
                } label: { Text("Parts · \(s.doneCount) of \(s.layers.count)") }
            }
        }
    }

    func word(_ st: LayerSwap.State) -> String {
        switch st {
        case .pending: return "Waiting"
        case .waiting: return "Paused"
        case .downloading: return "Downloading"
        case .switching: return "Finishing"
        case .done: return "Done"
        case .failed: return "Failed"
        }
    }

    func currentBytes(_ s: SwapStatus) -> Int64 {
        guard let k = s.current, s.layers[k].state != .done else { return 0 }
        return d.files.reduce(0) { $0 + ($1.state == .done ? 0 : $1.bytes) }
    }
    var speedLine: String {
        guard d.phase == .downloading, d.bytesPerSecond > 1e5 else { return "" }
        return String(format: "%.0f MB/s", d.bytesPerSecond / 1e6)
    }
}
