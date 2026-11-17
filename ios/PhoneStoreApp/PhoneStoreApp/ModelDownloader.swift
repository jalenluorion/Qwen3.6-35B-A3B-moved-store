import Foundation
import CryptoKit
import UIKit
import Network

// First-run model download (2026-11-06). The app fetches `manifest.json` from the hub repo (DownloadConfig), then every
// file it lists into Documents/model/ at its `local` path, so StoreStatus.check passes with no manual step.
//
// - One URLSession download task per file, all queued at once on a BACKGROUND session (nsurlsessiond keeps them going
//   while the app is suspended or killed; iOS relaunches the app for the completion events), at most
//   DownloadConfig.concurrentFiles transferring at once (httpMaximumConnectionsPerHost).
// - Resumable: pause = cancel(byProducingResumeData:), the resume data goes to Documents/model/.download/<key>.resume;
//   a network error's resume data likewise; a relaunch adopts the session's live tasks (getAllTasks) and resumes the
//   rest from their resume data (HTTP Range).
// - Each finished file is moved to .download/<key>.part, its SHA-256 computed in 4 MiB streaming reads (CryptoKit) and
//   compared with the manifest; only then is it renamed into place. Mismatch, bad HTTP status or network error: retry
//   with backoff (2, 4, 8 ... 60 s), DownloadConfig.maxAttempts times, then the file is marked failed.
// - Before starting: remaining bytes + 1 GB must fit in the volume's free space (volumeAvailableCapacityForImportantUsage;
//   `-fakeFreeBytes N` for the test), else a refusal with the numbers.
// - Verified files are recorded in .download/verified.json (size, mtime, sha256), so a relaunch does not rehash 20 GB.
// - Cellular: off by default (setting `download.cellular`); two background sessions, one with allowsCellularAccess off.
// - Compressed store (2026-11-07, docs/CODEC-STORE.md): the manifest's group "codec" lists a lossless PSC1 copy of every
//   store4 layer (store4/layer_XX.psc + .idx, 0.865x). The selection takes a layer's pair instead of its .bin when the
//   pair is in the folder, or for a layer with neither when "compressed store" is on (DownloadConfig.compressedStore).
//   startSwap(.codec) converts an installed store layer by layer (.idx, then .psc, each hash-checked and moved into place
//   by the normal path, then the .bin deleted; the store check must pass after every step); startSwap(.raw) is the
//   reverse. Resumable: the target is kept in UserDefaults (codec.swap.want) and a relaunch continues on the set-up screen.

struct ManifestFile: Codable, Identifiable, Hashable {
    var path: String          // in the repo (URL = DownloadConfig.baseURL + path)
    var local: String         // relative to the model folder
    var size: Int64
    var sha256: String
    var required: Bool
    var kind: String?
    var group: String?        // "codec": the compressed store's files (optional)
    var layer: Int?           // codec files: the store4 layer
    var id: String { local }
    var key: String { local.replacingOccurrences(of: "/", with: "__") }
    var name: String { (local as NSString).lastPathComponent }
}

struct Manifest: Codable {
    var format: String
    var version: String
    var model: String
    var license: String
    var repo: String?
    var requiredBytes: Int64
    var files: [ManifestFile]
}

enum FileState: Equatable {
    case queued, downloading, verifying, done, paused
    case waiting(String)      // retry scheduled (reason)
    case failed(String)
    var label: String {
        switch self {
        case .queued: return "queued"
        case .downloading: return "downloading"
        case .verifying: return "checking"
        case .done: return "done"
        case .paused: return "paused"
        case .waiting(let r): return "retrying (\(r))"
        case .failed(let r): return "failed: \(r)"
        }
    }
}

struct FileProgress: Identifiable {
    var file: ManifestFile
    var state: FileState = .queued
    var bytes: Int64 = 0              // received (downloading) or hashed (verifying)
    var attempts = 0
    var resumedFrom: Int64 = 0        // the offset of the last resume (resume data or adopted task)
    var started: Date?
    var finished: Date?
    var downloadedBytes: Int64 = 0    // bytes that came over the network for this file in this launch
    var shaOK: Bool?
    var source = ""                   // "network", "already here", "adopted", "resumed"
    var id: String { file.local }
    var fraction: Double { file.size > 0 ? min(1, Double(bytes) / Double(file.size)) : 1 }
}

enum DownloadPhase: Equatable {
    case idle                  // nothing started
    case fetchingManifest
    case checking              // hashing files already in the folder
    case refused(String)       // not enough space (or no manifest)
    case offline               // the file list could not be fetched and none is cached: waits for a connection
    case downloading
    case paused
    case complete
    case failed(String)
}

/// One verified file: what was hashed (size + mtime), so an unchanged file is not hashed again.
private struct VerifiedRecord: Codable { var size: Int64; var mtime: Double; var sha256: String }

@MainActor
final class ModelDownloader: ObservableObject {
    static let shared = ModelDownloader()

    @Published private(set) var manifest: Manifest?
    @Published private(set) var phase: DownloadPhase = .idle
    @Published private(set) var files: [FileProgress] = []
    @Published private(set) var bytesPerSecond: Double = 0
    @Published private(set) var freeBytes: Int64?
    @Published private(set) var note = ""
    @Published var cellular = DownloadConfig.cellularAllowed {
        didSet {
            UserDefaults.standard.set(cellular, forKey: DownloadConfig.cellularKey)
            if phase == .downloading { Task { await self.switchSession() } }
        }
    }
    /// Verify-files action (manage screen): files hashed so far / to hash, and the files that did not match.
    @Published private(set) var verifyResult: String?
    @Published private(set) var verifying = false

    /// The compressed-store conversion (startSwap): nil when none was started in this launch.
    @Published private(set) var swap: SwapStatus?
    /// Per layer the file the engine reads at its next load: "psc" or "bin" (StoreStatus.check), refreshed by the swap.
    @Published private(set) var layerFiles: [String] = []

    /// Set by the AppDelegate when iOS relaunches the app for background session events.
    var backgroundCompletion: (() -> Void)?
    /// Called on the main actor when every selected file is done.
    var onComplete: (() -> Void)?

    private let delegate = SessionDelegate()
    private var sessions: [Bool: URLSession] = [:]     // keyed by allowsCellularAccess
    private var index: [String: Int] = [:]              // local -> files[i]
    private var tasks: [String: URLSessionDownloadTask] = [:]
    private var pausing = false
    private var cancelling = false
    private var speedSamples: [(Date, Int64)] = []
    private var sessionBytes: Int64 = 0                 // network bytes this launch (all files)
    private var startedAt: Date?
    private var timer: Timer?
    private var lastLog = Date.distantPast
    private var verified: [String: VerifiedRecord] = [:]
    private var reportWritten = false
    private var testPauseFired = false
    /// The swap runs files through the same sessions one at a time (fetchFiles); this is resumed when they are all done.
    private var fetchWaiter: CheckedContinuation<Bool, Never>?
    private var swapMode = false
    private var swapTask: Task<Void, Never>?

    /// The network as NWPathMonitor sees it (the set-up screen's "Waiting for Wi-Fi"); a change to online retries a
    /// download that is waiting for one.
    @Published private(set) var network: NetworkState = .unknown
    enum NetworkState { case unknown, wifi, cellularOnly, none }
    private let monitor = NWPathMonitor()
    private var autoRetry: Timer?

    private init() {
        delegate.owner = self
        monitor.pathUpdateHandler = { [weak self] path in
            let st: NetworkState = path.status != .satisfied ? .none : (path.isExpensive ? .cellularOnly : .wifi)
            Task { @MainActor in self?.networkChanged(st) }
        }
        monitor.start(queue: DispatchQueue(label: "phonestore.download.path"))
        verified = Self.loadVerified()
        // create the session(s) at once: a relaunch for background events must reconnect before the events arrive
        _ = session(cellular: cellular)
        if DownloadConfig.useBackgroundSession { _ = session(cellular: !cellular) }
    }

    // MARK: derived figures

    var selected: [FileProgress] { files }
    var totalBytes: Int64 { files.isEmpty ? DownloadConfig.approxStoreBytes : files.reduce(0) { $0 + $1.file.size } }
    var doneBytes: Int64 {
        files.reduce(0) { a, f in a + (f.state == .done ? f.file.size : (f.state == .verifying ? 0 : f.bytes)) }
    }
    var remainingBytes: Int64 { max(0, totalBytes - doneBytes) }
    var fraction: Double { totalBytes > 0 ? Double(doneBytes) / Double(totalBytes) : 0 }
    var etaSeconds: Double? { bytesPerSecond > 1e5 ? Double(remainingBytes) / bytesPerSecond : nil }
    var active: Bool { phase == .downloading || phase == .fetchingManifest || phase == .checking }
    var doneCount: Int { files.filter { $0.state == .done }.count }

    static var wantKey = "download.want"
    /// The user started a download and has not paused or cancelled it: a relaunch continues it.
    var wantDownload: Bool {
        get { UserDefaults.standard.bool(forKey: Self.wantKey) }
        set { UserDefaults.standard.set(newValue, forKey: Self.wantKey) }
    }

    // MARK: sessions

    private func session(cellular: Bool) -> URLSession {
        if let s = sessions[cellular] { return s }
        let cfg: URLSessionConfiguration
        if DownloadConfig.useBackgroundSession {
            cfg = .background(withIdentifier: cellular ? DownloadConfig.backgroundCellularSessionID : DownloadConfig.backgroundSessionID)
            cfg.isDiscretionary = false
            cfg.sessionSendsLaunchEvents = true
        } else {
            cfg = .default
            // fail at once (the app's retry with backoff takes over): waiting for connectivity never ends when the path
            // stays up but the server went away (the simulator test, sim_onboarding.sh)
            cfg.waitsForConnectivity = false
        }
        cfg.allowsCellularAccess = cellular
        cfg.allowsExpensiveNetworkAccess = cellular
        cfg.httpMaximumConnectionsPerHost = DownloadConfig.concurrentFiles
        cfg.timeoutIntervalForRequest = 120
        cfg.timeoutIntervalForResource = 7 * 24 * 3600
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 1
        q.name = "phonestore.download"
        let s = URLSession(configuration: cfg, delegate: delegate, delegateQueue: q)
        sessions[cellular] = s
        return s
    }

    private func allTasks() async -> [URLSessionTask] {
        var out: [URLSessionTask] = []
        for s in sessions.values { out += await s.allTasks }
        return out
    }

    // MARK: launch

    /// At launch (setup screen shown): continue a download that was running, or start one for the tests.
    func resumeIfWanted() {
        log("DOWNLOAD config " + DownloadConfig.summary)
        if LaunchArgs.has("-deleteModel") {   // test: the Delete model action on -modelDir, then exit
            guard LaunchArgs.value("-modelDir") != nil else {   // never the real store from a launch flag
                log("DOWNLOAD -deleteModel refused: it needs -modelDir (the real model is deleted only from Settings)")
                log("DONE")
                if LaunchArgs.has("-exitWhenDone") { exit(1) }
                return
            }
            Task {
                await deleteModel()
                log("DONE")
                if LaunchArgs.has("-exitWhenDone") { exit(0) }
            }
            return
        }
        if let t = SwapTarget.wanted {   // a conversion that was started and has not finished (or the -codecSwap test)
            Task { await refreshFree(); startSwap(t) }
            return
        }
        if DownloadConfig.memoryTooLow {   // the set-up screen says so; nothing to download for
            log("DOWNLOAD not offered: this device's memory is below the model's minimum")
            Task { await refreshFree() }
        } else if DownloadConfig.autoStart || wantDownload || DownloadConfig.demoFlow {
            Task { await start() }
        } else {
            Task { await refreshFree(); await loadManifestQuietly() }
        }
    }

    private func loadManifestQuietly() async {
        if manifest != nil { return }
        if let m = try? await fetchManifest() { setManifest(m) }
    }

    func refreshFree() async {
        freeBytes = Self.volumeFree()
    }

    static func volumeFree() -> Int64? {
        if let f = DownloadConfig.fakeFreeBytes { return f }
        ensureModelDir()
        let v = try? modelDirURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey])
        if let i = v?.volumeAvailableCapacityForImportantUsage, i > 0 { return i }
        return v?.volumeAvailableCapacity.map { Int64($0) }
    }

    // MARK: manifest

    private func fetchManifest() async throws -> Manifest {
        let url = DownloadConfig.manifestURL
        let cached = DownloadConfig.stateDir.appendingPathComponent("manifest.json")
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        do {
            var req = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 30)
            req.allowsCellularAccess = true   // a few KB; the files themselves follow the cellular setting
            let (data, resp) = try await URLSession.shared.data(for: req)
            if let h = resp as? HTTPURLResponse, !(200..<300).contains(h.statusCode) {
                throw NSError(domain: "download", code: h.statusCode,
                              userInfo: [NSLocalizedDescriptionKey: "manifest: HTTP \(h.statusCode) from \(url.absoluteString)"])
            }
            let m = try dec.decode(Manifest.self, from: data)
            try? FileManager.default.createDirectory(at: DownloadConfig.stateDir, withIntermediateDirectories: true)
            try? data.write(to: cached, options: .atomic)
            log("DOWNLOAD manifest \(url.absoluteString): \(m.files.count) files, required \(m.requiredBytes) B, version \(m.version)")
            return m
        } catch {
            if let d = try? Data(contentsOf: cached), let m = try? dec.decode(Manifest.self, from: d) {
                log("DOWNLOAD manifest fetch failed (\(error.localizedDescription)); using the cached copy (version \(m.version))")
                return m
            }
            throw error
        }
    }

    /// The files a complete store needs: every required file, except that a layer is taken as its .psc + .idx pair
    /// when the pair is in the folder (a converted layer) or, for a layer with neither file, when compressed store is on.
    func storeSelection(_ m: Manifest) -> [ManifestFile] {
        let codec = Dictionary(grouping: m.files.filter { $0.group == "codec" && $0.layer != nil }, by: { $0.layer! })
        var out: [ManifestFile] = []
        for f in m.files where f.required {
            if let L = Self.binLayer(f), let pair = codec[L], pair.count == 2 {
                let here = Self.layerPresence(L)
                if here.pair || (DownloadConfig.compressedStore && !here.bin) {
                    out += pair.sorted { $0.local.hasSuffix(".idx") && !$1.local.hasSuffix(".idx") }
                    continue
                }
            }
            out.append(f)
        }
        return out
    }

    /// store4/layer_XX.bin -> XX
    nonisolated static func binLayer(_ f: ManifestFile) -> Int? {
        let n = f.local
        guard n.hasPrefix("store4/layer_"), n.hasSuffix(".bin") else { return nil }
        return Int(n.dropFirst("store4/layer_".count).dropLast(4))
    }

    /// Which of a layer's files are in the model folder (links followed: a Mac test folder of links counts).
    nonisolated static func layerPresence(_ L: Int) -> (bin: Bool, pair: Bool) {
        let base = modelDirURL.appendingPathComponent(String(format: "store4/layer_%02d", L)).path
        func has(_ p: String) -> Bool { resolvedSize(p) != nil }
        return (has(base + ".bin"), has(base + ".psc") && has(base + ".idx"))
    }

    /// Size of a regular file, following links (nil: absent or not a file).
    nonisolated static func resolvedSize(_ path: String) -> Int64? {
        guard let a = try? FileManager.default.attributesOfItem(atPath: (path as NSString).resolvingSymlinksInPath),
              (a[.type] as? FileAttributeType) == .typeRegular else { return nil }
        return (a[.size] as? NSNumber)?.int64Value
    }

    /// The compressed-store toggle of the set-up screen changed: re-select (only before anything was downloaded).
    func reselect() {
        guard let m = manifest, !active else { return }
        setManifest(m)
    }

    private func setManifest(_ m: Manifest) {
        manifest = m
        var sel = storeSelection(m)
        if let only = DownloadConfig.only {
            sel = m.files.filter { f in
                only.contains { o in
                    o == f.local || o == f.path || (o == "small" && f.required && f.size < DownloadConfig.smallLimit)
                }
            }
        }
        // keep the progress of files already tracked (a restart after pause)
        let old = Dictionary(uniqueKeysWithValues: files.map { ($0.file.local, $0) })
        files = sel.map { f in
            var p = old[f.local] ?? FileProgress(file: f)
            p.file = f
            // a file with a matching verified record (size + mtime) is shown as done without hashing it again
            if old[f.local] == nil, isVerified(f, at: modelDirURL.appendingPathComponent(f.local)) {
                p.state = .done; p.bytes = f.size; p.shaOK = true; p.source = "already here"
            }
            return p
        }
        index = Dictionary(uniqueKeysWithValues: files.enumerated().map { ($1.file.local, $0) })
    }

    // MARK: start / pause / cancel

    func start() async {
        guard !active, !swapMode else { return }
        ensureModelDir()
        pausing = false
        cancelling = false
        reportWritten = false
        note = ""
        // the user asked for the model: a relaunch, a reconnect or freed space continues by itself until it is done,
        // paused or cancelled (2026-11-07, onboarding)
        if DownloadConfig.only == nil { wantDownload = true }
        phase = .fetchingManifest
        let m: Manifest
        do { m = try await fetchManifest() } catch {
            log("DOWNLOAD failed manifest: \(error.localizedDescription)")
            if DownloadConfig.autoStart {
                phase = .failed("Could not get the file list: \(error.localizedDescription)")
                finishIfTest()
            } else {
                phase = .offline   // no dialog: retried when the network comes back, every 15 s, and when the app is opened
                scheduleAutoRetry(15)
            }
            return
        }
        setManifest(m)
        log("DOWNLOAD plan \(files.count) file(s), \(totalBytes) B" + (DownloadConfig.only != nil ? " (-downloadOnly)" : ""))

        // 1. what is already here: verified record, or a matching size to hash; leftovers in .download to finish
        phase = .checking
        let live = await allTasks()
        let liveLocals = Set(live.compactMap { $0.taskDescription })
        for i in files.indices where files[i].state != .done {
            let f = files[i].file
            if liveLocals.contains(f.local) { continue }   // being downloaded by the session (adopted below)
            let dst = modelDirURL.appendingPathComponent(f.local)
            let part = partURL(f)
            if let sz = Self.size(dst.path), sz == f.size {
                if isVerified(f, at: dst) {
                    files[i].state = .done; files[i].bytes = f.size; files[i].source = "already here"; files[i].shaOK = true
                    continue
                }
                if await verify(i, at: dst, moveTo: nil) { files[i].source = "already here"; continue }
            } else if let sz = Self.size(part.path), sz == f.size {
                if await verify(i, at: part, moveTo: dst) { files[i].source = "network (earlier launch)"; continue }
            }
            files[i].state = .queued
        }
        if cancelling { return }

        // 2. free space: what is left + 1 GB
        let need = files.filter { $0.state != .done }.reduce(Int64(0)) { $0 + $1.file.size }
        let free = Self.volumeFree()
        freeBytes = free
        if need > 0, let free, need + DownloadConfig.headroomBytes > free {
            let msg = String(format: "Not enough space: the download needs %.2f GB (%.2f GB of files + 1 GB to spare) and this device has %.2f GB free. Free up %.2f GB (Settings > General > iPhone Storage) and try again.",
                             Double(need + DownloadConfig.headroomBytes) / 1e9, Double(need) / 1e9, Double(free) / 1e9,
                             Double(need + DownloadConfig.headroomBytes - free) / 1e9)
            phase = .refused(msg)
            log("DOWNLOAD refused need=\(need) headroom=\(DownloadConfig.headroomBytes) free=\(free)")
            finishIfTest()
            if !DownloadConfig.autoStart { scheduleAutoRetry(20) }   // starts by itself once the space is there
            return
        }
        log("DOWNLOAD space need=\(need) + headroom \(DownloadConfig.headroomBytes) <= free \(free.map(String.init) ?? "unknown")")

        // 3. tasks: adopt the session's live ones, resume from resume data, or start fresh
        wantDownload = DownloadConfig.only == nil
        phase = .downloading
        startedAt = startedAt ?? Date()
        startTimer()
        for t in live {
            guard let local = t.taskDescription, let i = index[local], let dt = t as? URLSessionDownloadTask else {
                t.cancel(); continue
            }
            if files[i].state == .done { t.cancel(); continue }
            if t.state == .suspended { t.resume() }
            tasks[local] = dt
            files[i].state = .downloading
            files[i].bytes = t.countOfBytesReceived
            files[i].resumedFrom = t.countOfBytesReceived
            files[i].source = "adopted"
            files[i].started = files[i].started ?? Date()
            log("DOWNLOAD adopt \(local) task \(t.taskIdentifier) at \(t.countOfBytesReceived) B (state \(t.state.rawValue))")
        }
        for i in files.indices where files[i].state == .queued || files[i].state == .paused { launch(i) }
        checkAllDone()
    }

    // MARK: recovering by itself (2026-11-07, onboarding)

    /// The app came to the front (SetupScreen): pick up a download that waits for space or a connection, or failed.
    func appBecameActive() {
        Task { await refreshFree() }
        retryIfWanted("app opened")
    }

    private func networkChanged(_ st: NetworkState) {
        let was = network
        network = st
        if was != st { log("DOWNLOAD network \(st)") }
        if st != .none && was != st { retryIfWanted("network back") }
    }

    private func scheduleAutoRetry(_ seconds: Double) {
        autoRetry?.invalidate()
        autoRetry = Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) { [weak self] _ in
            Task { @MainActor in self?.retryIfWanted("timer") }
        }
    }

    /// Start again when the user wanted the download and it stopped for a reason that may have gone away.
    private func retryIfWanted(_ why: String) {
        guard wantDownload, !swapMode, DownloadConfig.only == nil, !SetupGate.shared.needsRestart else { return }
        switch phase {
        case .offline:
            log("DOWNLOAD retry (\(why)): waiting for the file list")
            Task { await start() }
        case .refused:
            let need = remainingBytes + DownloadConfig.headroomBytes
            if let f = Self.volumeFree(), f >= need {
                log("DOWNLOAD retry (\(why)): space is there now")
                Task { await start() }
            } else { scheduleAutoRetry(20) }
        case .failed:
            log("DOWNLOAD retry (\(why)): failed files")
            Task { await retryFailed() }
        default:
            break
        }
    }

    private func launch(_ i: Int) {
        let f = files[i].file
        guard tasks[f.local] == nil else { return }
        let s = session(cellular: cellular)
        let task: URLSessionDownloadTask
        let rd = resumeURL(f)
        if let data = try? Data(contentsOf: rd) {
            try? FileManager.default.removeItem(at: rd)
            task = s.downloadTask(withResumeData: data)
            let off = Self.resumeOffset(data)
            files[i].resumedFrom = off
            files[i].bytes = off
            files[i].source = "resumed"
            log("DOWNLOAD resume \(f.local) from resume data at \(off) B")
        } else {
            var req = URLRequest(url: DownloadConfig.baseURL.appendingPathComponent(f.path))
            req.timeoutInterval = 120
            task = s.downloadTask(with: req)
            files[i].bytes = 0
            files[i].source = "network"
            log("DOWNLOAD start \(f.local) \(f.size) B <- \(req.url!.absoluteString)")
        }
        task.taskDescription = f.local
        task.countOfBytesClientExpectsToReceive = f.size
        tasks[f.local] = task
        files[i].state = .queued   // becomes .downloading at the first bytes
        files[i].started = nil
        task.resume()
    }

    func pause() async {
        guard phase == .downloading else { return }
        pausing = true
        if swapMode { swapPausing = true; SwapTarget.wanted = nil }   // a paused conversion is resumed from Settings
        wantDownload = false
        for t in await allTasks() {
            guard let dt = t as? URLSessionDownloadTask, let local = dt.taskDescription else { t.cancel(); continue }
            let data = await withCheckedContinuation { (c: CheckedContinuation<Data?, Never>) in dt.cancel { c.resume(returning: $0) } }
            if let data, let i = index[local] {
                try? FileManager.default.createDirectory(at: DownloadConfig.stateDir, withIntermediateDirectories: true)
                try? data.write(to: resumeURL(files[i].file), options: .atomic)
                log("DOWNLOAD paused \(local) at \(Self.resumeOffset(data)) B (resume data saved)")
            }
        }
        tasks.removeAll()
        for i in files.indices where files[i].state != .done { files[i].state = .paused }
        phase = .paused
        bytesPerSecond = 0
        stopTimer()
        writeReport()
        if let w = fetchWaiter { fetchWaiter = nil; w.resume(returning: false) }
    }

    /// Stop and throw away partial files and resume data; finished, verified files stay.
    func cancel() async {
        cancelling = true
        wantDownload = false
        for t in await allTasks() { t.cancel() }
        tasks.removeAll()
        for f in files {
            try? FileManager.default.removeItem(at: partURL(f.file))
            try? FileManager.default.removeItem(at: resumeURL(f.file))
        }
        for i in files.indices where files[i].state != .done { files[i].state = .queued; files[i].bytes = 0 }
        phase = .idle
        bytesPerSecond = 0
        stopTimer()
        log("DOWNLOAD cancelled")
        if let w = fetchWaiter { fetchWaiter = nil; w.resume(returning: false) }
    }

    private func switchSession() async {
        log("DOWNLOAD cellular -> \(cellular): moving the transfers to the other session")
        await pause()
        await start()
    }

    // MARK: delegate events (main actor)

    fileprivate func didWrite(_ local: String, written: Int64, total: Int64) {
        guard let i = index[local] else { return }
        if files[i].state == .queued || files[i].state == .paused { files[i].state = .downloading }
        if files[i].started == nil { files[i].started = Date() }
        files[i].bytes = total
        files[i].downloadedBytes += written
        sessionBytes += written
        if let n = LaunchArgs.value("-pauseAfterBytes").flatMap({ Int64($0) }), sessionBytes >= n, !pausing, !testPauseFired {
            testPauseFired = true
            log("DOWNLOAD -pauseAfterBytes \(n): pausing at \(sessionBytes) B")
            Task {
                await self.pause()
                if LaunchArgs.has("-exitWhenDone") { log("DONE"); exit(0) }
            }
        }
    }

    fileprivate func didResume(_ local: String, offset: Int64) {
        guard let i = index[local] else { return }
        files[i].resumedFrom = offset
        files[i].bytes = offset
        log("DOWNLOAD resumed \(local) at offset \(offset) B (HTTP Range)")
    }

    fileprivate func didFinish(_ local: String, taskID: Int, outcome: SessionDelegate.Outcome, error: Error?, resumeData: Data?) {
        // an event of an older task for a file that has a newer one running (a force-quit's cancellations are delivered
        // at the next launch, possibly after start() launched the file again): ignore it, so the file is not fetched twice
        if let cur = tasks[local], cur.taskIdentifier != taskID {
            log("DOWNLOAD \(local): ignored the end of an older task \(taskID) (task \(cur.taskIdentifier) is running)")
            return
        }
        tasks[local] = nil
        // resume data first, even for an event that arrives before start() has the file list (a relaunch)
        if let resumeData, !cancelling {
            try? FileManager.default.createDirectory(at: DownloadConfig.stateDir, withIntermediateDirectories: true)
            try? resumeData.write(to: DownloadConfig.stateDir.appendingPathComponent(local.replacingOccurrences(of: "/", with: "__") + ".resume"),
                                  options: .atomic)
            log("DOWNLOAD \(local): task ended with resume data at \(Self.resumeOffset(resumeData)) B (saved)")
        }
        guard let i = index[local] else {
            log("DOWNLOAD event for \(local) before the file list (kept: .part or resume data)")
            return
        }
        if cancelling { return }
        if let error {
            let ns = error as NSError
            if pausing && ns.code == NSURLErrorCancelled { return }
            retry(i, reason: (resumeData != nil ? "network error, resumable: " : "network error: ") + ns.localizedDescription)
            return
        }
        switch outcome {
        case .moved:
            Task {
                let ok = await verify(i, at: partURL(files[i].file), moveTo: modelDirURL.appendingPathComponent(files[i].file.local))
                if !ok { retry(i, reason: "sha256 mismatch") }
                checkAllDone()
            }
        case .badStatus(let code):
            retry(i, reason: "HTTP \(code)")
        case .moveFailed(let why):
            retry(i, reason: "could not store the file: \(why)")
        case .none:
            retry(i, reason: "no file")
        }
    }

    private func retry(_ i: Int, reason: String) {
        files[i].attempts += 1
        let a = files[i].attempts
        if a >= DownloadConfig.maxAttempts {
            files[i].state = .failed(reason)
            log("DOWNLOAD failed \(files[i].file.local) after \(a) attempts: \(reason)")
            checkAllDone()
            return
        }
        let d = DownloadConfig.backoff(attempt: a)
        files[i].state = .waiting(reason)
        log("DOWNLOAD retry \(files[i].file.local) in \(Int(d)) s (attempt \(a + 1) of \(DownloadConfig.maxAttempts)): \(reason)")
        let local = files[i].file.local
        DispatchQueue.main.asyncAfter(deadline: .now() + d) { [weak self] in
            guard let self, self.phase == .downloading, let j = self.index[local], case .waiting = self.files[j].state else { return }
            self.launch(j)
        }
    }

    /// Retry the failed files (the screen's button).
    func retryFailed() async {
        for i in files.indices { if case .failed = files[i].state { files[i].state = .queued; files[i].attempts = 0 } }
        if phase == .downloading { for i in files.indices where files[i].state == .queued { launch(i) } } else { await start() }
    }

    fileprivate func sessionEventsFinished() {
        log("DOWNLOAD background session events delivered")
        backgroundCompletion?()
        backgroundCompletion = nil
    }

    private func checkAllDone() {
        guard phase == .downloading, !files.isEmpty else { return }
        func isFailed(_ p: FileProgress) -> Bool { if case .failed = p.state { return true }; return false }
        if files.contains(where: { $0.state != .done && !isFailed($0) }) { return }
        stopTimer()
        bytesPerSecond = 0
        if swapMode {   // one step of the conversion: hand the result to runSwap
            phase = .idle
            let ok = !files.contains(where: isFailed)
            if let w = fetchWaiter { fetchWaiter = nil; w.resume(returning: ok) }
            return
        }
        let failed = files.filter(isFailed)
        if !failed.isEmpty {
            phase = .failed("\(failed.count) file(s) could not be downloaded; tap Retry.")
            log("DOWNLOAD FAILED \(failed.count) file(s): " + failed.map { $0.file.local }.joined(separator: ", "))
            finishIfTest()
            if !DownloadConfig.autoStart { scheduleAutoRetry(60) }
            return
        }
        // the store's copy of the manifest, next to the files it describes
        if let d = try? Data(contentsOf: DownloadConfig.stateDir.appendingPathComponent("manifest.json")) {
            try? d.write(to: modelDirURL.appendingPathComponent("manifest.json"), options: .atomic)
        }
        phase = .complete
        wantDownload = false
        let dt = startedAt.map { Date().timeIntervalSince($0) } ?? 0
        log(String(format: "DOWNLOAD DONE %d file(s), %lld B; %lld B over the network this launch in %.1f s (%.1f MB/s)",
                   files.count, totalBytes, sessionBytes, dt, dt > 0 ? Double(sessionBytes) / 1e6 / dt : 0))
        let st = StoreStatus.check(dir: modelDirURL)
        log("DOWNLOAD store check: \(st.complete ? "complete" : "incomplete, \(st.problems.count) problem(s)")")
        finishIfTest()
        onComplete?()
    }

    private func finishIfTest() {
        writeReport()
        if LaunchArgs.has("-exitWhenDone") && (DownloadConfig.autoStart) {
            let code: Int32
            switch phase { case .complete: code = 0; case .refused: code = 2; default: code = 1 }
            log("DONE")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { exit(code) }
        }
    }

    // MARK: verification

    private func isVerified(_ f: ManifestFile, at url: URL) -> Bool {
        guard let r = verified[f.local], r.sha256 == f.sha256, r.size == f.size,
              let m = Self.mtime(url.path) else { return false }
        return abs(m - r.mtime) < 1
    }

    /// Hash `url` (streaming) against the manifest; on a match move it to `moveTo` (if given) and record it.
    private func verify(_ i: Int, at url: URL, moveTo: URL?) async -> Bool {
        let f = files[i].file
        files[i].state = .verifying
        files[i].bytes = 0
        let t0 = Date()
        let sha = await Self.sha256(url) { [weak self] n in
            Task { @MainActor in
                if let self, let j = self.index[f.local], self.files[j].state == .verifying { self.files[j].bytes = n }
            }
        }
        let dt = Date().timeIntervalSince(t0)
        guard let sha, sha == f.sha256 else {
            log("DOWNLOAD sha256 MISMATCH \(f.local): got \(sha ?? "unreadable"), want \(f.sha256)")
            try? FileManager.default.removeItem(at: url)
            files[i].shaOK = false
            files[i].state = .queued
            files[i].bytes = 0
            return false
        }
        var final = url
        if let dst = moveTo {
            let fm = FileManager.default
            try? fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fm.removeItem(at: dst)
            do { try fm.moveItem(at: url, to: dst) } catch {
                log("DOWNLOAD could not move \(f.local) into place: \(error.localizedDescription)")
                files[i].state = .queued
                return false
            }
            final = dst
        }
        if let m = Self.mtime(final.path) {
            verified[f.local] = VerifiedRecord(size: f.size, mtime: m, sha256: f.sha256)
            saveVerified()
        }
        try? FileManager.default.removeItem(at: resumeURL(f))
        files[i].state = .done
        files[i].bytes = f.size
        files[i].shaOK = true
        files[i].finished = Date()
        let net = files[i].started.map { files[i].finished!.timeIntervalSince($0) }
        log(String(format: "DOWNLOAD verified %@ %lld B sha256 ok (hash %.2f s, %.0f MB/s)%@", f.local, f.size, dt,
                   dt > 0 ? Double(f.size) / 1e6 / dt : 0,
                   net.map { String(format: "; transfer %.1f s, %.1f MB/s, resumed from %lld B", $0,
                                    $0 > 0 ? Double(files[i].downloadedBytes) / 1e6 / $0 : 0, files[i].resumedFrom) } ?? ""))
        return true
    }

    nonisolated static func sha256(_ url: URL, progress: @escaping @Sendable (Int64) -> Void) async -> String? {
        await Task.detached(priority: .utility) { () -> String? in
            guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
            defer { try? h.close() }
            var hasher = SHA256()
            var n: Int64 = 0
            var lastReport: Int64 = 0
            while true {
                let chunk: Data? = autoreleasepool { try? h.read(upToCount: 4 << 20) }
                guard let c = chunk, !c.isEmpty else { break }
                hasher.update(data: c)
                n += Int64(c.count)
                if n - lastReport >= 64 << 20 { progress(n); lastReport = n }
            }
            progress(n)
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }.value
    }

    /// Settings > Model files > Verify files: hash every required file in the folder against the manifest.
    func verifyAll() async {
        guard !verifying, !active, !swapMode else { return }
        verifying = true
        verifyResult = "Getting the file list..."
        defer { verifying = false }
        guard let m = try? await fetchManifest() else { verifyResult = "Could not get the file list (offline?)."; return }
        if manifest == nil || files.isEmpty { setManifest(m) }
        var bad: [String] = [], missing: [String] = [], ok = 0
        let req = storeSelection(m)   // a converted layer: its .psc + .idx instead of the .bin
        for (n, f) in req.enumerated() {
            verifyResult = "Checking \(n + 1) of \(req.count): \(f.local)"
            let url = modelDirURL.appendingPathComponent(f.local)
            guard let sz = Self.resolvedSize(url.path) else { missing.append(f.local); continue }
            if sz != f.size { bad.append(f.local); continue }
            let sha = await Self.sha256(url) { _ in }
            if sha == f.sha256 {
                ok += 1
                if let mt = Self.mtime(url.path) { verified[f.local] = VerifiedRecord(size: f.size, mtime: mt, sha256: f.sha256) }
            } else {
                bad.append(f.local)
                verified[f.local] = nil
            }
        }
        saveVerified()
        var s = "\(ok) of \(req.count) files match the manifest."
        if !bad.isEmpty { s += " Damaged: " + bad.joined(separator: ", ") + "." }
        if !missing.isEmpty { s += " Missing: " + missing.joined(separator: ", ") + "." }
        if !bad.isEmpty || !missing.isEmpty { s += " Tap Download to repair them." }
        verifyResult = s
        log("DOWNLOAD verify-all ok=\(ok) bad=\(bad) missing=\(missing)")
        // damaged files are removed so the next download replaces them (and the store check stops passing them)
        for b in bad { try? FileManager.default.removeItem(at: modelDirURL.appendingPathComponent(b)) }
    }

    /// Settings > Model files > Delete model: every file of the model folder (and the prefix cache, whose states belong
    /// to this model), then the empty folders again.
    func deleteModel() async {
        await cancel()
        let fm = FileManager.default
        if isLink(modelDirURL) { note = "The model folder is a link (a Mac run); not deleted."; return }
        for name in (try? fm.contentsOfDirectory(atPath: modelDirURL.path)) ?? [] {
            try? fm.removeItem(at: modelDirURL.appendingPathComponent(name))
        }
        // the prefix cache holds states of THIS model; a test folder (-modelDir) leaves the real model's states alone
        if LaunchArgs.value("-modelDir") == nil { try? fm.removeItem(at: PhoneStoreChatEngine.prefixDir) }
        verified = [:]
        SwapTarget.wanted = nil
        swap = nil
        ensureModelDir()
        files = []
        index = [:]
        manifest = nil
        phase = .idle
        verifyResult = nil
        cancelling = false
        freeBytes = Self.volumeFree()
        log("DOWNLOAD model deleted: \(modelDirURL.path)")
        await loadManifestQuietly()
    }

    // MARK: compressed store conversion (2026-11-07)

    var swapActive: Bool { swapTask != nil }

    /// Convert the installed store to the compressed layers (.codec) or back to the raw records (.raw), one layer at a
    /// time, in the background (the same URLSession as the model download). The target is remembered until it is done,
    /// so a relaunch continues it (on the set-up screen, SetupGate). Refused while a model download runs.
    func startSwap(_ target: SwapTarget) {
        guard swapTask == nil, !active else { return }
        if StoreStatusCodec.mode == "off" {
            swap = SwapStatus(target: target, phase: .failed("The app runs with -storeCodec off: the engine would not read the compressed layers."))
            return
        }
        SwapTarget.wanted = target
        swapTask = Task { await self.runSwap(target); self.swapTask = nil }
    }

    /// Pause the conversion: the file in transfer keeps its resume data; the next start continues from it. The store
    /// is complete at every step, so the chat works with the layers converted so far.
    func pauseSwap() async {
        guard swapTask != nil else { return }
        await pause()
    }

    private var swapPausing = false

    func refreshLayerFiles() { layerFiles = StoreStatus.check(dir: modelDirURL).layerFiles }

    private func runSwap(_ target: SwapTarget) async {
        swapMode = true
        swapPausing = false
        defer { swapMode = false; writeSwapReport() }
        let t0 = Date()
        var st = SwapStatus(target: target, phase: .running)
        swap = st
        func set(_ f: (inout SwapStatus) -> Void) { f(&st); swap = st }
        let m: Manifest
        do { m = try await fetchManifest() } catch {
            set { $0.phase = .failed("Could not get the file list: \(error.localizedDescription)") }
            log("SWAP failed: manifest: \(error.localizedDescription)")
            finishSwapTest(code: 1)
            return
        }
        manifest = m
        let codec = Dictionary(grouping: m.files.filter { $0.group == "codec" && $0.layer != nil }, by: { $0.layer! })
        let bins = Dictionary(uniqueKeysWithValues: m.files.compactMap { f in Self.binLayer(f).map { ($0, f) } })
        var layers = bins.keys.sorted().filter { codec[$0]?.count == 2 }
        if let only = DownloadConfig.swapLayers { layers = layers.filter { only.contains($0) } }
        guard !layers.isEmpty else {
            set { $0.phase = .failed("The file list has no compressed layers.") }
            finishSwapTest(code: 1)
            return
        }
        set { s in s.layers = layers.map { L in
            let pair = codec[L]!
            let bytes = target == .codec ? pair.reduce(0) { $0 + $1.size } : bins[L]!.size
            return LayerSwap(layer: L, bytes: bytes)
        } }
        log("SWAP start target=\(target.rawValue) layers=\(layers.map(String.init).joined(separator: ",")) engineOpen=\(SetupGate.engineCreated) "
            + "headroom=\(DownloadConfig.swapHeadroomBytes)")

        for (k, L) in layers.enumerated() {
            let pair = codec[L]!.sorted { $0.local.hasSuffix(".idx") && !$1.local.hasSuffix(".idx") }   // .idx first
            let idx = pair[0], psc = pair[1], bin = bins[L]!
            let base = modelDirURL.appendingPathComponent(String(format: "store4/layer_%02d", L))
            set { $0.current = k }
            // already in the target state (a relaunch, or a layer converted before)
            if layerIn(target, L, idx: idx, psc: psc, bin: bin) {
                set { $0.layers[k].state = .done; $0.layers[k].note = "already \(target == .codec ? "compressed" : "raw")" }
                continue
            }
            // space for this layer's downloads + the headroom
            let want = target == .codec ? [idx, psc] : [bin]
            // (a file already here at its full size is hashed in place, not downloaded: it needs no space)
            let need = want.filter { Self.resolvedSize(modelDirURL.appendingPathComponent($0.local).path) != $0.size }
                .reduce(Int64(0)) { $0 + $1.size }
            let free = Self.volumeFree() ?? 0
            freeBytes = free
            if need + DownloadConfig.swapHeadroomBytes > free {
                let held = st.heldBytes
                var msg = String(format: "Not enough space for layer %d: it needs %.2f GB (%.0f MB of files + %.0f MB to spare) and %.2f GB are free.",
                                 L, Double(need + DownloadConfig.swapHeadroomBytes) / 1e9, Double(need) / 1e6,
                                 Double(DownloadConfig.swapHeadroomBytes) / 1e6, Double(free) / 1e9)
                if held > 0 {
                    msg += String(format: " %.2f GB of replaced files are still open in the model: close the app and open it again; the conversion goes on from layer %d on the set-up screen, with the model closed.", Double(held) / 1e9, L)
                }
                set { $0.phase = .refused(msg); $0.layers[k].state = .waiting }
                log("SWAP refused layer \(L): need=\(need) headroom=\(DownloadConfig.swapHeadroomBytes) free=\(free) held=\(held)")
                finishSwapTest(code: 2)
                return
            }
            let tl = Date()
            set { $0.layers[k].state = .downloading }
            // 1. download + hash-check + move into place, one file after the other (the normal path: .download/<key>.part,
            //    SHA-256 against the manifest, rename). Codec: the .idx first, the .psc last, so the engine and the store check
            //    see the pair (and switch the layer to "psc") only when both are complete and verified.
            for f in want {
                if inPlace(f) { continue }
                set { $0.layers[k].file = f.name }
                let ok = await fetchFiles([f])
                if !ok {
                    if swapPausing {
                        set { $0.phase = .paused; $0.layers[k].state = .waiting }
                        log("SWAP paused at layer \(L) (\(f.local))")
                        finishSwapTest(code: 1)
                    } else {
                        set { $0.phase = .failed("Layer \(L): \(f.name) could not be downloaded (\(files.first?.state.label ?? "?")). Nothing was deleted; Resume tries again."); $0.layers[k].state = .failed }
                        log("SWAP failed layer \(L): \(f.local)")
                        finishSwapTest(code: 1)
                    }
                    return
                }
                if !storeGreen(L, step: "\(f.name) in place", &st) { set { _ in }; return }
            }
            set { $0.layers[k].state = .switching; $0.layers[k].file = "" }
            // 2. the layer reads the new files now; remove the old ones (only after the new ones are verified and in place)
            let fm = FileManager.default
            if target == .codec {
                guard Self.layerPresence(L).pair, inPlace(idx), inPlace(psc), StoreStatus.check(dir: modelDirURL).layerFiles[L] == "psc" else {
                    set { $0.phase = .failed("Layer \(L): the compressed pair is not in place; the .bin was kept."); $0.layers[k].state = .failed }
                    finishSwapTest(code: 1)
                    return
                }
                let freed = Self.resolvedSize(base.path + ".bin") ?? 0
                if isLink(base.appendingPathExtension("bin")) {
                    try? fm.removeItem(at: base.appendingPathExtension("bin"))   // a Mac test folder: the link only
                } else if freed > 0 {
                    try? fm.removeItem(at: base.appendingPathExtension("bin"))
                    if SetupGate.engineCreated { set { $0.heldBytes += freed } }   // the engine keeps the file open
                }
                verified[bin.local] = nil
            } else {
                guard inPlace(bin) else {
                    set { $0.phase = .failed("Layer \(L): the .bin is not in place; the compressed pair was kept."); $0.layers[k].state = .failed }
                    finishSwapTest(code: 1)
                    return
                }
                for f in [psc, idx] {   // the .psc first: the layer reads the .bin as soon as the pair is incomplete
                    let u = modelDirURL.appendingPathComponent(f.local)
                    let sz = isLink(u) ? 0 : (Self.resolvedSize(u.path) ?? 0)
                    try? fm.removeItem(at: u)
                    if SetupGate.engineCreated && sz > 0 { set { $0.heldBytes += sz } }
                    verified[f.local] = nil
                }
            }
            saveVerified()
            if !storeGreen(L, step: "old files removed", &st) { set { _ in }; return }
            let dt = Date().timeIntervalSince(tl)
            set { $0.layers[k].state = .done; $0.layers[k].seconds = dt; $0.doneBytes += $0.layers[k].bytes }
            log(String(format: "SWAP layer %d done (%@) in %.1f s; %d of %d", L, target == .codec ? "psc" : "bin", dt, k + 1, layers.count))
        }
        refreshLayerFiles()
        files = []
        index = [:]
        setManifest(m)
        let dt = Date().timeIntervalSince(t0)
        set { $0.phase = .complete; $0.seconds = dt; $0.current = nil }
        if DownloadConfig.swapLayers == nil { DownloadConfig.compressedStore = target == .codec }
        SwapTarget.wanted = nil
        let stc = StoreStatus.check(dir: modelDirURL)
        log(String(format: "SWAP DONE target=%@ %d layer(s) in %.1f s; store check %@; codec layers %d; held by the open model %lld B",
                   target.rawValue, layers.count, dt, stc.complete ? "complete" : "INCOMPLETE", stc.codecLayers.count, st.heldBytes))
        finishSwapTest(code: stc.complete ? 0 : 1)
        if !SetupGate.engineCreated && DownloadConfig.swapLayers == nil { onComplete?() }   // the set-up screen opens the chat
    }

    /// One file through the download path (adopt a live task, resume data, or a fresh request; SHA-256; move into place).
    /// Returns whether it is done.
    private func fetchFiles(_ fs: [ManifestFile]) async -> Bool {
        files = fs.map { FileProgress(file: $0) }
        index = Dictionary(uniqueKeysWithValues: files.enumerated().map { ($1.file.local, $0) })
        let live = await allTasks()
        let liveLocals = Set(live.compactMap { $0.taskDescription })
        for i in files.indices {
            let f = files[i].file
            if liveLocals.contains(f.local) { continue }
            let dst = modelDirURL.appendingPathComponent(f.local)
            let part = partURL(f)
            if let sz = Self.size(dst.path), sz == f.size {
                if isVerified(f, at: dst) { files[i].state = .done; files[i].bytes = f.size; files[i].source = "already here"; continue }
                if await verify(i, at: dst, moveTo: nil) { files[i].source = "already here"; continue }
            } else if let sz = Self.size(part.path), sz == f.size {
                if await verify(i, at: part, moveTo: dst) { files[i].source = "network (earlier launch)"; continue }
            }
            files[i].state = .queued
        }
        if !files.contains(where: { $0.state != .done }) { return true }
        pausing = false
        cancelling = false
        phase = .downloading
        startedAt = startedAt ?? Date()
        startTimer()
        for t in live {
            guard let local = t.taskDescription, let i = index[local], let dt = t as? URLSessionDownloadTask else { t.cancel(); continue }
            if files[i].state == .done { t.cancel(); continue }
            if t.state == .suspended { t.resume() }
            tasks[local] = dt
            files[i].state = .downloading
            files[i].bytes = t.countOfBytesReceived
            files[i].resumedFrom = t.countOfBytesReceived
            files[i].source = "adopted"
            log("SWAP adopt \(local) task \(t.taskIdentifier) at \(t.countOfBytesReceived) B")
        }
        for i in files.indices where files[i].state == .queued || files[i].state == .paused { launch(i) }
        return await withCheckedContinuation { (c: CheckedContinuation<Bool, Never>) in fetchWaiter = c }
    }

    /// The file is in the model folder with the manifest's size and a matching verified record (the download path's).
    private func inPlace(_ f: ManifestFile) -> Bool {
        let u = modelDirURL.appendingPathComponent(f.local)
        guard let sz = Self.resolvedSize(u.path), sz == f.size else { return false }
        if isLink(u) { return true }   // a Mac test folder's link to the Mac's copy
        return isVerified(f, at: u)
    }

    private func layerIn(_ t: SwapTarget, _ L: Int, idx: ManifestFile, psc: ManifestFile, bin: ManifestFile) -> Bool {
        let p = Self.layerPresence(L)
        return t == .codec ? (p.pair && !p.bin && inPlace(idx) && inPlace(psc)) : (p.bin && !p.pair && inPlace(bin))
    }

    /// The store check after a step: complete, or the conversion stops here (nothing more is deleted).
    private func storeGreen(_ L: Int, step: String, _ st: inout SwapStatus) -> Bool {
        let c = StoreStatus.check(dir: modelDirURL)
        layerFiles = c.layerFiles
        st.checks += 1
        if c.complete {
            log("SWAP layer \(L): \(step); store check complete, layer reads \(c.layerFiles[L]), \(c.codecLayers.count) compressed layer(s)")
            return true
        }
        st.phase = .failed("Layer \(L): the store check failed after \(step): " + c.problems.prefix(3).joined(separator: "; "))
        if let k = st.current { st.layers[k].state = .failed }
        swap = st
        log("SWAP store check FAILED after layer \(L) \(step): \(c.problems)")
        finishSwapTest(code: 1)
        return false
    }

    private func finishSwapTest(code: Int32) {
        writeSwapReport()
        if LaunchArgs.value("-codecSwap") != nil && LaunchArgs.has("-exitWhenDone") {
            log("DONE")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { exit(code) }
        }
    }

    private func writeSwapReport() {
        guard let s = swap else { return }
        let d: [String: Any] = [
            "time": ISO8601DateFormatter().string(from: Date()),
            "target": s.target.rawValue, "phase": s.phase.label, "model_dir": modelDirURL.path,
            "seconds": s.seconds, "held_by_open_model_bytes": s.heldBytes, "store_checks": s.checks,
            "free_bytes": freeBytes ?? -1, "headroom_bytes": DownloadConfig.swapHeadroomBytes,
            "network_bytes_this_launch": sessionBytes, "codec_layers": StoreStatus.check(dir: modelDirURL).codecLayers,
            "layers": s.layers.map { ["layer": $0.layer, "state": $0.state.rawValue, "bytes": $0.bytes, "seconds": $0.seconds, "note": $0.note] },
        ]
        if let data = try? JSONSerialization.data(withJSONObject: d, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: DownloadConfig.swapReportURL, options: .atomic)
        }
    }

    // MARK: speed, report

    private func startTimer() {
        guard timer == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    private func stopTimer() { timer?.invalidate(); timer = nil }

    private func tick() {
        let now = Date()
        speedSamples.append((now, sessionBytes))
        speedSamples.removeAll { now.timeIntervalSince($0.0) > 8 }
        if let first = speedSamples.first, now.timeIntervalSince(first.0) > 0.9 {
            bytesPerSecond = Double(sessionBytes - first.1) / now.timeIntervalSince(first.0)
        }
        if now.timeIntervalSince(lastLog) >= 5 {
            lastLog = now
            let act = files.filter { $0.state == .downloading }.map { String(format: "%@ %.0f%%", $0.file.name, 100 * $0.fraction) }
            log(String(format: "DOWNLOAD progress %.3f of %.3f GB (%.1f%%), %.1f MB/s, eta %@; active: %@", Double(doneBytes) / 1e9,
                       Double(totalBytes) / 1e9, 100 * fraction, bytesPerSecond / 1e6, etaSeconds.map { formatDuration($0) } ?? "-",
                       act.joined(separator: ", ")))
        }
    }

    private func writeReport() {
        let dt = startedAt.map { Date().timeIntervalSince($0) } ?? 0
        var d: [String: Any] = [
            "time": ISO8601DateFormatter().string(from: Date()),
            "phase": "\(phase)",
            "manifest_url": DownloadConfig.manifestURL.absoluteString,
            "model_dir": modelDirURL.path,
            "only": DownloadConfig.only ?? [],
            "background_session": DownloadConfig.useBackgroundSession,
            "cellular": cellular,
            "free_bytes": freeBytes ?? -1,
            "fake_free_bytes": DownloadConfig.fakeFreeBytes ?? -1,
            "headroom_bytes": DownloadConfig.headroomBytes,
            "total_bytes": totalBytes,
            "network_bytes_this_launch": sessionBytes,
            "seconds_since_start": dt,
            "mb_per_s_this_launch": dt > 0 ? Double(sessionBytes) / 1e6 / dt : 0,
            "store_complete": StoreStatus.check(dir: modelDirURL).complete,
        ]
        if case .refused(let m) = phase { d["refusal"] = m }
        if case .failed(let m) = phase { d["failure"] = m }
        d["files"] = files.map { p -> [String: Any] in
            var x: [String: Any] = ["local": p.file.local, "size": p.file.size, "state": p.state.label, "attempts": p.attempts,
                                    "resumed_from": p.resumedFrom, "network_bytes": p.downloadedBytes, "source": p.source]
            if let ok = p.shaOK { x["sha256_ok"] = ok }
            if let s = p.started, let e = p.finished {
                x["seconds"] = e.timeIntervalSince(s)
                x["mb_per_s"] = e.timeIntervalSince(s) > 0 ? Double(p.downloadedBytes) / 1e6 / e.timeIntervalSince(s) : 0
            }
            return x
        }
        if let data = try? JSONSerialization.data(withJSONObject: d, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: DownloadConfig.reportURL, options: .atomic)
            log("DOWNLOAD report \(DownloadConfig.reportURL.path)")
        }
    }

    // MARK: files

    func partURL(_ f: ManifestFile) -> URL { DownloadConfig.stateDir.appendingPathComponent(f.key + ".part") }
    func resumeURL(_ f: ManifestFile) -> URL { DownloadConfig.stateDir.appendingPathComponent(f.key + ".resume") }

    nonisolated static func size(_ path: String) -> Int64? {
        guard let a = try? FileManager.default.attributesOfItem(atPath: path), (a[.type] as? FileAttributeType) == .typeRegular
        else { return nil }
        return (a[.size] as? NSNumber)?.int64Value
    }

    nonisolated static func mtime(_ path: String) -> Double? {
        ((try? FileManager.default.attributesOfItem(atPath: path))?[.modificationDate] as? Date)?.timeIntervalSince1970
    }

    /// The byte offset a resume data blob continues from (its plist's NSURLSessionResumeBytesReceived), for the log.
    nonisolated static func resumeOffset(_ data: Data) -> Int64 {
        let key = "NSURLSessionResumeBytesReceived"
        // older systems: a plain plist dictionary; current ones: a keyed archive of one
        if let p = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
           let n = p[key] as? NSNumber { return n.int64Value }
        if let u = try? NSKeyedUnarchiver(forReadingFrom: data) {
            u.requiresSecureCoding = false
            defer { u.finishDecoding() }
            // CFNetwork archives under the literal key "NSKeyedArchiveRootObjectKey" (Foundation's constant is "root")
            for root in ["NSKeyedArchiveRootObjectKey", NSKeyedArchiveRootObjectKey] {
                if let d = u.decodeObject(forKey: root) as? [String: Any], let n = d[key] as? NSNumber { return n.int64Value }
            }
        }
        return -1
    }

    private static var verifiedURL: URL { DownloadConfig.stateDir.appendingPathComponent("verified.json") }
    private static func loadVerified() -> [String: VerifiedRecord] {
        guard let d = try? Data(contentsOf: verifiedURL) else { return [:] }
        return (try? JSONDecoder().decode([String: VerifiedRecord].self, from: d)) ?? [:]
    }
    private func saveVerified() {
        try? FileManager.default.createDirectory(at: DownloadConfig.stateDir, withIntermediateDirectories: true)
        if let d = try? JSONEncoder().encode(verified) { try? d.write(to: Self.verifiedURL, options: .atomic) }
    }
}

func formatDuration(_ s: Double) -> String {
    if s < 90 { return "\(Int(s.rounded())) s" }
    if s < 5400 { return "\(Int((s / 60).rounded())) min" }
    return String(format: "%.1f h", s / 3600)
}

/// The URLSession delegate. Runs on the session's serial queue; the finished file must be moved before
/// didFinishDownloadingTo returns, so that happens here; everything else goes to the main actor.
final class SessionDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    enum Outcome { case none, moved, badStatus(Int), moveFailed(String) }
    weak var owner: ModelDownloader?
    private var outcomes: [Int: Outcome] = [:]   // task id -> outcome (delegate queue only)
    private var lastWrite: [Int: Date] = [:]

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let local = downloadTask.taskDescription else { return }
        let status = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else { outcomes[downloadTask.taskIdentifier] = .badStatus(status); return }
        let part = DownloadConfig.stateDir.appendingPathComponent(local.replacingOccurrences(of: "/", with: "__") + ".part")
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: DownloadConfig.stateDir, withIntermediateDirectories: true)
            try? fm.removeItem(at: part)
            try fm.moveItem(at: location, to: part)
            outcomes[downloadTask.taskIdentifier] = .moved
        } catch {
            outcomes[downloadTask.taskIdentifier] = .moveFailed(error.localizedDescription)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard let local = downloadTask.taskDescription else { return }
        Task { @MainActor [weak owner] in owner?.didWrite(local, written: bytesWritten, total: totalBytesWritten) }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didResumeAtOffset fileOffset: Int64,
                    expectedTotalBytes: Int64) {
        guard let local = downloadTask.taskDescription else { return }
        Task { @MainActor [weak owner] in owner?.didResume(local, offset: fileOffset) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let local = task.taskDescription else { return }
        let outcome = outcomes.removeValue(forKey: task.taskIdentifier) ?? Outcome.none
        let resume = (error as NSError?)?.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        let id = task.taskIdentifier
        Task { @MainActor [weak owner] in owner?.didFinish(local, taskID: id, outcome: outcome, error: error, resumeData: resume) }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor [weak owner] in owner?.sessionEventsFinished() }
    }
}

// MARK: compressed store conversion types

enum SwapTarget: String {
    case codec, raw
    static let key = "codec.swap.want"
    /// The conversion that was started and has not finished (or `-codecSwap codec|raw`, a test).
    static var wanted: SwapTarget? {
        get {
            if let a = LaunchArgs.value("-codecSwap") { return SwapTarget(rawValue: a) }
            return UserDefaults.standard.string(forKey: key).flatMap { SwapTarget(rawValue: $0) }
        }
        set {   // a -codecSwap test leaves the setting alone (a refused test would otherwise resume at the next normal launch)
            if LaunchArgs.value("-codecSwap") == nil { UserDefaults.standard.set(newValue?.rawValue, forKey: key) }
        }
    }
}

struct LayerSwap: Identifiable {
    enum State: String { case pending, waiting, downloading, switching, done, failed }
    var layer: Int
    var bytes: Int64          // what this layer downloads
    var state: State = .pending
    var file = ""             // the file in transfer
    var seconds = 0.0
    var note = ""
    var id: Int { layer }
}

struct SwapStatus {
    enum Phase: Equatable {
        case running, paused, complete
        case refused(String), failed(String)
        var label: String {
            switch self {
            case .running: return "running"
            case .paused: return "paused"
            case .complete: return "complete"
            case .refused(let m): return "refused: " + m
            case .failed(let m): return "failed: " + m
            }
        }
    }
    var target: SwapTarget
    var phase: Phase
    var layers: [LayerSwap] = []
    var current: Int?         // index into layers
    var doneBytes: Int64 = 0
    var heldBytes: Int64 = 0  // removed while the model had them open: freed when the app is next opened
    var checks = 0            // store checks passed (after every step)
    var seconds = 0.0
    var doneCount: Int { layers.filter { $0.state == .done }.count }
    var totalBytes: Int64 { layers.reduce(0) { $0 + $1.bytes } }
}
