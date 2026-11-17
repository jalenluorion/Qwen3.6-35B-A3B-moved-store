import Foundation
import os

/// Where the app downloads the model from, and the downloader's knobs. The one place to change the repo.
///
/// The hub repo holds the slim layout exactly as `Documents/model/` wants it (store4/, cache/, embeddings, the small
/// files) plus `manifest.json` (tools/make_manifest.py: every file's path, local path, size, sha256 and `required`).
/// A file's URL is the manifest URL's folder + the file's `path`, so a local server works the same way
/// (tools/serve_store.py serves qwen35/manifest.json and every file of it from where it lives on the Mac).
///
/// Launch flags (tests; all logged as `DOWNLOAD config ...`):
///   -manifestURL URL     another manifest (e.g. http://127.0.0.1:8765/manifest.json); its folder is the file base URL
///   -downloadOnly LIST   only these files: comma-separated local paths (store4/layer_00.bin) or `small` (every required
///                        file under 50 MB: the config/tokenizer files and store4/meta.json); the store stays incomplete
///   -autoDownload        start without a tap (implied by -downloadOnly)
///   -fakeFreeBytes N     pretend the volume has N bytes free (the free-space refusal test)
///   -downloadCellular 1  allow cellular for this launch (the setting is off by default)
///   -foregroundDownload  a default (in-process) URLSession instead of the background one
///   -deleteModel         run the Delete model action on the model folder (-modelDir) and stop (phone test cleanup)
///   -pauseAfterBytes N   pause (resume data saved) after N bytes this launch; with -exitWhenDone exit after it
///   -exitWhenDone        exit when the selected files are done, refused or failed (writes download.json first)
///   -codecSwap codec|raw convert the installed store to the compressed layers (or back) at launch, on the set-up screen
///                        (ModelDownloader.startSwap); with -exitWhenDone exit when done (0), refused (2) or failed (1)
///   -fakeAvailableMemory N  pretend os_proc_available_memory is N bytes (the low-memory message)
///   -demoFlow            the first run end to end with a tiny manifest (sim_onboarding.sh): chat + setup card after it
///   -codecSwapLayers L   only these layers (comma-separated, e.g. 0,1,2,3); the compressed-store setting is left alone
enum DownloadConfig {
    static let repo = "jalenluorion/phone-store-qwen36-35b-a3b"
    static let revision = "main"
    static let defaultManifestURL = URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/manifest.json")!
    static var manifestURL: URL { LaunchArgs.value("-manifestURL").flatMap { URL(string: $0) } ?? defaultManifestURL }
    /// The folder the manifest's `path`s are relative to.
    static var baseURL: URL { manifestURL.deletingLastPathComponent() }
    static var repoPageURL: URL { URL(string: "https://huggingface.co/\(repo)")! }

    /// Shown before the manifest has been fetched (the slim layout: 51 files, 20,597,475,264 B).
    static let approxRequiredBytes: Int64 = 20_597_475_264
    /// Free space the volume must keep after the download: remaining bytes + this, or the start is refused.
    static let headroomBytes: Int64 = 1_000_000_000
    /// Files transferred at once (URLSessionConfiguration.httpMaximumConnectionsPerHost; the hub's CDN is one host).
    static let concurrentFiles = 3
    /// Retries per file (network errors, bad status, sha256 mismatch) before it is marked failed; backoff 2, 4, 8 ... 60 s.
    static let maxAttempts = 8
    static func backoff(attempt: Int) -> Double { min(60, pow(2, Double(max(1, attempt)))) }

    /// Compressed store (docs/CODEC-STORE.md): free space the volume must keep while a layer is converted (its files
    /// are downloaded before the old ones are deleted: peak extra space one layer, ~394 MB for a .psc + .idx pair).
    static let swapHeadroomBytes: Int64 = 400_000_000
    /// The slim layout with every layer compressed (required 20,597,475,264 - raw 18,119,393,280 + codec 15,672,533,424 B).
    static let approxCompressedBytes: Int64 = 18_150_615_408
    static let compressedKey = "download.compressed"
    /// Layers that have neither their .bin nor their .psc pair yet are downloaded compressed (set by a fresh install's
    /// choice and by a finished conversion).
    /// Default on (2026-11-07, docs/APP-COPY.md): a fresh install downloads the compressed files (fewer bytes on the
    /// phone and per token, the same answers); Settings > Model > Compressed files switches an installed store.
    /// The compressed store is always the smaller download, so when it does not fit, nothing does.
    static var compressedStore: Bool {
        get { (UserDefaults.standard.object(forKey: compressedKey) as? Bool) ?? true }
        set { UserDefaults.standard.set(newValue, forKey: compressedKey) }
    }
    /// What a fresh install downloads (before the manifest is in).
    static var approxStoreBytes: Int64 { compressedStore ? approxCompressedBytes : approxRequiredBytes }

    /// This iPhone gives the app too little memory to load the model at Budget.minCap (the same budget the load uses),
    /// so the set-up screen says so instead of offering an 18 GB download. iOS only (os_proc_available_memory is 0 on the
    /// Mac and in the simulator). `-fakeAvailableMemory N` pretends N bytes (the simulator screenshots).
    static var memoryTooLow: Bool {
        let avail = LaunchArgs.value("-fakeAvailableMemory").flatMap { Double($0) } ?? Double(os_proc_available_memory())
        guard avail > 0, only == nil else { return false }
        let staging = compressedStore ? Budget.codecStagingBytes : 0
        return Budget.cap(availableBytes: avail - staging, recordBytes: Budget.defaultRecordBytes) < Budget.minCap
    }

    /// `-demoFlow` (simulator only in practice): the whole first run with a tiny manifest: the finished download opens the
    /// chat although the store check fails, and the stub engine plays the one-time setup's card (ChatModel.launchApp).
    static var demoFlow: Bool { LaunchArgs.has("-demoFlow") }
    /// `-codecSwapLayers 0,1,2,3`: nil = every layer.
    static var swapLayers: [Int]? {
        LaunchArgs.value("-codecSwapLayers").map { $0.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) } }
    }
    static var swapReportURL: URL { docsURL.appendingPathComponent("codec_swap.json") }

    /// Derived from the bundle id, so every install under its own bundle id has its own background session.
    static let backgroundSessionID = (Bundle.main.bundleIdentifier ?? "phonestore") + ".model-download"
    static let backgroundCellularSessionID = (Bundle.main.bundleIdentifier ?? "phonestore") + ".model-download.cellular"
    static var useBackgroundSession: Bool { !LaunchArgs.has("-foregroundDownload") }

    /// `-downloadOnly LIST`: nil = every required file.
    static var only: [String]? {
        LaunchArgs.value("-downloadOnly").map { $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } }
    }
    static var autoStart: Bool { LaunchArgs.has("-autoDownload") || only != nil }
    static var fakeFreeBytes: Int64? { LaunchArgs.value("-fakeFreeBytes").flatMap { Int64($0) } }
    static let smallLimit: Int64 = 50_000_000

    static let cellularKey = "download.cellular"
    static var cellularAllowed: Bool {
        if let v = LaunchArgs.value("-downloadCellular") { return v == "1" }
        return UserDefaults.standard.bool(forKey: cellularKey)
    }

    /// Download bookkeeping inside the model folder (excluded from backup with it): partial files, resume data, the
    /// cached manifest, the verified-files record.
    static var stateDir: URL { modelDirURL.appendingPathComponent(".download", isDirectory: true) }
    static var reportURL: URL { docsURL.appendingPathComponent("download.json") }

    static var summary: String {
        "manifest=\(manifestURL.absoluteString) only=\(only?.joined(separator: ",") ?? "all-required") auto=\(autoStart) "
            + "fakeFree=\(fakeFreeBytes.map(String.init) ?? "-") cellular=\(cellularAllowed) background=\(useBackgroundSession) "
            + "modelDir=\(modelDirURL.path)"
    }
}
