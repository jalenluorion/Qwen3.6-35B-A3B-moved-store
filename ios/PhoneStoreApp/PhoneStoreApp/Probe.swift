import Foundation
import Darwin
import os
import PhoneStore

// Launch-time probe: memory, disk, thermal state, the slab cap, and the store's completeness.
// Written to Documents/probe.json on every launch.

func log(_ s: String) {
    print(s)
    fflush(stdout)
}

/// Launch arguments with a value (`-name value`). Every override below is written to probe.json.
enum LaunchArgs {
    static func value(_ k: String) -> String? {
        let a = CommandLine.arguments
        guard let i = a.firstIndex(of: k), i + 1 < a.count else { return nil }
        return a[i + 1]
    }
    static func has(_ k: String) -> Bool { CommandLine.arguments.contains(k) }
    static func dir(_ k: String) -> URL? { value(k).map { URL(fileURLWithPath: $0, isDirectory: true).standardizedFileURL } }
}

/// `-docs DIR` replaces the app's Documents folder (Mac runs: a run folder whose `model` is a link to qwen35/).
let docsURL = LaunchArgs.dir("-docs") ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
/// `-modelDir DIR` replaces Documents/model; a relative DIR is inside Documents/ (`-modelDir model_dl` on the phone,
/// whose container path is not known in advance).
let modelDirURL: URL = {
    guard let v = LaunchArgs.value("-modelDir") else { return docsURL.appendingPathComponent("model", isDirectory: true) }
    return v.hasPrefix("/") ? URL(fileURLWithPath: v, isDirectory: true).standardizedFileURL
                            : docsURL.appendingPathComponent(v, isDirectory: true).standardizedFileURL
}()
/// Where the engine keeps its one-time conversions (int4 resident 1.10 GB, int8 routers 0.34 GB): `-cacheDir DIR`, else
/// Documents/model/cache. The slim layout ships them there (ios/copy_store_to_phone.sh copies native/cache/*.safetensors
/// to Documents/model/cache/), so the phone never converts the bf16 files and needs neither resident.safetensors nor
/// routers_k4.npz.
let nativeCacheURL = LaunchArgs.dir("-cacheDir") ?? modelDirURL.appendingPathComponent("cache", isDirectory: true)
let probeURL = docsURL.appendingPathComponent("probe.json")
let autotestURL = docsURL.appendingPathComponent("autotest.json")
/// One MemProbe sample per line (JSON), appended as it is taken, so the samples survive a jetsam kill. Emptied at launch.
let memLogURL = docsURL.appendingPathComponent("mem.jsonl")
/// The process start, for MemSample.t (touched in ChatModel.init).
let appStart = Date()

/// One memory reading. Bytes; GB in the console line = 1e9 B.
struct MemSample: Codable, Sendable {
    var label: String                 // "launch", "after load", "reply 1 token 10", "memory warning", "end of reply 1", ...
    var t: Double                     // seconds since launch
    var time: String
    var reply: Int?                   // autotest / chat reply number (1-based) in progress, if any
    var token: Int?                   // generated tokens of that reply so far
    /// task_vm_info.phys_footprint: what jetsam charges the app (dirty + compressed + IOKit/Metal memory it owns).
    var physFootprint: Int
    /// task_vm_info.ledger_phys_footprint_peak: the footprint's high-water mark since launch.
    var physFootprintPeak: Int
    var residentSize: Int             // task_vm_info.resident_size (includes clean file-backed pages; not what jetsam uses)
    var compressed: Int               // task_vm_info.compressed
    /// task_vm_info.external: file-backed resident pages (the memory-mapped weights live here, outside the footprint).
    var external: Int? = nil
    /// task_events_info: page faults and page-ins since launch (mapped weights dropped under pressure come back as page-ins).
    var faults: Int? = nil
    var pageins: Int? = nil
    /// os_proc_available_memory(): how much more the app may take before jetsam (iOS; 0 on the Mac).
    var osProcAvailable: Int
    /// physFootprint + osProcAvailable: the app's memory limit as iOS sees it now (nil on the Mac, where available is 0).
    var limitEstimate: Int?
    var mlxActive: Int?               // MLX allocator: bytes in live arrays (resident + routers + slabs + KV/state + temporaries)
    var mlxPeak: Int?                 // MLX allocator high-water mark since launch
    var mlxCache: Int?                // MLX allocator: freed buffers kept for reuse (bounded by the cache limit)
    var thermal: String

    var line: String {
        func g(_ b: Int) -> String { String(format: "%.3f", Double(b) / 1e9) }
        func g(_ b: Int?) -> String { b.map { g($0) } ?? "n/a" }
        return "MEM \(label): footprint=\(g(physFootprint)) peak=\(g(physFootprintPeak)) avail=\(g(osProcAvailable)) limit~\(g(limitEstimate)) GB"
            + " | mlx active=\(g(mlxActive)) peak=\(g(mlxPeak)) cache=\(g(mlxCache)) GB"
            + " | resident=\(g(residentSize)) file-backed=\(g(external)) compressed=\(g(compressed)) GB | faults=\(faults ?? -1) pageins=\(pageins ?? -1) | thermal=\(thermal)"
    }
}

/// The process's memory as jetsam and MLX see it. Cheap (a task_info call and three counter reads); any thread.
enum MemProbe {
    static func vmInfo() -> task_vm_info_data_t? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info : nil
    }

    static func eventsInfo() -> (faults: Int, pageins: Int)? {
        var ev = task_events_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_events_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &ev) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_EVENTS_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? (Int(ev.faults), Int(ev.pageins)) : nil
    }

    static func read(_ label: String, reply: Int? = nil, token: Int? = nil) -> MemSample {
        let vm = vmInfo()
        let ev = eventsInfo()
        let avail = Int(os_proc_available_memory())
        let fp = Int(vm?.phys_footprint ?? 0)
        let mlx = PhoneStoreEngine.mlxMemory()  // nil until the engine has loaded (MLX not started)
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return MemSample(label: label, t: Date().timeIntervalSince(appStart), time: iso.string(from: Date()), reply: reply,
                         token: token, physFootprint: fp, physFootprintPeak: Int(vm?.ledger_phys_footprint_peak ?? 0),
                         residentSize: Int(vm?.resident_size ?? 0), compressed: Int(vm?.compressed ?? 0),
                         external: vm.map { Int($0.external) }, faults: ev?.faults, pageins: ev?.pageins,
                         osProcAvailable: avail, limitEstimate: avail > 0 ? fp + avail : nil,
                         mlxActive: mlx?.active, mlxPeak: mlx?.peak, mlxCache: mlx?.cache,
                         thermal: thermalName(ProcessInfo.processInfo.thermalState))
    }

    static func resetLog() { try? Data().write(to: memLogURL, options: .atomic) }

    static func append(_ m: MemSample) {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        e.outputFormatting = [.sortedKeys]
        guard var d = try? e.encode(m) else { return }
        d.append(0x0A)
        if let h = try? FileHandle(forWritingTo: memLogURL) {
            defer { try? h.close() }
            _ = try? h.seekToEnd()
            try? h.write(contentsOf: d)
        } else {
            try? d.write(to: memLogURL)
        }
    }
}

/// Memory budget for one cap (docs/G4-runtime.md, Phase B). GB = 1e9 B.
enum Budget {
    static let layers = 40
    static let experts = 256
    static let topK = 8
    static let defaultRecordBytes = 1_769_472
    static let residentBytesLoaded = 1.10e9  // int4 resident model, read into memory (-mappedWeights 0)
    static let routerBytesLoaded = 0.34e9    // int8 lag-4 routers, read into memory
    /// The resident model and the routers: 1.10 + 0.34 GB of physical memory however they are held. Memory-mapped
    /// (-mappedWeights 1) they leave phys_footprint (15 Plus, 2026-11-06: 1.47 GB after load instead of 2.89 at cap 19),
    /// but the GPU keeps them resident (wired by MLX's residency set), so they take the same RAM: the budget's cap 37
    /// for mapped weights was killed by jetsam (vm-pageshortage, 4.7 GB wired) during the load, twice. The cap is
    /// planned for the bytes in RAM, so a mapped load (or its fallback) never changes it (docs/G4-runtime.md
    /// "Mapped weights on the phone").
    static var residentBytes: Double { residentBytesLoaded }
    static var routerBytes: Double { routerBytesLoaded }
    static let stateBytes = 0.15e9         // KV + recurrent state + MLX cache
    /// Margin measured on the iPhone 15 Plus (2026-11-04, -memLimitTest): iOS warns when the footprint reaches
    /// 2.64 GB of the 3.20 GB it reports (0.58 GB headroom) without the increased-memory-limit entitlement, and 3.44 GB
    /// of 4.28 GB (0.85 GB) with it; the footprint runs 0.1-0.3 GB above the plan. 0.8 / 1.2 GB keep the plan under the
    /// warning point (cap 10 / 19 on the 15 Plus instead of 14 / 28).
    static var marginBytes: Double { hasIncreasedMemoryLimit ? 1.2e9 : 0.8e9 }
    static var hasIncreasedMemoryLimit: Bool {
        (Bundle.main.object(forInfoDictionaryKey: "PSIncreasedMemoryLimit") as? String) == "YES"
    }
    static let stagingRecordsPerCap = 3    // staging = 3 x cap records
    /// Compressed layers (.psc): the engine's codec staging (EngineOptions.codecStaging = 24 slots of ~1.67 MB).
    static let codecStagingBytes = 0.04e9
    static let minCap = topK               // refuse to load below this
    static let g4FloorCap = 16             // 2 x top-8, the G4 plan's floor; below it the app warns

    static var fixedBytes: Double { residentBytes + routerBytes + stateBytes + marginBytes }
    /// The layer caps for a uniform-equivalent cap: `-layerCaps auto` (the default when the cap comes from the budget):
    /// LayerCaps.auto(total: 40 x cap), the per-layer allocation of the same total (native/Sources/PhoneStore/
    /// LayerCaps.swift); "uniform": nil (cap in every layer); else a list of 40 ints (or a file holding one).
    /// This session's caps: `-layerCaps auto|uniform|LIST`, default auto when the cap comes from the budget and uniform
    /// with `-cap N` (the Mac references were made at uniform caps).
    static func sessionLayerCaps(cap: Int) -> (caps: [Int]?, label: String) {
        let spec = LaunchArgs.value("-layerCaps") ?? (LaunchArgs.value("-cap") != nil ? "uniform" : "auto")
        return layerCaps(spec: spec, cap: cap)
    }
    static func layerCaps(spec: String, cap: Int) -> (caps: [Int]?, label: String) {
        if spec == "uniform" || cap < minCap { return (nil, "uniform") }
        if spec == "auto" {
            guard let c = try? LayerCaps.auto(total: cap * layers) else { return (nil, "uniform (auto refused)") }
            return (c, "auto:\(cap * layers)")
        }
        return (try? LayerCaps.parse(spec)) ?? (nil, "uniform (bad -layerCaps \(spec))")
    }
    static func perCapBytes(recordBytes: Int) -> Double { Double(recordBytes) * Double(layers + stagingRecordsPerCap) }
    static func cap(availableBytes: Double, recordBytes: Int) -> Int {
        let c = Int(((availableBytes - fixedBytes) / perCapBytes(recordBytes: recordBytes)).rounded(.down))
        return max(0, min(experts, c))
    }
    static func planBytes(cap: Int, recordBytes: Int) -> Double {
        fixedBytes - marginBytes + Double(cap) * perCapBytes(recordBytes: recordBytes)
    }

    /// Batched prefill chunk activations (2026-11-07, docs/G4-runtime.md "Chunks as large as memory allows"): the auto
    /// chunk (EngineOptions.prefillAutoBytes) is the largest whose activations (PrefillMemory, native Prefill.swift) fit
    /// the room the plan leaves under iOS's first memory warning:
    ///   limit (footprint + os_proc_available_memory) - warning distance (0.85 GB with the increased-memory-limit
    ///   entitlement, 0.58 GB without: the 2026-11-04 -memLimitTest) - the plan at this cap - 0.20 GB for the footprint's
    ///   overshoot over the plan (0.1-0.3 GB measured; 0.0-0.1 GB after load on 2026-11-06).
    /// The margin (1.2 / 0.8 GB) is the warning distance + 0.35 / 0.22 GB, so this is the cap's rounding leftover + 0.15
    /// / 0.02 GB. 15 Plus, entitled, cap 19: 4.295 - 0.85 - 3.036 - 0.20 = 0.209 GB (auto chunk 752 on an empty
    /// context, 2 x 512 for a 1024-token input; 400-500 after 4096 tokens). Not on iOS (the Mac: os_proc_available_memory reads 0): macDefault.
    static let warnDistanceEntitled = 0.85e9, warnDistance = 0.58e9
    static let overshootBytes = 0.20e9
    static let prefillAutoMacDefault = 0.5e9
    static let prefillAutoMin = 0.02e9  // below this the auto chunk is the 16-position floor anyway
    static func prefillAutoBytes(cap: Int, recordBytes: Int, limitBytes: Double? = nil) -> Double {
        let lim = limitBytes ?? {
            let avail = Double(os_proc_available_memory())
            guard avail > 0 else { return 0 }
            return avail + Double(MemProbe.vmInfo()?.phys_footprint ?? 0)
        }()
        guard lim > 0 else { return prefillAutoMacDefault }
        let warn = hasIncreasedMemoryLimit ? warnDistanceEntitled : warnDistance
        return max(prefillAutoMin, lim - warn - planBytes(cap: cap, recordBytes: recordBytes) - overshootBytes)
    }
}

struct BudgetReport: Codable {
    var recordBytes: Int
    var residentBytes: Double
    var routerBytes: Double
    var stateBytes: Double
    var marginBytes: Double
    var slabBytesPerCap: Double       // 40 x record
    var stagingBytesPerCap: Double    // 3 x record
    var formula: String
    var planBytes: Double             // everything but the margin, at the chosen cap
    var minCap: Int
    var g4FloorCap: Int
    var meetsG4Floor: Bool
    var enoughForMinCap: Bool
    /// How this plan counts the resident model and the routers: "memory-mapped", "read into memory", or
    /// "FALLBACK: ..." with the reason when a mapped load failed this launch and the cap was re-planned (2026-11-06).
    var mappedWeights: String
    /// The auto prefill chunk's activation budget at this cap (Budget.prefillAutoBytes; 2026-11-07).
    var prefillAutoBytes: Double? = nil
    var prefillAutoFormula: String? = nil
}

struct StoreFile: Codable {
    var path: String
    var expectedBytes: Int
    var exact: Bool                   // exact size required (big files); else any non-zero size
    var actualBytes: Int?
    var ok: Bool
    var problem: String?
}

struct StoreStatus: Codable {
    var modelDir: String
    /// "full": resident.safetensors in the model folder (embedding rows read from it; routers_k4.npz required; the cache
    /// is built on first load if absent). "slim": no resident.safetensors; embeddings.safetensors + the converted cache
    /// (resident_q4 + routers_k4_r8 in the native cache folder) required. The engine picks the same way
    /// (ModelLayout.detect in native/Sources/PhoneStore/Weights.swift).
    var layout: String
    var cacheDir: String
    var complete: Bool
    var problems: [String]
    var files: [StoreFile]
    var presentBytes: Int
    var expectedBytes: Int
    var recordBytes: Int
    var layers: Int
    var experts: Int
    var metaSource: String            // "store4/meta.json" or "built-in"
    var optionalPresent: [String]
    /// Per layer the file the engine will read: "psc" (layer_XX.psc + .idx, compressed, decoded on the GPU; with
    /// -storeCodec auto, the default) or "bin" (raw records). codecLayers: the "psc" layers.
    var layerFiles: [String]
    var codecLayers: [Int]

    /// The native cache's files (native/cache on the Mac, 2026-11-03): exact sizes, a short copy would load garbage. Each
    /// has a page-aligned copy for the memory-mapped load (`.mmap.safetensors`, 2026-11-04, written once by the engine
    /// from the file; native/Sources/PhoneStore/Weights.swift MappedWeights): either one is enough.
    static let cacheFiles: [(String, Int, mmap: Int)] = [
        ("resident_q4.safetensors", 1_103_484_684, 1_109_917_696),
        ("routers_k4_r8.safetensors", 334_565_291, 337_592_320),
    ]
    static func mmapName(_ f: String) -> String { (f as NSString).deletingPathExtension + ".mmap.safetensors" }

    static func check(dir: URL, cacheDir: URL = nativeCacheURL) -> StoreStatus {
        let fm = FileManager.default
        // follows links (Mac runs: a folder of links to qwen35/), as the engine's open() does
        func sizeAt(_ path: String) -> Int? {
            guard let a = try? fm.attributesOfItem(atPath: (path as NSString).resolvingSymlinksInPath),
                  (a[.type] as? FileAttributeType) == .typeRegular else { return nil }
            return (a[.size] as? NSNumber)?.intValue
        }
        func size(_ rel: String) -> Int? {
            if rel.hasPrefix("/") { return sizeAt(rel) }
            return sizeAt(dir.appendingPathComponent(rel).path)
        }
        let layout = fm.fileExists(atPath: dir.appendingPathComponent("resident.safetensors").path) ? "full" : "slim"
        // cache files: shown relative to the model folder when the cache is inside it (the default, model/cache)
        let cachePrefix = cacheDir.standardizedFileURL.path.hasPrefix(dir.standardizedFileURL.path + "/")
            ? String(cacheDir.standardizedFileURL.path.dropFirst(dir.standardizedFileURL.path.count + 1)) + "/"
            : cacheDir.standardizedFileURL.path + "/"

        var recordBytes = Budget.defaultRecordBytes, layers = Budget.layers, experts = Budget.experts
        var metaSource = "built-in"
        if let d = try? Data(contentsOf: dir.appendingPathComponent("store4/meta.json")),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            if let r = j["record_bytes"] as? Int { recordBytes = r }
            if let l = j["layers"] as? Int { layers = l }
            if let e = j["experts"] as? Int { experts = e }
            metaSource = "store4/meta.json"
        }

        // Sizes of the Mac's qwen35/ files (2026-11-03). Exact checks only where a short copy would load garbage.
        var want: [(String, Int, Bool)] = [
            ("config.json", 3_686, false),
            ("tokenizer.json", 12_807_982, false),
            ("tokenizer_config.json", 16_718, false),
            ("vocab.json", 6_722_759, false),
            ("merges.txt", 3_353_259, false),
            ("chat_template.jinja", 7_764, false),
            ("generation_config.json", 202, false),
            ("store4/meta.json", 639, false),
        ]
        if layout == "full" {
            want += [("routers_k4.npz", 1_258_937_576, true), ("resident.safetensors", 4_896_793_680, true)]
        } else {
            want.append(("embeddings.safetensors", 1_017_119_000, true))
        }
        for (f, n, nm) in cacheFiles {
            // required in the slim layout (the file or its page-aligned copy); in the full layout checked only when present
            // (the engine builds it if absent); a page-aligned copy that is present is checked too
            let hasMmap = size(cachePrefix + mmapName(f)) != nil
            if size(cachePrefix + f) != nil || (layout == "slim" && !hasMmap) { want.append((cachePrefix + f, n, true)) }
            if hasMmap { want.append((cachePrefix + mmapName(f), nm, true)) }
        }
        // a layer is read compressed when layer_XX.psc + .idx are present (and -storeCodec is not off): their sizes are
        // checked against the index (the .psc's size is the index's last record offset), and no .bin is needed
        var layerFiles: [String] = []
        for l in 0..<layers {
            let base = String(format: "store4/layer_%02d", l)
            if StoreStatusCodec.mode != "off", size(base + ".psc") != nil, size(base + ".idx") != nil,
               let ix = StoreStatusCodec.indexSizes(dir.appendingPathComponent(base + ".idx").path, records: experts) {
                want.append((base + ".idx", ix.idx, true))
                want.append((base + ".psc", ix.psc, true))
                layerFiles.append("psc")
            } else {
                want.append((base + ".bin", recordBytes * experts, true))
                layerFiles.append("bin")
            }
        }

        var files: [StoreFile] = []
        var problems: [String] = []
        var present = 0, expected = 0
        for (rel, exp, exact) in want {
            expected += exp
            let got = size(rel)
            var f = StoreFile(path: rel, expectedBytes: exp, exact: exact, actualBytes: got, ok: false, problem: nil)
            if let g = got {
                present += g
                if exact && g != exp {
                    f.problem = "size \(g) != \(exp)" + (g < exp ? " (partial copy?)" : "")
                } else if g == 0 {
                    f.problem = "empty"
                } else {
                    f.ok = true
                }
            } else {
                f.problem = "missing"
            }
            if let p = f.problem { problems.append("\(rel): \(p)") }
            files.append(f)
        }
        let optional = ["store4/mtp_00.bin", "mtp.safetensors"].filter { size($0) != nil }
        return StoreStatus(modelDir: dir.path, layout: layout, cacheDir: cacheDir.path, complete: problems.isEmpty, problems: problems, files: files,
                           presentBytes: present, expectedBytes: expected, recordBytes: recordBytes, layers: layers,
                           experts: experts, metaSource: metaSource, optionalPresent: optional, layerFiles: layerFiles,
                           codecLayers: layerFiles.indices.filter { layerFiles[$0] == "psc" })
    }
}

/// `-mappedWeights 0|1` (2026-11-04; default 0): 1 = the int4 resident model and the int8 routers memory-mapped from
/// their page-aligned copies (native MappedWeights; EngineOptions.mappedWeights). Measured on the 15 Plus (2026-11-06,
/// docs/G4-runtime.md "Mapped weights on the phone"): it works (1395/1395 tensors zero-copy, footprint 1.47 GB instead of
/// 2.89 after load), but it frees no RAM (the GPU keeps the pages resident; the budget is unchanged, cap 37 was killed by
/// jetsam) and at cap 19 it decoded 1.81 tok/s vs 2.03 loaded with 37k page-ins vs 7k (the kernel dropped and re-read
/// ~0.5 GB of weight pages mid-reply). So it stays off: a diagnostic only.
enum MappedWeightsSetting {
    /// Set when a mapped load failed this launch (ChatModel.load falls back to reading the weights into memory).
    nonisolated(unsafe) static var failedThisLaunch = false
    /// probe.json budget.mappedWeights and, after a fallback, the console's `MAPPED WEIGHTS FALLBACK` line.
    static var status: String {
        if failedThisLaunch {
            return "FALLBACK: the memory-mapped load failed (" + (MappedWeights.lastFailure ?? "see the LOAD failed line")
                + "); loaded the weights into memory instead (same cap: the budget counts their 1.44 GB either way)"
        }
        return on ? "memory-mapped" : "read into memory"
    }
    static var on: Bool {
        if failedThisLaunch { return false }
        return LaunchArgs.value("-mappedWeights").map { $0 != "0" && $0 != "off" } ?? false
    }
}

/// `-mapTest` (2026-11-06): the mapped-weights path step by step on this device (native MappedWeights.selfTest over
/// the cache's resident_q4.safetensors; `-mapTest routers`: routers_k4_r8): phys_footprint, file-backed pages and
/// page-ins after the page-aligned copy, the mmap, a raw Metal no-copy buffer (+ residency set + GPU blit), the MLX
/// arrays as the engine builds them, a CPU touch of every page, a GPU op over every array, and the same op over the
/// file read into memory. Runs before any engine load (from the launch probe), writes Documents/maptest.json and the
/// console's `MAPTEST` lines, then exits. The work runs on a background thread while the main run loop keeps turning.
enum MapTest {
    struct Out: Encodable {
        var time: String
        var file: String
        var steps: [MapTestStep]
    }

    static func runAndExit() -> Never {
        let file = LaunchArgs.value("-mapTest") == "routers" ? "routers_k4_r8.safetensors" : "resident_q4.safetensors"
        let src = nativeCacheURL.appendingPathComponent(file).path
        log("MAPTEST start: \(src)")
        // the worker finishes the test itself: the launch Task runs on the main queue, so a block queued to the main
        // queue would never run while this loop turns the run loop (the first phone runs hung here after step 12)
        let th = Thread {
            let steps = MappedWeights.selfTest(cacheFile: src) { log($0) }
            let out = Out(time: ISO8601DateFormatter().string(from: Date()), file: src, steps: steps)
            try? jsonEncoder().encode(out).write(to: docsURL.appendingPathComponent("maptest.json"), options: .atomic)
            log("MAPTEST wrote maptest.json (\(steps.count) steps)")
            log("DONE")
            exit(0)
        }
        th.stackSize = 8 << 20
        th.start()
        while true { _ = RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.2)) }
    }
}

/// `-driveTest [SECONDS]` (2026-11-07): the engine's reader pool on the store, without the model (native DriveTest.swift,
/// SECONDS per cell, default 2; `-driveFresh N`: also N layers copied contiguously, `-driveFreshOnly`: only that
/// comparison): writes Documents/drivetest.json and the console's DRIVE lines, then exits. `-defragStore [all|LIST]`:
/// rewrite the fragmented layer files contiguously (native defragmentStore; verified copy, then rename), writes
/// Documents/defrag.json. Both run from the launch probe, before any engine load, on a background thread while the
/// main run loop keeps turning (as -mapTest).
enum DriveTestRun {
    static func runAndExit() -> Never {
        var c = DriveTestConfig(dir: modelDirURL.path)
        if let v = LaunchArgs.value("-driveTest").flatMap({ Double($0) }), v > 0 { c.cellSeconds = v }
        if let v = LaunchArgs.value("-driveFresh").flatMap({ Int($0) }), v > 0 { c.freshLayers = v }
        c.freshOnly = LaunchArgs.has("-driveFreshOnly")
        c.busyMarkers = ["autotest.json", "drivetest.json", "ui.json", "defrag.json"].map { docsURL.appendingPathComponent($0).path }
        log("DRIVE start: \(modelDirURL.path), \(c.cellSeconds) s per cell, thermal \(thermalName(ProcessInfo.processInfo.thermalState))")
        run("drivetest.json") { try runDriveTest(c) { log($0) } }
    }
    static func defragAndExit() -> Never {
        let v = LaunchArgs.value("-defragStore") ?? "all"
        let layers: [Int]? = v.hasPrefix("-") || v == "all" ? nil : v.split(separator: ",").compactMap { Int($0) }
        log("DEFRAG start: \(modelDirURL.path) layers \(layers.map { "\($0)" } ?? "all")")
        run("defrag.json") { try defragmentStore(dir: modelDirURL.path, layers: layers) { log($0) } }
    }
    static func run(_ file: String, _ job: @escaping () throws -> [String: Any]) -> Never {
        let th = Thread {
            do {
                let d = try JSONSerialization.data(withJSONObject: try job(), options: [.prettyPrinted, .sortedKeys])
                try d.write(to: docsURL.appendingPathComponent(file), options: .atomic)
                log("DRIVE wrote \(file)")
            } catch {
                log("ERROR \(file): \(error)")
            }
            log("DONE")
            exit(0)
        }
        th.stackSize = 8 << 20
        th.start()
        while true { _ = RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.2)) }
    }
}

/// The compressed-store files as the store check sees them (codec/CODEC.md, native/Sources/PhoneStore/Codec.swift).
enum StoreStatusCodec {
    /// `-storeCodec auto|off` (default auto).
    static var mode: String { LaunchArgs.value("-storeCodec") ?? "auto" }
    /// A PSC1 index's expected size and its .psc's size (the last record offset); nil if it is not a PSC1 index for
    /// `records` records. Header: 'PSC1', version, layer, nrec, record_bytes, 3 reserved (u32); lens 4 x 256 B; tables
    /// 4 x 4096 u16; offsets (nrec + 1) u64; SHA-256 nrec x 32 B.
    static func indexSizes(_ path: String, records: Int) -> (idx: Int, psc: Int)? {
        guard let d = FileManager.default.contents(atPath: path), d.count >= 32, String(data: d.prefix(4), encoding: .ascii) == "PSC1" else { return nil }
        let nrec = Int(d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 12, as: UInt32.self) })
        guard nrec == records else { return nil }
        let offEnd = 32 + 1024 + 2 * 4 * 4096 + 8 * nrec
        let size = offEnd + 8 + 32 * nrec
        guard d.count == size else { return nil }
        let psc = Int(d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offEnd, as: UInt64.self) })
        return (size, psc)
    }
}

struct Probe: Codable {
    var time: String
    var machine: String
    var os: String
    var app: String
    var physicalMemoryBytes: Int
    var osProcAvailableMemoryBytes: Int
    var diskAvailableImportantBytes: Int
    var diskAvailableBytes: Int
    var diskTotalBytes: Int
    var thermalState: String
    var lowPowerMode: Bool
    var activeProcessors: Int
    var cap: Int
    var capFromBudget: Int            // what the formula gives from os_proc_available_memory
    var capSource: String             // "budget" or "launch argument -cap N"
    var budget: BudgetReport
    /// "full" | "slim" (StoreStatus.layout): which model-folder layout the store check and the engine use.
    var layout: String
    var store: StoreStatus
    /// Per-layer slot caps (2026-11-04): the auto allocation of 40 x cap slots by default, "uniform" with -cap N or
    /// -layerCaps uniform. The engine's memory plan is the sum.
    var layerCapsSpec: String
    var layerCaps: [Int]
    var slotsTotal: Int
    /// -storeCodec auto|off (default auto): layers with layer_XX.psc + .idx are read compressed and decoded on the GPU.
    var storeCodec: String
    var modelDirExcludedFromBackup: Bool
    var modelDirIsLink: Bool
    var docsDir: String
    var nativeCacheDir: String
    var launchArguments: [String]
    /// Filled in after the engine is chosen and loaded (or refused): every setting the settings sheet shows.
    var engine: EngineReport?
    /// MemProbe samples so far (launch, after load, every 10 tokens, memory warnings, end of each reply).
    var memory: [MemSample]?
    /// Diagnostic launch settings: memWarn mode, overrides (README "Diagnostics").
    var diagnostics: [String: String]?

    static func run() -> Probe {
        ensureModelDir()
        if LaunchArgs.has("-mapTest") { MapTest.runAndExit() }
        // the engine's read path and reader QoS (native StoreTuning; docs/G4-runtime.md "Drive throughput on the phone")
        if let v = LaunchArgs.value("-readPath"), let p = ReadPath(rawValue: v), p != .flat { StoreTuning.readPath = p }
        if let v = LaunchArgs.value("-readerQoS"), let q = StoreTuning.qos(named: v) { StoreTuning.readerQoS = q }
        if LaunchArgs.has("-driveTest") { DriveTestRun.runAndExit() }
        if LaunchArgs.has("-defragStore") { DriveTestRun.defragAndExit() }
        let pi = ProcessInfo.processInfo
        let avail = Int(os_proc_available_memory())
        var important = -1, available = -1, total = -1
        if let v = try? docsURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey,
                                                         .volumeAvailableCapacityKey, .volumeTotalCapacityKey]) {
            important = Int(v.volumeAvailableCapacityForImportantUsage ?? -1)
            available = v.volumeAvailableCapacity ?? -1
            total = v.volumeTotalCapacity ?? -1
        }
        let store = StoreStatus.check(dir: modelDirURL)
        let rb = store.recordBytes
        // compressed layers need the codec staging (counted before the slots)
        let stagingBytes = store.codecLayers.isEmpty ? 0 : Budget.codecStagingBytes
        let capBudget = Budget.cap(availableBytes: Double(avail) - stagingBytes, recordBytes: rb)
        // `-cap N`: a fixed cap (Mac runs: os_proc_available_memory is an iOS jetsam figure and reads 0 on the Mac).
        let capArg = LaunchArgs.value("-cap").flatMap { Int($0) }
        var cap = capArg ?? capBudget
        // the one-time setup's cap is kept by the same build (2026-11-07, docs/G4-runtime.md "Launch: once per install"):
        // a prefix state is refused under other slot caps, so a budget one cap step off (76 MB, inside the margin's
        // slack) would void the cached system states; a larger change, or another build, uses the budget (and the setup
        // runs again). Not in the test flows.
        var capKept = ""
        if capArg == nil, !TestFlow.active, LaunchArgs.value("-layerCaps") == nil, let r = OneTimeSetup.load(),
           r.exe == OneTimeSetup.exeStamp, r.cap >= Budget.minCap, r.cap != capBudget, abs(r.cap - capBudget) <= 1 {
            capKept = " \(capBudget), kept at the one-time setup's cap \(r.cap)"
            cap = r.cap
        }
        let lc = Budget.sessionLayerCaps(cap: cap)
        let caps = lc.caps ?? Array(repeating: cap, count: store.layers)
        let budget = BudgetReport(
            recordBytes: rb, residentBytes: Budget.residentBytes, routerBytes: Budget.routerBytes,
            stateBytes: Budget.stateBytes, marginBytes: Budget.marginBytes,
            slabBytesPerCap: Double(rb * Budget.layers), stagingBytesPerCap: Double(rb * Budget.stagingRecordsPerCap),
            formula: "cap = floor((os_proc_available_memory - resident - routers - state - margin) / ((40 + 3) x record_bytes)); margin 1.2 GB with the increased-memory-limit entitlement, 0.8 GB without (15 Plus warning points, 2026-11-04); resident + routers 1.10 + 0.34 GB of RAM (also when memory-mapped: the GPU keeps them resident)",
            planBytes: Budget.planBytes(cap: cap, recordBytes: rb) + stagingBytes,
            minCap: Budget.minCap, g4FloorCap: Budget.g4FloorCap,
            meetsG4Floor: cap >= Budget.g4FloorCap, enoughForMinCap: cap >= Budget.minCap,
            mappedWeights: MappedWeightsSetting.status,
            prefillAutoBytes: Budget.prefillAutoBytes(cap: cap, recordBytes: rb, limitBytes: avail > 0 ? Double(avail) : nil),
            prefillAutoFormula: "auto prefill chunk: the largest whose activations fit limit - iOS warning distance (0.85 GB entitled, 0.58 without) - plan - 0.20 GB overshoot (native PrefillMemory, measured on the Mac: 30 MB + 189 KB per position + 65 B per position x key)")
        if MappedWeightsSetting.failedThisLaunch { log("MAPPED WEIGHTS " + MappedWeightsSetting.status) }
        let excluded = (try? modelDirURL.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup) ?? false
        let info = Bundle.main.infoDictionary
        return Probe(
            time: ISO8601DateFormatter().string(from: Date()), machine: machineName(),
            os: pi.operatingSystemVersionString,
            app: "\(info?["CFBundleShortVersionString"] ?? "?") (\(info?["CFBundleVersion"] ?? "?"))",
            physicalMemoryBytes: Int(pi.physicalMemory), osProcAvailableMemoryBytes: avail,
            diskAvailableImportantBytes: important, diskAvailableBytes: available, diskTotalBytes: total,
            thermalState: thermalName(pi.thermalState), lowPowerMode: pi.isLowPowerModeEnabled,
            activeProcessors: pi.activeProcessorCount, cap: cap, capFromBudget: capBudget,
            capSource: capArg != nil ? "launch argument -cap \(cap)" : "budget" + capKept, budget: budget, layout: store.layout, store: store,
            layerCapsSpec: lc.label, layerCaps: caps, slotsTotal: caps.reduce(0, +), storeCodec: StoreStatusCodec.mode,
            modelDirExcludedFromBackup: excluded, modelDirIsLink: isLink(modelDirURL), docsDir: docsURL.path,
            nativeCacheDir: nativeCacheURL.path, launchArguments: Array(CommandLine.arguments.dropFirst()), engine: nil,
            memory: nil, diagnostics: nil)
    }

    func write() {
        do {
            try jsonEncoder().encode(self).write(to: probeURL, options: .atomic)
        } catch {
            log("ERROR writing probe.json: \(error)")
        }
    }

    var summaryLine: String {
        String(format: "PROBE avail=%.2fGB phys=%.2fGB disk_free=%.2fGB thermal=%@ cap=%d (%@) layer_caps=%@ slots=%d layout=%@ store=%@ (%d problems) codec_layers=%d",
               Double(osProcAvailableMemoryBytes) / 1e9, Double(physicalMemoryBytes) / 1e9,
               Double(diskAvailableImportantBytes) / 1e9, thermalState, cap, capSource, layerCapsSpec, slotsTotal, layout,
               store.complete ? "complete" : "incomplete", store.problems.count, store.codecLayers.count)
    }
}

/// Documents/model/, Documents/model/store4/ and Documents/model/cache/ exist from first launch so `devicectl device copy to` can write into
/// them; the folder is excluded from iCloud/iTunes backup (24 GB of weights must not be backed up).
func ensureModelDir() {
    let fm = FileManager.default
    // A linked model folder (Mac runs: Documents/model -> qwen35/) is used as it is: nothing is created or marked in it.
    if isLink(modelDirURL) { return }
    try? fm.createDirectory(at: modelDirURL.appendingPathComponent("store4", isDirectory: true),
                            withIntermediateDirectories: true)
    try? fm.createDirectory(at: nativeCacheURL, withIntermediateDirectories: true)
    var u = modelDirURL
    var rv = URLResourceValues()
    rv.isExcludedFromBackup = true
    try? u.setResourceValues(rv)
}

func jsonEncoder() -> JSONEncoder {
    let e = JSONEncoder()
    e.keyEncodingStrategy = .convertToSnakeCase
    e.outputFormatting = [.prettyPrinted, .sortedKeys]
    return e
}

func thermalName(_ s: ProcessInfo.ThermalState) -> String {
    switch s {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

func machineName() -> String {
    var u = utsname()
    uname(&u)
    return withUnsafeBytes(of: &u.machine) { raw in
        String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
    }
}

func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1e9) }
func gb(_ b: Double) -> String { String(format: "%.2f GB", b / 1e9) }

func isLink(_ u: URL) -> Bool {
    (try? FileManager.default.attributesOfItem(atPath: u.path)[.type] as? FileAttributeType) == .typeSymbolicLink
}

/// The engine as the settings sheet shows it; part of probe.json.
struct EngineReport: Codable {
    var name: String
    var isStub: Bool
    var requiresStore: Bool
    var choice: String                // why this engine was chosen (EngineSeam.swift)
    var detail: String
    var state: String
    var thinking: Bool
    var settings: [String: String]    // every engine setting in force (empty until loaded)
    var loadLog: [String]
}
