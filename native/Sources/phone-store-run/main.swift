// phone-store-run: the G4 native runtime's CLI. The key flags and the bench JSON fields are mlx_run.py's.
//
//   phone-store-run --dir ../qwen35 --store store4 --mode exact --routers none --evict hazard --no-drop --max-slots 48 \
//       --keep 31 --bench --chat --max-new 200 --prompt "..."
//
// Every setting is in the startup line and in the JSON; nothing degrades silently (an unsupported flag value aborts).
import Foundation
import MLX
import PhoneStore
import Tokenizers

setvbuf(stdout, nil, _IOLBF, 0)
setQoSUserInteractive()

func die(_ s: String) -> Never {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
    exit(2)
}

let here = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
var args = Array(CommandLine.arguments.dropFirst())
var opt = EngineOptions(dir: "qwen35", cacheDir: "")
var prompt: String? = nil
var chat = true
var maxNew = 200
var bench = false
var promptIds: [Int]? = nil
var dumpPath: String? = nil
var forceIds: [Int]? = nil
var cacheDir: String? = nil
var unsupported: [String] = []
var replayPath: String? = nil
var logitsOut: String? = nil
var readBenchSec = 0.0
var driveTestSec = 0.0  // --drive-test S: DriveTest (no model), S seconds per cell
var driveOut: String? = nil
var driveCap = 19
var driveFresh = 0
var driveFreshOnly = false
var defragLayers: String? = nil
var tracePath: String? = nil
var tfRefPath: String? = nil  // --teacher-force REF.json: a reference run (prompt_ids + ids) fed token by token
var tfOut: String? = nil
var tfTopPath: String? = nil  // --tf-ref TOP.json: another teacher-forced run's top-K to compare against
var tfK = 32
// prefix state on disk (docs/PREFIX-CACHE.md, native port; all off by default)
var savePrefixPath: String? = nil
var loadPrefixPath: String? = nil
var prefixTokens = 0
var noReland = false
var kernelWarmupCLI = false
var prefixCacheDir: String? = nil
var prefixCacheGB = 16.0
var layerCapsArg: String? = nil
var namesOut: String? = nil
// prompting and sampling (2026-11-04, docs/G4-runtime.md "Prompting and sampling"): greedy stays the default
var thinkingArg: Bool? = nil  // --thinking on|off: enable_thinking (default: not passed = the template's default, on)
var systemPrompt: String? = nil
var templateCheckPath: String? = nil
var sampleOn = false
var sampling = SamplingParams.greedy
var samplingSet: [String] = []
var thinkingBudget: Int? = nil
var promptTokens = 0  // --prompt-tokens N: the first N prompt ids only
var chunkPlanTest: [Int]? = nil  // --chunk-plan-test C,C,...: Store.chunkPlan + the engine's pass schedule on synthetic routing, no model
var prefillPlanMB: Double? = nil  // --prefill-plan MB: the auto chunk's choices for that budget, no model
/// `--prefill-chunk auto` without a budget: the iPhone 15 Plus's at its budget cap 19 (ios Probe.swift
/// Budget.prefillAutoBytes: 4.295 GB limit - 0.85 GB iOS warning distance - 3.036 GB plan - 0.20 GB footprint overshoot)
let prefillAutoDefaultMB = 209.0

func val(_ i: inout Int) -> String {
    i += 1
    guard i < args.count else { die("missing value for \(args[i - 1])") }
    return args[i]
}

var i = 0
while i < args.count {
    let a = args[i]
    switch a {
    case "--dir": opt.dir = val(&i)
    case "--store": opt.store = val(&i)
    case "--mode": opt.mode = Mode(rawValue: val(&i)) ?? { die("--mode exact|prefetch") }()
    case "--routers": opt.routers = val(&i)
    case "--router-bits": opt.routerBits = Int(val(&i))!
    case "--m": opt.m = Int(val(&i))!
    case "--keep": opt.keep = Int(val(&i))!
    case "--evict": opt.evict = EvictKind(rawValue: val(&i)) ?? { die("--evict window|lru|hazard|prior") }()
    case "--no-drop": opt.noDrop = true
    case "--max-slots": opt.maxSlots = Int(val(&i))!
    case "--layer-caps": layerCapsArg = val(&i)  // uniform | auto:TOTAL | 40 ints (comma list or a file); budget = the sum
    case "--store-codec": opt.storeCodec = StoreCodec(rawValue: val(&i)) ?? { die("--store-codec auto|off") }()
    case "--codec-staging": opt.codecStaging = Int(val(&i))!  // staging slots for compressed records
    case "--codec-check": opt.codecCheck = true  // test: SHA-256 of every decoded slot against the .idx (slow)
    case "--names-out": namesOut = val(&i)  // per token clock, per layer: the experts named (prefetch) / released (exact), JSON
    case "--mem-cap": opt.memCap = Double(val(&i))!
    case "--resident-bits": opt.residentBits = Int(val(&i))!
    case "--threads", "--readers": opt.readers = Int(val(&i))!
    case "--sim-reads": opt.simReads = true
    case "--cache": opt.noCache = false
    case "--bw-cap": opt.bwCap = Double(val(&i))!
    case "--cache-mb": opt.cacheLimitMB = Int(val(&i))!
    case "--no-wire": opt.wire = false
    case "--mapped-weights":  // off (default since 2026-11-06, EngineOptions.mappedWeights): read into memory; on: resident + routers memory-mapped (Weights.swift MappedWeights)
        switch val(&i) {
        case "on", "1": opt.mappedWeights = true
        case "off", "0": opt.mappedWeights = false
        default: die("--mapped-weights on|off")
        }
    case "--early-wait": opt.lateWait = false  // read overlap off: wait for a layer's reads before its masked top-k
    case "--read-chunks": opt.readChunks = Int(val(&i))!  // reads per record (16 KB aligned byte ranges)
    case "--trace": tracePath = val(&i)  // timeline of generated tokens 100-109 (reads, namings, waits) as JSON
    case "--no-head-start": opt.headStart = false  // the sampler's argmax encoded at sampling time, not with the logits
    case "--teacher-force": tfRefPath = val(&i)
    case "--tf-out": tfOut = val(&i)
    case "--tf-ref": tfTopPath = val(&i)
    case "--tf-k": tfK = Int(val(&i))!
    case "--prompt": prompt = val(&i)
    case "--prompt-ids": promptIds = val(&i).split(separator: ",").map { Int($0)! }
    case "--force-ids": forceIds = val(&i).split(separator: ",").map { Int($0)! }  // teacher forcing (logits check)
    case "--chat": chat = true
    case "--no-chat": chat = false
    case "--max-new": maxNew = Int(val(&i))!
    case "--bench": bench = true
    case "--greedy": break  // greedy is the default (identity tests); --sample or any sampling flag below turns sampling on
    case "--sample": sampleOn = true  // the release's generation_config.json (temperature 1.0, top_k 20, top_p 0.95), then the flags below
    case "--temp", "--temperature": sampleOn = true; samplingSet.append("temperature"); sampling.temperature = Float(val(&i))!
    case "--top-k": sampleOn = true; samplingSet.append("top_k"); sampling.topK = Int(val(&i))!
    case "--top-p": sampleOn = true; samplingSet.append("top_p"); sampling.topP = Float(val(&i))!
    case "--min-p": sampleOn = true; samplingSet.append("min_p"); sampling.minP = Float(val(&i))!
    case "--presence-penalty": sampleOn = true; samplingSet.append("presence_penalty"); sampling.presencePenalty = Float(val(&i))!
    case "--repetition-penalty": sampleOn = true; samplingSet.append("repetition_penalty"); sampling.repetitionPenalty = Float(val(&i))!
    case "--frequency-penalty": sampleOn = true; samplingSet.append("frequency_penalty"); sampling.frequencyPenalty = Float(val(&i))!
    case "--seed": samplingSet.append("seed"); sampling.seed = UInt64(val(&i))!
    case "--preset":  // the Qwen3.6 card's presets: thinking (1.0 / 0.95 / 20, presence 1.5) | thinking-coding (0.6 / 0.95 / 20) | non-thinking (0.7 / 0.8 / 20, presence 1.5) | model (generation_config)
        sampleOn = true
        switch val(&i) {
        case "thinking": samplingSet.append("preset:thinking")
        case "non-thinking": samplingSet.append("preset:non-thinking")
        case "thinking-coding": samplingSet.append("preset:thinking-coding")
        case "model": samplingSet.append("preset:model")
        default: die("--preset thinking|thinking-coding|non-thinking|model")
        }
    case "--thinking":
        switch val(&i) {
        case "on", "1", "true": thinkingArg = true
        case "off", "0", "false": thinkingArg = false
        default: die("--thinking on|off")
        }
    case "--no-think": thinkingArg = false
    case "--thinking-budget": thinkingBudget = Int(val(&i))!  // tokens inside the think block, then the template's close is forced
    case "--system": systemPrompt = val(&i)
    case "--template-check": templateCheckPath = val(&i)  // compare the rendered prompt ids with docs/template/reference.json and exit
    case "--dump": dumpPath = val(&i)
    case "--logits-out": logitsOut = val(&i)
    case "--read-bench": readBenchSec = Double(val(&i))!  // the drive alone: random records, --readers threads  // as mlx_run.py: the next-token logits after the prompt, float32 .npy
    case "--native-cache": cacheDir = val(&i)
    case "--drive-test": driveTestSec = Double(val(&i))!  // the drive under the engine's reader pool, a matrix of settings (DriveTest.swift)
    case "--drive-out": driveOut = val(&i)  // --drive-test: the JSON (default: stdout only)
    case "--drive-cap": driveCap = Int(val(&i))!  // --drive-test: slots per layer of its Store (19, the phone's)
    case "--drive-fresh": driveFresh = Int(val(&i))!  // --drive-test: also the first N layers copied contiguously (deleted after)
    case "--drive-fresh-only": driveFreshOnly = true  // --drive-test: only the fresh-copy comparison
    case "--defrag-store": defragLayers = val(&i)  // all | 0,1,...: rewrite fragmented layer files contiguously (verified, renamed over)
    case "--read-path":  // how a record's bytes reach the slab: iovec (preadv into the 9 pieces) | staged (pread + memcpy)
        let v = val(&i)
        guard let p = ReadPath(rawValue: v), p != .flat else { die("--read-path iovec|staged") }
        StoreTuning.readPath = p
    case "--reader-qos":  // reader threads' QoS: ui | init | default | utility | background
        let v = val(&i)
        guard let q = StoreTuning.qos(named: v) else { die("--reader-qos ui|init|default|utility|background") }
        StoreTuning.readerQoS = q
    case "--router-dtype":
        let v = val(&i)
        if v != "float16" { unsupported.append("--router-dtype \(v)") }
    case "--embed":
        let v = val(&i)
        if v != "disk" { unsupported.append("--embed \(v)") }
    case "--dtype":
        let v = val(&i)
        if v != "float16" { unsupported.append("--dtype \(v)") }
    case "--chunk": _ = val(&i)  // mlx_run.py's flag, ignored here: the native prefill is --prefill-chunk
    case "--prefill-chunk":  // batched prefill: positions per chunk (0: token by token, the references' path) | auto[:MB]
        let v = val(&i)
        if v == "auto" || v.hasPrefix("auto:") {  // each chunk the largest whose activations fit MB (Prefill.swift PrefillMemory)
            let mb = v == "auto" ? prefillAutoDefaultMB : (Double(v.dropFirst(5)) ?? { die("--prefill-chunk auto:MB") }())
            opt.prefillAutoBytes = mb * 1e6
            if opt.prefillChunk <= 1 { opt.prefillChunk = 128 }
        } else {
            opt.prefillChunk = Int(v) ?? { die("--prefill-chunk N|auto|auto:MB") }()
            opt.prefillAutoBytes = 0
        }
    case "--chunk-plan-test": chunkPlanTest = val(&i).split(separator: ",").map { Int($0)! }
    case "--prefill-plan": prefillPlanMB = Double(val(&i))!  // print the auto chunks for an activation budget of MB (no model) and exit
    case "--prompt-file": prompt = try! String(contentsOfFile: val(&i), encoding: .utf8)  // the prompt text from a file
    case "--prompt-tokens": promptTokens = Int(val(&i))!  // keep the first N prompt ids
    case "--prefill-readers": opt.prefillReaders = Int(val(&i))!  // reader threads while a chunk's reads are queued
    case "--prefill-read-chunks": opt.prefillReadChunks = Int(val(&i))!  // reads per record during a chunk (0: --read-chunks)
    case "--save-prefix": savePrefixPath = val(&i)  // after prefilling the prompt (or its first --prefix-tokens), write the state
    case "--load-prefix": loadPrefixPath = val(&i)  // start from a saved state (its ids a prefix of the prompt's), re-land, prefill the rest
    case "--prefix-tokens": prefixTokens = Int(val(&i))!  // the first N prompt tokens are the prefix (the save point)
    case "--no-reland": noReland = true  // restore without reading the saved resident experts back (slots start empty)
    case "--reland":  // lazy (default since 2026-11-07): the held set restored, records read by the next chunk if used or kept; eager: all landed at the restore
        switch val(&i) {
        case "lazy": opt.relandLazy = true
        case "eager": opt.relandLazy = false
        default: die("--reland lazy|eager")
        }
    case "--prefill-read-ahead": opt.prefillReadAhead = Int(val(&i))!  // records per layer read ahead during a chunk from the lag-4 names into empty slots (-1: up to the cap, 0: off)
    case "--read-ahead-hprev":  // the routers' h_prev for chunk positions after the first: stale (the token before the chunk) | shift (the lag layer's input at the previous position)
        switch val(&i) {
        case "stale": opt.readAheadShift = false
        case "shift": opt.readAheadShift = true
        default: die("--read-ahead-hprev stale|shift")
        }
    case "--read-ahead-overwrite": opt.readAheadOverwrite = Int(val(&i))!  // read-ahead targets: 0 empty slots, 1 + stale unnamed records, 2 + held unnamed
    case "--kernel-warmup": kernelWarmupCLI = true  // the app's load-time warm-up (dry batched chunks + a dry decode step) before the prompt
    case "--no-decode-warmup": opt.decodeWarmup = false  // the warm-up without its dry decode step
    case "--prefix-cache": prefixCacheDir = val(&i)  // states keyed by SHA-256 of the ids; restore the longest cached prefix
    case "--prefix-cache-gb": prefixCacheGB = Double(val(&i))!  // LRU cap on --prefix-cache (0: never delete)
    case "--replay-policy": replayPath = val(&i)  // test: the cache policy on a routing dump (mlx_run.py --dump-routing), no model
    default: die("unknown flag \(a)")
    }
    i += 1
}
if !unsupported.isEmpty { die("not supported by the native runtime: \(unsupported.joined(separator: ", "))") }
if let layerCapsArg {
    do {
        let (c, label) = try LayerCaps.parse(layerCapsArg)
        opt.layerCaps = c
        opt.layerCapsSpec = label
    } catch { die("\(error)") }
}
guard opt.residentBits == 4 || opt.residentBits == 8 || opt.residentBits == 0 else { die("--resident-bits 0|4|8") }
guard opt.routerBits == 0 || opt.routerBits == 8 else { die("--router-bits 0|8") }
opt.dir = URL(fileURLWithPath: opt.dir).standardizedFileURL.path
// The converted cache: --native-cache DIR; else <dir>/cache when the model folder has one (the slim layout, as the app
// lays it out: Documents/model/cache); else native/cache (the full layout on the Mac).
let modelCache = opt.dir + "/cache"
var isDirFlag: ObjCBool = false
let cacheChoice: String
if let cacheDir {
    opt.cacheDir = cacheDir
    cacheChoice = "--native-cache"
} else if FileManager.default.fileExists(atPath: modelCache, isDirectory: &isDirFlag), isDirFlag.boolValue {
    opt.cacheDir = modelCache
    cacheChoice = "<dir>/cache"
} else {
    opt.cacheDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().path + "/cache"
    cacheChoice = "native/cache"
}
opt.cacheDir = URL(fileURLWithPath: opt.cacheDir).standardizedFileURL.path

func log(_ s: String) { print(s) }

// MLX loads a colocated mlx.metallib (built by metallib.sh with MLX's flags, -fno-fast-math) before the bundle's
// default.metallib (Xcode's fast-math build of the same kernels, NOT bit-identical to the Python runner).
let binDir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent().path
let metallib = FileManager.default.fileExists(atPath: binDir + "/mlx.metallib") ? "mlx.metallib (-fno-fast-math)" : "bundle default.metallib (FAST MATH: not bit-identical to mlx_run.py)"
if !metallib.hasPrefix("mlx.metallib") { FileHandle.standardError.write("WARNING: no mlx.metallib next to the binary; run native/metallib.sh\n".data(using: .utf8)!) }

if let defragLayers {
    let layers: [Int]? = defragLayers == "all" ? nil : defragLayers.split(separator: ",").compactMap { Int($0) }
    do {
        let j = try defragmentStore(dir: URL(fileURLWithPath: opt.dir).standardizedFileURL.path, store: opt.store, layers: layers) { print($0) }
        let d = try JSONSerialization.data(withJSONObject: j, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: d, as: UTF8.self))
    } catch { die("defrag: \(error)") }
    exit(0)
}

if driveTestSec > 0 {
    var dc = DriveTestConfig(dir: URL(fileURLWithPath: opt.dir).standardizedFileURL.path)
    dc.store = opt.store
    dc.cellSeconds = driveTestSec
    dc.cap = driveCap
    dc.freshLayers = driveFresh
    dc.freshOnly = driveFreshOnly
    do {
        let j = try runDriveTest(dc) { print($0) }
        let d = try JSONSerialization.data(withJSONObject: j, options: [.prettyPrinted, .sortedKeys])
        if let driveOut { try d.write(to: URL(fileURLWithPath: driveOut)); print("DRIVE wrote \(driveOut)") }
    } catch { die("drive test: \(error)") }
    exit(0)
}

if readBenchSec > 0 {
    let (gbps, ms, n) = try! readBench(dir: URL(fileURLWithPath: opt.dir).standardizedFileURL.path, store: opt.store, threads: opt.readers, seconds: readBenchSec, chunks: opt.readChunks)
    print("{\"read_bench\": true, \"readers\": \(opt.readers), \"read_chunks\": \(opt.readChunks), \"seconds\": \(readBenchSec), \"reads\": \(n), \"gb_per_s\": \(gbps), \"ms_per_read\": \(ms)}")
    exit(0)
}

/// Replay a mlx_run.py --dump-routing file through the native cache policy: exact mode from `rel` (the released top-k),
/// prefetch mode from `named` (the routers' names). Prints the counts over the generated tokens (t >= n_prompt), the
/// numbers the Python bench JSON reports (sync_reads / bytes after reset_stats).
if let replayPath {
    let z = try! readNpz(replayPath)
    func ints(_ k: String) -> (shape: [Int], v: [Int]) {
        let a = z[k]!
        precondition(a.descr == "<i4" || a.descr == "<i8", "\(k): \(a.descr)")
        let v: [Int] = a.descr == "<i4" ? a.data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)).map { Int($0) } }
            : a.data.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)).map { Int($0) } }
        return (a.shape, v)
    }
    let nPrompt = ints("n_prompt").v[0]
    let src = opt.mode == .exact ? ints("rel") : ints("named")
    let T = src.shape[0], NLr = src.shape[1], k = src.shape[2]
    let sopt = StoreOptions(cap: opt.maxSlots, keep: opt.keep, evict: opt.evict, noDrop: opt.noDrop, readers: 0, simReads: true, noCache: true, bwCap: 0, policyOnly: true,
                            layerCaps: opt.layerCaps)
    let store = try! Store(dir: URL(fileURLWithPath: opt.dir).standardizedFileURL.path, store: opt.store, opt: sopt)
    var waited = 0.0
    let tStart = now()
    for t in 0..<T {
        if t == nPrompt { store.resetStats() }
        for L in 0..<NLr {
            let row = Array(src.v[((t * NLr + L) * k)..<((t * NLr + L + 1) * k)])
            if opt.mode == .exact {
                store.exactPrepare(L, t)
                _ = store.exactChoose(L, t, row, waited: &waited)
            } else if row.allSatisfy({ $0 >= 0 }) {
                store.name(L, t, row)
                _ = store.layerReady(L, t, waited: &waited)
            }
        }
    }
    let gen = T - nPrompt
    let c = store.snapshot { s in (s.syncReads, s.reads, s.bytes, s.evicted, s.evictedEarly) }
    print("{\"replay\": \"\(replayPath)\", \"mode\": \"\(opt.mode.rawValue)\", \"evict\": \"\(opt.evict.rawValue)\", \"drop\": \(!opt.noDrop), \"cap\": \(opt.maxSlots), \"keep\": \(opt.keep), "
        + "\"fed_tokens\": \(T), \"n_prompt\": \(nPrompt), \"generated_fed\": \(gen), \"sync_reads\": \(c.0), \"reads\": \(c.1), \"mb_read_per_token\": \(Double(c.2) / Double(max(1, gen)) / 1048576.0), "
        + "\"evicted\": \(c.3), \"evicted_in_window\": \(c.4), \"policy_s\": \(now() - tStart)}")
    exit(0)
}

// --teacher-force: the reference's prompt ids replace the prompt; its generated ids are fed after them
var tfRef: [String: Any]? = nil
var tfTop: TFReference? = nil
if let tfRefPath {
    guard let d = FileManager.default.contents(atPath: tfRefPath),
          let j = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any],
          let pids = j["prompt_ids"] as? [Int], let rids = j["ids"] as? [Int] else { die("--teacher-force \(tfRefPath): needs prompt_ids and ids") }
    tfRef = j
    promptIds = pids
    if tfOut == nil { die("--teacher-force needs --tf-out PATH") }
    if let tfTopPath {
        guard let d = FileManager.default.contents(atPath: tfTopPath),
              let t = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any], let top = tfReference(t) else { die("--tf-ref \(tfTopPath): no positions[].top") }
        guard (t["ids"] as? [Int]) == rids else { die("--tf-ref \(tfTopPath): its ids differ from the reference's") }
        tfTop = top
    }
}

/// --chunk-plan-test C,C,...: Store.chunkPassTest (Prefill.swift) on a policy-only Store at --max-slots, no model.
if let sizes = chunkPlanTest {
    let sopt = StoreOptions(cap: opt.maxSlots, keep: opt.keep, evict: opt.evict, noDrop: opt.noDrop, readers: 0, simReads: true, noCache: true, bwCap: 0, policyOnly: true,
                            layerCaps: opt.layerCaps)
    let store = try! Store(dir: URL(fileURLWithPath: opt.dir).standardizedFileURL.path, store: opt.store, opt: sopt)
    let r = store.chunkPassTest(sizes: sizes)
    for l in r.lines { print(l) }
    print(r.problems == 0 ? "CHUNK PLAN TEST PASS (cap \(opt.maxSlots), chunks \(sizes))" : "CHUNK PLAN TEST FAIL: \(r.problems) problems")
    exit(r.problems == 0 ? 0 : 1)
}

// tokenizer + chat template (swift-transformers 1.4.0: the release's tokenizer.json and tokenizer_config.json template)
func makePromptIds() async throws -> ([Int], Tokenizer) {
    let tk = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: opt.dir))
    if let promptIds { return (promptIds, tk) }
    let p = prompt ?? "The history of the city of Pittsburgh begins with"
    if chat {
        var msgs: [[String: String]] = []
        if let systemPrompt { msgs.append(["role": "system", "content": systemPrompt]) }
        msgs.append(["role": "user", "content": p])
        return (try ChatPrompt.ids(tk, messages: msgs, thinking: thinkingArg), tk)
    }
    return (tk.encode(text: p, addSpecialTokens: false), tk)
}

let sem = DispatchSemaphore(value: 0)
var ids: [Int] = []
var tokenizer: Tokenizer! = nil
Task.detached {
    do { (ids, tokenizer) = try await makePromptIds() } catch { die("tokenizer: \(error)") }
    sem.signal()
}
sem.wait()
if promptTokens > 0 {
    if ids.count < promptTokens { die("--prompt-tokens \(promptTokens): the prompt has \(ids.count) tokens") }
    ids = Array(ids[..<promptTokens])
}
if let prefillPlanMB {  // config.json only: no model, no lock needed
    let cfg = try! TextConfig(dir: opt.dir)
    let budget = prefillPlanMB * 1e6
    for ra in [true, false] {
        let m = PrefillMemory(cfg: cfg, readAhead: ra)
        print(String(format: "PrefillMemory (read-ahead %@): %.1f KB per position (shapes %.1f KB x %.1f, measured) + %.0f B per (position, key) + %.1f MB; terms %@",
                     ra ? "on" : "off", PrefillMemory.scale * m.perPosition / 1e3, m.perPosition / 1e3, PrefillMemory.scale, m.perQueryKey, m.fixed / 1e6,
                     m.terms.map { "\($0.0) \(Int($0.1))" }.joined(separator: ", ")))
        for c in [128, 256, 512, 1024, 2048] {
            print(String(format: "  chunk %4d at offset 0: %.1f MB; at offset 2048: %.1f MB", c, m.bytes(chunk: c, keys: c) / 1e6, m.bytes(chunk: c, keys: 2048 + c) / 1e6))
        }
        for off in [0, 512, 1024, 2048, 4096, 8192] {
            print(String(format: "  budget %.0f MB, offset %5d: largest chunk %d", prefillPlanMB, off, m.largestChunk(budget: budget, offset: off, limit: 1 << 20)))
        }
    }
    exit(0)
}
if let templateCheckPath {  // tokenizer only: no model, no lock needed
    do {
        let r = try ChatPrompt.check(tokenizer, referencePath: templateCheckPath)
        for l in r.lines { print(l) }
        print(r.ok ? "TEMPLATE CHECK PASS: every case equals transformers' apply_chat_template" : "TEMPLATE CHECK FAIL")
        exit(r.ok ? 0 : 1)
    } catch { die("--template-check: \(error)") }
}
// sampling: greedy unless --sample / a sampling flag; the base is generation_config.json (or a --preset), the flags override it
if sampleOn {
    var base = SamplingParams.generationConfig(dir: opt.dir)
    if samplingSet.contains("preset:thinking") { base = .qwenThinking }
    if samplingSet.contains("preset:non-thinking") { base = .qwenNonThinking }
    if samplingSet.contains("preset:thinking-coding") { base = .qwenThinkingCoding }
    let given = sampling
    for k in samplingSet {
        switch k {
        case "temperature": base.temperature = given.temperature
        case "top_k": base.topK = given.topK
        case "top_p": base.topP = given.topP
        case "min_p": base.minP = given.minP
        case "presence_penalty": base.presencePenalty = given.presencePenalty
        case "repetition_penalty": base.repetitionPenalty = given.repetitionPenalty
        case "frequency_penalty": base.frequencyPenalty = given.frequencyPenalty
        default: break
        }
    }
    base.seed = given.seed
    sampling = base
} else {
    let s = sampling.seed
    sampling = .greedy
    sampling.seed = s
}
if !sampling.isPlainArgmax && (forceIds != nil || tfRefPath != nil) { die("sampling flags and --force-ids / --teacher-force are exclusive") }

let t00 = now()
let engine: Engine
do { engine = try Engine(opt: opt, log: log) } catch { die("\(error)") }
let planGB = engine.plan.mapValues { Double($0) / 1e9 }
let planSum = planGB.values.reduce(0, +)
let slabDesc = engine.capsUniform ? "\(engine.store.NL) x \(engine.cap) slots" : "\(engine.caps.reduce(0, +)) slots (per layer, \(opt.layerCapsSpec))"
log(String(format: "memory plan: resident %.2f GB (int%d, lm_head %.2f, embeddings on disk) + routers %.2f + slabs %@ x %.2f MB = %.2f%@ = %.2f GB%@",
           planGB["resident"]!, opt.residentBits, Double(engine.res.headBytes) / 1e9, planGB["routers"]!, slabDesc,
           Double(engine.store.REC) / 1e6, planGB["slabs"]!, planGB["codec_staging"].map { String(format: " + codec staging %.2f", $0) } ?? "",
           planSum, opt.memCap > 0 ? String(format: " of --mem-cap %.2f", opt.memCap) : ""))
if !engine.capsUniform { log("slots per layer (\(opt.layerCapsSpec)): \(engine.caps.map(String.init).joined(separator: " "))") }
let layerPaths = engine.store.layerPath
log("store codec: \(opt.storeCodec.rawValue); per layer: " + (0..<layerPaths.count).map { "\($0):\(layerPaths[$0])" }.joined(separator: " "))
let stopIds: Set<Int> = {
    var s = Set<Int>()
    if let d = try? Data(contentsOf: URL(fileURLWithPath: opt.dir + "/generation_config.json")),
       let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
        if let a = j["eos_token_id"] as? [NSNumber] { a.forEach { s.insert($0.intValue) } }
        if let a = j["eos_token_id"] as? NSNumber { s.insert(a.intValue) }
    }
    if let e = tokenizer.eosTokenId { s.insert(e) }
    return s
}()
let embFile: String = URL(fileURLWithPath: engine.embeddingsPath).lastPathComponent
let layoutDesc: String = "| layout \(engine.layout.rawValue), native cache \(opt.cacheDir) (\(cacheChoice)) "
let routerDesc = engine.routers == nil ? "no routers" : "\(opt.m > 0 ? opt.m : 8) named, lag \(engine.routers!.lag)"
var startup: String = "loaded in \(String(format: "%.1f", now() - t00))s | qwen3_5_moe | \(opt.store) (int4) | resident \(String(format: "%.2f", Double(engine.res.nbytes) / 1e9)) GB (int\(opt.residentBits)\(engine.weightsMapped ? ", memory-mapped" : "")) "
startup += "| \(opt.mode.rawValue), \(routerDesc), keep \(opt.keep), evict \(opt.evict.rawValue), \(opt.noDrop ? "no drop" : "drop after keep"), \(engine.capsUniform ? "\(engine.cap) slots/layer" : "\(engine.caps.reduce(0, +)) slots per layer by \(opt.layerCapsSpec)") "
let routerKind: String = engine.routers == nil ? "no routers" : "float16 routers" + (opt.routerBits > 0 ? " (int\(opt.routerBits) matrices)" : "")
startup += "(\(String(format: "%.2f", Double(engine.store.slabBytes) / 1e9)) GB) | \(routerKind), embeddings on disk (\(embFile)) "
startup += layoutDesc
startup += "| \(opt.readers) readers, read path \(engine.store.readPath.rawValue), reader QoS \(StoreTuning.qosName(engine.store.readerQoS)), \(opt.noCache ? "F_NOCACHE + no readahead" : "page cache ON")\(opt.bwCap > 0 ? ", read cap \(opt.bwCap) GB/s" : "")\(opt.simReads ? " | SIMULATED reads" : "") "
startup += "| prefill \(opt.prefillChunk > 1 ? "batched, chunks of \(opt.prefillAutoBytes > 0 ? String(format: "auto (activations <= %.0f MB)", opt.prefillAutoBytes / 1e6) : "\(opt.prefillChunk)"), \(engine.store.readerThreads) readers in a chunk\(opt.prefillReadChunks > 0 ? ", \(opt.prefillReadChunks) reads/record" : "")" : "token by token") "
startup += "| sampling: \(sampling.summary)\(thinkingBudget.map { ", thinking budget \($0)" } ?? "") | enable_thinking \(thinkingArg.map { String($0) } ?? "default") "
startup += "| kernels: \(metallib) | prompt \(ids.count) tokens | in memory \(String(format: "%.2f", Double(Memory.activeMemory) / 1e9)) GB"
log(startup)

if let tfRef {
    let rids = tfRef["ids"] as! [Int]
    let r = teacherForce(engine: engine, promptIds: ids, ids: rids, k: tfK, ref: tfTop)
    let meta: [String: Any] = [
        "engine": "swift-native CLI", "device": "mac", "mode": opt.mode.rawValue, "slab": engine.cap, "keep": opt.keep,
        "layer_caps": opt.layerCapsSpec, "slab_list": engine.caps, "layer_paths": engine.store.layerPath,
        "evict": opt.evict.rawValue, "drop": !opt.noDrop, "router_bits": engine.routers == nil ? 0 : opt.routerBits,
        "resident_bits": opt.residentBits, "readers": opt.readers, "read_chunks": opt.readChunks, "late_wait": opt.lateWait,
        "head_start": opt.headStart, "layout": engine.layout.rawValue, "model_dir": opt.dir, "metallib": metallib,
        "mlx_core": "0.32.3", "teacher_force_ref": tfRefPath!, "ref_cap": tfRef["slab"] ?? NSNull(),
        "tf_ref": tfTopPath.map { $0 as Any } ?? NSNull(),
    ]
    let rep = tfReport(r, meta: meta)
    try! JSONSerialization.data(withJSONObject: rep, options: [.sortedKeys]).write(to: URL(fileURLWithPath: tfOut!))
    for l in tfConsoleLines(r, label: "cap=\(engine.cap)") { print(l) }
    print("wrote \(tfOut!)")
    exit(0)
}
if kernelWarmupCLI {
    let w = engine.warmBatchedKernels(sampling: sampleOn ? sampling : nil)
    log(w.line)
}
engine.logRel = dumpPath != nil || (namesOut != nil && opt.mode == .exact)
if namesOut != nil && opt.mode == .prefetch { engine.store.snapshot { $0.nameLog = [:] } }
// prefill token by token (mlx_run.py --bench forces --chunk 0), from a restored prefix state when one is given or cached
if loadPrefixPath != nil && prefixCacheDir != nil { die("--load-prefix and --prefix-cache are exclusive") }
let tPre = now()
var logits = MLXArray(0)
var fed: [Int] = []
var prefixRun: [String: Any] = [:]
let bytesBefore = engine.store.snapshot { $0.bytes }
let pcache: PrefixCacheDir? = prefixCacheDir.map { try! PrefixCacheDir(dir: $0, capGB: prefixCacheGB) }
var restorePath: String? = loadPrefixPath
if let pcache {
    if let hit = pcache.longest(ids) {
        restorePath = hit.path
        pcache.touch(hit.path)
    } else {
        let u = pcache.usage
        log(String(format: "[prefix-cache] miss: no cached prefix of the %d tokens | cache: %d entries, %.2f GB", ids.count, u.entries, Double(u.bytes) / 1e9))
    }
}
if let restorePath {
    do {
        // lazy (default): nothing read here when a chunk follows; with nothing left to prefill or token-by-token prefill
        // the held set is landed at once (as the app's session does)
        let r = try engine.loadPrefix(path: restorePath, ids: ids, reland: !noReland, lazy: opt.relandLazy)
        fed = r.ids
        logits = r.logits
        var inf = r.info
        if fed.count >= ids.count || opt.prefillChunk <= 1 {
            let tl = now()
            let l = engine.store.landPending()
            inf = inf.landedAtOnce(l, seconds: now() - tl)
        }
        log(String(format: "[prefix] restored %d of %d tokens from %@ | state %.1f MiB loaded in %.3fs | re-landed %d records (%.1f MiB) in %.3fs, %d kept in place, %d moved, %d deferred to the chunk, %d held%@",
                   inf.n, ids.count, (restorePath as NSString).lastPathComponent, Double(inf.fileBytes) / 1048576, inf.loadSeconds, inf.relanded,
                   Double(inf.relandBytes) / 1048576, inf.relandSeconds, inf.kept, inf.moved, inf.deferred, inf.held, noReland ? " (--no-reland)" : (opt.relandLazy ? " (lazy)" : " (eager)")))
        prefixRun = ["restored_tokens": inf.n, "state_mib": Double(inf.fileBytes) / 1048576, "load_s": inf.loadSeconds, "reland_s": inf.relandSeconds,
                     "relanded": inf.relanded, "kept_in_slab": inf.kept, "moved_in_slab": inf.moved, "reland_mib": Double(inf.relandBytes) / 1048576,
                     "held": inf.held, "reland": !noReland, "reland_lazy": opt.relandLazy, "deferred": inf.deferred, "state_file": restorePath]
    } catch {
        if loadPrefixPath != nil { die("--load-prefix: \(error)") }
        log("[prefix-cache] skipped: \(error)")
    }
}
let tRestored = now()
// the save point: --prefix-tokens N (else the whole prompt), for --save-prefix and --prefix-cache
let savePoint = prefixTokens > 0 ? min(prefixTokens, ids.count) : ids.count
var savePath: String? = savePrefixPath
if let pcache, savePoint > fed.count {
    let p = pcache.path(for: ids[..<savePoint])
    savePath = FileManager.default.fileExists(atPath: p) ? nil : p
}
if savePrefixPath != nil && savePoint <= fed.count { die("--save-prefix: the save point (\(savePoint) tokens) is inside the restored prefix (\(fed.count))") }
var saveSec = 0.0
let pfBusy0 = engine.store.snapshot { $0.driveBusySec }
let pfReads0 = engine.store.snapshot { $0.reads }
let pfWait0 = engine.tm.readWait
let pfGpu0 = gpuStats()
var prefillSecs = 0.0
// the chunk's activations: MLX's peak during the prefill over its active memory before it (the peak reset here)
let mlxActiveBefore = Memory.activeMemory
Memory.peakMemory = 0
func feed(_ toks: ArraySlice<Int>) {
    let tf = now()
    logits = engine.prefill(toks)  // --prefill-chunk C: chunks of C positions; 0: step() per token
    fed += toks
    eval(logits)
    prefillSecs += now() - tf
}
let fedAtStart = fed.count
if savePoint > fed.count { feed(ids[fed.count..<savePoint]) }
eval(logits)
let tSeg1 = now()
if let savePath, fed.count == savePoint {
    let r = try! engine.savePrefix(path: savePath, ids: fed, logits: logits)
    saveSec = r.seconds
    prefixRun["saved_tokens"] = r.n
    prefixRun["save_s"] = r.seconds
    prefixRun["save_mib"] = Double(r.bytes) / 1048576
    prefixRun["save_file"] = savePath
    var extra = ""
    if let pcache {
        let gone = pcache.enforceCap()
        let u = pcache.usage
        extra = String(format: " | cache: %d entries, %.2f GB%@", u.entries, Double(u.bytes) / 1e9, gone.isEmpty ? "" : ", \(gone.count) deleted (LRU)")
    }
    log(String(format: "[prefix] saved %d tokens -> %@ (%.1f MiB, %.3fs)%@", r.n, (savePath as NSString).lastPathComponent, Double(r.bytes) / 1048576, r.seconds, extra))
}
let tSaved = now()
if fed.count < ids.count { feed(ids[fed.count...]) }
eval(logits)
let bytesPrefill = engine.store.snapshot { $0.bytes } - bytesBefore
let pfBusy = engine.store.snapshot { $0.driveBusySec } - pfBusy0
let pfGpu1 = gpuStats()
var prefillJSON: [String: Any] = [
    "prefill_chunk": opt.prefillChunk, "prefill_readers": engine.store.readerThreads, "prefill_read_chunks": opt.prefillReadChunks,
    "prefill_tokens": ids.count - fedAtStart, "prefill_s": prefillSecs, "prefill_mib": Double(bytesPrefill) / 1048576,
    "prefill_mb": Double(bytesPrefill) / 1e6, "prefill_records": engine.store.snapshot { $0.reads } - pfReads0,
    "prefill_drive_gb_per_s": prefillSecs > 0 ? Double(bytesPrefill) / prefillSecs / 1e9 : 0,
    "prefill_drive_gb_per_s_busy": pfBusy > 0 ? Double(bytesPrefill) / pfBusy / 1e9 : 0,
    "prefill_gpu_ms": 1000 * (pfGpu1.busy - pfGpu0.busy), "prefill_expert_wait_ms": 1000 * (engine.tm.readWait - pfWait0),
    "prefill_drive_busy_frac": prefillSecs > 0 ? pfBusy / prefillSecs : 0,
    "prefill_ms_per_token": ids.count > fedAtStart ? 1000 * prefillSecs / Double(ids.count - fedAtStart) : 0,
    "prefill_read_ahead": opt.prefillReadAhead, "read_ahead_hprev": opt.readAheadShift ? "shift" : "stale", "reland_lazy": opt.relandLazy,
    "prefill_auto_mb": opt.prefillAutoBytes / 1e6, "prefill_mlx_active_before_gb": Double(mlxActiveBefore) / 1e9,
    "prefill_mlx_peak_gb": Double(Memory.peakMemory) / 1e9, "prefill_act_mb": Double(Memory.peakMemory - mlxActiveBefore) / 1e6,
    "prefill_tok_per_s": prefillSecs > 0 ? Double(ids.count - fedAtStart) / prefillSecs : 0,
]
for (k, v) in engine.pf.dict { prefillJSON[k] = v }
log("{\"prefill\": " + String(data: try! JSONSerialization.data(withJSONObject: prefillJSON, options: [.sortedKeys]), encoding: .utf8)! + "}")
let how = fedAtStart > 0 ? "restored \(fedAtStart) + \(ids.count - fedAtStart) prefilled" : "cold"
log(String(format: "[feed] %@, %d tokens prefilled in %.1fs", how, ids.count - fedAtStart, (tSeg1 - tRestored) + (now() - tSaved)))
prefixRun["prefill_tokens"] = ids.count - fedAtStart
prefixRun["restore_s"] = tRestored - tPre
prefixRun["prefill_s"] = (tSeg1 - tRestored) + (now() - tSaved)
prefixRun["store_mib_before_first_token"] = Double(bytesPrefill) / 1048576
if let logitsOut {
    let v = logits.asType(.float32).asArray(Float.self)
    var hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(v.count),), }"
    while (10 + hdr.utf8.count + 1) % 64 != 0 { hdr += " " }
    hdr += "\n"
    var d = Data([0x93] + Array("NUMPY".utf8) + [1, 0, UInt8(hdr.utf8.count & 0xFF), UInt8(hdr.utf8.count >> 8)])
    d.append(hdr.data(using: .ascii)!)
    v.withUnsafeBytes { d.append(contentsOf: $0) }
    try! d.write(to: URL(fileURLWithPath: logitsOut))
}

engine.resetStats()
if tracePath != nil {
    engine.traceTokens = (ids.count + 100)..<(ids.count + 110)
    engine.store.snapshot { $0.missLog = [:] }
}
let g0 = gpuStats()
var new: [Int] = []
var dumpTop: [[(Int, Float)]] = []
let t0 = now()
var perToken: [Double] = []
var ttft = 0.0
// the thinking segment by token id, and the --thinking-budget close (ChatTemplate.swift); sampling after the prefill
var tracker = ThinkingTracker(promptIds: ids, tokens: ThinkTokens(tokenizer), budget: thinkingBudget)
var phaseCount: [String: Int] = [:]
var forcedAt: [Int] = []
var thinkIds: [Int] = [], answerIds: [Int] = []
if !sampling.isPlainArgmax { engine.sampler = Sampler(sampling) }
for j in 0..<maxNew {
    let forcedTok = forceIds == nil ? tracker.forced(lastText: { tokenizer.decode(tokens: new) }) : nil
    if forcedTok != nil { forcedAt.append(j) }
    let tk = forcedTok ?? engine.sample(logits)
    if j == 0 {
        ttft = now() - tPre  // restore + prefill + save + first sample (the model load excluded)
    }
    let use = forceIds.map { j < $0.count ? $0[j] : tk } ?? tk
    if dumpPath != nil {  // diagnostic: top-5 of the logits that chose this token (adds a read-back per token)
        let lg = logits.asType(.float32)
        let top = argPartition(-lg, kth: 4)[..<5]
        let vals = lg[top]
        eval(top, vals)
        dumpTop.append(zip(top.asArray(UInt32.self), vals.asArray(Float.self)).map { (Int($0), $1) }.sorted { $0.1 > $1.1 })
    }
    if stopIds.contains(use) && forceIds == nil { break }
    new.append(tk)
    let ph = tracker.observe(use)
    phaseCount[ph.rawValue, default: 0] += 1
    if ph == .thinking { thinkIds.append(use) } else if ph == .answer { answerIds.append(use) }
    let ts = now()
    logits = engine.step(use)
    perToken.append(now() - ts)
}
eval(logits)
engine.sampler = nil
let dt = now() - t0
let g1 = gpuStats()
let n = max(1, new.count)
let st = engine.store.snapshot { s in
    (bytes: s.bytes, reads: s.reads, sync: s.syncReads, evicted: s.evicted, early: s.evictedEarly, readSec: s.readSec, queueSec: s.queueSec,
     policySec: s.policySec, span: s.lastReadEnd - s.firstReadStart, chunkReads: s.chunkReads, recLat: s.recordLatSec, busy: s.driveBusySec,
     readsByLayer: s.readsByLayer, bytesByLayer: s.bytesByLayer, codecRecords: s.codecRecords, codecBytesExact: s.codecBytesExact,
     codecBatches: s.codecBatches, decodeGPU: s.decodeGPUSec, decodeWall: s.decodeWallSec, stgWaits: s.stgWaits,
     checked: s.codecChecked, checkBad: s.codecCheckBad)
}
let pol = engine.store.policySeconds
let tm = engine.tm
func r(_ x: Double, _ d: Int = 1) -> Double { (x * pow(10, Double(d))).rounded() / pow(10, Double(d)) }
let gpuMs = 1000 * (g1.busy - g0.busy) / Double(n)
let text = tokenizer.decode(tokens: new)
var j: [String: Any] = [
    "engine": "swift-native", "mode": opt.mode.rawValue, "m": opt.m > 0 ? opt.m : 8, "keep": opt.keep, "evict": opt.evict.rawValue, "drop": !opt.noDrop,
    "slab": engine.cap, "slots": engine.caps.reduce(0, +), "grown": 0, "layer_caps": opt.layerCapsSpec, "slab_list": engine.caps,
    "evicted_in_window_per_token": r(Double(st.early) / Double(n)), "evicted_per_token": r(Double(st.evicted) / Double(n), 2),
    "routers": engine.routers == nil ? "none" : (opt.routers ?? "routers_k4.npz"), "router_dtype": "float16", "router_bits": engine.routers == nil ? 0 : opt.routerBits, "metallib": metallib, "mlx_core": "0.32.3",
    "layout": engine.layout.rawValue, "embeddings_file": engine.embeddingsPath, "native_cache": opt.cacheDir, "native_cache_choice": cacheChoice,
    "resident_source": engine.residentSource, "router_source": engine.routerSource, "model_dir": opt.dir,
    "mapped_weights": engine.weightsMapped, "weights_footprint_gb": r(Double(engine.weightsFootprint) / 1e9, 3),
    "router_gb": r(Double(engine.routers?.nbytes ?? 0) / 1e9, 3),
    "mem_plan": planGB.mapValues { r($0, 3) }, "mem_plan_gb": r(planSum, 3),
    "sync_reads_per_token": r(Double(st.sync) / Double(n), 2), "sync_reads": st.sync,
    "tokens": new.count, "tok_per_s": r(Double(new.count) / dt, 3), "ms_per_token": r(1000 * dt / Double(n)),
    "expert_wait_ms": r(1000 * tm.readWait / Double(n)), "mb_read_per_token": r(Double(st.bytes) / Double(n) / 1048576.0),
    "kb_read_per_token": r(Double(st.bytes) / Double(n) / 1e3), "reads_per_token": r(Double(st.reads) / Double(n), 2),
    "held_experts": r(Double(engine.heldSum) / Double(max(1, engine.heldN))),
    "released_topk_in_memory": r(engine.inMemoryRate, 4),
    "peak_gb": r(Double(Memory.peakMemory) / 1073741824.0, 2), "active_gb": r(Double(Memory.activeMemory) / 1073741824.0, 2),
    "dtype": "float16", "sim_reads": opt.simReads, "store": "int4", "resident_bits": opt.residentBits, "resident_gb": r(Double(engine.res.nbytes) / 1e9, 3),
    "bw_cap": opt.bwCap, "readers": opt.readers, "f_nocache": opt.noCache, "read_path": engine.store.readPath.rawValue,
    "reader_qos": StoreTuning.qosName(engine.store.readerQoS),
    "syncs_per_token": r(Double(tm.syncs) / Double(n), 2), "sync_wait_ms": r(1000 * tm.syncWait / Double(n)),
    "host_ms": r(1000 * (dt - tm.syncWait - tm.readWait) / Double(n)),
    // the native split of the host time (encode thread), per token
    "host_encode_ms": r(1000 * tm.encode / Double(n)), "host_submit_ms": r(1000 * tm.submit / Double(n)),
    "host_graph_ms": r(1000 * (tm.encode - tm.submit) / Double(n)), "async_evals_per_token": r(Double(tm.asyncEvals) / Double(n), 1),
    "policy_ms": r(1000 * st.policySec / Double(n), 3), "evict_host_ms": r(1000 * pol.critical / Double(n), 3), "evict_prep_ms": r(1000 * pol.prepare / Double(n), 3),
    "reader_read_ms_mean": r(1000 * st.readSec / Double(max(1, st.reads)), 2), "reader_queue_ms_mean": r(1000 * st.queueSec / Double(max(1, st.reads)), 2),
    "reader_busy_ms_per_token": r(1000 * st.readSec / Double(n)), "reads_in_flight_mean": r(st.span > 0 ? st.readSec / dt : 0, 2),
    "expert_wait_ms_by_layer": engine.waitByLayer.map { r(1000 * $0 / Double(n), 2) },
    "namer_gpu_wait_ms": r(1000 * engine.namerGPUWait / Double(n)),
    // read overlap (2026-11-03): settings and the drive during the run
    "late_wait": opt.lateWait, "read_chunks": opt.readChunks, "head_start": opt.headStart,
    "name_wait_ms": r(1000 * tm.nameWait / Double(n)), "land_wait_ms": r(1000 * (tm.readWait - tm.nameWait) / Double(n)),
    "chunk_reads_per_token": r(Double(st.chunkReads) / Double(n), 1), "chunk_read_ms_mean": r(1000 * st.readSec / Double(max(1, st.chunkReads)), 2),
    "record_latency_ms_mean": r(1000 * st.recLat / Double(max(1, st.reads)), 2),
    "drive_gb_per_s": r(Double(st.bytes) / dt / 1e9, 3), "drive_busy_frac": r(st.busy / dt, 3),
    "drive_gb_per_s_busy": r(st.busy > 0 ? Double(st.bytes) / st.busy / 1e9 : 0, 3),
    "gpu_ms_per_token": r(gpuMs), "gpu_busy_frac": r((g1.busy - g0.busy) / dt, 3), "gpu_command_buffers_per_token": r(Double(g1.buffers - g0.buffers) / Double(n), 1),
    "gpu_kernels_per_token": r(Double(g1.kernels - g0.kernels) / Double(n), 1),
    "prompt_tokens": ids.count, "prompt_ids": ids,
    // per-layer caps and the compressed-store read path (2026-11-04)
    "reads_by_layer_per_token": st.readsByLayer.map { r(Double($0) / Double(n), 3) },
    "mb_read_by_layer_per_token": st.bytesByLayer.map { r(Double($0) / Double(n) / 1048576.0, 3) },
    "store_codec": opt.storeCodec.rawValue, "layer_paths": engine.store.layerPath, "codec_staging": opt.codecStaging,
    "codec_records_per_token": r(Double(st.codecRecords) / Double(n), 2),
    "codec_mb_exact_per_token": r(Double(st.codecBytesExact) / Double(n) / 1048576.0, 2),
    "codec_batches_per_token": r(Double(st.codecBatches) / Double(n), 2),
    "codec_records_per_batch": r(Double(st.codecRecords) / Double(max(1, st.codecBatches)), 2),
    "decode_gpu_ms_per_token": r(1000 * st.decodeGPU / Double(n), 3), "decode_wall_ms_per_token": r(1000 * st.decodeWall / Double(n), 3),
    "decode_gpu_ms_per_record": r(1000 * st.decodeGPU / Double(max(1, st.codecRecords)), 4),
    "codec_staging_waits": st.stgWaits, "codec_checked": st.checked, "codec_check_mismatches": st.checkBad,
    "text": String(text.prefix(80)), "ids": new,
    // prompting and sampling (2026-11-04)
    "sampling": sampling.dict, "sampling_summary": sampling.summary,
    "enable_thinking": thinkingArg.map { $0 as Any } ?? "default (not passed)",
    "thinking": tracker.summary.merging(["phases": phaseCount, "forced_at": forcedAt]) { a, _ in a },
    "reply_text": text, "thinking_text": tokenizer.decode(tokens: thinkIds), "answer_text": tokenizer.decode(tokens: answerIds),
]
if forceIds != nil { j["forced"] = true }
j["prefill"] = prefillJSON
j["ttft_s"] = ttft
j["first_step_ms"] = (perToken.first ?? 0) * 1000  // the first generated token's step() (its kernels' first use without --kernel-warmup)
if loadPrefixPath != nil || savePrefixPath != nil || prefixCacheDir != nil || prefixTokens > 0 {
    prefixRun["ttft_s"] = ttft
    prefixRun["ttft_excl_save_s"] = ttft - saveSec
    prefixRun["prompt_tokens"] = ids.count
    prefixRun["prefix_tokens"] = prefixTokens
    prefixRun["prefix_cache"] = prefixCacheDir ?? NSNull()
    prefixRun["decode_tok_per_s"] = r(Double(new.count) / dt, 3)
    prefixRun["ids"] = new
    j["prefix_run"] = prefixRun
    let pj = try! JSONSerialization.data(withJSONObject: ["prefix_run": prefixRun], options: [.sortedKeys])
    print(String(data: pj, encoding: .utf8)!)
}
let data = try! JSONSerialization.data(withJSONObject: j, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
if let dumpPath {
    let d: [String: Any] = ["ids": new, "prompt_ids": ids, "top5": dumpTop.map { $0.map { [$0.0, Double($0.1)] } }, "ms_per_token": perToken.map { $0 * 1000 }, "rel": engine.relLog]
    try! JSONSerialization.data(withJSONObject: d).write(to: URL(fileURLWithPath: dumpPath))
}
if let namesOut {  // docs/layer_caps/replay_caps.py replays these through evict_sim.py at the run's caps
    var rows: [[[Int]]] = []
    if opt.mode == .exact {
        let NLs = engine.store.NL
        for t in 0..<(engine.relLog.count / NLs) { rows.append((0..<NLs).map { engine.relLog[t * NLs + $0] }) }
    } else {
        let nl = engine.store.snapshot { $0.nameLog ?? [:] }
        for t in 0..<(ids.count + new.count) { rows.append(nl[t] ?? Array(repeating: [], count: engine.store.NL)) }
    }
    let d: [String: Any] = ["mode": opt.mode.rawValue, "n_prompt": ids.count, "generated": new.count, "slab_list": engine.caps,
                            "evict": opt.evict.rawValue, "drop": !opt.noDrop, "keep": opt.keep, "rows": rows,
                            "reads_by_layer_gen": st.readsByLayer, "bytes_gen": st.bytes]
    try! JSONSerialization.data(withJSONObject: d).write(to: URL(fileURLWithPath: namesOut))
}
if let tracePath {
    let ev = engine.store.snapshot { $0.events }
    let t0 = ev.map { $0.a }.min() ?? 0
    let d = ev.map { [$0.k, $0.L, $0.e, ($0.a - t0) * 1000, $0.b == 0 ? 0 : ($0.b - t0) * 1000] as [Any] }
    try! JSONSerialization.data(withJSONObject: d).write(to: URL(fileURLWithPath: tracePath))
    // every generated token: step start (ms) and records admitted per layer, for docs/g4_runs/lag4_model.py
    let ml = engine.store.snapshot { $0.missLog ?? [:] }
    let s0 = engine.stepStarts.values.min() ?? 0
    let all: [[String: Any]] = engine.stepStarts.keys.sorted().map { ["t": $0, "start_ms": (engine.stepStarts[$0]! - s0) * 1000, "misses": ml[$0] ?? []] }
    try! JSONSerialization.data(withJSONObject: all).write(to: URL(fileURLWithPath: tracePath + ".tokens.json"))
}
if !bench { print("\n" + text) }
exit(0)
