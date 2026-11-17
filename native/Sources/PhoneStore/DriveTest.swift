// DriveTest: the drive under the engine's own reader pool, without the model (docs/G4-runtime.md "Drive throughput on the
// phone"). A real Store (the engine's slabs, reader threads, chunking, F_NOCACHE handling, read paths) reads records of
// the real store files in a matrix of settings, each cell `cellSeconds` long:
//   readers {4, 8, 16} x read chunks {1, 4} x pattern {random, burst, contig} x cache {nocache, page} x load {none, gpu, cpu}
// plus the read path / destination (iovec into the slab pieces, staged + memcpy, one flat Metal buffer, one flat heap
// buffer) and the reader QoS, with a baseline cell repeated through the run to show drift (thermal, time). Patterns:
// random = records at random (layer, expert), kept queued (2 x readers, at least 32, in flight), as decode's naming keeps
// the queue full; burst = `recordsPerToken` records queued at once, all landed, then a `gapMs` idle gap (decode's cadence);
// contig = records in file order, kept queued. Loads: gpu = a loop of fp16 matmuls [5120 x 10240] @ [10240 x 64] (105 MB
// of weights per op) on its own MLX stream, cpu = one spinning thread at user-interactive QoS (the encode thread busy).
// Page-cache cells start cold: every cached cell is preceded and followed by msync(MS_INVALIDATE) of the store files, and
// the resident fraction (mincore) is sampled before every cell.
import Foundation
import MLX
import Cmlx

public struct DriveTestConfig {
    public var dir: String
    public var store: String = "store4"
    public var cellSeconds = 2.0
    public var cap = 19
    public var recordsPerToken = 132
    public var gapMs = 300.0
    public var readersSet = [4, 8, 16]
    public var chunksSet = [1, 4]
    public var patterns = ["random", "burst", "contig"]
    public var caches = ["nocache", "page"]
    public var loads = ["none", "gpu", "cpu"]
    /// the read-path / destination and QoS sections (before and after the main matrix)
    public var extraSections = true
    /// files whose newest mtime is "the last time the phone was busy with us" (the app: autotest.json, mem.jsonl, ...)
    public var busyMarkers: [String] = []
    /// > 0: copy the first N layer files to fresh, contiguously written files (<dir>/drivetest_fresh/, preallocated, one
    /// sequential pass) and read the same records from both, alternating ("fresh" section); deleted at the end
    public var freshLayers = 0
    /// only the fresh section (and the baseline), not the matrix
    public var freshOnly = false
    public init(dir: String) { self.dir = dir }
}

// MARK: - Store hooks (DriveTest only; the engine never calls these)

extension Store {
    /// Readers in use, reads per record, read path, reader QoS, F_NOCACHE (+ readahead off) or the page cache (+ readahead).
    func driveConfigure(readers: Int, chunks: Int, path: ReadPath, qos: qos_class_t, noCache: Bool) {
        qmon.withLock {
            readerLimit = readers
            chunkRanges = Store.ranges(rec: REC, max(1, chunks))
            readPath = path
            readerQoS = qos
            qmon.broadcast()
        }
        for fd in fds {
            _ = fcntl(fd, F_NOCACHE, noCache ? 1 : 0)
            _ = fcntl(fd, F_RDAHEAD, noCache ? 0 : 1)
        }
    }

    /// Queue record (L, e) into `slot` as a demand read (all its chunks); false if (L, e) is in flight already.
    func driveTrySubmit(_ L: Int, _ e: Int, _ slot: Int) -> Bool {
        mon.withLock {
            if inflight[L][e] { return false }
            inflight[L][e] = true
            nInflight[L] += 1
            submit(L, e, slot)
            return true
        }
    }

    var driveInflight: Int { mon.withLock { nInflight.reduce(0, +) } }

    /// Wait until at most `n` records are in flight.
    func driveWait(atMost n: Int) {
        mon.withLock { while nInflight.reduce(0, +) > n { mon.wait() } }
    }

    func driveLogging(_ on: Bool) -> (rec: [Double], chunk: [Double]) {
        mon.withLock {
            let r = (driveRecLat, driveChunkLat)
            driveLog = on
            driveRecLat = []
            driveChunkLat = []
            return r
        }
    }
}

// MARK: - the one-time setup's reader calibration (2026-11-07, docs/G4-runtime.md "Launch: once per install")

extension Store {
    /// Read throughput with `n` reader threads, for each n of `candidates` (at most readerThreads): random whole records
    /// (random layer, expert and slot, as decode and prefill name them), max(32, 2 n) records queued, `seconds` each,
    /// through the store's own path (F_NOCACHE, read chunks, compressed layers decoded into the slot). GB/s of bytes read
    /// from the drive. Writes only slots that hold no held record: call with nothing held (right after load), at a token
    /// boundary; afterwards every reader thread takes jobs again as before (readerLimit, boost). Records landed here are
    /// real records (slotHas says which), held by no one.
    public func calibrateReaders(_ candidates: [Int], seconds: Double) -> [(readers: Int, gbps: Double)] {
        var rng = SystemRandomNumberGenerator()
        var out: [(readers: Int, gbps: Double)] = []
        let held = mon.withLock { slotOf.reduce(0) { $0 + $1.reduce(0) { $0 + ($1 >= 0 ? 1 : 0) } } }
        guard held == 0, !slabsReleased, !opt.policyOnly else { return [] }
        for n in candidates where n >= 1 && n <= readerThreads {
            qmon.withLock { readerLimit = n; boost = true; qmon.broadcast() }
            let depth = max(32, 2 * n)
            let b0 = mon.withLock { bytes }
            let t0 = now()
            while now() - t0 < seconds {
                while driveInflight < depth {
                    let L = Int.random(in: 0..<NL, using: &rng), e = Int.random(in: 0..<NE, using: &rng)
                    _ = driveTrySubmit(L, e, Int.random(in: 0..<caps[L], using: &rng))
                }
                driveWait(atMost: depth - 1)
            }
            driveWait(atMost: 0)
            let wall = now() - t0
            out.append((n, Double(mon.withLock { bytes } - b0) / wall / 1e9))
        }
        qmon.withLock { readerLimit = Int.max; boost = false; qmon.broadcast() }
        return out
    }

    /// Reader threads that take jobs (the rest sleep): the calibrated count in the session that measured it.
    public func limitReaders(_ n: Int) {
        qmon.withLock { readerLimit = n; qmon.broadcast() }
    }
}

// MARK: - loads

/// A persistent worker that runs `body` repeatedly while switched on (one thread, created once).
final class LoadThread: @unchecked Sendable {
    let mon = Monitor()
    var on = false
    var count = 0
    var busySec = 0.0
    init(name: String, qos: qos_class_t = QOS_CLASS_USER_INTERACTIVE, setup: @escaping () -> (() -> Void)) {
        let t = Thread { [unowned self] in
            pthread_set_qos_class_self_np(qos, 0)
            let body = setup()
            while true {
                self.mon.lock()
                while !self.on { self.mon.wait() }
                self.mon.unlock()
                let a = now()
                body()
                let b = now()
                self.mon.withLock { self.count += 1; self.busySec += b - a }
            }
        }
        t.name = name
        t.stackSize = 4 << 20
        t.start()
    }
    func set(_ v: Bool) -> (count: Int, busy: Double) {
        mon.withLock {
            let r = (count, busySec)
            on = v
            count = 0
            busySec = 0
            mon.broadcast()
            return r
        }
    }
}

// MARK: - page cache and file layout

enum StoreFiles {
    /// Drop the store files' cached pages: mmap + msync(MS_INVALIDATE) per file (clean pages are discarded).
    static func invalidate(_ fds: [Int32], size: Int) {
        for fd in fds {
            guard let p = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0), p != MAP_FAILED else { continue }
            _ = msync(p, size, MS_INVALIDATE)
            munmap(p, size)
        }
    }

    /// Fraction of the store files' 16 KB pages resident in memory (mincore over each file mapped in turn).
    static func residentFraction(_ fds: [Int32], size: Int) -> Double {
        let pg = Int(getpagesize())
        var vec = [CChar](repeating: 0, count: (size + pg - 1) / pg)
        var res = 0, tot = 0
        for fd in fds {
            guard let p = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0), p != MAP_FAILED else { continue }
            if mincore(p, size, &vec) == 0 {
                for v in vec where (Int32(v) & MINCORE_INCORE) != 0 { res += 1 }
                tot += vec.count
            }
            munmap(p, size)
        }
        return tot > 0 ? Double(res) / Double(tot) : -1
    }

    /// Copy the first `size` bytes of `src` to a new file `dst`: preallocated (F_PREALLOCATE, contiguous when the volume can
    /// give it), written in order in 16 MB writes with F_NOCACHE, then F_FULLFSYNC.
    static func copyContiguous(src: Int32, dst: String, size: Int) throws -> [String: Any] {
        let t0 = now()
        let fd = open(dst, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        guard fd >= 0 else { throw PSError("create \(dst): \(String(cString: strerror(errno)))") }
        defer { close(fd) }
        var mode = "none"
        var fst = fstore_t(fst_flags: UInt32(F_ALLOCATECONTIG | F_ALLOCATEALL), fst_posmode: F_PEOFPOSMODE, fst_offset: 0,
                           fst_length: off_t(size), fst_bytesalloc: 0)
        if fcntl(fd, F_PREALLOCATE, &fst) == 0 {
            mode = "contiguous"
        } else {
            fst.fst_flags = UInt32(F_ALLOCATEALL)
            if fcntl(fd, F_PREALLOCATE, &fst) == 0 { mode = "all (not contiguous)" }
        }
        _ = fcntl(fd, F_NOCACHE, 1)
        _ = fcntl(src, F_NOCACHE, 1)
        let chunk = 16 << 20
        var raw: UnsafeMutableRawPointer? = nil
        posix_memalign(&raw, 16384, chunk)
        guard let buf = raw else { throw PSError("copy buffer") }
        defer { free(buf) }
        var off = 0
        while off < size {
            let n = min(chunk, size - off)
            guard pread(src, buf, n, off_t(off)) == n else { throw PSError("copy: short read at \(off)") }
            guard pwrite(fd, buf, n, off_t(off)) == n else { throw PSError("copy: short write at \(off) (\(String(cString: strerror(errno))))") }
            off += n
        }
        _ = fcntl(fd, F_FULLFSYNC)
        let dt = now() - t0
        return ["prealloc": mode, "seconds": dt, "gbps": Double(size) / dt / 1e9]
    }

    /// True if the first `size` bytes of the two files are equal (16 MB reads, F_NOCACHE).
    static func sameBytes(_ a: Int32, _ b: Int32, size: Int) -> Bool {
        _ = fcntl(a, F_NOCACHE, 1)
        _ = fcntl(b, F_NOCACHE, 1)
        let chunk = 16 << 20
        var ra: UnsafeMutableRawPointer? = nil, rb: UnsafeMutableRawPointer? = nil
        posix_memalign(&ra, 16384, chunk)
        posix_memalign(&rb, 16384, chunk)
        defer { free(ra); free(rb) }
        var off = 0
        while off < size {
            let n = min(chunk, size - off)
            if pread(a, ra, n, off_t(off)) != n || pread(b, rb, n, off_t(off)) != n || memcmp(ra, rb, n) != 0 { return false }
            off += n
        }
        return true
    }

    /// The file's extents (F_LOG2PHYS_EXT, physically adjacent ones merged) and how many records span an extent boundary.
    static func extents(_ fd: Int32, size: Int, record: Int) -> [String: Any] {
        var starts: [Int] = []  // file offsets where a (merged) extent starts
        var raw = 0
        var off = 0, prevEnd = -1
        while off < size {
            var l2p = log2phys()
            l2p.l2p_devoffset = off_t(off)
            l2p.l2p_contigbytes = off_t(size - off)
            if fcntl(fd, F_LOG2PHYS_EXT, &l2p) == -1 { return ["error": String(cString: strerror(errno)), "at": off] }
            let c = Int(l2p.l2p_contigbytes), dev = Int(l2p.l2p_devoffset)
            if c <= 0 { return ["error": "zero-length extent", "at": off] }
            raw += 1
            if dev != prevEnd { starts.append(off) }
            prevEnd = dev + c
            off += c
        }
        var split = 0, idx = 0
        let nrec = size / record
        for e in 0..<nrec {
            let a = e * record, b = a + record
            while idx < starts.count && starts[idx] <= a { idx += 1 }
            if idx < starts.count && starts[idx] < b { split += 1 }
        }
        return ["extents": starts.count, "raw_extents": raw, "records_split": split, "mean_extent_mb": Double(size) / Double(max(1, starts.count)) / 1e6]
    }
}

// MARK: - run

func pct(_ v: [Double], _ q: Double) -> Double {
    if v.isEmpty { return 0 }
    let s = v.sorted()
    return s[min(s.count - 1, max(0, Int((q * Double(s.count - 1)).rounded())))]
}

func thermalNow() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

struct DriveCell {
    var section: String
    var readers: Int
    var chunks: Int
    var pattern: String
    var cache: String
    var load: String
    var path: String  // "iovec" | "staged" | "flat-metal" | "flat-heap"
    var qos: String
    /// "store": the store files; "fresh": the contiguous copies of the first freshLayers layers (and only those layers)
    var file: String = "store"
    var label: String { "r\(readers) c\(chunks) \(pattern) \(cache) \(path) qos=\(qos) load=\(load)" + (file == "store" ? "" : " file=\(file)") }
}

/// The whole matrix. Returns the JSON object written as drivetest.json.
public func runDriveTest(_ c: DriveTestConfig, log: @escaping (String) -> Void) throws -> [String: Any] {
    let tStart = now()
    let wallStart = Date()
    var prevLimit = 0
    if let ws = MTLCreateSystemDefaultDeviceShim() { _ = mlx_set_wired_limit(&prevLimit, ws) }  // as the engine (opt.wire)
    let maxReaders = c.readersSet.max() ?? 8
    var so = StoreOptions(cap: c.cap, keep: 31, evict: .hazard, noDrop: true, readers: maxReaders, simReads: false, noCache: true,
                          bwCap: 0, chunks: 4)
    so.prefillReaders = 0
    let st = try Store(dir: c.dir, store: c.store, opt: so)
    let NL = st.NL, NE = st.NE, REC = st.REC, fileBytes = NE * REC
    log("DRIVE store \(c.dir)/\(c.store): \(NL) layers x \(NE) records of \(REC) B, cap \(c.cap) (slabs \(String(format: "%.2f", Double(st.slabBytes) / 1e9)) GB), \(maxReaders) reader threads")

    // the last time we kept the phone busy (newest marker file), and the state at the start
    var lastBusy: Double? = nil
    for p in c.busyMarkers {
        if let a = try? FileManager.default.attributesOfItem(atPath: p), let d = a[.modificationDate] as? Date {
            let s = Date().timeIntervalSince(d)
            lastBusy = min(lastBusy ?? s, s)
        }
    }

    // file layout
    var layout: [[String: Any]] = []
    var totalExt = 0, totalSplit = 0
    for L in 0..<NL {
        var x = StoreFiles.extents(st.fds[L], size: fileBytes, record: REC)
        x["layer"] = L
        totalExt += x["extents"] as? Int ?? 0
        totalSplit += x["records_split"] as? Int ?? 0
        layout.append(x)
    }
    log("DRIVE layout: \(totalExt) extents over \(NL) files, \(totalSplit) of \(NL * NE) records span an extent boundary"
        + ((layout.first?["error"]).map { " (F_LOG2PHYS_EXT: \($0))" } ?? ""))

    // destinations for the flat cells: one Metal (MLX, shared) buffer and one heap buffer, 64 records each
    let flatN = 64
    let flatArr = MLXArray.zeros([flatN * REC], dtype: .uint8)
    eval(flatArr)
    let flatMetal = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(mlx_array_data_uint8(flatArr.ctx)!))
    var heapRaw: UnsafeMutableRawPointer? = nil
    posix_memalign(&heapRaw, 16384, flatN * REC)
    memset(heapRaw!, 0, flatN * REC)

    // loads
    let gpuBytesPerOp = 5120 * 10240 * 2
    let gpu = LoadThread(name: "drive-gpu") {
        var w: MLXArray! = nil, x: MLXArray! = nil
        let s = Stream(.gpu)
        w = (MLXRandom.normal([5120, 10240], stream: .stream(s)) * 0.02).asType(.float16, stream: .stream(s))
        x = MLXRandom.normal([10240, 64], stream: .stream(s)).asType(.float16, stream: .stream(s))
        eval(w, x)
        return {
            let y = matmul(w, x, stream: .stream(s))
            eval(y)
        }
    }
    let cpu = LoadThread(name: "drive-cpu") {
        return {
            var a: UInt64 = 0x9E37_79B9_7F4A_7C15
            for _ in 0..<200_000 { a = a &* 6364136223846793005 &+ 1442695040888963407 }
            if a == 42 { print("") }
        }
    }
    // calibrate the GPU op alone (no reads)
    _ = gpu.set(true)
    usleep(1_500_000)
    let gcal = gpu.set(false)
    let gpuOpMs = gcal.count > 0 ? 1000 * gcal.busy / Double(gcal.count) : 0
    log(String(format: "DRIVE gpu load alone: %.2f ms per op, %.1f GB/s of weights read (%d ops in 1.5 s)", gpuOpMs,
               gcal.count > 0 ? Double(gpuBytesPerOp) / (gcal.busy / Double(gcal.count)) / 1e9 : 0, gcal.count))

    // the cells
    var cells: [DriveCell] = []
    let base = DriveCell(section: "baseline", readers: 8, chunks: 4, pattern: "random", cache: "nocache", load: "none", path: "iovec", qos: "ui")
    func addDest(_ sec: String) {
        for ch in [1, 4] { for p in ["iovec", "staged", "flat-metal", "flat-heap"] { for ld in ["none", "gpu"] {
            cells.append(DriveCell(section: sec, readers: 8, chunks: ch, pattern: "random", cache: "nocache", load: ld, path: p, qos: "ui"))
        } } }
    }
    // fresh: the first freshLayers layer files copied contiguously; the same reads from the originals (restricted to those
    // layers) and the copies, alternating, two rounds
    var freshFds: [Int32] = []
    var freshInfo: [[String: Any]] = []
    let freshDir = "\(c.dir)/drivetest_fresh"
    if c.freshLayers > 0 {
        try? FileManager.default.removeItem(atPath: freshDir)
        try FileManager.default.createDirectory(atPath: freshDir, withIntermediateDirectories: true)
        for L in 0..<min(NL, c.freshLayers) {
            let dst = "\(freshDir)/layer_\(String(format: "%02d", L)).bin"
            var x = try StoreFiles.copyContiguous(src: st.fds[L], dst: dst, size: fileBytes)
            let fd = open(dst, O_RDONLY)
            if fd < 0 { throw PSError("open \(dst)") }
            freshFds.append(fd)
            let before = StoreFiles.extents(st.fds[L], size: fileBytes, record: REC), after = StoreFiles.extents(fd, size: fileBytes, record: REC)
            x["layer"] = L
            x["extents_store"] = before["extents"] ?? -1
            x["records_split_store"] = before["records_split"] ?? -1
            x["extents_fresh"] = after["extents"] ?? -1
            x["records_split_fresh"] = after["records_split"] ?? -1
            x["identical"] = StoreFiles.sameBytes(st.fds[L], fd, size: fileBytes)
            freshInfo.append(x)
            log("DRIVE fresh copy of layer \(L): \(x["prealloc"]!), \(String(format: "%.2f", x["gbps"] as! Double)) GB/s written; extents \(x["extents_store"]!) -> \(x["extents_fresh"]!), records split \(x["records_split_store"]!) -> \(x["records_split_fresh"]!), identical \(x["identical"]!)")
        }
    }
    func addFresh() {
        for _ in 0..<2 { for f in ["store", "fresh"] {
            for (ch, ld, pa) in [(4, "none", "random"), (1, "none", "random"), (4, "gpu", "random"), (4, "none", "contig")] {
                cells.append(DriveCell(section: "fresh", readers: 8, chunks: ch, pattern: pa, cache: "nocache", load: ld, path: "iovec", qos: "ui", file: f))
            }
        } }
    }

    cells.append(base)
    if c.freshLayers > 0 { addFresh() }
    if c.extraSections && !c.freshOnly {
        addDest("dest")
        for q in ["ui", "utility"] { for ld in ["none", "cpu"] {
            cells.append(DriveCell(section: "qos", readers: 8, chunks: 4, pattern: "random", cache: "nocache", load: ld, path: "iovec", qos: q))
        } }
    }
    var n = 0
    for ca in c.freshOnly ? [] : c.caches { for pa in c.patterns { for r in c.readersSet { for ch in c.chunksSet {
        if n % 4 == 0 { cells.append(base) }
        for ld in c.loads { cells.append(DriveCell(section: "matrix", readers: r, chunks: ch, pattern: pa, cache: ca, load: ld, path: "iovec", qos: "ui")) }
        n += 1
    } } } }
    if !c.freshOnly {
        cells.append(base)
        if c.extraSections { addDest("dest-end") }
    }
    cells.append(base)

    var out: [[String: Any]] = []
    var rng = SystemRandomNumberGenerator()
    var contigPos = 0
    StoreFiles.invalidate(st.fds, size: fileBytes)
    let res0 = StoreFiles.residentFraction(st.fds, size: fileBytes)
    log(String(format: "DRIVE %d cells of %.1f s; store pages resident at start (after invalidate): %.4f", cells.count, c.cellSeconds, res0))
    for (ci, cell) in cells.enumerated() {
        let path: ReadPath = cell.path == "iovec" ? .iovec : (cell.path == "staged" ? .staged : .flat)
        if path == .flat {
            st.flatBase = cell.path == "flat-metal" ? flatMetal : heapRaw!
            st.flatSlots = flatN
        }
        st.driveConfigure(readers: cell.readers, chunks: cell.chunks, path: path, qos: StoreTuning.qos(named: cell.qos) ?? QOS_CLASS_USER_INTERACTIVE,
                          noCache: cell.cache == "nocache")
        let storeFds = st.fds
        let nl = cell.file == "fresh" ? freshFds.count : (cell.section == "fresh" ? freshFds.count : NL)  // fresh section: both on the same layers
        if cell.file == "fresh" { for L in 0..<freshFds.count { st.fds[L] = freshFds[L] } }
        defer { st.fds = storeFds }
        let resBefore = (ci % 10 == 0 || cell.cache == "page") ? StoreFiles.residentFraction(st.fds, size: fileBytes) : -1
        let th0 = thermalNow()
        if cell.load == "gpu" { _ = gpu.set(true) }
        if cell.load == "cpu" { _ = cpu.set(true) }
        if cell.load != "none" { usleep(200_000) }
        st.resetStats()
        _ = st.driveLogging(true)
        let depth = max(32, 2 * cell.readers)
        var tokMs: [Double] = []
        let t0 = now()
        func pick() -> (Int, Int) { (Int.random(in: 0..<nl, using: &rng), Int.random(in: 0..<NE, using: &rng)) }
        switch cell.pattern {
        case "burst":
            while now() - t0 < c.cellSeconds || tokMs.count < 3 {
                let a = now()
                var queued = 0
                while queued < c.recordsPerToken {
                    let (L, e) = pick()
                    if st.driveTrySubmit(L, e, Int.random(in: 0..<st.caps[L], using: &rng)) { queued += 1 }
                }
                st.driveWait(atMost: 0)
                tokMs.append(1000 * (now() - a))
                if now() - t0 >= c.cellSeconds && tokMs.count >= 3 { break }
                usleep(UInt32(c.gapMs * 1000))
            }
        default:
            while now() - t0 < c.cellSeconds {
                while st.driveInflight < depth {
                    let L: Int, e: Int
                    if cell.pattern == "contig" {
                        L = (contigPos / NE) % nl; e = contigPos % NE
                        contigPos += 1
                    } else {
                        (L, e) = pick()
                    }
                    _ = st.driveTrySubmit(L, e, Int.random(in: 0..<st.caps[L], using: &rng))
                }
                st.driveWait(atMost: depth - 1)
            }
        }
        st.driveWait(atMost: 0)
        let wall = now() - t0
        let lat = st.driveLogging(false)
        let g = cell.load == "gpu" ? gpu.set(false) : (count: 0, busy: 0.0)
        let cp = cell.load == "cpu" ? cpu.set(false) : (count: 0, busy: 0.0)
        let s = st.snapshot { ($0.bytes, $0.reads, $0.chunkReads, $0.driveBusySec, $0.readSec) }
        let th1 = thermalNow()
        // page cache: the share of the store now resident (what the cell left in memory; checks that mincore sees it), then drop it
        let resAfter = cell.cache == "page" ? StoreFiles.residentFraction(st.fds, size: fileBytes) : -1
        if cell.cache == "page" { StoreFiles.invalidate(st.fds, size: fileBytes) }
        var o: [String: Any] = [
            "i": ci, "section": cell.section, "readers": cell.readers, "chunks": cell.chunks, "pattern": cell.pattern,
            "cache": cell.cache, "load": cell.load, "path": cell.path, "qos": cell.qos, "file": cell.file, "layers": nl,
            "t": now() - tStart, "seconds": wall, "bytes": s.0, "records": s.1, "reads": s.2,
            "gbps": Double(s.0) / wall / 1e9, "gbps_busy": s.3 > 0 ? Double(s.0) / s.3 / 1e9 : 0, "busy_frac": s.3 / wall,
            "in_flight_mean": s.4 / wall,
            "read_ms_p50": 1000 * pct(lat.chunk, 0.5), "read_ms_p95": 1000 * pct(lat.chunk, 0.95), "read_ms_p99": 1000 * pct(lat.chunk, 0.99),
            "record_ms_p50": 1000 * pct(lat.rec, 0.5), "record_ms_p95": 1000 * pct(lat.rec, 0.95), "record_ms_p99": 1000 * pct(lat.rec, 0.99),
            "thermal_start": th0, "thermal_end": th1,
        ]
        if resBefore >= 0 { o["resident_before"] = resBefore }
        if resAfter >= 0 { o["resident_after"] = resAfter }
        if !tokMs.isEmpty {
            o["tokens"] = tokMs.count
            o["token_read_ms_mean"] = tokMs.reduce(0, +) / Double(tokMs.count)
            o["token_read_ms_p50"] = pct(tokMs, 0.5)
            o["token_read_ms_max"] = tokMs.max()!
        }
        if cell.load == "gpu" {
            o["gpu_ops"] = g.count
            o["gpu_ms_per_op"] = g.count > 0 ? 1000 * g.busy / Double(g.count) : 0
            o["gpu_weight_gbps"] = Double(g.count * gpuBytesPerOp) / wall / 1e9
        }
        if cell.load == "cpu" { o["cpu_spins"] = cp.count }
        out.append(o)
        log(String(format: "DRIVE %3d/%d %-9@ %@: %.3f GB/s (busy %.3f, %.0f%%), read p50 %.1f ms, record p50/p95 %.1f/%.1f ms%@ | %@",
                   ci + 1, cells.count, cell.section, cell.label, o["gbps"] as! Double, o["gbps_busy"] as! Double, 100 * (o["busy_frac"] as! Double),
                   o["read_ms_p50"] as! Double, o["record_ms_p50"] as! Double, o["record_ms_p95"] as! Double,
                   cell.load == "gpu" ? String(format: ", gpu %.1f ms/op", o["gpu_ms_per_op"] as! Double) : "",
                   th1))
    }
    for fd in freshFds { close(fd) }
    if c.freshLayers > 0 { try? FileManager.default.removeItem(atPath: freshDir) }
    st.driveConfigure(readers: Int.max, chunks: so.chunks, path: StoreTuning.readPath, qos: StoreTuning.readerQoS, noCache: true)
    free(heapRaw)
    var env: [String: Any] = [
        "time": ISO8601DateFormatter().string(from: wallStart), "seconds_total": now() - tStart,
        "machine": { var u = utsname(); uname(&u); return withUnsafeBytes(of: &u.machine) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) } }(),
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "uptime_s": ProcessInfo.processInfo.systemUptime, "thermal_at_start": cells.isEmpty ? "" : (out.first?["thermal_start"] ?? ""),
        "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
        "store": "\(c.dir)/\(c.store)", "record_bytes": REC, "cap": c.cap, "cell_seconds": c.cellSeconds,
        "records_per_token": c.recordsPerToken, "gap_ms": c.gapMs, "reader_threads": maxReaders,
        "gpu_load": ["shape": "[5120 x 10240] fp16 @ [10240 x 64]", "bytes_per_op": gpuBytesPerOp, "alone_ms_per_op": gpuOpMs],
        "resident_at_start": res0, "layout": ["extents": totalExt, "records_split": totalSplit, "per_layer": layout],
        "cells": out, "fresh": freshInfo,
    ]
    if let lastBusy { env["seconds_since_last_busy"] = lastBusy }
    return env
}

// MARK: - defragment the store

/// Rewrite the store's layer files whose records span extent boundaries (more than `minSplit` of them) as contiguously
/// written files: per layer a copy (StoreFiles.copyContiguous) to layer_XX.bin.defrag, kept only if it has fewer split
/// records and its bytes equal the original's (a full compare), then the original's times are given to it and it is
/// renamed over the original (atomic: a kill leaves the original and a stray .defrag). `layers`: nil = every layer.
/// Run with no Store open on the files. Stops when the volume has less than 1 GB free.
public func defragmentStore(dir: String, store: String = "store4", layers: [Int]? = nil, minSplit: Double = 0.05,
                            log: @escaping (String) -> Void) throws -> [String: Any] {
    let meta = try StoreMeta(path: "\(dir)/\(store)/meta.json")
    let REC = meta.recordBytes, size = meta.experts * REC
    var rows: [[String: Any]] = []
    let t0 = now()
    var done = 0, kept = 0
    for L in layers ?? Array(0..<meta.layers) {
        let path = "\(dir)/\(store)/layer_\(String(format: "%02d", L)).bin", tmp = path + ".defrag"
        var row: [String: Any] = ["layer": L]
        defer { rows.append(row) }
        let fd = open(path, O_RDONLY)
        if fd < 0 { row["skipped"] = "no .bin"; continue }
        defer { close(fd) }
        var sb = stat()
        fstat(fd, &sb)
        if Int(sb.st_size) != size { row["skipped"] = "size \(sb.st_size) != \(size)"; continue }
        let before = StoreFiles.extents(fd, size: size, record: REC)
        row["extents_before"] = before["extents"] ?? -1
        row["records_split_before"] = before["records_split"] ?? -1
        let splitBefore = before["records_split"] as? Int ?? meta.experts
        if Double(splitBefore) < minSplit * Double(meta.experts) { row["skipped"] = "not fragmented"; continue }
        if let a = try? FileManager.default.attributesOfFileSystem(forPath: dir), let free = (a[.systemFreeSize] as? NSNumber)?.intValue,
           free < size + (1 << 30) {
            row["skipped"] = "free space \(free) B"
            log("DEFRAG stop: \(free) B free")
            break
        }
        do {
            let x = try StoreFiles.copyContiguous(src: fd, dst: tmp, size: size)
            row["prealloc"] = x["prealloc"]
            row["write_gbps"] = x["gbps"]
            let nfd = open(tmp, O_RDWR)
            if nfd < 0 { throw PSError("open \(tmp)") }
            defer { close(nfd) }
            let after = StoreFiles.extents(nfd, size: size, record: REC)
            row["extents_after"] = after["extents"] ?? -1
            row["records_split_after"] = after["records_split"] ?? -1
            let splitAfter = after["records_split"] as? Int ?? meta.experts
            if splitAfter >= splitBefore {
                unlink(tmp)
                row["result"] = "no gain: copy removed"
            } else if !StoreFiles.sameBytes(fd, nfd, size: size) {
                unlink(tmp)
                row["result"] = "COPY DIFFERS: removed, original kept"
            } else {
                var ts = [sb.st_atimespec, sb.st_mtimespec]
                _ = futimens(nfd, &ts)
                if rename(tmp, path) != 0 { throw PSError("rename \(tmp): \(String(cString: strerror(errno)))") }
                row["result"] = "replaced"
                kept += 1
            }
        } catch {
            unlink(tmp)
            row["result"] = "error: \(error)"
        }
        done += 1
        log("DEFRAG layer \(L): extents \(row["extents_before"]!) -> \(row["extents_after"] ?? "-"), records split \(row["records_split_before"]!) -> \(row["records_split_after"] ?? "-"), \(row["prealloc"] ?? "") : \(row["result"] ?? row["skipped"] ?? "")")
    }
    return ["time": ISO8601DateFormatter().string(from: Date()), "seconds": now() - t0, "store": "\(dir)/\(store)", "layers_copied": done,
            "layers_replaced": kept, "per_layer": rows]
}
