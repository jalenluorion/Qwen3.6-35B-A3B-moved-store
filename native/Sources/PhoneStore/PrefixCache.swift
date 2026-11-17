// Prefix state on disk (docs/PREFIX-CACHE.md, the native port of mlx_run.py --save-prefix / --load-prefix / --prefix-cache).
//
// "The model after feeding tokens 0..n-1" is: the token ids; per attention layer the KV cache up to its offset (and the
// buffer capacity, so the restored buffer has the same layout: KVCache grows in steps of 256); per GatedDeltaNet layer the
// conv window and the recurrent state; the lag routers' h_prev per layer; the token clock; the next-token logits. All as
// held (fp16 / fp32, no precision change). For prefetch mode the computation also depends on which experts are in memory
// (the released top-8 is taken among them), and the cache policy's later choices depend on its history, so the store's
// host state is saved too: per layer expert -> slot, last use, insertion order, the free-slot list in order, the token
// clock of the last naming, and the eviction policy's history (Evict: last, tot, dec, the hazard table, the bucket state,
// its clock). A restore puts every held expert back in its saved slot: a record already in that slot stays, one sitting in
// another slot is copied over (memcpy inside the slab, no drive), the rest are read from the store straight into their
// slots, all in flight at once ("re-land"). Later victim choices, slot indices and kernel inputs are then those of the run
// that saved.
//
// One file per state, a safetensors container written here (device tensors with their MLX dtype, host tables as I64/F64/U8)
// with the metadata: format, n, t, key (SHA-256 of the ids as little-endian int32, the Python key), the cache layout and the
// engine configuration (a state is refused when it differs). Written to a temporary name and renamed, so a file under its
// final name is complete. Arrays are copied out and written one at a time (at most one 2 MiB array held in host memory).
import Foundation
import CryptoKit
import MLX

public let prefixFormat = "phone-store native prefix state 1"

/// The cache key of a token-id prefix: SHA-256 of the ids as little-endian int32 (mlx_run.py prefix_key).
public func prefixKey<C: Collection>(_ ids: C) -> String where C.Element == Int {
    var d = Data(capacity: ids.count * 4)
    for i in ids {
        var v = Int32(truncatingIfNeeded: i).littleEndian
        withUnsafeBytes(of: &v) { d.append(contentsOf: $0) }
    }
    return SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined()
}

public struct PrefixMismatch: Error, CustomStringConvertible {
    public let description: String
    init(_ s: String) { description = s }
}

public struct PrefixSaveInfo: Sendable {
    public let path: String
    public let n: Int
    public let bytes: Int
    public let seconds: Double  // settle reads + copy out + write + rename
}

public struct PrefixLoadInfo: Sendable {
    public let path: String
    public let n: Int  // tokens restored
    public let fileBytes: Int
    public let loadSeconds: Double  // header + model state into MLX (evaluated)
    public let relandSeconds: Double  // slot tables + moves + reads landed + policy
    public let relanded: Int  // records read from the store
    public let kept: Int  // records already in their saved slot
    public let moved: Int  // records copied from another slot of the slab
    public let relandBytes: Int  // bytes read from the store by the re-land
    public let held: Int  // experts held after the restore, all layers
    /// Lazy re-land: held experts not in their slot, left for the next chunk to read if it uses or keeps them (0: eager).
    public var deferred: Int = 0
    public var lazy: Bool = false
    /// This restore with its pending records landed at once (Store.landPending's result and seconds).
    public func landedAtOnce(_ l: (relanded: Int, kept: Int, moved: Int, bytes: Int), seconds: Double) -> PrefixLoadInfo {
        PrefixLoadInfo(path: path, n: n, fileBytes: fileBytes, loadSeconds: loadSeconds, relandSeconds: relandSeconds + seconds,
                       relanded: relanded + l.relanded, kept: kept + l.kept, moved: moved + l.moved, relandBytes: relandBytes + l.bytes, held: held)
    }
    public var dict: [String: Double] {
        ["prefix_tokens_restored": Double(n), "prefix_state_mb": Double(fileBytes) / 1e6, "prefix_load_ms": 1000 * loadSeconds,
         "prefix_reland_ms": 1000 * relandSeconds, "prefix_relanded": Double(relanded), "prefix_kept": Double(kept),
         "prefix_moved": Double(moved), "prefix_reland_mb": Double(relandBytes) / 1e6, "prefix_held": Double(held),
         "prefix_deferred": Double(deferred), "prefix_lazy": lazy ? 1 : 0]
    }
}

// MARK: - a minimal safetensors container

struct STEntry {
    let name: String
    let dtype: String
    let shape: [Int]
    let nbytes: Int
    let bytes: () -> Data
}

func stDType(_ d: DType) -> String {
    switch d {
    case .float16: return "F16"
    case .bfloat16: return "BF16"
    case .float32: return "F32"
    case .int32: return "I32"
    case .uint32: return "U32"
    case .int64: return "I64"
    case .uint8: return "U8"
    case .bool: return "BOOL"
    default: fatalError("prefix state: unsupported dtype \(d)")
    }
}

func mlxDType(_ s: String) throws -> DType {
    switch s {
    case "F16": return .float16
    case "BF16": return .bfloat16
    case "F32": return .float32
    case "I32": return .int32
    case "U32": return .uint32
    case "I64": return .int64
    case "U8": return .uint8
    case "BOOL": return .bool
    default: throw PrefixMismatch("unsupported dtype \(s)")
    }
}

func hostEntry(_ name: String, i64 v: [Int], shape: [Int]? = nil) -> STEntry {
    STEntry(name: name, dtype: "I64", shape: shape ?? [v.count], nbytes: 8 * v.count) {
        v.map { Int64($0).littleEndian }.withUnsafeBytes { Data($0) }
    }
}

func hostEntry(_ name: String, f64 v: [Double]) -> STEntry {
    STEntry(name: name, dtype: "F64", shape: [v.count], nbytes: 8 * v.count) { v.withUnsafeBytes { Data($0) } }
}

func hostEntry(_ name: String, i32 v: [Int]) -> STEntry {
    STEntry(name: name, dtype: "I32", shape: [v.count], nbytes: 4 * v.count) {
        v.map { Int32(truncatingIfNeeded: $0).littleEndian }.withUnsafeBytes { Data($0) }
    }
}

func deviceEntry(_ name: String, _ a: MLXArray) -> STEntry {
    STEntry(name: name, dtype: stDType(a.dtype), shape: a.shape, nbytes: a.size * a.itemSize) { a.asData(access: .copy).data }
}

/// Write `entries` (in order) and `metadata` to path via a temporary file; returns the file size.
func writeSafetensors(path: String, entries: [STEntry], metadata: [String: String]) throws -> Int {
    var hdr: [String: Any] = ["__metadata__": metadata]
    var off = 0
    for e in entries {
        hdr[e.name] = ["dtype": e.dtype, "shape": e.shape, "data_offsets": [off, off + e.nbytes]] as [String: Any]
        off += e.nbytes
    }
    var h = try JSONSerialization.data(withJSONObject: hdr, options: [.sortedKeys])
    while h.count % 8 != 0 { h.append(0x20) }
    let tmp = path + ".tmp"
    FileManager.default.createFile(atPath: tmp, contents: nil)
    guard let fh = FileHandle(forWritingAtPath: tmp) else { throw PSError("cannot write \(tmp)") }
    var n = UInt64(h.count).littleEndian
    try fh.write(contentsOf: withUnsafeBytes(of: &n) { Data($0) })
    try fh.write(contentsOf: h)
    for e in entries {
        let d = e.bytes()
        precondition(d.count == e.nbytes, "prefix state: \(e.name) is \(d.count) B, expected \(e.nbytes)")
        try fh.write(contentsOf: d)
    }
    try fh.close()
    if FileManager.default.fileExists(atPath: path) { try FileManager.default.removeItem(atPath: path) }
    try FileManager.default.moveItem(atPath: tmp, toPath: path)
    return 8 + h.count + off
}

/// A safetensors file opened for reading tensors one at a time (pread).
final class STReader {
    let path: String
    let fd: Int32
    let metadata: [String: String]
    let entries: [String: (dtype: String, shape: [Int], begin: Int, end: Int)]
    let size: Int

    init(path: String) throws {
        self.path = path
        let fd = open(path, O_RDONLY)
        if fd < 0 { throw PrefixMismatch("cannot open \(path)") }
        var n: UInt64 = 0
        guard pread(fd, &n, 8, 0) == 8, n < 1 << 26 else { close(fd); throw PrefixMismatch("\(path): bad header") }
        var h = Data(count: Int(n))
        let got = h.withUnsafeMutableBytes { pread(fd, $0.baseAddress, Int(n), 8) }
        guard got == Int(n), let obj = try? JSONSerialization.jsonObject(with: h) as? [String: Any] else {
            close(fd); throw PrefixMismatch("\(path): bad header")
        }
        self.fd = fd
        metadata = (obj["__metadata__"] as? [String: String]) ?? [:]
        var es: [String: (String, [Int], Int, Int)] = [:]
        for (k, v) in obj where k != "__metadata__" {
            guard let d = v as? [String: Any], let dt = d["dtype"] as? String, let sh = d["shape"] as? [Int], let o = d["data_offsets"] as? [Int]
            else { continue }
            es[k] = (dt, sh, 8 + Int(n) + o[0], 8 + Int(n) + o[1])
        }
        entries = es
        var st = stat()
        fstat(fd, &st)
        size = Int(st.st_size)
    }

    deinit { close(fd) }

    func has(_ name: String) -> Bool { entries[name] != nil }

    func data(_ name: String) throws -> Data {
        guard let e = entries[name] else { throw PrefixMismatch("\((path as NSString).lastPathComponent): no tensor \(name)") }
        var d = Data(count: e.end - e.begin)
        let got = d.withUnsafeMutableBytes { pread(fd, $0.baseAddress, e.end - e.begin, off_t(e.begin)) }
        guard got == e.end - e.begin else { throw PrefixMismatch("\(path): short read of \(name)") }
        return d
    }

    func array(_ name: String) throws -> MLXArray {
        let e = entries[name]!
        return MLXArray(try data(name), e.shape, dtype: try mlxDType(e.dtype))
    }

    func i64(_ name: String) throws -> [Int] {
        guard entries[name]?.dtype == "I64" else { throw PrefixMismatch("\(name): not I64") }
        return try data(name).withUnsafeBytes { Array($0.bindMemory(to: Int64.self)).map { Int(Int64(littleEndian: $0)) } }
    }

    func i32(_ name: String) throws -> [Int] {
        guard entries[name]?.dtype == "I32" else { throw PrefixMismatch("\(name): not I32") }
        return try data(name).withUnsafeBytes { Array($0.bindMemory(to: Int32.self)).map { Int(Int32(littleEndian: $0)) } }
    }

    func f64(_ name: String) throws -> [Double] {
        guard entries[name]?.dtype == "F64" else { throw PrefixMismatch("\(name): not F64") }
        return try data(name).withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
    }
}

// MARK: - the store's host state

struct StoreState {
    var slotOf: [[Int]], lastUse: [[Int]], insSeq: [[Int]]
    var seq: Int
    var free: [[Int]]
    var namedAt: [Int]
    var caps: [Int]
    struct Pol { var last, tot, dec, hzU, hzN: [Double]; var B: [Int]?; var t: Int?; var cur: [Bool] }
    var pol: Pol?

    static let noT = -(1 << 62)

    func entries() -> [STEntry] {
        let NL = slotOf.count
        var es = [
            hostEntry("s.slot_of", i64: slotOf.flatMap { $0 }, shape: [NL, slotOf[0].count]),
            hostEntry("s.last_use", i64: lastUse.flatMap { $0 }, shape: [NL, lastUse[0].count]),
            hostEntry("s.ins_seq", i64: insSeq.flatMap { $0 }, shape: [NL, insSeq[0].count]),
            hostEntry("s.seq", i64: [seq]), hostEntry("s.cap", i64: [caps.allSatisfy { $0 == caps[0] } ? caps[0] : -1]),
            hostEntry("s.caps", i64: caps), hostEntry("s.named_at", i64: namedAt),
            hostEntry("s.free", i64: free.flatMap { $0 }), hostEntry("s.free_n", i64: free.map(\.count)),
        ]
        if let p = pol {
            es += [hostEntry("p.last", f64: p.last), hostEntry("p.tot", f64: p.tot), hostEntry("p.dec", f64: p.dec),
                   hostEntry("p.hz_u", f64: p.hzU), hostEntry("p.hz_n", f64: p.hzN), hostEntry("p.t", i64: [p.t ?? Self.noT]),
                   hostEntry("p.cur", i64: p.cur.map { $0 ? 1 : 0 })]
            if let B = p.B { es.append(hostEntry("p.B", i64: B)) }
        }
        return es
    }

    init(slotOf: [[Int]], lastUse: [[Int]], insSeq: [[Int]], seq: Int, free: [[Int]], namedAt: [Int], caps: [Int], pol: Pol?) {
        self.slotOf = slotOf; self.lastUse = lastUse; self.insSeq = insSeq; self.seq = seq; self.free = free
        self.namedAt = namedAt; self.caps = caps; self.pol = pol
    }

    init(_ r: STReader, NL: Int, NE: Int) throws {
        func rows(_ v: [Int]) -> [[Int]] { (0..<NL).map { Array(v[($0 * NE)..<(($0 + 1) * NE)]) } }
        slotOf = rows(try r.i64("s.slot_of")); lastUse = rows(try r.i64("s.last_use")); insSeq = rows(try r.i64("s.ins_seq"))
        seq = try r.i64("s.seq")[0]
        caps = r.has("s.caps") ? try r.i64("s.caps") : Array(repeating: try r.i64("s.cap")[0], count: NL)  // pre-2026-11-04 files: one cap
        namedAt = try r.i64("s.named_at")
        let fl = try r.i64("s.free"), fn = try r.i64("s.free_n")
        var i = 0
        free = fn.map { n in defer { i += n }; return Array(fl[i..<(i + n)]) }
        if r.has("p.last") {
            let t = try r.i64("p.t")[0]
            pol = Pol(last: try r.f64("p.last"), tot: try r.f64("p.tot"), dec: try r.f64("p.dec"), hzU: try r.f64("p.hz_u"),
                      hzN: try r.f64("p.hz_n"), B: r.has("p.B") ? try r.i64("p.B") : nil, t: t == Self.noT ? nil : t,
                      cur: try r.i64("p.cur").map { $0 != 0 })
        } else {
            pol = nil
        }
    }
}

extension Store {
    /// The cache state at a token boundary. Waits for reads still in flight first (prefetch mode has none at a token
    /// boundary; exact mode with routers can have named-but-unused ones), as mlx_run.py settle_reads does.
    func captureState() -> StoreState {
        mon.lock()
        defer { mon.unlock() }
        while nInflight.contains(where: { $0 > 0 }) { mon.wait() }
        var p: StoreState.Pol? = nil
        if let pol {
            p = .init(last: pol.last, tot: pol.tot, dec: pol.dec, hzU: pol.hzU, hzN: pol.hzN, B: pol.B, t: pol.t, cur: pol.cur)
        }
        return StoreState(slotOf: slotOf, lastUse: lastUse, insSeq: insSeq, seq: seq, free: free, namedAt: namedAt, caps: caps, pol: p)
    }

    /// Copy record bytes between a slab slot and a host buffer laid out as the record on disk.
    func copyRecord(_ L: Int, slot: Int, toHost dst: UnsafeMutableRawPointer) {
        for (i, pc) in meta.pieces.enumerated() { memcpy(dst + pc.off, piecePtr[L][i] + slot * pc.n, pc.n) }
    }

    func copyRecord(_ L: Int, fromHost src: UnsafeRawPointer, slot: Int) {
        for (i, pc) in meta.pieces.enumerated() { memcpy(piecePtr[L][i] + slot * pc.n, src + pc.off, pc.n) }
    }

    /// Put the saved cache state back (mlx_run.py restore_slots + the policy restore). With `reland`, every held expert
    /// ends up landed in its saved slot: kept when the slot already holds its record, copied from another slot of the
    /// same layer when the slab has it elsewhere, else read from the store (all layers' reads in flight at once, waited
    /// for here). Without `reland` no expert is held (prefetch mode then computes differently: the slots start empty).
    /// With `lazy` (and reland), the held set, last uses, insertion order, free lists and the policy are restored exactly
    /// as with reland, but nothing is moved or read: a held expert whose record is already in its slot is landed, the
    /// others are pending (Store.pending: held, not landed). The next batched prefill chunk plans them like any absent
    /// expert (it reads those it uses or keeps, nothing else), and anything else lands them first (landPending: the same
    /// keep / move / read as reland). The computation is the same either way: a chunk's arithmetic and its policy replay
    /// do not depend on which records are in the slab, only on the held set (docs/G4-runtime.md "Time to first token").
    /// Call only at a token boundary with no GPU work in flight (the GPU never reads a slab during this).
    func restoreState(_ s: StoreState, reland: Bool, lazy: Bool = false)
        throws -> (relanded: Int, kept: Int, moved: Int, bytes: Int, held: Int, pending: Int) {
        guard s.caps == caps, s.slotOf.count == NL, s.slotOf[0].count == NE else {
            throw PrefixMismatch("saved with slots per layer \(s.caps) x \(s.slotOf.count) layers, this store has \(caps) x \(NL)")
        }
        mon.lock()
        while nInflight.contains(where: { $0 > 0 }) { mon.wait() }
        var kept = 0, pend = 0
        for L in 0..<NL {
            pending[L] = []
            for e in 0..<NE {
                let sl = reland ? s.slotOf[L][e] : -1
                slotOf[L][e] = sl
                lastUse[L][e] = reland ? s.lastUse[L][e] : -1
                insSeq[L][e] = s.insSeq[L][e]
                chunksLeft[L][e] = 0
                inflight[L][e] = false
                if sl >= 0 && slotHas[L][sl] == e {
                    tab[L][e] = Int32(sl)
                    kept += 1
                } else {
                    tab[L][e] = -1
                    if sl >= 0 { pending[L].append((e, sl)); pend += 1 }
                }
            }
            nInflight[L] = 0
            free[L] = reland ? s.free[L] : Array(0..<caps[L])
            namedAt[L] = s.namedAt[L]
            tabVersion[L] += 1
        }
        hasPending = pend > 0
        seq = s.seq
        if let pol {
            pol.reset()
            if let p = s.pol {
                pol.last = p.last; pol.tot = p.tot; pol.dec = p.dec; pol.hzU = p.hzU; pol.hzN = p.hzN
                pol.B = p.B; pol.t = p.t; pol.cur = p.cur
            }
        }
        let held = slotOf.reduce(0) { $0 + $1.reduce(0) { $0 + ($1 >= 0 ? 1 : 0) } }
        mon.unlock()
        if lazy || !reland { return (0, kept, 0, 0, held, pend) }
        let r = landPending()
        return (r.relanded, kept + r.kept, r.moved, r.bytes, held, 0)
    }

    /// Land every pending record (restoreState lazy, read-ahead overwrites): kept when its slot already holds it, copied
    /// from another slot of the layer that holds it (memcpy, no drive), else read (all layers in flight at once, waited
    /// for). Engine thread, at a token boundary with no GPU work in flight. Returns what it did.
    @discardableResult
    public func landPending() -> (relanded: Int, kept: Int, moved: Int, bytes: Int) {
        mon.lock()
        defer { mon.unlock() }
        guard hasPending else { return (0, 0, 0, 0) }
        while nInflight.contains(where: { $0 > 0 }) { mon.wait() }
        let b0 = bytes
        var toRead: [(L: Int, e: Int, s: Int)] = []
        var kept = 0, moved = 0
        for L in 0..<NL where !pending[L].isEmpty {
            var whereIs: [Int: Int] = [:]  // expert -> a slot whose bytes are its record
            for sl in 0..<caps[L] where slotHas[L][sl] >= 0 { whereIs[slotHas[L][sl]] = sl }
            var moves: [(e: Int, src: Int, dst: Int)] = []
            for (e, dst) in pending[L] where slotOf[L][e] == dst {
                if slotHas[L][dst] == e {
                    kept += 1
                    tab[L][e] = Int32(dst)
                } else if let src = whereIs[e], !opt.policyOnly {
                    moves.append((e, src, dst))
                } else {
                    toRead.append((L, e, dst))
                }
            }
            if !moves.isEmpty {  // sources out first (a destination can be another move's source), then into place
                let tmp = UnsafeMutableRawPointer.allocate(byteCount: moves.count * REC, alignment: 16384)
                for (i, m) in moves.enumerated() { copyRecord(L, slot: m.src, toHost: tmp + i * REC) }
                for (i, m) in moves.enumerated() { copyRecord(L, fromHost: tmp + i * REC, slot: m.dst) }
                tmp.deallocate()
                for m in moves {  // a source keeps its copy unless it is another move's destination (set here too)
                    slotHas[L][m.dst] = m.e
                    tab[L][m.e] = Int32(m.dst)
                }
                moved += moves.count
            }
            pending[L] = []
            tabVersion[L] += 1
        }
        hasPending = false
        for r in toRead {
            if opt.policyOnly {  // tests: no slabs, an admitted record lands at once
                tab[r.L][r.e] = Int32(r.s)
                bytes += REC
                continue
            }
            inflight[r.L][r.e] = true
            nInflight[r.L] += 1
            submit(r.L, r.e, r.s)
        }
        while nInflight.contains(where: { $0 > 0 }) { mon.wait() }
        return (toRead.count, kept, moved, bytes - b0)
    }
}

// MARK: - the engine's state

extension Engine {
    /// What a state is valid for (a state saved under another configuration is refused).
    public var prefixConfig: [String: String] {
        let c = res.cfg
        return [
            "store": opt.store, "record_bytes": "\(store.REC)", "layers": "\(NL)", "experts": "\(NE)", "hidden": "\(H)",
            "vocab": "\(c.vocab)", "mode": opt.mode.rawValue, "routers": routers == nil ? "none" : "lag-\(lag)",
            "router_bits": routers == nil ? "0" : "\(opt.routerBits)", "m": "\(M)", "resident_bits": "\(opt.residentBits)",
            "keep": "\(opt.keep)", "evict": opt.evict.rawValue, "no_drop": "\(opt.noDrop)",
            "cap": store.capsUniform ? "\(cap)" : store.caps.map(String.init).joined(separator: ","),
        ]
    }

    /// Write the state after the fed tokens `ids` (whose next-token logits are `logits`) to path. Call at a token
    /// boundary on the engine thread. Nothing it does changes the computation that follows.
    public func savePrefix(path: String, ids: [Int], logits: MLXArray) throws -> PrefixSaveInfo {
        let t0 = now()
        precondition(ids.count == t + 1, "prefix state: \(ids.count) ids fed but the token clock is \(t)")
        let host = store.captureState()
        var dev: [(String, MLXArray)] = [("logits", logits)]
        for L in 0..<NL { dev.append(("h_prev.\(L)", hPrev[L])) }
        var kinds: [[Any]] = []
        for L in 0..<NL {
            if let c = kv[L] {
                let capn = c.keys?.dim(2) ?? 0
                kinds.append(["kv", c.offset, capn])
                if c.offset > 0 {
                    dev.append(("kv.\(L).k", c.keys![.ellipsis, ..<c.offset, 0...]))
                    dev.append(("kv.\(L).v", c.values![.ellipsis, ..<c.offset, 0...]))
                }
            } else if let g = gdnc[L] {
                kinds.append(["gdn", g.conv != nil, g.state != nil])
                if let x = g.conv { dev.append(("gdn.\(L).conv", x)) }
                if let x = g.state { dev.append(("gdn.\(L).state", x)) }
            } else {
                kinds.append(["none"])
            }
        }
        eval(dev.map { $0.1 })
        let entries = [hostEntry("ids", i32: ids)] + dev.map { deviceEntry($0.0, $0.1) } + host.entries()
        let cfgJSON = String(data: try JSONSerialization.data(withJSONObject: prefixConfig, options: [.sortedKeys]), encoding: .utf8)!
        let cacheJSON = String(data: try JSONSerialization.data(withJSONObject: kinds), encoding: .utf8)!
        let meta = ["format": prefixFormat, "n": "\(ids.count)", "t": "\(t)", "key": prefixKey(ids), "cache": cacheJSON, "config": cfgJSON]
        let n = try writeSafetensors(path: path, entries: entries, metadata: meta)
        return PrefixSaveInfo(path: path, n: ids.count, bytes: n, seconds: now() - t0)
    }

    /// The saved ids, after checking the state belongs to this engine and (with ids) is a prefix of them.
    public func peekPrefix(path: String, ids: [Int]? = nil) throws -> [Int] {
        let r = try STReader(path: path)
        return try peek(r, ids: ids)
    }

    func peek(_ r: STReader, ids: [Int]?) throws -> [Int] {
        let name = (r.path as NSString).lastPathComponent
        guard r.metadata["format"] == prefixFormat else { throw PrefixMismatch("\(name): not a native prefix state (\(r.metadata["format"] ?? "no format"))") }
        guard let cj = r.metadata["config"]?.data(using: .utf8), let saved = try? JSONSerialization.jsonObject(with: cj) as? [String: String] else {
            throw PrefixMismatch("\(name): no configuration")
        }
        let now = prefixConfig
        if saved != now {
            let diff = Set(saved.keys).union(now.keys).filter { saved[$0] != now[$0] }.sorted().map { "\($0): \(saved[$0] ?? "-") -> \(now[$0] ?? "-")" }
            throw PrefixMismatch("\(name): saved by another engine configuration (\(diff.joined(separator: ", ")))")
        }
        let s = try r.i32("ids")
        if let ids, ids.count < s.count || Array(ids[..<s.count]) != s {
            throw PrefixMismatch("\(name): its \(s.count) tokens are not a prefix of the prompt")
        }
        return s
    }

    /// Restore the state saved at path: the model state (evaluated), then the store (re-land). Returns the saved ids and
    /// the next-token logits after them. Throws PrefixMismatch before touching any state when the file is not for this
    /// engine or not a prefix of ids. Call on the engine thread with no GPU work in flight.
    /// `lazy` (Store.restoreState): the held set is restored without reading or moving anything; the next chunk reads what
    /// it needs, and step() lands the rest first (Engine.step: store.landPending).
    public func loadPrefix(path: String, ids: [Int]?, reland: Bool = true, lazy: Bool = false) throws -> (ids: [Int], logits: MLXArray, info: PrefixLoadInfo) {
        let t0 = now()
        let r = try STReader(path: path)
        let saved = try peek(r, ids: ids)
        guard let cj = r.metadata["cache"]?.data(using: .utf8), let kinds = try JSONSerialization.jsonObject(with: cj) as? [[Any]],
              kinds.count == NL, let tt = Int(r.metadata["t"] ?? "") else { throw PrefixMismatch("\(path): bad cache layout") }
        let host = try StoreState(r, NL: NL, NE: NE)
        // the model state (built fully before anything is replaced)
        var nkv = [KVCache?](repeating: nil, count: NL), ng = [GDNCache?](repeating: nil, count: NL)
        var all: [MLXArray] = []
        for L in 0..<NL {
            let k = kinds[L]
            switch k[0] as? String {
            case "kv":
                let off = (k[1] as! NSNumber).intValue, capn = (k[2] as! NSNumber).intValue
                let c = KVCache()
                if off > 0 {
                    var kk = try r.array("kv.\(L).k"), vv = try r.array("kv.\(L).v")
                    if capn > off {  // the same buffer capacity as the run that saved it (zeros beyond the offset, as KVCache allocates)
                        var zs = kk.shape
                        zs[2] = capn - off
                        kk = concatenated([kk, MLXArray.zeros(zs, dtype: kk.dtype)], axis: 2)
                        vv = concatenated([vv, MLXArray.zeros(zs, dtype: vv.dtype)], axis: 2)
                    }
                    c.keys = kk
                    c.values = vv
                    all += [kk, vv]
                }
                c.offset = off
                nkv[L] = c
            case "gdn":
                let c = GDNCache()
                if (k[1] as! NSNumber).boolValue { c.conv = try r.array("gdn.\(L).conv"); all.append(c.conv!) }
                if (k[2] as! NSNumber).boolValue { c.state = try r.array("gdn.\(L).state"); all.append(c.state!) }
                ng[L] = c
            default: break
            }
            guard (nkv[L] != nil) == (kv[L] != nil), (ng[L] != nil) == (gdnc[L] != nil) else {
                throw PrefixMismatch("\(path): layer \(L) has another cache kind")
            }
        }
        let hp = try (0..<NL).map { try r.array("h_prev.\($0)") }
        let lg = try r.array("logits")
        eval(all + hp + [lg])
        kv = nkv
        gdnc = ng
        hPrev = hp
        t = tt
        tabDev = Array(repeating: nil, count: NL)
        nextArg = nil
        let t1 = now()
        let rl = try store.restoreState(host, reland: reland, lazy: lazy)
        let t2 = now()
        var info = PrefixLoadInfo(path: path, n: saved.count, fileBytes: r.size, loadSeconds: t1 - t0, relandSeconds: t2 - t1,
                                  relanded: rl.relanded, kept: rl.kept, moved: rl.moved, relandBytes: rl.bytes, held: rl.held)
        info.deferred = rl.pending
        info.lazy = lazy && reland
        return (saved, lg, info)
    }
}

// MARK: - a directory of states keyed by the SHA-256 of the token-id prefix

/// `n<len>-<sha256 of the ids>.safetensors` files; a lookup finds the longest cached prefix of a prompt's ids. Least
/// recently used entries (file modification time; a hit touches its file) are deleted above `capBytes` (0: never), except
/// the `pinned` ones (the session pins the system-prompt state of the conversation in progress), the `kept` ones (the
/// app's one-time setup: its system-prompt states, listed in pinned.txt) and the newest.
public final class PrefixCacheDir: @unchecked Sendable {
    public let dir: String
    public var capBytes: Int
    /// Paths the LRU never deletes (set by the session on the engine thread: the current conversation's system-prompt
    /// state). They still count toward the cap.
    public var pinned: Set<String> {
        get { pinLock.withLock { _pinned } }
        set { pinLock.withLock { _pinned = newValue } }
    }
    private var _pinned: Set<String> = []
    private let pinLock = NSLock()

    /// States the LRU never deletes across launches (2026-11-07, docs/G4-runtime.md "Launch: once per install"): the app's
    /// one-time setup keeps its system-prompt states here (tools off and on), in `pinned.txt` beside them (file names, one
    /// per line), so they survive any number of chats and relaunches. They still count toward the cap.
    public var kept: Set<String> {
        get { pinLock.withLock { _kept } }
        set {
            pinLock.withLock { _kept = newValue }
            let names = newValue.map { ($0 as NSString).lastPathComponent }.sorted().joined(separator: "\n")
            try? (names.isEmpty ? "" : names + "\n").write(toFile: keptFile, atomically: true, encoding: .utf8)
        }
    }
    private var _kept: Set<String> = []
    var keptFile: String { dir + "/pinned.txt" }

    public init(dir: String, capGB: Double) throws {
        self.dir = dir
        capBytes = Int(capGB * 1e9)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        if let t = try? String(contentsOfFile: dir + "/pinned.txt", encoding: .utf8) {
            _kept = Set(t.split(separator: "\n").map { dir + "/" + $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.hasSuffix("/") })
        }
    }

    public func path<C: Collection>(for ids: C) -> String where C.Element == Int {
        dir + "/" + String(format: "n%07d-", ids.count) + prefixKey(ids) + ".safetensors"
    }

    public func entries() -> [(path: String, n: Int, bytes: Int, mtime: Date)] {
        let fm = FileManager.default
        let names = (try? fm.contentsOfDirectory(atPath: dir)) ?? []
        return names.compactMap { nm in
            guard nm.hasPrefix("n"), nm.hasSuffix(".safetensors"), let dash = nm.firstIndex(of: "-"),
                  let n = Int(nm[nm.index(after: nm.startIndex)..<dash]) else { return nil }
            let p = dir + "/" + nm
            guard let a = try? fm.attributesOfItem(atPath: p) else { return nil }
            return (p, n, (a[.size] as? NSNumber)?.intValue ?? 0, (a[.modificationDate] as? Date) ?? .distantPast)
        }
    }

    public var usage: (entries: Int, bytes: Int) {
        let e = entries()
        return (e.count, e.reduce(0) { $0 + $1.bytes })
    }

    /// The longest cached prefix of ids (at most `maxLen` tokens): its length and file.
    public func longest(_ ids: [Int], maxLen: Int? = nil) -> (n: Int, path: String)? {
        let lens = Set(entries().map(\.n)).filter { $0 > 0 && $0 <= min(ids.count, maxLen ?? ids.count) }.sorted(by: >)
        for n in lens {
            let p = path(for: ids[..<n])
            if FileManager.default.fileExists(atPath: p) { return (n, p) }
        }
        return nil
    }

    public func touch(_ path: String) {
        try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: path)
    }

    /// Delete the least recently used entries until the directory is under the cap (the newest entry and the pinned ones
    /// are always kept).
    @discardableResult
    public func enforceCap() -> [String] {
        guard capBytes > 0 else { return [] }
        let keep = pinned.union(kept)
        let all = entries().sorted { $0.mtime < $1.mtime }
        var used = all.reduce(0) { $0 + $1.bytes }
        var es = Array(all.dropLast()).filter { !keep.contains($0.path) }  // candidates, oldest first
        var gone: [String] = []
        while used > capBytes && !es.isEmpty {
            let e = es.removeFirst()
            try? FileManager.default.removeItem(atPath: e.path)
            used -= e.bytes
            gone.append(e.path)
        }
        return gone
    }

    /// Delete every state (the kept ones too, and the list is emptied; `keepKept`: all but the kept ones).
    public func clear(keepKept: Bool = false) {
        let k = keepKept ? kept : []
        for e in entries() where !k.contains(e.path) { try? FileManager.default.removeItem(atPath: e.path) }
        if !keepKept { kept = [] }
    }
}
