// The expert store on flash: per-layer slabs of slots in GPU-visible memory, the reader threads that read records straight
// into them (preadv, F_NOCACHE, readahead off), the slot table, and the cache policy (mlx_run.py take_slot / Evict).
//
// Ownership and ordering (the invariants mlx_run.py relies on, kept here):
// - All cache state (slotOf, slotBy, last, inflight, tab, free, the policy) is guarded by `mon`.
// - A record is read into a slot only after the slot left every held expert's entry; a victim is never named or used for
//   the current token (last < t) and never in flight. The GPU uses layer L's slots only in layer L's expert step of the
//   current token, which is encoded after layer L's names were applied and its reads landed, and every previous token's GPU
//   work is complete (the sampler's read-back drains the queue). So no slot is overwritten while the GPU can read it.
// - tab[L][e] (expert -> slot once landed) is set by the reader when the record is in memory. With --read-chunks N a
//   record is read as N byte ranges (16 KB aligned) by up to N readers; it is landed when the last of them is in.
// - Late wait (Engine, prefetch): the encode thread waits for layer L's names (layerNamed) before it encodes the masked
//   top-k, and for layer L's reads (layerLanded) only right before it encodes layer L's routed expert step, the only op
//   that reads the slab. The table it uploads at layerNamed is slotOf, which equals tab once nInflight[L] == 0 (every held
//   expert not in flight is landed), so the GPU's choice is the same as with the landed table.
import Foundation
import MLX
import Cmlx

public struct StoreMeta {
    public let hidden: Int, inter: Int, layers: Int, experts: Int, topK: Int, recordBytes: Int
    public let dtype: String, bits: Int, group: Int
    /// record pieces in file order: (offset, matrix 0/1/2, kind 0 codes / 1 scales / 2 biases, bytes)
    public let pieces: [(off: Int, m: Int, kind: Int, n: Int)]

    public init(path: String) throws {
        let d = try Data(contentsOf: URL(fileURLWithPath: path))
        let j = try JSONSerialization.jsonObject(with: d) as! [String: Any]
        func i(_ k: String) -> Int { (j[k] as! NSNumber).intValue }
        hidden = i("hidden"); inter = i("intermediate"); layers = i("layers"); experts = i("experts")
        topK = (j["top_k"] as? NSNumber)?.intValue ?? 8
        recordBytes = i("record_bytes")
        dtype = (j["dtype"] as? String) ?? "int8"
        bits = (j["bits"] as? NSNumber)?.intValue ?? 8
        group = (j["group"] as? NSNumber)?.intValue ?? 64
        guard dtype == "int4", bits == 4, group == 64 else { throw PSError("native runtime reads int4 stores (mx.quantize bits 4 group 64); \(path) is \(dtype)") }
        let names = ["gate_proj", "up_proj", "down_proj"]
        let offs = j["offsets"] as! [String: NSNumber], so = j["scale_offsets"] as! [String: NSNumber], bo = j["bias_offsets"] as! [String: NSNumber]
        let mat = hidden * inter
        var p: [(Int, Int, Int, Int)] = []
        for (m, nm) in names.enumerated() {
            p.append((offs[nm]!.intValue, m, 0, mat / 2))
            p.append((so[nm]!.intValue, m, 1, mat / G * 2))
            p.append((bo[nm]!.intValue, m, 2, mat / G * 2))
        }
        pieces = p.sorted { $0.0 < $1.0 }.map { (off: $0.0, m: $0.1, kind: $0.2, n: $0.3) }
        var pos = 0
        for pc in pieces {
            guard pc.off == pos else { throw PSError("record pieces are not contiguous at \(pc.off)") }
            pos += pc.n
        }
        guard pos == recordBytes else { throw PSError("record pieces cover \(pos) of \(recordBytes) bytes") }
    }
}

public enum EvictKind: String { case window, lru, hazard, prior }

/// mlx_run.py's Evict (docs/evict_runs/evict_sim.py hazard_h128 / online_prior_256): per-layer history of every expert on
/// the token clock; the victim is the held expert with the smallest key. Same buckets, decay, table prior, tie-breaks.
///
/// The clock is kept per layer (2026-11-06, batched prefill): `tick(L, t)` advances layer L only. mlx_run.py's tick
/// advances every layer at the token's first touch; the per-layer state (last, tot, dec, the hazard table, B, cur) of one
/// layer never reads another layer's, and every layer is ticked at every token in both modes (prefetch names all 40,
/// exact prepares all 40), so the values are identical when tokens are fed one by one. A batched prefill chunk runs the
/// chunk's tokens through layer L before layer L + 1 sees any of them (layer-major); with a per-layer clock that is the
/// same history as feeding them in order.
public final class Evict {
    let kind: EvictKind
    let NL: Int, NE: Int
    static let NA = 13, NF = 10
    let fac = pow(2.0, -1.0 / 128.0)
    let w = 256.0
    var last: [Double], tot: [Double], dec: [Double]
    var hzU: [Double], hzN: [Double]
    var B: [Int]? = nil
    var bOK: [Bool]  // per layer: B holds this layer's buckets (it was ticked before)
    var tl: [Int?]  // per layer: the clock of its last tick
    /// The clock (prefix state): every layer's, equal at a token boundary. Setting it sets every layer's (and B's validity).
    var t: Int? {
        get { tl[0] }
        set { for L in 0..<NL { tl[L] = newValue; bOK[L] = newValue != nil && B != nil } }
    }
    var cur: [Bool]
    var keys: [(Int, [Double])?]
    var kb: [(Int, [Int])?]
    public var secCritical = 0.0  // touch + keys + victims
    public var secPrepare = 0.0

    init(kind: EvictKind, layers: Int, experts: Int) {
        self.kind = kind; NL = layers; NE = experts
        last = []; tot = []; dec = []; hzU = []; hzN = []; cur = []; keys = []; kb = []; tl = []; bOK = []
        reset()
    }

    func reset() {
        last = [Double](repeating: -1, count: NL * NE)
        tot = [Double](repeating: 0, count: NL * NE)
        dec = [Double](repeating: 0, count: NL * NE)
        hzU = [Double](repeating: 1, count: NL * Evict.NA * Evict.NF)
        hzN = [Double](repeating: 2, count: NL * Evict.NA * Evict.NF)
        B = nil
        tl = [Int?](repeating: nil, count: NL)
        bOK = [Bool](repeating: false, count: NL)
        cur = [Bool](repeating: false, count: NL * NE)
        keys = [(Int, [Double])?](repeating: nil, count: NL)
        kb = [(Int, [Int])?](repeating: nil, count: NL)
    }

    @inline(__always) func bin(_ lastV: Double, _ decV: Double, _ t: Int) -> Int {
        let ab: Int
        if lastV >= 0 {
            let age = t - Int(lastV)  // >= 1
            ab = min(Int.bitWidth - 1 - age.leadingZeroBitCount, Evict.NA - 1)  // floor(log2(age)), exact
        } else {
            ab = min(14, Evict.NA - 1)  // age 1 << 14
        }
        let fb = min(Int(log2(1 + 4 * decV)), Evict.NF - 1)
        return ab * Evict.NF + fb
    }

    /// Layer L's clock to t (mlx_run.py Evict.tick, one layer): its hazard buckets counted at t, its decay, cur, keys.
    func tick(_ L: Int, _ t: Int) {
        if tl[L] == t { return }
        let base = L * NE
        if kind == .hazard {
            if let tp = tl[L] {
                precondition(t > tp, "policy clock of layer \(L) going back: \(tp) -> \(t)")
                if B == nil { B = [Int](repeating: 0, count: NL * NE) }
                let nb = Evict.NA * Evict.NF
                for e in 0..<NE {
                    let i = base + e
                    let v = bin(last[i], dec[i], t)
                    B![i] = v
                    hzN[L * nb + v] += 1
                }
                bOK[L] = true
            }
            let g = tl[L] == nil ? 1 : t - tl[L]!
            let f = g == 1 ? fac : pow(fac, Double(g))
            for i in base..<(base + NE) { dec[i] *= f }
        }
        for i in base..<(base + NE) { cur[i] = false }
        tl[L] = t
        keys[L] = nil
    }

    func prepare(_ L: Int, _ t: Int) {
        let t0 = now()
        tick(L, t)
        if kind == .hazard {
            var k = [Int](repeating: 0, count: NE)
            for e in 0..<NE { k[e] = bin(last[L * NE + e], dec[L * NE + e], t + 1) }
            kb[L] = (t, k)
        }
        secPrepare += now() - t0
    }

    func touch(_ L: Int, _ t: Int, _ experts: [Int]) {
        let t0 = now()
        tick(L, t)
        var changed = false
        let nb = Evict.NA * Evict.NF
        for e in Set(experts) {
            let i = L * NE + e
            if cur[i] { continue }
            cur[i] = true
            last[i] = Double(t)
            tot[i] += 1
            if kind == .hazard {
                dec[i] += 1
                if bOK[L], let B { hzU[L * nb + B[i]] += 1 }
            }
            changed = true
        }
        if changed { keys[L] = nil }
        secCritical += now() - t0
    }

    func key(_ L: Int, _ t: Int) -> [Double] {
        if let k = keys[L], k.0 == t { return k.1 }
        var k = [Double](repeating: 0, count: NE)
        let base = L * NE
        if kind == .hazard {
            let nb = Evict.NA * Evict.NF
            let kbL: [Int]
            if let p = kb[L], p.0 == t { kbL = p.1 } else { kbL = (0..<NE).map { bin(last[base + $0], dec[base + $0], t + 1) } }
            for e in 0..<NE { k[e] = hzU[L * nb + kbL[e]] / hzN[L * nb + kbL[e]] + last[base + e] * 1e-9 }
        } else {
            // r = argsort(argsort(tot[L], stable), stable): the rank of each expert by count, ties by id
            let order = (0..<NE).sorted { (tot[base + $0], $0) < (tot[base + $1], $1) }
            var r = [Int](repeating: 0, count: NE)
            for (rank, e) in order.enumerated() { r[e] = rank }
            for e in 0..<NE { k[e] = last[base + e] + w * (Double(r[e]) / Double(NE - 1)) }
        }
        keys[L] = (t, k)
        return k
    }

    func victim(_ L: Int, _ t: Int, _ cands: [Int]) -> Int {
        let t0 = now()
        let k = key(L, t)
        var best = -1
        var bv = Double.infinity
        for e in cands.sorted() where k[e] < bv {  // argmin, first (lowest id) on ties
            bv = k[e]
            best = e
        }
        secCritical += now() - t0
        return best
    }
}

public struct StoreOptions {
    public var cap: Int
    public var keep: Int
    public var evict: EvictKind
    public var noDrop: Bool
    public var readers: Int
    public var simReads: Bool
    public var noCache: Bool
    public var bwCap: Double  // GB/s, 0 off
    public var policyOnly = false  // tests: no slabs, no files, no readers; an admitted expert lands at once
    public var chunks = 1  // reads per record (byte ranges, 16 KB aligned), spread over the readers
    /// Per-layer slot caps (--layer-caps); nil: `cap` in every layer.
    public var layerCaps: [Int]? = nil
    /// --store-codec: auto reads a layer from layer_XX.psc/.idx (decoded on the GPU into the slot) when both exist.
    public var codec: StoreCodec = .off
    public var codecStaging = 24  // staging slots for compressed records (read or waiting to be decoded)
    public var codecCheck = false  // test: SHA-256 of every decoded slot against the index (CPU, slow)
    /// Batched prefill (Engine.prefill, --prefill-chunk): reader threads while a chunk's reads are issued (the extra
    /// threads beyond `readers` sleep otherwise) and reads per record then (0: `chunks`).
    public var prefillReaders = 0
    public var prefillReadChunks = 0
    public init(cap: Int, keep: Int, evict: EvictKind, noDrop: Bool, readers: Int, simReads: Bool, noCache: Bool, bwCap: Double, policyOnly: Bool = false,
                chunks: Int = 1, layerCaps: [Int]? = nil, codec: StoreCodec = .off, codecStaging: Int = 24, codecCheck: Bool = false) {
        self.cap = cap; self.keep = keep; self.evict = evict; self.noDrop = noDrop; self.readers = readers
        self.simReads = simReads; self.noCache = noCache; self.bwCap = bwCap; self.policyOnly = policyOnly; self.chunks = chunks
        self.layerCaps = layerCaps; self.codec = codec; self.codecStaging = codecStaging; self.codecCheck = codecCheck
    }
}

/// --bw-cap: a token bucket of depth zero on one clock (mlx_run.py Throttle).
final class Throttle {
    let rate: Double
    var freeAt = 0.0
    let lock = NSLock()
    init(gbps: Double) { rate = gbps * 1e9 }
    func reserve(_ n: Int) -> Double {
        lock.lock(); defer { lock.unlock() }
        freeAt = max(now(), freeAt) + Double(n) / rate
        return freeAt
    }
    func settle(_ due: Double) {
        while true {
            let d = due - now()
            if d <= 0 { return }
            usleep(UInt32(max(50, (d > 0.002 ? d - 0.0015 : min(d, 0.0002)) * 1e6)))
        }
    }
}

/// How the codec decode thread groups records into one command buffer (docs/G4-runtime.md "Codec read path").
public enum CodecBatch: String, Sendable {
    /// every record of a layer together: a record waits until no record of its layer is still being read (or waiting
    /// for staging), then all of that layer's read records (and those of every other layer in the same state) go in one
    /// command buffer. If no layer is complete while a record waits for a staging slot, everything read is decoded (forced)
    /// so the staging slots come back. Since 2026-11-07.
    case layer
    /// the 2026-11-04 behaviour: whatever has been read when the decode thread is free (1.15-1.2 records per batch on the Mac)
    case landed
}

/// How a raw record's byte range reaches its slot (docs/G4-runtime.md "Drive throughput on the phone").
public enum ReadPath: String, Sendable {
    /// one `preadv` with an iovec per record piece: the 9 pieces (3 x codes 512 KB, 6 x scales/biases 32 KB) straight into
    /// their slab arrays. The kernel builds one UPL, and issues at least one device I/O, per iovec.
    case iovec
    /// one `pread` of the contiguous range into the reader's own page-aligned buffer, then a memcpy of each piece into its
    /// slab array: one device I/O per range (a whole record at --read-chunks 1), 1.77 MB of CPU copy per record.
    case staged
    /// DriveTest only: the range into one contiguous buffer (no slab, nothing landed in the arrays).
    case flat
}

/// Process-wide read settings, read when a Store is created (the CLI's --read-path / --reader-qos, the app's -readPath /
/// -readerQoS launch arguments). Defaults: the engine as measured (DESIGN.md, "Drive throughput on the phone").
public enum StoreTuning {
    nonisolated(unsafe) public static var readPath: ReadPath = .iovec
    /// Compressed layers: how the decode thread forms a batch (CodecBatch; environment PS_CODEC_BATCH=layer|landed
    /// overrides it when a Store is created) and how many decode command buffers may be in flight (PS_CODEC_INFLIGHT=1|2).
    /// Default .layer since 2026-11-07 (docs/G4-runtime.md "Codec read path" > "Decode batching": on the Mac 4.6 / 3.9
    /// records per batch instead of 1.2 / 1.1 and 45-60% less decode GPU per token at caps 19 / 48, ids identical); two in
    /// flight measured no better, so 1.
    nonisolated(unsafe) public static var codecBatch: CodecBatch = .layer
    nonisolated(unsafe) public static var codecInflight = 1
    /// QoS of the reader threads (QOS_CLASS_USER_INTERACTIVE since 2026-11-02).
    nonisolated(unsafe) public static var readerQoS: qos_class_t = QOS_CLASS_USER_INTERACTIVE
    /// Batched prefill pass planner (Store.chunkPlan; docs/G4-runtime.md "Batched prefill", "Passes that keep the drive
    /// busy", 2026-11-07): 2 = every slot the chunk's remaining work does not need is a read target (held experts the
    /// chunk is done with are given up and read back when that costs less than the passes they cost), each pass's reads
    /// issued as soon as the GPU is done with their slots; 1 = the planner of commit 3ace2ed (A/B). Environment
    /// PS_CHUNK_PLAN=1|2 overrides it when a Store is created.
    nonisolated(unsafe) public static var chunkPlanVersion = 2
    /// v2 pass size: a pass takes up to ceil(frac x the slots it may write), leaving at least `ahead` of them to the next
    /// pass's reads, which are then in flight while it computes (PS_CHUNK_PASS_FRAC, PS_CHUNK_PASS_AHEAD).
    /// With the lagged fence (chunkPassLag 1) a pass of P reads needs P other slots for the next pass, so the default
    /// is a third of the slots: pass j + 2's reads are queued while pass j + 1's land (no drain between passes).
    nonisolated(unsafe) public static var chunkPassFrac = 0.34
    nonisolated(unsafe) public static var chunkPassAhead = 0
    /// v2 fence lag (PS_CHUNK_PASS_LAG): 1 = a slot pass f computed with is rewritten only after pass f + 1 has been
    /// submitted, and the engine then waits for pass f's GPU work (done by then: the GPU runs in order and the host has
    /// waited for pass f + 1's reads meanwhile), so the host never blocks on the pass it just submitted; 0 = right after
    /// pass f itself (the host waits for that pass's GPU work).
    nonisolated(unsafe) public static var chunkPassLag = 1
    /// v2: what one pass costs, in record reads, when the planner weighs giving up a held expert's slot (one read back)
    /// against the passes that slot saves (PS_CHUNK_PASS_COST).
    nonisolated(unsafe) public static var chunkPassCost = 1.0
    public static func qos(named s: String) -> qos_class_t? {
        switch s {
        case "ui", "userInteractive", "user-interactive": return QOS_CLASS_USER_INTERACTIVE
        case "init", "userInitiated", "user-initiated": return QOS_CLASS_USER_INITIATED
        case "default": return QOS_CLASS_DEFAULT
        case "utility": return QOS_CLASS_UTILITY
        case "background": return QOS_CLASS_BACKGROUND
        default: return nil
        }
    }
    public static func qosName(_ q: qos_class_t) -> String {
        switch q {
        case QOS_CLASS_USER_INTERACTIVE: return "ui"
        case QOS_CLASS_USER_INITIATED: return "init"
        case QOS_CLASS_DEFAULT: return "default"
        case QOS_CLASS_UTILITY: return "utility"
        case QOS_CLASS_BACKGROUND: return "background"
        default: return "\(q.rawValue)"
        }
    }
}

public final class Store: @unchecked Sendable {
    public let meta: StoreMeta
    public let opt: StoreOptions
    public let NL: Int, NE: Int, K: Int, REC: Int
    /// Slots per layer (StoreOptions.layerCaps, else cap everywhere).
    public let caps: [Int]
    public var capsUniform: Bool { caps.allSatisfy { $0 == caps[0] } }
    let mon = Monitor()

    // slabs: per layer, per matrix m: codes [cap, rows, cols/8] uint32, scales and biases [cap, rows, cols/64] float16
    public private(set) var W: [[MLXArray]] = [], S: [[MLXArray]] = [], Bi: [[MLXArray]] = []
    var piecePtr: [[UnsafeMutableRawPointer]] = []  // per layer, per record piece (file order): the slot-0 address
    var fds: [Int32] = []
    public var slabBytes: Int { caps.reduce(0, +) * REC }

    // compressed layers (Codec.swift): per layer the index when the layer is read from .psc, the decoder, staging slots
    public private(set) var codecIdx: [CodecIndex?] = []
    var decoder: SlotDecoder? = nil
    var stgFree: [Int] = []  // free staging slots (guarded by mon)
    var stgWait: [(L: Int, e: Int, slot: Int)] = []  // records waiting for a staging slot, FIFO (guarded by mon)
    var stgWaitHead = 0
    var stgOf: [[Int]] = []  // per (layer, expert) in flight: its staging slot
    var decodePending: [(L: Int, e: Int, slot: Int, stg: Int)] = []  // read, not yet decoded (guarded by mon)
    /// Per layer: compressed records submitted whose bytes are not all in yet (queued, being read, or waiting for a staging
    /// slot); CodecBatch.layer decodes a layer only when this is 0 (guarded by mon).
    var codecReading: [Int] = []
    var decInflight = 0  // decode command buffers committed and not completed (guarded by mon)
    var descFree: [Int] = []  // free descriptor regions of the decoder (guarded by mon)
    public let codecBatchMode: CodecBatch
    public let codecMaxInflight: Int
    /// Batched prefill planner settings in force (StoreTuning.chunkPlanVersion etc., environment overrides at init).
    public let chunkPlanVersion: Int
    public let chunkPassFrac: Double
    public let chunkPassAhead: Int
    public let chunkPassLag: Int
    public let chunkPassCost: Double
    /// Per layer: "psc" (compressed, decoded on the GPU) or "bin" (raw records).
    public var layerPath: [String] { (0..<NL).map { codecIdx.indices.contains($0) && codecIdx[$0] != nil ? "psc" : "bin" } }
    public var codecStagingBytes: Int { decoder.map { $0.slots * $0.stride } ?? 0 }
    // codec stats (guarded by mon)
    public var codecRecords = 0  // records decoded
    public var codecBytesExact = 0  // their compressed bytes (the index's sizes)
    public var codecBatches = 0
    public var decodeGPUSec = 0.0  // GPU time of the decode command buffers
    public var decodeWallSec = 0.0  // commit -> completed, on the decode thread
    public var stgWaits = 0  // records that waited for a staging slot
    public var codecForcedBatches = 0  // CodecBatch.layer: batches decoded before their layer was complete (staging full)
    public var codecChecked = 0, codecCheckBad = 0
    public var readsByLayer: [Int] = []  // records landed per layer (raw or decoded)
    public var bytesByLayer: [Int] = []  // bytes read from the drive per layer

    // cache state (guarded by mon)
    var slotOf: [[Int]]  // expert -> slot (-1: not held)
    var lastUse: [[Int]]  // expert -> token clock it was last named/used (valid while held)
    var insSeq: [[Int]]  // expert -> insertion order into the held set (Python dict order, for lru/window ties)
    var seq = 0
    var inflight: [[Bool]]
    var nInflight: [Int]
    public var tab: [[Int32]]  // expert -> slot once landed (-1 otherwise)
    var tabVersion: [Int]
    var free: [[Int]]
    var namedAt: [Int]  // token clock of the layer's last naming (prefetch)
    let pol: Evict?
    /// What each slot physically holds (layer, slot) -> expert whose record is completely in it, -1 if none or being
    /// overwritten. Unlike slotOf it survives forgetSlots(): the prefix restore (PrefixCache.swift) keeps or moves records
    /// that are already in the slab instead of reading them again. Guarded by mon.
    var slotHas: [[Int]]
    /// Lazy re-land (2026-11-07, PrefixCache.swift restoreState(lazy:)): per layer, held experts whose record is not in
    /// their slot (expert, slot). They are held as far as the policy is concerned (slotOf, last use, insertion order) but
    /// not landed (tab -1). A batched prefill chunk plans them like any absent expert (Store.chunkPlan reads what the
    /// chunk uses or keeps); anything else (step(), a save) lands them first (landPending). Engine thread only.
    var pending: [[(e: Int, slot: Int)]] = []
    /// Some layer has pending records (Engine.step lands them before its first token). Engine thread only.
    public internal(set) var hasPending = false

    // read-ahead during a batched prefill chunk (2026-11-07, Engine.prefillChunk): low-priority whole-record reads of the
    // experts the lag-4 routers name for a later layer of the chunk, into that layer's slots, taken by a reader only when
    // no demand read is queued. They change nothing but what a slot physically holds (slotHas); the chunk's plan for that
    // layer then finds them present. Queue guarded by qmon; ahead / aheadInflight / aheadDone by mon.
    var lowQueue: [(L: Int, e: Int, slot: Int, tSub: Double)] = []
    var lowHead = 0
    var aheadInflight: [Int] = []  // per layer: read-ahead records queued or being read
    var aheadDone: [Set<Int>] = []  // per layer: experts landed by read-ahead since the layer's last plan
    var aheadPrev: [[Int: Int]] = []  // per layer: slot -> the record it held when its read-ahead was queued
    public var aheadIssued = 0, aheadLanded = 0, aheadCancelled = 0, aheadBytes = 0, aheadUsed = 0, aheadOverwroteHeld = 0

    // read queue: one job per chunk of a record (byte range [lo, hi) of the record)
    /// raw record: [lo, hi) of the record, read into the slot's pieces; compressed (stg >= 0): [lo, hi) of the .psc file,
    /// read into staging slot stg at byte dst
    var queue: [(L: Int, e: Int, slot: Int, lo: Int, hi: Int, tSub: Double, stg: Int, dst: Int)] = []
    var chunkRanges: [(lo: Int, hi: Int)] = []
    var prefillChunkRanges: [(lo: Int, hi: Int)] = []
    /// Batched prefill: while set (guarded by qmon), all reader threads take jobs (else only the first opt.readers) and
    /// raw records are queued as prefillChunkRanges. Engine.prefill sets it around a chunk.
    var boost = false
    public var readerThreads: Int { max(opt.readers, opt.prefillReaders) }
    var chunksLeft: [[Int]]  // per (layer, expert) in flight: chunks not yet in (guarded by mon)
    var subAt: [[Double]]  // per (layer, expert): when its first chunk was queued
    var qHead = 0
    let qmon = Monitor()
    let throttle: Throttle?
    /// The read path and reader QoS in force (StoreTuning at init; DriveTest changes them between cells; guarded by qmon).
    public internal(set) var readPath: ReadPath = StoreTuning.readPath
    public internal(set) var readerQoS: qos_class_t = StoreTuning.readerQoS
    /// DriveTest: reader threads at idx >= readerLimit take no jobs (guarded by qmon); the engine never sets it.
    var readerLimit = Int.max
    /// DriveTest, ReadPath.flat: the destination (flatSlots records, contiguous) of every range.
    var flatBase: UnsafeMutableRawPointer? = nil
    var flatSlots = 1
    /// DriveTest: per-record (first chunk queued -> last chunk in) and per-read (one pread/preadv) latencies, seconds
    /// (guarded by mon; off in the engine).
    var driveLog = false
    var driveRecLat: [Double] = []
    var driveChunkLat: [Double] = []

    // stats (guarded by mon unless noted)
    public var bytes = 0
    public var reads = 0
    public var syncReads = 0
    public var evicted = 0
    public var evictedEarly = 0
    public var readSec = 0.0  // sum of read durations (reader threads)
    public var queueSec = 0.0  // submission -> start
    public var policySec = 0.0  // take_slot + naming bookkeeping (host)
    public var firstReadStart = 0.0, lastReadEnd = 0.0
    public var chunkReads = 0  // preadv calls
    public var recordLatSec = 0.0  // sum over records: first chunk queued -> last chunk in
    public var driveBusySec = 0.0  // wall time with at least one read in progress
    var active = 0, activeSince = 0.0
    /// Read depth while a batched prefill chunk runs (boost on; 2026-11-07): seconds, and the time integrals of the
    /// reader threads inside a read (`active`), of the records queued or being read (`recInflight`, demand and
    /// read-ahead), and the seconds with every reader thread inside a read. Never reset (callers take differences).
    public var depthSec = 0.0, depthActiveInt = 0.0, depthRecInt = 0.0, depthSatSec = 0.0
    var depthOn = false, depthT = 0.0
    var recInflight = 0
    /// Call with mon held, before `active` or `recInflight` change.
    @inline(__always) func depthTick() {
        guard depthOn else { return }
        let t = now(), dt = t - depthT
        depthT = t
        depthSec += dt
        depthActiveInt += Double(active) * dt
        depthRecInt += Double(recInflight) * dt
        if active >= readerThreads { depthSatSec += dt }
    }
    // --trace: a timeline of a few tokens (reads, namings, encoder waits), guarded by mon
    public var traceOn = false
    public var missLog: [Int: [Int]]? = nil  // --trace: token clock -> records admitted per layer (prefetch naming)
    /// Teacher forcing (Engine.collectReleased): token clock -> per layer the ids the routers named (sorted; [] when the
    /// layer was not named at that token). nil: not recorded.
    public var nameLog: [Int: [[Int]]]? = nil
    public var events: [(k: String, L: Int, e: Int, a: Double, b: Double)] = []
    public func event(_ k: String, _ L: Int, _ e: Int, _ a: Double, _ b: Double) {
        mon.withLock { if traceOn { events.append((k, L, e, a, b)) } }
    }

    public init(dir: String, store: String, opt: StoreOptions) throws {
        meta = try StoreMeta(path: "\(dir)/\(store)/meta.json")
        self.opt = opt
        NL = meta.layers; NE = meta.experts; K = meta.topK; REC = meta.recordBytes
        if let lc = opt.layerCaps {
            guard lc.count == meta.layers else { throw PSError("--layer-caps: \(lc.count) caps for \(meta.layers) layers") }
            let lo = meta.topK, hi = meta.experts
            guard lc.allSatisfy({ $0 >= lo && $0 <= hi }) else { throw PSError("--layer-caps: every cap must be in \(lo)...\(hi): \(lc)") }
            caps = lc
        } else {
            caps = Array(repeating: opt.cap, count: meta.layers)
        }
        slotOf = Array(repeating: Array(repeating: -1, count: NE), count: NL)
        lastUse = Array(repeating: Array(repeating: -1, count: NE), count: NL)
        insSeq = Array(repeating: Array(repeating: 0, count: NE), count: NL)
        inflight = Array(repeating: Array(repeating: false, count: NE), count: NL)
        nInflight = Array(repeating: 0, count: NL)
        tab = Array(repeating: Array(repeating: -1, count: NE), count: NL)
        tabVersion = Array(repeating: 0, count: NL)
        free = caps.map { Array(0..<$0) }
        namedAt = Array(repeating: -1, count: NL)
        slotHas = caps.map { Array(repeating: -1, count: $0) }
        pending = Array(repeating: [], count: NL)
        aheadInflight = Array(repeating: 0, count: NL)
        aheadDone = Array(repeating: [], count: NL)
        aheadPrev = Array(repeating: [:], count: NL)
        stgOf = Array(repeating: Array(repeating: -1, count: NE), count: NL)
        readsByLayer = Array(repeating: 0, count: NL)
        bytesByLayer = Array(repeating: 0, count: NL)
        chunksLeft = Array(repeating: Array(repeating: 0, count: NE), count: NL)
        subAt = Array(repeating: Array(repeating: 0, count: NE), count: NL)
        precondition(meta.recordBytes % 16384 == 0, "record not a whole number of 16 KB pages")
        chunkRanges = Store.ranges(rec: meta.recordBytes, max(1, opt.chunks))
        prefillChunkRanges = Store.ranges(rec: meta.recordBytes, max(1, opt.prefillReadChunks > 0 ? opt.prefillReadChunks : opt.chunks))
        pol = (opt.evict == .hazard || opt.evict == .prior) ? Evict(kind: opt.evict, layers: NL, experts: NE) : nil
        throttle = opt.bwCap > 0 ? Throttle(gbps: opt.bwCap) : nil
        codecIdx = Array(repeating: nil, count: NL)
        codecReading = Array(repeating: 0, count: NL)
        let env = ProcessInfo.processInfo.environment
        codecBatchMode = env["PS_CODEC_BATCH"].flatMap { CodecBatch(rawValue: $0) } ?? StoreTuning.codecBatch
        codecMaxInflight = min(2, max(1, env["PS_CODEC_INFLIGHT"].flatMap { Int($0) } ?? StoreTuning.codecInflight))
        chunkPlanVersion = env["PS_CHUNK_PLAN"].flatMap { Int($0) }.map { $0 == 1 ? 1 : 2 } ?? StoreTuning.chunkPlanVersion
        chunkPassFrac = min(1, max(0.05, env["PS_CHUNK_PASS_FRAC"].flatMap { Double($0) } ?? StoreTuning.chunkPassFrac))
        chunkPassAhead = max(0, env["PS_CHUNK_PASS_AHEAD"].flatMap { Int($0) } ?? StoreTuning.chunkPassAhead)
        chunkPassLag = min(1, max(0, env["PS_CHUNK_PASS_LAG"].flatMap { Int($0) } ?? StoreTuning.chunkPassLag))
        chunkPassCost = max(0, env["PS_CHUNK_PASS_COST"].flatMap { Double($0) } ?? StoreTuning.chunkPassCost)
        if opt.policyOnly { return }
        for L in 0..<NL {
            var path = "\(dir)/\(store)/layer_\(String(format: "%02d", L)).bin"
            if opt.codec == .auto, let cf = codecFiles(dir: dir, store: store, layer: L) {
                let ix = try CodecIndex(path: cf.idx)
                guard ix.nrec == NE, ix.recordBytes == REC else { throw PSError("\(cf.idx): \(ix.nrec) records of \(ix.recordBytes) B, the store has \(NE) of \(REC)") }
                codecIdx[L] = ix
                path = cf.psc
            }
            let fd = open(path, O_RDONLY)
            if fd < 0 { throw PSError("open layer \(L): \(path)") }
            if opt.noCache {
                _ = fcntl(fd, F_NOCACHE, 1)
                _ = fcntl(fd, F_RDAHEAD, 0)
            }
            fds.append(fd)
        }
        try allocSlabs()
        if codecIdx.contains(where: { $0 != nil }) {
            var big = 0
            for ix in codecIdx.compactMap({ $0 }) { for e in 0..<ix.nrec { let r = ix.alignedRange(e); big = max(big, r.hi - r.lo) } }
            let stride = (big + 16383) & ~16383
            let n = max(opt.codecStaging, 2)
            let dec = try SlotDecoder(layers: NL, stagingSlots: n, stride: stride, inflight: codecMaxInflight)
            for L in 0..<NL {
                guard let ix = codecIdx[L] else { continue }
                dec.setTables(L, ix.tables)
                try dec.setSlab(L, pieces: meta.pieces.map { ($0.m, $0.kind, $0.n) }, ptrs: piecePtr[L], cap: caps[L])
            }
            decoder = dec
            stgFree = Array((0..<n).reversed())
            descFree = Array((0..<codecMaxInflight).reversed())
            startThread(name: "decoder") { [unowned self] in self.decodeLoop() }
        }
        for i in 0..<readerThreads { startThread(name: "reader-\(i)") { [unowned self] in self.readerLoop(i) } }
    }

    /// A record's n byte ranges (16 KB aligned; the last one takes the rest).
    static func ranges(rec: Int, _ n: Int) -> [(lo: Int, hi: Int)] {
        let pg = 16384, pages = rec / pg
        var r: [(lo: Int, hi: Int)] = []
        var lo = 0
        for c in 0..<n {
            let hi = c == n - 1 ? rec : min(rec, lo + ((pages + n - 1) / n) * pg)
            if hi > lo { r.append((lo, hi)) }
            lo = hi
        }
        return r
    }

    func allocSlabs() throws {
        let H = meta.hidden, I = meta.inter
        let rows = [I, I, H], cols = [H, H, I]
        var all: [MLXArray] = []
        for L in 0..<NL {
            let cap = caps[L]
            var w: [MLXArray] = [], s: [MLXArray] = [], b: [MLXArray] = []
            for m in 0..<3 {
                w.append(MLXArray.zeros([cap, rows[m], cols[m] / 8], dtype: .uint32))
                s.append(MLXArray.zeros([cap, rows[m], cols[m] / G], dtype: .float16))
                b.append(MLXArray.zeros([cap, rows[m], cols[m] / G], dtype: .float16))
            }
            W.append(w); S.append(s); Bi.append(b)
            all += w + s + b
        }
        eval(all)
        for L in 0..<NL {
            var ptrs: [UnsafeMutableRawPointer] = []
            for pc in meta.pieces {
                let arr = pc.kind == 0 ? W[L][pc.m] : (pc.kind == 1 ? S[L][pc.m] : Bi[L][pc.m])
                let p = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(mlx_array_data_uint8(arr.ctx)!))
                precondition(arr.nbytes == pc.n * caps[L], "slab piece size")
                precondition(Int(bitPattern: p) % 16384 == 0, "slab buffer not page aligned: F_NOCACHE reads would bounce")
                ptrs.append(p)
            }
            piecePtr.append(ptrs)
        }
    }

    // MARK: reader

    /// Queue every chunk of record (L, e) into `slot` (call with mon held).
    func submit(_ L: Int, _ e: Int, _ slot: Int) {
        let ts = now()
        depthTick()
        recInflight += 1
        subAt[L][e] = ts
        slotHas[L][slot] = -1
        if codecIdx[L] != nil {
            codecReading[L] += 1
            if let s = stgFree.popLast() {
                queueCompressed(L, e, slot, s, ts)
            } else {
                stgWait.append((L, e, slot))
                stgWaits += 1
                mon.broadcast()  // CodecBatch.layer: a record waiting for staging lets the decode thread force a batch
            }
            return
        }
        qmon.withLock {
            let rs = boost ? prefillChunkRanges : chunkRanges
            chunksLeft[L][e] = rs.count
            for c in rs { queue.append((L, e, slot, c.lo, c.hi, ts, -1, 0)) }
            qmon.broadcast()
        }
    }

    /// Queue the compressed record (L, e) into staging slot `stg`: its 4 KiB-aligned range of the .psc, in up to
    /// `chunks` parts (call with mon held).
    func queueCompressed(_ L: Int, _ e: Int, _ slot: Int, _ stg: Int, _ ts: Double) {
        let r = codecIdx[L]!.alignedRange(e)
        let pages = (r.hi - r.lo + 4095) / 4096, n = max(1, min(opt.chunks, pages)), per = (pages + n - 1) / n
        var parts: [(Int, Int)] = []
        var lo = r.lo
        while lo < r.hi {
            let hi = min(r.hi, lo + per * 4096)
            parts.append((lo, hi))
            lo = hi
        }
        chunksLeft[L][e] = parts.count
        stgOf[L][e] = stg
        qmon.withLock {
            for (a, b) in parts { queue.append((L, e, slot, a, b, ts, stg, a - r.lo)) }
            qmon.broadcast()
        }
    }

    /// A record is in its slot (raw: its last chunk read; compressed: decoded). Call with mon held.
    func land(_ L: Int, _ e: Int, _ slot: Int, _ t1: Double) {
        depthTick()
        recInflight -= 1
        reads += 1
        readsByLayer[L] += 1
        recordLatSec += t1 - subAt[L][e]
        if driveLog { driveRecLat.append(t1 - subAt[L][e]) }
        inflight[L][e] = false
        nInflight[L] -= 1
        slotHas[L][slot] = e
        if slotOf[L][e] == slot {  // still ours (never evicted while in flight)
            tab[L][e] = Int32(slot)
            tabVersion[L] += 1
        }
    }

    /// Give a staging slot back; the oldest record waiting for one is queued into it (call with mon held).
    func releaseStaging(_ s: Int) {
        if stgWaitHead < stgWait.count {
            let w = stgWait[stgWaitHead]
            stgWaitHead += 1
            if stgWaitHead == stgWait.count { stgWait.removeAll(keepingCapacity: true); stgWaitHead = 0 }
            queueCompressed(w.L, w.e, w.slot, s, now())
        } else {
            stgFree.append(s)
        }
    }

    /// The next decode batch (call with mon held), nil if none is due. CodecBatch.layer: the read records of every layer
    /// with nothing left to read; if there is none and a record waits for a staging slot, every read record (forced: the
    /// staging slots must come back, or the waiting record's layer could never complete). CodecBatch.landed: everything read.
    /// A batch never exceeds the staging slots (each pending record holds one).
    func takeDecodeBatch() -> [(L: Int, e: Int, slot: Int, stg: Int)]? {
        if decodePending.isEmpty { return nil }
        if codecBatchMode == .landed {
            let b = decodePending
            decodePending.removeAll(keepingCapacity: true)
            return b
        }
        var take: [(L: Int, e: Int, slot: Int, stg: Int)] = [], keep: [(L: Int, e: Int, slot: Int, stg: Int)] = []
        for p in decodePending { if codecReading[p.L] == 0 { take.append(p) } else { keep.append(p) } }
        if take.isEmpty {
            guard stgWaitHead < stgWait.count else { return nil }
            take = decodePending
            keep = []
            codecForcedBatches += 1
        }
        decodePending = keep
        return take
    }

    /// The decode thread: takes a batch (takeDecodeBatch) whenever fewer than codecMaxInflight command buffers are in
    /// flight, commits it, and goes on; the batch's completion handler checks (--codec-check), lands its records (only now:
    /// the fence) and gives back their staging slots and its descriptor region.
    func decodeLoop() {
        let dec = decoder!
        mon.lock()
        while true {
            var next: [(L: Int, e: Int, slot: Int, stg: Int)]? = nil
            while next == nil {
                if decInflight < codecMaxInflight { next = takeDecodeBatch() }
                if next == nil { mon.wait() }
            }
            let batch = next!
            let region = descFree.removeLast()
            decInflight += 1
            let recs = batch.map { b in (L: b.L, slot: b.slot, stg: b.stg, off: Int(codecIdx[b.L]!.offsets[b.e]) - codecIdx[b.L]!.alignedRange(b.e).lo) }
            mon.unlock()
            let t0 = now()
            do {
                try dec.submit(recs, region: region) { [unowned self] gpu, err in
                    if let err { fatalError("\(err)") }
                    self.decodeCompleted(batch, region: region, gpu: gpu, t0: t0)
                }
            } catch { fatalError("\(error)") }
            mon.lock()
        }
    }

    /// A decode command buffer completed (Metal's completion thread).
    func decodeCompleted(_ batch: [(L: Int, e: Int, slot: Int, stg: Int)], region: Int, gpu: Double, t0: Double) {
        let t1 = now()
        var bad = 0
        if opt.codecCheck {  // the slot's 9 pieces, laid out as the record, against the index's SHA-256 of the original
            let tmp = UnsafeMutableRawPointer.allocate(byteCount: REC, alignment: 16384)
            for b in batch {
                for (i, pc) in meta.pieces.enumerated() { memcpy(tmp + pc.off, piecePtr[b.L][i] + b.slot * pc.n, pc.n) }
                if sha256Of(tmp, REC) != codecIdx[b.L]!.sha[b.e] {
                    bad += 1
                    FileHandle.standardError.write("codec check: layer \(b.L) expert \(b.e) slot \(b.slot) MISMATCH\n".data(using: .utf8)!)
                }
            }
            tmp.deallocate()
        }
        mon.withLock {
            for b in batch {
                codecRecords += 1
                codecBytesExact += codecIdx[b.L]!.size(b.e)
                land(b.L, b.e, b.slot, t1)
                stgOf[b.L][b.e] = -1
                releaseStaging(b.stg)
            }
            codecBatches += 1
            decodeGPUSec += gpu
            decodeWallSec += t1 - t0
            if opt.codecCheck { codecChecked += batch.count; codecCheckBad += bad }
            decInflight -= 1
            descFree.append(region)
            mon.broadcast()
        }
    }

    /// Reader thread `idx`: threads at idx >= opt.readers take jobs only while `boost` is set (a batched prefill chunk).
    func readerLoop(_ idx: Int) {
        var iov = [iovec](repeating: iovec(), count: meta.pieces.count)
        let extra = idx >= opt.readers
        var stage: UnsafeMutableRawPointer? = nil  // ReadPath.staged: this reader's page-aligned buffer (one record), on first use
        var curQoS = QOS_CLASS_USER_INTERACTIVE  // startThread's
        while true {
            qmon.lock()
            // demand reads first; a read-ahead record only when none is queued (and only during a chunk: boost)
            while (qHead >= queue.count && !(boost && lowHead < lowQueue.count)) || (extra && !boost) || idx >= readerLimit { qmon.wait() }
            let path = readPath, q = readerQoS
            let low: (L: Int, e: Int, slot: Int, tSub: Double)? = qHead >= queue.count ? popLow() : nil
            var job = (L: 0, e: 0, slot: 0, lo: 0, hi: 0, tSub: 0.0, stg: -1, dst: 0)
            if low == nil {
                job = queue[qHead]
                qHead += 1
                if qHead > 4096 && qHead == queue.count {
                    queue.removeAll(keepingCapacity: true)
                    qHead = 0
                }
            }
            qmon.unlock()  // the one unlock (an early unlock in a branch trips the exclusivity check on the merged lock access)
            if q != curQoS {
                pthread_set_qos_class_self_np(q, 0)
                curQoS = q
            }
            if path == .staged && stage == nil {
                posix_memalign(&stage, 16384, REC)
            }
            if let low {
                readAheadRecord(low, path, &iov, stage)
                continue
            }
            let nb = job.hi - job.lo
            let t0 = now()
            mon.withLock {
                depthTick()
                if active == 0 { activeSince = t0 }
                active += 1
            }
            let due = throttle?.reserve(nb) ?? 0
            if !opt.simReads && job.stg >= 0 {  // compressed: the .psc byte range into the staging slot
                let d = decoder!
                let got = pread(fds[job.L], d.staging.contents() + job.stg * d.stride + job.dst, nb, off_t(job.lo))
                if got != nb { fatalError("short read: \(got) of \(nb) (layer \(job.L) expert \(job.e) .psc at \(job.lo))") }
            } else if !opt.simReads {
                readRaw(job.L, job.e, job.slot, job.lo, job.hi, path, &iov, stage)
            }
            if let throttle { throttle.settle(due) }
            let t1 = now()
            mon.withLock {
                depthTick()
                active -= 1
                if active == 0 { driveBusySec += t1 - activeSince }
                readSec += t1 - t0
                queueSec += t0 - job.tSub
                if firstReadStart == 0 { firstReadStart = t0 }
                lastReadEnd = t1
                bytes += nb
                bytesByLayer[job.L] += nb
                chunkReads += 1
                if driveLog { driveChunkLat.append(t1 - t0) }
                if traceOn { events.append(("r", job.L, job.e, t0, t1)); events.append(("q", job.L, job.e, job.tSub, t0)) }
                chunksLeft[job.L][job.e] -= 1
                if chunksLeft[job.L][job.e] == 0 {
                    if job.stg >= 0 { codecReading[job.L] -= 1 }
                    if job.stg >= 0 && !opt.simReads {
                        decodePending.append((job.L, job.e, job.slot, job.stg))  // the decode thread lands it
                    } else {
                        if job.stg >= 0 {  // simulated reads: nothing to decode (the slot keeps what it holds)
                            stgOf[job.L][job.e] = -1
                            releaseStaging(job.stg)
                        }
                        land(job.L, job.e, job.slot, t1)
                    }
                    mon.broadcast()
                }
            }
        }
    }

    /// Bytes [lo, hi) of raw record (L, e) into `slot` by `path` (ReadPath). On the reader threads.
    @inline(__always) func readRaw(_ L: Int, _ e: Int, _ slot: Int, _ lo: Int, _ hi: Int, _ path: ReadPath, _ iov: inout [iovec],
                                   _ stage: UnsafeMutableRawPointer?) {
        let nb = hi - lo, off = off_t(e * REC + lo)
        switch path {
        case .iovec:
            var k = 0
            for (i, pc) in meta.pieces.enumerated() {  // the pieces' parts inside [lo, hi), each to its place in the slot
                let a = max(lo, pc.off), b = min(hi, pc.off + pc.n)
                if b <= a { continue }
                iov[k] = iovec(iov_base: piecePtr[L][i] + slot * pc.n + (a - pc.off), iov_len: b - a)
                k += 1
            }
            let got = iov.withUnsafeBufferPointer { preadv(fds[L], $0.baseAddress, Int32(k), off) }
            precondition(got == nb, "short read: \(got) of \(nb) (layer \(L) expert \(e) at \(lo))")
        case .staged:
            let st = stage!
            let got = pread(fds[L], st, nb, off)
            precondition(got == nb, "short read: \(got) of \(nb) (layer \(L) expert \(e) at \(lo))")
            for (i, pc) in meta.pieces.enumerated() {
                let a = max(lo, pc.off), b = min(hi, pc.off + pc.n)
                if b <= a { continue }
                memcpy(piecePtr[L][i] + slot * pc.n + (a - pc.off), st + (a - lo), b - a)
            }
        case .flat:
            let got = pread(fds[L], flatBase! + (slot % flatSlots) * REC + lo, nb, off)
            precondition(got == nb, "short read: \(got) of \(nb) (layer \(L) expert \(e) at \(lo))")
        }
    }

    /// A read-ahead record (lowQueue): the whole record into its slot with one read (readRaw, the read path in force), then slotHas (not held: slotOf and
    /// tab stay as they are). Counted in the drive stats like any read.
    /// The next read-ahead record (call with qmon held, lowQueue not empty).
    func popLow() -> (L: Int, e: Int, slot: Int, tSub: Double) {
        let j = lowQueue[lowHead]
        lowHead += 1
        if lowHead == lowQueue.count {
            lowQueue.removeAll(keepingCapacity: true)
            lowHead = 0
        }
        return j
    }

    func readAheadRecord(_ j: (L: Int, e: Int, slot: Int, tSub: Double), _ path: ReadPath, _ iov: inout [iovec], _ stage: UnsafeMutableRawPointer?) {
        let t0 = now()
        mon.withLock {
            depthTick()
            if active == 0 { activeSince = t0 }
            active += 1
        }
        let due = throttle?.reserve(REC) ?? 0
        if !opt.simReads { readRaw(j.L, j.e, j.slot, 0, REC, path, &iov, stage) }
        if let throttle { throttle.settle(due) }
        let t1 = now()
        mon.withLock {
            depthTick()
            active -= 1
            if active == 0 { driveBusySec += t1 - activeSince }
            readSec += t1 - t0
            queueSec += t0 - j.tSub
            if firstReadStart == 0 { firstReadStart = t0 }
            lastReadEnd = t1
            bytes += REC
            bytesByLayer[j.L] += REC
            chunkReads += 1
            aheadBytes += REC
            aheadLanded += 1
            if traceOn { events.append(("ra", j.L, j.e, t0, t1)) }
            if !opt.simReads { slotHas[j.L][j.slot] = j.e }
            aheadDone[j.L].insert(j.e)
            aheadInflight[j.L] -= 1
            recInflight -= 1  // depthTick ran above (active -= 1)
            mon.broadcast()
        }
    }

    // MARK: policy (call with mon held)

    func takeSlot(_ L: Int, _ t: Int) -> Int {
        if let s = free[L].popLast() { return s }
        var victim = -1
        if opt.evict != .window {
            var cands: [Int] = []
            for e in 0..<NE where slotOf[L][e] >= 0 && lastUse[L][e] < t && !inflight[L][e] { cands.append(e) }
            precondition(!cands.isEmpty, "layer \(L): every held expert is this token's or in flight; cap \(caps[L]) too small")
            if let pol { victim = pol.victim(L, t, cands) } else { victim = argminLast(L, cands) }
            if lastUse[L][victim] >= t - opt.keep { evictedEarly += 1 }
        } else {
            var outside: [Int] = []
            for e in 0..<NE where slotOf[L][e] >= 0 && lastUse[L][e] < t - opt.keep && !inflight[L][e] { outside.append(e) }
            if !outside.isEmpty {
                victim = argminLast(L, outside)
            } else {
                var cands: [Int] = []
                for e in 0..<NE where slotOf[L][e] >= 0 && lastUse[L][e] < t && !inflight[L][e] { cands.append(e) }
                precondition(!cands.isEmpty, "layer \(L): cap \(caps[L]) too small")
                victim = argminLast(L, cands)
                evictedEarly += 1
            }
        }
        evicted += 1
        let slot = slotOf[L][victim]
        slotOf[L][victim] = -1
        lastUse[L][victim] = -1
        if tab[L][victim] >= 0 {
            tab[L][victim] = -1
            tabVersion[L] += 1
        }
        return slot
    }

    /// min by last use, ties by the order experts entered the held set (Python's min over a dict).
    func argminLast(_ L: Int, _ cands: [Int]) -> Int {
        var best = cands[0]
        for e in cands.dropFirst() where (lastUse[L][e], insSeq[L][e]) < (lastUse[L][best], insSeq[L][best]) { best = e }
        return best
    }

    func admit(_ L: Int, _ e: Int, _ t: Int) {
        let slot = takeSlot(L, t)
        slotOf[L][e] = slot
        lastUse[L][e] = t
        seq += 1
        insSeq[L][e] = seq
        if opt.policyOnly {
            bytes += REC
            reads += 1
            tab[L][e] = Int32(slot)
            tabVersion[L] += 1
            return
        }
        inflight[L][e] = true
        nInflight[L] += 1
        submit(L, e, slot)
    }

    /// The keep-window drop (mlx_run.py bookkeeping stage), skipped with --no-drop.
    func drop(_ L: Int, _ t: Int) {
        if opt.noDrop { return }
        for e in 0..<NE where slotOf[L][e] >= 0 && lastUse[L][e] < t - opt.keep {
            while inflight[L][e] { mon.wait() }
            free[L].append(slotOf[L][e])
            slotOf[L][e] = -1
            lastUse[L][e] = -1
            if tab[L][e] >= 0 {
                tab[L][e] = -1
                tabVersion[L] += 1
            }
        }
    }

    // MARK: the two modes' entry points

    /// Prefetch mode, I/O side: the routers named `names` for layer L at token t (mlx_run.py name()).
    public func name(_ L: Int, _ t: Int, _ names: [Int]) {
        mon.withLock {
            let t0 = now()
            pol?.touch(L, t, names)
            for c in names where slotOf[L][c] >= 0 { lastUse[L][c] = t }
            for c in names where slotOf[L][c] < 0 { admit(L, c, t) }
            namedAt[L] = t
            if nameLog != nil {
                var row = nameLog![t] ?? [[Int]](repeating: [], count: NL)
                row[L] = names.sorted()
                nameLog![t] = row
            }
            if missLog != nil {
                var row = missLog![t] ?? [Int](repeating: 0, count: NL)
                row[L] = names.filter { slotOf[L][$0] >= 0 && inflight[L][$0] }.count
                missLog![t] = row
            }
            if traceOn { events.append(("n", L, names.filter { slotOf[L][$0] >= 0 && inflight[L][$0] }.count, t0, now())) }
            policySec += now() - t0
            mon.broadcast()
        }
    }

    /// Prefetch mode, encode side: wait until layer L's names for token t were applied and all its reads landed; then the
    /// keep-window drop; returns the slot table (and its version) and the held count. Seconds blocked in `waited`.
    public func layerReady(_ L: Int, _ t: Int, waited: inout Double) -> (tab: [Int32], version: Int, held: Int) {
        mon.lock()
        defer { mon.unlock() }
        let t0 = now()
        while namedAt[L] != t || nInflight[L] > 0 { mon.wait() }
        waited += now() - t0
        drop(L, t)
        let held = slotOf[L].reduce(0) { $0 + ($1 >= 0 ? 1 : 0) }
        return (tab[L], tabVersion[L], held)
    }

    /// Prefetch mode, late wait, first half: true once layer L's names for token t were applied (no wait).
    public func isNamed(_ L: Int, _ t: Int) -> Bool { mon.withLock { namedAt[L] == t } }

    /// Prefetch mode, late wait: wait until layer L's names for token t were applied; then the keep-window drop. Returns
    /// the slot table as it will be once the layer's reads are in (slotOf: every held expert, landed or in flight) and the
    /// held count. The GPU may index with this table but must not read the slab before layerLanded(L, t).
    public func layerNamed(_ L: Int, _ t: Int, waited: inout Double) -> (tab: [Int32], held: Int) {
        mon.lock()
        defer { mon.unlock() }
        let t0 = now()
        while namedAt[L] != t { mon.wait() }
        waited += now() - t0
        drop(L, t)
        var held = 0
        let tb = slotOf[L].map { s -> Int32 in
            if s >= 0 { held += 1 }
            return Int32(s)
        }
        return (tb, held)
    }

    /// Prefetch mode, late wait, second half: true once every read of layer L is in (no wait).
    public func isLanded(_ L: Int) -> Bool { mon.withLock { nInflight[L] == 0 } }

    /// Prefetch mode, late wait: wait until every read of layer L (named for token t) is in; checks that the landed table
    /// equals the one layerNamed returned.
    public func layerLanded(_ L: Int, _ t: Int, waited: inout Double) {
        mon.lock()
        defer { mon.unlock() }
        let t0 = now()
        while namedAt[L] != t || nInflight[L] > 0 { mon.wait() }
        waited += now() - t0
        assert(zip(tab[L], slotOf[L]).allSatisfy { Int($0.0) == $0.1 }, "landed table != slotOf")
    }

    /// Exact mode, before the released choice is known (bookkeeping stage, while the GPU runs the gate).
    public func exactPrepare(_ L: Int, _ t: Int) {
        mon.withLock {
            let t0 = now()
            pol?.prepare(L, t)
            drop(L, t)
            policySec += now() - t0
        }
    }

    /// Exact mode: the released top-k `rel` (in the gate's order); misses are read now, synchronously (8 in flight).
    /// Returns the slots in rel's order and the held count. Seconds blocked on reads in `waited`.
    public func exactChoose(_ L: Int, _ t: Int, _ rel: [Int], waited: inout Double) -> (slots: [UInt32], held: Int) {
        mon.lock()
        defer { mon.unlock() }
        let t0 = now()
        pol?.touch(L, t, rel)
        for e in rel where slotOf[L][e] >= 0 { lastUse[L][e] = t }
        for e in rel {
            if slotOf[L][e] < 0 {
                admit(L, e, t)
                syncReads += 1
            }
            lastUse[L][e] = t
        }
        policySec += now() - t0
        let t1 = now()
        while rel.contains(where: { inflight[L][$0] }) { mon.wait() }
        waited += now() - t1
        let held = slotOf[L].reduce(0) { $0 + ($1 >= 0 ? 1 : 0) }
        return (rel.map { UInt32(slotOf[L][$0]) }, held)
    }

    /// mlx_run.py reset_slots(): forget every held expert (a new conversation: the token clock restarts, so the window
    /// and policy state are void). Waits for reads in flight first; the free lists go back to their initial order.
    public func forgetSlots() {
        mon.lock()
        defer { mon.unlock() }
        while nInflight.contains(where: { $0 > 0 }) { mon.wait() }
        for L in 0..<NL {
            for e in 0..<NE {
                slotOf[L][e] = -1
                lastUse[L][e] = -1
                inflight[L][e] = false
                chunksLeft[L][e] = 0
                if tab[L][e] >= 0 { tab[L][e] = -1 }
            }
            tabVersion[L] += 1
            free[L] = Array(0..<caps[L])
            namedAt[L] = -1
            pending[L] = []  // slotHas (what each slot physically holds) is kept: a later restore or chunk uses it
        }
        hasPending = false
        pol?.reset()
    }

    /// The slabs are freed (releaseSlabs) and not allocated again yet. Engine thread only.
    public private(set) var slabsReleased = false

    /// Free the expert slabs (and the codec decoder's aliases of them) while the app is in the background (2026-11-07,
    /// docs/G4-runtime.md "Launch: once per install"): every landed record becomes pending (held, not landed: the held
    /// set, last uses, insertion order, free lists and policy are untouched, as after a lazy restore) and no slot holds a
    /// record any more. reallocSlabs() allocates them again; the next chunk reads what it uses and step() / landPending()
    /// lands the rest, so the computation is the same as if the slabs had been kept (only slot contents, never the held
    /// set, depend on this). Waits for reads and decodes in flight. Engine thread, at a token boundary, no GPU work in
    /// flight. Returns the bytes released (0 when already released or there are no slabs).
    @discardableResult
    public func releaseSlabs() -> Int {
        guard !slabsReleased, !opt.policyOnly, !W.isEmpty else { return 0 }
        mon.lock()
        while nInflight.contains(where: { $0 > 0 }) || aheadInflight.contains(where: { $0 > 0 }) || decInflight > 0
                || !decodePending.isEmpty { mon.wait() }
        var pend = 0
        for L in 0..<NL {
            var p: [(e: Int, slot: Int)] = []
            for e in 0..<NE where slotOf[L][e] >= 0 {
                p.append((e, slotOf[L][e]))
                tab[L][e] = -1
            }
            pending[L] = p
            pend += p.count
            for sl in 0..<caps[L] { slotHas[L][sl] = -1 }
            tabVersion[L] += 1
        }
        hasPending = pend > 0
        let bytes = slabBytes
        if let dec = decoder { for L in 0..<NL { dec.dst[L] = nil } }
        W = []; S = []; Bi = []
        piecePtr = []
        slabsReleased = true
        mon.unlock()
        return bytes
    }

    /// Allocate the slabs again after releaseSlabs() (empty: every held record is pending until a chunk or landPending()
    /// reads it). No-op when they are allocated.
    public func reallocSlabs() throws {
        guard slabsReleased else { return }
        mon.lock()
        defer { mon.unlock() }
        try allocSlabs()
        if let dec = decoder {
            for L in 0..<NL where codecIdx[L] != nil {
                try dec.setSlab(L, pieces: meta.pieces.map { ($0.m, $0.kind, $0.n) }, ptrs: piecePtr[L], cap: caps[L])
            }
        }
        for L in 0..<NL { tabVersion[L] += 1 }
        slabsReleased = false
    }

    /// The experts layer L holds right now (landed or in flight), sorted.
    public func heldIds(_ L: Int) -> [Int] { mon.withLock { (0..<NE).filter { slotOf[L][$0] >= 0 } } }

    /// Experts held right now, summed over the layers (landed or in flight).
    public var heldNow: Int { mon.withLock { slotOf.reduce(0) { $0 + $1.reduce(0) { $0 + ($1 >= 0 ? 1 : 0) } } } }

    public func resetStats() {
        mon.withLock {
            bytes = 0; reads = 0; syncReads = 0; evicted = 0; evictedEarly = 0
            readSec = 0; queueSec = 0; policySec = 0; firstReadStart = 0; lastReadEnd = 0
            chunkReads = 0; recordLatSec = 0; driveBusySec = 0
            codecRecords = 0; codecBytesExact = 0; codecBatches = 0; decodeGPUSec = 0; decodeWallSec = 0; stgWaits = 0
            codecForcedBatches = 0
            codecChecked = 0; codecCheckBad = 0
            aheadIssued = 0; aheadLanded = 0; aheadCancelled = 0; aheadBytes = 0; aheadUsed = 0; aheadOverwroteHeld = 0
            readsByLayer = Array(repeating: 0, count: NL); bytesByLayer = Array(repeating: 0, count: NL)
            if active > 0 { activeSince = now() }
            pol?.secCritical = 0; pol?.secPrepare = 0
        }
    }

    public func snapshot<R>(_ f: (Store) -> R) -> R { mon.withLock { f(self) } }
    public var policySeconds: (critical: Double, prepare: Double) { mon.withLock { (pol?.secCritical ?? 0, pol?.secPrepare ?? 0) } }
}

/// The drive alone: `threads` readers, each reading random whole records (layer, expert) of the store into its own
/// page-aligned buffer with F_NOCACHE and readahead off, for `seconds`. Returns (GB/s, mean ms per read, reads).
public func readBench(dir: String, store: String, threads: Int, seconds: Double, chunks: Int = 1) throws -> (Double, Double, Int) {
    let meta = try StoreMeta(path: "\(dir)/\(store)/meta.json")
    let RECORD = meta.recordBytes
    let pages = RECORD / 16384, nChunks = max(1, chunks)
    let REC = ((pages + nChunks - 1) / nChunks) * 16384  // one read = one chunk of a record (a whole record at chunks 1)
    var fds: [Int32] = []
    for L in 0..<meta.layers {
        let fd = open("\(dir)/\(store)/layer_\(String(format: "%02d", L)).bin", O_RDONLY)
        _ = fcntl(fd, F_NOCACHE, 1); _ = fcntl(fd, F_RDAHEAD, 0)
        fds.append(fd)
    }
    let mon = Monitor()
    var total = 0, sumLat = 0.0, finished = 0
    let t0 = now()
    for i in 0..<threads {
        let fdsCopy = fds
        startThread(name: "bench-\(i)") {
            var buf: UnsafeMutableRawPointer? = nil
            posix_memalign(&buf, 16384, REC)
            var rng = SystemRandomNumberGenerator()
            var n = 0, lat = 0.0
            while now() - t0 < seconds {
                let L = Int.random(in: 0..<meta.layers, using: &rng), e = Int.random(in: 0..<meta.experts, using: &rng)
                let c = Int.random(in: 0..<nChunks, using: &rng)  // (was 0..<n: the per-thread count, 0 at the first read)
                let a = now()
                let got = pread(fdsCopy[L], buf, REC, off_t(e * RECORD + min(c * REC, RECORD - REC)))
                precondition(got == REC)
                lat += now() - a
                n += 1
            }
            free(buf)
            mon.withLock { total += n; sumLat += lat; finished += 1; mon.broadcast() }
        }
    }
    mon.lock()
    while finished < threads { mon.wait() }
    mon.unlock()
    let dt = now() - t0
    for fd in fds { close(fd) }
    return (Double(total * REC) / dt / 1e9, 1000 * sumLat / Double(max(1, total)), total)
}

// MARK: - batched prefill (Engine.prefill, docs/G4-runtime.md "Batched prefill")

/// One layer of a batched prefill chunk: what the policy keeps, which experts each pass computes with and where they are,
/// and when each read is queued.
///
/// The engine's protocol (Engine.prefillChunk; replayed without a model by Store.chunkPassTest): `issue[0]` is queued at
/// once; then per pass j: wait for `reads[j]`, compute `passes[j]`, and when `issue[j + 1]` is not empty, wait for the
/// GPU work of every pass <= fence[j + 1] (the overwrite fence) and queue `issue[j + 1]`. The planner puts a read in
/// `issue[i]` only when every pass that computes with its slot's earlier record is <= fence[i], so a slot the GPU may
/// still read is never overwritten, and each pass waits for the reads of its own records. `restores` (also in the issue lists, waited for by
/// no pass) are records the held set after the chunk needs but no pass computes with; chunkCommit waits for them (the
/// engine commits a layer after the next layer's mixer has run, so they are read while it computes).
public struct ChunkLayerPlan {
    /// Per pass: the experts first computed in it and their slots (every pass's slots are in memory when it runs).
    public var passes: [[(e: Int, slot: Int)]] = []
    /// Per pass: the records read in this chunk that it computes with (it waits for them).
    public var reads: [[(e: Int, slot: Int)]] = []
    /// passes.count + 1 lists: issue[0] at the start, issue[j + 1] after pass j has been submitted and the GPU work of
    /// every pass <= fence[j + 1] is done (fence -1: none needed).
    public var issue: [[(e: Int, slot: Int)]] = []
    public var fence: [Int] = []
    /// Passes that compute something (a pass list may be empty: with the lagged fence a pass can find no slot free).
    public var computing: Int { passes.reduce(0) { $0 + ($1.isEmpty ? 0 : 1) } }
    /// Records read only for the held set after the chunk (v2: held experts whose slots the chunk used, read back).
    public var restores: [(e: Int, slot: Int)] = []
    /// The held set after the chunk (the policy's), expert -> slot, with its last use and insertion order.
    var finalSlot: [Int: Int] = [:]
    var finalLast: [Int: Int] = [:]
    var finalIns: [Int: Int] = [:]
    var freeSlots: [Int] = []
    /// demoted: held experts that stay held but whose slots the passes used (each read back once: v2's restores, v1's
    /// demotions).
    public var union = 0, present = 0, readCount = 0, demoted = 0
    /// Read-ahead records that landed in this layer before its plan, and of those the ones the chunk used or keeps.
    public var aheadLanded = 0, aheadUsed = 0
    /// Held at the chunk's start but not in the slab (a lazy restore's pending records, or overwritten by read-ahead),
    /// and of those the ones this plan reads (used by the chunk or kept after it).
    public var heldAbsent = 0, heldAbsentRead = 0
    /// The planner that made it (StoreTuning.chunkPlanVersion) and its largest pass in reads.
    public var version = 2, maxPassReads = 0
}

/// The v2 pass schedule of one layer (Store.schedulePasses): see ChunkLayerPlan for the lists.
struct PassSchedule {
    var passes: [[(e: Int, slot: Int)]] = []
    var reads: [[(e: Int, slot: Int)]] = []
    var issue: [[(e: Int, slot: Int)]] = []
    var fence: [Int] = []
    var computing = 0
    var restores: [(e: Int, slot: Int)] = []
    var finalLoaded: [Int: Int] = [:]
}

extension Store {
    /// Batched prefill, layer L, chunk tokens at clocks t0 + 1 ... t0 + C, whose released top-k are `rows` (C rows).
    ///
    /// Policy (the semantics): the chunk's tokens are applied to the layer in position order exactly as exact mode does
    /// one token at a time (exactPrepare + exactChoose: tick, keep-window drop, touch, last use, a miss admitted with the
    /// policy's victim), without reading anything. That leaves the held set, last uses, insertion order and the policy's
    /// history a token-by-token exact-mode feed of these released top-k would leave. Which slot holds which expert is a
    /// physical detail (nothing reads slot numbers but the GPU's gather), so the slots are then chosen to read well:
    ///
    /// Physical: every expert of the chunk's union U is computed once, from a slot that holds its record while that
    /// pass runs. Experts already in the slab (held, or a stale record left in a slot: slotHas) are used in place.
    /// v2 (StoreTuning.chunkPlanVersion, 2026-11-07): the held set after the chunk (hf) is known before any slot is
    /// chosen, so the slots are planned around it. Passes take up to P = min(ceil(frac x W), W - ahead) experts (W: the
    /// slots the chunk may write): first the present experts of U (no read; P per pass, so the first ones' slots take
    /// reads while the later ones compute), then the absent ones, those hf does not keep first, the kept ones last (they
    /// stay in the slot they are read into). Each read takes the slot that became free earliest; with the lagged fence
    /// (chunkPassLag 1) a slot pass f computed with is written only once pass f + 1 has been submitted, behind a wait for
    /// pass f's GPU work, which the GPU has finished by then, so the host never blocks on the pass it just submitted and,
    /// at P ~ W / 3, pass j + 2's reads are queued while pass j + 1's land (the drive does not drain between passes).
    /// Experts that hf keeps and that are in the slab ("stable") either keep their slot for the whole chunk or give it to
    /// the passes and are read back at the end (a restore: one extra read, queued as early as its slot allows and waited
    /// for only at the layer's commit, which the engine makes after the next layer's mixer). The number given up, m (those
    /// the chunk does not compute first, then those computed early), is chosen on a grid of ~9 values from 0 to |stable|
    /// as the schedule with the least computing passes x chunkPassCost + m.
    /// v1: the planner of commit 3ace2ed (stable experts never written, passes of half the free slots), expressed in the
    /// same lists.
    /// Call with no reads in flight in layer L (the engine drains the namer and the reads before a chunk).
    public func chunkPlan(_ L: Int, t0: Int, rows: [[Int]]) -> ChunkLayerPlan {
        mon.lock()
        defer { mon.unlock() }
        let tp0 = now()
        while nInflight[L] > 0 { mon.wait() }
        let ahead = settleReadAhead(L)
        // a lazy restore's pending records of this layer: planned below like any absent expert (read if used or kept)
        pending[L] = []
        hasPending = pending.contains { !$0.isEmpty }
        let cap = caps[L]
        let slot0 = slotOf[L]
        // where each expert's record physically is (its held slot first, else any slot whose bytes are its record)
        var presentSlot: [Int: Int] = [:]
        for s in 0..<cap where slotHas[L][s] >= 0 && presentSlot[slotHas[L][s]] == nil { presentSlot[slotHas[L][s]] = s }
        for e in 0..<NE where slot0[e] >= 0 && slotHas[L][slot0[e]] == e { presentSlot[e] = slot0[e] }
        let heldAbsent = Set((0..<NE).filter { slot0[$0] >= 0 && presentSlot[$0] == nil })
        // the policy, token by token (exact mode's bookkeeping, no reads)
        for (p, rel) in rows.enumerated() {
            let t = t0 + 1 + p
            pol?.prepare(L, t)
            drop(L, t)
            pol?.touch(L, t, rel)
            for e in rel where slotOf[L][e] >= 0 { lastUse[L][e] = t }
            for e in rel where slotOf[L][e] < 0 {
                let s = takeSlot(L, t)
                slotOf[L][e] = s
                seq += 1
                insSeq[L][e] = seq
                lastUse[L][e] = t
            }
        }
        var plan = ChunkLayerPlan()
        var hf = Set<Int>()
        for e in 0..<NE where slotOf[L][e] >= 0 {
            hf.insert(e)
            plan.finalLast[e] = lastUse[L][e]
            plan.finalIns[e] = insSeq[L][e]
        }
        // the layer holds nothing while the chunk's passes run (land() then leaves tab alone); chunkCommit sets it
        for e in 0..<NE {
            slotOf[L][e] = -1
            lastUse[L][e] = -1
            if tab[L][e] >= 0 { tab[L][e] = -1 }
        }
        tabVersion[L] += 1
        var uSet = Set<Int>()
        for r in rows { for e in r { uSet.insert(e) } }
        let U = uSet.sorted()
        plan.union = U.count
        var stable: [Int: Int] = [:]  // kept experts already in the slab: their slot
        for e in hf { if let s = presentSlot[e] { stable[e] = s } }
        let computed: Set<Int>
        if chunkPlanVersion == 1 {
            computed = chunkPassesV1(L, cap: cap, U: U, uSet: uSet, hf: hf, stable: stable, presentSlot: presentSlot, plan: &plan)
        } else {
            computed = chunkPassesV2(L, cap: cap, U: U, uSet: uSet, hf: hf, stable: stable, presentSlot: presentSlot, plan: &plan)
        }
        precondition(plan.issue.count == plan.passes.count + 1 && plan.fence.count == plan.issue.count && plan.reads.count == plan.passes.count,
                     "chunk plan: lists out of shape")
        precondition(plan.finalSlot.count == hf.count, "chunk plan: \(plan.finalSlot.count) slots for \(hf.count) held experts")
        let used = Set(plan.finalSlot.values)
        precondition(used.count == plan.finalSlot.count, "chunk plan: two held experts in one slot")
        plan.freeSlots = (0..<cap).filter { !used.contains($0) }
        precondition(computed == uSet, "chunk plan: \(uSet.subtracting(computed).count) experts of the union not computed")
        plan.readCount = plan.issue.reduce(0) { $0 + $1.count }
        plan.maxPassReads = plan.reads.map(\.count).max() ?? 0
        plan.aheadLanded = ahead.count
        plan.aheadUsed = ahead.filter { uSet.contains($0) || hf.contains($0) }.count
        aheadUsed += plan.aheadUsed
        plan.heldAbsent = heldAbsent.count
        plan.heldAbsentRead = plan.issue.reduce(0) { $0 + $1.filter { heldAbsent.contains($0.e) }.count }
        policySec += now() - tp0
        return plan
    }

    /// v2 passes (see chunkPlan). Returns the experts computed.
    func chunkPassesV2(_ L: Int, cap: Int, U: [Int], uSet: Set<Int>, hf: Set<Int>, stable: [Int: Int], presentSlot: [Int: Int],
                       plan: inout ChunkLayerPlan) -> Set<Int> {
        var p0: [(e: Int, slot: Int)] = []  // the union's present experts, computed where they are
        for e in U { if let s = presentSlot[e] { p0.append((e, s)) } }
        plan.present = p0.count
        let absent = U.filter { presentSlot[$0] == nil }
        let transient = absent.filter { !hf.contains($0) }.map { (e: $0, compute: true, final: false) }
        let keptRead = absent.filter { hf.contains($0) }.map { (e: $0, compute: true, final: true) }
        let heldUnused = hf.filter { presentSlot[$0] == nil && !uSet.contains($0) }.sorted()  // held, absent, not used (rare)
        // stable experts in the order they are given up: those the chunk does not compute (their slot is free at once),
        // then those computed in the first pass
        let order = stable.keys.filter { !uSet.contains($0) }.sorted() + stable.keys.filter { uSet.contains($0) }.sorted()
        var best: (cost: Double, m: Int, s: PassSchedule)? = nil
        let base = transient + keptRead
        // m on a coarse grid (0, every ~8th, all): the cost is flat near its minimum and each schedule costs host time
        let step = max(1, order.count / 8)
        let ms = base.isEmpty ? [0] : Array(Set(Array(stride(from: 0, through: order.count, by: step)) + [order.count])).sorted()
        for m in ms {
            let keptSlots = Set(order[m...].map { stable[$0]! })
            let queue = base + (heldUnused + order[..<m]).map { (e: $0, compute: false, final: true) }
            let W = cap - keptSlots.count
            let P = max(1, min(Int((chunkPassFrac * Double(W)).rounded(.up)), W - chunkPassAhead))
            // merged (the present experts with the first reads in one pass) only pays when one pass could take everything
            for merged in p0.isEmpty || p0.count + base.count > cap ? [false] : [false, true] {
                let s = Store.schedulePasses(cap: cap, keptSlots: keptSlots, p0: p0, merged: merged, queue: queue, P: P, lag: chunkPassLag)
                let cost = Double(s.computing) * chunkPassCost + Double(m)
                if best == nil || cost < best!.cost { best = (cost, m, s) }
            }
        }
        let s = best!.s
        plan.passes = s.passes
        plan.reads = s.reads
        plan.issue = s.issue
        plan.fence = s.fence
        plan.restores = s.restores
        plan.demoted = best!.m
        for e in order[best!.m...] { plan.finalSlot[e] = stable[e]! }
        for (e, sl) in s.finalLoaded { plan.finalSlot[e] = sl }
        return Set(s.passes.flatMap { $0.map(\.e) })
    }

    /// One v2 schedule: slots `keptSlots` are never written; `p0` (present experts) computed in pass 0 (`merged`: in
    /// pass 1, with the first reads); `queue` read in order, each into the non-kept slot that became free earliest
    /// (ties: lowest slot), up to P reads per pass (restores, after every computed expert, take what is left). A slot
    /// last computed with by pass f is free for pass j's read when f + lag < j, and the read is queued after pass
    /// f + lag has been submitted, behind the fence on pass f (f = -1, never used: at the start). A `final` read's slot
    /// is never written again. A pass that finds no free slot stays empty (lag 1, the pool nearly all final); passes
    /// after the last computing one are dropped and their reads (restores) move to after it.
    static func schedulePasses(cap: Int, keptSlots: Set<Int>, p0: [(e: Int, slot: Int)], merged: Bool,
                               queue: [(e: Int, compute: Bool, final: Bool)], P: Int, lag: Int) -> PassSchedule {
        let never = Int.max
        var freeAfter = [Int](repeating: -1, count: cap)  // the last pass that computes with the slot (-1: none yet)
        for s in keptSlots { freeAfter[s] = never }
        var passes: [[(e: Int, slot: Int)]] = [[], []], reads: [[(e: Int, slot: Int)]] = [[], []]
        var pending: [(e: Int, slot: Int, f: Int)] = []  // every read and the pass it must be fenced behind
        var out = PassSchedule()
        // the present experts: all in pass 1 with the first reads (merged), or P per pass from pass 0 on, so the slots of
        // the first ones take reads while the later ones compute
        var pi0 = 0
        func placePresent(_ j: Int, _ n: Int) -> Int {
            var k = 0
            while pi0 < p0.count && k < n {
                let p = p0[pi0]
                passes[j].append(p)
                if !keptSlots.contains(p.slot) { freeAfter[p.slot] = j }
                pi0 += 1
                k += 1
            }
            return k
        }
        _ = placePresent(merged ? 1 : 0, merged ? p0.count : P)
        var qi = 0, j = 1, idle = 0
        while qi < queue.count || pi0 < p0.count {
            while passes.count <= j { passes.append([]); reads.append([]) }
            let placed = placePresent(j, P)
            var avail = (0..<cap).filter { freeAfter[$0] == -1 || (freeAfter[$0] != never && freeAfter[$0] + lag < j) }
            avail.sort { (freeAfter[$0], $0) < (freeAfter[$1], $1) }
            idle = avail.isEmpty && placed == 0 ? idle + 1 : 0
            precondition(idle <= lag + 1, "chunk plan: no slot for pass \(j) (\(queue.count - qi) reads left)")
            var n = placed
            for s in avail {
                guard qi < queue.count else { break }
                let q = queue[qi]
                if q.compute && n >= P { break }
                qi += 1
                n += 1
                pending.append((q.e, s, freeAfter[s]))
                if q.compute {
                    passes[j].append((q.e, s))
                    reads[j].append((q.e, s))
                } else {
                    out.restores.append((q.e, s))
                }
                if q.final {
                    freeAfter[s] = never
                    out.finalLoaded[q.e] = s
                } else {
                    freeAfter[s] = j
                }
            }
            j += 1
        }
        let last = passes.lastIndex { !$0.isEmpty } ?? 0
        out.passes = Array(passes[...last])
        out.reads = Array(reads[...last])
        out.computing = out.passes.reduce(0) { $0 + ($1.isEmpty ? 0 : 1) }
        out.issue = Array(repeating: [], count: last + 2)
        out.fence = Array(repeating: -1, count: last + 2)
        for r in pending {  // issue[i + 1]: after pass i is submitted
            let i = r.f < 0 ? 0 : min(r.f + lag, last) + 1
            out.issue[i].append((r.e, r.slot))
            out.fence[i] = max(out.fence[i], r.f)
        }
        return out
    }

    /// v1 passes: the planner of commit 3ace2ed (stable experts never written; while more remains than one pass can
    /// take, a pass takes half the slots it may write, and pass k + 2's reads go in after pass k's GPU work), its
    /// per-pass reads mapped onto the issue lists (reads[0] and reads[1] at the start, reads[k + 2] after pass k).
    func chunkPassesV1(_ L: Int, cap: Int, U: [Int], uSet: Set<Int>, hf: Set<Int>, stable st: [Int: Int], presentSlot: [Int: Int],
                       plan: inout ChunkLayerPlan) -> Set<Int> {
        plan.version = 1
        var stable = st
        let stableSlots = Set(stable.values)
        var work = (0..<cap).filter { !stableSlots.contains($0) }  // slots the chunk may write
        var p1: [(e: Int, slot: Int)] = []
        var busy = Set<Int>()
        for e in U {
            if let s = stable[e] { p1.append((e, s)) } else if let s = presentSlot[e], !stableSlots.contains(s) {
                p1.append((e, s))
                busy.insert(s)
            }
        }
        plan.present = p1.count
        let absent = U.filter { stable[$0] == nil && (presentSlot[$0] == nil || stableSlots.contains(presentSlot[$0]!)) }
        var queue = absent.filter { !hf.contains($0) } + absent.filter { hf.contains($0) }
        queue += hf.filter { stable[$0] == nil && !uSet.contains($0) }.sorted()  // held, not in the slab, not used (none expected)
        var finalLoaded: [Int: Int] = [:]  // kept experts read during the chunk: their slot
        var freeFrom: [Int: Int] = [:]  // work slot -> first pass that may write it
        for s in work { freeFrom[s] = busy.contains(s) ? 3 : 1 }
        var computed = Set(p1.map { $0.e })
        var passes: [[(e: Int, slot: Int)]] = [], reads: [[(e: Int, slot: Int)]] = []
        var k = 1
        var idle = 0
        while true {
            var pass: [(e: Int, slot: Int)] = k == 1 ? p1 : []
            var rd: [(e: Int, slot: Int)] = []
            let finals = Set(finalLoaded.values)
            let pool = work.filter { !finals.contains($0) }
            var avail = pool.filter { freeFrom[$0]! <= k }
            if !queue.isEmpty && avail.isEmpty && pool.allSatisfy({ freeFrom[$0]! <= k }) {
                // nothing will free up: every slot is stable or holds a kept expert; give up a stable one (read back later)
                if let (x, s) = stable.min(by: { $0.key < $1.key }) {
                    stable[x] = nil
                    work.append(s)
                    freeFrom[s] = uSet.contains(x) ? max(k, 3) : k  // pass 1 computed with it
                    avail = freeFrom[s]! <= k ? [s] : []
                    queue.append(x)
                    plan.demoted += 1
                }
            }
            let limit = queue.count <= avail.count ? avail.count : max(1, (pool.count + 1) / 2)
            for s in avail.prefix(min(limit, queue.count)) {
                let e = queue.removeFirst()
                rd.append((e, s))
                if uSet.contains(e) && !computed.contains(e) {
                    pass.append((e, s))
                    computed.insert(e)
                }
                if hf.contains(e) { finalLoaded[e] = s } else { freeFrom[s] = k + 2 }
            }
            passes.append(pass)
            reads.append(rd)
            idle = pass.isEmpty && rd.isEmpty ? idle + 1 : 0
            precondition(idle < 4, "chunk plan: no progress (layer \(L), queue \(queue.count))")
            k += 1
            if queue.isEmpty { break }
        }
        while let last = passes.last, last.isEmpty, reads.last!.isEmpty { passes.removeLast(); reads.removeLast() }
        for (e, s) in stable { plan.finalSlot[e] = s }
        for (e, s) in finalLoaded { plan.finalSlot[e] = s }
        let np = passes.count
        plan.passes = passes
        plan.reads = reads  // v1 waits for every read of the pass's list, including read-backs it does not compute with
        var issue: [[(e: Int, slot: Int)]] = [reads[0]]
        if np > 1 { issue[0] += reads[1] }
        for j in 0..<np { issue.append(j + 2 < np ? reads[j + 2] : []) }
        plan.issue = issue
        plan.fence = [-1] + (0..<np).map { $0 }  // v1: pass k + 2's reads behind pass k's GPU work
        return computed
    }

    /// Read-ahead (Engine.prefillChunk): queue up to `budget` low-priority reads of the experts the routers named for
    /// layer L at the chunk's positions (`rows`, one list per position), most named first, skipping those whose record is
    /// already in the slab, into the layer's empty slots (no record in them). With `overwrite` 1 also stale records the
    /// routers did not name, with 2 also held experts the routers did not name (diagnostics; 0 by
    /// default since the 2026-11-07 Mac app runs: about 44% of the records the routers did not name were needed by the
    /// chunk after all, so overwriting them cost a re-read each) also stale records the routers did not name, then held
    /// experts the routers did not name (least recently used first; they become pending: still held, not landed). A slot
    /// whose record the routers named is never taken. Nothing else changes: the held set, last uses, insertion order and
    /// the policy are untouched, and the chunk's plan for layer L (chunkPlan) decides from what it then finds in the slab.
    /// Call during a chunk (boost on), before layer L's plan, on the engine thread. Returns the records queued.
    @discardableResult
    public func issueReadAhead(_ L: Int, rows: [[Int]], budget: Int, overwrite: Int = 0) -> Int {
        guard budget > 0, !opt.policyOnly, codecIdx[L] == nil else { return 0 }
        return mon.withLock { () -> Int in
            precondition(aheadInflight[L] == 0, "read-ahead issued twice for layer \(L) in one chunk")
            var votes: [Int: Int] = [:], first: [Int: Int] = [:]
            var i = 0
            for r in rows { for e in r { votes[e, default: 0] += 1; if first[e] == nil { first[e] = i; i += 1 } } }
            let present = Set(slotHas[L].filter { $0 >= 0 })
            let want = votes.keys.filter { !present.contains($0) }.sorted { (votes[$0]!, -first[$0]!) > (votes[$1]!, -first[$1]!) }
            if want.isEmpty { return 0 }
            let cap = caps[L]
            var empty: [Int] = [], stale: [Int] = [], held: [(s: Int, last: Int)] = []
            for s in 0..<cap {
                let x = slotHas[L][s]
                if x < 0 { empty.append(s) } else if votes[x] != nil { continue } else if slotOf[L][x] == s { held.append((s, lastUse[L][x])) } else { stale.append(s) }
            }
            let cands = empty + (overwrite >= 1 ? stale : []) + (overwrite >= 2 ? held.sorted { ($0.last, $0.s) < ($1.last, $1.s) }.map(\.s) : [])
            let n = min(budget, want.count, cands.count)
            if n == 0 { return 0 }
            let ts = now()
            var jobs: [(L: Int, e: Int, slot: Int, tSub: Double)] = []
            for k in 0..<n {
                let s = cands[k], e = want[k]
                let x = slotHas[L][s]
                if x >= 0 && slotOf[L][x] == s {  // a held expert's record is overwritten: still held, now pending
                    tab[L][x] = -1
                    tabVersion[L] += 1
                    pending[L].append((x, s))
                    hasPending = true
                    aheadOverwroteHeld += 1
                }
                aheadPrev[L][s] = x
                slotHas[L][s] = -1
                jobs.append((L, e, s, ts))
            }
            aheadInflight[L] += n
            aheadIssued += n
            depthTick()
            recInflight += n
            qmon.withLock {
                lowQueue += jobs
                qmon.broadcast()
            }
            return n
        }
    }

    /// Layer L's read-ahead before its plan (mon held): records still queued are dropped (their slots keep what they
    /// held), those being read are waited for. Returns the experts read ahead into layer L since its last plan.
    func settleReadAhead(_ L: Int) -> Set<Int> {
        if aheadInflight[L] > 0 {
            var gone: [(e: Int, slot: Int)] = []
            qmon.withLock {
                var keep: [(L: Int, e: Int, slot: Int, tSub: Double)] = []
                for k in lowHead..<lowQueue.count {
                    let j = lowQueue[k]
                    if j.L == L { gone.append((j.e, j.slot)) } else { keep.append(j) }
                }
                lowQueue = keep
                lowHead = 0
            }
            aheadInflight[L] -= gone.count
            aheadCancelled += gone.count
            depthTick()
            recInflight -= gone.count
            for g in gone { if let p = aheadPrev[L][g.slot], p >= 0 { slotHas[L][g.slot] = p } }  // never overwritten
            while aheadInflight[L] > 0 { mon.wait() }
        }
        aheadPrev[L] = [:]
        let d = aheadDone[L]
        aheadDone[L] = []
        return d
    }

    /// Queue a pass's reads (all at once: with `boost` every reader thread takes them).
    public func chunkIssue(_ L: Int, _ reads: [(e: Int, slot: Int)]) {
        guard !reads.isEmpty else { return }
        mon.withLock {
            for r in reads {
                if opt.policyOnly {
                    bytes += REC
                    self.reads += 1
                    slotHas[L][r.slot] = r.e
                    continue
                }
                inflight[L][r.e] = true
                nInflight[L] += 1
                submit(L, r.e, r.slot)
            }
        }
    }

    /// Wait until every read of layer L is in. Seconds blocked in `waited`.
    public func chunkWait(_ L: Int, waited: inout Double) {
        mon.lock()
        defer { mon.unlock() }
        let t0 = now()
        while nInflight[L] > 0 { mon.wait() }
        waited += now() - t0
    }

    /// The layer's state after the chunk (its last token at clock t): the policy's held set in the plan's slots, landed.
    public func chunkCommit(_ L: Int, _ plan: ChunkLayerPlan, t: Int) {
        mon.withLock {
            while nInflight[L] > 0 { mon.wait() }
            for e in 0..<NE { slotOf[L][e] = -1; lastUse[L][e] = -1; tab[L][e] = -1 }
            for (e, s) in plan.finalSlot {
                precondition(slotHas[L][s] == e || opt.simReads, "chunk commit: layer \(L) slot \(s) holds \(slotHas[L][s]), not \(e)")
                slotOf[L][e] = s
                lastUse[L][e] = plan.finalLast[e]!
                insSeq[L][e] = plan.finalIns[e]!
                tab[L][e] = Int32(s)
            }
            free[L] = plan.freeSlots
            namedAt[L] = t
            tabVersion[L] += 1
            mon.broadcast()
        }
    }

    /// Batched prefill: all reader threads on (and reads queued as prefill chunks) while set.
    public func setBoost(_ on: Bool) {
        mon.withLock {
            depthTick()
            depthOn = on
            depthT = now()
        }
        qmon.withLock { boost = on; qmon.broadcast() }
    }

    /// The read-depth integrals (depthSec, depthActiveInt, depthRecInt, depthSatSec).
    public var depthStats: (sec: Double, active: Double, records: Double, saturated: Double) {
        mon.withLock { depthTick(); return (depthSec, depthActiveInt, depthRecInt, depthSatSec) }
    }

    /// Wait until no read is in flight in any layer.
    public func waitAllLanded() { mon.withLock { while nInflight.contains(where: { $0 > 0 }) { mon.wait() } } }
}

extension Store {
    /// Wait until the records `reads` of layer L are in (one pass of a chunk; later passes' reads may still be in flight).
    public func chunkWait(_ L: Int, _ reads: [(e: Int, slot: Int)], waited: inout Double) {
        mon.lock()
        defer { mon.unlock() }
        let t0 = now()
        while reads.contains(where: { inflight[L][$0.e] }) { mon.wait() }
        waited += now() - t0
    }
}
