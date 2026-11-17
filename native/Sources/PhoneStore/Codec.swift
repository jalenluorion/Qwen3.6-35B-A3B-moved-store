// The compressed-store read path (2026-11-04, docs/G4-runtime.md "Codec read path"): PSC1 records (codec/CODEC.md,
// codec/encode_layers.py) decoded on the GPU straight into the slab slots.
//
// A layer whose `<store>/layer_XX.psc` + `layer_XX.idx` exist (and --store-codec auto) is read compressed: the reader pool
// reads a record's compressed bytes (4 KiB-aligned byte range of the .psc, from the index, split into --read-chunks
// parts like a raw record) into a staging slot; when the last part is in, the record goes to the decode list. The decode
// thread takes a batch (Store.takeDecodeBatch; since 2026-11-07 every listed record of each layer that has no record
// still being read, before that everything on the list, still PS_CODEC_BATCH=landed), encodes one command buffer on its own Metal queue (two
// dispatches: the 96 nibble blocks and the 12 side blocks of every record) that writes each record's 9 pieces directly
// into its slot of the layer's slab, and only when it has completed marks the records landed (the slot table), so
// the engine's layerLanded wait covers the decode exactly as it covers a raw read. Which slot a record may be written into
// is decided by the same policy as the raw path (Store.takeSlot): a victim is never named or used for the current token
// and never in flight, and every earlier token's GPU work is complete, so the decode never writes a slot the GPU may
// still read.
//
// The kernels are codec/Sources/CodecCore/Shader.swift's (same bit reader, tables, bias prediction, -fno-fast-math),
// with the output addressing changed: per record a descriptor holds the GPU addresses of its 9 destination pieces
// (MTLBuffer.gpuAddress of an alias of each slab piece, Metal 3) and the offset of its layer's tables, and a threadgroup
// never straddles two records (it loads one layer's table).
import Foundation
import Metal
import CryptoKit

public enum StoreCodec: String { case off, auto }

/// PSC1 index (codec/encode_layers.py): u32 header ('PSC1', version 1, layer, nrec, record_bytes, 3 reserved), u8
/// lens[4][256], u16 tables[4][4096], u64 offsets[nrec+1] into the .psc, sha256[nrec][32] of the original records.
public struct CodecIndex {
    public let layer: Int, nrec: Int, recordBytes: Int
    public let tables: [UInt16]  // 4 x 4096
    public let offsets: [UInt64]  // nrec + 1
    public let sha: [[UInt8]]  // nrec x 32

    public init(path: String) throws {
        guard let d = FileManager.default.contents(atPath: path) else { throw PSError("codec index \(path): unreadable") }
        func u32(_ o: Int) -> Int { Int(d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) }) }
        guard d.count >= 32, String(data: d.prefix(4), encoding: .ascii) == "PSC1", u32(4) == 1 else { throw PSError("codec index \(path): not PSC1") }
        layer = u32(8); nrec = u32(12); recordBytes = u32(16)
        var p = 32 + 4 * 256
        let pp = p
        tables = d.withUnsafeBytes { b in (0..<(4 * 4096)).map { b.loadUnaligned(fromByteOffset: pp + 2 * $0, as: UInt16.self) } }
        p += 2 * 4 * 4096
        let po = p, n = nrec
        offsets = d.withUnsafeBytes { b in (0...n).map { b.loadUnaligned(fromByteOffset: po + 8 * $0, as: UInt64.self) } }
        p += 8 * (nrec + 1)
        guard p + 32 * nrec == d.count else { throw PSError("codec index \(path): size \(d.count) != \(p + 32 * nrec)") }
        let ps = p
        sha = (0..<nrec).map { i in [UInt8](d[(ps + 32 * i)..<(ps + 32 * i + 32)]) }
    }

    public func size(_ e: Int) -> Int { Int(offsets[e + 1] - offsets[e]) }
    /// The 4 KiB-aligned byte range of the .psc that holds record e (what the drive reads).
    public func alignedRange(_ e: Int) -> (lo: Int, hi: Int) {
        let lo = Int(offsets[e]) & ~4095
        let hi = min((Int(offsets[e + 1]) + 4095) & ~4095, Int(offsets[nrec]))
        return (lo, hi)
    }
}

/// `<dir>/<store>/layer_XX.{psc,idx}` present?
public func codecFiles(dir: String, store: String, layer L: Int) -> (psc: String, idx: String)? {
    let base = "\(dir)/\(store)/layer_\(String(format: "%02d", L))"
    let fm = FileManager.default
    guard fm.fileExists(atPath: base + ".psc"), fm.fileExists(atPath: base + ".idx") else { return nil }
    return (base + ".psc", base + ".idx")
}

let pscSlotShaderSource = """
#include <metal_stdlib>
using namespace metal;

constant uint HDR = 108 * 4;      // u32 block offsets

// dst: codes of gate/up/down, scales of gate/up/down, biases of gate/up/down, each already at the record's slot
struct RecDesc { device uint* dst[9]; uint compOff; uint tabOff; uint pad0; uint pad1; };

struct BitReader {
    uint buf;
    int  cnt;
    uint pos;
    device const ushort* s;

    inline void refill(uint lane) {
        bool need = cnt < 12;
        ulong m = static_cast<ulong>(static_cast<simd_vote::vote_t>(simd_ballot(need)));
        if (need) {
            uint idx = pos + popcount(m & ((1ul << lane) - 1ul));
            buf |= uint(s[idx]) << uint(16 - cnt);
            cnt += 16;
        }
        pos += popcount(m);
    }
    inline uint sym(threadgroup const ushort* T, uint lane) {
        refill(lane);
        uint e = T[buf >> 20];
        uint len = e >> 8;
        buf <<= len; cnt -= int(len);
        return e & 255u;
    }
    inline uint raw8(bool take, uint lane) {
        refill(lane);
        uint v = take ? (buf >> 24) : 0u;
        uint len = take ? 8u : 0u;
        buf <<= len; cnt -= int(len);
        return v;
    }
};

// 96 nibble blocks per record (matrix m = b / 32, block b % 32 = 16 KiB of code words); one record per threadgroup range
kernel void decode_nib_slot(device const uchar*   comp [[buffer(0)]],
                            device const RecDesc* desc [[buffer(1)]],
                            device const ushort*  tabs [[buffer(3)]],
                            uint tg   [[threadgroup_position_in_grid]],
                            uint tid  [[thread_index_in_threadgroup]],
                            uint tpg  [[threads_per_threadgroup]],
                            uint sg   [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]])
{
    uint sgs = tpg / 32, per = 96 / sgs;
    uint r = tg / per, b = (tg % per) * sgs + sg;
    device const RecDesc& d = desc[r];
    threadgroup ushort T[4096];
    for (uint i = tid; i < 4096; i += tpg) T[i] = tabs[d.tabOff + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar* rec = comp + d.compOff;
    BitReader br;
    br.buf = 0; br.cnt = 0; br.pos = 0;
    br.s = (device const ushort*)(rec + HDR) + ((device const uint*)rec)[b];
    device uint* o = d.dst[b / 32] + (b % 32) * 4096 + lane;
    for (uint k = 0; k < 128; ++k) {
        uint w = br.sym(T, lane);
        w |= br.sym(T, lane) << 8;
        w |= br.sym(T, lane) << 16;
        w |= br.sym(T, lane) << 24;
        o[32 * k] = w;
    }
}

// 12 side blocks per record (matrix m = b / 4, 4096 groups each): fp16 scale + fp16 bias per group
kernel void decode_side_slot(device const uchar*   comp [[buffer(0)]],
                             device const RecDesc* desc [[buffer(1)]],
                             device const ushort*  tabs [[buffer(3)]],
                             uint tg   [[threadgroup_position_in_grid]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint tpg  [[threads_per_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]])
{
    uint sgs = tpg / 32, per = 12 / sgs;
    uint r = tg / per, b = (tg % per) * sgs + sg;
    device const RecDesc& d = desc[r];
    threadgroup ushort T[3 * 4096];
    for (uint i = tid; i < 3 * 4096; i += tpg) T[i] = tabs[d.tabOff + 4096 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar* rec = comp + d.compOff;
    BitReader br;
    br.buf = 0; br.cnt = 0; br.pos = 0;
    br.s = (device const ushort*)(rec + HDR) + ((device const uint*)rec)[96 + b];
    device uint* os = d.dst[3 + b / 4] + (b % 4) * 2048 + lane;
    device uint* ob = d.dst[6 + b / 4] + (b % 4) * 2048 + lane;
    for (uint k = 0; k < 64; ++k) {
        uint sw = 0, bw = 0;
        for (uint gi = 0; gi < 2; ++gi) {
            uint hi = br.sym(T, lane);
            uint lo = br.sym(T + 4096, lane);
            uint q  = br.sym(T + 8192, lane);
            bool esc = q == 255u;
            uint rh = br.raw8(esc, lane);
            uint rl = br.raw8(esc, lane);
            uint sv = (hi << 8) | lo;
            uint bv;
            if (esc) bv = (rh << 8) | rl;
            else if (q == 128u) bv = 0u;
            else bv = uint(as_type<ushort>(half(float(int(q) - 128) * float(as_type<half>(ushort(sv))))));
            sw |= sv << (16 * gi);
            bw |= bv << (16 * gi);
        }
        os[32 * k] = sw;
        ob[32 * k] = bw;
    }
}
"""

/// The GPU side: staging memory the readers fill, per-layer tables, aliases of the codec layers' slab pieces, and the
/// two kernels. Up to `inflight` batches at once (Store's decode thread, StoreTuning.codecInflight): each in-flight batch
/// has its own region of the descriptor buffer.
final class SlotDecoder {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let nibPSO: MTLComputePipelineState, sidePSO: MTLComputePipelineState
    let nibSGs: Int, sideSGs: Int
    let tables: MTLBuffer
    let staging: MTLBuffer
    let stride: Int
    let slots: Int
    let descBuf: MTLBuffer
    let inflight: Int
    /// per layer, per destination d (kind * 3 + m): the alias of that slab piece and its bytes per slot (nil: raw layer)
    var dst: [[(buf: MTLBuffer, perSlot: Int)]?]
    static let descBytes = 88

    init(layers: Int, stagingSlots: Int, stride: Int, inflight: Int = 1) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw PSError("codec: no Metal device") }
        device = dev
        guard let q = dev.makeCommandQueue() else { throw PSError("codec: no command queue") }
        queue = q
        let opts = MTLCompileOptions()
        opts.languageVersion = .version3_0
        if #available(macOS 15.0, iOS 18.0, *) { opts.mathMode = .safe } else { opts.fastMathEnabled = false }
        let lib = try dev.makeLibrary(source: pscSlotShaderSource, options: opts)
        nibPSO = try dev.makeComputePipelineState(function: lib.makeFunction(name: "decode_nib_slot")!)
        sidePSO = try dev.makeComputePipelineState(function: lib.makeFunction(name: "decode_side_slot")!)
        // simdgroups per threadgroup: a divisor of the record's block count, so a threadgroup never straddles two records
        func pick(_ blocks: Int, _ pso: MTLComputePipelineState, _ pref: [Int]) -> Int {
            pref.first { blocks % $0 == 0 && $0 * 32 <= pso.maxTotalThreadsPerThreadgroup } ?? 1
        }
        nibSGs = pick(96, nibPSO, [8, 4, 2, 1])
        sideSGs = pick(12, sidePSO, [12, 6, 4, 3, 2, 1])
        guard let t = dev.makeBuffer(length: layers * 4 * 4096 * 2, options: .storageModeShared) else { throw PSError("codec: tables") }
        tables = t
        self.stride = stride
        slots = stagingSlots
        guard let s = dev.makeBuffer(length: stagingSlots * stride, options: [.storageModeShared, .hazardTrackingModeUntracked]) else {
            throw PSError("codec: staging \(stagingSlots) x \(stride) B")
        }
        staging = s
        self.inflight = max(1, inflight)
        guard let db = dev.makeBuffer(length: self.inflight * stagingSlots * SlotDecoder.descBytes, options: .storageModeShared) else {
            throw PSError("codec: descriptors")
        }
        descBuf = db
        dst = Array(repeating: nil, count: layers)
    }

    func setTables(_ L: Int, _ t: [UInt16]) {
        precondition(t.count == 4 * 4096)
        t.withUnsafeBytes { memcpy(tables.contents() + L * 4 * 4096 * 2, $0.baseAddress!, $0.count) }
    }

    /// Alias the 9 slab pieces of layer L (page-aligned shared memory owned by MLX) for the GPU's writes.
    func setSlab(_ L: Int, pieces: [(m: Int, kind: Int, n: Int)], ptrs: [UnsafeMutableRawPointer], cap: Int) throws {
        var d = [(buf: MTLBuffer, perSlot: Int)?](repeating: nil, count: 9)
        for (i, pc) in pieces.enumerated() {
            guard let b = device.makeBuffer(bytesNoCopy: ptrs[i], length: pc.n * cap, options: .storageModeShared, deallocator: nil) else {
                throw PSError("codec: cannot alias layer \(L) slab piece \(i) (\(pc.n * cap) B)")
            }
            d[pc.kind * 3 + pc.m] = (b, pc.n)
        }
        dst[L] = d.map { $0! }
    }

    /// Decode `recs` (layer, slot, staging slot, byte offset of the record inside its staging slot) into their slots;
    /// returns the GPU seconds of the command buffer. Blocks until the GPU is done.
    func decode(_ recs: [(L: Int, slot: Int, stg: Int, off: Int)]) throws -> Double {
        let done = DispatchSemaphore(value: 0)
        var gpu = 0.0
        var err: Error? = nil
        try submit(recs, region: 0) { g, e in gpu = g; err = e; done.signal() }
        done.wait()
        if let err { throw err }
        return gpu
    }

    /// Encode and commit one batch, its descriptors in region `region` (0 ..< inflight; the caller keeps one batch per
    /// region in flight). `completed(gpuSeconds, error)` runs on Metal's completion thread once the GPU is done.
    func submit(_ recs: [(L: Int, slot: Int, stg: Int, off: Int)], region: Int,
                completed: @escaping (Double, Error?) -> Void) throws {
        let n = recs.count
        precondition(n <= slots && region >= 0 && region < inflight)
        let base = region * slots * SlotDecoder.descBytes
        let p = descBuf.contents() + base
        var used: [ObjectIdentifier: MTLBuffer] = [:]
        for (i, r) in recs.enumerated() {
            let q = p + i * SlotDecoder.descBytes
            let ds = dst[r.L]!
            for k in 0..<9 {
                q.storeBytes(of: ds[k].buf.gpuAddress + UInt64(r.slot * ds[k].perSlot), toByteOffset: 8 * k, as: UInt64.self)
                used[ObjectIdentifier(ds[k].buf)] = ds[k].buf
            }
            q.storeBytes(of: UInt32(r.stg * stride + r.off), toByteOffset: 72, as: UInt32.self)
            q.storeBytes(of: UInt32(r.L * 4 * 4096), toByteOffset: 76, as: UInt32.self)
            q.storeBytes(of: UInt64(0), toByteOffset: 80, as: UInt64.self)
        }
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else { throw PSError("codec: encoder") }
        enc.setBuffer(staging, offset: 0, index: 0)
        enc.setBuffer(descBuf, offset: base, index: 1)
        enc.setBuffer(tables, offset: 0, index: 3)
        enc.useResources(Array(used.values), usage: .write)
        enc.setComputePipelineState(nibPSO)
        enc.dispatchThreadgroups(MTLSize(width: n * 96 / nibSGs, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: nibSGs * 32, height: 1, depth: 1))
        enc.setComputePipelineState(sidePSO)
        enc.dispatchThreadgroups(MTLSize(width: n * 12 / sideSGs, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: sideSGs * 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.addCompletedHandler { cb in
            completed(cb.gpuEndTime - cb.gpuStartTime, cb.error.map { PSError("codec: decode command buffer failed: \($0)") })
        }
        cb.commit()
    }
}

func sha256Of(_ p: UnsafeRawPointer, _ n: Int) -> [UInt8] { Array(SHA256.hash(data: UnsafeRawBufferPointer(start: p, count: n))) }
