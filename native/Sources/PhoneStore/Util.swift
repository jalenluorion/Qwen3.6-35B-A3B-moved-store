// Small utilities: clocks, a mutex + condition variable, safetensors headers, .npy / .npz (stored) readers.
import Foundation

@inline(__always) public func now() -> Double {
    var ts = timespec()
    clock_gettime(CLOCK_UPTIME_RAW, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) * 1e-9
}

/// pthread mutex + condition variable (the reader workers, the name coordinator and the encode thread share one).
public final class Monitor: @unchecked Sendable {
    private var m = pthread_mutex_t()
    private var c = pthread_cond_t()
    public init() {
        pthread_mutex_init(&m, nil)
        pthread_cond_init(&c, nil)
    }
    deinit {
        pthread_mutex_destroy(&m)
        pthread_cond_destroy(&c)
    }
    @inline(__always) public func lock() { pthread_mutex_lock(&m) }
    @inline(__always) public func unlock() { pthread_mutex_unlock(&m) }
    @inline(__always) public func wait() { pthread_cond_wait(&c, &m) }
    @inline(__always) public func broadcast() { pthread_cond_broadcast(&c) }
    @inline(__always) public func withLock<R>(_ f: () throws -> R) rethrows -> R {
        pthread_mutex_lock(&m)
        defer { pthread_mutex_unlock(&m) }
        return try f()
    }
}

/// LocalizedError: `error.localizedDescription` is the message (without it, Foundation bridges the struct to
/// "The operation couldn't be completed. (PhoneStore.PSError error 1.)", which is how the app logged the phone's
/// mapped-weights failure of 2026-11-06 without its reason).
public struct PSError: Error, CustomStringConvertible, LocalizedError {
    public let description: String
    public init(_ s: String) { description = s }
    public var errorDescription: String? { description }
}

/// Ask for the performance cores (QOS_CLASS_USER_INTERACTIVE), as the Python runner does for its threads.
public func setQoSUserInteractive() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
}

public func startThread(name: String, _ body: @escaping () -> Void) {
    let t = Thread {
        setQoSUserInteractive()
        body()
    }
    t.name = name
    t.stackSize = 1 << 20
    t.start()
}

// MARK: - safetensors header

public struct TensorInfo {
    public let dtype: String
    public let shape: [Int]
    public let begin: Int  // absolute file offset
    public let end: Int
}

public func safetensorsHeader(_ path: String) throws -> [String: TensorInfo] {
    guard let fh = FileHandle(forReadingAtPath: path) else { throw PSError("cannot open \(path)") }
    defer { try? fh.close() }
    let lenData = fh.readData(ofLength: 8)
    let n = lenData.withUnsafeBytes { $0.load(as: UInt64.self) }
    let hdr = fh.readData(ofLength: Int(n))
    guard let obj = try JSONSerialization.jsonObject(with: hdr) as? [String: Any] else { throw PSError("bad header \(path)") }
    var out: [String: TensorInfo] = [:]
    for (k, v) in obj where k != "__metadata__" {
        guard let d = v as? [String: Any], let dt = d["dtype"] as? String, let sh = d["shape"] as? [Int], let off = d["data_offsets"] as? [Int]
        else { continue }
        out[k] = TensorInfo(dtype: dt, shape: sh, begin: 8 + Int(n) + off[0], end: 8 + Int(n) + off[1])
    }
    return out
}

// MARK: - .npz (zip, stored entries only, as np.savez writes them)

public struct NpyArray {
    public let descr: String
    public let shape: [Int]
    public let data: Data
}

/// Parse a .npy blob: magic, version, header dict ('descr', 'fortran_order', 'shape'), raw little-endian data.
public func parseNpy(_ d: Data) throws -> NpyArray {
    guard d.count > 10, d[0] == 0x93, d[1] == 0x4E else { throw PSError("not npy") }
    let major = d[6]
    var hlen = 0, start = 0
    if major == 1 {
        hlen = Int(d[8]) | (Int(d[9]) << 8)
        start = 10
    } else {
        hlen = Int(d[8]) | (Int(d[9]) << 8) | (Int(d[10]) << 16) | (Int(d[11]) << 24)
        start = 12
    }
    let header = String(decoding: d[start..<(start + hlen)], as: UTF8.self)
    func field(_ name: String) -> String? {
        guard let r = header.range(of: "'\(name)':") else { return nil }
        return String(header[r.upperBound...]).trimmingCharacters(in: .whitespaces)
    }
    guard let dsc = field("descr"), let q1 = dsc.firstIndex(of: "'") else { throw PSError("npy descr") }
    let rest = dsc[dsc.index(after: q1)...]
    let descr = String(rest[..<rest.firstIndex(of: "'")!])
    if let fo = field("fortran_order"), fo.hasPrefix("True") { throw PSError("fortran order npy not supported") }
    guard let shp = field("shape"), let p0 = shp.firstIndex(of: "("), let p1 = shp.firstIndex(of: ")") else { throw PSError("npy shape") }
    let shape = shp[shp.index(after: p0)..<p1].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    return NpyArray(descr: descr, shape: shape, data: d.subdata(in: (start + hlen)..<d.count))
}

/// Every entry of a stored (uncompressed) zip: name -> .npy contents. The file is memory-mapped.
public func readNpz(_ path: String) throws -> [String: NpyArray] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path), options: .alwaysMapped)
    var out: [String: NpyArray] = [:]
    var p = 0
    func u16(_ o: Int) -> Int { Int(d[o]) | (Int(d[o + 1]) << 8) }
    func u32(_ o: Int) -> Int { u16(o) | (u16(o + 2) << 16) }
    while p + 30 <= d.count, u32(p) == 0x0403_4B50 {
        let method = u16(p + 8)
        var csize = u32(p + 18)
        let nlen = u16(p + 26), xlen = u16(p + 28)
        let name = String(decoding: d[(p + 30)..<(p + 30 + nlen)], as: UTF8.self)
        if csize == 0xFFFF_FFFF {  // zip64 extra field
            var q = p + 30 + nlen
            let xe = q + xlen
            while q + 4 <= xe {
                let id = u16(q), sz = u16(q + 2)
                if id == 1 { csize = u32(q + 12) | (u32(q + 16) << 32) }
                q += 4 + sz
            }
        }
        guard method == 0 else { throw PSError("npz entry \(name) is compressed") }
        let start = p + 30 + nlen + xlen
        let key = name.hasSuffix(".npy") ? String(name.dropLast(4)) : name
        out[key] = try parseNpy(d.subdata(in: start..<(start + csize)))
        p = start + csize
    }
    return out
}

// MARK: - process memory (what jetsam charges on iOS; the same ledger on macOS)

/// task_vm_info: phys_footprint (dirty + compressed + IOKit/Metal memory the task owns; clean file-backed pages are not
/// in it), resident_size, external (file-backed resident pages), compressed; task_events_info: faults and pageins.
public struct ProcMemory: Sendable {
    public let footprint: Int, resident: Int, external: Int, compressed: Int
    public let faults: Int, pageins: Int
    public static func read() -> ProcMemory {
        var vm = task_vm_info_data_t()
        var n = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &vm) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(n)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &n) }
        }
        var ev = task_events_info_data_t()
        var m = mach_msg_type_number_t(MemoryLayout<task_events_info_data_t>.size / MemoryLayout<natural_t>.size)
        let ke = withUnsafeMutablePointer(to: &ev) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(m)) { task_info(mach_task_self_, task_flavor_t(TASK_EVENTS_INFO), $0, &m) }
        }
        let ok = kr == KERN_SUCCESS
        return ProcMemory(footprint: ok ? Int(vm.phys_footprint) : 0, resident: ok ? Int(vm.resident_size) : 0,
                          external: ok ? Int(vm.external) : 0, compressed: ok ? Int(vm.compressed) : 0,
                          faults: ke == KERN_SUCCESS ? Int(ev.faults) : 0, pageins: ke == KERN_SUCCESS ? Int(ev.pageins) : 0)
    }
}
