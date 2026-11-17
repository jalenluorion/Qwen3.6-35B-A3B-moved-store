// Per-layer slot caps (2026-11-04, docs/G4-runtime.md "Per-layer caps"). The misses are front-loaded (layers 0-3 miss
// most and are named last, after the sample), so the same total of slots cuts bytes when layers 0-3 get more of them.
// `auto:TOTAL` looks the allocation up in LayerCapsTable (docs/evict_runs/layer_caps_table.py: the greedy rule of
// layer_caps.py on the hazard_h128 miss curves of the wiki / code / chat traces); the memory budget is the sum.
import Foundation

public enum LayerCaps {
    /// The table's allocation for `total` slots over 40 layers: total is rounded down to the table's step (40); below the
    /// table (< 400) the caps are uniform floor(total / 40) (>= 8), above it (> 5120) the largest row is used.
    public static func auto(total: Int, layers: Int = 40) throws -> [Int] {
        guard layers == 40 else { throw PSError("--layer-caps auto: the table is for 40 layers, the model has \(layers)") }
        if total < LayerCapsTable.minTotal {
            let u = total / layers
            guard u >= 8 else { throw PSError("--layer-caps auto:\(total): fewer than 8 slots per layer") }
            return Array(repeating: u, count: layers)
        }
        let t = min(total, LayerCapsTable.maxTotal)
        return LayerCapsTable.rows[(t - LayerCapsTable.minTotal) / LayerCapsTable.step]
    }

    /// The total `auto:TOTAL` actually allocates (the table's step, the bounds above).
    public static func autoTotal(_ total: Int) -> Int { (try? auto(total: total).reduce(0, +)) ?? 0 }

    /// `--layer-caps` value: "uniform" (nil: the uniform cap), "auto:TOTAL", a comma/space separated list of 40 ints, or
    /// a file holding such a list (or a JSON array). Returns the caps and a label for logs.
    public static func parse(_ spec: String, layers: Int = 40) throws -> (caps: [Int]?, label: String) {
        if spec == "uniform" { return (nil, "uniform") }
        if spec.hasPrefix("auto:") {
            guard let t = Int(spec.dropFirst(5)) else { throw PSError("--layer-caps auto:TOTAL, got \(spec)") }
            let c = try auto(total: t, layers: layers)
            return (c, "auto:\(t)" + (c.reduce(0, +) != t ? " (\(c.reduce(0, +)) allocated)" : ""))
        }
        var text = spec
        var label = "list"
        if FileManager.default.fileExists(atPath: spec), let d = FileManager.default.contents(atPath: spec), let s = String(data: d, encoding: .utf8) {
            text = s
            label = (spec as NSString).lastPathComponent
        }
        let v = text.split(whereSeparator: { ",[] \n\t".contains($0) }).map { Int($0) }
        guard !v.isEmpty, v.allSatisfy({ $0 != nil }) else { throw PSError("--layer-caps: not a list of ints: \(spec.prefix(80))") }
        let c = v.map { $0! }
        guard c.count == layers else { throw PSError("--layer-caps: \(c.count) values, need \(layers)") }
        return (c, label)
    }
}
