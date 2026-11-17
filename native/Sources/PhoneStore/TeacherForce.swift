// Teacher forcing (numerical agreement between devices, 2026-11-03): feed a reference run's token ids (prompt + its
// generated ids) one at a time through the normal pipeline (Engine.step: the same cache policy, lag-4 routers, slots and
// read path as a greedy reply) and record, at every generated position i (the logits after prompt + ids[0..<i]):
//   argmax      the sampler's pick (Engine.sampleGreedy, exactly what a greedy reply would emit there)
//   match       argmax == ids[i]
//   margin      top-1 minus top-2 logit (float16 logits, read as float32)
//   ref_logit   the logit of ids[i]; gap = top-1 logit - ref_logit (0 when they match)
//   top         the top-K (id, logit) pairs, sorted by logit (desc), K = 32 by default
// and, when another run's top-K is given for the same ids (e.g. the Mac's, bundled with the app):
//   ref_max_abs  max |logit[id] - other[id]| over the other run's top-K ids (the max-abs logit difference on those ids)
//   ref_argmax / ref_margin   the other run's top-1 id and top-1 minus top-2 margin
// Every step (prompt tokens included) also records the released top-k expert ids of every layer (the experts actually
// computed: in prefetch mode the masked top-k among the experts in memory), `released[s]` = layers x top-k ids, each
// layer's ids sorted; step s = promptIds.count - 1 + i produced position i's logits. Against the other run's `released`:
// the first (step, layer) whose expert set differs, and per position the number of layers that differ in its step.
// Since 2026-11-03 (after the first phone runs) every step also records, per layer, the ids the routers named at that step
// (`named[s][L]`, sorted, [] when the layer was not named) and the experts held when the released choice was made
// (`held[s][L]`, sorted: prefetch = the slot table the masked top-k used, landed or in flight (the routed step waits for
// the reads, so an expert in flight counts as in memory); exact = before that layer's misses were admitted). Against the
// other run's: the first step whose names / held sets differ and, at the first released difference, which cause it was
// (summary.routing_cause).
// One extra GPU read-back per position (the top-K); the arithmetic of the model is unchanged.
// File format (CLI --tf-out and the app's Documents/tf-<cap>.json): tfReport() below; ios/PhoneStoreApp/README.md
// "Teacher forcing" and native/ref/tf_summary.py.
import Foundation
import MLX

public struct TFPosition: Sendable {
    public let i: Int
    public let refNext: Int
    public let argmax: Int
    public let margin: Float
    public let refLogit: Float
    public let top: [(Int, Float)]
    public let refMaxAbs: Float?
    public let refArgmax: Int?
    public let refMargin: Float?
    public let routingDiffLayers: Int?  // layers whose released experts differ from the other run's, in this position's step
    public var match: Bool { argmax == refNext }
}

/// Another teacher-forced run of the same ids (tfReference(report)): its top-K per position, released experts per step,
/// and (files since 2026-11-03) the named and held experts per step and layer.
public struct TFReference: Sendable {
    public var top: [[(Int, Float)]]
    public var released: [[Int]]?
    public var named: [[[Int]]]? = nil
    public var held: [[[Int]]]? = nil
    public init(top: [[(Int, Float)]], released: [[Int]]?, named: [[[Int]]]? = nil, held: [[[Int]]]? = nil) {
        self.top = top; self.released = released; self.named = named; self.held = held
    }
}

public struct TFRun: Sendable {
    public var promptIds: [Int]
    public var ids: [Int]
    public var positions: [TFPosition]
    public var k: Int
    public var hasRef: Bool
    public var prefillSec: Double
    public var seconds: Double
    public var stopped: Bool  // cancelled before every position ran
    public var split: [String: Double]
    public var released: [[Int]]  // per step (prompt + positions run): layers x top-k expert ids, each layer sorted
    public var named: [[[Int]]] = []  // per step, per layer: the ids the routers named (sorted; [] when not named)
    public var held: [[[Int]]] = []  // per step, per layer: the experts held when the released choice was made (sorted)
    public var topK: Int  // the model's top-k (ids per layer in `released`)
    public var refReleased: [[Int]]?
    public var refNamed: [[[Int]]]? = nil
    public var refHeld: [[[Int]]]? = nil
    public var mode: String = "prefetch"
}

/// The other run in a teacher-forced report (tfReport's JSON): positions[i].top and `released`.
public func tfReference(_ report: [String: Any]) -> TFReference? {
    guard let top = tfTopK(report) else { return nil }
    return TFReference(top: top, released: report["released"] as? [[Int]], named: report["named"] as? [[[Int]]],
                       held: report["held"] as? [[[Int]]])
}

/// The top-K of every position from a teacher-forced report (tfReport's `positions[i].top`), or nil.
public func tfTopK(_ report: [String: Any]) -> [[(Int, Float)]]? {
    guard let ps = report["positions"] as? [[String: Any]] else { return nil }
    var out: [[(Int, Float)]] = []
    for p in ps {
        guard let t = p["top"] as? [[Any]] else { return nil }
        out.append(t.compactMap { e in
            guard e.count == 2, let id = (e[0] as? NSNumber)?.intValue, let v = (e[1] as? NSNumber)?.floatValue else { return nil }
            return (id, v)
        })
    }
    return out
}

/// Feed `promptIds` then `ids` through `engine` (a fresh session: forget() first) and record every position.
/// `ref`: another run of the same ids (its top-K per position, its released experts per step), compared when present.
public func teacherForce(engine: Engine, promptIds: [Int], ids: [Int], k: Int = 32, ref: TFReference? = nil,
                         isCancelled: () -> Bool = { false }, progress: (TFPosition) -> Void = { _ in }) -> TFRun {
    let refTop = ref?.top
    let NL = engine.res.cfg.layers, TK = engine.res.cfg.topK
    engine.forget()
    engine.resetStats()
    engine.collectReleased = true
    defer { engine.collectReleased = false }
    var released: [[Int]] = [], named: [[[Int]]] = [], held: [[[Int]]] = []
    func takeSteps() {  // the steps since the last call: released (one flat row, NL x TK, per step), named, held
        let r = engine.takeRouting()
        for row in r.released { released.append(Array(row.joined())) }
        named += r.named
        held += r.held
    }
    func diffLayers(_ s: Int) -> Int? {
        guard let rr = ref?.released, s < rr.count, s < released.count, rr[s].count == NL * TK else { return nil }
        return (0..<NL).filter { Array(rr[s][($0 * TK)..<(($0 + 1) * TK)]) != Array(released[s][($0 * TK)..<(($0 + 1) * TK)]) }.count
    }
    let tp = now()
    var logits = MLXArray(0)
    logits = engine.prefill(promptIds)  // EngineOptions.prefillChunk (0: step() per token, as before)
    eval(logits)
    let prefillSec = now() - tp
    takeSteps()
    engine.resetStats()
    let g0 = gpuStats()
    let t0 = now()
    var out: [TFPosition] = []
    var stopped = false
    for i in 0..<ids.count {
        if isCancelled() { stopped = true; break }
        let am = engine.sampleGreedy(logits)  // the normal flow's pick (uses the head-start argmax)
        let lg = logits.asType(.float32)
        let topI = argPartition(-lg, kth: k - 1)[..<k]
        let topV = lg[topI]
        let rt = (refTop != nil && i < refTop!.count) ? refTop![i] : nil
        let gIdx = MLXArray([Int32(ids[i])] + (rt?.map { Int32($0.0) } ?? []))
        let gV = lg[gIdx]
        eval(topI, topV, gV)
        let ti = topI.asArray(UInt32.self), tv = topV.asArray(Float.self), gv = gV.asArray(Float.self)
        if i > 0 { takeSteps() }  // the step that fed ids[i - 1] (its logits are evaluated)
        let top = zip(ti, tv).map { (Int($0), $1) }.sorted { $0.1 != $1.1 ? $0.1 > $1.1 : $0.0 < $1.0 }
        var maxAbs: Float? = nil, rArg: Int? = nil, rMargin: Float? = nil
        if let rt, !rt.isEmpty {
            maxAbs = zip(rt, gv.dropFirst()).map { abs($0.1 - $1) }.max()
            let rs = rt.sorted { $0.1 != $1.1 ? $0.1 > $1.1 : $0.0 < $1.0 }
            rArg = rs[0].0
            rMargin = rs.count > 1 ? rs[0].1 - rs[1].1 : nil
        }
        let p = TFPosition(i: i, refNext: ids[i], argmax: am, margin: top[0].1 - top[1].1, refLogit: gv[0], top: top,
                           refMaxAbs: maxAbs, refArgmax: rArg, refMargin: rMargin,
                           routingDiffLayers: diffLayers(promptIds.count - 1 + i))
        out.append(p)
        progress(p)
        logits = engine.step(ids[i])  // the reference's token, whatever the argmax was
    }
    eval(logits)
    takeSteps()  // the last step (its logits are not a recorded position)
    let dt = now() - t0
    let g1 = gpuStats()
    let split = PhoneStoreEngine.split(engine, tokens: out.count, dt: dt, g0: g0, g1: g1)
    return TFRun(promptIds: promptIds, ids: ids, positions: out, k: k, hasRef: refTop != nil, prefillSec: prefillSec,
                 seconds: dt, stopped: stopped, split: split, released: released, named: named, held: held, topK: TK,
                 refReleased: ref?.released, refNamed: ref?.named, refHeld: ref?.held, mode: engine.opt.mode.rawValue)
}

/// Margin thresholds of the summary (positions with a top-1 minus top-2 margin below each).
public let tfMarginBins: [Float] = [0.05, 0.1, 0.5]

/// The summary of a run: matches, mismatched positions with their margins, the margin distribution, the max-abs logit
/// difference against the other run's top-K when there is one.
public func tfSummary(_ r: TFRun) -> [String: Any] {
    let ps = r.positions
    let mism = ps.filter { !$0.match }
    var s: [String: Any] = [
        "positions": ps.count, "matches": ps.count - mism.count, "mismatches": mism.count,
        "first_mismatch": mism.first.map { $0.i as Any } ?? NSNull(),
        "mismatch_positions": mism.map { $0.i },
        "min_margin": ps.map { Double($0.margin) }.min() ?? 0,
        "min_margin_position": ps.min { $0.margin < $1.margin }.map { $0.i as Any } ?? NSNull(),
        "stopped": r.stopped,
    ]
    for b in tfMarginBins { s["margin_below_\(b)"] = ps.filter { $0.margin < b }.count }
    if r.hasRef {
        let d = ps.compactMap { $0.refMaxAbs }.map(Double.init)
        s["ref_max_abs_max"] = d.max() ?? 0
        s["ref_max_abs_mean"] = d.isEmpty ? 0 : d.reduce(0, +) / Double(d.count)
        s["ref_max_abs_max_position"] = ps.max { ($0.refMaxAbs ?? 0) < ($1.refMaxAbs ?? 0) }.map { $0.i as Any } ?? NSNull()
        let am = ps.filter { $0.refArgmax != nil && $0.refArgmax != $0.argmax }
        s["argmax_differs_from_ref_run"] = am.map { $0.i }
    }
    if let fr = tfFirstRoutingDiff(r) {
        s["routing_first_diff"] = ["step": fr.step, "layer": fr.layer, "position": fr.step - (r.promptIds.count - 1),
                                   "ids": fr.ids, "ref_ids": fr.refIds]
        s["routing_diff_steps"] = fr.steps
    } else if r.refReleased != nil {
        s["routing_first_diff"] = NSNull()
        s["routing_diff_steps"] = 0
    }
    if let a = tfFirstListDiff(r.named, r.refNamed) {
        s["named_first_diff"] = ["step": a.step, "layer": a.layer, "ids": a.ids, "ref_ids": a.refIds]
        s["named_diff_steps"] = a.steps
    } else if r.refNamed != nil { s["named_first_diff"] = NSNull(); s["named_diff_steps"] = 0 }
    if let a = tfFirstListDiff(r.held, r.refHeld) {
        s["held_first_diff"] = ["step": a.step, "layer": a.layer, "ids": a.ids, "ref_ids": a.refIds]
        s["held_diff_steps"] = a.steps
    } else if r.refHeld != nil { s["held_first_diff"] = NSNull(); s["held_diff_steps"] = 0 }
    if let c = tfRoutingCause(r) { s["routing_cause"] = c }
    return s
}

/// The first (step, layer) whose per-layer id lists differ between two runs, and how many steps differ at all.
public func tfFirstListDiff(_ a: [[[Int]]], _ b: [[[Int]]]?) -> (step: Int, layer: Int, ids: [Int], refIds: [Int], steps: Int)? {
    guard let b else { return nil }
    var first: (Int, Int, [Int], [Int])? = nil
    var steps = 0
    for s in 0..<min(a.count, b.count) where a[s] != b[s] {
        steps += 1
        if first == nil, let L = (0..<min(a[s].count, b[s].count)).first(where: { a[s][$0] != b[s][$0] }) {
            first = (s, L, a[s][L], b[s][L])
        }
    }
    guard let f = first else { return nil }
    return (f.0, f.1, f.2, f.3, steps)
}

/// At the first released-expert difference (step s, layer L) against the other run, with both runs' named / held sets:
/// whether the two runs held the same experts there (then the masked top-k ranked the same set differently: the gate
/// scores), or different ones (then: the first step <= s at which the routers named different experts for layer L).
public func tfRoutingCause(_ r: TFRun) -> [String: Any]? {
    guard let fr = tfFirstRoutingDiff(r), let rh = r.refHeld, let rn = r.refNamed, fr.step < rh.count, fr.step < r.held.count
    else { return nil }
    let s = fr.step, L = fr.layer
    let held = r.held[s][L], refHeld = rh[s][L]
    let ours = fr.ids.filter { !fr.refIds.contains($0) }, theirs = fr.refIds.filter { !fr.ids.contains($0) }
    var c: [String: Any] = [
        "step": s, "layer": L, "released_only_here": ours, "released_only_ref": theirs,
        "held": held, "ref_held": refHeld, "held_equal": held == refHeld,
        "named": s < r.named.count ? r.named[s][L] : [], "ref_named": s < rn.count ? rn[s][L] : [],
        "only_here_in_ref_held": ours.map { refHeld.contains($0) }, "only_ref_in_held": theirs.map { held.contains($0) },
    ]
    if r.mode == "exact" {
        c["cause"] = "exact mode: the true top-k of the gate differs (gate scores; the choice does not depend on what is in memory)"
    } else if held == refHeld {
        c["cause"] = "gate order: both runs held the same experts at this layer; the masked top-k ranked them differently (gate scores)"
    } else if let s2 = (0...s).first(where: { $0 < r.named.count && $0 < rn.count && r.named[$0][L] != rn[$0][L] }) {
        c["names_first_diff_step"] = s2
        c["names_at_that_step"] = r.named[s2][L]
        c["ref_names_at_that_step"] = rn[s2][L]
        c["cause"] = "different names: the routers named different experts for layer \(L) at step \(s2), so the held sets differ"
    } else {
        c["cause"] = "held sets differ with identical names for this layer up to this step (cache policy)"
    }
    return c
}

/// The first (step, layer) whose released experts differ from the other run's, and how many steps differ at all.
public func tfFirstRoutingDiff(_ r: TFRun) -> (step: Int, layer: Int, ids: [Int], refIds: [Int], steps: Int)? {
    guard let rr = r.refReleased else { return nil }
    let K = r.topK
    var first: (Int, Int, [Int], [Int])? = nil
    var steps = 0
    for s in 0..<min(rr.count, r.released.count) where rr[s] != r.released[s] {
        steps += 1
        if first == nil {
            let a = r.released[s], b = rr[s]
            let L = (0..<(a.count / K)).first { Array(a[($0 * K)..<(($0 + 1) * K)]) != Array(b[($0 * K)..<(($0 + 1) * K)]) } ?? 0
            first = (s, L, Array(a[(L * K)..<((L + 1) * K)]), Array(b[(L * K)..<((L + 1) * K)]))
        }
    }
    guard let f = first else { return nil }
    return (f.0, f.1, f.2, f.3, steps)
}

/// The JSON written by the CLI (--tf-out) and the app (Documents/tf-<cap>.json); `meta` adds the run's settings.
public func tfReport(_ r: TFRun, meta: [String: Any]) -> [String: Any] {
    func d(_ x: Float?) -> Any { x.map { Double($0) } ?? NSNull() }
    var j = meta
    j["format"] = "phone-store teacher-forced v1: positions[i] = logits after prompt_ids + ids[0..<i]; top = top-K (id, logit) sorted desc"
    j["prompt_ids"] = r.promptIds
    j["ids"] = r.ids
    j["k"] = r.k
    j["prefill_s"] = r.prefillSec
    j["seconds"] = r.seconds
    j["split"] = r.split
    j["summary"] = tfSummary(r)
    j["released"] = r.released
    j["released_format"] = "released[s]: step s (prompt tokens, then one per fed id) = layers x top-k expert ids, each layer sorted; position i's logits came from step prompt_ids.count - 1 + i"
    j["named"] = r.named
    j["held"] = r.held
    j["named_held_format"] = "named[s][L]: the ids the routers named for layer L at step s (sorted; [] when not named); held[s][L]: the experts layer L held when its released choice was made (sorted; prefetch: the masked top-k's table, landed or in flight; exact: before that layer's misses were read)"
    j["mode"] = r.mode
    j["positions"] = r.positions.map { p -> [String: Any] in
        var e: [String: Any] = ["i": p.i, "ref_next": p.refNext, "argmax": p.argmax, "match": p.match, "margin": Double(p.margin),
                                "ref_logit": Double(p.refLogit), "gap": Double(p.top[0].1 - p.refLogit),
                                "top": p.top.map { [$0.0, Double($0.1)] as [Any] }]
        if r.hasRef { e["ref_max_abs"] = d(p.refMaxAbs); e["ref_argmax"] = p.refArgmax ?? -1; e["ref_margin"] = d(p.refMargin) }
        if let n = p.routingDiffLayers { e["routing_diff_layers"] = n }
        return e
    }
    return j
}

/// One console line per run, then one per position whose argmax differs from the fed id (with the margins).
public func tfConsoleLines(_ r: TFRun, label: String) -> [String] {
    let s = tfSummary(r)
    func f(_ x: Any?) -> String { (x as? Double).map { String(format: "%.4f", $0) } ?? "\(x ?? "n/a")" }
    var head = "TF \(label) positions=\(s["positions"]!) matches=\(s["matches"]!) mismatches=\(s["mismatches"]!)"
        + " first_mismatch=\(s["first_mismatch"] is NSNull ? "none" : "\(s["first_mismatch"]!)")"
        + " margin<0.05=\(s["margin_below_0.05"]!) margin<0.1=\(s["margin_below_0.1"]!) margin<0.5=\(s["margin_below_0.5"]!)"
        + " min_margin=\(f(s["min_margin"])) at \(s["min_margin_position"]!)"
    if r.hasRef {
        head += " ref_max_abs(top\(r.k)) max=\(f(s["ref_max_abs_max"])) at \(s["ref_max_abs_max_position"]!) mean=\(f(s["ref_max_abs_mean"]))"
    }
    head += String(format: " prefill=%.1fs %.1fs %.2f tok/s", r.prefillSec, r.seconds, Double(r.positions.count) / max(1e-9, r.seconds))
    if r.stopped { head += " STOPPED at \(r.positions.count)" }
    var lines = [head]
    if r.refReleased != nil {
        if let fr = tfFirstRoutingDiff(r) {
            lines.append("TF ROUTING first released-expert difference vs the ref run: step \(fr.step) (position \(fr.step - (r.promptIds.count - 1))), layer \(fr.layer): \(fr.ids) vs \(fr.refIds); \(fr.steps) step(s) differ")
        } else {
            lines.append("TF ROUTING released experts identical to the ref run at every step (\(r.released.count))")
        }
    }
    if let c = tfRoutingCause(r) {
        lines.append("TF CAUSE step \(c["step"]!) layer \(c["layer"]!): \(c["cause"]!); held \(c["held"]!) vs ref \(c["ref_held"]!); named \(c["named"]!) vs ref \(c["ref_named"]!)")
    } else if r.refReleased != nil && r.refHeld == nil {
        lines.append("TF CAUSE n/a: the ref run's file has no named/held sets")
    }
    for p in r.positions where !p.match || (p.refArgmax != nil && p.refArgmax != p.argmax) {
        let gap = p.top[0].1 - p.refLogit
        var l = String(format: "TF DIFF pos=%d fed=%d argmax=%d margin=%.4f gap(top1-fed)=%.4f", p.i, p.refNext, p.argmax, p.margin, gap)
        if let m = p.refMaxAbs {
            l += String(format: " | ref run: argmax=%d margin=%.4f max_abs(top%d)=%.4f", p.refArgmax ?? -1, p.refMargin ?? .nan, r.k, m)
        }
        if let n = p.routingDiffLayers { l += " routing_diff_layers=\(n)" }
        lines.append(l)
    }
    return lines
}
