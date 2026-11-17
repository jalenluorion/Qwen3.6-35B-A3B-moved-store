// Batched prefill (2026-11-06, docs/G4-runtime.md "Batched prefill"): the prompt in chunks of C positions, layer by layer.
//
// Per chunk and layer: the attention / GatedDeltaNet, the norms, the released gate, the shared expert run for all C
// positions at once (MLX's batched ops, Model.swift *Seq; the fused kernels of Kernels.swift stay the decode path); the
// released top-8 of every position comes to the host in one read-back; the store plans the layer (Store.chunkPlan: the
// policy applied token by token, then which records to read into which slots, in one or more passes) and all of a pass's
// reads are queued at once (Store.setBoost: up to prefillReaders reads in flight); the routed experts of every
// (position, expert) pair of the pass run as one gather_qmm over the slab. No lag-4 router runs during a chunk: the
// released gate's choice for all C positions is known before any expert is read, so there is nothing to predict. What a
// router needs from the prompt (h_prev, the MoE input of the last position per layer) is left as step() would leave it.
//
// Arithmetic: every position uses its released top-8 (exact mode's choice; prefetch mode's token-by-token prefill takes
// the top-8 among the experts held). The batched kernels round differently from the single-position ones, so the
// logits after a chunked prefill are not bit-identical to the token-by-token path (verified by teacher forcing).
import Foundation
import MLX
import MLXFast
import MLXNN

public struct PrefillCounters {
    public var chunks = 0, passes = 0, union = 0, present = 0, reads = 0, demoted = 0, tokens = 0
    public var readWait = 0.0  // seconds blocked on a pass's reads
    public var gpuWait = 0.0  // seconds blocked on the GPU: the released top-8 read-back and the waits between passes
    public var plan = 0.0  // host: the policy replay and the pass plan
    /// Read-ahead (2026-11-07): records queued, landed before their layer's plan, used by the chunk (in its union or kept),
    /// and the union experts the routers named (named_in_union / union = the names' recall over the chunk's union).
    public var aheadIssued = 0, aheadLanded = 0, aheadUsed = 0, aheadNamed = 0, aheadNamedInUnion = 0
    /// Lazy re-land: held experts absent from the slab when their layer was planned, and of those the ones read.
    public var heldAbsent = 0, heldAbsentRead = 0
    /// Chunk sizes (2026-11-07, auto chunk): the largest chunk fed, and the activation bytes PrefillMemory predicted for it.
    public var chunkMax = 0
    public var actEstimate = 0.0
    /// The pass planner (2026-11-07, Store.chunkPlan v2): layers planned, the most passes in one layer, the GPU fences
    /// before a pass's reads, the read-backs of held experts whose slots the passes used (restores), the seconds the
    /// engine waited for a layer's commit (its restores), the planner version; and the read depth while the chunks ran
    /// (Store.depthStats): seconds, time integrals of the reader threads inside a read and of the records in flight, and
    /// the seconds with every reader thread inside a read.
    public var layers = 0, passMaxLayer = 0, fences = 0, restores = 0, planVersion = 0
    public var commitWait = 0.0
    public var depthSec = 0.0, depthActive = 0.0, depthRecords = 0.0, depthSat = 0.0
    /// Where the chunk's time goes and how busy the readers are meanwhile (seconds, and the time integral of the reader
    /// threads inside a read), per phase: mixer = from a layer's start to its released top-8 on the host (attention or
    /// GatedDeltaNet, norms, gate, shared expert, read-ahead queued), plan = the previous layer's commit and this layer's
    /// plan, wait = blocked on a pass's reads, fence = blocked on the GPU before a pass's reads, host = the rest.
    public static let phaseNames = ["mixer", "plan", "wait", "fence", "host"]
    public var phaseSec = [Double](repeating: 0, count: 5), phaseActive = [Double](repeating: 0, count: 5)
    public var dict: [String: Double] {
        var d: [String: Double] = [
            "prefill_chunks": Double(chunks), "prefill_passes": Double(passes), "prefill_union_experts": Double(union),
            "prefill_present_experts": Double(present), "prefill_records_read": Double(reads), "prefill_demoted": Double(demoted),
            "prefill_read_wait_ms": 1000 * readWait, "prefill_gpu_wait_ms": 1000 * gpuWait, "prefill_plan_ms": 1000 * plan]
        d["prefill_ahead_issued"] = Double(aheadIssued)
        d["prefill_ahead_landed"] = Double(aheadLanded)
        d["prefill_ahead_used"] = Double(aheadUsed)
        d["prefill_ahead_named"] = Double(aheadNamed)
        d["prefill_ahead_named_in_union"] = Double(aheadNamedInUnion)
        d["prefill_held_absent"] = Double(heldAbsent)
        d["prefill_held_absent_read"] = Double(heldAbsentRead)
        d["prefill_chunk_max"] = Double(chunkMax)
        d["prefill_act_est_mb"] = actEstimate / 1e6
        d["prefill_plan_version"] = Double(planVersion)
        d["prefill_passes_per_layer"] = layers > 0 ? Double(passes) / Double(layers) : 0
        d["prefill_passes_layer_max"] = Double(passMaxLayer)
        d["prefill_reads_per_pass"] = passes > 0 ? Double(reads) / Double(passes) : 0
        d["prefill_fences"] = Double(fences)
        d["prefill_restores"] = Double(restores)
        d["prefill_commit_wait_ms"] = 1000 * commitWait
        d["prefill_readers_active_mean"] = depthSec > 0 ? depthActive / depthSec : 0
        d["prefill_records_in_flight_mean"] = depthSec > 0 ? depthRecords / depthSec : 0
        d["prefill_readers_saturated_frac"] = depthSec > 0 ? depthSat / depthSec : 0
        for (i, n) in PrefillCounters.phaseNames.enumerated() {
            d["prefill_phase_\(n)_ms"] = 1000 * phaseSec[i]
            d["prefill_phase_\(n)_readers"] = phaseSec[i] > 0 ? phaseActive[i] / phaseSec[i] : 0
        }
        return d
    }
}

extension Engine {
    /// Feed `tokens` after what was fed: chunks of opt.prefillChunk positions (0 or 1: step() per token). `done(n)` is
    /// called after every chunk (or token) with the tokens fed so far. Returns the last token's next-token logits
    /// (submitted, not waited for).
    /// `layer(f)`: inside a chunk, after each layer, f = tokens before the chunk + C x (layers done / layers) (a UI's
    /// progress; the state is mid-chunk then).
    /// With EngineOptions.prefillAutoBytes > 0 (and no `chunk` given) each chunk's size is autoChunk's: the largest whose
    /// activations fit the budget at the current clock, the tokens left split evenly over as many chunks as that needs.
    public func prefill(_ tokens: ArraySlice<Int>, chunk: Int? = nil, layer: (Double) -> Void = { _ in }, done: (Int) -> Void = { _ in }) -> MLXArray {
        let c = chunk ?? opt.prefillChunk
        let auto = chunk == nil && opt.prefillAutoBytes > 0
        var logits = MLXArray(0)
        var n = 0
        if c <= 1 {
            for tok in tokens {
                logits = step(tok)
                n += 1
                done(n)
            }
            return logits
        }
        var i = tokens.startIndex
        while i < tokens.endIndex {
            let j = i + (auto ? autoChunk(left: tokens.endIndex - i) : min(c, tokens.endIndex - i))
            let n0 = n, cn = j - i
            pf.chunkMax = max(pf.chunkMax, cn)
            pf.actEstimate = max(pf.actEstimate, prefillMemory.bytes(chunk: cn, keys: t + cn))
            logits = prefillChunk(Array(tokens[i..<j])) { l in layer(Double(n0) + Double(cn) * Double(l) / Double(self.NL)) }
            n += j - i
            done(n)
            i = j
        }
        return logits
    }

    public func prefill(_ tokens: [Int], chunk: Int? = nil, done: (Int) -> Void = { _ in }) -> MLXArray {
        prefill(tokens[...], chunk: chunk, done: done)
    }

    /// The activation model of this engine's chunks (its shapes; the read-ahead's extra rows when it runs).
    public var prefillMemory: PrefillMemory {
        PrefillMemory(cfg: res.cfg, readAhead: routers != nil && opt.prefillReadAhead != 0)
    }

    /// Auto chunk: `left` tokens still to feed after the `t` fed. The largest chunk whose activations fit
    /// opt.prefillAutoBytes with its keys (t + chunk) is cmax; the tokens left go in ceil(left / cmax) chunks of equal size
    /// (a 600-token segment at cmax 512: 300 + 300, not 512 + 88: the same passes over the store, less memory).
    public func autoChunk(left: Int) -> Int {
        let cmax = prefillMemory.largestChunk(budget: opt.prefillAutoBytes, offset: t, limit: left)
        let k = (left + cmax - 1) / cmax
        return (left + k - 1) / k
    }

    /// One chunk: tokens at clocks t + 1 ... t + C. Leaves the state step() leaves after the last of them (KV and
    /// GatedDeltaNet caches, h_prev, the token clock, the store's held set and policy history).
    /// `dry` (the kernel warm-up, warmBatchedKernels): the same op graph on the warm-up's own throwaway caches, with no
    /// store call at all (nothing planned, read, evicted or committed: each pair's slot is a slot of the slab as it is,
    /// whatever it holds) and none of the engine's state touched (h_prev, the clock, the sampler, the last logits).
    func prefillChunk(_ toks: [Int], layerDone: (Int) -> Void = { _ in }, dry: DryChunkState? = nil) -> MLXArray {
        tEnter = now()
        let C = toks.count
        let t0 = t
        let cfg = res.cfg
        let kvs = dry?.kv ?? kv, gdns = dry?.gdnc ?? gdnc
        // the store belongs to this chunk: the namer's jobs applied, no read in flight, no earlier GPU work running
        let tw = blockedBegin()
        namer?.drain()
        store.waitAllLanded()
        if let lg = lastLogits { eval(lg) }
        pf.gpuWait += now() - tw
        blockedEnd()
        if dry == nil {
            for tok in toks { sampler?.append(tok) }
            if collectReleased { for p in 0..<C { collectT.append(t0 + 1 + p) } }
        }
        let e0 = stacked(toks.map { emb.row($0) })  // [C, H]
        var h = e0.reshaped([1, C, H])
        var hPrevNext = hPrev
        var relRows: [[[Int]]] = []  // per layer, per position (teacher forcing)
        if dry == nil { store.setBoost(true) }
        let depth0 = dry == nil ? store.depthStats : (0, 0, 0, 0)
        var phT = now(), phA = depth0.1
        func phase(_ k: Int) {  // the time since the last mark belongs to phase k
            guard dry == nil else { return }
            let d = store.depthStats, t = now()
            pf.phaseSec[k] += t - phT; pf.phaseActive[k] += d.active - phA
            phT = t; phA = d.active
        }
        // a layer's commit (Store.chunkCommit: the held set after the chunk, landed) waits for its restores; it runs after
        // the next layer's mixer and gate, so they are read while those compute
        var pendingCommit: ChunkLayerPlan? = nil, pendingL = -1
        func commitPending() {
            guard let pc = pendingCommit else { return }
            let tc = blockedBegin()
            store.chunkCommit(pendingL, pc, t: t0 + C)
            blockedEnd()
            let w = now() - tc
            pf.commitWait += w; pf.readWait += w; tm.readWait += w
            if !waitByLayer.isEmpty { waitByLayer[pendingL] += w }
            pendingCommit = nil
        }
        // Read-ahead (2026-11-07): the lag-4 routers name, for every position of the chunk, the experts of layer L + 4 from
        // layer L's MoE input (layers 1-3: the early routers, from the embeddings), and the most named ones that are not in
        // the slab are queued at low priority into that layer's slots (Store.issueReadAhead), so the drive keeps reading
        // while layers L ... L + 3 compute. The released gate still decides what each position computes; the names only
        // change which records the layer's plan then finds in the slab (never what is held after the chunk).
        let raOn = dry == nil && routers != nil && opt.prefillReadAhead != 0
        func budget(_ L: Int) -> Int { opt.prefillReadAhead > 0 ? min(opt.prefillReadAhead, caps[L]) : caps[L] }
        let embN: MLXArray? = raOn ? fwd.routerEmbNorm(e0) : nil
        func hPrevRows(_ Lr: Int, _ xf: MLXArray?) -> MLXArray {  // the routers' h_prev per position (see readAheadShift)
            let h0 = hPrev[Lr].reshaped([1, H]).asType(.float16)
            if C == 1 { return h0 }
            if opt.readAheadShift, let xf { return concatenated([h0, xf[0..<(C - 1)].asType(.float16)], axis: 0) }
            return broadcast(h0, to: [C, H])
        }
        func names(_ Lr: Int, _ xf: MLXArray?) -> MLXArray {
            let r = routers!.layers[Lr]
            return argPartition(-fwd.routeLogits(r, embN: embN!, hPrev: hPrevRows(Lr, xf), hLag: r.lagged ? xf! : nil), kth: M - 1, axis: -1)[.ellipsis, ..<M]
        }
        // diagnostics (PS_MEMTRACE=1): MLX's active memory and the peak since the last mark, at each layer's stages
        // (synchronous there: the layer's h is evaluated), the peak reset after each line
        let memTrace = dry == nil && ProcessInfo.processInfo.environment["PS_MEMTRACE"] != nil
        func mt(_ L: Int, _ what: String) {
            guard memTrace else { return }
            print(String(format: "MEMTRACE C=%d L%02d %@: active %.1f MB, peak since last %.1f MB", C, L, what,
                         Double(Memory.activeMemory) / 1e6, Double(Memory.peakMemory) / 1e6))
            Memory.peakMemory = 0
        }
        mt(-1, "chunk start")
        var aheadRows: [Int: [[Int]]] = [:]  // layer -> the routers' names per position (recall stats)
        func splitRows(_ v: [Int], _ m: Int) -> [[Int]] { (0..<(v.count / m)).map { Array(v[($0 * m)..<(($0 + 1) * m)]) } }
        var earlyNames: [(L: Int, a: MLXArray)] = []
        if raOn, let routers {
            for Lr in 1..<NL where !routers.layers[Lr].lagged && budget(Lr) > 0 { earlyNames.append((Lr, names(Lr, nil))) }
        }
        for L in 0..<NL {
            let lw = res.layers[L]
            let xin = MLXFast.rmsNorm(h, weight: lw.inNorm, eps: cfg.eps)
            let r = cfg.isLinear(L) ? fwd.gdnSeq(lw.gdn!, xin, gdns[L]!) : fwd.attentionSeq(lw.attn!, xin, kvs[L]!)
            h = h + r
            let x = MLXFast.rmsNorm(h, weight: lw.postNorm, eps: cfg.eps)
            let xf = x.reshaped([C, H])
            // its own buffer (evaluated below with the layer): a slice would keep the layer's whole [C, H] input alive
            hPrevNext[L] = contiguous(xf[C - 1])
            let gates = softmax(lw.gate(xf), axis: -1, precise: true)
            let inds = argPartition(gates, kth: NE - K, axis: -1)[.ellipsis, (NE - K)...]
            let scores = releasedScores(gates, inds)
            let sh = fwd.sharedSeq(lw, x)
            let xh = expandedDimensions(xf.asType(.float16), axis: -2)  // [C, 1, H]
            var ahead: [(L: Int, a: MLXArray)] = L == 0 ? earlyNames : []
            if raOn, let routers, L + lag < NL, routers.layers[L + lag].lagged, budget(L + lag) > 0 {
                ahead.append((L + lag, names(L + lag, xf)))
            }
            // the layer's carried state evaluated with it: unevaluated, the GatedDeltaNet's conv state (a 3-row copy of
            // the conv input) and h_prev (a row of x) keep their whole [C, ...] inputs alive to the next step, 16 KB +
            // 4 KB per position per layer (2026-11-07 MEMTRACE: 0.33 GB of a 512-token chunk's 0.50 GB)
            var carried: [MLXArray] = [hPrevNext[L]]
            if let g = gdns[L], let cs = g.conv { carried.append(cs) }
            submit([inds, scores, sh, xh] + ahead.map(\.a) + carried)
            let tg = now()
            let flat = readInts(inds)  // the chunk's one read-back per layer
            for (Lr, a) in ahead {  // evaluated with inds: no further wait
                let rr = splitRows(readInts(a), M)
                aheadRows[Lr] = rr
                pf.aheadIssued += store.issueReadAhead(Lr, rows: rr, budget: budget(Lr), overwrite: opt.readAheadOverwrite)
            }
            pf.gpuWait += now() - tg
            phase(0)
            mt(L, "mixer + gate")
            if let dry {  // the warm-up: the pairs' kernels on the slab as it is (Y = yp, or the zeros + scatter of passes)
                let Y = dryRouted(xh, L, flat, C, dry)
                let y = (Y.reshaped([C, K, H]) * expandedDimensions(scores, axis: -1)).sum(axis: -2)
                h = h + (y.reshaped(x.shape).asType(x.dtype) + sh)
                submit([h])
                layerDone(L + 1)
                continue
            }
            let rows = (0..<C).map { Array(flat[($0 * K)..<(($0 + 1) * K)]) }
            if collectReleased { relRows.append(rows) }
            commitPending()
            let tpl = now()
            let plan = store.chunkPlan(L, t0: t0, rows: rows)
            pf.plan += now() - tpl
            phase(1)
            pf.union += plan.union; pf.present += plan.present; pf.reads += plan.readCount; pf.demoted += plan.demoted
            pf.aheadLanded += plan.aheadLanded; pf.aheadUsed += plan.aheadUsed
            pf.heldAbsent += plan.heldAbsent; pf.heldAbsentRead += plan.heldAbsentRead
            if let named = aheadRows[L] {
                let u = Set(flat), n = Set(named.flatMap { $0 })
                pf.aheadNamed += n.count
                pf.aheadNamedInUnion += n.intersection(u).count
            }
            pf.passes += plan.computing
            pf.layers += 1
            pf.passMaxLayer = max(pf.passMaxLayer, plan.computing)
            pf.restores += plan.restores.count
            pf.planVersion = plan.version
            heldSum += plan.finalSlot.count
            asked += C * K
            var Y: MLXArray? = nil
            // the plan's protocol (ChunkLayerPlan): issue[0] now; per pass, its reads, its experts, and before
            // issue[pi + 1] the GPU work of every pass <= fence[pi + 1] (the overwrite fence; with the lagged fence that
            // is the pass before, already done by then)
            let np = plan.passes.count
            let single = plan.computing == 1  // one pass computes every pair: Y = yp in (position, k) order
            // each pass's gathered rows (they read the slots), kept until a fence covers that pass: waiting for them is
            // the fence (the scatter into Y does not read the slab)
            var unfenced: [(j: Int, a: MLXArray)] = []
            // each expert's (position, k) pairs, once per layer (a pass then costs its own pairs, not a scan of all C x K)
            var pairsOf: [Int: [Int]] = [:]
            if !single { for (q, e) in flat.enumerated() { pairsOf[e, default: []].append(q) } }
            store.chunkIssue(L, plan.issue[0])
            for (pi, pass) in plan.passes.enumerated() {
                var w = 0.0
                phase(4)
                _ = blockedBegin()
                store.chunkWait(L, plan.reads[pi], waited: &w)
                blockedEnd()
                phase(2)
                tm.readWait += w
                pf.readWait += w
                if !waitByLayer.isEmpty { waitByLayer[L] += w }
                var pairIdx: [UInt32] = [], tokIdx: [UInt32] = [], slotIdx: [UInt32] = []
                if single {  // every pair, in (position, k) order (Y = yp below)
                    var slotOf: [Int: Int] = [:]
                    for p in pass { slotOf[p.e] = p.slot }
                    for (q, e) in flat.enumerated() {
                        if let s = slotOf[e] {
                            pairIdx.append(UInt32(q)); tokIdx.append(UInt32(q / K)); slotIdx.append(UInt32(s))
                        }
                    }
                } else {  // grouped by expert: the rows are independent, the scatter puts them in place
                    for p in pass {
                        for q in pairsOf[p.e] ?? [] {
                            pairIdx.append(UInt32(q)); tokIdx.append(UInt32(q / K)); slotIdx.append(UInt32(p.slot))
                        }
                    }
                }
                if !pairIdx.isEmpty {
                    let yp = routedPairs(xh, L, MLXArray(tokIdx), MLXArray(slotIdx))  // [P, H]
                    unfenced.append((pi, yp))
                    if single {
                        Y = yp  // every pair, in (position, k) order
                    } else {
                        var Yv = Y ?? MLXArray.zeros([C * K, H], dtype: yp.dtype)
                        Yv[MLXArray(pairIdx)] = yp
                        Y = Yv
                        // the scatter evaluated with its pass: yp is freed then (submitting yp alone kept every pass's
                        // rows alive to the layer's end, C x 8 x 4 KB), and the fence below covers the pass all the same
                        submit([Yv])
                    }
                }
                if !plan.issue[pi + 1].isEmpty {
                    let f = plan.fence[pi + 1]
                    if let g = unfenced.last(where: { $0.j <= f }) {  // these reads write slots passes <= f read
                        phase(4)
                        let tg2 = blockedBegin()
                        eval(g.a)
                        tm.syncWait += now() - tg2
                        tm.syncs += 1
                        pf.gpuWait += now() - tg2
                        blockedEnd()
                        phase(3)
                        unfenced.removeAll { $0.j <= f }
                    }
                    pf.fences += 1
                    store.chunkIssue(L, plan.issue[pi + 1])
                }
            }
            pendingCommit = plan
            pendingL = L
            if memTrace { eval(Y!); mt(L, "routed passes (\(np))") }
            let y = (Y!.reshaped([C, K, H]) * expandedDimensions(scores, axis: -1)).sum(axis: -2)
            h = h + (y.reshaped(x.shape).asType(x.dtype) + sh)
            submit([h])  // the GPU runs the experts while the host builds the next layer
            if memTrace { eval(h); mt(L, "layer end") }
            layerDone(L + 1)
            phase(4)
        }
        let hl = h[0..., (C - 1)..., 0...]
        let logits = res.lmHead(MLXFast.rmsNorm(hl, weight: res.norm, eps: cfg.eps))[0, -1]
        submit([logits])
        if dry != nil { return logits }  // the warm-up's: the engine's state is as it was
        commitPending()  // the last layer's (the head computes meanwhile)
        store.setBoost(false)
        let depth1 = store.depthStats
        pf.depthSec += depth1.sec - depth0.0; pf.depthActive += depth1.active - depth0.1
        pf.depthRecords += depth1.records - depth0.2; pf.depthSat += depth1.saturated - depth0.3
        if collectReleased {  // per position, per layer: released = held (every released expert is computed in a chunk)
            for p in 0..<C {
                for L in 0..<NL {
                    releasedDev.append(MLXArray(relRows[L][p].map { UInt32($0) }))
                    heldLog.append(relRows[L][p].sorted())
                }
            }
        }
        t = t0 + C
        hPrev = hPrevNext
        lastLogits = logits
        nextArg = nil
        pf.chunks += 1
        pf.tokens += C
        tm.encode += now() - tEnter
        return logits
    }

    /// The routed experts of P (position, expert) pairs: x [C, 1, H] float16, the pairs' positions and slots ([P]).
    /// Three gather_qmm over layer L's slab as routed() runs them, with one row per pair. Returns [P, H].
    func routedPairs(_ xh: MLXArray, _ L: Int, _ tok: MLXArray, _ slots: MLXArray) -> MLXArray {
        let W = store.W[L], S = store.S[L], B = store.Bi[L]
        let gt = gatherQuantizedMM(xh, W[0], scales: S[0], biases: B[0], lhsIndices: tok, rhsIndices: slots, transpose: true, groupSize: G, bits: 4)
        let up = gatherQuantizedMM(xh, W[1], scales: S[1], biases: B[1], lhsIndices: tok, rhsIndices: slots, transpose: true, groupSize: G, bits: 4)
        let P = tok.dim(0)
        let own = MLXArray(Array(0..<UInt32(P)))
        return gatherQuantizedMM(silu(gt) * up, W[2], scales: S[2], biases: B[2], lhsIndices: own, rhsIndices: slots, transpose: true, groupSize: G, bits: 4)
            .squeezed(axis: -2)
    }
}

/// The kernel warm-up's throwaway state (Engine.warmBatchedKernels): its own KV and GatedDeltaNet caches, discarded after.
final class DryChunkState {
    let kv: [KVCache?]
    let gdnc: [GDNCache?]
    /// Per layer: the routed pairs of a chunk run as one pass (Y = yp: every pair, as a chunk whose union fits the slots)
    /// or as two passes of one position each (zeros + scatter, as a chunk of several passes; the other pairs stay 0, so
    /// a large warm-up chunk does not run 8 x C expert products). Even layers one pass when C <= onePassMax, odd layers two.
    let onePassMax = 8
    init(layers: Int, isLinear: (Int) -> Bool) {
        kv = (0..<layers).map { isLinear($0) ? nil : KVCache() }
        gdnc = (0..<layers).map { isLinear($0) ? GDNCache() : nil }
    }
}

/// What the kernel warm-up did (probe.json, the load log, the settings sheet).
public struct KernelWarmup: Sendable {
    public let seconds: Double
    public let chunks: [Int]  // the dry chunks' sizes, in order (one throwaway state)
    /// Seconds of the dry decode step (0: not run, EngineOptions.decodeWarmup off), included in `seconds`.
    public var decodeSeconds = 0.0
    public var tokens: Int { chunks.reduce(0, +) }
    public var line: String {
        String(format: "warm-up: batched kernels compiled in %.2f s (dry chunks %@ = %d positions%@ on a throwaway state; no reads, store and prefix cache untouched)",
               seconds, chunks.map(String.init).joined(separator: "/"), tokens,
               decodeSeconds > 0 ? String(format: " + 1 dry decode step %.2f s", decodeSeconds) : "")
    }
}

extension Engine {
    /// The routed experts of a dry chunk (prefillChunk dry): slot = expert id mod the layer's slots, no read.
    func dryRouted(_ xh: MLXArray, _ L: Int, _ flat: [Int], _ C: Int, _ dry: DryChunkState) -> MLXArray {
        let nSlots = store.W[L][0].dim(0)
        let groups: [[Int]] = (C <= dry.onePassMax && L % 2 == 0) ? [Array(0..<C)] : [[0], [C - 1]]
        var Y: MLXArray? = nil
        for g in groups {
            let q = g.flatMap { p in (0..<K).map { p * K + $0 } }
            let yp = routedPairs(xh, L, MLXArray(q.map { UInt32($0 / K) }), MLXArray(q.map { UInt32(flat[$0] % nSlots) }))
            if groups.count == 1 {
                Y = yp
            } else {
                var Yv = Y ?? MLXArray.zeros([C * K, H], dtype: yp.dtype)
                Yv[MLXArray(q.map { UInt32($0) })] = yp
                Y = Yv
                submit([yp])
            }
        }
        return Y!
    }

    /// The batched prefill's first-use cost moved to load (docs/G4-runtime.md "Batched prefill", "Kernel warm-up"): the
    /// first chunk after an install builds every Metal pipeline the chunked path uses (iPhone 15 Plus: ~4.7 s inside the
    /// first message's prefill). This runs dry chunks (prefillChunk dry) of synthetic tokens through all layers on one
    /// throwaway state, in sizes that reach every kernel variant a real chunk can pick: the quantized matmuls switch
    /// between qmv, qmv_wide, qmm_splitk and qmm by the number of positions (and the N tiles), SDPA between the vector
    /// kernel (<= 8 queries) and the full one, whose pipeline is specialised on the query and key lengths' alignment
    /// (Q % 64 or 32, K % 32): 128 at offset 0 (aligned, aligned), 37 (no, no), 64 (yes, no), 27 (no, yes: 256 keys),
    /// 5 (vector, qmv_wide), 1 (qmv). Sizes above the engine's chunk are dropped (the chunk itself is used instead of
    /// 128). Nothing of the engine's state, the store (held set, policy, stats, readers) or the prefix cache changes; the
    /// engine's timers and counters are restored. Returns the wall time (compile + the dry chunks' GPU work).
    /// With EngineOptions.decodeWarmup, one dry decode step follows on the same throwaway state (step(dry:): the fused
    /// decode kernels, the routers, the masked top-k, the head start's argmax and, given `sampling`, the reply sampler's
    /// graph on a throwaway Sampler), so the reply's first generated token does not compile them either.
    public func warmBatchedKernels(sampling: SamplingParams? = nil) -> KernelWarmup {
        let c = opt.prefillChunk > 1 ? opt.prefillChunk : 128  // forced on with token-by-token prefill: the default chunk's shapes
        var sizes: [Int] = [c]
        for s in [37, 64, 27, 5, 1] where s < c { sizes.append(s) }
        let saved = (pf, tm, asked, heldSum, heldN, tEnter)
        let t0 = now()
        let dry = DryChunkState(layers: NL, isLinear: res.cfg.isLinear)
        var off = 0
        for n in sizes {
            let toks = (0..<n).map { 1000 + 37 * ((off + $0) % 256) }  // ordinary text ids, all different within a chunk
            off += n
            let lg = prefillChunk(toks, dry: dry)
            eval(lg)
        }
        var decodeSec = 0.0
        if opt.decodeWarmup {
            let td = now()
            let ws = sampling.flatMap { $0.isPlainArgmax ? nil : Sampler($0) }
            let lg = step(1000 + 37 * (off % 256), dry: dry, warmSampler: ws)
            eval(lg)
            decodeSec = now() - td
        }
        let dt = now() - t0
        (pf, tm, asked, heldSum, heldN, tEnter) = saved
        var w = KernelWarmup(seconds: dt, chunks: sizes)
        w.decodeSeconds = decodeSec
        return w
    }
}

/// The activation memory of one batched prefill chunk (2026-11-07, docs/G4-runtime.md "Batched prefill", "Chunks as large
/// as memory allows"): what a chunk of C positions with `keys` attention keys (the tokens before it + C) allocates beyond
/// the engine's standing state (weights, slabs, the KV and GatedDeltaNet caches, MLX's buffer cache), from the shapes.
/// Activations are float16 (2 B).
///   Per position: the chunk's residual stream, its embeddings and the routers' normalized embeddings (3 x H; the last
///   only with read-ahead), the layer's input norm (H), and the largest live set inside a layer: the MoE's (Y, the K
///   routed outputs of every position, Y x scores before the sum over K, the MoE input and its float16 copy: (2K + 2) x
///   H), the GatedDeltaNet's (in_proj qkv, its concatenation with the conv state, conv1d's output and its silu: 4 x
///   convDim, + z: valueDim) or the attention's (q_proj with its gate 2 x heads x headDim, the normed and the rotated
///   queries, the output: 5 x heads x headDim).
///   Per (position, key): MLX's SDPA with head dim 256 and more than 8 queries is the unfused fallback
///   (mlx/backend/metal/scaled_dot_product_attention.cpp use_fallback), which materializes the scores [heads, C, keys]
///   in float16, the causal mask (bool), and the masked scores / the softmax: 2 x heads x 2 B + 1 B.
///   Fixed: the head's logits, the gate's per-layer arrays and what the pipelined layers keep (measured below).
/// The GatedDeltaNet kernel keeps its state in registers over the chunk (one state out, none per position), and
/// conv1d's carried state is a copy (contiguous of a 3-row slice), so neither grows with C beyond the terms above.
/// Measured (Mac, cap 19 and 48, 2026-11-07, native/ref/bigchunk/out; the CLI's prefill_act_mb = MLX's peak during the
/// prefill over its active memory before it, minus the standing state a cold prefill creates: the GatedDeltaNet states
/// 63 MB and the KV cache 20 KB per token): 57 / 81 / 131 / 246 MB at chunks 128 / 256 / 512 / 1024 from an empty
/// context, 153 MB for 512 positions after 512. The pipelined layers (layer L's routed experts still in flight while L + 1
/// is built and its mixer runs) roughly double the per-position live set of the shapes above, so the per-position term is
/// scaled by `scale` (2.1: 189 KB) and `fixed` is the measured 30 MB; the per-key term is the shapes' (65 B; measured
/// ~80). Every measured point is at or under the model (docs/G4-runtime.md "Chunks as large as memory allows").
public struct PrefillMemory: Sendable {
    public let perPosition: Double
    public let perQueryKey: Double
    public let fixed: Double
    public static let scale = 2.1
    /// The smallest chunk auto picks however tight the budget (still batched: the bytes per chunk barely change below it).
    public static let minChunk = 16
    public let terms: [(String, Double)]

    public init(hidden H: Int, heads: Int, headDim: Int, convDim: Int, valueDim: Int, topK K: Int, readAhead: Bool) {
        let b = 2.0
        let stream = Double((readAhead ? 3 : 2) * H) * b
        let xin = Double(H) * b
        let moe = Double((2 * K + 2) * H) * b
        let gdn = Double(4 * convDim + valueDim) * b
        let attn = Double(5 * heads * headDim) * b
        perPosition = stream + xin + max(moe, gdn, attn)
        perQueryKey = Double(heads) * 2 * b + 1
        fixed = 30e6
        terms = [("stream", stream), ("xin", xin), ("moe", moe), ("gdn", gdn), ("attn", attn), ("per_qk", perQueryKey)]
    }

    public init(cfg: TextConfig, readAhead: Bool) {
        self.init(hidden: cfg.hidden, heads: cfg.heads, headDim: cfg.headDim, convDim: cfg.convDim, valueDim: cfg.valueDim,
                  topK: cfg.topK, readAhead: readAhead)
    }

    /// Qwen3.6-35B-A3B's shapes (the app's budget before the model is loaded: Probe).
    public static func qwen36(readAhead: Bool = true) -> PrefillMemory {
        PrefillMemory(hidden: 2048, heads: 16, headDim: 256, convDim: 8192, valueDim: 4096, topK: 8, readAhead: readAhead)
    }

    /// Bytes for a chunk of `chunk` positions with `keys` keys in its attention.
    public func bytes(chunk: Int, keys: Int) -> Double {
        fixed + Double(chunk) * (Self.scale * perPosition + perQueryKey * Double(keys))
    }

    /// The largest chunk (<= limit, >= minChunk or limit) whose bytes fit `budget` after `offset` tokens fed.
    public func largestChunk(budget: Double, offset: Int, limit: Int) -> Int {
        let a = Self.scale * perPosition + perQueryKey * Double(offset), q = perQueryKey
        let room = max(0, budget - fixed)
        let c = q > 0 ? (-a + (a * a + 4 * q * room).squareRoot()) / (2 * q) : room / a
        return min(limit, max(Self.minChunk, Int(c.rounded(.down))))
    }
}

extension Store {
    /// The pass schedule under test, no model (phone-store-run --chunk-plan-test; this is a policy-only Store: no slabs,
    /// no files, chunkIssue sets slotHas at once). Chunks of `sizes`, one after the other, routing drawn from a skewed
    /// distribution (seeded, a different permutation per layer: a 128-token chunk names ~150-200 of 256 experts per
    /// layer, 512 and more nearly all); each layer planned by chunkPlan (the version in force: PS_CHUNK_PLAN) and its
    /// passes run in Engine.prefillChunk's order (ChunkLayerPlan: issue[0]; per pass: its waits, its GPU work, and after
    /// the fence on the passes <= fence[pass + 1], issue[pass + 1]). A read lands the moment it is queued here (the
    /// earliest it could overwrite its slot). Checks: no read goes into a slot that a submitted pass the fence does not
    /// cover computes with (the overwrite fence: the GPU may still be reading it); every pass finds its experts' records
    /// in their slots (a read queued before a later pass that still needs the slot's older record); every record a pass
    /// computes with that was read in the chunk is in its waits; every read is waited for by a later-or-same pass or is a
    /// restore; no expert is read twice; every pair is computed once; the held set after the chunk is where the plan put
    /// it (restores included). Returns a line per chunk (passes, union, reads per layer and per pass, restores, fences, the
    /// host's plan and pair-grouping time) and the problems found.
    public func chunkPassTest(sizes: [Int]) -> (lines: [String], problems: Int) {
        precondition(opt.policyOnly, "chunkPassTest needs a policy-only Store")
        var out: [String] = []
        let NE = 256, K = 8, NLt = NL
        var rng = UInt64(12345)
        func rnd() -> Double { rng = rng &* 6364136223846793005 &+ 1442695040888963407; return Double(rng >> 11) / Double(1 << 53) }
        // expert popularity ~ 1 / (rank + 8): a 128-token chunk names ~150-200 of 256 per layer, 512+ nearly all
        let w = (0..<NE).map { 1.0 / Double($0 + 8) }
        let wsum = w.reduce(0, +)
        let cdf = w.reduce(into: [Double]()) { $0.append(($0.last ?? 0) + $1 / wsum) }
        let perms: [[Int]] = (0..<NLt).map { _ in
            var a = Array(0..<NE)
            for i in stride(from: NE - 1, to: 0, by: -1) { a.swapAt(i, Int(rnd() * Double(i + 1))) }
            return a
        }
        var contents: [[Int: Int]] = Array(repeating: [:], count: NLt)  // per layer: slot -> record, as chunkIssue leaves slotHas
        var t0 = 0, bad = 0
        for C in sizes {
            var passes = 0, union = 0, reads = 0, maxPasses = 0, restores = 0, fences = 0, maxPassReads = 0
            var planSec = 0.0, groupSec = 0.0
            for L in 0..<NLt {
                var rows: [[Int]] = []
                for _ in 0..<C {
                    var r: [Int] = []
                    while r.count < K {
                        let u = rnd()
                        var lo = 0, hi = NE - 1
                        while lo < hi { let m = (lo + hi) / 2; if cdf[m] < u { lo = m + 1 } else { hi = m } }
                        let e = perms[L][lo]
                        if !r.contains(e) { r.append(e) }
                    }
                    rows.append(r)
                }
                for s in 0..<caps[L] where (contents[L][s] ?? -1) != slotHas[L][s] {
                    bad += 1
                    out.append("SLAB: chunk \(C) layer \(L): slot \(s) holds \(slotHas[L][s]), the schedule says \(contents[L][s] ?? -1)")
                }
                let tp = now()
                let plan = chunkPlan(L, t0: t0, rows: rows)
                planSec += now() - tp
                let np = plan.passes.count
                passes += plan.computing; maxPasses = max(maxPasses, plan.computing); union += plan.union; reads += plan.readCount
                restores += plan.restores.count
                maxPassReads = max(maxPassReads, plan.maxPassReads)
                if plan.issue.count != np + 1 || plan.fence.count != np + 1 || plan.reads.count != np {
                    bad += 1
                    out.append("SHAPE: chunk \(C) layer \(L): \(np) passes, \(plan.reads.count) wait lists, \(plan.issue.count) issue lists")
                    continue
                }
                // the slab as the GPU would see it (policy-only: chunkIssue sets slotHas at once, i.e. a read lands the moment
                // it is queued: the earliest it could overwrite a slot)
                var content = contents[L]
                var readNow = Set<Int>()  // experts read in this chunk
                let restoreSet = Set(plan.restores.map(\.e))
                let waitSets = plan.reads.map { Set($0.map(\.e)) }
                var lastUser = [Int](repeating: -1, count: caps[L])  // the last pass submitted that computes with the slot
                func issue(_ i: Int) {  // queued after pass i - 1 is submitted and passes <= fence[i] are done on the GPU
                    for r in plan.issue[i] {
                        if lastUser[r.slot] > plan.fence[i] {
                            bad += 1
                            out.append("FENCE VIOLATION: chunk \(C) layer \(L): read of \(r.e) into slot \(r.slot) at \(i) while pass \(lastUser[r.slot]) (fenced through \(plan.fence[i])) may still compute with it")
                        }
                        if readNow.contains(r.e) { bad += 1; out.append("TWICE: chunk \(C) layer \(L): \(r.e) read twice") }
                        readNow.insert(r.e)
                        if !restoreSet.contains(r.e) && !waitSets[i...].contains(where: { $0.contains(r.e) }) {
                            bad += 1
                            out.append("UNWAITED: chunk \(C) layer \(L): read of \(r.e) issued at \(i), no pass >= \(i) waits for it")
                        }
                        content[r.slot] = r.e
                    }
                    chunkIssue(L, plan.issue[i])
                }
                let tg = now()
                var pairsOf: [Int: [Int]] = [:]
                if np > 1 { for (q, e) in rows.joined().enumerated() { pairsOf[e, default: []].append(q) } }
                var pairs = 0
                groupSec += now() - tg
                issue(0)
                for (pi, pass) in plan.passes.enumerated() {
                    // every expert computes from its record: a read issued before this pass into one of its slots for another
                    // expert would show here (the overwrite fence), and a record read in this chunk must be waited for
                    for p in pass {
                        if content[p.slot] != p.e {
                            bad += 1
                            out.append("WRONG RECORD (fence): chunk \(C) layer \(L) pass \(pi): expert \(p.e) in slot \(p.slot) holds \(content[p.slot] ?? -9)")
                        }
                        if readNow.contains(p.e) && !waitSets[pi].contains(p.e) {
                            bad += 1
                            out.append("NOT WAITED: chunk \(C) layer \(L) pass \(pi): expert \(p.e) read in this chunk, not in the pass's waits")
                        }
                    }
                    let tq = now()
                    for p in pass { pairs += np > 1 ? (pairsOf[p.e]?.count ?? 0) : rows.joined().filter { $0 == p.e }.count }
                    groupSec += now() - tq
                    for p in pass { lastUser[p.slot] = pi }
                    if !plan.issue[pi + 1].isEmpty { fences += 1; issue(pi + 1) }
                }
                if pairs != C * K { bad += 1; out.append("PAIRS: chunk \(C) layer \(L): \(pairs) of \(C * K) computed") }
                for (e, s) in plan.finalSlot where content[s] != e {
                    bad += 1
                    out.append("HELD: chunk \(C) layer \(L): expert \(e) held in slot \(s), which holds \(content[s] ?? -9)")
                }
                chunkCommit(L, plan, t: t0 + C)
                contents[L] = content
            }
            t0 += C
            out.append(String(format: "chunk %d (clock %d): passes %d (%.1f / layer, max %d), union %.1f / layer, reads %d (%.1f / layer, %.1f / pass, max %d), restores %d, fences %d, plan %.1f ms, pair grouping %.2f ms",
                         C, t0, passes, Double(passes) / Double(NLt), maxPasses, Double(union) / Double(NLt), reads,
                         Double(reads) / Double(NLt), Double(reads) / Double(max(1, passes)), maxPassReads, restores, fences,
                         1000 * planSec, 1000 * groupSec))
        }

        return (out, bad)
    }
}
