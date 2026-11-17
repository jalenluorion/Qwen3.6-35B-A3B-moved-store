// Sampling for the native runtime (docs/G4-runtime.md "Prompting and sampling", 2026-11-04).
//
// The order is transformers' (and mlx_run.py sample()'s): penalties on the raw logits, then the temperature, then top-k,
// then top-p and min-p on the tempered distribution of the top-k survivors, then one categorical draw. Everything is one
// lazy MLX graph on the fp32 logits on the GPU, so with the engine's head start (Engine.step) the draw rides in the same
// submission as the logits and the token's only GPU wait is still the read-back of one uint32.
//
// Penalties apply to the tokens fed since the prompt (generated and forced: vLLM's / SGLang's presence and frequency
// penalties, which Qwen's model cards recommend; the prompt's tokens are not penalised, unlike transformers'
// repetition_penalty):
//   repetition r   logit > 0 ? logit / r : logit * r   (CTRL / transformers form), r = 1 off
//   presence p     logit - p                           once per distinct token, p = 0 off
//   frequency f    logit - f * count                    f = 0 off
// The draw for the n-th token of a reply (n = tokens fed since the prompt) uses the key MLXRandom.key(mix(seed, n)), so a
// reply is a pure function of (logits, seed, the tokens so far): reproducible across runs, with or without the head start,
// and independent of what else ran on the device.
//
// Greedy (temperature 0, no penalties) never builds a Sampler: the engine's own argmax path (sampleGreedy) is the
// identity-tested path and stays byte for byte what it was.
import Foundation
import MLX

public struct SamplingParams: Sendable, Equatable, Codable {
    public var temperature: Float
    public var topK: Int  // 0: off
    public var topP: Float  // 1: off
    public var minP: Float  // 0: off (relative to the most likely token after top-k / temperature)
    public var presencePenalty: Float  // 0: off
    public var repetitionPenalty: Float  // 1: off
    public var frequencyPenalty: Float  // 0: off
    public var seed: UInt64

    public init(temperature: Float, topK: Int = 0, topP: Float = 1, minP: Float = 0, presencePenalty: Float = 0,
                repetitionPenalty: Float = 1, frequencyPenalty: Float = 0, seed: UInt64 = 0) {
        self.temperature = temperature; self.topK = topK; self.topP = topP; self.minP = minP
        self.presencePenalty = presencePenalty; self.repetitionPenalty = repetitionPenalty
        self.frequencyPenalty = frequencyPenalty; self.seed = seed
    }

    /// Argmax: the identity-test default of the CLI and the app's autotest.
    public static let greedy = SamplingParams(temperature: 0)
    /// The release's generation_config.json (Qwen3.6-35B-A3B): do_sample, temperature 1.0, top_k 20, top_p 0.95.
    public static let modelDefault = SamplingParams(temperature: 1.0, topK: 20, topP: 0.95)
    /// The Qwen3.6-35B-A3B model card's presets ("We recommend using the following set of sampling parameters"):
    /// thinking mode, general tasks: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, repetition 1.0.
    public static let qwenThinking = SamplingParams(temperature: 1.0, topK: 20, topP: 0.95, presencePenalty: 1.5)
    /// thinking mode, precise coding tasks: temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.
    public static let qwenThinkingCoding = SamplingParams(temperature: 0.6, topK: 20, topP: 0.95)
    /// instruct (non-thinking) mode: temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5.
    public static let qwenNonThinking = SamplingParams(temperature: 0.7, topK: 20, topP: 0.8, presencePenalty: 1.5)

    /// No randomness: the argmax of the (penalised) logits.
    public var isGreedy: Bool { temperature <= 0 }
    public var hasPenalties: Bool { presencePenalty != 0 || repetitionPenalty != 1 || frequencyPenalty != 0 }
    /// The engine's plain argmax path applies (no Sampler is built).
    public var isPlainArgmax: Bool { isGreedy && !hasPenalties }

    /// generation_config.json in `dir` (temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty), each key
    /// falling back to `modelDefault`'s value; `do_sample: false` gives greedy.
    public static func generationConfig(dir: String) -> SamplingParams {
        var p = modelDefault
        guard let d = try? Data(contentsOf: URL(fileURLWithPath: dir + "/generation_config.json")),
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return p }
        func f(_ k: String) -> Float? { (j[k] as? NSNumber).map { $0.floatValue } }
        if let v = f("temperature") { p.temperature = v }
        if let v = (j["top_k"] as? NSNumber)?.intValue { p.topK = v }
        if let v = f("top_p") { p.topP = v }
        if let v = f("min_p") { p.minP = v }
        if let v = f("repetition_penalty") { p.repetitionPenalty = v }
        if let v = f("presence_penalty") { p.presencePenalty = v }
        if let s = j["do_sample"] as? Bool, !s { p.temperature = 0 }
        return p
    }

    /// One line for the startup line, the settings sheet and the JSON.
    public var summary: String {
        if isPlainArgmax { return "greedy (argmax)" }
        var s = isGreedy ? "greedy (argmax of the penalised logits)" : String(format: "temperature %g, top_k %d, top_p %g, min_p %g, seed %llu",
                                                                       temperature, topK, topP, minP, seed)
        if repetitionPenalty != 1 { s += String(format: ", repetition penalty %g", repetitionPenalty) }
        if presencePenalty != 0 { s += String(format: ", presence penalty %g", presencePenalty) }
        if frequencyPenalty != 0 { s += String(format: ", frequency penalty %g", frequencyPenalty) }
        return s
    }

    public var dict: [String: Any] {
        ["temperature": Double(temperature), "top_k": topK, "top_p": Double(topP), "min_p": Double(minP),
         "presence_penalty": Double(presencePenalty), "repetition_penalty": Double(repetitionPenalty),
         "frequency_penalty": Double(frequencyPenalty), "seed": seed, "greedy": isPlainArgmax]
    }
}

/// One reply's sampler: the parameters, the tokens fed since the prompt (for the penalties and the key index).
public final class Sampler {
    public let params: SamplingParams
    public private(set) var history: [Int] = []
    private var counts: [Int: Int] = [:]

    public init(_ params: SamplingParams) { self.params = params }

    /// A token fed after the prompt (sampled or forced). Engine.step calls this for every token it feeds while the
    /// sampler is installed.
    public func append(_ token: Int) {
        history.append(token)
        counts[token, default: 0] += 1
    }

    /// splitmix64 of seed and index: the n-th draw's key.
    static func mix(_ seed: UInt64, _ n: UInt64) -> UInt64 {
        var z = seed &+ (n &+ 1) &* 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// The next token for `logits` ([V] or [1, V], any float dtype) as a lazy uint32 scalar.
    public func token(_ logits: MLXArray) -> MLXArray {
        let p = params
        var lg = logits.asType(.float32).reshaped([-1])
        let V = lg.dim(0)
        if p.hasPenalties && !counts.isEmpty {
            let ids = counts.keys.sorted()
            let idx = MLXArray(ids.map { Int32($0) })
            var v = take(lg, idx)
            if p.repetitionPenalty != 1 { v = which(v .> Float(0), v / p.repetitionPenalty, v * p.repetitionPenalty) }
            if p.presencePenalty != 0 { v = v - p.presencePenalty }
            if p.frequencyPenalty != 0 { v = v - p.frequencyPenalty * MLXArray(ids.map { Float(counts[$0]!) }) }
            lg = putAlong(lg, idx, values: v, axis: 0)
        }
        if p.isGreedy { return argMax(lg).asType(.uint32) }
        lg = lg / p.temperature
        var cand: MLXArray? = nil  // vocabulary ids of the top-k survivors (nil: the whole vocabulary)
        var vals = lg
        if p.topK > 0 && p.topK < V {
            let top = argPartition(-lg, kth: p.topK - 1)[..<p.topK]
            cand = top
            vals = take(lg, top)
        }
        let key = MLXRandom.key(Sampler.mix(p.seed, UInt64(history.count)))
        var local: MLXArray
        if p.topP < 1 || p.minP > 0 {
            let order = argSort(-vals)  // descending
            let sv = take(vals, order)
            let probs = softmax(sv)
            var keep = probs .>= Float(0)
            if p.topP < 1 { keep = logicalAnd(keep, (cumsum(probs) - probs) .< p.topP) }  // the smallest set reaching top-p
            if p.minP > 0 { keep = logicalAnd(keep, probs .>= p.minP * probs[0]) }
            let pick = MLXRandom.categorical(which(keep, sv, MLXArray(-Float.infinity)), key: key)
            local = take(order, pick)
        } else {
            local = MLXRandom.categorical(vals, key: key)
        }
        if let cand { local = take(cand, local) }
        return local.asType(.uint32)
    }
}
