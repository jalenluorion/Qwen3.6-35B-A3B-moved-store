// The chat prompt and the thinking segment (docs/G4-runtime.md "Prompting and sampling", 2026-11-04).
//
// Prompt: the release's own chat template (qwen35/chat_template.jinja = tokenizer_config.json's), rendered by
// swift-transformers' Jinja with add_generation_prompt = true and enable_thinking passed as given (nil: not passed, the
// template's default = thinking on). The ids equal transformers' apply_chat_template for the checked cases
// (docs/template/reference.json, `phone-store-run --template-check`): one user turn with thinking on / off / default, a
// system prompt + two turns whose earlier assistant reply carries a think block (the template strips it), and the
// generation-prompt suffix (thinking on: `<|im_start|>assistant\n<think>\n`; off: `...<think>\n\n</think>\n\n`).
//
// Tools (2026-11-06, docs/TOOLS.md): `ChatPrompt.ids(..., tools:)` passes JSON-schema tool specs to the Jinja render (the
// template writes them into the system turn; messages may carry structured `tool_calls` and the `tool` role), and
// `ToolPrompt.systemContent` renders the same tools system turn into a plain system message's content, which is how the
// app declares its web_search tool through the engine's string-message API. The t_ cases of the template check compare
// both against transformers' apply_chat_template(tools=...): the tools system turn alone, a full tool round trip with
// thinking off and on, and the follow-up after it.
//
// Thinking segment, by token id (`<think>` 248068 and `</think>` 248069 are single added tokens, special = false, so they
// also decode as text): Qwen3.6's generation prompt already opens the block, so with thinking on the reply starts INSIDE
// it and its first token is thought, not `<think>`. ThinkingTracker starts inside when the prompt's last assistant header
// is followed by an unclosed `<think>`, marks each generated token open / thinking / close / answer, and, with a budget,
// forces the template's close (`\n</think>\n\n`, or `</think>\n\n` after a newline) once the block has used `budget`
// tokens; the forced tokens are fed like sampled ones and are part of the reply.
import Foundation
import Tokenizers

public enum ChatPrompt {
    /// The prompt ids for `messages` (role/content dictionaries), add_generation_prompt = true, enable_thinking =
    /// `thinking` (nil: not passed).
    public static func ids(_ tk: Tokenizer, messages: [[String: String]], thinking: Bool?) throws -> [Int] {
        let msgs: [Message] = messages.map { $0 as [String: any Sendable] }
        return try tk.applyChatTemplate(messages: msgs, tools: nil, additionalContext: thinking.map { ["enable_thinking": $0] })
    }

    /// The template's own tools path (transformers' apply_chat_template(tools=...)): `tools` (JSON-schema function specs)
    /// are passed to the Jinja render, which writes them into the system turn; messages may carry structured `tool_calls`
    /// and the `tool` role. swift-jinja converts Swift dictionaries with their keys sorted, so `tojson` writes sorted keys
    /// (the reference's specs are written with sorted keys for that reason). The app does not call this: it renders the
    /// tools block into the system content itself (ToolPrompt.systemContent), which gives the same ids (template check).
    public static func ids(_ tk: Tokenizer, messages: [[String: any Sendable]], thinking: Bool?, tools: [[String: any Sendable]]?,
                           addGenerationPrompt: Bool = true) throws -> [Int] {
        try tk.applyChatTemplate(messages: messages, chatTemplate: nil, addGenerationPrompt: addGenerationPrompt, truncation: false,
                                 maxLength: nil, tools: tools, additionalContext: thinking.map { ["enable_thinking": $0] })
    }

    /// JSON (JSONSerialization output: NSDictionary / NSArray / NSNumber / NSString) as Swift values the Jinja bridge takes.
    static func sendable(_ v: Any) -> any Sendable {
        if let d = v as? [String: Any] { return d.mapValues { sendable($0) } as [String: any Sendable] }
        if let a = v as? [Any] { return a.map { sendable($0) } as [any Sendable] }
        if let n = v as? NSNumber {
            if CFGetTypeID(n) == CFBooleanGetTypeID() { return n.boolValue }
            return n.doubleValue == n.doubleValue.rounded() && abs(n.doubleValue) < 1e15 ? n.intValue : n.doubleValue
        }
        if let s = v as? String { return s }
        return String(describing: v)
    }

    /// The template check: renders every case of a reference written by docs/template/make_reference.py (transformers'
    /// apply_chat_template) and compares the ids. Returns (all equal, report lines).
    public static func check(_ tk: Tokenizer, referencePath: String) throws -> (ok: Bool, lines: [String]) {
        guard let d = FileManager.default.contents(atPath: referencePath),
              let j = try JSONSerialization.jsonObject(with: d) as? [String: Any],
              let cases = j["cases"] as? [[String: Any]] else {
            throw NSError(domain: "template-check", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(referencePath): no cases"])
        }
        var ok = true
        var lines: [String] = ["reference: \(referencePath) (transformers \(j["transformers"] ?? "?"))"]
        let tt = ThinkTokens(tk)
        lines.append("ids: <think> \(tt.open.map(String.init) ?? "-"), </think> \(tt.close.map(String.init) ?? "-"), <|im_start|> \(tt.imStart.map(String.init) ?? "-"), <|im_end|> \(tk.convertTokenToId("<|im_end|>").map(String.init) ?? "-"), <|endoftext|> \(tk.convertTokenToId("<|endoftext|>").map(String.init) ?? "-")")
        if let sp = j["special_ids"] as? [String: Int] {
            for (s, v) in sp.sorted(by: { $0.key < $1.key }) where tk.convertTokenToId(s) != v {
                ok = false
                lines.append("MISMATCH special id \(s): swift \(tk.convertTokenToId(s).map(String.init) ?? "nil"), transformers \(v)")
            }
        }
        if let jl = j["tools_json_lines"] as? [String] {
            let same = jl == [ToolPrompt.webSearchJSON]
            if !same { ok = false }
            lines.append((same ? "OK   " : "DIFF ") + "ToolPrompt.webSearchJSON vs json.dumps of the reference's tools")
        }
        if let sc = j["tools_system_content"] as? [String: String], let sys = sc["system"] {
            for (k, s) in [("content", sys), ("content_no_system", "")] {
                let same = ToolPrompt.systemContent(system: s, toolsJSON: [ToolPrompt.webSearchJSON]) == sc[k]
                if !same { ok = false }
                lines.append((same ? "OK   " : "DIFF ") + "ToolPrompt.systemContent (\(k)) vs the template's tools system turn")
            }
        }
        for c in cases {
            let name = c["name"] as? String ?? "?"
            let think = c["enable_thinking"] as? Bool
            let agp = c["add_generation_prompt"] as? Bool ?? true
            let ref = c["ids"] as? [Int] ?? []
            if let tools = c["tools"] as? [Any] {
                // tool cases: (1) the template's tools path through swift-jinja with structured tool_calls / tool turns;
                // (2) the app's path (app_messages: the tools block in the system content, raw tool-call text), no tools
                let msgsAny = ((c["messages"] as? [Any]) ?? []).map { sendable($0) as? [String: any Sendable] ?? [:] }
                let specs = tools.map { sendable($0) as? [String: any Sendable] ?? [:] }
                var variants: [(String, [Int])] = [("jinja tools", try ids(tk, messages: msgsAny, thinking: think, tools: specs, addGenerationPrompt: agp))]
                if let app = c["app_messages"] as? [[String: String]] {
                    variants.append(("app path", try ids(tk, messages: app.map { $0 as [String: any Sendable] }, thinking: think, tools: nil, addGenerationPrompt: agp)))
                }
                for (how, mine) in variants {
                    report(name: "\(name) [\(how)]", mine: mine, ref: ref, tk: tk, ok: &ok, lines: &lines)
                }
                continue
            }
            let msgs = (c["messages"] as? [[String: String]]) ?? []
            let mine: [Int]
            if agp {
                mine = try ids(tk, messages: msgs, thinking: think)
            } else {
                mine = try tk.applyChatTemplate(messages: msgs.map { $0 as [String: any Sendable] }, chatTemplate: nil,
                                                addGenerationPrompt: false, truncation: false, maxLength: nil, tools: nil,
                                                additionalContext: think.map { ["enable_thinking": $0] })
            }
            report(name: name, mine: mine, ref: ref, tk: tk, ok: &ok, lines: &lines)
        }
        return (ok, lines)
    }

    static func report(name: String, mine: [Int], ref: [Int], tk: Tokenizer, ok: inout Bool, lines: inout [String]) {
        if mine == ref {
            lines.append("OK   \(name): \(mine.count) ids equal; tail \(mine.suffix(7))")
        } else {
            ok = false
            let n = zip(mine, ref).prefix { $0 == $1 }.count
            lines.append("DIFF \(name): swift \(mine.count) ids, transformers \(ref.count); first difference at \(n): swift \(Array(mine[n..<min(mine.count, n + 8)])) vs \(Array(ref[n..<min(ref.count, n + 8)]))")
            lines.append("     swift text: \(tk.decode(tokens: Array(mine[max(0, n - 8)...])).debugDescription)")
            lines.append("     ref text:   \(tk.decode(tokens: Array(ref[max(0, n - 8)...])).debugDescription)")
        }
        lines.append("     ids \(mine)")
    }
}

/// Tools in the system turn without the engine's help (docs/TOOLS.md). The engine takes string messages and renders them
/// without `tools`; with tools the release's template writes `<|im_start|>system\n` + the tools block + ("\n\n" + system
/// prompt, trimmed, if any) + `<|im_end|>\n`, and without tools `<|im_start|>system\n` + content (trimmed) + `<|im_end|>\n`.
/// So a system message whose content is the tools block + "\n\n" + the prompt renders the same bytes (the template check's
/// "app path" variant of the t_ cases compares the ids with transformers' apply_chat_template(tools=...)). The block's text
/// is the template's, verbatim; each tool is one line of JSON as transformers' `tojson` writes it (json.dumps: ", " and
/// ": " separators, keys in the spec's order; the app's specs are written with sorted keys, as swift-jinja would).
public enum ToolPrompt {
    public static let head = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
    public static let tail = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

    /// The system message content that renders as the template's tools system turn. `toolsJSON`: one JSON line per tool.
    public static func systemContent(system: String, toolsJSON: [String]) -> String {
        var s = head
        for j in toolsJSON { s += "\n" + j }
        s += tail
        let t = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { s += "\n\n" + t }
        return s
    }

    /// The app's one tool (docs/TOOLS.md): terse, one parameter, keys sorted. `n` (number of results, default 3) is accepted
    /// by the app but not declared: declaring it costs 19 more system-turn tokens (272 -> 291 with a one-line system prompt).
    public static let webSearchJSON = #"{"function": {"description": "Search the web.", "name": "web_search", "parameters": {"properties": {"query": {"type": "string"}}, "required": ["query"], "type": "object"}}, "type": "function"}"#

    /// The tool call tags (single added tokens 248058 / 248059 in Qwen3.6's vocabulary, special = false: they decode as text).
    public static let callOpen = "<tool_call>", callClose = "</tool_call>"
}

/// The think-block token ids of a tokenizer (nil when it has none) and the template's close continuation.
public struct ThinkTokens: Sendable {
    public let open: Int?  // <think>
    public let close: Int?  // </think>
    public let imStart: Int?  // <|im_start|>
    /// "\n</think>\n\n" and "</think>\n\n" (after a newline): the template's `reasoning + '\n</think>\n\n' + content`.
    public let closeAfterText: [Int]
    public let closeAfterNewline: [Int]

    public init(_ tk: Tokenizer) {
        open = tk.convertTokenToId("<think>")
        close = tk.convertTokenToId("</think>")
        imStart = tk.convertTokenToId("<|im_start|>")
        closeAfterText = close == nil ? [] : tk.encode(text: "\n</think>\n\n", addSpecialTokens: false)
        closeAfterNewline = close == nil ? [] : tk.encode(text: "</think>\n\n", addSpecialTokens: false)
    }
}

public enum ThinkPhase: String, Sendable {
    case open  // a <think> token (the model opening a block itself, or a redundant one inside the block)
    case thinking  // inside the block
    case close  // the </think> token
    case answer  // after the block (or no block)
}

/// The thinking segment of one reply, by token id.
public struct ThinkingTracker: Sendable {
    public let tokens: ThinkTokens
    /// Tokens inside the block (the open/close tags excluded) after which the close is forced; nil: no budget.
    public let budget: Int?
    /// The prompt ends inside a think block (Qwen3.6's generation prompt with thinking on: `<|im_start|>assistant\n<think>\n`).
    public let startedInside: Bool
    public private(set) var inside: Bool
    public private(set) var thinkingTokens = 0
    public private(set) var closed = false
    public private(set) var forcedClose = false
    private var pendingForce: [Int] = []

    public init(promptIds: [Int], tokens: ThinkTokens, budget: Int?) {
        self.tokens = tokens
        self.budget = budget
        var ins = false
        if let o = tokens.open {
            let from = tokens.imStart.flatMap { promptIds.lastIndex(of: $0) } ?? 0
            for id in promptIds[from...] {
                if id == o { ins = true } else if id == tokens.close { ins = false }
            }
        }
        startedInside = ins
        inside = ins
    }

    /// The phase of a token fed after the prompt (sampled or forced), updating the state.
    public mutating func observe(_ id: Int) -> ThinkPhase {
        if id == tokens.open { inside = true; return .open }
        if inside && id == tokens.close { inside = false; closed = true; return .close }
        if inside { thinkingTokens += 1; return .thinking }
        return .answer
    }

    /// The next token to feed instead of sampling, when the budget forces the close (nil: sample). `lastText` is the
    /// decoded reply so far (decides whether the close needs its leading newline).
    public mutating func forced(lastText: () -> String) -> Int? {
        if pendingForce.isEmpty, let b = budget, inside, !forcedClose, thinkingTokens >= b, tokens.close != nil {
            pendingForce = lastText().hasSuffix("\n") ? tokens.closeAfterNewline : tokens.closeAfterText
            forcedClose = true
        }
        return pendingForce.isEmpty ? nil : pendingForce.removeFirst()
    }

    public var summary: [String: Any] {
        ["started_inside": startedInside, "thinking_tokens": thinkingTokens, "closed": closed, "forced_close": forcedClose,
         "budget": budget.map { $0 as Any } ?? NSNull()]
    }
}
