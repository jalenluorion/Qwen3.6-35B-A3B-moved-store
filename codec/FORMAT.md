# PSC1: the lossless format of the compressed expert store

`codec.py` (format, encoder, a NumPy reference decoder) and `encode_layers.py` (writes `layer_XX.psc` + `layer_XX.idx`
from a raw `store4/layer_XX.bin`; `PSC_STORE` names the raw store4 folder, default `../model/store4`). The app decodes
the records on the GPU (`native/Sources/PhoneStore/Codec.swift`) bit for bit into the raw record bytes; the `.idx`
holds the SHA-256 of every raw record.

## Format

Files: `codec.py` (format, encoder, numpy reference decoder), `encode_layers.py` (layer files, stats, phone pack),
`survey_layers.py` (all 40 layers on a sample), `Sources/CodecCore/Shader.swift` (the Metal kernels),
`Sources/CodecCore/{PSC,Decoder,Bench}.swift` (host side, shared by the macOS CLI and the iOS app),
`Sources/codecbench/main.swift` (CLI).

A record is cut into **108 independent blocks**, each decoded by one 32-lane SIMD group:

- **96 nibble blocks** (3 matrices x 32 x 16 KiB of uint32 code words). Lane j owns output words j, j+32, ..., j+32*127 of
  the block (writes are coalesced 128 B rows). Each output byte (two nibbles) is one symbol of a static per-layer
  **Huffman code over the 256 byte values** (pairs, not single nibbles: free on layers 1-39, and on layer 0 the dead
  experts make the two nibbles of a byte correlated, 2.99 vs 3.35 bits/nibble). Code lengths limited to 12 bits by
  package-merge, so decoding is one lookup in a 4096-entry u16 table (sym | len << 8) held in threadgroup memory.
- **12 side blocks** (3 matrices x 4 x 4096 groups). Per group: scale hi byte (table T1), scale lo byte (T2), then a
  symbol q (T3). **The bias is predicted from the scale**: MLX affine quantization stores bias = edge and
  scale = edge / q0 with q0 a small integer, so `bias == fp16(float(q) * float(scale))` for all but ~11 of the 49,152
  groups of a record (q = 0 predicts +0: the dead groups of layer 0 have scale -1e-7, bias +0). The bias costs
  H(q) = 1.15 bits instead of 8.25. When the prediction fails, q = 255 (escape) and the 16 raw bits follow.
- **Bit streams**: every lane keeps a 32-bit MSB-first buffer and, before every symbol, takes one 16-bit unit if it
  holds < 12 bits. The units of the 32 lanes are interleaved in the order those refills happen (symbol step major,
  lane minor), so a lane finds its unit with `simd_ballot` + `popcount` and a block needs only its start offset. The
  encoder replays the same refill rule to lay the units out. All lanes run the same step sequence (escape raw bytes are
  two extra steps that consume 0 or 8 bits), so the ballots stay in lockstep.
- **Record** = u32[108] block start offsets (in 16-bit units) + the u16 payload, zero padded to 16 B. Decoded output is
  the store4 record layout byte for byte (code words, fp16 scales, fp16 biases per matrix), so a slot can take it
  unchanged.
- **Layer files**: `out/layer_XX.psc` (records back to back) + `out/layer_XX.idx` ('PSC1' header, 4 x 256 code lengths,
  4 x 4096 decode tables, u64 record offsets, SHA-256 of every original record). Tables are per layer, 33 KB per layer
  file (0.007% of the layer).

Overheads per record: block offsets 432 B (0.024%), lane tail waste (last partial unit of each of the 3,456 lanes,
~5.6 KB, 0.31%), padding <16 B. Total 0.34%, inside the 0.5% budget. Padding records to 4 KiB for direct flash reads
would add ~0.12% (16 KiB: ~0.46%).

## Compressed size (all 256 records of each layer)

| layer | ratio (mean) | min - max | 0.885 bound (zero-order nibble + fp16 scale + fp16 bias) | entropy of this model | Huffman payload | nibble bits/weight |
|---|---|---|---|---|---|---|
| 0  | **0.7063** | 0.493 - 0.933 | 0.7956 | 0.6967 | 0.7026 | 3.01 |
| 10 | **0.8689** | 0.865 - 0.870 | 0.8846 | 0.8602 | 0.8655 | 3.74 |
| 20 | **0.8694** | 0.862 - 0.881 | 0.8855 | 0.8608 | 0.8660 | 3.74 |
| 39 | **0.8688** | 0.863 - 0.889 | 0.8858 | 0.8607 | 0.8654 | 3.74 |

- All 40 layers, 16 records each with tables from the same sample (`out/survey.json`): layers 1-39 at 0.8668-0.8702,
  mean 0.8689; with layer 0's full-layer 0.706 the model mean is **0.865**.
- The result beats the 0.885 bound because the bound coded the bias independently (8.25 bits); predicting it from the
  scale takes ~2.4% of the record off. The remaining gap to this model's entropy (0.861 -> 0.869) is Huffman
  integer-length loss (0.4% of the nibble stream: 7.456 vs 7.431 bits per byte) plus the 0.34% overhead. rANS would buy
  back ~0.3-0.5% at the cost of per-lane state words; not worth it.
- Escapes: 11 per record on layer 20, 28 on layer 39, 34 on layer 0.

