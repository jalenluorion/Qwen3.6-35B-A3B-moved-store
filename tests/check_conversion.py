"""Post-conversion checks of a converted model folder (default ../model).

  python tests/check_conversion.py [OUT_DIR]

- every store/layer_XX.bin and store4/layer_XX.bin is exactly experts * record_bytes;
- both meta.json files (printed);
- resident.safetensors loads (header + every tensor readable) and its key count equals the checkpoint's
  non-expert, non-mtp, non-visual key count from model.safetensors.index.json;
- a few experts are read from the bf16 release (HTTP range reads of only those bytes, no shard download) and compared
  with the records in both stores: max abs error and relative Frobenius error, plus a check that the int4 bytes equal
  mx.quantize of the original and the int8 bytes equal the formula.
"""
import json, os, struct, sys

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import HfFileSystem
from safetensors import safe_open

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
m8 = json.load(open(os.path.join(OUT, "store", "meta.json")))
m4 = json.load(open(os.path.join(OUT, "store4", "meta.json")))
print("store/meta.json ", json.dumps(m8))
print("store4/meta.json", json.dumps(m4))
H, I, NL, NE = m8["hidden"], m8["intermediate"], m8["layers"], m8["experts"]
ok = True
for sub, m in (("store", m8), ("store4", m4)):
    sizes = [os.path.getsize(os.path.join(OUT, sub, f"layer_{L:02d}.bin")) for L in range(NL)]
    good = all(s == NE * m["record_bytes"] for s in sizes)
    print(f"{sub}: {NL} files, each {NE} x {m['record_bytes']} = {NE * m['record_bytes']} B: {good}; total {sum(sizes) / 2**30:.2f} GiB")
    ok &= good

wm = json.load(open(os.path.join(OUT, "model.safetensors.index.json")))["weight_map"]
want = sorted(k for k in wm if not k.startswith("mtp.") and not k.startswith("model.visual") and ".mlp.experts." not in k)
res = os.path.join(OUT, "resident.safetensors")
nbytes, dtypes = 0, {}
with safe_open(res, framework="pt") as f:
    keys = sorted(f.keys())
    for k in keys:
        t = f.get_tensor(k)
        nbytes += t.numel() * t.element_size()
        dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
good = keys == want
print(f"resident.safetensors: {len(keys)} tensors (expected {len(want)}), names match: {good}, {nbytes / 2**30:.2f} GiB of data, "
      f"file {os.path.getsize(res) / 2**30:.2f} GiB, dtypes {dtypes}")
for k in ("lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.layers.0.mlp.gate.weight",
          "model.language_model.layers.0.mlp.shared_expert_gate.weight", "model.language_model.layers.3.self_attn.q_proj.weight"):
    print("  has", k, k in keys)
ok &= good

# ---- experts against the bf16 release, read by range from the hub
fs = HfFileSystem()
REPO = m8["repo"]


def read_expert(key, e):
    path = f"{REPO}/{wm[key]}"
    with fs.open(path, "rb", block_size=1 << 20) as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))[key]
        assert hdr["dtype"] == "BF16", hdr
        shape = hdr["shape"]
        per = int(np.prod(shape[1:])) * 2
        fh.seek(8 + n + hdr["data_offsets"][0] + e * per)
        raw = fh.read(per)
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape[1:]).float()


def errs(d, w):
    return float(np.abs(d - w).max()), float(np.linalg.norm(d - w) / np.linalg.norm(w))


SH = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
for L, e in ((0, 0), (39, 255), (17, 128)):
    pre = f"model.language_model.layers.{L}.mlp.experts."
    gu = read_expert(pre + "gate_up_proj", e)
    W = {"gate_proj": gu[:I].numpy(), "up_proj": gu[I:].numpy(), "down_proj": read_expert(pre + "down_proj", e).numpy()}
    with open(os.path.join(OUT, "store", f"layer_{L:02d}.bin"), "rb") as fh:
        fh.seek(e * m8["record_bytes"])
        r8 = fh.read(m8["record_bytes"])
    with open(os.path.join(OUT, "store4", f"layer_{L:02d}.bin"), "rb") as fh:
        fh.seek(e * m4["record_bytes"])
        r4 = fh.read(m4["record_bytes"])
    tot = {"8": [0.0, 0.0, 0.0], "4": [0.0, 0.0, 0.0]}  # max abs, sum sq err, sum sq w
    for k in ("gate_proj", "up_proj", "down_proj"):
        rows, cols = SH[k]
        w = W[k]
        q8 = np.frombuffer(r8, np.int8, rows * cols, m8["offsets"][k]).reshape(rows, cols).astype(np.float32)
        s8 = np.frombuffer(r8, np.float16, I, m8["scale_offsets"][k]).astype(np.float32)
        d8 = q8 * (s8[None, :] if k == "down_proj" else s8[:, None])
        wq = np.frombuffer(r4, np.uint32, rows * cols // 8, m4["offsets"][k]).reshape(rows, cols // 8)
        s4 = np.frombuffer(r4, np.float16, rows * cols // 64, m4["scale_offsets"][k]).reshape(rows, cols // 64)
        b4 = np.frombuffer(r4, np.float16, rows * cols // 64, m4["bias_offsets"][k]).reshape(rows, cols // 64)
        d4 = np.array(mx.dequantize(mx.array(wq), mx.array(s4), mx.array(b4), group_size=64, bits=4).astype(mx.float32))
        rq, rs, rb = mx.quantize(mx.array(w), group_size=64, bits=4)
        same4 = wq.tobytes() == np.array(rq).tobytes() and s4.tobytes() == np.array(rs.astype(mx.float16)).tobytes() \
            and b4.tobytes() == np.array(rb.astype(mx.float16)).tobytes()
        wt = torch.from_numpy(w)
        sf = (wt.abs().amax(dim=0 if k == "down_proj" else 1).clamp_min(1e-8) / 127.0).to(torch.float16)
        qf = torch.clamp((wt / (sf.float()[None, :] if k == "down_proj" else sf.float()[:, None])).round(), -127, 127)
        same8 = np.array_equal(q8, qf.numpy()) and np.array_equal(s8, sf.float().numpy())
        a8, f8 = errs(d8, w)
        a4, f4 = errs(d4, w)
        print(f"L{L:02d} e{e:03d} {k:9s} int8 max|err| {a8:.3e} rel-F {f8:.4f} | int4 max|err| {a4:.3e} rel-F {f4:.4f} | "
              f"|w|max {np.abs(w).max():.3e} | int4 == mx.quantize(orig): {same4}, int8 == formula: {same8}")
        ok &= same4 and same8
        for key, d in (("8", d8), ("4", d4)):
            tot[key][0] = max(tot[key][0], float(np.abs(d - w).max()))
            tot[key][1] += float(((d - w) ** 2).sum())
            tot[key][2] += float((w ** 2).sum())
    for key in ("8", "4"):
        print(f"L{L:02d} e{e:03d} whole expert int{key}: max|err| {tot[key][0]:.3e}, rel-F {np.sqrt(tot[key][1] / tot[key][2]):.4f}")
print("CHECK_OK" if ok else "CHECK_FAILED")
