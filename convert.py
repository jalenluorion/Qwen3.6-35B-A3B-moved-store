"""One-time conversion of a Qwen MoE release into the expert stores, on a machine with little free disk.

  REPO=Qwen/Qwen3.6-35B-A3B python convert.py --out model

Works for releases whose experts are per-expert tensors (Qwen3-30B-A3B, Qwen3-Next-80B-A3B, Qwen3-Coder-Next:
model.layers.N.mlp.experts.M.{gate,up,down}_proj.weight) and for the fused layout of Qwen3.5/3.6 MoE
(model.language_model.layers.N.mlp.experts.gate_up_proj [E, 2I, H], first I rows gate, next I rows up, as mlx-lm's
qwen3_5_moe sanitize splits it; .experts.down_proj [E, H, I]). Config values are read from config["text_config"] when
present.

Downloads the released weights one shard at a time and, in the same pass over each shard, writes every routed expert
once into two stores, then deletes the shard:

  store/layer_XX.bin   int8, one fp16 scale per neuron (the quality reference). Record (REC bytes, one expert):
                       int8 gate [I, H] | int8 up [I, H] | int8 down [H, I] | fp16 scales gate[I] | up[I] | down[I] |
                       zero padding to a 16 KB page.  W_gate = q * s[:, None] (per output neuron), W_down = q * s[None, :]
                       (per input neuron); s = amax over the non-neuron axis / 127.
  store4/layer_XX.bin  int4, exactly mx.quantize(mx.array(w_fp32), group_size=64, bits=4) per matrix, scales and biases
                       cast to fp16. Record (REC4 bytes): for gate [I, H], up [I, H], down [H, I] in turn:
                       w_q uint32 [rows, cols/8] | scales fp16 [rows, cols/64] | biases fp16 [rows, cols/64]; zero
                       padding to a 16 KB page. Offsets per matrix are in store4/meta.json. The runner hands the bytes
                       to mx.gather_qmm(bits=4, group_size=64) unchanged.

Everything that is not a routed expert goes into resident.safetensors (original names and dtypes). Records sit at
expert_id * record_bytes. Env: REPO (required), SKIP_KEYS (regex of tensors to drop; default the multi-token-prediction
head and the vision tower), STATE_DIR (shared directory for resuming on a cluster), HF_REPO (upload each finished layer
of both stores at once), OUT (same as --out; default: next to this file), CHUNK (experts quantized per batch, 32).
"""
import argparse, gc, hashlib, json, os, re, shutil, struct, time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

PAGE = 16384
GROUP, BITS4 = 64, 4
KINDS = ("gate_proj", "up_proj", "down_proj")


def page_round(n):
    return -(-n // PAGE) * PAGE


def layout8(H, I):
    """int8 record: codes of gate, up, down, then the three fp16 scale vectors (one per neuron, I each)."""
    MAT = I * H
    OFF = {"gate_proj": 0, "up_proj": MAT, "down_proj": 2 * MAT}
    SOFF = {"gate_proj": 3 * MAT, "up_proj": 3 * MAT + 2 * I, "down_proj": 3 * MAT + 4 * I}
    return page_round(3 * MAT + 3 * I * 2), OFF, SOFF


def layout4(H, I):
    """int4 record: per matrix [w_q uint32 rows x cols/8 | scales fp16 rows x cols/64 | biases fp16 rows x cols/64]."""
    shapes = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
    OFF, SOFF, BOFF, pos = {}, {}, {}, 0
    for kind in KINDS:
        r, c = shapes[kind]
        assert c % GROUP == 0, (kind, c)
        OFF[kind] = pos
        pos += r * (c // 8) * 4
        SOFF[kind] = pos
        pos += r * (c // GROUP) * 2
        BOFF[kind] = pos
        pos += r * (c // GROUP) * 2
    return page_round(pos), OFF, SOFF, BOFF


def quant8(w, kind):
    """w: float32 [E, rows, cols] (gate/up: [E, I, H]; down: [E, H, I]). Returns int8 codes [E, rows, cols] and fp16
    scales [E, I]: amax over the non-neuron axis / 127, identical per matrix to the original per-expert code."""
    dim = 1 if kind == "down_proj" else 2  # reduce over the non-neuron axis
    s = (w.abs().amax(dim=dim).clamp_min(1e-8) / 127.0).to(torch.float16)
    sf = s.float()
    q = torch.clamp((w / (sf[:, None, :] if kind == "down_proj" else sf[:, :, None])).round(), -127, 127).to(torch.int8)
    return q, s


def quant4(w):
    """w: float32 numpy [E, rows, cols]. mx.quantize quantizes each row along the last axis in groups of 64, so the
    batched call returns, per matrix, exactly what mx.quantize returns on that matrix alone (checked in tests/).
    Returns numpy w_q uint32 [E, rows, cols/8], scales fp16 and biases fp16 [E, rows, cols/64]."""
    import mlx.core as mx

    wq, s, b = mx.quantize(mx.array(w), group_size=GROUP, bits=BITS4)
    s, b = s.astype(mx.float16), b.astype(mx.float16)
    mx.eval(wq, s, b)
    out = np.array(wq), np.array(s), np.array(b)
    del wq, s, b
    return out


def write_expert_mats(fd8, fd4, e0, w, kind, REC, OFF, SOFF, REC4, OFF4, SOFF4, BOFF4):
    """Quantize a batch of experts' matrices [E_c, rows, cols] (torch float32) into both stores at expert e0..."""
    q, s = quant8(w, kind)
    q, s = q.numpy(), s.numpy()
    wq4, s4, b4 = quant4(w.numpy())
    for j in range(w.shape[0]):
        e = e0 + j
        os.pwrite(fd8, q[j].tobytes(), e * REC + OFF[kind])
        os.pwrite(fd8, s[j].tobytes(), e * REC + SOFF[kind])
        os.pwrite(fd4, wq4[j].tobytes(), e * REC4 + OFF4[kind])
        os.pwrite(fd4, s4[j].tobytes(), e * REC4 + SOFF4[kind])
        os.pwrite(fd4, b4[j].tobytes(), e * REC4 + BOFF4[kind])


def merge_safetensors(parts, out_path):
    """Concatenate safetensors files into one by copying raw tensor bytes (dtypes and names untouched, little RAM)."""
    entries = []  # (name, dtype, shape, part, begin, end)
    for p in parts:
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            entries.append((k, v["dtype"], v["shape"], p, base + v["data_offsets"][0], base + v["data_offsets"][1]))
    names = [e[0] for e in entries]
    assert len(names) == len(set(names)), "duplicate tensor names across parts"
    entries.sort(key=lambda e: e[0])
    header, pos = {"__metadata__": {"format": "pt"}}, 0
    for k, dt, shape, _, b, e in entries:
        header[k] = {"dtype": dt, "shape": shape, "data_offsets": [pos, pos + (e - b)]}
        pos += e - b
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        for k, dt, shape, p, b, e in entries:
            with open(p, "rb") as fh:
                fh.seek(b)
                left = e - b
                while left:
                    buf = fh.read(min(left, 64 << 20))
                    out.write(buf)
                    left -= len(buf)
    return len(entries)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(16 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT", os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()
    REPO = os.environ["REPO"]
    HERE = os.path.abspath(args.out)
    WORK, STORE, STORE4 = os.path.join(HERE, "_download"), os.path.join(HERE, "store"), os.path.join(HERE, "store4")
    for d in (WORK, STORE, STORE4):
        os.makedirs(d, exist_ok=True)
    CHUNK = int(os.environ.get("CHUNK", 32))
    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)

    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json",
              "model.safetensors.index.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        try:
            shutil.copy(hf_hub_download(REPO, f, local_dir=WORK), os.path.join(HERE, f))
        except Exception as e:
            print("no", f, type(e).__name__, flush=True)
    cfg_full = json.load(open(os.path.join(HERE, "config.json")))
    cfg = cfg_full.get("text_config", cfg_full)
    H, I, NL = cfg["hidden_size"], cfg.get("moe_intermediate_size", cfg.get("intermediate_size")), cfg["num_hidden_layers"]
    NE = cfg.get("num_experts", cfg.get("num_local_experts"))
    MAT = I * H
    REC, OFF, SOFF = layout8(H, I)
    REC4, OFF4, SOFF4, BOFF4 = layout4(H, I)
    # Qwen3.5/3.6 MoE renormalizes the top-k router weights unconditionally (HF router; mlx-lm ModelArgs default True)
    # but its config has no norm_topk_prob key, so the default depends on the architecture.
    common = {"repo": REPO, "model_type": cfg_full.get("model_type"), "hidden": H, "intermediate": I, "layers": NL, "experts": NE,
              "top_k": cfg.get("num_experts_per_tok", 8), "norm_topk_prob": bool(cfg.get("norm_topk_prob", cfg_full.get("model_type") == "qwen3_5_moe")), "page": PAGE,
              "matrix_bytes": MAT}
    meta8 = dict(common, record_bytes=REC, offsets=OFF, scale_offsets=SOFF, dtype="int8", bits=8, scale_dtype="float16")
    meta4 = dict(common, record_bytes=REC4, offsets=OFF4, scale_offsets=SOFF4, bias_offsets=BOFF4, dtype="int4", bits=BITS4,
                 group=GROUP, scale_dtype="float16")
    log(f"H={H} I={I} layers={NL} experts={NE} | int8 record {REC} B, int4 record {REC4} B | out {HERE}")
    fds8, fds4 = [], []
    for L in range(NL):
        for store, rec, fds in ((STORE, REC, fds8), (STORE4, REC4, fds4)):
            p = os.path.join(store, f"layer_{L:02d}.bin")
            if not os.path.exists(p) or os.path.getsize(p) != NE * rec:
                with open(p, "wb") as fh:
                    fh.truncate(NE * rec)
            fds.append(os.open(p, os.O_RDWR))
    index = json.load(open(os.path.join(HERE, "model.safetensors.index.json")))["weight_map"]
    shards = sorted(set(index.values()))
    LP = r"(?:model\.language_model\.|model\.)layers\.(\d+)\.mlp\.experts\."
    pat = re.compile(LP + r"(\d+)\.(gate_proj|up_proj|down_proj)\.weight")  # per-expert tensors
    pat_fused = re.compile(LP + r"(gate_up_proj|down_proj)")  # fused [E, 2I, H] / [E, H, I]
    SKIP = re.compile(os.environ.get("SKIP_KEYS", r"^(mtp\.|model\.visual)"))
    # STATE_DIR (default: out) holds the per-shard state and the small resident parts; on a cluster it is a shared
    # directory so a job requeued onto another node resumes. With HF_REPO set, every store layer is uploaded the moment
    # its last expert is written and skipped afterwards, so the large files never have to survive a node change.
    STATE_DIR = os.environ.get("STATE_DIR", HERE)
    os.makedirs(STATE_DIR, exist_ok=True)
    state_path = os.path.join(STATE_DIR, "_convert_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"done": [], "mats": {}, "uploaded": []}
    for k, v in (("mats", {}), ("uploaded", []), ("shard_layers", {}), ("node", None), ("parts", [])):
        state.setdefault(k, v)
    HF_REPO_UP = os.environ.get("HF_REPO")
    NODE = os.uname().nodename
    if state["node"] != NODE and state["done"]:
        # A different machine: the local store files are gone. Layers already uploaded are safe; a layer that was only
        # partly written must be rebuilt, so every shard that touched it is redone.
        redo = {sh for sh, Ls in state["shard_layers"].items() for L in Ls if L not in state["uploaded"]}
        for L in list(state["mats"]):
            if int(L) not in state["uploaded"]:
                state["mats"][L] = 0
        state["done"] = [sh for sh in state["done"] if sh not in redo]
        print(f"resuming on {NODE}: {len(state['uploaded'])} layers uploaded, {len(redo)} shards to redo", flush=True)
    state["node"] = NODE
    n_exp = 0
    part_path = os.path.join(STATE_DIR, "_resident_part_{}.safetensors")

    def upload_complete_layers():
        """Layers whose 3*NE matrices have all been written are complete: push both stores' files now (idempotent)."""
        if not HF_REPO_UP:
            return
        from huggingface_hub import HfApi

        for L in range(NL):
            if state["mats"].get(str(L), 0) >= 3 * NE and L not in state["uploaded"]:
                for store, sub, rec, fds in ((STORE, "store", REC, fds8), (STORE4, "store4", REC4, fds4)):
                    os.fsync(fds[L])
                    path = os.path.join(store, f"layer_{L:02d}.bin")
                    assert os.path.getsize(path) == NE * rec, (path, os.path.getsize(path))
                    HfApi().upload_file(path_or_fileobj=path, path_in_repo=f"{sub}/layer_{L:02d}.bin", repo_id=HF_REPO_UP,
                                        commit_message=f"{sub} layer {L}")
                state["uploaded"].append(L)
                json.dump(state, open(state_path, "w"))
                log("uploaded", f"layer_{L:02d}.bin")

    args8_4 = (REC, OFF, SOFF, REC4, OFF4, SOFF4, BOFF4)
    for si, shard in enumerate(shards):
        if shard in state["done"]:
            log("skip", shard, "(already converted)")
            continue
        log("downloading", shard)
        path = hf_hub_download(REPO, shard, local_dir=WORK)
        log("converting", shard, f"{os.path.getsize(path) / 2**30:.1f} GB")
        res_part = {}
        touched = set()
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                if SKIP.match(k):  # auxiliary heads the served text model does not use (MTP block, vision tower)
                    continue
                m, mf = pat.fullmatch(k), pat_fused.fullmatch(k)
                if m:
                    L, e, kind = int(m.group(1)), int(m.group(2)), m.group(3)
                    w = f.get_tensor(k).to(torch.float32)[None]
                    write_expert_mats(fds8[L], fds4[L], e, w, kind, *args8_4)
                    nm = 1
                elif mf:
                    L, kind = int(mf.group(1)), mf.group(2)
                    sl = f.get_slice(k)
                    shape = sl.get_shape()
                    want = [NE, 2 * I, H] if kind == "gate_up_proj" else [NE, H, I]
                    assert list(shape) == want, (k, shape, want)
                    for e0 in range(0, NE, CHUNK):
                        t = sl[e0:e0 + CHUNK].to(torch.float32)
                        if kind == "gate_up_proj":  # mlx-lm qwen3_5_moe: mid = shape[-2] // 2, gate first
                            write_expert_mats(fds8[L], fds4[L], e0, t[:, :I].contiguous(), "gate_proj", *args8_4)
                            write_expert_mats(fds8[L], fds4[L], e0, t[:, I:].contiguous(), "up_proj", *args8_4)
                        else:
                            write_expert_mats(fds8[L], fds4[L], e0, t.contiguous(), "down_proj", *args8_4)
                        del t
                    nm = NE * (2 if kind == "gate_up_proj" else 1)
                    gc.collect()
                    try:
                        import mlx.core as mx

                        mx.clear_cache()
                    except Exception:
                        pass
                else:
                    res_part[k] = f.get_tensor(k)
                    continue
                touched.add(L)
                n_exp += nm
                state["mats"][str(L)] = state["mats"].get(str(L), 0) + nm
        if res_part:
            save_file(res_part, part_path.format(si))
            if si not in state["parts"]:
                state["parts"].append(si)
        del res_part
        gc.collect()
        for fd in fds8 + fds4:
            os.fsync(fd)
        os.remove(path)
        state["done"].append(shard)
        state["shard_layers"][shard] = sorted(touched)
        json.dump(state, open(state_path, "w"))
        upload_complete_layers()
        log("done", shard, f"({si + 1}/{len(shards)})", "| expert matrices so far this run:", n_exp, "| free GB:",
            round(shutil.disk_usage(HERE).free / 2**30, 1))
    for fd in fds8 + fds4:
        os.close(fd)
    incomplete = {L: state["mats"].get(str(L), 0) for L in range(NL) if state["mats"].get(str(L), 0) != 3 * NE}
    assert not incomplete, f"layers with a matrix count != 3*NE: {incomplete}"
    parts = [part_path.format(si) for si in sorted(state["parts"])]
    res_path = os.path.join(HERE, "resident.safetensors")
    n_res = merge_safetensors(parts, res_path)
    resident_params = 0
    with safe_open(res_path, framework="pt") as f:
        for k in f.keys():
            resident_params += int(np.prod(f.get_slice(k).get_shape()))
    for p in parts:
        os.remove(p)
    if HF_REPO_UP:
        missing = [L for L in range(NL) if L not in state["uploaded"]]
        assert not missing, f"layers never completed/uploaded: {missing}"
    for meta, store in ((meta8, STORE), (meta4, STORE4)):
        meta["resident_params"] = resident_params
        meta["expert_params"] = NL * NE * 3 * MAT
        json.dump(meta, open(os.path.join(store, "meta.json"), "w"), indent=1)
    log("hashing")
    with open(os.path.join(HERE, "sha256.txt"), "w") as fh:
        for rel in ("store/layer_00.bin", "store4/layer_00.bin", "resident.safetensors"):
            fh.write(f"{sha256(os.path.join(HERE, rel))}  {rel}\n")
    shutil.rmtree(WORK, ignore_errors=True)
    os.remove(state_path)
    log("CONVERT_DONE resident tensors", n_res, "resident params", resident_params, "expert params", NL * NE * 3 * MAT,
        "store GB", round(NL * NE * REC / 2**30, 2), "store4 GB", round(NL * NE * REC4 / 2**30, 2))


if __name__ == "__main__":
    main()
