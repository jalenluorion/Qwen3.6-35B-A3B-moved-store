"""Slim phone layout: write DIR/embeddings.safetensors, the input embedding table alone.

The tensor keeps its name, dtype (BF16) and shape from resident.safetensors and its bytes are copied verbatim, so the
native runtime's per-token row read (DiskEmbedding: one pread at data_start + token * row_bytes) is the same code on
either file. The script verifies the copied bytes against the slice of resident.safetensors (SHA-256 of both) and
records the new file's SHA-256 in DIR/sha256.txt.

    python3 make_embeddings.py [--dir model]
"""
import argparse, hashlib, json, os, struct, sys

KEY_SUFFIX = "embed_tokens.weight"
CHUNK = 64 << 20


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
    a = ap.parse_args()
    src = os.path.join(a.dir, "resident.safetensors")
    dst = os.path.join(a.dir, "embeddings.safetensors")
    h, data0 = header(src)
    keys = [k for k in h if k.endswith(KEY_SUFFIX) and not k.startswith("mtp.")]
    assert len(keys) == 1, keys
    key = keys[0]
    e = h[key]
    b0, b1 = e["data_offsets"]
    nbytes = b1 - b0
    rows, H = e["shape"]
    assert e["dtype"] == "BF16" and nbytes == rows * H * 2, e
    meta = {"format": "pt", "source": "resident.safetensors " + key + ", bytes copied verbatim (phone-store slim layout)"}
    hj = json.dumps({"__metadata__": meta, key: {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [0, nbytes]}},
                    separators=(",", ":")).encode()
    hj += b" " * ((-len(hj)) % 8)  # data starts 8-byte aligned, as the safetensors writers pad it
    tmp = dst + ".part"
    hs, hd, hf = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        pre = struct.pack("<Q", len(hj)) + hj
        fo.write(pre); hf.update(pre)
        fi.seek(data0 + b0)
        left = nbytes
        while left:
            buf = fi.read(min(CHUNK, left))
            assert buf, "short read"
            left -= len(buf)
            hs.update(buf); fo.write(buf); hd.update(buf); hf.update(buf)
    os.replace(tmp, dst)
    # verify: re-read the new file's data region from disk and compare with the source slice's hash
    h2, d2 = header(dst)
    assert h2[key]["data_offsets"] == [0, nbytes] and os.path.getsize(dst) == d2 + nbytes
    hv = hashlib.sha256()
    with open(dst, "rb") as f:
        f.seek(d2)
        while True:
            buf = f.read(CHUNK)
            if not buf: break
            hv.update(buf)
    ok = hv.hexdigest() == hs.hexdigest()
    print(json.dumps({"key": key, "shape": e["shape"], "dtype": e["dtype"], "bytes": nbytes, "file_bytes": os.path.getsize(dst),
                      "header_bytes": d2, "src_slice_sha256": hs.hexdigest(), "dst_data_sha256": hv.hexdigest(),
                      "file_sha256": hf.hexdigest(), "equal": ok}, indent=1))
    if not ok:
        sys.exit("MISMATCH")
    shaf = os.path.join(a.dir, "sha256.txt")
    lines = [l for l in open(shaf).read().splitlines() if not l.endswith("  embeddings.safetensors")] if os.path.exists(shaf) else []
    lines.append(f"{hf.hexdigest()}  embeddings.safetensors")
    open(shaf, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
