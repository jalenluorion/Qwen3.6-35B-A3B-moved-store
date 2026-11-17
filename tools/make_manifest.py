#!/usr/bin/env python3
"""Write MODEL/manifest.json: the file list the app's ModelDownloader fetches from the Hugging Face repo.

    python3 tools/make_manifest.py [--model model] [--out model/manifest.json] [--no-codec] [--rehash]
                                   [--codec-manifest F] [--codec-dir codec/out] [--repo USER/NAME]

One entry per file of the slim layout, all under --model:
  store4/meta.json, store4/layer_00..39.bin
  cache/resident_q4.safetensors, cache/routers_k4_r8.safetensors   (NOT the *.mmap.* copies the engine may write)
  embeddings.safetensors + the seven small files
plus, optional (required: false, group "codec"), the compressed codec layers layer_XX.{psc,idx} (repo path
optional/codec/store4/..., local path store4/...; the app reads a layer compressed when both files are present).
The codec entries come from MODEL/codec_manifest.json when it exists (sizes, sha256 and ratios from the build), else
from whatever CODEC_DIR/layer_XX.{psc,idx} exist. codec_manifest.json itself is listed too (group "codec").

Each entry: path (in the repo), local (relative to Documents/model/), size, sha256, required, source.
sha256: reused from MODEL/sha256.txt and from a previous manifest when the size and mtime match; otherwise computed
(1 MiB streaming reads). --rehash recomputes everything.
"""
import argparse, hashlib, json, os, sys, time

PS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PS, "model")       # --model
CACHE = os.path.join(SRC, "cache")
CODEC = os.path.join(PS, "codec", "out")  # --codec-dir
SMALL = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
         "chat_template.jinja", "generation_config.json"]
REPO = "jalenluorion/phone-store-qwen36-35b-a3b"


def codec_from_manifest(p):
    cm = json.load(open(p))
    out = [dict(path="optional/codec/codec_manifest.json", local="codec_manifest.json", src=p, size=os.path.getsize(p),
                sha256=None, layer=None, ratio=cm["ratio"])]
    for f in cm["files"]:
        out.append(dict(path=f["path"], local=f["local"], src=None, size=f["size"], sha256=f["sha256"], layer=f["layer"],
                        ratio=f["ratio"]))
    return cm, out


def entries(codec):
    e = [("store4/meta.json", "store4/meta.json", os.path.join(SRC, "store4/meta.json"), True, "store")]
    for l in range(40):
        e.append((f"store4/layer_{l:02d}.bin", f"store4/layer_{l:02d}.bin", os.path.join(SRC, f"store4/layer_{l:02d}.bin"), True, "store"))
    for f in ["resident_q4.safetensors", "routers_k4_r8.safetensors"]:
        e.append((f"cache/{f}", f"cache/{f}", os.path.join(CACHE, f), True, "cache"))
    e.append(("embeddings.safetensors", "embeddings.safetensors", os.path.join(SRC, "embeddings.safetensors"), True, "embeddings"))
    for f in SMALL:
        e.append((f, f, os.path.join(SRC, f), True, "small"))
    if codec:
        for l in range(40):
            psc, idx = os.path.join(CODEC, f"layer_{l:02d}.psc"), os.path.join(CODEC, f"layer_{l:02d}.idx")
            if os.path.isfile(psc) and os.path.isfile(idx):
                for ext, p in (("idx", idx), ("psc", psc)):
                    e.append((f"optional/codec/store4/layer_{l:02d}.{ext}", f"store4/layer_{l:02d}.{ext}", p, False, "codec"))
    return e


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb", buffering=0) as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    global SRC, CACHE, CODEC, REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SRC, help="the slim-layout model folder")
    ap.add_argument("--out", default=None, help="default: MODEL/manifest.json")
    ap.add_argument("--no-codec", action="store_true")
    ap.add_argument("--rehash", action="store_true")
    ap.add_argument("--codec-manifest", default=None, help="default: MODEL/codec_manifest.json")
    ap.add_argument("--codec-dir", default=CODEC, help="encoded layers (codec/encode_layers.py --write)")
    ap.add_argument("--repo", default=REPO, help="the Hugging Face repo the manifest describes")
    a = ap.parse_args()
    SRC = os.path.abspath(a.model)
    CACHE, CODEC, REPO = os.path.join(SRC, "cache"), os.path.abspath(a.codec_dir), a.repo
    a.out = a.out or os.path.join(SRC, "manifest.json")
    a.codec_manifest = a.codec_manifest or os.path.join(SRC, "codec_manifest.json")
    cm = None
    if not a.no_codec and os.path.isfile(a.codec_manifest):
        cm, cfiles = codec_from_manifest(a.codec_manifest)

    known = {}  # source path -> (size, mtime or None, sha)
    st = os.path.join(SRC, "sha256.txt")
    if os.path.isfile(st):
        for line in open(st):
            parts = line.split()
            if len(parts) == 2:
                known[os.path.join(SRC, parts[1])] = (None, None, parts[0])
    if os.path.isfile(a.out) and not a.rehash:
        try:
            old = json.load(open(a.out))
            for f in old.get("files", []):
                if f.get("source") and f.get("mtime") is not None:
                    known[os.path.join(PS, f["source"])] = (f["size"], f["mtime"], f["sha256"])
        except Exception as ex:
            print(f"previous manifest ignored: {ex}", file=sys.stderr)

    files, t0, hashed = [], time.time(), 0
    for path, local, src, req, kind in entries(not a.no_codec and cm is None):
        if not os.path.isfile(src):
            sys.exit(f"missing: {src}")
        s = os.stat(src)
        k = None if a.rehash else known.get(src)
        if k and (k[0] is None or (k[0] == s.st_size and k[1] == int(s.st_mtime))):
            sha, how = k[2], "reused"
        else:
            t = time.time()
            sha = sha256_file(src)
            dt = time.time() - t
            hashed += s.st_size
            how = f"hashed {s.st_size / 1e6 / max(dt, 1e-6):.0f} MB/s"
        print(f"{path:44s} {s.st_size:14d}  {sha[:16]}  {how}")
        files.append({"path": path, "local": local, "size": s.st_size, "sha256": sha, "required": req, "kind": kind,
                      "source": os.path.relpath(src, PS), "mtime": int(s.st_mtime)})

    if cm is not None:
        for c in cfiles:
            sha = c["sha256"] or sha256_file(c["src"])
            files.append({"path": c["path"], "local": c["local"], "size": c["size"], "sha256": sha, "required": False,
                          "kind": "codec", "group": "codec", "layer": c["layer"], "ratio": c["ratio"],
                          "source": os.path.relpath(c["src"], PS) if c["src"] else "hub:" + c["path"], "mtime": None})
            print(f"{c['path']:44s} {c['size']:14d}  {sha[:16]}  codec manifest")
    else:
        for f in files:
            if f["kind"] == "codec":
                f["group"] = "codec"
    req = [f for f in files if f["required"]]
    cod = [f for f in files if f.get("group") == "codec" and f["path"].endswith((".psc", ".idx"))]
    man = {
        "format": "phonestore.manifest.v1",
        "version": time.strftime("%Y-%m-%d"),
        "model": "Qwen3.6-35B-A3B",
        "upstream": "https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
        "license": "Apache-2.0",
        "note": "A converted copy of Qwen/Qwen3.6-35B-A3B for the phone-store iOS app: int4 expert store (store4/, one "
                "file per layer, 256 records of 1,769,472 B), int4 resident model and int8 lag-4 routers (cache/), "
                "bf16 embeddings, the upstream tokenizer and config files. Apache-2.0, as the original.",
        "repo": REPO,
        "layout": "slim",
        "required_bytes": sum(f["size"] for f in req),
        "required_files": len(req),
        "total_bytes": sum(f["size"] for f in files),
        "total_files": len(files),
        "codec_total_bytes": sum(f["size"] for f in cod),
        "codec_files": len(cod),
        "codec_layers": sorted({int(f["local"][13:15]) for f in cod}),
        "codec_ratio": (sum(f["size"] for f in cod) /
                        sum(f["size"] for f in files if f["kind"] == "store" and f["path"][13:15].isdigit()
                            and int(f["path"][13:15]) in {int(c["local"][13:15]) for c in cod})) if cod else None,
        "codec_note": "optional/codec/: lossless PSC1 copies of the store4 layers (codec/CODEC.md). "
                      "A layer with both store4/layer_XX.psc and .idx is read compressed and its .bin is not needed; "
                      "decoded records are bit-identical to the .bin records (SHA-256 of every record in the .idx).",
        "files": files,
    }
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    os.replace(tmp, a.out)
    print(f"required: {len(req)} files, {man['required_bytes']:,} B ({man['required_bytes'] / 1e9:.2f} GB); "
          f"all: {len(files)} files, {man['total_bytes']:,} B; hashed {hashed / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
