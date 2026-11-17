"""Encode store4 layers into PSC1 files.

usage: python encode_layers.py --layers 0 20 [--write] [--stats-only 10 39] [--pack 20 200]
  --write        write out/layer_XX.psc + out/layer_XX.idx for the --layers
  --pack L N     write out/pack_LXX/ with the first N records of layer L (a small test pack)
Index file (.idx, little-endian): 'PSC1', version=1, layer, nrec, record_bytes, 3 x reserved (u32 each, 32 B);
  u8 lens[4][256]; u16 tables[4][4096] (entry = sym | len << 8); u64 offsets[nrec+1] into the .psc;
  u8 sha256[nrec][32] of the ORIGINAL record bytes.
"""
import argparse, hashlib, json, os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codec import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')


def write_idx(path, layer, lens, tables, offsets, shas):
    with open(path, 'wb') as f:
        f.write(b'PSC1' + struct.pack('<7I', 1, layer, len(shas), R, 0, 0, 0))
        f.write(lens.astype(np.uint8).tobytes())
        f.write(tables.astype('<u2').tobytes())
        f.write(np.asarray(offsets, '<u8').tobytes())
        for h in shas:
            f.write(h)


def do_layer(layer, write, nrec=256, out_dir=None, tag=None, check_ref=2):
    t0 = time.time()
    counts, nesc = layer_counts(layer, range(256))      # tables always from the whole layer
    lens, codes, tables = build_tables(counts)
    t1 = time.time()
    sizes, st_all, shas = [], [], []
    f = None
    if write:
        os.makedirs(out_dir, exist_ok=True)
        f = open(os.path.join(out_dir, f'{tag}.psc'), 'wb')
    offsets = [0]
    for e in range(nrec):
        rec = read_record(layer, e)
        blob, st = encode_record(rec, lens, codes)
        if e < check_ref:   # numpy reference decoder round trip
            assert np.array_equal(decode_record_ref(blob, tables), rec), f'ref decode mismatch L{layer} e{e}'
        sizes.append(len(blob)); st_all.append(st)
        shas.append(hashlib.sha256(rec.tobytes()).digest())
        if f:
            f.write(blob)
        offsets.append(offsets[-1] + len(blob))
    if f:
        f.close()
        write_idx(os.path.join(out_dir, f'{tag}.idx'), layer, lens, tables, offsets, shas)
    sizes = np.array(sizes)
    # entropy bounds (per record averages over the layer)
    n_nib = 3 * 1048576; n_grp = 3 * GROUPS
    nibc = np.zeros(16)
    for b in range(256):
        nibc[b & 15] += counts[0][b]; nibc[b >> 4] += counts[0][b]
    # zero-order fp16 entropy of scales and biases needs the 16-bit histograms
    sc = np.zeros(65536); bc = np.zeros(65536)
    for e in range(0, 256, 8):
        rec = read_record(layer, e)
        for m in range(3):
            base = m * MAT
            sc += np.bincount(rec[base + CODE_B: base + CODE_B + 2 * GROUPS].view(np.uint16), minlength=65536)
            bc += np.bincount(rec[base + CODE_B + 2 * GROUPS: base + MAT].view(np.uint16), minlength=65536)
    H_nib, H_byte = entropy(nibc), entropy(counts[0])
    H_s, H_b, H_q = entropy(sc), entropy(bc), entropy(counts[3])
    esc_per_rec = nesc / 256
    bound_design = (n_nib * H_nib + n_grp * (H_s + H_b)) / 8 / R
    bound_model = (n_nib / 2 * H_byte + n_grp * (H_s + H_q) + esc_per_rec * 16) / 8 / R
    huff_bits = lambda t: (counts[t] * lens[t]).sum() / 256
    huff_payload = (huff_bits(0) + huff_bits(1) + huff_bits(2) + huff_bits(3) + esc_per_rec * 16) / 8
    res = dict(layer=layer, records=nrec, ratio_mean=float(sizes.mean() / R), ratio_min=float(sizes.min() / R),
               ratio_max=float(sizes.max() / R), bytes_mean=float(sizes.mean()),
               bound_design_0885=bound_design, bound_model=bound_model,
               huffman_payload_ratio=float(huff_payload / R),
               header_ratio=HDR / R, tail_waste_ratio=float(np.mean([s['waste_bits'] for s in st_all]) / 8 / R),
               nib_bits_per_weight=float(np.mean([s['nib_units'] for s in st_all]) * 16 / n_nib),
               side_bytes=float(np.mean([s['side_units'] for s in st_all]) * 2),
               H_nibble=H_nib, H_byte_per_nibble=H_byte / 2, H_scale16=H_s, H_bias16=H_b, H_q=H_q,
               huff_T0_bits_per_byte=float(huff_bits(0) / (n_nib / 2)), esc_per_record=esc_per_rec,
               maxlen=lens.max(1).tolist(), sec_tables=t1 - t0, sec_encode=time.time() - t1)
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, nargs='*', default=[])
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--pack', type=int, nargs=2, default=None)
    a = ap.parse_args()
    allres = []
    for L in a.layers:
        allres.append(do_layer(L, a.write, out_dir=OUT, tag=f'layer_{L:02d}'))
    if a.pack:
        L, N = a.pack
        allres.append(do_layer(L, True, nrec=N, out_dir=os.path.join(OUT, f'pack_L{L:02d}'), tag='pack', check_ref=0))
    p = os.path.join(OUT, 'encode_stats.json')
    old = json.load(open(p)) if os.path.exists(p) else {}
    for r in allres:
        old[f"L{r['layer']}" + ('' if r['records'] == 256 else f"_n{r['records']}")] = r
    json.dump(old, open(p, 'w'), indent=1)
