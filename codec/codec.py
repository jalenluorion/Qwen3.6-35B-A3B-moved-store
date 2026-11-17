"""Lossless block-parallel entropy coder for phone-store expert records (format "PSC1").

See CODEC.md for the format. Summary:
  record = 108 blocks: 96 nibble blocks (3 matrices x 32 blocks x 16 KiB of uint32 code words) and
           12 side blocks (3 matrices x 4 blocks x 4096 groups of fp16 scale + fp16 bias).
  A block is decoded by one 32-lane SIMD group. Lane j owns output words j, j+32, j+64, ... of the block.
  Every lane runs the same sequence of "steps"; before each step a lane refills its 32-bit MSB-first bit
  buffer with one 16-bit unit if it holds < 12 bits. The units of all 32 lanes are interleaved in the
  order the refills happen (step-major, lane-minor), so a lane finds its unit by a ballot + popcount and
  the block needs only one start offset.
  nibble step: one byte (two nibbles) from Huffman table T0 (length-limited to 12 bits).
  side group (5 steps): scale hi byte (T1), scale lo byte (T2), q symbol (T3: q+128, 255 = escape),
           raw bias hi, raw bias lo (8 raw bits each, only consumed when q is the escape).
           bias = fp16(float(q) * float(scale)) (+0 when q = 0) unless escaped (exact for >99.99% of groups, MLX affine
           quantization stores bias = edge and scale = edge / q0).
  record layout: u32[108] block start offsets (in 16-bit units, from the payload start), then the payload
           (u16 units), zero-padded to 16 bytes.
"""
import numpy as np, json, os, sys, hashlib, time

STORE = os.environ.get('PSC_STORE', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'store4'))  # the raw store4/ to encode
R = 1769472
MAT = 589824          # bytes per matrix in a record
CODE_B = 524288       # code-word bytes per matrix
GROUPS = 16384        # groups per matrix
LMAX = 12             # max Huffman code length (decode table 4096 entries)
REFILL = 12           # refill when fewer bits than this are buffered
NB_NIB = 32           # nibble blocks per matrix
NIB_BLOCK = CODE_B // NB_NIB      # 16384 bytes
NIB_STEPS = NIB_BLOCK // 32       # 512 bytes per lane
NB_SIDE = 4           # side blocks per matrix
SIDE_GROUPS = GROUPS // NB_SIDE   # 4096 groups per block, 128 per lane
SIDE_STEPS = (SIDE_GROUPS // 32) * 5   # 640
NBLK = 3 * (NB_NIB + NB_SIDE)     # 108
HDR = NBLK * 4
ESC = 255


def read_record(layer, e):
    return np.fromfile(f'{STORE}/layer_{layer:02d}.bin', dtype=np.uint8, count=R, offset=e * R)


# ---------------- Huffman ----------------
def limited_lengths(freq, L=LMAX):
    """Package-merge: optimal prefix code lengths with max length L. freq: (256,) counts."""
    freq = np.asarray(freq, dtype=np.float64)
    syms = np.nonzero(freq > 0)[0]
    n = len(syms)
    lens = np.zeros(256, np.int64)
    if n == 1:
        lens[syms[0]] = 1
        return lens
    assert n <= 2 ** L
    leaves = [(freq[s], np.bincount([i], minlength=n)) for i, s in enumerate(syms)]
    leaves.sort(key=lambda x: x[0])
    pk = leaves
    for _ in range(L - 1):
        merged = [(pk[i][0] + pk[i + 1][0], pk[i][1] + pk[i + 1][1]) for i in range(0, len(pk) - 1, 2)]
        pk = sorted(leaves + merged, key=lambda x: x[0])
    cnt = np.zeros(n, np.int64)
    for w, v in pk[:2 * n - 2]:
        cnt += v
    lens[syms] = cnt
    assert (2.0 ** -lens[syms]).sum() <= 1 + 1e-12
    return lens


def canonical(lens):
    """Canonical codes (MSB-first) and the 4096-entry decode table (u16 = sym | len << 8)."""
    order = sorted([s for s in range(256) if lens[s] > 0], key=lambda s: (lens[s], s))
    codes = np.zeros(256, np.int64)
    code, prev = 0, lens[order[0]]
    table = np.zeros(1 << LMAX, np.uint16)
    for i, s in enumerate(order):
        if i:
            code = (code + 1) << (lens[s] - prev)
        prev = lens[s]
        codes[s] = code
        lo = code << (LMAX - lens[s])
        table[lo: lo + (1 << (LMAX - lens[s]))] = s | (lens[s] << 8)
    assert code + 1 == 1 << prev or len(order) == 1, 'code not complete'
    return codes, table


# ---------------- symbol extraction ----------------
def side_symbols(s16, b16):
    """s16, b16: (G,) uint16 scale / bias bit patterns -> hi, lo, qsym, esc mask, raw."""
    sf = s16.view(np.float16).astype(np.float32)
    bf = b16.view(np.float16).astype(np.float32)
    with np.errstate(all='ignore'):
        q = np.round(bf / sf)
    ok = np.isfinite(q) & (np.abs(q) <= 127)
    qi = np.where(ok, q, 0).astype(np.int64)
    with np.errstate(all='ignore'):
        pred = (qi.astype(np.float32) * sf).astype(np.float16).view(np.uint16)
    pred = np.where(qi == 0, 0, pred)          # q = 0 predicts +0 (dead groups: scale -1e-7, bias +0)
    esc = ~(ok & (pred == b16))
    qsym = np.where(esc, ESC, qi + 128)
    return (s16 >> 8).astype(np.int64), (s16 & 255).astype(np.int64), qsym, esc


def record_symbols(rec):
    """Returns nibble-byte symbols (3*32 blocks, 32 lanes, 512) and side symbol tuples per matrix."""
    nib, side = [], []
    for m in range(3):
        base = m * MAT
        codes = rec[base: base + CODE_B]
        # block, k (128 words per lane), lane, byte -> block, lane, k, byte
        nib.append(codes.reshape(NB_NIB, NIB_STEPS // 4, 32, 4).transpose(0, 2, 1, 3).reshape(NB_NIB, 32, NIB_STEPS))
        s16 = rec[base + CODE_B: base + CODE_B + 2 * GROUPS].view(np.uint16)
        b16 = rec[base + CODE_B + 2 * GROUPS: base + MAT].view(np.uint16)
        side.append((s16, b16))
    return np.concatenate(nib).astype(np.int64), side


# ---------------- tables ----------------
def layer_counts(layer, records):
    c = np.zeros((4, 256), np.int64)
    nesc = 0
    for e in records:
        rec = read_record(layer, e)
        nib, side = record_symbols(rec)
        c[0] += np.bincount(nib.ravel(), minlength=256)
        for s16, b16 in side:
            hi, lo, qs, esc = side_symbols(s16, b16)
            c[1] += np.bincount(hi, minlength=256)
            c[2] += np.bincount(lo, minlength=256)
            c[3] += np.bincount(qs, minlength=256)
            nesc += int(esc.sum())
    return c, nesc


def build_tables(counts):
    lens = np.stack([limited_lengths(counts[t]) for t in range(4)])
    codes, tables = [], []
    for t in range(4):
        cd, tb = canonical(lens[t])
        codes.append(cd); tables.append(tb)
    return lens, np.stack(codes), np.stack(tables)


# ---------------- lane interleaving ----------------
def encode_blocks(vals, nbits):
    """vals, nbits: (nb, 32, ns). Returns list of per-block u16 unit arrays."""
    nb, nl, ns = nbits.shape
    lanes = nb * nl
    nbf = nbits.reshape(lanes, ns)
    vf = vals.reshape(lanes, ns)
    cnt = np.zeros(lanes, np.int64)
    need = np.zeros((lanes, ns), bool)
    for t in range(ns):
        nd = cnt < REFILL
        need[:, t] = nd
        cnt += 16 * nd - nbf[:, t]
    assert (cnt >= 0).all()
    nunits = need.sum(1)                              # units per lane
    unit_off = np.concatenate([[0], np.cumsum(nunits)])[:-1]
    bit_off = unit_off * 16
    # bits of each lane, MSB-first, zero padded to nunits*16
    bits = np.zeros(int(nunits.sum()) * 16, np.uint8)
    excl = np.cumsum(nbf, axis=1) - nbf
    start = bit_off[:, None] + excl
    for j in range(LMAX):
        mk = nbf > j
        if not mk.any():
            break
        bits[start[mk] + j] = (vf[mk] >> (nbf[mk] - 1 - j)) & 1
    by = np.packbits(bits)
    units = (by[0::2].astype(np.uint16) << 8) | by[1::2]
    # stream order: block, step, lane
    kidx = np.cumsum(need, axis=1) - 1
    need3 = need.reshape(nb, nl, ns).transpose(0, 2, 1)       # (nb, ns, nl)
    b, t, l = np.nonzero(need3)
    lane = b * nl + l
    stream = units[unit_off[lane] + kidx[lane, t]]
    per_block = need.reshape(nb, nl * ns).sum(1)
    out, p = [], 0
    for k in per_block:
        out.append(stream[p: p + k]); p += k
    return out, int((nunits * 16).sum() - nbf.sum())          # wasted tail bits


def encode_record(rec, lens, codes):
    nib, side = record_symbols(rec)
    blocks, waste = encode_blocks(codes[0][nib], lens[0][nib])
    sv, sn = [], []
    nesc = 0
    for s16, b16 in side:
        hi, lo, qs, esc = side_symbols(s16, b16)
        nesc += int(esc.sum())
        raw = b16.astype(np.int64)
        st = np.stack([codes[1][hi], codes[2][lo], codes[3][qs], raw >> 8, raw & 255], 1)       # (G, 5)
        sb = np.stack([lens[1][hi], lens[2][lo], lens[3][qs], 8 * esc, 8 * esc], 1)
        # group g = blk*4096 + 2*(32k + lane) + gi  -> (blk, k, lane, gi, 5) -> (blk, lane, k, gi, 5)
        sh = (NB_SIDE, SIDE_GROUPS // 64, 32, 2, 5)
        sv.append(st.reshape(sh).transpose(0, 2, 1, 3, 4).reshape(NB_SIDE, 32, SIDE_STEPS))
        sn.append(sb.reshape(sh).transpose(0, 2, 1, 3, 4).reshape(NB_SIDE, 32, SIDE_STEPS))
    sblocks, w2 = encode_blocks(np.concatenate(sv), np.concatenate(sn))
    allb = blocks + sblocks
    offs = np.concatenate([[0], np.cumsum([len(b) for b in allb])])[:-1].astype(np.uint32)
    payload = np.concatenate(allb).astype('<u2')
    out = offs.astype('<u4').tobytes() + payload.tobytes()
    out += b'\0' * ((-len(out)) % 16)
    st = dict(nib_units=int(sum(len(b) for b in blocks)), side_units=int(sum(len(b) for b in sblocks)),
              waste_bits=waste + w2, esc=nesc)
    return out, st


# ---------------- reference decoder (numpy, lockstep lanes) ----------------
def decode_blocks(units, nsteps, step_table, raw_steps=None, esc_from=None):
    """units: list of per-block u16 arrays. step_table[t] = table (4096,) or None for raw.
    Returns (nb, 32, nsteps) symbols."""
    nb = len(units)
    out = np.zeros((nb, 32, nsteps), np.int64)
    for b in range(nb):
        u = units[b].astype(np.int64)
        buf = np.zeros(32, np.int64); cnt = np.zeros(32, np.int64); pos = 0
        esc = np.zeros(32, bool)
        for t in range(nsteps):
            nd = cnt < REFILL
            idx = pos + np.cumsum(nd) - nd
            buf = np.where(nd, buf | (u[np.minimum(idx, len(u) - 1)] << (16 - cnt)), buf) & 0xFFFFFFFF
            cnt = cnt + 16 * nd
            pos += int(nd.sum())
            tb = step_table(t)
            if tb is None:          # raw 8 bits if escaped
                sym = np.where(esc, buf >> 24, 0); ln = 8 * esc
            else:
                e = tb[buf >> 20].astype(np.int64); sym = e & 255; ln = e >> 8
            buf = (buf << ln) & 0xFFFFFFFF; cnt -= ln
            out[b, :, t] = sym
            if esc_from is not None and esc_from(t):
                esc = sym == ESC
    return out


def decode_record_ref(blob, tables):
    offs = np.frombuffer(blob[:HDR], '<u4').astype(np.int64)
    pay = np.frombuffer(blob[HDR:], '<u2')
    ends = np.concatenate([offs[1:], [len(pay)]])
    units = [pay[offs[i]: ends[i]] for i in range(NBLK)]
    rec = np.zeros(R, np.uint8)
    nibs = decode_blocks(units[:96], NIB_STEPS, lambda t: tables[0])
    side = decode_blocks(units[96:], SIDE_STEPS, lambda t: [tables[1], tables[2], tables[3], None, None][t % 5],
                         esc_from=lambda t: t % 5 == 2)
    for m in range(3):
        base = m * MAT
        blk = nibs[m * 32:(m + 1) * 32].reshape(NB_NIB, 32, NIB_STEPS // 4, 4).transpose(0, 2, 1, 3)
        rec[base: base + CODE_B] = blk.reshape(-1).astype(np.uint8)
        sd = side[m * 4:(m + 1) * 4].reshape(NB_SIDE, 32, SIDE_GROUPS // 64, 2, 5).transpose(0, 2, 1, 3, 4).reshape(GROUPS, 5)
        s16 = ((sd[:, 0] << 8) | sd[:, 1]).astype(np.uint16)
        q = sd[:, 2]
        pred = ((np.where(q == ESC, 0, q - 128)).astype(np.float32) * s16.view(np.float16).astype(np.float32)).astype(np.float16).view(np.uint16)
        pred = np.where(q == 128, 0, pred)
        b16 = np.where(q == ESC, (sd[:, 3] << 8) | sd[:, 4], pred).astype(np.uint16)
        rec[base + CODE_B: base + CODE_B + 2 * GROUPS] = s16.view(np.uint8)
        rec[base + CODE_B + 2 * GROUPS: base + MAT] = b16.view(np.uint8)
    return rec


def entropy(c):
    c = np.asarray(c, np.float64); p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


# ---------------- fast reference decoder (numpy, all blocks of a kind in lockstep) ----------------
# Same bit-level rule as decode_blocks/decode_record_ref, vectorized over blocks: every nibble block runs the same 512
# steps and every side block the same 640, so one step is a handful of (nb, 32) array ops. Used to
# verify every record of every layer on CPU (~0.1 s per record instead of ~seconds).
def _decode_blocks_vec(units, nsteps, tabsel, esc_after=None):
    nb = len(units)
    width = max(len(u) for u in units) + 1
    U = np.zeros((nb, width), np.int64)
    for b, u in enumerate(units):
        U[b, :len(u)] = u
    rows = np.arange(nb)[:, None]
    buf = np.zeros((nb, 32), np.int64); cnt = np.zeros((nb, 32), np.int64); pos = np.zeros((nb, 1), np.int64)
    esc = np.zeros((nb, 32), bool)
    out = np.empty((nb, 32, nsteps), np.int64)
    for t in range(nsteps):
        nd = cnt < REFILL
        c = np.cumsum(nd, axis=1)
        idx = np.minimum(pos + c - nd, width - 1)
        sh = np.where(nd, 16 - cnt, 0)
        buf = np.where(nd, (buf | (U[rows, idx] << sh)) & 0xFFFFFFFF, buf)
        cnt += 16 * nd
        pos += c[:, -1:]
        tb = tabsel(t)
        if tb is None:
            sym = np.where(esc, buf >> 24, 0); ln = 8 * esc
        else:
            e = tb[buf >> 20]; sym = e & 255; ln = e >> 8
        buf = (buf << ln) & 0xFFFFFFFF; cnt -= ln
        out[:, :, t] = sym
        if esc_after is not None and esc_after(t):
            esc = sym == ESC
    return out


def decode_record_fast(blob, tables):
    """Bit-identical to decode_record_ref (same format rules), vectorized; returns the (R,) uint8 record."""
    T = [np.asarray(t, np.int64) for t in tables]
    offs = np.frombuffer(blob[:HDR], '<u4').astype(np.int64)
    pay = np.frombuffer(blob[HDR:], '<u2')
    ends = np.concatenate([offs[1:], [len(pay)]])
    units = [pay[offs[i]: ends[i]] for i in range(NBLK)]
    rec = np.zeros(R, np.uint8)
    nibs = _decode_blocks_vec(units[:96], NIB_STEPS, lambda t: T[0])
    side = _decode_blocks_vec(units[96:], SIDE_STEPS, lambda t: [T[1], T[2], T[3], None, None][t % 5],
                              esc_after=lambda t: t % 5 == 2)
    for m in range(3):
        base = m * MAT
        blk = nibs[m * 32:(m + 1) * 32].reshape(NB_NIB, 32, NIB_STEPS // 4, 4).transpose(0, 2, 1, 3)
        rec[base: base + CODE_B] = blk.reshape(-1).astype(np.uint8)
        sd = side[m * 4:(m + 1) * 4].reshape(NB_SIDE, 32, SIDE_GROUPS // 64, 2, 5).transpose(0, 2, 1, 3, 4).reshape(GROUPS, 5)
        s16 = ((sd[:, 0] << 8) | sd[:, 1]).astype(np.uint16)
        q = sd[:, 2]
        with np.errstate(all='ignore'):
            pred = ((np.where(q == ESC, 0, q - 128)).astype(np.float32) * s16.view(np.float16).astype(np.float32)).astype(np.float16).view(np.uint16)
        pred = np.where(q == 128, 0, pred)
        b16 = np.where(q == ESC, (sd[:, 3] << 8) | sd[:, 4], pred).astype(np.uint16)
        rec[base + CODE_B: base + CODE_B + 2 * GROUPS] = s16.view(np.uint8)
        rec[base + CODE_B + 2 * GROUPS: base + MAT] = b16.view(np.uint8)
    return rec


def read_idx(path):
    """Parse a PSC1 .idx: returns dict(layer, nrec, record_bytes, lens (4,256), tables (4,4096), offsets, sha256 list)."""
    b = open(path, 'rb').read()
    assert b[:4] == b'PSC1', 'bad magic'
    ver, layer, nrec, rb = np.frombuffer(b[4:20], '<u4').tolist()
    p = 32
    lens = np.frombuffer(b[p: p + 1024], np.uint8).reshape(4, 256); p += 1024
    tables = np.frombuffer(b[p: p + 4 * 4096 * 2], '<u2').reshape(4, 4096); p += 4 * 4096 * 2
    offsets = np.frombuffer(b[p: p + 8 * (nrec + 1)], '<u8').astype(np.int64); p += 8 * (nrec + 1)
    shas = [b[p + 32 * i: p + 32 * (i + 1)] for i in range(nrec)]
    assert p + 32 * nrec == len(b), 'idx size mismatch'
    return dict(version=ver, layer=layer, nrec=nrec, record_bytes=rb, lens=lens, tables=tables, offsets=offsets, shas=shas)
