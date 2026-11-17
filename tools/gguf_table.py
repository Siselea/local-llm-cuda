import struct, sys, collections

QK = {0:('F32',4),1:('F16',2),2:('Q4_0',0.5625),3:('Q4_1',0.625),6:('Q5_0',0.6875),7:('Q5_1',0.75),
      8:('Q8_0',1.0625),9:('Q8_1',1.125),10:('Q2_K',0.328125),11:('Q3_K',0.4375),12:('Q4_K',0.5625),
      13:('Q5_K',0.6875),14:('Q6_K',0.8203125),15:('Q8_K',1.03125),16:('IQ2_XXS',0.25),17:('IQ2_XS',0.3125)}

f = None

def u32():
    return struct.unpack('<I', f.read(4))[0]

def u64():
    return struct.unpack('<Q', f.read(8))[0]

def s():
    n = u64()
    return f.read(n).decode('utf-8', 'replace')

def read_kv(t):
    if t == 0: return struct.unpack('<B', f.read(1))[0]
    if t == 1: return struct.unpack('<b', f.read(1))[0]
    if t == 2: return struct.unpack('<H', f.read(2))[0]
    if t == 3: return struct.unpack('<h', f.read(2))[0]
    if t == 4: return u32()
    if t == 5: return struct.unpack('<i', f.read(4))[0]
    if t == 6: return struct.unpack('<f', f.read(4))[0]
    if t == 7: return struct.unpack('<?', f.read(1))[0]
    if t == 8: return s()
    if t in (9, 10, 11, 12):
        at = u32(); n = u64()
        base = {9:0,10:4,11:4,12:4}[t]
        for _ in range(n):
            read_kv(at)
        return '<array>'
    raise SystemExit('bad kv type %d' % t)

def main(path, allflag):
    global f
    f = open(path, 'rb')
    if f.read(4) != b'GGUF': raise SystemExit('not gguf')
    ver = u32(); n_tensors = u64(); n_kv = u64()
    kv = {}
    for _ in range(n_kv):
        k = s(); t = u32(); kv[k] = read_kv(t)
    tensors = []
    for _ in range(n_tensors):
        name = s()
        n_dims = u32()
        dims = [u64() for _ in range(n_dims)]
        tt = u32()
        off = u64()
        tensors.append((name, dims, tt, off))
    print('version', ver, 'tensors', n_tensors, 'kv', n_kv)
    for k in ('general.architecture', 'general.name'):
        if k in kv: print(k, '=', kv[k])
    tot = 0
    bycat = collections.Counter(); byq = collections.Counter(); rows = []
    for (name, dims, tt, off) in tensors:
        qn, bpp = QK.get(tt, ('T%d' % tt, 0))
        n = 1
        for d in dims: n *= d
        nb = int(n * bpp)
        tot += nb
        rows.append((name, dims, qn, nb))
        byq[qn] += nb
        nm = name
        if '_exps' in nm: cat = 'experts'
        elif 'output.weight' in nm: cat = 'lm_head'
        elif 'token_embd' in nm: cat = 'embed'
        elif any(x in nm for x in ('attn', 'ssm', 'linear', 'conv', 'time')): cat = 'attn'
        elif 'ffn' in nm: cat = 'ffn'
        else: cat = 'other'
        bycat[cat] += nb
    print('total %.1f MiB' % (tot / 1048576.0))
    print('--- by quant ---')
    for q, b in byq.most_common(): print('  %-8s %8.1f MiB' % (q, b / 1048576.0))
    print('--- by category ---')
    for c, b in bycat.most_common(): print('  %-8s %8.1f MiB' % (c, b / 1048576.0))
    print('--- top 30 by size ---')
    for (name, dims, qn, nb) in sorted(rows, key=lambda r: -r[3])[:30]:
        print('  %-46s %-6s %-20s %8.2f MiB' % (name, qn, 'x'.join(map(str, dims)), nb / 1048576.0))
    if allflag:
        print('--- all ---')
        for (name, dims, qn, nb) in rows:
            print('  %-46s %-6s %-20s %8.2f MiB' % (name, qn, 'x'.join(map(str, dims)), nb / 1048576.0))

main(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == '-all')
