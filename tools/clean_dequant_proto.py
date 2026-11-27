#!/usr/bin/env python3
"""Fix up the dequant prototype: remove invalid leftovers and the extrapolation block."""
p = 'tests/dequant_proto.h'
s = open(p, encoding='utf-8').read()

# 1) q4k x8 kernel: bogus 'memcpy:' label + local array copy
old = '''  const unsigned char* qp = qs + g * 32 + l0;
  unsigned char qb[8];
  memcpy: for (int i = 0; i < 8; ++i) qb[i] = qp[i];
  const int sh = half * 4;
  unsigned short v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = (qb[i] >> sh) & 0xF;
    v[i] = pf_half(scale * (float)q - bias);
  }
  out[b * 32 + lane] = pack8(v);'''
new = '''  const unsigned char* qp = qs + g * 32 + l0;
  const int sh = half * 4;
  unsigned short v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = (qp[i] >> sh) & 0xF;
    v[i] = pf_half(scale * (float)q - bias);
  }
  out[b * 32 + lane] = pack8(v);'''
assert s.count(old) == 1
s = s.replace(old, new)

# 2) host: remove bogus label and the extrapolation block
old2 = '''    using_share: (void)0;
    total_old += best_old;
    total_new += best_new;
    total_gb += c.model_gb;
  }
  // extrapolate to the whole 9B dense weight set (6.1 GB): the three matrices above
  // together move 110 MB in the same proportion as the model.
  std::printf("  -> per-110 MB: old %.3f ms, x8 %.3f ms; whole-model (6.1 GB) ~ old %.1f ms, "
              "x8 %.1f ms\\n",
              total_old, total_new, total_old * (6.1 / (110.0 / 1024.0)),
              total_new * (6.1 / (110.0 / 1024.0)));
}'''
new2 = '''    total_old += best_old;
    total_new += best_new;
    total_gb += c.model_gb;
  }
  std::printf("  -> the 110 MB of slabs above: old %.3f ms, x8 %.3f ms (%.2fx); whole-model "
              "6.1 GB at these rates: old ~%.0f ms, x8 ~%.0f ms\\n",
              total_old, total_new, total_old / total_new, total_old * 6.1 * 1024.0 / 110.0,
              total_new * 6.1 * 1024.0 / 110.0);
}'''
assert s.count(old2) == 1
s = s.replace(old2, new2)

# 3) Case.model_gb is now only kept for the summary; keep it but document it
s = s.replace('double model_gb;        // share of the 9B weights this type covers (for the estimate)',
              'double model_gb;        // approximate share of the 9B weights of this type')
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('dequant prototype cleaned')
