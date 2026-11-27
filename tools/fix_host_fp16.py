#!/usr/bin/env python3
"""Fix the two host-side fp16 decoders in runner.cpp (F16 subnormal renormalisation).

Both copies used `bits = sign | (112u << 23) | (mant << 13)` for exp == 0, which treats
an fp16 subnormal as if it were a normal number (512x too large for the smallest one).
They now call engine::fp16::half_to_float, the shared implementation that
tests/test_fp16.cpp checks bit-exactly for all 65536 patterns.
"""
P = 'src/runner.cpp'
s = open(P, encoding='utf-8').read()

# include the shared header next to the other engine includes
if '#include "engine/fp16.h"' not in s:
    anchor = '#include "engine/gguf.h"'
    assert s.count(anchor) >= 1
    s = s.replace(anchor, '#include "engine/cuda.h"\n#include "engine/fp16.h"', 1)

old1 = '''        uint32_t bits;
        if (t.dtype == "BF16") {
          bits = static_cast<uint32_t>(raw[i]) << 16;
        } else {  // F16
          const uint32_t h = raw[i];
          const uint32_t sign = (h & 0x8000u) << 16;
          uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
          if (exp == 0) bits = sign | (mant ? (112u << 23) | (mant << 13) : 0);
          else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
          else bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
        std::memcpy(&out[i], &bits, 4);'''
assert s.count(old1) == 1, s.count(old1)
new1 = '''        if (t.dtype == "BF16") {
          const uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
          std::memcpy(&out[i], &bits, 4);
        } else {  // F16
          out[i] = engine::fp16::half_to_float(raw[i]);
        }'''
s = s.replace(old1, new1)

old2 = '''      uint32_t bits;
      if (ti->type_name == "BF16") {
        bits = static_cast<uint32_t>(h) << 16;
      } else {
        const uint32_t sign = (h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
        if (exp == 0) bits = sign | (mant ? (112u << 23) | (mant << 13) : 0);
        else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
        else bits = sign | ((exp + 112u) << 23) | (mant << 13);
      }
      std::memcpy(&out[i], &bits, 4);'''
assert s.count(old2) == 1, s.count(old2)
new2 = '''      if (ti->type_name == "BF16") {
        const uint32_t bits = static_cast<uint32_t>(h) << 16;
        std::memcpy(&out[i], &bits, 4);
      } else {
        out[i] = engine::fp16::half_to_float(h);
      }'''
s = s.replace(old2, new2)

# any other hand-rolled fp16 decode in the loaders must not come back
open(P, 'w', encoding='utf-8', newline='\n').write(s)
print('runner.cpp: both host F16 decoders now use engine::fp16::half_to_float')
