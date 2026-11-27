#!/usr/bin/env python3
"""Move the device half_to_float out of kernels.h into the shared engine/fp16.h.

The physical copy now lives in include/engine/fp16.h as engine::fp16::kDeviceSrc, right
next to the host implementation that the loaders use (single file, single review point).
kernels::kQuantCommon is composed from it, so every NVRTC module that included the old
helpers keeps getting them with no call-site changes.
"""
P = 'include/engine/kernels.h'
s = open(P, encoding='utf-8').read()

# 1) include the shared header
inc_old = '#pragma once\n\nnamespace kernels {'
assert s.count(inc_old) == 1
s = s.replace(inc_old, '#pragma once\n\n#include <string>\n\n#include "engine/fp16.h"\n\nnamespace kernels {')

# 2) compose kQuantCommon instead of redefining half_to_float inside the raw string
old = '''// Shared helpers are emitted once and prepended to both dequant programs.
inline const char* kQuantCommon = R"cuda(
__device__ __forceinline__ float half_to_float(unsigned short h) {
  unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
  unsigned int exp = (h >> 10) & 0x1Fu;
  unsigned int mant = h & 0x3FFu;
  unsigned int bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int e = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) { mant <<= 1; --e; }
      mant &= 0x3FFu;
      bits = sign | ((unsigned int)e << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  return __uint_as_float(bits);
}

// 6-bit scale/min pair per 32-element sub-block (ggml get_scale_min_k4)'''
assert s.count(old) == 1
new = '''// Shared helpers are emitted once and prepended to both dequant programs.
// half_to_float comes from engine::fp16::kDeviceSrc (include/engine/fp16.h), which is
// also the source of the host-side conversion -- so the device and host decoders can
// never drift apart again (tests/test_fp16.cpp checks all 65536 patterns).
inline const std::string kQuantCommon = engine::fp16::kDeviceSrc + R"cuda(
// 6-bit scale/min pair per 32-element sub-block (ggml get_scale_min_k4)'''
s = s.replace(old, new)
open(P, 'w', encoding='utf-8', newline='\n').write(s)
print('kernels.h: kQuantCommon now composes engine::fp16::kDeviceSrc')
