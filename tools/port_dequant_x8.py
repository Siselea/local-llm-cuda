#!/usr/bin/env python3
"""Port the x8 dequant kernels into the product kernel module and use them by default.

Prototype (tests/dequant_proto.h, test_gen --bench-dequant) measured:
  Q4_K 1.70x, Q5_K 1.64x, Q6_K 1.48x faster, ~305 GB/s vs 181-207 GB/s, and the fp16
  slab is BIT-IDENTICAL to the product kernels (memcmp), so model output cannot change.
  Whole-model dequant extrapolates to ~60 ms instead of ~95-115 ms.

The product path launches these as: grid = (n_blocks + 7) / 8, block = 256, one warp
per super-block, 32 lanes x 8 values, uint4 store.  ENGINE_DQ_OLD=1 restores the old
kernels for A/B.
"""
KERNELS = 'include/engine/kernels.h'
RUNNER = 'src/runner.cpp'

X8 = '''
// ---- dequant to fp16, one super-block per warp ---------------------------------
// The original kernels above give every super-block its own 128-thread CUDA block whose
// whole work is two 4-byte stores, so block scheduling dominates: measured 181-207 GB/s
// (HBM is ~331).  This variant packs 8 super-blocks into one 256-thread block (one warp
// each) and has every lane decode 8 consecutive values: one (scale, min) pair per lane
// instead of a branchy q4k_scale_min per value, 8+8 consecutive quant bytes and one
// 16-byte uint4 store.  Measured ~305 GB/s (1.5-1.7x) with a BIT-IDENTICAL fp16 slab.
__device__ __forceinline__ uint4 pf_pack8(unsigned short v[8]) {
  uint4 r;
  r.x = (unsigned int)v[0] | ((unsigned int)v[1] << 16);
  r.y = (unsigned int)v[2] | ((unsigned int)v[3] << 16);
  r.z = (unsigned int)v[4] | ((unsigned int)v[5] << 16);
  r.w = (unsigned int)v[6] | ((unsigned int)v[7] << 16);
  return r;
}

__device__ __forceinline__ unsigned short pf_half(float a) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(a));
  return h;
}

extern "C" __global__ void dequant_q4k_f16_x8(const unsigned char* __restrict__ blocks,
                                              uint4* __restrict__ out, long long n_blocks) {
  const int lane = threadIdx.x & 31;
  const long long b = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (b >= n_blocks) return;
  const unsigned char* p = blocks + b * 144;
  const float d = half_to_float(*(const unsigned short*)(p));
  const float dmin = half_to_float(*(const unsigned short*)(p + 2));
  const unsigned char* scales = p + 4;
  const unsigned char* qs = p + 16;
  const int e0 = lane * 8;
  const int g = e0 >> 6, half = (e0 & 63) >> 5, l0 = e0 & 31;
  const int sub = g * 2 + half;
  int sc = 0, mn = 0;
  q4k_scale_min(sub, scales, &sc, &mn);
  const float scale = d * (float)sc, bias = dmin * (float)mn;
  const unsigned char* qp = qs + g * 32 + l0;
  const int sh = half * 4;
  unsigned short v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = (qp[i] >> sh) & 0xF;
    v[i] = pf_half(scale * (float)q - bias);
  }
  out[b * 32 + lane] = pf_pack8(v);
}

extern "C" __global__ void dequant_q5k_f16_x8(const unsigned char* __restrict__ blocks,
                                              uint4* __restrict__ out, long long n_blocks) {
  const int lane = threadIdx.x & 31;
  const long long b = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (b >= n_blocks) return;
  const unsigned char* p = blocks + b * 176;
  const float d = half_to_float(*(const unsigned short*)(p));
  const float dmin = half_to_float(*(const unsigned short*)(p + 2));
  const unsigned char* scales = p + 4;
  const unsigned char* qh = p + 16;
  const unsigned char* qs = p + 48;
  const int e0 = lane * 8;
  const int g = e0 >> 6, half = (e0 & 63) >> 5, l0 = e0 & 31;
  const int sub = g * 2 + half;
  int sc = 0, mn = 0;
  q4k_scale_min(sub, scales, &sc, &mn);
  const float scale = d * (float)sc, bias = dmin * (float)mn;
  const unsigned char* qp = qs + (sub >> 1) * 32 + l0;
  const int sh = (sub & 1) * 4;
  unsigned short v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = ((qp[i] >> sh) & 0xF) | (((qh[l0 + i] >> sub) & 1) << 4);
    v[i] = pf_half(scale * (float)q - bias);
  }
  out[b * 32 + lane] = pf_pack8(v);
}

extern "C" __global__ void dequant_q6k_f16_x8(const unsigned char* __restrict__ blocks,
                                              uint4* __restrict__ out, long long n_blocks) {
  const int lane = threadIdx.x & 31;
  const long long b = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (b >= n_blocks) return;
  const unsigned char* p = blocks + b * 210;
  const float d = half_to_float(*(const unsigned short*)(p + 208));
  const unsigned char* ql = p;
  const unsigned char* qh = p + 128;
  const signed char* sc = (const signed char*)(p + 192);
  const int e0 = lane * 8;
  const int n2 = e0 >> 7, q = (e0 & 127) >> 5, l0 = e0 & 31;
  const int is = l0 >> 4;
  const int qi = n2 * 64 + l0 + ((q & 1) ? 32 : 0);
  const int hi = n2 * 32 + l0;
  const int sh = 2 * q, nb = (q >= 2) ? 4 : 0;
  const float s = d * (float)sc[n2 * 8 + is + sh];
  unsigned short v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int qv = (int)(((ql[qi + i] >> nb) & 0xF) | (((qh[hi + i] >> sh) & 3) << 4)) - 32;
    v[i] = pf_half(s * (float)qv);
  }
  out[b * 32 + lane] = pf_pack8(v);
}
'''

# ---------------------------------------------------------------- kernels.h
s = open(KERNELS, encoding='utf-8').read()
anchor = '''  const int l = threadIdx.x;
  out[b * 32 + l] = (unsigned short)f32_to_f16_bits(d * (float)q[l]);
}
)cuda";

'''
assert s.count(anchor) == 1, s.count(anchor)
s = s.replace(anchor, '''  const int l = threadIdx.x;
  out[b * 32 + l] = (unsigned short)f32_to_f16_bits(d * (float)q[l]);
}
''' + X8 + ''')cuda";

''')
open(KERNELS, 'w', encoding='utf-8', newline='\n').write(s)
print('kernels.h: x8 dequant kernels added')

# ---------------------------------------------------------------- runner.cpp
s = open(RUNNER, encoding='utf-8').read()

# 1) kernel handles
old = '''  dq_q4k_f16_ = mod_->kernel("dequant_q4k_f16");
  dq_q5k_f16_ = mod_->kernel("dequant_q5k_f16");
  dq_q6k_f16_ = mod_->kernel("dequant_q6k_f16");'''
assert s.count(old) == 1
s = s.replace(old, old + '''
  dq_q4k_f16_x8_ = mod_->kernel("dequant_q4k_f16_x8");
  dq_q5k_f16_x8_ = mod_->kernel("dequant_q5k_f16_x8");
  dq_q6k_f16_x8_ = mod_->kernel("dequant_q6k_f16_x8");''')

# 2) dispatch: prefer the x8 kernels (bit-identical slab, 1.5-1.7x faster)
old2 = '''    void* f16fn = nullptr;
    int f16_threads = 128;
    if (w.type == "Q4_K") f16fn = dq_q4k_f16_;
    else if (w.type == "Q5_K") f16fn = dq_q5k_f16_;
    else if (w.type == "Q6_K") f16fn = dq_q6k_f16_;
    else if (w.type == "Q8_0") { f16fn = dq_q8_0_f16_; f16_threads = 32; }'''
assert s.count(old2) == 1
new2 = '''    // x8 variants: one warp per super-block, 8 values/lane and a 16 B store.  Measured
    // ~305 GB/s vs 181-207 GB/s for the per-block kernels, with a bit-identical fp16
    // slab (so the generated tokens cannot change).  ENGINE_DQ_OLD=1 forces the old
    // kernels for A/B.
    static const bool dq_old = ENGINE_ENV_ON("ENGINE_DQ_OLD");
    void* f16fn = nullptr;
    void* f16x8 = nullptr;
    int f16_threads = 128;
    if (w.type == "Q4_K") { f16fn = dq_q4k_f16_; f16x8 = dq_q4k_f16_x8_; }
    else if (w.type == "Q5_K") { f16fn = dq_q5k_f16_; f16x8 = dq_q5k_f16_x8_; }
    else if (w.type == "Q6_K") { f16fn = dq_q6k_f16_; f16x8 = dq_q6k_f16_x8_; }
    else if (w.type == "Q8_0") { f16fn = dq_q8_0_f16_; f16_threads = 32; }
    if (dq_old) f16x8 = nullptr;'''
s = s.replace(old2, new2)

# 3) the launch itself: x8 wants grid/8 and 256 threads
old3 = '''        void* a[] = {&ps, &po, &nb};
        cuda::launch(f16fn, static_cast<int>(nblk), 1, 1, f16_threads, 1, 1, 0, a);
        if (want_acc16 && k_f16tof32_) {'''
assert s.count(old3) == 1
new3 = '''        void* a[] = {&ps, &po, &nb};
        if (f16x8)
          cuda::launch(f16x8, static_cast<int>((nblk + 7) / 8), 1, 1, 256, 1, 1, 0, a);
        else
          cuda::launch(f16fn, static_cast<int>(nblk), 1, 1, f16_threads, 1, 1, 0, a);
        if (want_acc16 && k_f16tof32_) {'''
s = s.replace(old3, new3)
open(RUNNER, 'w', encoding='utf-8', newline='\n').write(s)
print('runner.cpp: x8 kernels wired in (ENGINE_DQ_OLD=1 = old path)')
