// Prototype: faster Q4_K / Q5_K / Q6_K -> fp16 dequant kernels.
//
// Why: one whole-model dequant is ~95 ms of the ~130 ms fixed cost of every prefill
// forward (measured: the product kernel moves 1.12 MB read + 2.25 MB written per
// 1.6k-token prompt... i.e. only ~115 GB/s of the 331 GB/s HBM when the 34.7M
// super-blocks of a 9B are each given their own 128-thread CUDA block: the per-block
// work is 2 stores, so block scheduling dominates).
//
// The variant here assigns ONE SUPER-BLOCK PER WARP: 32 lanes x 8 consecutive values,
// each lane decoding a single scale/min pair once (branch-free per lane, no per-element
// q4k_scale_min), 8-byte vector loads for the quant bytes and one 16-byte uint4 store.
// 8 warps per CUDA block => 8 super-blocks per block => 8x fewer blocks.
//
// Correctness bar: the fp16 slab must be BIT-IDENTICAL to the product kernel (same
// arithmetic order: scale * (float)q - bias, then cvt.rn.f16.f32), which makes the
// end-to-end token output identical by construction.  The bench checks that with memcmp.
//
//   test_gen --bench-dequant
#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "engine/cuda.h"

namespace dequant_proto {

inline const char* kSrc = R"cuda(
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
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  return __uint_as_float(bits);
}

__device__ __forceinline__ unsigned int pf_pack_half2(float a, float b) {
  unsigned short lo, hi;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(lo) : "f"(a));
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hi) : "f"(b));
  return (unsigned int)lo | ((unsigned int)hi << 16);
}

__device__ __forceinline__ unsigned short pf_half(float a) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(a));
  return h;
}

__device__ __forceinline__ void q4k_scale_min(int j, const unsigned char* q, int* sc, int* mn) {
  if (j < 4) {
    *sc = q[j] & 63;
    *mn = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 2) & 0x30);
    *mn = (q[j + 4] >> 4) | ((q[j] >> 2) & 0x30);
  }
}

// ============================ product kernels (copied) ============================
extern "C" __global__ void old_dequant_q4k_f16(const unsigned char* __restrict__ blocks,
                                               unsigned int* __restrict__ out,
                                               long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const int t = threadIdx.x;
  const unsigned char* p = blocks + b * 144;
  const float d = half_to_float(*(const unsigned short*)(p));
  const float dmin = half_to_float(*(const unsigned short*)(p + 2));
  const unsigned char* scales = p + 4;
  const unsigned char* qs = p + 16;
  const int e = t * 2;
  const int g = e >> 6, r = e & 63, half = r >> 5, l = r & 31;
  int sc = 0, mn = 0;
  q4k_scale_min(g * 2 + half, scales, &sc, &mn);
  const float scale = d * (float)sc, bias = dmin * (float)mn;
  const unsigned char* qp = qs + g * 32 + l;
  const int qb0 = qp[0], qb1 = qp[1];
  const int n0 = half ? (qb0 >> 4) : (qb0 & 0xF);
  const int n1 = half ? (qb1 >> 4) : (qb1 & 0xF);
  out[b * 128 + t] = pf_pack_half2(scale * (float)n0 - bias, scale * (float)n1 - bias);
}

extern "C" __global__ void old_dequant_q5k_f16(const unsigned char* __restrict__ blocks,
                                               unsigned int* __restrict__ out,
                                               long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const int t = threadIdx.x;
  const unsigned char* p = blocks + b * 176;
  const float d = half_to_float(*(const unsigned short*)(p));
  const float dmin = half_to_float(*(const unsigned short*)(p + 2));
  const unsigned char* scales = p + 4;
  const unsigned char* qh = p + 16;
  const unsigned char* qs = p + 48;
  const int e = t * 2;
  const int g = e >> 6, r = e & 63, half = r >> 5, l = r & 31;
  const int sub = g * 2 + half;
  int sc = 0, mn = 0;
  q4k_scale_min(sub, scales, &sc, &mn);
  const float scale = d * (float)sc, bias = dmin * (float)mn;
  const unsigned char* qp = qs + (sub >> 1) * 32 + l;
  const int qb0 = qp[0], qb1 = qp[1];
  const int nib_sh = (sub & 1) * 4;
  const int h0 = (qh[l] >> sub) & 1, h1 = (qh[l + 1] >> sub) & 1;
  const int q0 = ((qb0 >> nib_sh) & 0xF) | (h0 << 4);
  const int q1 = ((qb1 >> nib_sh) & 0xF) | (h1 << 4);
  out[b * 128 + t] = pf_pack_half2(scale * (float)q0 - bias, scale * (float)q1 - bias);
}

extern "C" __global__ void old_dequant_q6k_f16(const unsigned char* __restrict__ blocks,
                                               unsigned int* __restrict__ out,
                                               long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const int t = threadIdx.x;
  const unsigned char* p = blocks + b * 210;
  const float d = half_to_float(*(const unsigned short*)(p + 208));
  const unsigned char* ql = p;
  const unsigned char* qh = p + 128;
  const signed char* sc = (const signed char*)(p + 192);
  const int e = t * 2;
  const int n2 = e >> 7, r = e & 127, q = r >> 5, l = r & 31;
  const int is = l >> 4;
  const int qi = n2 * 64 + l + ((q & 1) ? 32 : 0);
  const int hi = n2 * 32 + l;
  const int sh = 2 * q, nb = (q >= 2) ? 4 : 0;
  const int v0 = (int)(((ql[qi] >> nb) & 0xF) | (((qh[hi] >> sh) & 3) << 4)) - 32;
  const int v1 = (int)(((ql[qi + 1] >> nb) & 0xF) | (((qh[hi + 1] >> sh) & 3) << 4)) - 32;
  const float s = d * (float)sc[n2 * 8 + is + sh];
  out[b * 128 + t] = pf_pack_half2(s * (float)v0, s * (float)v1);
}

// ================================ x8 kernels =====================================
// One super-block per warp, 32 lanes x 8 consecutive values, one 16 B store per lane.
// Lane l covers values 8l..8l+7, which always sit inside ONE 32-value sub-block, so a
// lane needs exactly one (scale, min) pair and 8+8 consecutive quant bytes (uint2 loads).
__device__ __forceinline__ uint4 pack8(unsigned short v[8]) {
  uint4 r;
  r.x = (unsigned int)v[0] | ((unsigned int)v[1] << 16);
  r.y = (unsigned int)v[2] | ((unsigned int)v[3] << 16);
  r.z = (unsigned int)v[4] | ((unsigned int)v[5] << 16);
  r.w = (unsigned int)v[6] | ((unsigned int)v[7] << 16);
  return r;
}

extern "C" __global__ void x8_dequant_q4k_f16(const unsigned char* __restrict__ blocks,
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
  out[b * 32 + lane] = pack8(v);
}

extern "C" __global__ void x8_dequant_q5k_f16(const unsigned char* __restrict__ blocks,
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
  out[b * 32 + lane] = pack8(v);
}

extern "C" __global__ void x8_dequant_q6k_f16(const unsigned char* __restrict__ blocks,
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
  out[b * 32 + lane] = pack8(v);
}
)cuda";

struct Case {
  const char* label;
  int blk_bytes;          // 144 / 176 / 210
  const char* old_kernel; // product kernel name
  const char* new_kernel; // x8 kernel name
  double model_gb;        // approximate share of the 9B weights of this type
};

inline void run_bench() {
  cuda::init();
  const cuda::DeviceInfo& di = cuda::device_info();
  std::printf("[dequant] old (1 block/super-block, 128 thr, 4 B stores) vs x8 (1 warp/"
              "super-block, 16 B stores)\n");
  std::printf("[dequant] device: %s, %d SMs, HBM ~331 GB/s\n", di.name.c_str(), di.sm_count);
  std::string ptx;
  try {
    ptx = cuda::compile_ptx(kSrc, "dequant_proto");
  } catch (const std::exception& e) {
    std::printf("[dequant] NVRTC compile FAILED: %s\n", e.what());
    return;
  }
  cuda::Module mod(ptx);
  // Representative Q5_K_M 9B tensor: 12288 x 4096 ffn_up (34.6 MB of Q5_K blocks).
  const Case cases[] = {
      {"Q4_K 12288x4096 (34.6 MB)", 144, "old_dequant_q4k_f16", "x8_dequant_q4k_f16", 0.4},
      {"Q5_K 12288x4096 (34.6 MB)", 176, "old_dequant_q5k_f16", "x8_dequant_q5k_f16", 5.2},
      {"Q6_K 12288x4096 (41.3 MB)", 210, "old_dequant_q6k_f16", "x8_dequant_q6k_f16", 0.5},
  };
  const int n_vals = 12288 * 4096;               // elements per tensor
  const size_t n_blocks = (size_t)n_vals / 256;
  double total_old = 0, total_new = 0, total_gb = 0;
  for (const auto& c : cases) {
    const size_t w_bytes = n_blocks * (size_t)c.blk_bytes;
    std::vector<unsigned char> w_host(w_bytes);
    unsigned s = 1234567u;
    for (auto& byte : w_host) {
      s = s * 1664525u + 1013904223u;
      byte = (unsigned char)(s >> 24);
    }
    // sane fp16 d / dmin for the dequant to produce finite values
    for (size_t off = 0; off + (size_t)c.blk_bytes <= w_bytes; off += (size_t)c.blk_bytes) {
      const unsigned short dh = 0x2C00u;
      std::memcpy(&w_host[off], &dh, 2);
      if (c.blk_bytes != 210) {
        const unsigned short dm = 0x2400u;
        std::memcpy(&w_host[off + 2], &dm, 2);
      }
    }
    cuda::Buffer w_dev(w_bytes), out_old(n_blocks * 512), out_new(n_blocks * 512);
    w_dev.upload(w_host.data(), w_bytes);
    void* old_fn = mod.kernel(c.old_kernel);
    void* new_fn = mod.kernel(c.new_kernel);
    void* w_ptr = w_dev.data();
    void* o_ptr = out_old.data();
    void* n_ptr = out_new.data();
    long long nb = (long long)n_blocks;
    void* old_args[] = {&w_ptr, &o_ptr, &nb};
    void* new_args[] = {&w_ptr, &n_ptr, &nb};
    // warm-up + min-of-5 event timing
    double best_old = 1e9, best_new = 1e9;
    for (int i = 0; i < 6; ++i) {
      cuda::Event e0, e1;
      e0.record();
      cuda::launch(old_fn, (int)n_blocks, 1, 1, 128, 1, 1, 0, old_args);
      e1.record();
      e1.synchronize();
      if (i) best_old = std::min(best_old, e0.elapsed_ms(e1));
      cuda::Event e2, e3;
      e2.record();
      cuda::launch(new_fn, (int)((n_blocks + 7) / 8), 1, 1, 256, 1, 1, 0, new_args);
      e3.record();
      e3.synchronize();
      if (i) best_new = std::min(best_new, e2.elapsed_ms(e3));
    }
    // bit-identical check (the whole point: same fp16 slab => identical model output)
    std::vector<unsigned char> ho(n_blocks * 512), hn(n_blocks * 512);
    out_old.download(ho.data(), ho.size());
    out_new.download(hn.data(), hn.size());
    const bool same = std::memcmp(ho.data(), hn.data(), ho.size()) == 0;
    const double mb = (double)(w_bytes + n_blocks * 512) / 1048576.0;
    std::printf("  %-26s old %6.3f ms (%5.0f GB/s) | x8 %6.3f ms (%5.0f GB/s) | %4.2fx | "
                "slab bit-identical: %s\n",
                c.label, best_old, mb / best_old, best_new, mb / best_new, best_old / best_new,
                same ? "YES" : "NO");
    total_old += best_old;
    total_new += best_new;
    total_gb += c.model_gb;
  }
  std::printf("  -> the 110 MB of slabs above: old %.3f ms, x8 %.3f ms (%.2fx); whole-model "
              "6.1 GB at these rates: old ~%.0f ms, x8 ~%.0f ms\n",
              total_old, total_new, total_old / total_new, total_old * 6.1 * 1024.0 / 110.0,
              total_new * 6.1 * 1024.0 / 110.0);
}
}  // namespace dequant_proto
