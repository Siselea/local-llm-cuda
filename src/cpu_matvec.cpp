#include "engine/cpu_matvec.h"
#include "engine/config.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <algorithm>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace cpu {
namespace {

struct QK {
  int block_bytes;
  int block_elems;
};

QK qk_of(const std::string& type) {
  if (type == "Q4_K") return {144, 256};
  if (type == "Q5_K") return {176, 256};
  if (type == "Q6_K") return {210, 256};
  if (type == "Q8_0") return {34, 32};
  if (type == "Q5_0") return {22, 32};
  if (type == "Q5_1") return {24, 32};
  throw std::runtime_error("cpu::matvec: unsupported quant type " + type);
}

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t exp = (h >> 10) & 0x1Fu;
  std::uint32_t mant = h & 0x3FFu;
  if (exp == 0) {
    if (mant == 0) {
      const std::uint32_t bits = sign;
      float f;
      std::memcpy(&f, &bits, 4);
      return f;
    }
    float v = static_cast<float>(mant) * 5.9604644775390625e-08f;
    return sign ? -v : v;
  }
  if (exp == 31) {
    const std::uint32_t bits = sign | 0x7F800000u | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  }
  exp += 112;
  const std::uint32_t bits = sign | (exp << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Physical core count.  Windows reports one RelationProcessorCore entry per physical
// core; a VM whose vCPUs are all logical-only yields 0 and the caller keeps the
// logical-CPU formula.  Used to derive the worker thread count from the topology
// instead of a fixed fraction of the logical CPUs.
int physical_cores() {
#if defined(_WIN32)
  DWORD len = 0;
  if (!GetLogicalProcessorInformation(nullptr, &len)) {
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) return 0;
  }
  if (len == 0) return 0;
  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
      len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) + 1);
  if (!GetLogicalProcessorInformation(buf.data(), &len)) return 0;
  const size_t cnt = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
  int n = 0;
  for (size_t i = 0; i < cnt; ++i)
    if (buf[i].Relationship == RelationProcessorCore) ++n;
  return n;
#else
  return 0;
#endif
}

// Logical CPU ids ordered "one thread per physical core first, SMT siblings last".
// Used only by the opt-in win_affinity pinning.  Empty when the topology is not
// available (non-Windows / more than one processor group).
std::vector<int> physical_first_cpu_order() {
  std::vector<int> order;
#if defined(_WIN32)
  DWORD len = 0;
  if (!GetLogicalProcessorInformation(nullptr, &len)) {
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) return order;
  }
  if (len == 0) return order;
  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
      len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) + 1);
  if (!GetLogicalProcessorInformation(buf.data(), &len)) return order;
  const size_t cnt = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
  std::vector<int> first, second;
  for (size_t i = 0; i < cnt; ++i) {
    if (buf[i].Relationship != RelationProcessorCore) continue;
    ULONG_PTR mask = buf[i].ProcessorMask;
    bool primary_done = false;
    for (int bit = 0; bit < 64 && mask; ++bit, mask >>= 1) {
      if (!(mask & 1)) continue;
      (primary_done ? second : first).push_back(bit);
      primary_done = true;
    }
  }
  order.reserve(first.size() + second.size());
  order.insert(order.end(), first.begin(), first.end());
  order.insert(order.end(), second.begin(), second.end());
#endif
  return order;
}

void scale_min(int sub, const unsigned char* s, int* sc, int* mn) {
  // must match q4k_scale_min() in kernels.h (validated bit-exact vs gguf.quants)
  if (sub < 4) {
    *sc = s[sub] & 63;
    *mn = s[sub + 4] & 63;
  } else {
    *sc = (s[sub + 4] & 0xF) | ((s[sub - 4] >> 2) & 0x30);
    *mn = (s[sub + 4] >> 4) | ((s[sub] >> 2) & 0x30);
  }
}

}  // namespace

void dequant_block(const std::string& type, const unsigned char* p, float* out) {
  if (type == "Q4_K") {
    const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const float dmin = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
    const unsigned char* scales = p + 4;
    const unsigned char* qs = p + 16;
    for (int e = 0; e < 256; ++e) {
      const int g = e >> 6, half = (e & 63) >> 5, l = e & 31;
      const int sub = g * 2 + half;
      const unsigned char qb = qs[g * 32 + l];
      const int nib = half ? (qb >> 4) : (qb & 0xF);
      int sc = 0, mn = 0;
      scale_min(sub, scales, &sc, &mn);
      out[e] = d * static_cast<float>(sc) * static_cast<float>(nib) - dmin * static_cast<float>(mn);
    }
    return;
  }
  if (type == "Q5_K") {
    const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const float dmin = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
    const unsigned char* scales = p + 4;
    const unsigned char* qh = p + 16;
    const unsigned char* qs = p + 48;
    for (int e = 0; e < 256; ++e) {
      const int g = e >> 6, half = (e & 63) >> 5, l = e & 31;
      const int sub = g * 2 + half;
      const unsigned char qb = qs[g * 32 + l];
      const int nib = half ? (qb >> 4) : (qb & 0xF);
      const int hi = (qh[l] >> sub) & 1;
      int sc = 0, mn = 0;
      scale_min(sub, scales, &sc, &mn);
      out[e] = d * static_cast<float>(sc) * static_cast<float>(nib | (hi << 4)) -
               dmin * static_cast<float>(mn);
    }
    return;
  }
  if (type == "Q6_K") {
    const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 208));
    const unsigned char* ql = p;
    const unsigned char* qh = p + 128;
    const signed char* sc = reinterpret_cast<const signed char*>(p + 192);
    for (int n2 = 0; n2 < 2; ++n2) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int qi = n2 * 64, hi = n2 * 32;
        const int q1 = static_cast<int>((ql[qi + l] & 0xF) | (((qh[hi + l] >> 0) & 3) << 4)) - 32;
        const int q2 =
            static_cast<int>((ql[qi + l + 32] & 0xF) | (((qh[hi + l] >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>((ql[qi + l] >> 4) | (((qh[hi + l] >> 4) & 3) << 4)) - 32;
        const int q4 =
            static_cast<int>((ql[qi + l + 32] >> 4) | (((qh[hi + l] >> 6) & 3) << 4)) - 32;
        float* o = out + n2 * 128;
        o[l + 0] = d * static_cast<float>(sc[n2 * 8 + is + 0]) * static_cast<float>(q1);
        o[l + 32] = d * static_cast<float>(sc[n2 * 8 + is + 2]) * static_cast<float>(q2);
        o[l + 64] = d * static_cast<float>(sc[n2 * 8 + is + 4]) * static_cast<float>(q3);
        o[l + 96] = d * static_cast<float>(sc[n2 * 8 + is + 6]) * static_cast<float>(q4);
      }
    }
    return;
  }
  if (type == "Q8_0") {
    const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const signed char* q = reinterpret_cast<const signed char*>(p + 2);
    for (int i = 0; i < 32; ++i) out[i] = d * static_cast<float>(q[i]);
    return;
  }
  if (type == "Q5_0") {
    const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const unsigned int qh = static_cast<unsigned int>(p[2]) |
                            (static_cast<unsigned int>(p[3]) << 8) |
                            (static_cast<unsigned int>(p[4]) << 16) |
                            (static_cast<unsigned int>(p[5]) << 24);
    const unsigned char* qs = p + 6;
    for (int i = 0; i < 32; ++i) {
      const int low = (i < 16) ? (qs[i] & 0xF) : (qs[i - 16] >> 4);
      const int hi = static_cast<int>((qh >> i) & 1u);
      out[i] = d * static_cast<float>((low | (hi << 4)) - 16);
    }
    return;
  }
  if (type == "Q5_1") {
    const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const float m = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
    const unsigned int qh = static_cast<unsigned int>(p[4]) |
                            (static_cast<unsigned int>(p[5]) << 8) |
                            (static_cast<unsigned int>(p[6]) << 16) |
                            (static_cast<unsigned int>(p[7]) << 24);
    const unsigned char* qs = p + 8;
    for (int i = 0; i < 32; ++i) {
      const int low = (i < 16) ? (qs[i] & 0xF) : (qs[i - 16] >> 4);
      const int hi = static_cast<int>((qh >> i) & 1u);
      out[i] = d * static_cast<float>(low | (hi << 4)) + m;
    }
    return;
  }
  throw std::runtime_error("cpu::dequant_block: unsupported type " + type);
}

namespace {

// Fused dequant + dot product for the hot types: the per-element scale lookup and
// nibble branch in dequant_block costs more than the arithmetic, so the block scales
// are hoisted and both halves of each 32-element group are handled in one pass.

#if defined(__AVX2__) && defined(__FMA__)
#define ENGINE_HAVE_AVX2 1
#include <immintrin.h>

static inline __m256 q4k_fma8(__m256 acc, const unsigned char* q8, __m256 scale, __m256 bias,
                              const float* x) {
  // 8 packed nibbles -> 8 floats: scale*val - bias, then acc += that * x
  const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q8));
  __m256i v = _mm256_cvtepu8_epi32(raw);
  v = _mm256_and_si256(v, _mm256_set1_epi32(0xF));
  __m256 f = _mm256_fmsub_ps(scale, _mm256_cvtepi32_ps(v), bias);
  return _mm256_fmadd_ps(f, _mm256_loadu_ps(x), acc);
}

static inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}


// ---- two-level int8 activation quantisation (for the int8 row kernels) --------
// The fp32 kernels below are limited by instruction throughput, not by DRAM: one
// core reaches ~5.7 GB/s on a quantized row while the same thread reading the same
// bytes with AVX2 does 15-18 GB/s, and a pure-unpack probe runs at full kernel speed.
// The int8 kernels replace the float multiply-accumulate with int8 ones
// (_mm256_maddubs_epi16 does 32 MACs in 2 instructions), which needs the
// activation as int8 as well.
//
// A single int8 level is far too coarse: the quantisation error is d/sqrt(12) =
// amax/(127*sqrt(12)) which is ~3e-3 relative for a 32-value block -- the GPU
// dp4a GEMV measured the same thing and went to two levels (base + residual),
// each with its own scale, which brings the error to ~1e-5 relative.  Blocks are
// 32 consecutive values, i.e. exactly the granularity of one AVX2 int8 vector and
// exactly one Q4_K sub-block; sx32 / sx16 hold the sums of the *reconstructed*
// activation per 32 / 16 values that the Q4_K / Q6_K bias terms need.
struct XQ8 {
  int n = 0;
  int nblk = 0;
  std::vector<signed char> q1, q2;  // two int8 levels, n values each
  std::vector<float> d1, d2;        // scale of each level, per 32-value block
  std::vector<float> sx32;          // sum of the reconstructed activation per 32
  std::vector<float> sx16;          // ... per 16 values (Q6_K scale granularity)
};

// Helpers for the vectorised quantiser.
inline int hsum_epi32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  lo = _mm_add_epi32(lo, hi);
  lo = _mm_hadd_epi32(lo, lo);
  lo = _mm_hadd_epi32(lo, lo);
  return _mm_cvtsi128_si32(lo);
}

inline float hmax_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_max_ps(lo, hi);
  lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 1));
  return _mm_cvtss_f32(lo);
}

// 32 int32 (4 ymm) -> 32 int8, in element order.
inline __m256i pack_i32_i8(const __m256i q[4]) {
  const __m128i a = _mm_packs_epi32(_mm256_castsi256_si128(q[0]),
                                    _mm256_extracti128_si256(q[0], 1));
  const __m128i b = _mm_packs_epi32(_mm256_castsi256_si128(q[1]),
                                    _mm256_extracti128_si256(q[1], 1));
  const __m128i c = _mm_packs_epi32(_mm256_castsi256_si128(q[2]),
                                    _mm256_extracti128_si256(q[2], 1));
  const __m128i d = _mm_packs_epi32(_mm256_castsi256_si128(q[3]),
                                    _mm256_extracti128_si256(q[3], 1));
  return _mm256_set_m128i(_mm_packs_epi16(c, d), _mm_packs_epi16(a, b));
}

void xq_build(const float* x, int n, XQ8* out) {
  const int nb = n >> 5;
  out->n = n;
  out->nblk = nb;
  out->q1.resize(static_cast<std::size_t>(n));
  out->q2.resize(static_cast<std::size_t>(n));
  out->d1.resize(static_cast<std::size_t>(nb));
  out->d2.resize(static_cast<std::size_t>(nb));
  out->sx32.resize(static_cast<std::size_t>(nb));
  out->sx16.resize(static_cast<std::size_t>(nb) * 2);
  const __m256 signmask = _mm256_set1_ps(-0.0f);
  const __m256 k127 = _mm256_set1_ps(127.f);
  for (int b = 0; b < nb; ++b) {
    const float* xb = x + static_cast<std::size_t>(b) * 32;
    __m256 v[4];
    for (int i = 0; i < 4; ++i) v[i] = _mm256_loadu_ps(xb + 8 * i);
    __m256 m = _mm256_max_ps(
        _mm256_max_ps(_mm256_andnot_ps(signmask, v[0]), _mm256_andnot_ps(signmask, v[1])),
        _mm256_max_ps(_mm256_andnot_ps(signmask, v[2]), _mm256_andnot_ps(signmask, v[3])));
    const float amax = hmax_ps(m);
    const float d1 = amax * (1.f / 127.f);
    const float inv1 = d1 > 0.f ? 1.f / d1 : 0.f;
    const __m256 sv1 = _mm256_set1_ps(inv1);
    __m256i q[4];
    for (int i = 0; i < 4; ++i) q[i] = _mm256_cvtps_epi32(_mm256_mul_ps(v[i], sv1));
    const __m256i s01 = _mm256_add_epi32(q[0], q[1]);
    const __m256i s23 = _mm256_add_epi32(q[2], q[3]);
    const int s1h0 = hsum_epi32(s01), s1h1 = hsum_epi32(s23);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out->q1[static_cast<std::size_t>(b) * 32]),
                        pack_i32_i8(q));

    // residual after the first level
    const __m256 d1v = _mm256_set1_ps(d1);
    __m256 r[4];
    for (int i = 0; i < 4; ++i)
      r[i] = _mm256_fnmadd_ps(d1v, _mm256_cvtepi32_ps(q[i]), v[i]);
    __m256 mr = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(signmask, r[0]),
                                            _mm256_andnot_ps(signmask, r[1])),
                              _mm256_max_ps(_mm256_andnot_ps(signmask, r[2]),
                                            _mm256_andnot_ps(signmask, r[3])));
    const float amax2 = hmax_ps(mr);
    const float d2 = amax2 * (1.f / 127.f);
    const float inv2 = d2 > 0.f ? 1.f / d2 : 0.f;
    const __m256 sv2 = _mm256_set1_ps(inv2);
    for (int i = 0; i < 4; ++i) q[i] = _mm256_cvtps_epi32(_mm256_mul_ps(r[i], sv2));
    const int s2h0 = hsum_epi32(_mm256_add_epi32(q[0], q[1]));
    const int s2h1 = hsum_epi32(_mm256_add_epi32(q[2], q[3]));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out->q2[static_cast<std::size_t>(b) * 32]),
                        pack_i32_i8(q));
    out->d1[static_cast<std::size_t>(b)] = d1;
    out->d2[static_cast<std::size_t>(b)] = d2;
    out->sx32[static_cast<std::size_t>(b)] =
        d1 * static_cast<float>(s1h0 + s1h1) + d2 * static_cast<float>(s2h0 + s2h1);
    out->sx16[static_cast<std::size_t>(b) * 2] = d1 * static_cast<float>(s1h0) + d2 * static_cast<float>(s2h0);
    out->sx16[static_cast<std::size_t>(b) * 2 + 1] = d1 * static_cast<float>(s1h1) + d2 * static_cast<float>(s2h1);
  }
  (void)k127;
}

// ---- per-type row dot kernels -------------------------------------------------
// These are the hot loops of the CPU-expert path: one quantized row x the
// activation vector, called once per (expert, row) of the 35B's 25 CPU layers.
//
// Two changes vs. the first AVX2 version, both measured on this machine
// (tools/bench_cpu_layer.cpp, rotating copies so the reads come from DRAM):
//  * 4 independent accumulators: the old single-accumulator loop chained every
//    FMA, so one row was latency bound at ~4 cycles per FMA (3.5 GB/s per core).
//  * explicit prefetch of the current row into L1 and the next row into L2.
//    The kernel turned out to be load-latency bound, not ALU bound: prefetching
//    alone is worth ~20% at 12 threads.  row + row_bytes is prefetched
//    speculatively; prefetch never faults and the caller's rows are contiguous.
// The summation order changes vs. the single-accumulator version (allowed: float
// addition is commutative but not associative); tests/test_cpu_moe.cpp pins the
// resulting difference against a double-precision reference to < 1e-6.
static inline void avx2_dot_q4k(const unsigned char* row, const float* x, int n_in, float* out) {
  const int nb = n_in / 256;
  const int row_bytes = nb * 144;
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + i), _MM_HINT_T0);
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + row_bytes + i), _MM_HINT_T1);
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  const __m256i c15 = _mm256_set1_epi32(0xF);
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 144;
    const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const float dmin = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
    const unsigned char* scales = p + 4;
    const unsigned char* qs = p + 16;
    float scf[8], mnf[8];
    for (int sub = 0; sub < 8; ++sub) {
      int sc = 0, mn = 0;
      scale_min(sub, scales, &sc, &mn);
      scf[sub] = d * static_cast<float>(sc);
      mnf[sub] = dmin * static_cast<float>(mn);
    }
    const float* xb = x + static_cast<std::size_t>(b) * 256;
    for (int g = 0; g < 4; ++g) {
      const unsigned char* q = qs + g * 32;
      const __m256 s0 = _mm256_set1_ps(scf[2 * g]);
      const __m256 m0 = _mm256_set1_ps(mnf[2 * g]);
      const __m256 s1 = _mm256_set1_ps(scf[2 * g + 1]);
      const __m256 m1 = _mm256_set1_ps(mnf[2 * g + 1]);
      const float* x0 = xb + g * 64;
      for (int l = 0; l < 32; l += 16) {
        const __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
        const __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l + 8));
        const __m256i lo0 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw0), c15);
        const __m256i hi0 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw0), 4);
        const __m256i lo1 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw1), c15);
        const __m256i hi1 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw1), 4);
        const __m256 f00 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo0), m0);
        const __m256 f01 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi0), m1);
        const __m256 f10 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo1), m0);
        const __m256 f11 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi1), m1);
        a0 = _mm256_fmadd_ps(f00, _mm256_loadu_ps(x0 + l), a0);
        a1 = _mm256_fmadd_ps(f01, _mm256_loadu_ps(x0 + 32 + l), a1);
        a2 = _mm256_fmadd_ps(f10, _mm256_loadu_ps(x0 + l + 8), a2);
        a3 = _mm256_fmadd_ps(f11, _mm256_loadu_ps(x0 + 32 + l + 8), a3);
      }
    }
  }
  a0 = _mm256_add_ps(a0, a1);
  a2 = _mm256_add_ps(a2, a3);
  *out = hsum256(_mm256_add_ps(a0, a2));
}

static inline void avx2_dot_q6k(const unsigned char* row, const float* x, int n_in, float* out) {
  const int nb = n_in / 256;
  const int row_bytes = nb * 210;
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + i), _MM_HINT_T0);
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + row_bytes + i), _MM_HINT_T1);
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  const __m256i c15 = _mm256_set1_epi32(0xF);
  const __m256i c3 = _mm256_set1_epi32(3);
  const __m256i c32 = _mm256_set1_epi32(32);
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 210;
    const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 208));
    const signed char* sc = reinterpret_cast<const signed char*>(p + 192);
    const float* xb = x + static_cast<std::size_t>(b) * 256;
    // the per-16-element scales are loop invariant per half-block: hoist them out
    float sfl[16];
    for (int i = 0; i < 16; ++i) sfl[i] = d * static_cast<float>(sc[i]);
    for (int n2 = 0; n2 < 2; ++n2) {
      const unsigned char* ql = p + n2 * 64;
      const unsigned char* qh = p + 128 + n2 * 32;
      const float* x0 = xb + n2 * 128;
      for (int l = 0; l < 32; l += 8) {
        // scale byte index: n2*8 + (l>>4), then every second byte for q1..q4
        const float* sf = sfl + n2 * 8 + (l >> 4);
        const __m128i l1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l));
        const __m128i l2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l + 32));
        const __m128i h8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l));
        const __m256i h = _mm256_cvtepu8_epi32(h8);
        const __m256i lo1 = _mm256_cvtepu8_epi32(l1);
        const __m256i lo2 = _mm256_cvtepu8_epi32(l2);
        const __m256i q1 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_and_si256(lo1, c15),
                            _mm256_slli_epi32(_mm256_and_si256(h, c3), 4)), c32);
        const __m256i q2 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_and_si256(lo2, c15),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 2), c3), 4)),
            c32);
        const __m256i q3 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(lo1, 4), c15),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 4), c3), 4)),
            c32);
        const __m256i q4 = _mm256_sub_epi32(
            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(lo2, 4), c15),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 6), c3), 4)),
            c32);
        a0 = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_set1_ps(sf[0]), _mm256_cvtepi32_ps(q1)),
            _mm256_loadu_ps(x0 + l), a0);
        a1 = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_set1_ps(sf[2]), _mm256_cvtepi32_ps(q2)),
            _mm256_loadu_ps(x0 + l + 32), a1);
        a2 = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_set1_ps(sf[4]), _mm256_cvtepi32_ps(q3)),
            _mm256_loadu_ps(x0 + l + 64), a2);
        a3 = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_set1_ps(sf[6]), _mm256_cvtepi32_ps(q4)),
            _mm256_loadu_ps(x0 + l + 96), a3);
      }
    }
  }
  a0 = _mm256_add_ps(a0, a1);
  a2 = _mm256_add_ps(a2, a3);
  *out = hsum256(_mm256_add_ps(a0, a2));
}

// ---- int8 row kernels (Q4_K / Q6_K) ------------------------------------------
// Both quant types are unpacked straight into int8 vectors with one load + one mask
// per 32 values (Q4_K) or one load + 4 mask/shift/or chains per 128 values (Q6_K),
// and the two int8 activation levels are dotted in with _mm256_maddubs_epi16
// (unsigned weight x signed activation) + _mm256_madd_epi16.  The 8 int32 lanes
// that come out are 4-value partial sums, so the sub-block scale can be applied to
// the vector and no horizontal reduction happens inside the block.  The bias term
// (Q4_K's -dmin*mn*sum(x), Q6_K's -32*d*sc*sum(x)) comes from the precomputed
// sx32 / sx16 sums of the reconstructed activation.
//
// The contributions are returned by value and summed in fixed-name accumulators
// (never through an indexed array / reference: that forces the compiler to spill the
// four accumulators to the stack and costs the whole speed-up of the int8 path).

// One 32-value sub-block of a Q4_K row: unsigned nibble weights (0..15) against the
// two int8 activation levels; returns sum(q*x1)*c1 + sum(q*x2)*c2 (8-lane partials).
static inline __m256 q4k_i8_contrib(__m256i w, const signed char* q1, const signed char* q2,
                                    std::size_t off, float c1s, float c2s) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q1 + off));
  const __m256i x2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q2 + off));
  const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x1), ones);
  const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x2), ones);
  return _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(c1s),
                         _mm256_mul_ps(_mm256_cvtepi32_ps(p2), _mm256_set1_ps(c2s)));
}

static inline void avx2_dot_q4k_i8(const unsigned char* row, const XQ8& x, int n_in, float* out) {
  const int nb = n_in / 256;
  const int row_bytes = nb * 144;
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + i), _MM_HINT_T0);
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + row_bytes + i), _MM_HINT_T1);
  const __m256i m0f = _mm256_set1_epi8(0x0F);
  const signed char* q1 = x.q1.data();
  const signed char* q2 = x.q2.data();
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  float bias = 0.f;
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 144;
    const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const float dmin = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
    const unsigned char* scales = p + 4;
    const unsigned char* qs = p + 16;
    float scf[8], mnf[8];
    for (int sub = 0; sub < 8; ++sub) {
      int sc = 0, mn = 0;
      scale_min(sub, scales, &sc, &mn);
      scf[sub] = d * static_cast<float>(sc);
      mnf[sub] = dmin * static_cast<float>(mn);
    }
    const __m256i raw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
    const __m256i raw1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32));
    const __m256i raw2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 64));
    const __m256i raw3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 96));
    // g-th 64-value group: low nibbles -> sub-block 2g, high nibbles -> 2g+1
#define Q4K_G(RAW, G)                                                                        \
  do {                                                                                       \
    const __m256i wlo = _mm256_and_si256(RAW, m0f);                                           \
    const __m256i whi = _mm256_and_si256(_mm256_srli_epi16(RAW, 4), m0f);                     \
    const int s_lo = (G) * 2, s_hi = (G) * 2 + 1;                                            \
    const std::size_t o_lo = static_cast<std::size_t>(b * 8 + s_lo) * 32;                     \
    const std::size_t o_hi = static_cast<std::size_t>(b * 8 + s_hi) * 32;                     \
    __m256& A = acc##G;                                                                       \
    const __m256 clo = q4k_i8_contrib(wlo, q1, q2, o_lo, scf[s_lo] * x.d1[b * 8 + s_lo],      \
                                      scf[s_lo] * x.d2[b * 8 + s_lo]);                        \
    const __m256 chi = q4k_i8_contrib(whi, q1, q2, o_hi, scf[s_hi] * x.d1[b * 8 + s_hi],      \
                                      scf[s_hi] * x.d2[b * 8 + s_hi]);                        \
    A = _mm256_add_ps(A, _mm256_add_ps(clo, chi));                                            \
    bias -= mnf[s_lo] * x.sx32[b * 8 + s_lo] + mnf[s_hi] * x.sx32[b * 8 + s_hi];              \
  } while (0)
    Q4K_G(raw0, 0);
    Q4K_G(raw1, 1);
    Q4K_G(raw2, 2);
    Q4K_G(raw3, 3);
#undef Q4K_G
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  acc2 = _mm256_add_ps(acc2, acc3);
  *out = hsum256(_mm256_add_ps(acc0, acc2)) + bias;
}

// Q6_K: 6-bit values are kept unsigned (0..63) so maddubs can take them directly;
// sum((q-32)*x) = sum(q*x) - 32*sum(x) is restored from sx16.  One Q6_K block has
// 16 scales of 16 values, i.e. two different scales inside every 32-value int8
// vector, so the 8 int32 lanes get a [sa sa sa sa sb sb sb sb] scale vector.
static inline __m256 q6k_i8_contrib(__m256i w, const signed char* q1, const signed char* q2,
                                    std::size_t off, float d1s, float d2s, float sa, float sb,
                                    float* bias_acc) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q1 + off));
  const __m256i x2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q2 + off));
  const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x1), ones);
  const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x2), ones);
  const __m256 pf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p2), _mm256_set1_ps(d2s),
                                    _mm256_mul_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(d1s)));
  const __m256 sv = _mm256_blend_ps(_mm256_set1_ps(sa), _mm256_set1_ps(sb), 0xF0);
  (void)bias_acc;
  return _mm256_mul_ps(pf, sv);
}

static inline void avx2_dot_q6k_i8(const unsigned char* row, const XQ8& x, int n_in, float* out) {
  const int nb = n_in / 256;
  const int row_bytes = nb * 210;
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + i), _MM_HINT_T0);
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + row_bytes + i), _MM_HINT_T1);
  const __m256i m0f = _mm256_set1_epi8(0x0F);
  const __m256i m03 = _mm256_set1_epi8(0x03);
  const __m256i m30 = _mm256_set1_epi8(0x30);
  const __m256i mc0 = _mm256_set1_epi8(static_cast<char>(0xC0));
  const signed char* q1 = x.q1.data();
  const signed char* q2 = x.q2.data();
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  float bias = 0.f;
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 210;
    const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 208));
    const signed char* sc = reinterpret_cast<const signed char*>(p + 192);
    for (int n2 = 0; n2 < 2; ++n2) {
      const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + n2 * 64));
      const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + n2 * 64 + 32));
      const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 128 + n2 * 32));
      const __m256i w0 = _mm256_or_si256(_mm256_and_si256(ql0, m0f),
                                         _mm256_slli_epi16(_mm256_and_si256(qh, m03), 4));
      const __m256i w1 = _mm256_or_si256(
          _mm256_and_si256(ql1, m0f),
          _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2), m03), 4));
      const __m256i w2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m0f),
                                         _mm256_and_si256(qh, m30));
      const __m256i w3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m0f),
                                         _mm256_srli_epi16(_mm256_and_si256(qh, mc0), 2));
      const int bio = b * 8 + n2 * 4;
      const signed char* scb = sc + n2 * 8;
      const std::size_t o0 = static_cast<std::size_t>(bio) * 32;
      const float dd1 = x.d1[bio], dd2 = x.d2[bio];
      acc0 = _mm256_add_ps(acc0, q6k_i8_contrib(w0, q1, q2, o0, dd1, dd2, d * scb[0], d * scb[1], &bias));
      acc1 = _mm256_add_ps(acc1, q6k_i8_contrib(w1, q1, q2, o0 + 32, x.d1[bio + 1], x.d2[bio + 1],
                                                d * scb[2], d * scb[3], &bias));
      acc2 = _mm256_add_ps(acc2, q6k_i8_contrib(w2, q1, q2, o0 + 64, x.d1[bio + 2], x.d2[bio + 2],
                                                d * scb[4], d * scb[5], &bias));
      acc3 = _mm256_add_ps(acc3, q6k_i8_contrib(w3, q1, q2, o0 + 96, x.d1[bio + 3], x.d2[bio + 3],
                                                d * scb[6], d * scb[7], &bias));
      bias -= 32.f * (d * scb[0] * x.sx16[static_cast<std::size_t>(bio) * 2] +
                      d * scb[1] * x.sx16[static_cast<std::size_t>(bio) * 2 + 1] +
                      d * scb[2] * x.sx16[static_cast<std::size_t>(bio + 1) * 2] +
                      d * scb[3] * x.sx16[static_cast<std::size_t>(bio + 1) * 2 + 1] +
                      d * scb[4] * x.sx16[static_cast<std::size_t>(bio + 2) * 2] +
                      d * scb[5] * x.sx16[static_cast<std::size_t>(bio + 2) * 2 + 1] +
                      d * scb[6] * x.sx16[static_cast<std::size_t>(bio + 3) * 2] +
                      d * scb[7] * x.sx16[static_cast<std::size_t>(bio + 3) * 2 + 1]);
    }
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  acc2 = _mm256_add_ps(acc2, acc3);
  *out = hsum256(_mm256_add_ps(acc0, acc2)) + bias;
}

static bool avx2_dot_row(const std::string& type, const unsigned char* row, const float* x,
                         int n_in, float* out) {
  if (type == "Q4_K") {
    avx2_dot_q4k(row, x, n_in, out);
    return true;
  }
  if (type == "Q8_0") {
    const int nb = n_in / 32;
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
      const unsigned char* p = row + static_cast<std::size_t>(b) * 34;
      const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
      const __m256 dv = _mm256_set1_ps(d);
      const signed char* q = reinterpret_cast<const signed char*>(p + 2);
      const float* xb = x + static_cast<std::size_t>(b) * 32;
      for (int l = 0; l < 32; l += 8) {
        const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
        const __m256 f = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw)));
        acc = _mm256_fmadd_ps(f, _mm256_loadu_ps(xb + l), acc);
      }
    }
    *out = hsum256(acc);
    return true;
  }
  if (type == "Q5_K") {
    const int nb = n_in / 256;
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
      const unsigned char* p = row + static_cast<std::size_t>(b) * 176;
      const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
      const float dmin = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
      const unsigned char* scales = p + 4;
      const unsigned char* qh = p + 16;
      const unsigned char* qs = p + 48;
      const float* xb = x + static_cast<std::size_t>(b) * 256;
      const __m256i c15 = _mm256_set1_epi32(0xF);
      const __m256i c1 = _mm256_set1_epi32(1);
      for (int g = 0; g < 4; ++g) {
        int sc0 = 0, mn0 = 0, sc1 = 0, mn1 = 0;
        scale_min(2 * g, scales, &sc0, &mn0);
        scale_min(2 * g + 1, scales, &sc1, &mn1);
        const __m256 s0 = _mm256_set1_ps(d * static_cast<float>(sc0));
        const __m256 b0 = _mm256_set1_ps(dmin * static_cast<float>(mn0));
        const __m256 s1 = _mm256_set1_ps(d * static_cast<float>(sc1));
        const __m256 b1 = _mm256_set1_ps(dmin * static_cast<float>(mn1));
        const unsigned char* q = qs + g * 32;
        const float* x0 = xb + g * 64;
        for (int l = 0; l < 32; l += 8) {
          const __m128i qbv = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
          const __m128i qhv = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l));
          const __m256i qb = _mm256_cvtepu8_epi32(qbv);
          const __m256i hv = _mm256_cvtepu8_epi32(qhv);
          const __m256i sh0 = _mm256_set1_epi32(2 * g);
          const __m256i sh1 = _mm256_set1_epi32(2 * g + 1);
          const __m256i h0 = _mm256_and_si256(_mm256_srlv_epi32(hv, sh0), c1);
          const __m256i h1 = _mm256_and_si256(_mm256_srlv_epi32(hv, sh1), c1);
          const __m256i v0 = _mm256_or_si256(_mm256_and_si256(qb, c15), _mm256_slli_epi32(h0, 4));
          const __m256i v1 = _mm256_or_si256(_mm256_srli_epi32(qb, 4), _mm256_slli_epi32(h1, 4));
          const __m256 f0 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(v0), b0);
          const __m256 f1 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(v1), b1);
          acc = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x0 + l), acc);
          acc = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x0 + 32 + l), acc);
        }
      }
    }
    *out = hsum256(acc);
    return true;
  }
  if (type == "Q5_1") {
    const int nb = n_in / 32;
    __m256 acc = _mm256_setzero_ps();
    const __m256i c15 = _mm256_set1_epi32(0xF);
    const __m256i c1 = _mm256_set1_epi32(1);
    for (int b = 0; b < nb; ++b) {
      const unsigned char* p = row + static_cast<std::size_t>(b) * 24;
      const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
      const float m = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
      const unsigned int qh = static_cast<unsigned int>(p[4]) |
                              (static_cast<unsigned int>(p[5]) << 8) |
                              (static_cast<unsigned int>(p[6]) << 16) |
                              (static_cast<unsigned int>(p[7]) << 24);
      const unsigned char* qs = p + 8;
      const float* xb = x + static_cast<std::size_t>(b) * 32;
      const __m256 dv = _mm256_set1_ps(d);
      const __m256 mv = _mm256_set1_ps(m);
      for (int g = 0; g < 4; ++g) {
        const int base = g * 8;
        const int src = (base < 16) ? base : (base - 16);
        const __m128i qbv = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + src));
        __m256i v = _mm256_cvtepu8_epi32(qbv);
        v = (base < 16) ? _mm256_and_si256(v, c15) : _mm256_srli_epi32(v, 4);
        const __m256i shifts = _mm256_add_epi32(
            _mm256_set1_epi32(base), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        const __m256i hv = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(qh)), shifts), c1);
        v = _mm256_or_si256(v, _mm256_slli_epi32(hv, 4));
        const __m256 f = _mm256_fmadd_ps(dv, _mm256_cvtepi32_ps(v), mv);
        acc = _mm256_fmadd_ps(f, _mm256_loadu_ps(xb + base), acc);
      }
    }
    *out = hsum256(acc);
    return true;
  }
  if (type == "Q6_K") {
    avx2_dot_q6k(row, x, n_in, out);
    return true;
  }
  return false;
}
#endif  // __AVX2__ && __FMA__

bool dot_row(const std::string& type, const unsigned char* row, const float* x, int n_in,
             float* out) {
#if defined(ENGINE_HAVE_AVX2)
  static const bool no_avx2 = engine::config::get_bool("no_avx2");
  if (!no_avx2 && avx2_dot_row(type, row, x, n_in, out)) return true;
#endif
  if (type == "Q4_K") {
    const int nb = n_in / 256;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
      const unsigned char* p = row + static_cast<std::size_t>(b) * 144;
      const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
      const float dmin = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
      const unsigned char* scales = p + 4;
      const unsigned char* qs = p + 16;
      float scf[8], mnf[8];
      for (int sub = 0; sub < 8; ++sub) {
        int sc = 0, mn = 0;
        scale_min(sub, scales, &sc, &mn);
        scf[sub] = d * static_cast<float>(sc);
        mnf[sub] = dmin * static_cast<float>(mn);
      }
      const float* xb = x + static_cast<std::size_t>(b) * 256;
      for (int g = 0; g < 4; ++g) {
        const unsigned char* q = qs + g * 32;
        const float s0 = scf[2 * g], m0 = mnf[2 * g];
        const float s1 = scf[2 * g + 1], m1 = mnf[2 * g + 1];
        const float* x0 = xb + g * 64;
        const float* x1 = x0 + 32;
        for (int l = 0; l < 32; ++l) {
          const unsigned char qb = q[l];
          acc += (s0 * static_cast<float>(qb & 0xF) - m0) * x0[l];
          acc += (s1 * static_cast<float>(qb >> 4) - m1) * x1[l];
        }
      }
    }
    *out = acc;
    return true;
  }
  if (type == "Q5_K") {
    const int nb = n_in / 256;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
      const unsigned char* p = row + static_cast<std::size_t>(b) * 176;
      const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
      const float dmin = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
      const unsigned char* scales = p + 4;
      const unsigned char* qh = p + 16;
      const unsigned char* qs = p + 48;
      float scf[8], mnf[8];
      for (int sub = 0; sub < 8; ++sub) {
        int sc = 0, mn = 0;
        scale_min(sub, scales, &sc, &mn);
        scf[sub] = d * static_cast<float>(sc);
        mnf[sub] = dmin * static_cast<float>(mn);
      }
      const float* xb = x + static_cast<std::size_t>(b) * 256;
      for (int g = 0; g < 4; ++g) {
        const unsigned char* q = qs + g * 32;
        const int sub0 = 2 * g, sub1 = sub0 + 1;
        const float s0 = scf[sub0], m0 = mnf[sub0];
        const float s1 = scf[sub1], m1 = mnf[sub1];
        const float* x0 = xb + g * 64;
        const float* x1 = x0 + 32;
        for (int l = 0; l < 32; ++l) {
          const unsigned char qb = q[l];
          const int hi = (qh[l] >> sub0) & 1;
          acc += (s0 * static_cast<float>((qb & 0xF) | (hi << 4)) - m0) * x0[l];
          const int hi2 = (qh[l] >> sub1) & 1;
          acc += (s1 * static_cast<float>((qb >> 4) | (hi2 << 4)) - m1) * x1[l];
        }
      }
    }
    *out = acc;
    return true;
  }
  if (type == "Q6_K") {
    const int nb = n_in / 256;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
      const unsigned char* p = row + static_cast<std::size_t>(b) * 210;
      const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 208));
      const unsigned char* ql = p;
      const unsigned char* qh = p + 128;
      const signed char* sc = reinterpret_cast<const signed char*>(p + 192);
      const float* xb = x + static_cast<std::size_t>(b) * 256;
      for (int n2 = 0; n2 < 2; ++n2) {
        const int qi = n2 * 64, hi = n2 * 32;
        const float* x0 = xb + n2 * 128;
        for (int l = 0; l < 32; ++l) {
          const int is = l / 16;
          const unsigned char h = qh[hi + l];
          const int q1 = static_cast<int>((ql[qi + l] & 0xF) | (((h >> 0) & 3) << 4)) - 32;
          const int q2 = static_cast<int>((ql[qi + l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32;
          const int q3 = static_cast<int>((ql[qi + l] >> 4) | (((h >> 4) & 3) << 4)) - 32;
          const int q4 = static_cast<int>((ql[qi + l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;
          const signed char* s0 = sc + n2 * 8 + is;
          acc += d * static_cast<float>(s0[0]) * static_cast<float>(q1) * x0[l];
          acc += d * static_cast<float>(s0[2]) * static_cast<float>(q2) * x0[l + 32];
          acc += d * static_cast<float>(s0[4]) * static_cast<float>(q3) * x0[l + 64];
          acc += d * static_cast<float>(s0[6]) * static_cast<float>(q4) * x0[l + 96];
        }
      }
    }
    *out = acc;
    return true;
  }
  if (type == "Q8_0") {
    const int nb = n_in / 32;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
      const unsigned char* p = row + static_cast<std::size_t>(b) * 34;
      const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
      const signed char* q = reinterpret_cast<const signed char*>(p + 2);
      const float* xb = x + static_cast<std::size_t>(b) * 32;
      float lp = 0.f;
      for (int i = 0; i < 32; ++i) lp += static_cast<float>(q[i]) * xb[i];
      acc += d * lp;
    }
    *out = acc;
    return true;
  }
  return false;
}

}  // namespace

namespace {

// Persistent worker pool: matvec_mt used to spawn/join 11 threads per call, and the
// CPU expert path makes hundreds of calls per token (thread creation alone cost
// 100-200 ms/token on the 35B).
//
// A condvar handoff for one parallel region measured ~35 us on this (virtualised)
// machine: 11 waiters all take the pool mutex on wake-up and the main thread then
// blocks on a second condvar.  moe_forward_mt has three regions per CPU layer and
// the 35B has 25 CPU layers per token, so that handoff alone was ~2.6 ms/token.
// The pool therefore hands work over through an atomic generation counter: an
// idle worker spins (with PAUSE) for a short window that covers the tail of the
// preceding region, and only falls back to the condvar for long idle gaps
// (GPU-only layers).  A side effect is that the main thread no longer sleeps in
// the middle of a region.
namespace {
#if defined(ENGINE_HAVE_AVX2)
inline void pool_pause() { _mm_pause(); }
#else
inline void pool_pause() {}
#endif
// ~140 cycles per PAUSE on this core family: 512 pauses is roughly 20 us.
int pool_spin_count = 512;
}  // namespace

class Pool {
 public:
  static Pool& instance() {
    static Pool p;
    return p;
  }
  int size() const { return nthreads_; }

  // Runs fn(i) for i = thread_id, thread_id + nthreads, ... across the pool and
  // returns only when every worker has finished its share.
  void run(int n, const std::function<void(int)>& fn) {
    const int nw = static_cast<int>(workers_.size());
    if (nw == 0 || n <= 0) {
      for (int i = 0; i < n; ++i) fn(i);
      return;
    }
    {
      // The generation bump and the notification must happen under the same mutex the
      // workers use for their predicate check, otherwise a worker that is descheduled
      // between its predicate check and cv_.wait() (which happens under CPU contention
      // on a loaded VM) can miss the wake-up and sleep forever while the main thread
      // spins on remaining_ -- a hard hang.  Observed twice in a row on an otherwise
      // idle-looking box; fixed here.
      std::lock_guard<std::mutex> lk(m_);
      fn_ = &fn;
      n_ = n;
      remaining_.store(nw, std::memory_order_relaxed);
      gen_.fetch_add(1, std::memory_order_release);
      cv_.notify_all();
    }
    for (int i = 0; i < n; i += nthreads_) fn(i);
    int spins = 0;
    while (remaining_.load(std::memory_order_acquire) != 0) {
      pool_pause();
      if (++spins > 4096) {  // a worker is late: stop burning issue slots
        std::this_thread::yield();
        spins = 0;
      }
    }
  }

 private:
  Pool() {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 2;
    pool_spin_count = static_cast<int>(engine::config::get_int("cpu_pool_spin"));
    // config key `threads`: total pool threads, 0 = auto.
    //
    // Auto does NOT use every logical CPU.  This path is a memory-latency-bound unpack
    // loop and filling every SMT slot is measurably slower (bench_cpu_layer: 12 threads
    // 17.4 GB/s, 9 threads 23 GB/s); leaving ~1/4 of the logical CPUs free also keeps
    // the OS / WDDM driver (submitting this layer's GPU work concurrently) responsive.
    // Thread count.  The measured optimum on this host is 1.5x the *physical* core
    // count (9 threads on 6 cores / 12 logical): the SMT siblings do help this
    // latency-bound unpack loop, but the full logical count overprovisions (12 threads
    // were ~25% slower than 9).  Derive it from the topology, clamped to [2, hw-1] so a
    // slot stays free for the OS / WDDM driver; fall back to the historical
    // "3/4 of logical" rule when the physical count is unknown (VM without topology
    // information).
    const int phys = physical_cores();
    int nthreads;
    if (phys > 0 && phys < static_cast<int>(hw)) {
      nthreads = (3 * phys + 1) / 2;
      if (nthreads > static_cast<int>(hw) - 1) nthreads = static_cast<int>(hw) - 1;
    } else {
      nthreads = static_cast<int>(hw) - std::max(2, static_cast<int>(hw) / 4);
    }
    if (nthreads < 2) nthreads = 2;
    const int64_t want = engine::config::get_int("threads");
    if (want > 0) nthreads = static_cast<int>(want);
    if (nthreads < 1) nthreads = 1;
    const int extra = nthreads - 1;
    nthreads_ = nthreads;
#if defined(_WIN32)
    // Real-desktop scheduling knobs (see include/engine/config.h and
    // WIN-PRIORITY-REPORT.md).  The *process* class is applied once at startup by
    // config::apply_win_priority(); here the workers get their relative thread
    // priority and the optional affinity pinning.
    //
    // Default (win_priority=1): process HIGH (base 13) + workers THREAD_PRIORITY_HIGHEST
    // => 15.  That is still below the REALTIME range, so input/IO and the WDDM
    // submission threads of every other process keep running; it only makes the
    // GEMV workers win the time-slice lottery against desktop background apps.
    {
      const int64_t wp = engine::config::get_int("win_priority");
      int64_t wtp = engine::config::get_int("win_thread_priority");
      if (wtp < 0) wtp = wp > 0 ? 2 : 0;
      if (wtp > 2) wtp = 2;
      worker_rel_priority_ = wtp == 2 ? static_cast<int>(THREAD_PRIORITY_HIGHEST)
                             : wtp == 1 ? static_cast<int>(THREAD_PRIORITY_ABOVE_NORMAL)
                                        : 0;
      affinity_mode_ = static_cast<int>(engine::config::get_int("win_affinity"));
      if (affinity_mode_ > 0) affinity_order_ = physical_first_cpu_order();
      if (affinity_order_.empty()) affinity_mode_ = 0;
      if (affinity_mode_ > 0 && nthreads_ + 1 > static_cast<int>(affinity_order_.size()))
        affinity_mode_ = 0;  // not enough CPUs to pin without sharing
      if ((affinity_mode_ > 0 || worker_rel_priority_ != 0) &&
          std::getenv("ENGINE_DBG_SCHED") == nullptr) {
        // Keep the default logs quiet: one line only when the user asks for it.
      }
      if (std::getenv("ENGINE_DBG_SCHED")) {
        std::fprintf(stderr,
                     "[cpu] sched: win_priority=%lld worker_rel=%d affinity=%d cpus=%zu\n",
                     static_cast<long long>(wp), worker_rel_priority_, affinity_mode_,
                     affinity_order_.size());
      }
    }
#endif
    for (int t = 0; t < extra; ++t) {
      workers_.emplace_back([this, t] {
#if defined(_WIN32)
        if (worker_rel_priority_ != 0)
          SetThreadPriority(GetCurrentThread(), worker_rel_priority_);
        if (affinity_mode_ > 0 && !affinity_order_.empty()) {
          const int cpu = affinity_order_[static_cast<size_t>(t + 1) % affinity_order_.size()];
          SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu);
        }
#endif
        loop(t + 1);
      });
    }
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_.store(true, std::memory_order_relaxed);
      fn_ = nullptr;
    }
    gen_.fetch_add(1, std::memory_order_release);
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }
  void loop(int id) {
    long long seen = 0;
    for (;;) {
      long long g = gen_.load(std::memory_order_acquire);
      if (g == seen) {
        bool got = false;
        for (int i = 0; i < pool_spin_count; ++i) {
          pool_pause();
          g = gen_.load(std::memory_order_acquire);
          if (g != seen) {
            got = true;
            break;
          }
        }
        if (!got) {
          std::unique_lock<std::mutex> lk(m_);
          if (stop_.load(std::memory_order_relaxed)) return;
          cv_.wait(lk, [&] {
            return stop_.load(std::memory_order_relaxed) ||
                   gen_.load(std::memory_order_acquire) != seen;
          });
          if (stop_.load(std::memory_order_relaxed)) return;
          g = gen_.load(std::memory_order_acquire);
        }
      }
      if (stop_.load(std::memory_order_relaxed)) return;
      seen = g;
      const std::function<void(int)>* fn = fn_;
      const int n = n_;
      if (fn == nullptr) continue;
      for (int i = id; i < n; i += nthreads_) (*fn)(i);
      remaining_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  std::vector<std::thread> workers_;
  int nthreads_ = 1;
  int worker_rel_priority_ = 0;  // THREAD_PRIORITY_* relative value for the workers
  int affinity_mode_ = 0;        // win_affinity: 0 = off, 1 = physical-core-first pinning
  std::vector<int> affinity_order_;
  std::mutex m_;
  std::condition_variable cv_;
  const std::function<void(int)>* fn_ = nullptr;
  int n_ = 0;
  std::atomic<long long> gen_{0};
  std::atomic<bool> stop_{false};
  std::atomic<int> remaining_{0};
};

enum class RowKind { Other = 0, Q4K, Q6K, Q5K, Q5_1, Q8_0 };

RowKind row_kind_of(const std::string& t) {
  if (t == "Q4_K") return RowKind::Q4K;
  if (t == "Q6_K") return RowKind::Q6K;
  if (t == "Q5_K") return RowKind::Q5K;
  if (t == "Q5_1") return RowKind::Q5_1;
  if (t == "Q8_0") return RowKind::Q8_0;
  return RowKind::Other;
}

// Everything needed to dot one row: the quant type (dispatched once per phase
// instead of once per row), the input width and the row stride in bytes (the old
// per-row path recomputed `n_in / block_elems * block_bytes` with a runtime
// integer division for every row).
struct RowDot {
  RowKind kind = RowKind::Other;
  std::string type;
  int n_in = 0;
  int row_bytes = 0;
};

RowDot make_row_dot(const std::string& type, int n_in) {
  RowDot rd;
  rd.kind = row_kind_of(type);
  rd.type = type;
  rd.n_in = n_in;
  const QK qk = qk_of(type);
  rd.row_bytes = (n_in / qk.block_elems) * qk.block_bytes;
  return rd;
}

inline void prefetch_row(const unsigned char* row, int row_bytes) {
#if defined(ENGINE_HAVE_AVX2)
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + i), _MM_HINT_T0);
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + row_bytes + i), _MM_HINT_T1);
#else
  (void)row;
  (void)row_bytes;
#endif
}

// One row dot product with the type dispatch hoisted out of the row loop.
inline void row_dot(const RowDot& rd, const unsigned char* row, const float* x, float* out) {
#if defined(ENGINE_HAVE_AVX2)
  static const bool no_avx2 = engine::config::get_bool("no_avx2");
  if (!no_avx2) {
    switch (rd.kind) {
      case RowKind::Q4K:
        avx2_dot_q4k(row, x, rd.n_in, out);  // kernel prefetches internally
        return;
      case RowKind::Q6K:
        avx2_dot_q6k(row, x, rd.n_in, out);
        return;
      case RowKind::Q5K:
      case RowKind::Q5_1:
      case RowKind::Q8_0:
        prefetch_row(row, rd.row_bytes);
        if (dot_row(rd.type, row, x, rd.n_in, out)) return;
        break;
      default:
        break;
    }
  }
#endif
  if (!dot_row(rd.type, row, x, rd.n_in, out)) matvec(rd.type, row, 1, rd.n_in, x, out);
}

// Kernel *selection* is by quant type and width, never by model.  Both Q4_K and Q6_K
// rows that are a whole number of 256-value super-blocks have a two-level int8
// kernel; which of the two types actually uses it is a measured property of the host
// (the fp32 Q6_K kernel spends 1.4 instructions per value unpacking 6-bit weights,
// the fp32 Q4_K kernel only 0.9, so int8 wins on Q6_K and loses on Q4_K on this
// machine -- config `cpu_int8`: 0 off, 1 = measured set, 2 = both).
// Everything else (Q5_K/Q5_1/Q8_0, odd widths) keeps the fp32 kernel above.
inline bool int8_row_supported(const RowDot& rd, int level) {
  if (level <= 0) return false;
  if (rd.n_in < 256 || (rd.n_in % 256) != 0) return false;
  if (rd.kind == RowKind::Q6K) return true;
  if (rd.kind == RowKind::Q4K) return level >= 2;
  return false;
}
inline int cpu_int8_level() {
  const int64_t v = engine::config::get_int("cpu_int8");
  return v < 0 ? 0 : (v > 2 ? 2 : static_cast<int>(v));
}

inline void row_dot_i8(const RowDot& rd, const unsigned char* row, const XQ8& x, float* out) {
  switch (rd.kind) {
    case RowKind::Q4K:
      avx2_dot_q4k_i8(row, x, rd.n_in, out);
      return;
    case RowKind::Q6K:
      avx2_dot_q6k_i8(row, x, rd.n_in, out);
      return;
    default:
      row_dot(rd, row, nullptr, out);  // unreachable: guarded by int8_row_supported
      return;
  }
}

}  // namespace

// Multithreaded variant: rows are independent, so split them across cores. The CPU
// expert path is 10-100x slower than the GPU one, and this was the 35B bottleneck.
void matvec_mt(const std::string& type, const unsigned char* w, int n_out, int n_in,
               const float* x, float* y) {
  Pool& pool = Pool::instance();
  const int nthreads = pool.size();
  static bool once = false;
  if (!once) { once = true; std::fprintf(stderr, "[cpu] pool threads: %d\n", nthreads); }
  if (nthreads <= 1 || n_out < 64) {
    matvec(type, w, n_out, n_in, x, y);
    return;
  }
  const RowDot rd = make_row_dot(type, n_in);
  // int8 path: quantise the activation once (two levels / 32 values) and run the
  // int8 kernels.  Config `cpu_int8` (default true) selects it; the choice is per
  // (quant type, width), see int8_row_supported().
  const bool i8 = int8_row_supported(rd, cpu_int8_level());
  XQ8 xq;
  if (i8) xq_build(x, n_in, &xq);
  const int chunk = (n_out + nthreads - 1) / nthreads;
  const int nchunks = (n_out + chunk - 1) / chunk;
  pool.run(nchunks, [&](int ci) {
    const int r0 = ci * chunk;
    const int r1 = std::min(n_out, r0 + chunk);
    for (int r = r0; r < r1; ++r) {
      const unsigned char* row = w + static_cast<std::size_t>(r) * rd.row_bytes;
      float acc = 0.f;
      if (i8) row_dot_i8(rd, row, xq, &acc);
      else row_dot(rd, row, x, &acc);
      y[r] = acc;
    }
  });
}

void matvec(const std::string& type, const unsigned char* w, int n_out, int n_in,
            const float* x, float* y) {
  const QK qk = qk_of(type);
  const int nb_row = n_in / qk.block_elems;
  std::vector<float> tmp(static_cast<std::size_t>(qk.block_elems));
  for (int r = 0; r < n_out; ++r) {
    const unsigned char* row = w + static_cast<std::size_t>(r) * nb_row * qk.block_bytes;
    float acc = 0.f;
    for (int b = 0; b < nb_row; ++b) {
      dequant_block(type, row + static_cast<std::size_t>(b) * qk.block_bytes, tmp.data());
      const float* xb = x + static_cast<std::size_t>(b) * qk.block_elems;
      for (int e = 0; e < qk.block_elems; ++e) acc += tmp[e] * xb[e];
    }
    y[r] = acc;
  }
}

namespace {

// Scratch for moe_forward_mt.  Plain statics (NOT thread_local): the pool workers
// access them through a captured reference, so every thread has to see the same
// buffer.  The pool itself is a process-wide singleton, so the callers are already
// serialised.
std::vector<float> g_moe_a;
std::vector<float> g_moe_d;

}  // namespace

void moe_forward_mt(const std::string& tg, const unsigned char* const* wg,
                    const std::string& tu, const unsigned char* const* wu,
                    const std::string& td, const unsigned char* const* wd,
                    const int* tokens, const float* act_w, int n_act, int n_tokens, int hidden,
                    int inter, const float* x, float* xout) {
  if (n_act <= 0) return;
  Pool& pool = Pool::instance();
  const int nthreads = pool.size();
  static const bool dbg_moe = std::getenv("ENGINE_DBG_MOE_PHASES") != nullptr;
  const auto t_ph0 = std::chrono::steady_clock::now();
  const RowDot rg = make_row_dot(tg, hidden);
  const RowDot ru = make_row_dot(tu, hidden);
  const RowDot rd = make_row_dot(td, inter);
  // Kernel selection by (quant type, width): the two int8 paths are independent, so
  // a layer whose gate/up are Q4_K but whose down is Q5_K uses int8 only for phase A.
  const int i8lvl = cpu_int8_level();
  const bool i8a = int8_row_supported(rg, i8lvl) && int8_row_supported(ru, i8lvl) &&
                   rg.n_in == ru.n_in;
  const bool i8b = int8_row_supported(rd, i8lvl);
  // Activation quantisation is done on the calling thread before the parallel region;
  // phase A has one activation vector per *token* (all experts of a token share it),
  // phase B one per routed expert activation.
  static std::vector<XQ8> g_xq_tok, g_xq_act;
  if (i8a) {
    g_xq_tok.resize(static_cast<std::size_t>(n_tokens));
    for (int t = 0; t < n_tokens; ++t) xq_build(x + static_cast<std::size_t>(t) * hidden, hidden, &g_xq_tok[static_cast<std::size_t>(t)]);
  }
  g_moe_a.resize(static_cast<std::size_t>(n_act) * inter);
  g_moe_d.resize(static_cast<std::size_t>(n_act) * hidden);
  float* a_buf = g_moe_a.data();
  float* d_buf = g_moe_d.data();

  auto run_chunked = [&](long long total, const std::function<void(long long, long long)>& body) {
    if (nthreads <= 1 || total <= 0) {
      body(0, total);
      return;
    }
    const long long chunk = (total + nthreads - 1) / nthreads;
    const int nch = static_cast<int>((total + chunk - 1) / chunk);
    pool.run(nch, [&](int ci) {
      const long long b = static_cast<long long>(ci) * chunk;
      const long long e = std::min(total, b + chunk);
      body(b, e);
    });
  };

  // Phase A: gate row + up row -> act = silu(gate) * up.  One parallel region for all
  // experts (one barrier per layer instead of one per matvec).
  run_chunked(static_cast<long long>(n_act) * inter, [&](long long b, long long e) {
    for (long long idx = b; idx < e; ++idx) {
      const int act = static_cast<int>(idx / inter);
      const int r = static_cast<int>(idx - static_cast<long long>(act) * inter);
      const float* xv = x + static_cast<std::size_t>(tokens[act]) * hidden;
      const unsigned char* gr = wg[act] + static_cast<std::size_t>(r) * rg.row_bytes;
      const unsigned char* ur = wu[act] + static_cast<std::size_t>(r) * ru.row_bytes;
      float g = 0.f, u = 0.f;
      if (i8a) {
        const XQ8& xq = g_xq_tok[static_cast<std::size_t>(tokens[act])];
        row_dot_i8(rg, gr, xq, &g);
        row_dot_i8(ru, ur, xq, &u);
      } else {
        row_dot(rg, gr, xv, &g);
        row_dot(ru, ur, xv, &u);
      }
      a_buf[static_cast<std::size_t>(act) * inter + r] = (g / (1.f + std::exp(-g))) * u;
    }
  });

  const auto t_ph1 = std::chrono::steady_clock::now();
  // Phase B: down row -> per-act partial output row.
  if (i8b) {
    g_xq_act.resize(static_cast<std::size_t>(n_act));
    for (int act = 0; act < n_act; ++act)
      xq_build(a_buf + static_cast<std::size_t>(act) * inter, inter,
               &g_xq_act[static_cast<std::size_t>(act)]);
  }
  run_chunked(static_cast<long long>(n_act) * hidden, [&](long long b, long long e) {
    for (long long idx = b; idx < e; ++idx) {
      const int act = static_cast<int>(idx / hidden);
      const int r = static_cast<int>(idx - static_cast<long long>(act) * hidden);
      const float* av = a_buf + static_cast<std::size_t>(act) * inter;
      const unsigned char* dr = wd[act] + static_cast<std::size_t>(r) * rd.row_bytes;
      float d = 0.f;
      if (i8b) row_dot_i8(rd, dr, g_xq_act[static_cast<std::size_t>(act)], &d);
      else row_dot(rd, dr, av, &d);
      d_buf[static_cast<std::size_t>(act) * hidden + r] = d;
    }
  });

  const auto t_ph2 = std::chrono::steady_clock::now();
  // Merge: same order as the per-expert loop (acts are ordered by expert, then token),
  // so the accumulation into xout keeps the historical order.
  run_chunked(hidden, [&](long long b, long long e) {
    for (long long j = b; j < e; ++j) {
      for (int act = 0; act < n_act; ++act) {
        xout[static_cast<std::size_t>(tokens[act]) * hidden + j] +=
            act_w[act] * d_buf[static_cast<std::size_t>(act) * hidden + j];
      }
    }
  });
  if (dbg_moe) {
    const auto t_ph3 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr, "[moeph] n_act=%d thr=%d A %.3f B %.3f merge %.3f ms\n", n_act, nthreads,
                 ms(t_ph0, t_ph1), ms(t_ph1, t_ph2), ms(t_ph2, t_ph3));
  }
}

}  // namespace cpu
