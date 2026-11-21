// Prototype: fused Q5_K dequant + tensor-core GEMM in ONE kernel (no fp16 mirror),
// A/B'd against the product path (dequant kernel -> fp16 slab -> cuBLAS f16).
//
// Why: a 9B prefill pays ~130 ms of fixed cost per forward because every weight slab is
// dequantised to a 12.2 GB fp16 mirror and then re-read by cuBLAS (~30 GB of traffic).
// A fused kernel reads only the 6.1 GB of quantised weights. This prototype measures
// whether that traffic reduction actually materialises on representative single-matrix
// shapes.  It is test-only code (lives under tests/, synthetic data, no product path
// changes) so the idea can be judged before any rewrite.
//
//   test_gen --bench-fused-gemm
#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "engine/cublas.h"
#include "engine/cuda.h"

namespace fused_proto {

inline const char* kSrc = R"cuda(
// ---- helpers copied from the product kernel module (self-contained prototype) ----
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

__device__ __forceinline__ void q4k_scale_min(int j, const unsigned char* q, int* sc, int* mn) {
  if (j < 4) {
    *sc = q[j] & 63;
    *mn = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 2) & 0x30);
    *mn = (q[j + 4] >> 4) | ((q[j] >> 2) & 0x30);
  }
}

__device__ __forceinline__ float q5k_elem(const unsigned char* p, int e) {
  const float d = half_to_float(*(const unsigned short*)(p));
  const float dmin = half_to_float(*(const unsigned short*)(p + 2));
  const unsigned char* scales = p + 4;
  const unsigned char* qh = p + 16;
  const unsigned char* qs = p + 48;
  const int g = e >> 6, r = e & 63, hi = r >> 5, l = r & 31;
  const int sub = g * 2 + hi;
  int sc = 0, mn = 0;
  q4k_scale_min(sub, scales, &sc, &mn);
  const float scale = d * (float)sc, bias = dmin * (float)mn;
  const unsigned char* qp = qs + (sub >> 1) * 32 + l;
  const int qb0 = qp[0];
  const int nib_sh = (sub & 1) * 4;
  const int h0 = (qh[l] >> sub) & 1;
  const int q0 = ((qb0 >> nib_sh) & 0xF) | (h0 << 4);
  return scale * (float)q0 - bias;
}

// ---- baseline half: Q5_K blocks -> fp16 slab (same as the product dequant kernel) ----
extern "C" __global__ void proto_dequant_q5k_f16(const unsigned char* __restrict__ blocks,
                                                 unsigned int* __restrict__ out,
                                                 long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const unsigned char* p = blocks + b * 176;
  out[b * 128 + threadIdx.x] =
      pf_pack_half2(q5k_elem(p, threadIdx.x * 2), q5k_elem(p, threadIdx.x * 2 + 1));
}

// ---- variant: vectorised dequant, 8 halves per thread, 16 B stores --------------
// One super-block = 256 values = 512 B; 32 threads cover it with one uint4 store each
// (thread t writes elements 8t..8t+7).  Fewer, wider transactions than the product
// kernel's 128 threads x 4 B, and each thread reuses the block scale once.
extern "C" __global__ void proto_dequant_q5k_f16_v2(const unsigned char* __restrict__ blocks,
                                                    uint4* __restrict__ out,
                                                    long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const int t = threadIdx.x & 31;
  const unsigned char* p = blocks + b * 176;
  unsigned short v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float f = q5k_elem(p, t * 8 + i);
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(v[i]) : "f"(f));
  }
  uint4 r;
  r.x = (unsigned int)v[0] | ((unsigned int)v[1] << 16);
  r.y = (unsigned int)v[2] | ((unsigned int)v[3] << 16);
  r.z = (unsigned int)v[4] | ((unsigned int)v[5] << 16);
  r.w = (unsigned int)v[6] | ((unsigned int)v[7] << 16);
  out[b * 32 + t] = r;
}

// ---- fused kernel: out[m][n] = sum_k x[m][k] * dequant(w[n][k]) ------------------
// Requires M % 16 == 0, N % 64 == 0, K % 256 == 0 (prototype shapes are chosen so).
// Block = 128 threads (4 warps), tile = 16 (m) x 64 (n); one mma.m16n8k16 per warp
// covers 16x8, each warp does two of them.  The whole K range is walked in 256-wide
// super-block steps: x tile and the dequantised weight tile are staged in shared memory,
// then the warps issue 16 mma steps.  Only the quantised weights are read from DRAM.
extern "C" __global__ void proto_gemm_q5k_f16_wmma(const unsigned int* __restrict__ x,
                                                   const unsigned char* __restrict__ w,
                                                   float* __restrict__ out, int M, int N,
                                                   int K) {
  const int m0 = blockIdx.x * 16;
  const int n0 = blockIdx.y * 64;
  __shared__ unsigned short sw[64][264];
  __shared__ unsigned short sx[16][264];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nblk = K >> 8;
  float acc0[4] = {0.f, 0.f, 0.f, 0.f};
  float acc1[4] = {0.f, 0.f, 0.f, 0.f};
  const int fr = lane >> 2;             // fragment row (0..7)
  const int fc = (lane & 3) * 2;        // fragment k-column (0,2,4,6)
  for (int kb = 0; kb < nblk; ++kb) {
    for (int i = threadIdx.x; i < 16 * 256; i += blockDim.x) {
      const int r = i >> 8, c = i & 255;
      sx[r][c] = reinterpret_cast<const unsigned short*>(x)[(size_t)(m0 + r) * (size_t)K +
                                                            (size_t)(kb << 8) + c];
    }
    for (int i = threadIdx.x; i < 64 * 256; i += blockDim.x) {
      const int col = i >> 8, e = i & 255;
      const unsigned char* p = w + (size_t)(n0 + col) * (size_t)nblk * 176 + (size_t)kb * 176;
      const float v = q5k_elem(p, e);
      unsigned short h;
      asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
      sw[col][e] = h;
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      const int kk = ks * 16;
      const unsigned int a0 = *reinterpret_cast<const unsigned int*>(&sx[fr][kk + fc]);
      const unsigned int a1 = *reinterpret_cast<const unsigned int*>(&sx[fr + 8][kk + fc]);
      const unsigned int a2 = *reinterpret_cast<const unsigned int*>(&sx[fr][kk + fc + 8]);
      const unsigned int a3 = *reinterpret_cast<const unsigned int*>(&sx[fr + 8][kk + fc + 8]);
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const int col = warp * 16 + t * 8 + (lane >> 2);
        const unsigned int b0 = *reinterpret_cast<const unsigned int*>(&sw[col][kk + fc]);
        const unsigned int b1 = *reinterpret_cast<const unsigned int*>(&sw[col][kk + fc + 8]);
        if (t == 0)
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        else
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
    __syncthreads();
  }
  // epilogue: C fragment (16x8): row = lane/4 (+8), col = (lane%4)*2 (+1)
  const int r0 = m0 + (lane >> 2);
  const int c0 = n0 + warp * 16 + (lane & 3) * 2;
  const size_t s = (size_t)N;
  out[(size_t)r0 * s + c0 + 0] = acc0[0];
  out[(size_t)r0 * s + c0 + 1] = acc0[1];
  out[(size_t)(r0 + 8) * s + c0 + 0] = acc0[2];
  out[(size_t)(r0 + 8) * s + c0 + 1] = acc0[3];
  out[(size_t)r0 * s + c0 + 8] = acc1[0];
  out[(size_t)r0 * s + c0 + 9] = acc1[1];
  out[(size_t)(r0 + 8) * s + c0 + 8] = acc1[2];
  out[(size_t)(r0 + 8) * s + c0 + 9] = acc1[3];
}
)cuda";

struct Shape {
  int M, N, K;
  const char* label;
};

// Deterministic synthetic Q5_K data: random bytes, but d/dmin are set to small positive
// fp16 values so the dequantised values are in a sane range.
inline void fill_q5k(std::vector<unsigned char>& w, unsigned seed) {
  unsigned s = seed;
  for (auto& b : w) {
    s = s * 1664525u + 1013904223u;
    b = (unsigned char)(s >> 24);
  }
  for (size_t off = 0; off + 176 <= w.size(); off += 176) {
    const unsigned short d = 0x2C00u;     // 0.0625
    const unsigned short dmin = 0x2400u;  // 0.03125
    std::memcpy(&w[off], &d, 2);
    std::memcpy(&w[off + 2], &dmin, 2);
  }
}

inline void fill_f16(std::vector<unsigned short>& x, unsigned seed) {
  unsigned s = seed;
  for (auto& h : x) {
    s = s * 1664525u + 1013904223u;
    const float f = ((float)((s >> 8) & 0x3FF) / 1024.f - 0.5f) * 2.f;
    unsigned short bits;
    unsigned int fb;
    std::memcpy(&fb, &f, 4);
    const unsigned int sign = (fb >> 16) & 0x8000u;
    const int e = (int)((fb >> 23) & 0xFF) - 127 + 15;
    const unsigned int m = (fb >> 13) & 0x3FFu;
    bits = (unsigned short)(sign | ((unsigned int)(e < 1 ? 1 : (e > 30 ? 30 : e)) << 10) | m);
    h = bits;
  }
}

inline double time_ms(void* fn, int gx, int gy, int gz, int bx, int by, int bz, size_t shm,
                      void** args, int iters) {
  // warm-up
  cuda::launch(fn, gx, gy, gz, bx, by, bz, shm, args);
  cuda::sync();
  double best = 1e9;
  for (int i = 0; i < iters; ++i) {
    cuda::Event e0, e1;
    e0.record();
    cuda::launch(fn, gx, gy, gz, bx, by, bz, shm, args);
    e1.record();
    e1.synchronize();
    const double ms = e0.elapsed_ms(e1);
    if (ms < best) best = ms;
  }
  return best;
}

inline void bench_one(const Shape& s, void* dq_fn, void* dq_v2_fn, void* fused_fn) {
  const size_t nblk_total = (size_t)s.N * (size_t)s.K / 256;
  const size_t w_bytes = nblk_total * 176;
  std::vector<unsigned char> w_host(w_bytes);
  fill_q5k(w_host, 12345u);
  std::vector<unsigned short> x_host((size_t)s.M * s.K);
  fill_f16(x_host, 777u);
  std::vector<unsigned short> slab_host((size_t)s.N * s.K);
  std::vector<float> out_ref((size_t)s.M * s.N);

  cuda::Buffer w_dev(w_bytes), x_dev((size_t)s.M * s.K * 2), slab_dev((size_t)s.N * s.K * 2),
      out_ref_dev((size_t)s.M * s.N * 4), out_fused_dev((size_t)s.M * s.N * 4);
  w_dev.upload(w_host.data(), w_bytes);
  x_dev.upload(x_host.data(), (size_t)s.M * s.K * 2);

  // ---- baseline: dequant into the fp16 slab, then cuBLAS ----
  void* dq_args_real[3];
  void* w_ptr = w_dev.data();
  void* slab_ptr = slab_dev.data();
  long long nb = (long long)nblk_total;
  dq_args_real[0] = &w_ptr;
  dq_args_real[1] = &slab_ptr;
  dq_args_real[2] = &nb;
  const double dq_ms = time_ms(dq_fn, (int)nblk_total, 1, 1, 128, 1, 1, 0, dq_args_real, 5);
  const double dq_v2_ms =
      dq_v2_fn ? time_ms(dq_v2_fn, (int)nblk_total, 1, 1, 32, 1, 1, 0, dq_args_real, 5)
               : -1.0;

  int m = s.M, n = s.N, k = s.K;
  const double gemm_ms = [&]() {
    cublas::gemm_f16_nt_ld(m, n, k, x_dev.data(), slab_dev.data(), (float*)out_ref_dev.data(), n);
    cuda::sync();
    double best = 1e9;
    for (int i = 0; i < 5; ++i) {
      cuda::Event e0, e1;
      e0.record();
      cublas::gemm_f16_nt_ld(m, n, k, x_dev.data(), slab_dev.data(), (float*)out_ref_dev.data(),
                             n);
      e1.record();
      e1.synchronize();
      const double ms = e0.elapsed_ms(e1);
      if (ms < best) best = ms;
    }
    return best;
  }();
  // ---- fused ----
  void* fused_args[6];
  void* xf_ptr = x_dev.data();
  void* wf_ptr = w_dev.data();
  void* of_ptr = out_fused_dev.data();
  int M2 = s.M, N2 = s.N, K2 = s.K;
  fused_args[0] = &xf_ptr;
  fused_args[1] = &wf_ptr;
  fused_args[2] = &of_ptr;
  fused_args[3] = &M2;
  fused_args[4] = &N2;
  fused_args[5] = &K2;
  const double fused_ms =
      time_ms(fused_fn, s.M / 16, s.N / 64, 1, 128, 1, 1, 0, fused_args, 5);

  // ---- numerics: fused vs dequant+cuBLAS ----
  out_ref_dev.download(out_ref.data(), (size_t)s.M * s.N * 4);
  std::vector<float> out_fused((size_t)s.M * s.N);
  out_fused_dev.download(out_fused.data(), (size_t)s.M * s.N * 4);
  double num = 0, den = 0, mx = 0;
  for (size_t i = 0; i < out_ref.size(); ++i) {
    const double d = (double)out_fused[i] - (double)out_ref[i];
    num += d * d;
    den += (double)out_ref[i] * out_ref[i];
    if (std::fabs(d) > mx) mx = std::fabs(d);
  }
  const double rel = std::sqrt(num / (den > 0 ? den : 1));
  std::printf(
      "  [%s] baseline(dequant+cuBLAS) %8.3f ms = dq %6.3f + gemm %6.3f"
      " | dq_v2 %6.3f ms (%.2fx) | fused %8.3f ms | rel_l2 %.2e max|diff| %.2e\n",
      s.label, dq_ms + gemm_ms, dq_ms, gemm_ms, dq_v2_ms,
      dq_v2_ms > 0 ? dq_ms / dq_v2_ms : 0.0, fused_ms, rel, mx);
}

inline void run_bench() {
  cuda::init();
  cublas::init();
  std::printf("[fused-gemm] prototype: fused Q5_K dequant+GEMM vs dequant->fp16->cuBLAS\n");
  std::printf("[fused-gemm] note: synthetic weights, min-of-5 event timings, exclusive GPU\n");
  const cuda::DeviceInfo& di = cuda::device_info();
  std::printf("[fused-gemm] device: %s (sm_%d%d, %d SMs)\n", di.name.c_str(), di.cc_major,
              di.cc_minor, di.sm_count);
  std::string ptx;
  try {
    ptx = cuda::compile_ptx(kSrc, "fused_gemm_proto");
  } catch (const std::exception& e) {
    std::printf("[fused-gemm] NVRTC compile FAILED: %s\n", e.what());
    return;
  }
  cuda::Module mod(ptx);
  void* dq_fn = mod.kernel("proto_dequant_q5k_f16");
  void* dq_v2_fn = mod.kernel("proto_dequant_q5k_f16_v2");
  void* fused_fn = mod.kernel("proto_gemm_q5k_f16_wmma");
  const Shape shapes[] = {
      {512, 8192, 2048, "M512  N8192 K2048 (ffn_up-like)"},
      {512, 2048, 8192, "M512  N2048 K8192 (ffn_down-like)"},
      {512, 2048, 2048, "M512  N2048 K2048"},
      {16, 8192, 2048, "M16   N8192 K2048 (tail)"},
  };
  for (const auto& s : shapes) bench_one(s, dq_fn, dq_v2_fn, fused_fn);
  std::printf(
      "[fused-gemm] traffic per matrix: baseline reads quantised weights once, writes and "
      "re-reads the fp16 slab (3x the quantised bytes); fused reads the quantised weights "
      "only\n");
}

}  // namespace fused_proto
