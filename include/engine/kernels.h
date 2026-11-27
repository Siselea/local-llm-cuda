// CUDA kernel sources compiled at runtime by NVRTC (no nvcc needed).
// All kernels are float32 and correctness-first; optimized variants come later.
#pragma once

#include <string>

#include "engine/fp16.h"

namespace kernels {

inline const char* kRmsNorm = R"cuda(
extern "C" __global__ void rmsnorm(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   float* __restrict__ y,
                                   int n, float eps) {
  extern __shared__ float s[];
  const int row = blockIdx.x;
  const float* xr = x + (long long)row * n;
  float sum = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) sum += xr[i] * xr[i];
  s[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(s[0] / (float)n + eps);
  float* yr = y + (long long)row * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * scale * w[i];
}
)cuda";

// rotary embedding, in-place on q or k: x[L, n_heads*head_dim], positions 0..L-1
inline const char* kRope = R"cuda(
extern "C" __global__ void rope(float* __restrict__ x, const int* __restrict__ pstart,
                                int n_heads, int head_dim, const float* __restrict__ freqs,
                                float mscale) {
  const int row = blockIdx.x;          // row in this launch (decode: always 0)
  const int pos = *pstart + row;       // absolute position for theta
  const int h = blockIdx.y;
  const int half = head_dim / 2;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const float ang = (float)pos * freqs[i];
    const float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
    float* base_ptr = x + ((long long)row * n_heads + h) * head_dim;
    const float x1 = base_ptr[i];
    const float x2 = base_ptr[i + half];
    base_ptr[i] = x1 * c - x2 * s;
    base_ptr[i + half] = x1 * s + x2 * c;
  }
}
)cuda";

// Reference causal GQA attention used by tests/test_layer.cpp for the synthetic-layer
// check.  The engine itself does not launch this kernel (the tiled attention kernels
// above/below are the production paths), so it lives on as a test asset only.
// q: [L, n_head*D], k/v: [L, n_kv*D], out: [L, n_head*D]
inline const char* kAttn = R"cuda(
extern "C" __global__ void attn(const float* __restrict__ q, const float* __restrict__ k,
                                const float* __restrict__ v, float* __restrict__ out,
                                int L, int n_head, int n_kv, int head_dim) {
  extern __shared__ float sh[];
  float* scores = sh;                 // [L]
  float* acc = sh + L;                // [head_dim]
  const int pos = blockIdx.x;
  const int h = blockIdx.y;
  const int rep = n_head / n_kv;
  const int kv = h / rep;
  const float scale = rsqrtf((float)head_dim);

  const float* qh = q + ((long long)pos * n_head + h) * head_dim;

  // thread 0 does the dot products / softmax (head_dim and L are small here)
  if (threadIdx.x == 0) {
    float maxs = -1e30f;
    for (int j = 0; j <= pos; ++j) {
      const float* kh = k + ((long long)j * n_kv + kv) * head_dim;
      float dot = 0.f;
      for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
      scores[j] = dot * scale;
      maxs = fmaxf(maxs, scores[j]);
    }
    float sum = 0.f;
    for (int j = 0; j <= pos; ++j) {
      scores[j] = expf(scores[j] - maxs);
      sum += scores[j];
    }
    const float inv = 1.0f / sum;
    for (int j = 0; j <= pos; ++j) scores[j] *= inv;
  }
  __syncthreads();

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float a = 0.f;
    for (int j = 0; j <= pos; ++j)
      a += scores[j] * v[((long long)j * n_kv + kv) * head_dim + d];
    acc[d] = a;
  }
  __syncthreads();
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    out[((long long)pos * n_head + h) * head_dim + d] = acc[d];
}
)cuda";

// elementwise silu(gate) * up, row-major [rows, n]
inline const char* kSiluMul = R"cuda(
extern "C" __global__ void silu_mul(const float* __restrict__ gate,
                                    const float* __restrict__ up,
                                    float* __restrict__ out, long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    const float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
  }
}
)cuda";

// elementwise a += b
inline const char* kAdd = R"cuda(
extern "C" __global__ void add_inplace(float* __restrict__ a, const float* __restrict__ b,
                                       long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) a[i] += b[i];
}
)cuda";

// elementwise a += (b + c) -- one submission for the two host-part / residual adds.
// Rounding is bit-identical to `b += c; a += b` (the temporary is still rounded once).
inline const char* kAddThree = R"cuda(
extern "C" __global__ void add_three_inplace(float* __restrict__ a, const float* __restrict__ b,
                                            const float* __restrict__ c, long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) a[i] += (b[i] + c[i]);
}
)cuda";

// ---- quantized block dequantization (K-quants) ----------------------------
// Shared helpers are emitted once and prepended to both dequant programs.
// half_to_float comes from engine::fp16::kDeviceSrc (include/engine/fp16.h), which is
// also the source of the host-side conversion -- so the device and host decoders can
// never drift apart again (tests/test_fp16.cpp checks all 65536 patterns).
inline const std::string kQuantCommon = engine::fp16::kDeviceSrc + R"cuda(
// 6-bit scale/min pair per 32-element sub-block (ggml get_scale_min_k4)
__device__ __forceinline__ void q4k_scale_min(int j, const unsigned char* q, int* sc, int* mn) {
  if (j < 4) {
    *sc = q[j] & 63;
    *mn = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 2) & 0x30);
    *mn = (q[j + 4] >> 4) | ((q[j] >> 2) & 0x30);
  }
}
)cuda";

// one thread per element; grid = n_blocks, block = 256
inline const char* kDequantQ4K = R"cuda(
extern "C" __global__ void dequant_q4k(const unsigned char* __restrict__ blocks,
                                       float* __restrict__ out, long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const unsigned char* p = blocks + b * 144;
  const float d = half_to_float(*(const unsigned short*)(p));
  const float dmin = half_to_float(*(const unsigned short*)(p + 2));
  const unsigned char* scales = p + 4;
  const unsigned char* qs = p + 16;
  const int e = threadIdx.x;
  const int g = e >> 6, r = e & 63, half = r >> 5, l = r & 31;
  const int sub = g * 2 + half;
  const int qb = qs[g * 32 + l];
  const int nib = half ? (qb >> 4) : (qb & 0xF);
  int sc = 0, mn = 0;
  q4k_scale_min(sub, scales, &sc, &mn);
  out[b * 256 + e] = d * (float)sc * (float)nib - dmin * (float)mn;
}
)cuda";

inline const char* kDequantQ5K = R"cuda(
extern "C" __global__ void dequant_q5k(const unsigned char* __restrict__ blocks,
                                       float* __restrict__ out, long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const unsigned char* p = blocks + b * 176;
  const float d = half_to_float(*(const unsigned short*)(p));
  const float dmin = half_to_float(*(const unsigned short*)(p + 2));
  const unsigned char* scales = p + 4;
  const unsigned char* qh = p + 16;
  const unsigned char* qs = p + 48;
  const int e = threadIdx.x;
  const int g = e >> 6, r = e & 63, half = r >> 5, l = r & 31;
  const int sub = g * 2 + half;
  const int qb = qs[g * 32 + l];
  const int nib = half ? (qb >> 4) : (qb & 0xF);
  const int hi = (qh[l] >> sub) & 1;
  int sc = 0, mn = 0;
  q4k_scale_min(sub, scales, &sc, &mn);
  out[b * 256 + e] = d * (float)sc * (float)(nib | (hi << 4)) - dmin * (float)mn;
}
)cuda";

inline const char* kDequantQ6K = R"cuda(
extern "C" __global__ void dequant_q6k(const unsigned char* __restrict__ blocks,
                                       float* __restrict__ out, long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const unsigned char* p = blocks + b * 210;
  const float d = half_to_float(*(const unsigned short*)(p + 208));
  const unsigned char* ql = p;
  const unsigned char* qh = p + 128;
  const signed char* sc = (const signed char*)(p + 192);
  const int l = threadIdx.x;  // 0..31
  const int is = l / 16;
  for (int n2 = 0; n2 < 2; ++n2) {
    const int qi = n2 * 64, hi = n2 * 32;
    const int q1 = (int)((ql[qi + l] & 0xF) | (((qh[hi + l] >> 0) & 3) << 4)) - 32;
    const int q2 = (int)((ql[qi + l + 32] & 0xF) | (((qh[hi + l] >> 2) & 3) << 4)) - 32;
    const int q3 = (int)((ql[qi + l] >> 4) | (((qh[hi + l] >> 4) & 3) << 4)) - 32;
    const int q4 = (int)((ql[qi + l + 32] >> 4) | (((qh[hi + l] >> 6) & 3) << 4)) - 32;
    float* o = out + b * 256 + n2 * 128;
    o[l + 0] = d * (float)sc[n2 * 8 + is + 0] * (float)q1;
    o[l + 32] = d * (float)sc[n2 * 8 + is + 2] * (float)q2;
    o[l + 64] = d * (float)sc[n2 * 8 + is + 4] * (float)q3;
    o[l + 96] = d * (float)sc[n2 * 8 + is + 6] * (float)q4;
  }
}
)cuda";

inline const char* kDequantQ8_0 = R"cuda(
extern "C" __global__ void dequant_q8_0(const unsigned char* __restrict__ blocks,
                                        float* __restrict__ out, long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const unsigned char* p = blocks + b * 34;
  const float d = half_to_float(*(const unsigned short*)(p));
  const signed char* q = (const signed char*)(p + 2);
  const int l = threadIdx.x;  // 0..31
  out[b * 32 + l] = d * (float)q[l];
}
)cuda";

// ---- fused dequant + matvec (decode: batch=1) -----------------------------
// y[o] = dot(dequant(W[o, :]), x).  grid = n_out, block = 128 threads.
// Each thread walks the row in stride-128 steps; scales are unpacked per super-block.
// 4 rows per block (1 warp per row), x staged in shared memory once per block.
// Each lane owns 8 consecutive elements inside one 32-element sub-block, so the
// 6-bit scale/min is unpacked once per (lane, super-block).
inline const char* kGemvQ4K = R"cuda(
extern "C" __global__ void gemv_q4k(const unsigned char* __restrict__ W,
                                    const float* __restrict__ x,
                                    float* __restrict__ y, int n_in, int n_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_out) return;
  const int sb_count = n_in / 256;
  const int k0 = lane * 8;
  const int g = k0 >> 6, r = k0 & 63, half = r >> 5, l0 = r & 31;
  const int sub = g * 2 + half;
  const unsigned char* rp = W + (long long)row * sb_count * 144;
  float acc = 0.f;
  for (int sb = 0; sb < sb_count; ++sb) {
    const unsigned char* p = rp + sb * 144;
    const float d = half_to_float(*(const unsigned short*)(p));
    const float dmin = half_to_float(*(const unsigned short*)(p + 2));
    int sc = 0, mn = 0;
    q4k_scale_min(sub, p + 4, &sc, &mn);
    const float scale = d * (float)sc, bias = dmin * (float)mn;
    const unsigned char* q = p + 16 + g * 32 + l0;
    const float* xp = x + sb * 256 + k0;
    float lp = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int byte = q[i];
      const int nib = half ? (byte >> 4) : (byte & 0xF);
      lp += (scale * (float)nib - bias) * xp[i];
    }
    acc += lp;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) y[row] = acc;
}
)cuda";

inline const char* kGemvQ5K = R"cuda(
extern "C" __global__ void gemv_q5k(const unsigned char* __restrict__ W,
                                    const float* __restrict__ x,
                                    float* __restrict__ y, int n_in, int n_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_out) return;
  const int sb_count = n_in / 256;
  const int k0 = lane * 8;
  const int g = k0 >> 6, r = k0 & 63, half = r >> 5, l0 = r & 31;
  const int sub = g * 2 + half;
  const unsigned char* rp = W + (long long)row * sb_count * 176;
  float acc = 0.f;
  for (int sb = 0; sb < sb_count; ++sb) {
    const unsigned char* p = rp + sb * 176;
    const float d = half_to_float(*(const unsigned short*)(p));
    const float dmin = half_to_float(*(const unsigned short*)(p + 2));
    const unsigned char* qh = p + 16;
    int sc = 0, mn = 0;
    q4k_scale_min(sub, p + 4, &sc, &mn);
    const float scale = d * (float)sc, bias = dmin * (float)mn;
    const unsigned char* q = p + 48 + g * 32 + l0;
    const float* xp = x + sb * 256 + k0;
    float lp = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int byte = q[i];
      const int nib = half ? (byte >> 4) : (byte & 0xF);
      const int hi = (qh[l0 + i] >> sub) & 1;
      lp += (scale * (float)(nib | (hi << 4)) - bias) * xp[i];
    }
    acc += lp;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) y[row] = acc;
}
)cuda";

inline const char* kGemvQ8_0 = R"cuda(
extern "C" __global__ void gemv_q8_0(const unsigned char* __restrict__ W,
                                     const float* __restrict__ x,
                                     float* __restrict__ y, int n_in, int n_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_out) return;
  const int nb = n_in / 32;
  const unsigned char* rp = W + (long long)row * nb * 34;
  float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = rp + b * 34;
    const float d = half_to_float(*(const unsigned short*)(p));
    const int q = (int)(signed char)p[2 + lane];
    acc += d * (float)q * x[b * 32 + lane];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) y[row] = acc;
}
)cuda";

// ---- KV cache (single sequence, contiguous per layer) ----------------------
// Kc/Vc layout: [n_kv][max_seq][D]. Prefill writes L positions; decode writes 1.
inline const char* kKvWrite = R"cuda(
extern "C" __global__ void kv_write(const float* __restrict__ k, const float* __restrict__ v,
                                    float* __restrict__ Kc, float* __restrict__ Vc,
                                    const int* __restrict__ pstart, int n_kv, int D,
                                    int max_seq) {
  const int idx = blockIdx.x;
  const int j = idx / n_kv, h = idx % n_kv;
  const int pos = *pstart + j;
  const long long dst = ((long long)h * max_seq + pos) * D;
  const long long src = ((long long)j * n_kv + h) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    Kc[dst + d] = k[src + d];
    Vc[dst + d] = v[src + d];
  }
}
)cuda";

// Reference causal KV-cache attention (one q row per block, warp per query head) used
// by tests/test_kv.cpp as the independent implementation to check kv_write.  The engine
// itself launches attn_tiled / attn_cached_w / attn_cached_cfg only.
// One warp per (query, head): lanes split the keys, scores staged in shared memory.
inline const char* kAttnCached = R"cuda(
extern "C" __global__ void attn_cached(const float* __restrict__ q,
                                       const float* __restrict__ Kc,
                                       const float* __restrict__ Vc,
                                       float* __restrict__ out,
                                       const int* __restrict__ pstart, int n_head, int n_kv,
                                       int D, int max_seq) {
  extern __shared__ float sh[];
  const unsigned mask = 0xffffffffu;
  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int qpos = *pstart + i;
  const int rep = n_head / n_kv;
  const int kvh = h / rep;
  const float scale = rsqrtf((float)D);
  const float* qh = q + ((long long)i * n_head + h) * D;
  const float* Kh = Kc + (long long)kvh * max_seq * D;
  const float* Vh = Vc + (long long)kvh * max_seq * D;

  for (int j = lane; j <= qpos; j += 32) {
    const float* kh = Kh + (long long)j * D;
    float dot = 0.f;
    for (int d = 0; d < D; ++d) dot += qh[d] * kh[d];
    sh[j] = dot * scale;
  }
  __syncwarp();
  float lmax = -1e30f;
  for (int j = lane; j <= qpos; j += 32) lmax = fmaxf(lmax, sh[j]);
  for (int off = 16; off > 0; off >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(mask, lmax, off));
  float lsum = 0.f;
  for (int j = lane; j <= qpos; j += 32) {
    const float e = expf(sh[j] - lmax);
    sh[j] = e;
    lsum += e;
  }
  for (int off = 16; off > 0; off >>= 1) lsum += __shfl_xor_sync(mask, lsum, off);
  __syncwarp();
  const float inv = 1.f / lsum;
  for (int d = lane; d < D; d += 32) {
    float a = 0.f;
    for (int j = 0; j <= qpos; ++j) a += sh[j] * inv * Vh[(long long)j * D + d];
    out[((long long)i * n_head + h) * D + d] = a;
  }
}
)cuda";

inline const char* kDequantQ5_1 = R"cuda(
extern "C" __global__ void dequant_q5_1(const unsigned char* __restrict__ blocks,
                                        float* __restrict__ out, long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const unsigned char* p = blocks + b * 24;
  const float d = half_to_float(*(const unsigned short*)(p));
  const float m = half_to_float(*(const unsigned short*)(p + 2));
  const unsigned int qh = (unsigned int)p[4] | ((unsigned int)p[5] << 8) |
                          ((unsigned int)p[6] << 16) | ((unsigned int)p[7] << 24);
  const unsigned char* qs = p + 8;
  const int i = threadIdx.x;
  const int low = (i < 16) ? (qs[i] & 0xF) : (qs[i - 16] >> 4);
  // gguf.quants reference: q = (low | bit_i<<4)  (no -16 offset), value = d*q + m
  const int hi = (int)((qh >> i) & 1u);
  out[b * 32 + i] = d * (float)(low | (hi << 4)) + m;
}
)cuda";

inline const char* kDequantQ5_0 = R"cuda(
extern "C" __global__ void dequant_q5_0(const unsigned char* __restrict__ blocks,
                                        float* __restrict__ out, long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const unsigned char* p = blocks + b * 22;
  const float d = half_to_float(*(const unsigned short*)(p));
  const unsigned int qh = (unsigned int)p[2] | ((unsigned int)p[3] << 8) |
                          ((unsigned int)p[4] << 16) | ((unsigned int)p[5] << 24);
  const unsigned char* qs = p + 6;
  const int i = threadIdx.x;  // 0..31
  const int low = (i < 16) ? (qs[i] & 0xF) : (qs[i - 16] >> 4);
  const int hi = (qh >> i) & 1u;
  const int q = (low | (hi << 4)) - 16;
  out[b * 32 + i] = d * (float)q;
}
)cuda";

// out[i, j] += bias[j]
inline const char* kAddBias = R"cuda(
extern "C" __global__ void add_bias(float* __restrict__ x, const float* __restrict__ bias,
                                    int n_cols, long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) x[i] += bias[i % n_cols];
}
)cuda";

// MoE routing helpers: gather rows by token index, scatter weighted rows back.
inline const char* kGatherRows = R"cuda(
extern "C" __global__ void gather_rows(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       const int* __restrict__ idx, int width) {
  const int row = blockIdx.x;
  const int s = idx[row];
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    dst[(long long)row * width + c] = src[(long long)s * width + c];
}
)cuda";

inline const char* kScatterAddW = R"cuda(
extern "C" __global__ void scatter_add_weighted(float* __restrict__ dst,
                                                const float* __restrict__ src,
                                                const int* __restrict__ idx,
                                                const float* __restrict__ w, int width) {
  const int row = blockIdx.x;
  const int s = idx[row];
  const float wt = w[row];
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    atomicAdd(&dst[(long long)s * width + c], src[(long long)row * width + c] * wt);
}
)cuda";

// ---- MLA (deepseek2) helpers ---------------------------------------------
// rotate only the [offset, offset+rope_dim) slice of each head (head_stride apart)
inline const char* kRopeStrided = R"cuda(
extern "C" __global__ void rope_strided(float* __restrict__ x, const int* __restrict__ pstart,
                                        int n_heads, int head_stride, int rope_dim, int offset,
                                        const float* __restrict__ freqs, float mscale) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int half = rope_dim / 2;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const float ang = (float)(*pstart + row) * freqs[i];
    const float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
    float* p = x + ((long long)row * n_heads + h) * head_stride + offset;
    const float x1 = p[i], x2 = p[i + half];
    p[i] = x1 * c - x2 * s;
    p[i + half] = x1 * s + x2 * c;
  }
}
)cuda";

// split a row into two column ranges: a = src[0:na], b = src[na:na+nb]
// rope_strided with per-pair frequency divisors (GGUF gemma4 rope_freqs.weight).
//   theta_i = pos * freqs[i] / ff[i]
// ff == nullptr keeps the plain behaviour.  The GGUF sentinel 1e30 makes
// theta_i underflow to 0 (cos=1, sin=0), i.e. that pair is not rotated: this is
// exactly llama.cpp's "proportional rope" for full-attention layers.
inline const char* kRopeStridedFF = R"cuda(
extern "C" __global__ void rope_strided_ff(float* __restrict__ x, const int* __restrict__ pstart,
                                           int n_heads, int head_stride, int rope_dim, int offset,
                                           const float* __restrict__ freqs,
                                           const float* __restrict__ ff, float mscale) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int half = rope_dim / 2;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    float ang = (float)(*pstart + row) * freqs[i];
    if (ff != nullptr) ang /= ff[i];
    const float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
    float* p = x + ((long long)row * n_heads + h) * head_stride + offset;
    const float x1 = p[i], x2 = p[i + half];
    p[i] = x1 * c - x2 * s;
    p[i + half] = x1 * s + x2 * c;
  }
}
)cuda";

inline const char* kSplit2 = R"cuda(
extern "C" __global__ void split2(const float* __restrict__ src, float* __restrict__ a,
                                  float* __restrict__ b, int rows, int na, int nb) {
  const int r = blockIdx.x;
  for (int i = threadIdx.x; i < na; i += blockDim.x) a[(long long)r * na + i] = src[(long long)r * (na + nb) + i];
  for (int i = threadIdx.x; i < nb; i += blockDim.x) b[(long long)r * nb + i] = src[(long long)r * (na + nb) + na + i];
}
)cuda";

// k[t, h] = [kn[t, h*qk_nope ...] | kpe[t, :]]   (rope applied afterwards)
inline const char* kMlaBuildK = R"cuda(
extern "C" __global__ void mla_build_k(const float* __restrict__ kn, const float* __restrict__ kpe,
                                       float* __restrict__ k, int n_head, int qk_nope,
                                       int qk_rope) {
  const int t = blockIdx.x;
  const int qk = qk_nope + qk_rope;
  for (int h = 0; h < n_head; ++h) {
    float* dst = k + ((long long)t * n_head + h) * qk;
    const float* src = kn + ((long long)t * n_head + h) * qk_nope;
    for (int d = threadIdx.x; d < qk_nope; d += blockDim.x) dst[d] = src[d];
    for (int d = threadIdx.x; d < qk_rope; d += blockDim.x)
      dst[qk_nope + d] = kpe[(long long)t * qk_rope + d];
  }
}
)cuda";

// GEMV for Q6_K / Q5_0 (warp per row, 8 rows per block), matching the 4-row family above.
inline const char* kGemvQ6K = R"cuda(
extern "C" __global__ void gemv_q6k(const unsigned char* __restrict__ W,
                                    const float* __restrict__ x,
                                    float* __restrict__ y, int n_in, int n_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_out) return;
  const int sb_count = n_in / 256;
  const unsigned char* rp = W + (long long)row * sb_count * 210;
  float acc = 0.f;
  const int is = lane / 16;
#pragma unroll 2
  for (int sb = 0; sb < sb_count; ++sb) {
    const unsigned char* p = rp + sb * 210;
    const float d = half_to_float(*(const unsigned short*)(p + 208));
    const unsigned char* ql = p;
    const unsigned char* qh = p + 128;
    const signed char* sc = (const signed char*)(p + 192);
    const float* xp = x + sb * 256;
#pragma unroll
    for (int n2 = 0; n2 < 2; ++n2) {
      const int qi = n2 * 64, hi = n2 * 32;
      const int q1 = (int)((ql[qi + lane] & 0xF) | (((qh[hi + lane] >> 0) & 3) << 4)) - 32;
      const int q2 = (int)((ql[qi + lane + 32] & 0xF) | (((qh[hi + lane] >> 2) & 3) << 4)) - 32;
      const int q3 = (int)((ql[qi + lane] >> 4) | (((qh[hi + lane] >> 4) & 3) << 4)) - 32;
      const int q4 = (int)((ql[qi + lane + 32] >> 4) | (((qh[hi + lane] >> 6) & 3) << 4)) - 32;
      const int base = n2 * 128;
      acc += d * (float)sc[n2 * 8 + is + 0] * (float)q1 * xp[base + lane];
      acc += d * (float)sc[n2 * 8 + is + 2] * (float)q2 * xp[base + lane + 32];
      acc += d * (float)sc[n2 * 8 + is + 4] * (float)q3 * xp[base + lane + 64];
      acc += d * (float)sc[n2 * 8 + is + 6] * (float)q4 * xp[base + lane + 96];
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) y[row] = acc;
}
)cuda";

inline const char* kGemvQ5_0 = R"cuda(
extern "C" __global__ void gemv_q5_0(const unsigned char* __restrict__ W,
                                     const float* __restrict__ x,
                                     float* __restrict__ y, int n_in, int n_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_out) return;
  const int nb = n_in / 32;
  const unsigned char* rp = W + (long long)row * nb * 22;
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  int b = 0;
  // four independent accumulator chains keep more loads in flight
  for (; b + 3 < nb; b += 4) {
    float d[4];
    int q[4];
    for (int u = 0; u < 4; ++u) {
      const unsigned char* p = rp + (b + u) * 22;
      d[u] = half_to_float(*(const unsigned short*)(p));
      const unsigned int qh = (unsigned int)p[2] | ((unsigned int)p[3] << 8) |
                              ((unsigned int)p[4] << 16) | ((unsigned int)p[5] << 24);
      const unsigned char* qs = p + 6;
      const int low = (lane < 16) ? (qs[lane] & 0xF) : (qs[lane - 16] >> 4);
      const int hi = (qh >> lane) & 1u;
      q[u] = (low | (hi << 4)) - 16;
    }
    acc0 += d[0] * (float)q[0] * x[(b + 0) * 32 + lane];
    acc1 += d[1] * (float)q[1] * x[(b + 1) * 32 + lane];
    acc2 += d[2] * (float)q[2] * x[(b + 2) * 32 + lane];
    acc3 += d[3] * (float)q[3] * x[(b + 3) * 32 + lane];
  }
  float acc = acc0 + acc1 + acc2 + acc3;
  for (; b < nb; ++b) {
    const unsigned char* p = rp + b * 22;
    const float d = half_to_float(*(const unsigned short*)(p));
    const unsigned int qh = (unsigned int)p[2] | ((unsigned int)p[3] << 8) |
                            ((unsigned int)p[4] << 16) | ((unsigned int)p[5] << 24);
    const unsigned char* qs = p + 6;
    const int low = (lane < 16) ? (qs[lane] & 0xF) : (qs[lane - 16] >> 4);
    const int hi = (qh >> lane) & 1u;
    const int q = (low | (hi << 4)) - 16;
    acc += d * (float)q * x[b * 32 + lane];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) y[row] = acc;
}
)cuda";


// ---- GEMV, 4 rows per warp: x loads are shared across the 4 rows -------------
// grid = ceil(n_out/32), block = 256 (8 warps x 4 rows)
inline const char* kGemvQ4K_x4 = R"cuda(
extern "C" __global__ void gemv_q4k_x4(const unsigned char* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ y, int n_in, int n_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * ((blockDim.x >> 5) * 4) + warp * 4;
  const int sb_count = n_in / 256;
  const int k0 = lane * 8;
  const int g = k0 >> 6, rr = k0 & 63, half = rr >> 5, l0 = rr & 31;
  const int sub = g * 2 + half;
  const size_t row_bytes = (size_t)sb_count * 144;
  const unsigned char* ptr[4];
  bool valid[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    valid[r] = (row0 + r) < n_out;
    const int row = valid[r] ? (row0 + r) : (n_out - 1);
    ptr[r] = W + (long long)row * row_bytes;
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int sb = 0; sb < sb_count; ++sb) {
    const float* xp = x + sb * 256 + k0;
    const float4 xa = *reinterpret_cast<const float4*>(xp);
    const float4 xb = *reinterpret_cast<const float4*>(xp + 4);
    const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    const int qoff = 16 + g * 32 + l0;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const unsigned char* p = ptr[r] + sb * 144;
      const float d = half_to_float(*(const unsigned short*)(p));
      const float dmin = half_to_float(*(const unsigned short*)(p + 2));
      int sc = 0, mn = 0;
      q4k_scale_min(sub, p + 4, &sc, &mn);
      const float scale = d * (float)sc, bias = dmin * (float)mn;
      const unsigned long long bits =
          *reinterpret_cast<const unsigned long long*>(p + qoff);
      float lp = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int byte = (int)((bits >> (8 * i)) & 0xFFull);
        const int nib = half ? (byte >> 4) : (byte & 0xF);
        lp += (scale * (float)nib - bias) * xs[i];
      }
      acc[r] += lp;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
    if (lane == 0 && valid[r]) y[row0 + r] = acc[r];
  }
}
)cuda";

inline const char* kGemvQ5K_x4 = R"cuda(
extern "C" __global__ void gemv_q5k_x4(const unsigned char* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ y, int n_in, int n_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * ((blockDim.x >> 5) * 4) + warp * 4;
  const int sb_count = n_in / 256;
  const int k0 = lane * 8;
  const int g = k0 >> 6, rr = k0 & 63, half = rr >> 5, l0 = rr & 31;
  const int sub = g * 2 + half;
  const size_t row_bytes = (size_t)sb_count * 176;
  const unsigned char* ptr[4];
  bool valid[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    valid[r] = (row0 + r) < n_out;
    const int row = valid[r] ? (row0 + r) : (n_out - 1);
    ptr[r] = W + (long long)row * row_bytes;
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int sb = 0; sb < sb_count; ++sb) {
    const float* xp = x + sb * 256 + k0;
    const float4 xa = *reinterpret_cast<const float4*>(xp);
    const float4 xb = *reinterpret_cast<const float4*>(xp + 4);
    const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    const int qoff = 48 + g * 32 + l0;
    const int qhoff = 16 + l0;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const unsigned char* p = ptr[r] + sb * 176;
      const float d = half_to_float(*(const unsigned short*)(p));
      const float dmin = half_to_float(*(const unsigned short*)(p + 2));
      int sc = 0, mn = 0;
      q4k_scale_min(sub, p + 4, &sc, &mn);
      const float scale = d * (float)sc, bias = dmin * (float)mn;
      const unsigned long long bits = *reinterpret_cast<const unsigned long long*>(p + qoff);
      const unsigned long long hbits = *reinterpret_cast<const unsigned long long*>(p + qhoff);
      float lp = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int byte = (int)((bits >> (8 * i)) & 0xFFull);
        const int nib = half ? (byte >> 4) : (byte & 0xF);
        const int hb = (int)((hbits >> (8 * i)) & 0xFFull);
        const int q = nib | (((hb >> sub) & 1) << 4);
        lp += (scale * (float)q - bias) * xs[i];
      }
      acc[r] += lp;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
    if (lane == 0 && valid[r]) y[row0 + r] = acc[r];
  }
}
)cuda";

inline const char* kGemvQ6K_x4 = R"cuda(
extern "C" __global__ void gemv_q6k_x4(const unsigned char* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ y, int n_in, int n_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * ((blockDim.x >> 5) * 4) + warp * 4;
  const int sb_count = n_in / 256;
  const int is = lane / 16;
  const size_t row_bytes = (size_t)sb_count * 210;
  const unsigned char* ptr[4];
  bool valid[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    valid[r] = (row0 + r) < n_out;
    const int row = valid[r] ? (row0 + r) : (n_out - 1);
    ptr[r] = W + (long long)row * row_bytes;
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int sb = 0; sb < sb_count; ++sb) {
    const float* xp = x + sb * 256;
#pragma unroll
    for (int n2 = 0; n2 < 2; ++n2) {
      const int base = n2 * 128;
      const float xv[4] = {xp[base + lane], xp[base + lane + 32], xp[base + lane + 64],
                           xp[base + lane + 96]};
      const int ql0 = n2 * 64, qh0 = n2 * 32;
      const int sc0 = n2 * 8 + is;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const unsigned char* p = ptr[r] + sb * 210;
        const float d = half_to_float(*(const unsigned short*)(p + 208));
        const unsigned char* ql = p + ql0;
        const unsigned char* qh = p + 128 + qh0;
        const signed char* sc = (const signed char*)(p + 192);
        const int q1 = (int)((ql[lane] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
        const int q2 = (int)((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
        const int q3 = (int)((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
        const int q4 = (int)((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
        acc[r] += d * (float)sc[sc0 + 0] * (float)q1 * xv[0];
        acc[r] += d * (float)sc[sc0 + 2] * (float)q2 * xv[1];
        acc[r] += d * (float)sc[sc0 + 4] * (float)q3 * xv[2];
        acc[r] += d * (float)sc[sc0 + 6] * (float)q4 * xv[3];
      }
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
    if (lane == 0 && valid[r]) y[row0 + r] = acc[r];
  }
}
)cuda";

inline const char* kGemvQ50_x4 = R"cuda(
extern "C" __global__ void gemv_q5_0_x4(const unsigned char* __restrict__ W,
                                        const float* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * ((blockDim.x >> 5) * 4) + warp * 4;
  const int nb = n_in / 32;
  const size_t row_bytes = (size_t)nb * 22;
  const unsigned char* ptr[4];
  bool valid[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    valid[r] = (row0 + r) < n_out;
    const int row = valid[r] ? (row0 + r) : (n_out - 1);
    ptr[r] = W + (long long)row * row_bytes;
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int b = 0; b < nb; ++b) {
    const float xv = x[b * 32 + lane];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const unsigned char* p = ptr[r] + b * 22;
      const float d = half_to_float(*(const unsigned short*)(p));
      const unsigned int qh = (unsigned int)p[2] | ((unsigned int)p[3] << 8) |
                              ((unsigned int)p[4] << 16) | ((unsigned int)p[5] << 24);
      const unsigned char* qs = p + 6;
      const int low = (lane < 16) ? (qs[lane] & 0xF) : (qs[lane - 16] >> 4);
      const int hi = (qh >> lane) & 1u;
      const int q = (low | (hi << 4)) - 16;
      acc[r] += d * (float)q * xv;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
    if (lane == 0 && valid[r]) y[row0 + r] = acc[r];
  }
}
)cuda";

// Split-K GEMV for short rows: two warps cooperate on one row (each does half of k),
// partials reduced through shared memory. grid = ceil(n_out/4), block = 256.
inline const char* kGemvQ50K2 = R"cuda(
extern "C" __global__ void gemv_q5_0_k2(const unsigned char* __restrict__ W,
                                        const float* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
  __shared__ float red[8];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int half = warp & 1, pair = warp >> 1;
  const int row = blockIdx.x * 4 + pair;
  const bool valid = row < n_out;
  const int nb = n_in / 32;
  const int lo = half ? nb / 2 : 0;
  const int hi = half ? nb : nb / 2;
  float acc = 0.f;
  if (valid) {
    const size_t row_bytes = (size_t)nb * 22;
    const unsigned char* rp = W + (long long)row * row_bytes;
    for (int b = lo; b < hi; ++b) {
      const unsigned char* p = rp + b * 22;
      const float d = half_to_float(*(const unsigned short*)(p));
      const unsigned int qh = (unsigned int)p[2] | ((unsigned int)p[3] << 8) |
                              ((unsigned int)p[4] << 16) | ((unsigned int)p[5] << 24);
      const unsigned char* qs = p + 6;
      const int low = (lane < 16) ? (qs[lane] & 0xF) : (qs[lane - 16] >> 4);
      const int hb = (qh >> lane) & 1u;
      const int q = (low | (hb << 4)) - 16;
      acc += d * (float)q * x[b * 32 + lane];
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (valid && half == 0 && lane == 0) y[row] = red[warp] + red[warp + 1];
}
)cuda";

inline const char* kGemvQ4KK2 = R"cuda(
extern "C" __global__ void gemv_q4k_k2(const unsigned char* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ y, int n_in, int n_out) {
  __shared__ float red[8];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int half = warp & 1, pair = warp >> 1;
  const int row = blockIdx.x * 4 + pair;
  const bool valid = row < n_out;
  const int sb_count = n_in / 256;
  const int lo = half ? sb_count / 2 : 0;
  const int hi = half ? sb_count : sb_count / 2;
  const int k0 = lane * 8;
  const int g = k0 >> 6, rr = k0 & 63, hf = rr >> 5, l0 = rr & 31;
  const int sub = g * 2 + hf;
  float acc = 0.f;
  if (valid) {
    const size_t row_bytes = (size_t)sb_count * 144;
    const unsigned char* rp = W + (long long)row * row_bytes;
    for (int sb = lo; sb < hi; ++sb) {
      const unsigned char* p = rp + sb * 144;
      const float d = half_to_float(*(const unsigned short*)(p));
      const float dmin = half_to_float(*(const unsigned short*)(p + 2));
      int sc = 0, mn = 0;
      q4k_scale_min(sub, p + 4, &sc, &mn);
      const float scale = d * (float)sc, bias = dmin * (float)mn;
      const unsigned char* q = p + 16 + g * 32 + l0;
      const float* xp = x + sb * 256 + k0;
      float lp = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int byte = q[i];
        const int nib = hf ? (byte >> 4) : (byte & 0xF);
        lp += (scale * (float)nib - bias) * xp[i];
      }
      acc += lp;
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (valid && half == 0 && lane == 0) y[row] = red[warp] + red[warp + 1];
}
)cuda";

// Q5_0 GEMV with 4-deep software pipelining (independent loads -> more MLP).
// One warp per row, 8 rows per block.
inline const char* kGemvQ50P4 = R"cuda(
extern "C" __global__ void gemv_q5_0_p4(const unsigned char* __restrict__ W,
                                        const float* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_out) return;
  const int nb = n_in / 32;
  const size_t row_bytes = (size_t)nb * 22;
  const unsigned char* rp = W + (long long)row * row_bytes;
  float acc = 0.f;
  int b = 0;
  for (; b + 4 <= nb; b += 4) {
    const unsigned char* p0 = rp + (b + 0) * 22;
    const unsigned char* p1 = rp + (b + 1) * 22;
    const unsigned char* p2 = rp + (b + 2) * 22;
    const unsigned char* p3 = rp + (b + 3) * 22;
    const float d0 = half_to_float(*(const unsigned short*)(p0));
    const float d1 = half_to_float(*(const unsigned short*)(p1));
    const float d2 = half_to_float(*(const unsigned short*)(p2));
    const float d3 = half_to_float(*(const unsigned short*)(p3));
    const unsigned int h0 = (unsigned int)p0[2] | ((unsigned int)p0[3] << 8) |
                            ((unsigned int)p0[4] << 16) | ((unsigned int)p0[5] << 24);
    const unsigned int h1 = (unsigned int)p1[2] | ((unsigned int)p1[3] << 8) |
                            ((unsigned int)p1[4] << 16) | ((unsigned int)p1[5] << 24);
    const unsigned int h2 = (unsigned int)p2[2] | ((unsigned int)p2[3] << 8) |
                            ((unsigned int)p2[4] << 16) | ((unsigned int)p2[5] << 24);
    const unsigned int h3 = (unsigned int)p3[2] | ((unsigned int)p3[3] << 8) |
                            ((unsigned int)p3[4] << 16) | ((unsigned int)p3[5] << 24);
    const int lo0 = (lane < 16) ? (p0[6 + lane] & 0xF) : (p0[6 + lane - 16] >> 4);
    const int lo1 = (lane < 16) ? (p1[6 + lane] & 0xF) : (p1[6 + lane - 16] >> 4);
    const int lo2 = (lane < 16) ? (p2[6 + lane] & 0xF) : (p2[6 + lane - 16] >> 4);
    const int lo3 = (lane < 16) ? (p3[6 + lane] & 0xF) : (p3[6 + lane - 16] >> 4);
    const int q0 = (lo0 | (((h0 >> lane) & 1u) << 4)) - 16;
    const int q1 = (lo1 | (((h1 >> lane) & 1u) << 4)) - 16;
    const int q2 = (lo2 | (((h2 >> lane) & 1u) << 4)) - 16;
    const int q3 = (lo3 | (((h3 >> lane) & 1u) << 4)) - 16;
    acc += d0 * (float)q0 * x[(b + 0) * 32 + lane];
    acc += d1 * (float)q1 * x[(b + 1) * 32 + lane];
    acc += d2 * (float)q2 * x[(b + 2) * 32 + lane];
    acc += d3 * (float)q3 * x[(b + 3) * 32 + lane];
  }
  for (; b < nb; ++b) {
    const unsigned char* p = rp + b * 22;
    const float d = half_to_float(*(const unsigned short*)(p));
    const unsigned int qh = (unsigned int)p[2] | ((unsigned int)p[3] << 8) |
                            ((unsigned int)p[4] << 16) | ((unsigned int)p[5] << 24);
    const int low = (lane < 16) ? (p[6 + lane] & 0xF) : (p[6 + lane - 16] >> 4);
    const int q = (low | (((qh >> lane) & 1u) << 4)) - 16;
    acc += d * (float)q * x[b * 32 + lane];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) y[row] = acc;
}
)cuda";

// MLA decode: cache stores expanded k (per head, qk) and v (per head, v_dim).
inline const char* kMlaKvWrite = R"cuda(
extern "C" __global__ void mla_kv_write(const float* __restrict__ k, const float* __restrict__ v,
                                        float* __restrict__ Kc, float* __restrict__ Vc,
                                        const int* __restrict__ pstart, int n_head, int qk,
                                        int vdim, int max_seq) {
  const int idx = blockIdx.x;
  const int j = idx / n_head, h = idx % n_head;
  const int pos = *pstart + j;
  const long long ksrc = ((long long)j * n_head + h) * qk;
  const long long kdst = ((long long)h * max_seq + pos) * qk;
  for (int d = threadIdx.x; d < qk; d += blockDim.x) Kc[kdst + d] = k[ksrc + d];
  const long long vsrc = ((long long)j * n_head + h) * vdim;
  const long long vdst = ((long long)h * max_seq + pos) * vdim;
  for (int d = threadIdx.x; d < vdim; d += blockDim.x) Vc[vdst + d] = v[vsrc + d];
}
)cuda";

inline const char* kMlaAttnCached = R"cuda(
extern "C" __global__ void mla_attn_cached(const float* __restrict__ q,
                                           const float* __restrict__ Kc,
                                           const float* __restrict__ Vc,
                                           float* __restrict__ out,
                                           const int* __restrict__ pstart, int n_head, int qk,
                                           int vdim, int max_seq) {
  extern __shared__ float sh[];
  const unsigned mask = 0xffffffffu;
  const int i = blockIdx.x, h = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int qpos = *pstart + i;
  const float scale = rsqrtf((float)qk);
  const float* qh = q + ((long long)i * n_head + h) * qk;
  const float* Kh = Kc + (long long)h * max_seq * qk;
  const float* Vh = Vc + (long long)h * max_seq * vdim;
  for (int j = lane; j <= qpos; j += 32) {
    const float* kh = Kh + (long long)j * qk;
    float dot = 0.f;
    for (int d = 0; d < qk; ++d) dot += qh[d] * kh[d];
    sh[j] = dot * scale;
  }
  __syncwarp();
  float lmax = -1e30f;
  for (int j = lane; j <= qpos; j += 32) lmax = fmaxf(lmax, sh[j]);
  for (int off = 16; off > 0; off >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(mask, lmax, off));
  float lsum = 0.f;
  for (int j = lane; j <= qpos; j += 32) {
    const float e = expf(sh[j] - lmax);
    sh[j] = e;
    lsum += e;
  }
  for (int off = 16; off > 0; off >>= 1) lsum += __shfl_xor_sync(mask, lsum, off);
  __syncwarp();
  const float inv = lsum > 0.f ? 1.f / lsum : 0.f;   // all -inf row -> zeros, not NaN
  for (int d = lane; d < vdim; d += 32) {
    float a = 0.f;
    for (int j = 0; j <= qpos; ++j) a += sh[j] * inv * Vh[(long long)j * vdim + d];
    out[((long long)i * n_head + h) * vdim + d] = a;
  }
}
)cuda";

// MLA layout helpers (GGUF may store tensors per-head or globally split).
inline const char* kMlaTmpLayout = R"cuda(
// q: [T, n_head*qk] global-split ([all nope][all rope]) -> per-head [nope|rope]
extern "C" __global__ void mla_q_reorder(const float* __restrict__ src, float* __restrict__ dst,
                                         int n_head, int qk_nope, int qk_rope, int pe_first) {
  const int t = blockIdx.x;
  const int qk = qk_nope + qk_rope;
  for (int h = 0; h < n_head; ++h) {
    // source layout: [all q_nope | all q_rope] (pe_first=0) or [all q_rope | all q_nope]
    const float* sn = src + (long long)t * n_head * qk +
                      (pe_first ? n_head * qk_rope : 0) + h * qk_nope;
    const float* sr = src + (long long)t * n_head * qk +
                      (pe_first ? 0 : n_head * qk_nope) + h * qk_rope;
    float* d = dst + ((long long)t * n_head + h) * qk;
    for (int i = threadIdx.x; i < qk_nope; i += blockDim.x) d[i] = sn[i];
    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x) d[qk_nope + i] = sr[i];
  }
}
// kv_b: [T, n_head*(qk_nope+v)] per-head [nope|v] -> separate k_nope / v buffers
extern "C" __global__ void mla_kv_split(const float* __restrict__ src, float* __restrict__ kn,
                                        float* __restrict__ vv, int n_head, int qk_nope,
                                        int vdim, int vfirst) {
  const int t = blockIdx.x;
  const int row = qk_nope + vdim;
  const int ko = vfirst ? vdim : 0;
  const int vo = vfirst ? 0 : qk_nope;
  for (int h = 0; h < n_head; ++h) {
    const float* s = src + ((long long)t * n_head + h) * row;
    float* dk = kn + ((long long)t * n_head + h) * qk_nope;
    float* dv = vv + ((long long)t * n_head + h) * vdim;
    for (int i = threadIdx.x; i < qk_nope; i += blockDim.x) dk[i] = s[ko + i];
    for (int i = threadIdx.x; i < vdim; i += blockDim.x) dv[i] = s[vo + i];
  }
}
)cuda";


// ---- qwen35 gated delta net (linear attention) ------------------------------
// depthwise causal conv over the token axis + silu.  state layout: [K-1][C]
inline const char* kGdConv1d = R"cuda(
extern "C" __global__ void gd_conv1d_silu(const float* __restrict__ x,
                                          const float* __restrict__ w,
                                          float* __restrict__ state,
                                          float* __restrict__ y, int T, int C, int K) {
  const int t = blockIdx.x;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float acc = 0.f;
    const float* wc = w + c * K;
    for (int k = 0; k < K; ++k) {
      const int j = t - (K - 1) + k;
      float xv;
      if (j >= 0) xv = x[j * C + c];
      else xv = state[(-j - 1) * C + c];
      acc += wc[k] * xv;
    }
    y[t * C + c] = acc / (1.f + expf(-acc));
  }
}
)cuda";

// keep the last K-1 input rows; must run after gd_conv1d_silu for the chunk.
// The K=4 tap shift below is the only implementation: q35_geometry::validate_ssm()
// rejects conv_kernel != 4 at load time, so models with another K never reach here.
inline const char* kGdConvState = R"cuda(
extern "C" __global__ void gd_conv_state(const float* __restrict__ x,
                                         float* __restrict__ state, int T, int C, int K) {
  // Shift the conv history: state[a][c] = input at (T-1-a); entries beyond the
  // chunk (a >= T) fall back to the previous state shifted by T. K is always 4
  // for qwen35, so the taps are unrolled to keep the read/write order explicit.
  if (K != 4) return;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < C;
       c += blockDim.x * gridDim.x) {
    const float s0 = state[c];
    const float s1 = state[C + c];
    const float n0 = x[(T - 1) * C + c];
    const float n1 = (T >= 2) ? x[(T - 2) * C + c] : s0;
    const float n2 = (T >= 3) ? x[(T - 3) * C + c] : ((T >= 2) ? s0 : s1);
    state[c] = n0;
    state[C + c] = n1;
    state[2 * C + c] = n2;
  }
}
)cuda";

// q,k: [T, n_head_k, D]  -> l2-normalised; q additionally scaled by `scale`
inline const char* kGdNormQk = R"cuda(
extern "C" __global__ void gd_norm_qk(float* __restrict__ q, float* __restrict__ k, int rows,
                                      int D, float scale) {
  const int r = blockIdx.x;
  __shared__ float sh[2];
  if (threadIdx.x < 2) sh[threadIdx.x] = 0.f;
  __syncthreads();
  float sq = 0.f, sk = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float a = q[r * D + i], b = k[r * D + i];
    sq += a * a;
    sk += b * b;
  }
  for (int off = 16; off > 0; off >>= 1) {
    sq += __shfl_xor_sync(0xffffffffu, sq, off);
    sk += __shfl_xor_sync(0xffffffffu, sk, off);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&sh[0], sq);
    atomicAdd(&sh[1], sk);
  }
  __syncthreads();
  const float iq = rsqrtf(sh[0] + 1e-6f) * scale;
  const float ik = rsqrtf(sh[1] + 1e-6f);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    q[r * D + i] *= iq;
    k[r * D + i] *= ik;
  }
}
)cuda";

// gated delta rule: grid = n_vh, block = VD threads (one per value dim).
// S: [n_vh][VD][KD] row-major (row j = value dim j, KD contiguous)
inline const char* kGdDelta = R"cuda(
extern "C" __global__ void gd_delta_rule(const float* __restrict__ q,
                                         const float* __restrict__ k,
                                         const float* __restrict__ v,
                                         const float* __restrict__ g,
                                         const float* __restrict__ beta,
                                         float* __restrict__ S, float* __restrict__ o, int T,
                                         int n_vh, int n_kh, int KD, int VD, int kh_mode) {
  extern __shared__ float sh[];
  float* sk = sh;            // [KD] key of the current step
  float* sq = sh + KD;       // [KD] query of the current step
  const int h = blockIdx.x;
  const int j = threadIdx.x; // value dim
  const int rep = n_vh / n_kh;
  // q/k head -> value head broadcast.  GGUF checkpoints are written with the
  // llama.cpp convention (v_head h reads k_head h % n_kh, "block repeat"; see
  // ggml/src/ggml-cuda/gated_delta_net.cu `iq1 = fastmodulo(h_idx, neqk1)` and
  // [TAG_GGML_GDN_BCAST] in ggml.h).  HF safetensors checkpoints use
  // repeat_interleave (v_head h reads k_head h / rep).  kh_mode 0 = HF, 1 = GGUF.
  const int kh = kh_mode ? (h % n_kh) : (h / rep);
  float* Sj = S + ((long long)h * VD + j) * KD;
  for (int t = 0; t < T; ++t) {
    if (j < KD) {
      sk[j] = k[((long long)t * n_kh + kh) * KD + j];
      sq[j] = q[((long long)t * n_kh + kh) * KD + j];
    }
    __syncthreads();
    const float decay = expf(g[t * n_vh + h]);
    float kv = 0.f;
    for (int i = 0; i < KD; ++i) {
      Sj[i] *= decay;
      kv += Sj[i] * sk[i];
    }
    const float delta =
        (v[((long long)t * n_vh + h) * VD + j] - kv) * beta[t * n_vh + h];
    float acc = 0.f;
    for (int i = 0; i < KD; ++i) {
      Sj[i] += sk[i] * delta;
      acc += Sj[i] * sq[i];
    }
    o[((long long)t * n_vh + h) * VD + j] = acc;
    __syncthreads();
  }
}
)cuda";

// Decode-optimised variant, specialised for key_head_dim = value_head_dim = 128
// (every qwen35 stack on this machine: 9B / Qwythos / 35B-A3B all use 128/128).
//
// Why: the generic kernel above indexes S as [n_vh][VD][KD] with one thread per
// value dim, so at every inner step a warp touches 32 *different* 128-byte lines
// (uncoalesced) and the 2 MB state is streamed through L1 eight times.  Measured
// 0.13 ms/call in isolation, 24 calls/token on the 9B.
//
// This version uses the qwen35 reference layout [n_vh][KD][VD] (value dim
// contiguous) and tiles the key dim 4-way inside a 512-thread block: each thread
// keeps its 32 state values in registers, so one launch reads the 2 MB state once
// and writes it once (4 MB traffic).  Measured 0.0125 ms/call, rel_l2 7e-9 vs the
// generic kernel.
inline const char* kGdDelta128 = R"cuda(
extern "C" __global__ void gd_delta_rule_128(const float* __restrict__ q,
                                             const float* __restrict__ k,
                                             const float* __restrict__ v,
                                             const float* __restrict__ g,
                                             const float* __restrict__ beta,
                                             float* __restrict__ S, float* __restrict__ o, int T,
                                             int n_vh, int n_kh, int KD, int VD, int kh_mode) {
  constexpr int D = 128;
  constexpr int GRP = 4;
  constexpr int KPER = D / GRP;
  __shared__ float sk[D];
  __shared__ float sq[D];
  __shared__ float part[GRP][D];
  __shared__ float sdelta[D];
  const int h = blockIdx.x;
  const int grp = threadIdx.x / D;
  const int j = threadIdx.x % D;
  const int rep = n_vh / n_kh;
  // q/k head -> value head broadcast.  GGUF checkpoints are written with the
  // llama.cpp convention (v_head h reads k_head h % n_kh, "block repeat"; see
  // ggml/src/ggml-cuda/gated_delta_net.cu `iq1 = fastmodulo(h_idx, neqk1)` and
  // [TAG_GGML_GDN_BCAST] in ggml.h).  HF safetensors checkpoints use
  // repeat_interleave (v_head h reads k_head h / rep).  kh_mode 0 = HF, 1 = GGUF.
  const int kh = kh_mode ? (h % n_kh) : (h / rep);
  float* Sh = S + (long long)h * D * D;
  float reg[KPER];
  for (int t = 0; t < T; ++t) {
    if (grp == 0) {
      sk[j] = k[((long long)t * n_kh + kh) * D + j];
      sq[j] = q[((long long)t * n_kh + kh) * D + j];
    }
    __syncthreads();
    const float decay = expf(g[t * n_vh + h]);
    const float bh = beta[t * n_vh + h];
    float kv = 0.f;
    for (int i = 0; i < KPER; ++i) {
      const float s = Sh[(grp * KPER + i) * D + j] * decay;
      reg[i] = s;
      kv += s * sk[grp * KPER + i];
    }
    part[grp][j] = kv;
    __syncthreads();
    if (grp == 0) {
      float tot = part[0][j] + part[1][j] + part[2][j] + part[3][j];
      sdelta[j] = (v[((long long)t * n_vh + h) * D + j] - tot) * bh;
    }
    __syncthreads();
    const float delta = sdelta[j];
    float acc = 0.f;
    for (int i = 0; i < KPER; ++i) {
      const int ii = grp * KPER + i;
      const float s = reg[i] + sk[ii] * delta;
      Sh[ii * D + j] = s;
      acc += s * sq[ii];
    }
    part[grp][j] = acc;
    __syncthreads();
    if (grp == 0) {
      o[((long long)t * n_vh + h) * D + j] =
          part[0][j] + part[1][j] + part[2][j] + part[3][j];
    }
    __syncthreads();
  }
}
)cuda";

// Fused decode preparation for a gated-delta-net layer (T == 1 only).
//
// Replaces four launches (gd_conv1d_silu + gd_conv_state + q35_split3 +
// q35_beta_g) with one.  The conv output is written straight into q/k/v (the
// conv dimension is q | k | v), the conv history is shifted in the same kernel
// (each block owns a disjoint column range, so the read-then-write of `state` is
// race free), and the block past the last column range computes beta/g.
// 0.69+0.15+0.26+0.41 ms/token of kprof time -> one launch.
inline const char* kGdSsmPrep = R"cuda(
extern "C" __global__ void gd_ssm_prep(const float* __restrict__ x,
                                       const float* __restrict__ w,
                                       float* __restrict__ state,
                                       float* __restrict__ q,
                                       float* __restrict__ k,
                                       float* __restrict__ v,
                                       int T, int C, int K, int key_dim, int value_dim,
                                       const float* __restrict__ b,
                                       const float* __restrict__ a,
                                       const float* __restrict__ a_log,
                                       const float* __restrict__ dt_bias,
                                       float* __restrict__ beta,
                                       float* __restrict__ g, int n_vh, int raw_a) {
  const int nb = (C + blockDim.x - 1) / blockDim.x;
  if (blockIdx.x >= nb) {
    // beta = sigmoid(b); g = -exp(A) * softplus(a + dt_bias)  (independent data)
    const int total = T * n_vh;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
      const int h = i % n_vh;
      beta[i] = 1.f / (1.f + expf(-b[i]));
      const float xx = a[i] + dt_bias[h];
      const float sp = xx > 20.f ? xx : log1pf(expf(xx));
      g[i] = raw_a ? a_log[h] * sp : -expf(a_log[h]) * sp;
    }
    return;
  }
  if (K > 9) return;   // host gate uses the fused path only for K <= 8 (qwen35: K=4)
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float hist[8];
  const int nh_hist = K - 1;
  for (int i = 0; i < nh_hist && i < 8; ++i) hist[i] = state[i * C + c];
  float acc = 0.f;
  for (int kk = 0; kk < K; ++kk) {
    const int j = kk - (K - 1);
    const float xv = (j >= 0) ? x[j * C + c] : hist[-j - 1];
    acc += w[c * K + kk] * xv;
  }
  acc = acc / (1.f + expf(-acc));
  if (c < key_dim) q[c] = acc;
  else if (c < 2 * key_dim) k[c - key_dim] = acc;
  else v[c - 2 * key_dim] = acc;
  // T == 1: state[a] = input at (T-1-a) -> state[0] = x[0], state[a] = old state[a-1]
  state[c] = x[c];
  for (int i = 1; i < nh_hist && i < 8; ++i) state[i * C + c] = hist[i - 1];
}
)cuda";

// Fused variant of gd_delta_rule_128: also does the q/k L2 norm (gd_norm_qk) in
// the prologue and the gated RMSNorm (gd_norm_gate) in the epilogue, so a linear
// attention layer needs one prep launch + one delta launch instead of five.
// q/k are consumed in raw form; the normalised copies are kept in shared memory.
inline const char* kGdDelta128F = R"cuda(
extern "C" __global__ void gd_delta_rule_128f(const float* __restrict__ qraw,
                                              const float* __restrict__ kraw,
                                              const float* __restrict__ v,
                                              const float* __restrict__ g,
                                              const float* __restrict__ beta,
                                              float* __restrict__ S, float* __restrict__ o,
                                              const float* __restrict__ z,
                                              const float* __restrict__ nw, float qscale,
                                              float eps, int T, int n_vh, int n_kh, int KD,
                                              int VD, int kh_mode) {
  constexpr int D = 128;
  constexpr int GRP = 4;
  constexpr int KPER = D / GRP;
  __shared__ float sk[D];
  __shared__ float sq[D];
  __shared__ float part[GRP][D];
  __shared__ float sdelta[D];
  __shared__ float red[16];
  const int h = blockIdx.x;
  const int grp = threadIdx.x / D;
  const int j = threadIdx.x % D;
  const int rep = n_vh / n_kh;
  // q/k head -> value head broadcast.  GGUF checkpoints are written with the
  // llama.cpp convention (v_head h reads k_head h % n_kh, "block repeat"; see
  // ggml/src/ggml-cuda/gated_delta_net.cu `iq1 = fastmodulo(h_idx, neqk1)` and
  // [TAG_GGML_GDN_BCAST] in ggml.h).  HF safetensors checkpoints use
  // repeat_interleave (v_head h reads k_head h / rep).  kh_mode 0 = HF, 1 = GGUF.
  const int kh = kh_mode ? (h % n_kh) : (h / rep);
  float* Sh = S + (long long)h * D * D;
  float reg[KPER];
  for (int t = 0; t < T; ++t) {
    // ---- prologue: l2-normalise q/k for this key head (4 warps of grp 0) ----
    if (grp == 0) {
      const float kk = kraw[((long long)t * n_kh + kh) * D + j];
      const float qq = qraw[((long long)t * n_kh + kh) * D + j];
      sk[j] = kk;
      sq[j] = qq;
      part[0][j] = kk * kk;
      part[1][j] = qq * qq;
    }
    __syncthreads();
    if (grp == 0) {
      float a = part[0][j], b = part[1][j];
      for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, off);
        b += __shfl_xor_sync(0xffffffffu, b, off);
      }
      if ((threadIdx.x & 31) == 0) {
        red[threadIdx.x >> 5] = a;
        red[4 + (threadIdx.x >> 5)] = b;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      red[8] = rsqrtf(red[0] + red[1] + red[2] + red[3] + 1e-6f);
      red[9] = rsqrtf(red[4] + red[5] + red[6] + red[7] + 1e-6f) * qscale;
    }
    __syncthreads();
    if (grp == 0) {
      sk[j] *= red[8];
      sq[j] *= red[9];
    }
    __syncthreads();
    const float decay = expf(g[t * n_vh + h]);
    const float bh = beta[t * n_vh + h];
    float kv = 0.f;
    for (int i = 0; i < KPER; ++i) {
      const float s = Sh[(grp * KPER + i) * D + j] * decay;
      reg[i] = s;
      kv += s * sk[grp * KPER + i];
    }
    part[grp][j] = kv;
    __syncthreads();
    if (grp == 0) {
      const float tot = part[0][j] + part[1][j] + part[2][j] + part[3][j];
      sdelta[j] = (v[((long long)t * n_vh + h) * D + j] - tot) * bh;
    }
    __syncthreads();
    const float delta = sdelta[j];
    float acc = 0.f;
    for (int i = 0; i < KPER; ++i) {
      const int ii = grp * KPER + i;
      const float s = reg[i] + sk[ii] * delta;
      Sh[ii * D + j] = s;
      acc += s * sq[ii];
    }
    part[grp][j] = acc;
    __syncthreads();
    // ---- epilogue: gated RMSNorm over the value head ----
    if (grp == 0) {
      const float o0 = part[0][j] + part[1][j] + part[2][j] + part[3][j];
      sdelta[j] = o0;
      float a = o0 * o0;
      for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
      if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = a;
    }
    __syncthreads();
    if (threadIdx.x == 0)
      red[10] = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)D + eps);
    __syncthreads();
    if (grp == 0) {
      const float zz = z[((long long)t * n_vh + h) * D + j];
      o[((long long)t * n_vh + h) * D + j] =
          sdelta[j] * red[10] * nw[j] * (zz / (1.f + expf(-zz)));
    }
    __syncthreads();
  }
}
)cuda";

// split the conv output into q / k / v (all [T, *] row-major)
inline const char* kQ35Split3 = R"cuda(
extern "C" __global__ void q35_split3(const float* __restrict__ src, float* __restrict__ q,
                                      float* __restrict__ k, float* __restrict__ v, int T,
                                      int key_dim, int value_dim, int conv_dim) {
  const int t = blockIdx.x;
  const float* s = src + (long long)t * conv_dim;
  for (int i = threadIdx.x; i < key_dim; i += blockDim.x) {
    q[(long long)t * key_dim + i] = s[i];
    k[(long long)t * key_dim + i] = s[key_dim + i];
  }
  for (int i = threadIdx.x; i < value_dim; i += blockDim.x) {
    v[(long long)t * value_dim + i] = s[2 * key_dim + i];
  }
}
)cuda";

// beta = sigmoid(b);  g = -exp(A[h]) * softplus(a + dt_bias[h])
inline const char* kQ35Beta = R"cuda(
extern "C" __global__ void q35_beta_g(const float* __restrict__ b, const float* __restrict__ a,
                                      const float* __restrict__ a_log,
                                      const float* __restrict__ dt_bias, float* __restrict__ beta,
                                      float* __restrict__ g, int total, int n_vh) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int h = i % n_vh;
  const float bv = b[i];
  beta[i] = 1.f / (1.f + expf(-bv));
  const float x = a[i] + dt_bias[h];
  const float sp = x > 20.f ? x : log1pf(expf(x));
  g[i] = -expf(a_log[h]) * sp;
}
)cuda";

// variant: ssm_a already holds -A (llama.cpp style) instead of log(A)
inline const char* kQ35BetaRaw = R"cuda(
extern "C" __global__ void q35_beta_g_raw(const float* __restrict__ b,
                                          const float* __restrict__ a,
                                          const float* __restrict__ a_log,
                                          const float* __restrict__ dt_bias,
                                          float* __restrict__ beta, float* __restrict__ g,
                                          int total, int n_vh) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int h = i % n_vh;
  beta[i] = 1.f / (1.f + expf(-b[i]));
  const float x = a[i] + dt_bias[h];
  const float sp = x > 20.f ? x : log1pf(expf(x));
  g[i] = a_log[h] * sp;   // a_log already carries the (negative) log-decay
}
)cuda";

// gated RMSNorm over the value head: o = rmsnorm(o) * w * silu(z)
inline const char* kGdNormGate = R"cuda(
extern "C" __global__ void gd_norm_gate(float* __restrict__ o, const float* __restrict__ z,
                                        const float* __restrict__ w, int rows, int D,
                                        float eps) {
  const int r = blockIdx.x;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float a = o[r * D + i];
    ss += a * a;
  }
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float tot_pos = 0.f;
    for (int b = 0; b < (blockDim.x + 31) / 32; ++b) tot_pos += red[b];
    red[0] = rsqrtf(tot_pos / (float)D + eps);
  }
  __syncthreads();
  const float inv = red[0];
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float zz = z[r * D + i];
    o[r * D + i] = o[r * D + i] * inv * w[i] * (zz / (1.f + expf(-zz)));
  }
}
)cuda";

// shared-expert gate: scale each token's shared-expert output by sigmoid(w . h)
inline const char* kMoeSharedGate = R"cuda(
extern "C" __global__ void moe_shared_gate(float* __restrict__ out,
                                           const float* __restrict__ h,
                                           const float* __restrict__ w, int n, int hidden) {
  const int t = blockIdx.x;
  float s = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) s += h[t * hidden + i] * w[i];
  for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float tot = 0.f;
    // only the warps that actually wrote a partial sum take part
    for (int b = 0; b < (blockDim.x + 31) / 32; ++b) tot += red[b];
    red[0] = 1.f / (1.f + expf(-tot));
  }
  __syncthreads();
  const float g = red[0];
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) out[t * hidden + i] *= g;
  (void)n;
}
)cuda";

// qwen35 full attention: q_proj emits [head][query | gate]; split them
inline const char* kQ35SplitGate = R"cuda(
extern "C" __global__ void q35_split_qgate(const float* __restrict__ qfull,
                                           float* __restrict__ q, float* __restrict__ gate,
                                           int T, int n_head, int D) {
  // grid = (token, head): splitting one token's q|gate used to serialise all heads in
  // a single block, which cost ~27 us on the 8 full-attention layers.
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const float* src = qfull + ((long long)t * n_head + h) * 2 * D;
  float* dq = q + ((long long)t * n_head + h) * D;
  float* dg = gate + ((long long)t * n_head + h) * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    dq[i] = src[i];
    dg[i] = src[D + i];
  }
}
)cuda";

// multiply attention output by sigmoid(gate)
inline const char* kQ35GateMul = R"cuda(
extern "C" __global__ void q35_gate_mul(float* __restrict__ a, const float* __restrict__ gate,
                                        long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] *= 1.f / (1.f + expf(-gate[i]));
}
)cuda";


// ---- int8 dot-product path (llama.cpp style): quantise x to Q8 per 32-elt block,
// then use __dp4a against the already-int8 weights. 4x fewer arithmetic ops than f32.
inline const char* kQuantXQ8 = R"cuda(
// n must be a multiple of 32: the caller passes nblocks = n/32 and every lane reads
// x[b*32 + lane].  This holds for Q8_0/Q5_0 weight formats by definition; the dp4a
// dispatch in runner.cpp also asserts it before launching (audit D8).
extern "C" __global__ void quantize_x_q8(const float* __restrict__ x, signed char* __restrict__ qx,
                                         float* __restrict__ dx, int nblocks) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  const float* xb = x + b * 32;
  float amax = 0.f;
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xb[i]));
  const float d = amax / 127.f;
  const float id = d > 0.f ? 1.f / d : 0.f;
  dx[b] = d;
  signed char* q = qx + b * 32;
  for (int i = 0; i < 32; ++i)
    q[i] = (signed char)lrintf(fminf(127.f, fmaxf(-127.f, xb[i] * id)));
}
)cuda";

// Q8_0 weights (34 B / 32 values: fp16 scale + 32 int8); one warp per row.
inline const char* kGemvQ80Dp4a = R"cuda(
extern "C" __global__ void gemv_q8_0_dp4a(const void* __restrict__ w,
                                          const signed char* __restrict__ qx,
                                          const float* __restrict__ dx, float* __restrict__ y,
                                          int n_in, int n_out) {
  extern __shared__ signed char shx[];   // quantised x for this block
  for (int i = threadIdx.x; i < n_in; i += blockDim.x) shx[i] = qx[i];
  __syncthreads();
  const int nb = n_in / 32;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n_out) return;
  const unsigned char* rowp = static_cast<const unsigned char*>(w) + (long long)row * nb * 34;
  float acc = 0.f;
  for (int b = lane; b < nb; b += 32) {
    const unsigned char* blk = rowp + b * 34;
    const float d = half_to_float(*reinterpret_cast<const unsigned short*>(blk));
    const signed char* qw = reinterpret_cast<const signed char*>(blk + 2);
    const signed char* qxs = shx + b * 32;
    int s = 0;
    for (int i = 0; i < 32; i += 4) {
      int a, c;
      memcpy(&a, qw + i, 4);
      memcpy(&c, qxs + i, 4);
      s = __dp4a(a, c, s);
    }
    acc += d * dx[b] * (float)s;
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if (lane == 0) y[row] = acc;
}
)cuda";


// Q5_0 weights (22 B / 32 values): fp16 d, 32-bit high bits, 16 packed nibbles.
inline const char* kGemvQ50Dp4a = R"cuda(
extern "C" __global__ void gemv_q5_0_dp4a(const void* __restrict__ w,
                                          const signed char* __restrict__ qx,
                                          const float* __restrict__ dx, float* __restrict__ y,
                                          int n_in, int n_out) {
  extern __shared__ signed char shx[];
  for (int i = threadIdx.x; i < n_in; i += blockDim.x) shx[i] = qx[i];
  __syncthreads();
  const int nb = n_in / 32;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n_out) return;
  const unsigned char* rowp = static_cast<const unsigned char*>(w) + (long long)row * nb * 22;
  float acc = 0.f;
  for (int b = lane; b < nb; b += 32) {
    const unsigned char* blk = rowp + b * 22;
    const float d = half_to_float(*reinterpret_cast<const unsigned short*>(blk));
    const unsigned qh = *reinterpret_cast<const unsigned*>(blk + 2);
    const unsigned char* qs = blk + 6;
    const signed char* qxs = shx + b * 32;
    int v[32];
    for (int i = 0; i < 16; ++i) {
      const unsigned x = qs[i];
      v[i] = (int)(x & 0xF) + (int)((qh >> i) & 1u) * 16 - 16;
      v[i + 16] = (int)(x >> 4) + (int)((qh >> (16 + i)) & 1u) * 16 - 16;
    }
    int s0 = 0;
    for (int i = 0; i < 32; i += 4) {
      int a, c;
      memcpy(&a, &v[i], 4);
      memcpy(&c, qxs + i, 4);
      s0 = __dp4a(a, c, s0);
    }
    acc += d * dx[b] * (float)s0;
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if (lane == 0) y[row] = acc;
}
)cuda";


// Multi-warp causal attention over the KV cache: all warps of the block split the
// key range (phase 1 scores, phase 3 output) so long contexts do not serialise on
// a single warp.  blockDim.x must be a multiple of 32.
inline const char* kAttnCachedW = R"cuda(
extern "C" __global__ void attn_cached_w(const float* __restrict__ q,
                                         const float* __restrict__ Kc,
                                         const float* __restrict__ Vc,
                                         float* __restrict__ out,
                                         const int* __restrict__ pstart, int n_head, int n_kv,
                                         int D, int max_seq) {
  extern __shared__ float sh[];
  __shared__ float red[32];
  const int i = blockIdx.x, h = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int nwarp = blockDim.x >> 5;
  const int qpos = *pstart + i;
  const int rep = n_head / n_kv, kvh = h / rep;
  const float scale = rsqrtf((float)D);
  const float* qh = q + ((long long)i * n_head + h) * D;
  const float* Kh = Kc + (long long)kvh * max_seq * D;
  const float* Vh = Vc + (long long)kvh * max_seq * D;
  const int msz = qpos + 1;
  float* accs = sh + (((msz + 31) & ~31));

  for (int j = warp * 32 + lane; j <= qpos; j += nwarp * 32) {
    const float* kh = Kh + (long long)j * D;
    float dot = 0.f;
    for (int d = 0; d < D; ++d) dot += qh[d] * kh[d];
    sh[j] = dot * scale;
  }
  __syncthreads();
  float lmax = -1e30f;
  for (int j = tid; j <= qpos; j += blockDim.x) lmax = fmaxf(lmax, sh[j]);
  for (int off = 16; off > 0; off >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, off));
  if (lane == 0) red[warp] = lmax;
  __syncthreads();
  lmax = red[0];
  for (int w = 1; w < nwarp; ++w) lmax = fmaxf(lmax, red[w]);
  __syncthreads();
  float lsum = 0.f;
  for (int j = tid; j <= qpos; j += blockDim.x) {
    const float e = expf(sh[j] - lmax);
    sh[j] = e;
    lsum += e;
  }
  for (int off = 16; off > 0; off >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
  if (lane == 0) red[warp] = lsum;
  __syncthreads();
  float tot = 0.f;
  for (int w = 0; w < nwarp; ++w) tot += red[w];
  const float inv = 1.f / tot;
  __syncthreads();
  float* acc = accs + warp * D;
  for (int d = lane; d < D; d += 32) {
    float a = 0.f;
    for (int j = warp; j <= qpos; j += nwarp) a += sh[j] * inv * Vh[(long long)j * D + d];
    acc[d] = a;
  }
  __syncthreads();
  for (int d = tid; d < D; d += blockDim.x) {
    float a = 0.f;
    for (int w = 0; w < nwarp; ++w) a += accs[w * D + d];
    out[((long long)i * n_head + h) * D + d] = a;
  }
}
)cuda";


// ---- block-major repack: dst[b][r] = src[r][b] ----------------------------
// Coalescing-friendly layout for decode GEMV: a warp handling 32 consecutive rows
// reads 32*bb contiguous bytes per block instead of striding across the matrix.
inline const char* kRepackBM = R"cuda(
extern "C" __global__ void repack_block_major(const unsigned char* __restrict__ src,
                                              unsigned char* __restrict__ dst, int n_out, int nb,
                                              int bb) {
  const long long total = (long long)n_out * nb * bb;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long long)blockDim.x * gridDim.x) {
    const int r = (int)(i / ((long long)nb * bb));
    const int rem = (int)(i % ((long long)nb * bb));
    const int b = rem / bb;
    const int o = rem % bb;
    dst[((long long)b * n_out + r) * bb + o] = src[i];
  }
}
)cuda";

// Q8_0 GEMV over the block-major layout: one lane per row (warp = 32 rows), so the
// warp reads 32*34 contiguous bytes per block. Keeps f32 activations (no int8
// quantisation of x) so the numerics match the row-major path.
inline const char* kGemvQ80BM = R"cuda(
extern "C" __global__ void gemv_q8_0_bm(const unsigned char* __restrict__ w,
                                        const float* __restrict__ x, float* __restrict__ y,
                                        int n_in, int n_out) {
  const int nb = n_in / 32;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.x * (blockDim.x >> 5) + warp) * 32 + lane;
  if (row >= n_out) return;
  float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const unsigned char* blk = w + ((long long)b * n_out + row) * 34;
    const float d = half_to_float(*reinterpret_cast<const unsigned short*>(blk));
    const signed char* qw = reinterpret_cast<const signed char*>(blk + 2);
    const float* xb = x + b * 32;
    float s = 0.f;
    for (int i = 0; i < 32; i += 4) {
      int q4;
      memcpy(&q4, qw + i, 4);
      const int c0 = (signed char)(q4 & 0xff), c1 = (signed char)((q4 >> 8) & 0xff);
      const int c2 = (signed char)((q4 >> 16) & 0xff), c3 = (signed char)((q4 >> 24) & 0xff);
      s += c0 * xb[i] + c1 * xb[i + 1] + c2 * xb[i + 2] + c3 * xb[i + 3];
    }
    acc += d * s;
  }
  y[row] = acc;
}
)cuda";


// Q5_0 GEMV over the block-major layout (22 B per 32 values).
inline const char* kGemvQ50BM = R"cuda(
extern "C" __global__ void gemv_q5_0_bm(const unsigned char* __restrict__ w,
                                        const float* __restrict__ x, float* __restrict__ y,
                                        int n_in, int n_out) {
  const int nb = n_in / 32;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.x * (blockDim.x >> 5) + warp) * 32 + lane;
  if (row >= n_out) return;
  float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const unsigned char* blk = w + ((long long)b * n_out + row) * 22;
    const float d = half_to_float(*reinterpret_cast<const unsigned short*>(blk));
    const unsigned qh = *reinterpret_cast<const unsigned*>(blk + 2);
    const unsigned char* qs = blk + 6;
    const float* xb = x + b * 32;
    float s = 0.f;
    for (int i = 0; i < 16; ++i) {
      const unsigned v = qs[i];
      const int lo = (int)(v & 0xF) + (int)((qh >> i) & 1u) * 16 - 16;
      const int hi = (int)(v >> 4) + (int)((qh >> (16 + i)) & 1u) * 16 - 16;
      s += lo * xb[i] + hi * xb[i + 16];
    }
    acc += d * s;
  }
  y[row] = acc;
}
)cuda";

// Q6_K GEMV over the block-major layout (210 B per 256 values).
inline const char* kGemvQ6KBM = R"cuda(
extern "C" __global__ void gemv_q6_k_bm(const unsigned char* __restrict__ w,
                                        const float* __restrict__ x, float* __restrict__ y,
                                        int n_in, int n_out) {
  const int nb = n_in / 256;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.x * (blockDim.x >> 5) + warp) * 32 + lane;
  if (row >= n_out) return;
  float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = w + ((long long)b * n_out + row) * 210;
    const float d = half_to_float(*reinterpret_cast<const unsigned short*>(p + 208));
    const unsigned char* ql = p;
    const unsigned char* qh = p + 128;
    const signed char* sc = reinterpret_cast<const signed char*>(p + 192);
    const float* xb = x + b * 256;
    float tot = 0.f;
    for (int n2 = 0; n2 < 2; ++n2) {
      const int qi = n2 * 64, hi = n2 * 32;
      const float* g = xb + n2 * 128;
      // the sub-block scale changes every 16 lanes, so keep it inside the loop
      const signed char* sc2 = sc + n2 * 8;
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int q1 = (int)((ql[qi + l] & 0xF) | (((qh[hi + l] >> 0) & 3) << 4)) - 32;
        const int q2 = (int)((ql[qi + l + 32] & 0xF) | (((qh[hi + l] >> 2) & 3) << 4)) - 32;
        const int q3 = (int)((ql[qi + l] >> 4) | (((qh[hi + l] >> 4) & 3) << 4)) - 32;
        const int q4 = (int)((ql[qi + l + 32] >> 4) | (((qh[hi + l] >> 6) & 3) << 4)) - 32;
        tot += (float)sc2[is + 0] * (float)q1 * g[l] +
               (float)sc2[is + 2] * (float)q2 * g[l + 32] +
               (float)sc2[is + 4] * (float)q3 * g[l + 64] +
               (float)sc2[is + 6] * (float)q4 * g[l + 96];
      }
    }
    acc += d * tot;
  }
  y[row] = acc;
}
)cuda";

inline const char* kF32ToHalf = R"cuda(
__device__ __forceinline__ unsigned short f32_to_f16_bits(float f) {
  unsigned int x;
  memcpy(&x, &f, 4);
  const unsigned int sign = (x >> 16) & 0x8000u;
  int e = (int)((x >> 23) & 0xff) - 127 + 15;
  const unsigned int mant = x & 0x7fffffu;
  if (e <= 0) return (unsigned short)sign;
  if (e >= 31) return (unsigned short)(sign | 0x7c00u);
  return (unsigned short)(sign | ((unsigned int)e << 10) | (mant >> 13));
}

extern "C" __global__ void f32_to_half(const float* __restrict__ in,
                                       unsigned short* __restrict__ out, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = f32_to_f16_bits(in[i]);
}
)cuda";

// Interleaved (GPT-NeoX) rotary: pairs (0,1), (2,3), ... Used by DeepSeek-V2 MLA,
// whose reference implementation views the rope dims as complex numbers.
inline const char* kRopeInterleaved = R"cuda(
extern "C" __global__ void rope_interleaved(float* __restrict__ x, const int* __restrict__ pstart,
                                            int n_heads, int head_stride, int rope_dim,
                                            int offset, const float* __restrict__ freqs,
                                            float mscale) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int half = rope_dim / 2;
  float* p = x + ((long long)row * n_heads + h) * head_stride + offset;
  const float pos = (float)(*pstart + row);
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const float ang = pos * freqs[i];
    const float c = cosf(ang) * mscale, s = sinf(ang) * mscale;
    const float x1 = p[2 * i], x2 = p[2 * i + 1];
    p[2 * i] = x1 * c - x2 * s;
    p[2 * i + 1] = x1 * s + x2 * c;
  }
}
)cuda";

// Transpose a row-major f32 matrix and store it as fp16 (for the TN GEMM layout).
inline const char* kTransposeF16 = R"cuda(
extern "C" __global__ void transpose_f16(const float* __restrict__ src,
                                         unsigned short* __restrict__ dst, int rows, int cols) {
  __shared__ float tile[32][33];
  const int x = blockIdx.x * 32 + threadIdx.x;   // column in src
  const int y = blockIdx.y * 32 + threadIdx.y;   // row in src
  for (int j = 0; j < 32; j += 8)
    if (x < cols && (y + j) < rows) tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * cols + x];
  __syncthreads();
  const int tx = blockIdx.y * 32 + threadIdx.x;  // row in dst (= column in src)
  const int ty = blockIdx.x * 32 + threadIdx.y;  // column in dst (= row in src)
  for (int j = 0; j < 32; j += 8)
    if (tx < rows && (ty + j) < cols) {
      const float v = tile[threadIdx.x][threadIdx.y + j];
      unsigned int u;
      memcpy(&u, &v, 4);
      const unsigned int sign = (u >> 16) & 0x8000u;
      int e = (int)((u >> 23) & 0xff) - 127 + 15;
      const unsigned int mant = u & 0x7fffffu;
      unsigned short h;
      if (e <= 0) h = (unsigned short)sign;
      else if (e >= 31) h = (unsigned short)(sign | 0x7c00u);
      else h = (unsigned short)(sign | ((unsigned int)e << 10) | (mant >> 13));
      dst[(long long)tx * cols + (ty + j)] = h;
    }
}
)cuda";

// Plain f32 GEMV for small token counts (cuBLAS Sgemm at m<=4 runs at ~1 GB/s).
// Four independent accumulators keep four 128B loads in flight per warp.
inline const char* kGemvF32 = R"cuda(
extern "C" __global__ void gemv_f32(const float* __restrict__ W, const float* __restrict__ x,
                                    float* __restrict__ y, int n_in, int n_out, int n_tok) {
  const int t = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_out) return;
  const float* xp = x + (long long)t * n_in;
  const float* rp = W + (long long)row * n_in;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  int i = lane;
  for (; i + 96 < n_in; i += 128) {
    a0 += __ldg(rp + i) * __ldg(xp + i);
    a1 += __ldg(rp + i + 32) * __ldg(xp + i + 32);
    a2 += __ldg(rp + i + 64) * __ldg(xp + i + 64);
    a3 += __ldg(rp + i + 96) * __ldg(xp + i + 96);
  }
  for (; i < n_in; i += 32) a0 += __ldg(rp + i) * __ldg(xp + i);
  float acc = (a0 + a1) + (a2 + a3);
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if (lane == 0) y[(long long)t * n_out + row] = acc;
}
)cuda";

inline const char* kMoeFused = R"cuda(
// ---------------------------------------------------------------------------
// Fused MoE expert application for decode (one token per expert slot).
// Two kernels replace the per-expert chain (gather + gate GEMV + up GEMV + silu +
// down GEMV + scatter): on WDDM every extra launch costs far more than the maths.
// ---------------------------------------------------------------------------
template <int T>
__device__ __forceinline__ float qrow_dot(const unsigned char* __restrict__ W, long long row,
                                          const float* __restrict__ x, int n_in, int lane) {
  if (T == 0) {
    const int nb = n_in / 256;
    const unsigned char* rp = W + row * nb * 144;
    const int k0 = lane * 8;
    const int g = k0 >> 6, r = k0 & 63, half = r >> 5, l0 = r & 31;
    const int sub = g * 2 + half;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
      const unsigned char* p = rp + b * 144;
      const float d = half_to_float(*(const unsigned short*)(p));
      const float dmin = half_to_float(*(const unsigned short*)(p + 2));
      int sc = 0, mn = 0;
      q4k_scale_min(sub, p + 4, &sc, &mn);
      const float scale = d * (float)sc, bias = dmin * (float)mn;
      const unsigned char* q = p + 16 + g * 32 + l0;
      const float* xp = x + b * 256 + k0;
      float lp = 0.f;
      for (int i = 0; i < 8; ++i) {
        const unsigned char qb = q[i];
        const int nib = half ? (qb >> 4) : (qb & 0xF);
        lp += (scale * (float)nib - bias) * xp[i];
      }
      acc += lp;
    }
    return acc;
  } else if (T == 1) {
    const int nb = n_in / 256;
    const unsigned char* rp = W + row * nb * 176;
    const int k0 = lane * 8;
    const int g = k0 >> 6, r = k0 & 63, half = r >> 5, l0 = r & 31;
    const int sub = g * 2 + half;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
      const unsigned char* p = rp + b * 176;
      const float d = half_to_float(*(const unsigned short*)(p));
      const float dmin = half_to_float(*(const unsigned short*)(p + 2));
      int sc = 0, mn = 0;
      q4k_scale_min(sub, p + 4, &sc, &mn);
      const float scale = d * (float)sc, bias = dmin * (float)mn;
      const unsigned char* qh = p + 16;
      const unsigned char* q = p + 48 + g * 32 + l0;
      const float* xp = x + b * 256 + k0;
      float lp = 0.f;
      for (int i = 0; i < 8; ++i) {
        const unsigned char qb = q[i];
        const int nib = half ? (qb >> 4) : (qb & 0xF);
        const int hi = (qh[l0 + i] >> sub) & 1;
        lp += (scale * (float)(nib | (hi << 4)) - bias) * xp[i];
      }
      acc += lp;
    }
    return acc;
  } else if (T == 2) {
    const int sb_count = n_in / 256;
    const unsigned char* rp = W + row * sb_count * 210;
    const int is = lane / 16;
    float acc = 0.f;
    for (int sb = 0; sb < sb_count; ++sb) {
      const unsigned char* p = rp + sb * 210;
      const float d = half_to_float(*(const unsigned short*)(p + 208));
      const unsigned char* ql = p;
      const unsigned char* qh = p + 128;
      const signed char* sc = (const signed char*)(p + 192);
      const float* xp = x + sb * 256;
      for (int n2 = 0; n2 < 2; ++n2) {
        const int qi = n2 * 64, hi = n2 * 32;
        const int q1 = (int)((ql[qi + lane] & 0xF) | (((qh[hi + lane] >> 0) & 3) << 4)) - 32;
        const int q2 = (int)((ql[qi + lane + 32] & 0xF) | (((qh[hi + lane] >> 2) & 3) << 4)) - 32;
        const int q3 = (int)((ql[qi + lane] >> 4) | (((qh[hi + lane] >> 4) & 3) << 4)) - 32;
        const int q4 = (int)((ql[qi + lane + 32] >> 4) | (((qh[hi + lane] >> 6) & 3) << 4)) - 32;
        const int base = n2 * 128;
        acc += d * (float)sc[n2 * 8 + is + 0] * (float)q1 * xp[base + lane];
        acc += d * (float)sc[n2 * 8 + is + 2] * (float)q2 * xp[base + lane + 32];
        acc += d * (float)sc[n2 * 8 + is + 4] * (float)q3 * xp[base + lane + 64];
        acc += d * (float)sc[n2 * 8 + is + 6] * (float)q4 * xp[base + lane + 96];
      }
    }
    return acc;
  } else if (T == 3) {
    const int nb = n_in / 32;
    const unsigned char* rp = W + row * nb * 34;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
      const unsigned char* p = rp + b * 34;
      const float d = half_to_float(*(const unsigned short*)(p));
      const signed char* q = (const signed char*)(p + 2);
      acc += d * (float)q[lane] * x[b * 32 + lane];
    }
    return acc;
  } else {   // Q5_1
    const int nb = n_in / 32;
    const unsigned char* rp = W + row * nb * 24;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
      const unsigned char* p = rp + b * 24;
      const float d = half_to_float(*(const unsigned short*)(p));
      const float m = half_to_float(*(const unsigned short*)(p + 2));
      const unsigned int qh = (unsigned int)p[4] | ((unsigned int)p[5] << 8) |
                              ((unsigned int)p[6] << 16) | ((unsigned int)p[7] << 24);
      const unsigned char* qs = p + 8;
      const int low = (lane < 16) ? (qs[lane] & 0xF) : (qs[lane - 16] >> 4);
      const int hi = (qh >> lane) & 1;
      acc += (d * (float)(low | (hi << 4)) + m) * x[b * 32 + lane];
    }
    return acc;
  }
}

__device__ __forceinline__ float warp_reduce(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Four rows per warp, sharing the x loads (mirrors the gemv_*_x4 kernels).
template <int T>
__device__ __forceinline__ void qrows4(const unsigned char* __restrict__ W, long long row0,
                                       const float* __restrict__ x, int n_in, int lane,
                                       int n_out, float acc[4]) {
  const long long row_bytes = (T == 0)   ? (long long)(n_in / 256) * 144
                              : (T == 1) ? (long long)(n_in / 256) * 176
                              : (T == 2) ? (long long)(n_in / 256) * 210
                              : (T == 3) ? (long long)(n_in / 32) * 34
                                         : (long long)(n_in / 32) * 24;
  const int nb_k = n_in / 256;
  const int nb_8 = n_in / 32;
  const int k0 = lane * 8;
  const int gq = k0 >> 6, rc = k0 & 63, halfq = rc >> 5, l0 = rc & 31;
  const int subq = gq * 2 + halfq;
  const int is = lane / 16;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const long long rr = row0 + r;
    const long long row = rr < n_out ? rr : (long long)(n_out - 1);
    const unsigned char* rp = W + row * row_bytes;
    if (T == 0) {
      for (int b = 0; b < nb_k; ++b) {
        const unsigned char* p = rp + b * 144;
        const float d = half_to_float(*(const unsigned short*)(p));
        const float dmin = half_to_float(*(const unsigned short*)(p + 2));
        int sc = 0, mn = 0;
        q4k_scale_min(subq, p + 4, &sc, &mn);
        const float scale = d * (float)sc, bias = dmin * (float)mn;
        const unsigned char* q = p + 16 + gq * 32 + l0;
        const float* xp = x + b * 256 + k0;
        float lp = 0.f;
        for (int i = 0; i < 8; ++i) {
          const int byte = q[i];
          const int nib = halfq ? (byte >> 4) : (byte & 0xF);
          lp += (scale * (float)nib - bias) * xp[i];
        }
        acc[r] += lp;
      }
    } else if (T == 1) {
      for (int b = 0; b < nb_k; ++b) {
        const unsigned char* p = rp + b * 176;
        const float d = half_to_float(*(const unsigned short*)(p));
        const float dmin = half_to_float(*(const unsigned short*)(p + 2));
        int sc = 0, mn = 0;
        q4k_scale_min(subq, p + 4, &sc, &mn);
        const float scale = d * (float)sc, bias = dmin * (float)mn;
        const unsigned char* qh = p + 16;
        const unsigned char* q = p + 48 + gq * 32 + l0;
        const float* xp = x + b * 256 + k0;
        float lp = 0.f;
        for (int i = 0; i < 8; ++i) {
          const int byte = q[i];
          const int nib = halfq ? (byte >> 4) : (byte & 0xF);
          const int hi = (qh[l0 + i] >> subq) & 1;
          lp += (scale * (float)(nib | (hi << 4)) - bias) * xp[i];
        }
        acc[r] += lp;
      }
    } else if (T == 2) {
      for (int sb = 0; sb < nb_k; ++sb) {
        const unsigned char* p = rp + sb * 210;
        const float d = half_to_float(*(const unsigned short*)(p + 208));
        const unsigned char* ql = p;
        const unsigned char* qh = p + 128;
        const signed char* sc = (const signed char*)(p + 192);
        const float* xp = x + sb * 256;
        for (int n2 = 0; n2 < 2; ++n2) {
          const int qi = n2 * 64, hi = n2 * 32;
          const int q1 = (int)((ql[qi + lane] & 0xF) | (((qh[hi + lane] >> 0) & 3) << 4)) - 32;
          const int q2 = (int)((ql[qi + lane + 32] & 0xF) | (((qh[hi + lane] >> 2) & 3) << 4)) - 32;
          const int q3 = (int)((ql[qi + lane] >> 4) | (((qh[hi + lane] >> 4) & 3) << 4)) - 32;
          const int q4 = (int)((ql[qi + lane + 32] >> 4) | (((qh[hi + lane] >> 6) & 3) << 4)) - 32;
          const int base = n2 * 128;
          acc[r] += d * (float)sc[n2 * 8 + is + 0] * (float)q1 * xp[base + lane];
          acc[r] += d * (float)sc[n2 * 8 + is + 2] * (float)q2 * xp[base + lane + 32];
          acc[r] += d * (float)sc[n2 * 8 + is + 4] * (float)q3 * xp[base + lane + 64];
          acc[r] += d * (float)sc[n2 * 8 + is + 6] * (float)q4 * xp[base + lane + 96];
        }
      }
    } else if (T == 3) {
      for (int b = 0; b < nb_8; ++b) {
        const unsigned char* p = rp + b * 34;
        const float d = half_to_float(*(const unsigned short*)(p));
        const signed char* q = (const signed char*)(p + 2);
        acc[r] += d * (float)q[lane] * x[b * 32 + lane];
      }
    } else {   // Q5_1
      for (int b = 0; b < nb_8; ++b) {
        const unsigned char* p = rp + b * 24;
        const float d = half_to_float(*(const unsigned short*)(p));
        const float m = half_to_float(*(const unsigned short*)(p + 2));
        const unsigned int qh = (unsigned int)p[4] | ((unsigned int)p[5] << 8) |
                                ((unsigned int)p[6] << 16) | ((unsigned int)p[7] << 24);
        const unsigned char* qs = p + 8;
        const int low = (lane < 16) ? (qs[lane] & 0xF) : (qs[lane - 16] >> 4);
        const int hi = (qh >> lane) & 1;
        acc[r] += (d * (float)(low | (hi << 4)) + m) * x[b * 32 + lane];
      }
    }
  }
}

__device__ __forceinline__ void qrows4_t(int t, const unsigned char* W, long long row0,
                                         const float* x, int n_in, int lane, int n_out,
                                         float acc[4]) {
  if (t == 0) qrows4<0>(W, row0, x, n_in, lane, n_out, acc);
  else if (t == 1) qrows4<1>(W, row0, x, n_in, lane, n_out, acc);
  else if (t == 2) qrows4<2>(W, row0, x, n_in, lane, n_out, acc);
  else if (t == 3) qrows4<3>(W, row0, x, n_in, lane, n_out, acc);
  else qrows4<4>(W, row0, x, n_in, lane, n_out, acc);
}

__device__ __forceinline__ float qrow_t(int t, const unsigned char* W, long long row,
                                        const float* x, int n_in, int lane) {
  if (t == 0) return qrow_dot<0>(W, row, x, n_in, lane);
  if (t == 1) return qrow_dot<1>(W, row, x, n_in, lane);
  if (t == 2) return qrow_dot<2>(W, row, x, n_in, lane);
  if (t == 3) return qrow_dot<3>(W, row, x, n_in, lane);
  return qrow_dot<4>(W, row, x, n_in, lane);
}

// gate+up+silu for every active expert slot of one MoE layer
extern "C" __global__ void moe_gateup_silu(const unsigned char* const* __restrict__ gptrs,
                                           const unsigned char* const* __restrict__ uptrs,
                                           const int* __restrict__ idx,
                                           const float* __restrict__ x,
                                           float* __restrict__ act, int hidden, int inter,
                                           int type_g, int type_u) {
  const int slot = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long row = blockIdx.x * 8 + warp;
  if (row >= inter) return;
  const float* xr = x + (long long)idx[slot] * hidden;
  const float g = warp_reduce(qrow_t(type_g, gptrs[slot], row, xr, hidden, lane));
  const float u = warp_reduce(qrow_t(type_u, uptrs[slot], row, xr, hidden, lane));
  if (lane == 0) act[(long long)slot * inter + row] = (g / (1.f + expf(-g))) * u;
}

// down + weighted scatter-add into xout
extern "C" __global__ void moe_down_scatter(const unsigned char* const* __restrict__ dptrs,
                                            const int* __restrict__ idx,
                                            const float* __restrict__ wts,
                                            const float* __restrict__ act,
                                            float* __restrict__ xout, int hidden, int inter,
                                            int type_d) {
  const int slot = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long row = blockIdx.x * 8 + warp;
  if (row >= hidden) return;
  const float* ar = act + (long long)slot * inter;
  const float v = warp_reduce(qrow_t(type_d, dptrs[slot], row, ar, inter, lane));
  if (lane == 0) atomicAdd(&xout[(long long)idx[slot] * hidden + row], wts[slot] * v);
}
// ---- device-side router (removes the per-layer host round-trip) ----
__device__ __forceinline__ float sigmoidf_(float v) { return 1.f / (1.f + expf(-v)); }

extern "C" __global__ void moe_route(const float* __restrict__ logits, int ne, int topk,
                                     int router_sigmoid, int norm_topk,
                                     int* __restrict__ slot_exp, int* __restrict__ slot_tok,
                                     float* __restrict__ slot_w) {
  if (threadIdx.x != 0) return;
  const int t = blockIdx.x;
  if (ne > 256) return;
  const float* row = logits + (long long)t * ne;
  float pr[256];
  if (router_sigmoid) {
    for (int e = 0; e < ne; ++e) pr[e] = sigmoidf_(row[e]);
  } else {
    float mx = -1e30f;
    for (int e = 0; e < ne; ++e) mx = fmaxf(mx, row[e]);
    float sum = 0.f;
    for (int e = 0; e < ne; ++e) {
      pr[e] = expf(row[e] - mx);
      sum += pr[e];
    }
    for (int e = 0; e < ne; ++e) pr[e] /= sum;
  }
  float wsum = 0.f;
  for (int k = 0; k < topk; ++k) {
    int best = 0;
    float bv = -1e30f;
    for (int e = 0; e < ne; ++e)
      if (pr[e] > bv) {
        bv = pr[e];
        best = e;
      }
    wsum += bv;
    pr[best] = -1e30f;
    const int slot = t * topk + k;
    slot_exp[slot] = best;
    slot_w[slot] = bv;
    slot_tok[slot] = t;
  }
  if (norm_topk && topk > 0) {
    for (int k = 0; k < topk; ++k) {
      const int slot = t * topk + k;
      slot_w[slot] = slot_w[slot] / (wsum > 0.f ? wsum : 1.f);
    }
  }
}

extern "C" __global__ void zero_fill(float* __restrict__ p, long long n) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x)
    p[i] = 0.f;
}

extern "C" __global__ void moe_gateup_silu_dev(const unsigned char* const* __restrict__ gptrs,
                                               const unsigned char* const* __restrict__ uptrs,
                                               const int* __restrict__ slot_exp,
                                               const int* __restrict__ slot_tok,
                                               const float* __restrict__ x,
                                               float* __restrict__ act, int hidden, int inter,
                                               int type_g, int type_u) {
  const int slot = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long row0 = (long long)(blockIdx.x * (blockDim.x >> 5) + warp) * 4;
  if (row0 >= inter) return;
  const int e = slot_exp[slot];
  const float* xr = x + (long long)slot_tok[slot] * hidden;
  float ga[4] = {0.f, 0.f, 0.f, 0.f}, ua[4] = {0.f, 0.f, 0.f, 0.f};
  qrows4_t(type_g, gptrs[e], row0, xr, hidden, lane, inter, ga);
  qrows4_t(type_u, uptrs[e], row0, xr, hidden, lane, inter, ua);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float g = warp_reduce(ga[r]);
    const float u = warp_reduce(ua[r]);
    if (lane == 0 && row0 + r < inter)
      act[(long long)slot * inter + row0 + r] = (g / (1.f + expf(-g))) * u;
  }
}

extern "C" __global__ void moe_down_scatter_dev(const unsigned char* const* __restrict__ dptrs,
                                                const int* __restrict__ slot_exp,
                                                const int* __restrict__ slot_tok,
                                                const float* __restrict__ slot_w,
                                                const float* __restrict__ act,
                                                float* __restrict__ xout, int hidden, int inter,
                                                int type_d) {
  const int slot = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long row0 = (long long)(blockIdx.x * (blockDim.x >> 5) + warp) * 4;
  if (row0 >= hidden) return;
  const int e = slot_exp[slot];
  const float* ar = act + (long long)slot * inter;
  float va[4] = {0.f, 0.f, 0.f, 0.f};
  qrows4_t(type_d, dptrs[e], row0, ar, inter, lane, hidden, va);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float v = warp_reduce(va[r]);
    if (lane == 0 && row0 + r < hidden)
      atomicAdd(&xout[(long long)slot_tok[slot] * hidden + row0 + r], slot_w[slot] * v);
  }
}

)cuda";


inline const char* kRouterGemm = R"cuda(
// Router logits without cuBLAS (cuBLAS cannot be used inside captured layers).
extern "C" __global__ void router_gemm(const float* __restrict__ hn, const float* __restrict__ w,
                                       float* __restrict__ out, int n, int hidden, int ne) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * ne) return;
  const int t = idx / ne, e = idx % ne;
  const float* x = hn + (long long)t * hidden;
  const float* r = w + (long long)e * hidden;
  float acc = 0.f;
  for (int i = 0; i < hidden; ++i) acc += x[i] * r[i];
  out[idx] = acc;
}
)cuda";

inline const char* kClockWarmer = R"cuda(
// Keeps the GPU in a high power state: Windows drops the SM clock to ~1/3 when the
// GPU idles between CPU-side MoE layers, which halves MoE decode throughput.
extern "C" __global__ void clock_warmer(volatile int* stop) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  while (*stop == 0) __nanosleep(20000);
}
)cuda";

inline const char* kQuantGemm = R"cuda(
// ---- Self-developed quantized GEMM (G1) ------------------------------------
// C[M,N] = A[M,K] (f32 row-major) @ W[N,K]^T (quantized rows).
// One block computes a 32x32 tile; K is streamed in 32-element tiles whose
// dequantized values are staged in shared memory. Weights are dequantized on the
// fly, one 32-element slice per row per k-tile.
__device__ __forceinline__ void deq32_slice(int type, const unsigned char* rowp, int k0, int nb,
                                            float* out) {
  const int blk = k0 >> 5;
  if (type == 3) {   // Q8_0
    const unsigned char* p = rowp + (long long)blk * 34;
    const float d = half_to_float(*(const unsigned short*)(p));
    const signed char* q = (const signed char*)(p + 2);
    for (int i = 0; i < 32; ++i) out[i] = d * (float)q[i];
    return;
  }
  const int blk256 = k0 >> 8;
  const int inblk = k0 & 255;
  const int sub = inblk >> 5;              // 0..7 sub-block of 32
  if (type == 0) {   // Q4_K
    const unsigned char* p = rowp + (long long)blk256 * 144;
    const float d = half_to_float(*(const unsigned short*)(p));
    const float dmin = half_to_float(*(const unsigned short*)(p + 2));
    int sc = 0, mn = 0;
    q4k_scale_min(sub, p + 4, &sc, &mn);
    const float scale = d * (float)sc, bias = dmin * (float)mn;
    const unsigned char* qs = p + 16 + (sub >> 1) * 32;
    const int hi = sub & 1;
    for (int i = 0; i < 32; ++i) {
      const int nb2 = hi ? (qs[i] >> 4) : (qs[i] & 0xF);
      out[i] = scale * (float)nb2 - bias;
    }
    return;
  }
  if (type == 1) {   // Q5_K
    const unsigned char* p = rowp + (long long)blk256 * 176;
    const float d = half_to_float(*(const unsigned short*)(p));
    const float dmin = half_to_float(*(const unsigned short*)(p + 2));
    int sc = 0, mn = 0;
    q4k_scale_min(sub, p + 4, &sc, &mn);
    const float scale = d * (float)sc, bias = dmin * (float)mn;
    const unsigned char* qh = p + 16;
    const unsigned char* qs = p + 48 + (sub >> 1) * 32;
    const int hi = sub & 1;
    for (int i = 0; i < 32; ++i) {
      const int nib = hi ? (qs[i] >> 4) : (qs[i] & 0xF);
      const int h = (qh[i] >> sub) & 1;
      out[i] = scale * (float)(nib | (h << 4)) - bias;
    }
    return;
  }
  // Q6_K slices are 16 elements wide; the GEMM path falls back for it.
  for (int i = 0; i < 32; ++i) out[i] = 0.f;
}

extern "C" __global__ void quant_gemm(const float* __restrict__ A,
                                      const unsigned char* __restrict__ W,
                                      float* __restrict__ C, int M, int N, int K, int type,
                                      long long row_bytes) {
  __shared__ float As[32][33];
  __shared__ float Ws[32][33];
  const int bm = blockIdx.y * 32, bn = blockIdx.x * 32;
  const int tid = threadIdx.x;
  const int tr = tid / 16, tc = tid % 16;
  float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;
  const int nb = (K + 31) / 32;
  for (int k0 = 0; k0 < K; k0 += 32) {
    // A tile: 32 rows x 32 cols, 4 elements per thread
    for (int i = 0; i < 4; ++i) {
      const int idx = tid * 4 + i;
      const int r = idx / 32, c = idx % 32;
      const int gr = bm + r;
      As[r][c] = (gr < M && (k0 + c) < K) ? A[(long long)gr * K + k0 + c] : 0.f;
    }
    // W tile: 32 rows x 32 cols, one thread per row dequantizes the slice
    if (tid < 32) {
      const int gr = bn + tid;
      float tmp[32];
      if (gr < N) deq32_slice(type, W + (long long)gr * row_bytes, k0, nb, tmp);
      else
        for (int i = 0; i < 32; ++i) tmp[i] = 0.f;
      for (int i = 0; i < 32; ++i) Ws[tid][i] = tmp[i];
    }
    __syncthreads();
    for (int kk = 0; kk < 32; ++kk) {
      const float a0 = As[tr * 2 + 0][kk];
      const float a1 = As[tr * 2 + 1][kk];
      const float b0 = Ws[tc * 2 + 0][kk];
      const float b1 = Ws[tc * 2 + 1][kk];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }
    __syncthreads();
  }
  const int r0 = bm + tr * 2, c0 = bn + tc * 2;
  if (r0 < M && c0 < N) C[(long long)r0 * N + c0] = acc00;
  if (r0 < M && c0 + 1 < N) C[(long long)r0 * N + c0 + 1] = acc01;
  if (r0 + 1 < M && c0 < N) C[(long long)(r0 + 1) * N + c0] = acc10;
  if (r0 + 1 < M && c0 + 1 < N) C[(long long)(r0 + 1) * N + c0 + 1] = acc11;
}
)cuda";

inline const char* kGemvSK = R"cuda(
// Split-K variant: the K dimension is split across blockIdx.y so medium-sized matrices
// (one layer's projection: 4096x4096, only 128 blocks in the plain x4 launch) get
// enough blocks to hide latency. Partials are combined with atomicAdd, so `y` must be
// zeroed by the caller.
extern "C" __global__ void gemv_q5k_x4_sk(const unsigned char* __restrict__ W,
                                          const float* __restrict__ x, float* __restrict__ y,
                                          int n_in, int n_out, int ksplit) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * ((blockDim.x >> 5) * 4) + warp * 4;
  const int sb_count = n_in / 256;
  const int per = (sb_count + ksplit - 1) / ksplit;
  const int sb0 = blockIdx.y * per;
  const int sb1 = min(sb_count, sb0 + per);
  if (sb0 >= sb_count) return;
  const int k0 = lane * 8;
  const int g = k0 >> 6, rr = k0 & 63, half = rr >> 5, l0 = rr & 31;
  const int sub = g * 2 + half;
  const size_t row_bytes = (size_t)sb_count * 176;
  const unsigned char* ptr[4];
  bool valid[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    valid[r] = (row0 + r) < n_out;
    const int row = valid[r] ? (row0 + r) : (n_out - 1);
    ptr[r] = W + (long long)row * row_bytes;
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int sb = sb0; sb < sb1; ++sb) {
    const float* xp = x + sb * 256 + k0;
    const float4 xa = *reinterpret_cast<const float4*>(xp);
    const float4 xb = *reinterpret_cast<const float4*>(xp + 4);
    const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    const int qoff = 48 + g * 32 + l0;
    const int qhoff = 16 + l0;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const unsigned char* p = ptr[r] + sb * 176;
      const float d = half_to_float(*(const unsigned short*)(p));
      const float dmin = half_to_float(*(const unsigned short*)(p + 2));
      int sc = 0, mn = 0;
      q4k_scale_min(sub, p + 4, &sc, &mn);
      const float scale = d * (float)sc, bias = dmin * (float)mn;
      const unsigned long long bits = *reinterpret_cast<const unsigned long long*>(p + qoff);
      const unsigned long long hbits = *reinterpret_cast<const unsigned long long*>(p + qhoff);
      float lp = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int byte = (int)((bits >> (8 * i)) & 0xFFull);
        const int nib = half ? (byte >> 4) : (byte & 0xF);
        const int hb = (int)((hbits >> (8 * i)) & 0xFFull);
        const int q = nib | (((hb >> sub) & 1) << 4);
        lp += (scale * (float)q - bias) * xs[i];
      }
      acc[r] += lp;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float s = acc[r];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0 && valid[r]) atomicAdd(&y[row0 + r], s);
  }
}

extern "C" __global__ void gemv_q6k_x4_sk(const unsigned char* __restrict__ W,
                                          const float* __restrict__ x, float* __restrict__ y,
                                          int n_in, int n_out, int ksplit) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * ((blockDim.x >> 5) * 4) + warp * 4;
  const int nb = n_in / 256;
  const int per = (nb + ksplit - 1) / ksplit;
  const int sb0 = blockIdx.y * per;
  const int sb1 = min(nb, sb0 + per);
  if (sb0 >= nb) return;
  const size_t row_bytes = (size_t)nb * 210;
  const unsigned char* ptr[4];
  bool valid[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    valid[r] = (row0 + r) < n_out;
    const int row = valid[r] ? (row0 + r) : (n_out - 1);
    ptr[r] = W + (long long)row * row_bytes;
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int is = lane / 16;
  for (int sb = sb0; sb < sb1; ++sb) {
    const float* xb = x + sb * 256;
    for (int n2 = 0; n2 < 2; ++n2) {
      const int qi = n2 * 64, hi = n2 * 32;
      const float x0 = xb[n2 * 128 + lane];
      const float x1 = xb[n2 * 128 + lane + 32];
      const float x2 = xb[n2 * 128 + lane + 64];
      const float x3 = xb[n2 * 128 + lane + 96];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const unsigned char* p = ptr[r] + sb * 210;
        const float d = half_to_float(*(const unsigned short*)(p + 208));
        const unsigned char* ql = p;
        const unsigned char* qh = p + 128;
        const signed char* sc = (const signed char*)(p + 192);
        const int q1 = (int)((ql[qi + lane] & 0xF) | (((qh[hi + lane] >> 0) & 3) << 4)) - 32;
        const int q2 = (int)((ql[qi + lane + 32] & 0xF) | (((qh[hi + lane] >> 2) & 3) << 4)) - 32;
        const int q3 = (int)((ql[qi + lane] >> 4) | (((qh[hi + lane] >> 4) & 3) << 4)) - 32;
        const int q4 = (int)((ql[qi + lane + 32] >> 4) | (((qh[hi + lane] >> 6) & 3) << 4)) - 32;
        acc[r] += d * (float)sc[n2 * 8 + is + 0] * (float)q1 * x0;
        acc[r] += d * (float)sc[n2 * 8 + is + 2] * (float)q2 * x1;
        acc[r] += d * (float)sc[n2 * 8 + is + 4] * (float)q3 * x2;
        acc[r] += d * (float)sc[n2 * 8 + is + 6] * (float)q4 * x3;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float s = acc[r];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0 && valid[r]) atomicAdd(&y[row0 + r], s);
  }
}
)cuda";

inline const char* kGemvQ8SK = R"cuda(
// Tiny-output GEMV (e.g. 32 rows x 4096: the SSM beta/gate projections) leaves only
// 4 blocks in the plain launch, so the kernel is pure latency. Split K across
// blockIdx.y and combine with atomicAdd (caller zeroes y).
extern "C" __global__ void gemv_q8_0_sk(const unsigned char* __restrict__ W,
                                        const float* __restrict__ x, float* __restrict__ y,
                                        int n_in, int n_out, int ksplit) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n_out) return;
  const int nb = n_in / 32;
  const int per = (nb + ksplit - 1) / ksplit;
  const int b0 = blockIdx.y * per;
  const int b1 = min(nb, b0 + per);
  const unsigned char* rp = W + (long long)row * nb * 34;
  float acc = 0.f;
  for (int b = b0; b < b1; ++b) {
    const unsigned char* p = rp + b * 34;
    const float d = half_to_float(*(const unsigned short*)(p));
    const signed char* q = (const signed char*)(p + 2);
    const int i = lane;
    acc += d * (float)q[i] * x[b * 32 + i];
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if (lane == 0) atomicAdd(&y[row], acc);
}
)cuda";

inline const char* kGemvDp4a = R"cuda(// ---- int8 dp4a GEMV (Q5_K weights, int8-quantized activations) --------------
// The activation vector is quantised once per token and the whole matvec is then
// an int8 dot product: one __dp4a performs 4 MACs, so the 5-bit weight unpack
// cost that dominates the fp32 kernel drops by ~4x.
//
// Q5_K block layout (176 B / 256 values):
//   [0,2) d (fp16)   [2,4) dmin (fp16)   [4,16) 12 B 6-bit (scale,min) pairs
//   [16,48) qh (32 B high bits, bit s of qh[i] = element 32*s+i)   [48,176) qs
// Element e = 32*s+i sits in qs[(s>>1)*32 + i], nibble (s&1), high bit s of qh[i].
// w = d*sc*q - dmin*mn, so with x ~ dx1*q1 + dx2*q2:
//   sum w*x = d*sc*(dx1*sum(q*q1) + dx2*sum(q*q2)) - dmin*mn*sum(x_hat).
// The second (residual) int8 level costs one more dp4a per 4 values but makes the
// activation error ~1e-5 relative (a single int8 level is ~5e-3).
__device__ __forceinline__ float dp4a_h2f(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}

// One warp per 32 values.  dx[2*blk], dx[2*blk+1] = first/residual scale,
// xs8[blk] = sum of the reconstructed activation over the block.
// Two-level int8 quantisation with a tail guard for i >= n; the dx/xs8 writes use
// ceil(n/32) slots, which the host buffers cover up to kDp4aMaxElems (audit D8).
extern "C" __global__ void quant_x_i8(const float* __restrict__ x, signed char* __restrict__ qx,
                                      float* __restrict__ dx, float* __restrict__ xs8, int n) {
  const int lane = threadIdx.x & 31;
  const int blk = blockIdx.x;
  const int i = blk * 32 + lane;
  const float v = (i < n) ? x[i] : 0.f;
  float amax = fabsf(v);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const float d1 = amax * (1.f / 127.f);
  const float inv1 = d1 > 0.f ? 1.f / d1 : 0.f;
  int q1 = (int)lrintf(v * inv1);
  q1 = max(-127, min(127, q1));
  const float r = v - d1 * (float)q1;
  float amax2 = fabsf(r);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) amax2 = fmaxf(amax2, __shfl_xor_sync(0xffffffffu, amax2, off));
  const float d2 = amax2 * (1.f / 127.f);
  const float inv2 = d2 > 0.f ? 1.f / d2 : 0.f;
  int q2 = (int)lrintf(r * inv2);
  q2 = max(-127, min(127, q2));
  if (i < n) {
    qx[i] = (signed char)q1;
    qx[n + i] = (signed char)q2;
  }
  int s1 = q1, s2 = q2;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s1 += __shfl_xor_sync(0xffffffffu, s1, off);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, off);
  if (lane == 0) {
    dx[2 * blk] = d1;
    dx[2 * blk + 1] = d2;
    xs8[blk] = d1 * (float)s1 + d2 * (float)s2;
  }
}

// Assemble the 4 int8 weights of one 32-bit qs word: nibbles from qs, +16 for the
// high bit stored in bit `sub` of the four corresponding qh bytes.
__device__ __forceinline__ int dp4a_q5k_word(unsigned qs, unsigned qh, int nib_sh, int sub) {
  const unsigned nib = (qs >> nib_sh) & 0x0F0F0F0Fu;
  const unsigned hi = ((qh >> sub) & 0x01010101u) << 4;
  return (int)(nib | hi);
}

// 4 rows per warp, lane = (row 0..3, Q5_K sub-block 0..7): each lane owns 32
// contiguous elements of its row, i.e. one full sub-block => its own scale/min.
extern "C" __global__ void gemv_q5k_x4_dp4a(const unsigned char* __restrict__ W,
                                            const signed char* __restrict__ qx,
                                            const float* __restrict__ dx,
                                            const float* __restrict__ xs8, float* __restrict__ y,
                                            int n_in, int n_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * ((blockDim.x >> 5) * 4) + warp * 4;
  const int sb_count = n_in / 256;
  const int r = lane >> 3;
  const int sub = lane & 7;
  const int row = row0 + r;
  const bool valid = row < n_out;
  const int rc = valid ? row : (n_out - 1);
  const unsigned char* rp = W + (long long)rc * (size_t)sb_count * 176;
  const int qs_off = 48 + (sub >> 1) * 32;
  const int nib_sh = (sub & 1) * 4;
  const int sh = (sub & 3) * 8;
  const bool lo_sub = sub < 4;
  const signed char* xp = qx + 32 * sub;
  const signed char* xp2 = xp + n_in;
  float acc = 0.f;
  for (int sb = 0; sb < sb_count; ++sb) {
    const unsigned char* p = rp + (size_t)sb * 176;
    const int4 hd = *reinterpret_cast<const int4*>(p);
    const int4 ha = *reinterpret_cast<const int4*>(p + 16);
    const int4 hb = *reinterpret_cast<const int4*>(p + 32);
    const int4 qa = *reinterpret_cast<const int4*>(p + qs_off);
    const int4 qb = *reinterpret_cast<const int4*>(p + qs_off + 16);
    const int4 xa = *reinterpret_cast<const int4*>(xp + sb * 256);
    const int4 xb = *reinterpret_cast<const int4*>(xp + sb * 256 + 16);
    const int4 ya = *reinterpret_cast<const int4*>(xp2 + sb * 256);
    const int4 yb = *reinterpret_cast<const int4*>(xp2 + sb * 256 + 16);
    // 6-bit (sc,mn) pair of this lane's sub-block, branchless (see
    // q4k_scale_min: subs 0..3 read bytes j / j+4, subs 4..7 read bytes 8..11
    // plus the top two bits of bytes j-4 / j).
    const unsigned w0 = (unsigned)hd.y, w1 = (unsigned)hd.z, w2 = (unsigned)hd.w;
    const unsigned byteA = (w0 >> sh) & 0xFFu;
    const unsigned byteB = ((lo_sub ? w0 : w2) >> sh) & 0xFFu;
    const unsigned byteC = (w1 >> sh) & 0xFFu;
    const int sc = (int)((byteB & 0x0Fu) | ((byteA >> (lo_sub ? 0 : 2)) & 0x30u));
    const unsigned dmb = lo_sub ? (byteC & 0x0Fu) : (((w2 >> sh) >> 4) & 0x0Fu);
    const int mn = (int)(dmb | ((byteC >> (lo_sub ? 0 : 2)) & 0x30u));
    const int wi0 = dp4a_q5k_word((unsigned)qa.x, (unsigned)ha.x, nib_sh, sub);
    const int wi1 = dp4a_q5k_word((unsigned)qa.y, (unsigned)ha.y, nib_sh, sub);
    const int wi2 = dp4a_q5k_word((unsigned)qa.z, (unsigned)ha.z, nib_sh, sub);
    const int wi3 = dp4a_q5k_word((unsigned)qa.w, (unsigned)ha.w, nib_sh, sub);
    const int wi4 = dp4a_q5k_word((unsigned)qb.x, (unsigned)hb.x, nib_sh, sub);
    const int wi5 = dp4a_q5k_word((unsigned)qb.y, (unsigned)hb.y, nib_sh, sub);
    const int wi6 = dp4a_q5k_word((unsigned)qb.z, (unsigned)hb.z, nib_sh, sub);
    const int wi7 = dp4a_q5k_word((unsigned)qb.w, (unsigned)hb.w, nib_sh, sub);
    int P1 = 0, P2 = 0;
    P1 = __dp4a(wi0, xa.x, P1);
    P1 = __dp4a(wi1, xa.y, P1);
    P1 = __dp4a(wi2, xa.z, P1);
    P1 = __dp4a(wi3, xa.w, P1);
    P1 = __dp4a(wi4, xb.x, P1);
    P1 = __dp4a(wi5, xb.y, P1);
    P1 = __dp4a(wi6, xb.z, P1);
    P1 = __dp4a(wi7, xb.w, P1);
    P2 = __dp4a(wi0, ya.x, P2);
    P2 = __dp4a(wi1, ya.y, P2);
    P2 = __dp4a(wi2, ya.z, P2);
    P2 = __dp4a(wi3, ya.w, P2);
    P2 = __dp4a(wi4, yb.x, P2);
    P2 = __dp4a(wi5, yb.y, P2);
    P2 = __dp4a(wi6, yb.z, P2);
    P2 = __dp4a(wi7, yb.w, P2);
    const int si = sb * 8 + sub;
    const float d = dp4a_h2f((unsigned short)((unsigned)hd.x & 0xFFFFu));
    const float dmin = dp4a_h2f((unsigned short)((unsigned)hd.x >> 16));
    const float dot = dx[2 * si] * (float)P1 + dx[2 * si + 1] * (float)P2;
    acc += d * (float)sc * dot - dmin * (float)mn * xs8[si];
  }
#pragma unroll
  for (int off = 1; off < 8; off <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if ((lane & 7) == 0 && valid) y[row] = acc;
}
// Short-row variant of the kernel above: the two int8 activation levels are first
// staged in shared memory, with a 48-byte stride per 32-value sub-block so the
// 16-byte LDS accesses of the 8 sub-blocks hit 8 distinct bank groups.  Without
// staging the same 32 activation bytes are re-fetched through L1 once per row
// (4x per warp), which measurably limits the gate/up shape.
// Dynamic shared bytes = 3*n_in (<= 48 KiB); callers must fall back to the
// global-load kernel for wider rows.
extern "C" __global__ void gemv_q5k_x4_dp4a_s(const unsigned char* __restrict__ W,
                                              const signed char* __restrict__ qx,
                                              const float* __restrict__ dx,
                                              const float* __restrict__ xs8, float* __restrict__ y,
                                              int n_in, int n_out) {
  extern __shared__ signed char shx[];
  const int sb_count = n_in / 256;
  const int nblk = sb_count * 8;   // (sb, sub) 32-byte chunks per quantisation level
  for (int i = threadIdx.x; i < nblk * 4; i += blockDim.x) {
    const int chunk = i >> 1, half = i & 1;
    const int level = chunk / nblk, rem = chunk - level * nblk;
    const int4 v = reinterpret_cast<const int4*>(qx + level * n_in)[rem * 2 + half];
    reinterpret_cast<int4*>(shx)[(level * nblk + rem) * 3 + half] = v;
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * ((blockDim.x >> 5) * 4) + warp * 4;
  const int r = lane >> 3;
  const int sub = lane & 7;
  const int row = row0 + r;
  const bool valid = row < n_out;
  const unsigned char* rp = W + (long long)(valid ? row : (n_out - 1)) * (size_t)sb_count * 176;
  const int qs_off = 48 + (sub >> 1) * 32;
  const int nib_sh = (sub & 1) * 4;
  const int sh = (sub & 3) * 8;
  const bool lo_sub = sub < 4;
  const int xo = sub * 48;
  const int lvl = nblk * 3 * 16;
  float acc = 0.f;
#pragma unroll 2
  for (int sb = 0; sb < sb_count; ++sb) {
    const unsigned char* p = rp + (size_t)sb * 176;
    const int4 hd = *reinterpret_cast<const int4*>(p);
    const int4 ha = *reinterpret_cast<const int4*>(p + 16);
    const int4 hb = *reinterpret_cast<const int4*>(p + 32);
    const int4 qa = *reinterpret_cast<const int4*>(p + qs_off);
    const int4 qb = *reinterpret_cast<const int4*>(p + qs_off + 16);
    const int xb0 = sb * 384 + xo;
    const int4 xa = *reinterpret_cast<const int4*>(shx + xb0);
    const int4 xb = *reinterpret_cast<const int4*>(shx + xb0 + 16);
    const int4 ya = *reinterpret_cast<const int4*>(shx + lvl + xb0);
    const int4 yb = *reinterpret_cast<const int4*>(shx + lvl + xb0 + 16);
    // 6-bit (sc,mn) pair of this lane's sub-block, branchless.
    const unsigned w0 = (unsigned)hd.y, w1 = (unsigned)hd.z, w2 = (unsigned)hd.w;
    const unsigned byteA = (w0 >> sh) & 0xFFu;
    const unsigned byteB = ((lo_sub ? w0 : w2) >> sh) & 0xFFu;
    const unsigned byteC = (w1 >> sh) & 0xFFu;
    const int sc = (int)((byteB & 0x0Fu) | ((byteA >> (lo_sub ? 0 : 2)) & 0x30u));
    const unsigned dmb = lo_sub ? (byteC & 0x0Fu) : (((w2 >> sh) >> 4) & 0x0Fu);
    const int mn = (int)(dmb | ((byteC >> (lo_sub ? 0 : 2)) & 0x30u));
    const int wi0 = dp4a_q5k_word((unsigned)qa.x, (unsigned)ha.x, nib_sh, sub);
    const int wi1 = dp4a_q5k_word((unsigned)qa.y, (unsigned)ha.y, nib_sh, sub);
    const int wi2 = dp4a_q5k_word((unsigned)qa.z, (unsigned)ha.z, nib_sh, sub);
    const int wi3 = dp4a_q5k_word((unsigned)qa.w, (unsigned)ha.w, nib_sh, sub);
    const int wi4 = dp4a_q5k_word((unsigned)qb.x, (unsigned)hb.x, nib_sh, sub);
    const int wi5 = dp4a_q5k_word((unsigned)qb.y, (unsigned)hb.y, nib_sh, sub);
    const int wi6 = dp4a_q5k_word((unsigned)qb.z, (unsigned)hb.z, nib_sh, sub);
    const int wi7 = dp4a_q5k_word((unsigned)qb.w, (unsigned)hb.w, nib_sh, sub);
    int P1 = __dp4a(wi0, xa.x, 0);
    P1 = __dp4a(wi1, xa.y, P1);
    P1 = __dp4a(wi2, xa.z, P1);
    P1 = __dp4a(wi3, xa.w, P1);
    P1 = __dp4a(wi4, xb.x, P1);
    P1 = __dp4a(wi5, xb.y, P1);
    P1 = __dp4a(wi6, xb.z, P1);
    P1 = __dp4a(wi7, xb.w, P1);
    int P2 = __dp4a(wi0, ya.x, 0);
    P2 = __dp4a(wi1, ya.y, P2);
    P2 = __dp4a(wi2, ya.z, P2);
    P2 = __dp4a(wi3, ya.w, P2);
    P2 = __dp4a(wi4, yb.x, P2);
    P2 = __dp4a(wi5, yb.y, P2);
    P2 = __dp4a(wi6, yb.z, P2);
    P2 = __dp4a(wi7, yb.w, P2);
    const int si = sb * 8 + sub;
    const float d = dp4a_h2f((unsigned short)((unsigned)hd.x & 0xFFFFu));
    const float dmin = dp4a_h2f((unsigned short)((unsigned)hd.x >> 16));
    const float2 dxx = *reinterpret_cast<const float2*>(dx + 2 * si);
    const float dot = dxx.x * (float)P1 + dxx.y * (float)P2;
    acc += d * (float)sc * dot - dmin * (float)mn * xs8[si];
  }
#pragma unroll
  for (int off = 1; off < 8; off <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if ((lane & 7) == 0 && valid) y[row] = acc;
}
)cuda";

// ---- int8 dp4a GEMV for Q6_K weights ------------------------------------------
inline const char* kGemvDp4aQ6K = R"cuda(
// Same idea as the Q5_K dp4a kernel above: quantise the activation once per token to
// two int8 levels (quant_x_i8, dx = 2 scales / 32 values) and run the matvec as an
// int8 dot product, so one __dp4a replaces 4 float FMAs and the 6-bit unpack.
//
// Q6_K block layout (210 B / 256 values): ql 128 B, qh 64 B, scales 16 B (int8, one
// per 16 values), d (fp16) at +208.  Element index -> source bytes, for the n2 half:
//   q1(x = +l)      = ql[qi+l]      & 0xF | (qh[hi+l] >> 0 & 3) << 4   sc[+0]
//   q2(x = 32+l)    = ql[qi+l+32]   & 0xF | (qh[hi+l] >> 2 & 3) << 4   sc[+2]
//   q3(x = 64+l)    = ql[qi+l]      >> 4  | (qh[hi+l] >> 4 & 3) << 4   sc[+4]
//   q4(x = 96+l)    = ql[qi+l+32]   >> 4  | (qh[hi+l] >> 6 & 3) << 4   sc[+6]
// (qi = n2*64, hi = n2*32, sc base = n2*8 + (l/16)).  So one 32-bit ql word plus one
// 32-bit qh word build *four* 6-bit weights (low/high nibbles of the ql word times two
// bits of the qh word) -- the 4-element group at l = 4g..4g+3 of two chunks whose x
// positions are 64 apart.
//
// `lane = (n2, p, g)`: n2 = half, p = chunk pair (0: q1/q3, 1: q2/q4), g = 4-element
// group, so 8 elements per lane and 256 per warp, i.e. one row per warp.  The 6-bit
// value is folded to signed int8 with q-32 = ((q + 0x60) ^ 0x80) per byte: both steps
// are carry-free because 0 <= q <= 63.
// Note the block stride 210 is not 16-byte aligned, so ql/qh/sc are read as 2-byte
// words (the whole row block does coalesce into full 32-byte sectors anyway).
__device__ __forceinline__ float q6k_dp4a_range(const unsigned char* __restrict__ W,
                                                const signed char* __restrict__ qx,
                                                const float* __restrict__ dx, int n_in, int row,
                                                int sb0, int sb1) {
  const int lane = threadIdx.x & 31;
  const int sb_count = n_in >> 8;
  const unsigned char* rp = W + (long long)row * (size_t)sb_count * 210;
  const int n2 = lane >> 4, p = (lane >> 3) & 1, g = lane & 7;
  const int qlb = n2 * 64 + p * 32 + 4 * g;
  const int qhb = n2 * 32 + 4 * g;
  const int xo = n2 * 128 + p * 32 + 4 * g;
  const int blk1 = n2 * 4 + p;
  const int sci = n2 * 8 + (g >> 2) + (p ? 2 : 0);
  const int sh1 = p * 2, sh2 = 4 + p * 2;
  float acc = 0.f;
#pragma unroll 2
  for (int sb = sb0; sb < sb1; ++sb) {
    const unsigned char* blk = rp + (size_t)sb * 210;
    const unsigned qlw = (unsigned)*(const unsigned short*)(blk + qlb) |
                         ((unsigned)*(const unsigned short*)(blk + qlb + 2) << 16);
    const unsigned qhw = (unsigned)*(const unsigned short*)(blk + 128 + qhb) |
                         ((unsigned)*(const unsigned short*)(blk + 128 + qhb + 2) << 16);
    const unsigned w1 = (((qlw & 0x0F0F0F0Fu) | ((((qhw >> sh1) & 0x03030303u) << 4))) +
                         0x60606060u) ^
                        0x80808080u;
    const unsigned w2 = ((((qlw >> 4) & 0x0F0F0F0Fu) | ((((qhw >> sh2) & 0x03030303u) << 4))) +
                         0x60606060u) ^
                        0x80808080u;
    const int xa = *(const int*)(qx + sb * 256 + xo);
    const int xb = *(const int*)(qx + sb * 256 + xo + 64);
    const int ya = *(const int*)(qx + n_in + sb * 256 + xo);
    const int yb = *(const int*)(qx + n_in + sb * 256 + xo + 64);
    const float d = half_to_float(*(const unsigned short*)(blk + 208));
    const float2 da = *reinterpret_cast<const float2*>(dx + 2 * (sb * 8 + blk1));
    const float2 db = *reinterpret_cast<const float2*>(dx + 2 * (sb * 8 + blk1 + 2));
    const signed char* sc = (const signed char*)(blk + 192);
    const int P1a = __dp4a((int)w1, xa, 0);
    const int P1b = __dp4a((int)w2, xb, 0);
    const int P2a = __dp4a((int)w1, ya, 0);
    const int P2b = __dp4a((int)w2, yb, 0);
    acc += d * ((float)sc[sci] * (da.x * (float)P1a + da.y * (float)P2a) +
                (float)sc[sci + 4] * (db.x * (float)P1b + db.y * (float)P2b));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  return acc;
}

// One row per warp.  grid = ceil(n_out / warps_per_block), block = 256 (or 128).
extern "C" __global__ void gemv_q6k_dp4a(const unsigned char* __restrict__ W,
                                         const signed char* __restrict__ qx,
                                         const float* __restrict__ dx, float* __restrict__ y,
                                         int n_in, int n_out) {
  const int warps = blockDim.x >> 5;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= n_out) return;
  const float acc = q6k_dp4a_range(W, qx, dx, n_in, row, 0, n_in >> 8);
  if ((threadIdx.x & 31) == 0) y[row] = acc;
}

// Split-K variant for shapes whose row count leaves the SMs half idle: `split` warps
// cooperate on one row, each taking a contiguous slice of the superblock range, and the
// partials are reduced through shared memory (no atomics, no zero-fill).
// grid = ceil(n_out / (warps_per_block / split)); requires (n_in/256) % split == 0.
extern "C" __global__ void gemv_q6k_dp4a_sk(const unsigned char* __restrict__ W,
                                            const signed char* __restrict__ qx,
                                            const float* __restrict__ dx, float* __restrict__ y,
                                            int n_in, int n_out, int split) {
  __shared__ float part[32];
  const int warps = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nr = warps / split;
  const int row0 = blockIdx.x * nr + (warp / split);
  const int ps = warp % split;
  const int sb_count = n_in >> 8;
  const int chunk = sb_count / split;
  const int row = row0 < n_out ? row0 : n_out - 1;
  const float acc = q6k_dp4a_range(W, qx, dx, n_in, row, ps * chunk, (ps + 1) * chunk);
  if (lane == 0) part[warp] = acc;
  __syncthreads();
  if (threadIdx.x < nr) {
    float s = 0.f;
    for (int j = 0; j < split; ++j) s += part[threadIdx.x * split + j];
    const int r = blockIdx.x * nr + threadIdx.x;
    if (r < n_out) y[r] = s;
  }
}
)cuda";

// ---------------------------------------------------------------------------
// Vectorised RMSNorm (same signature as `rmsnorm`).
// The scalar kernel reads/writes one float per instruction with a 256-float
// stride, which on a 1-block launch is latency bound (~8-13 us per call, 81
// calls per decode token on the 9B).  This variant moves float4 packets: the
// sum-of-squares is accumulated per float4 lane (a different -- equally valid --
// summation order) and the scaling pass is element-wise so it is bit-identical.
// Falls back to the scalar path in-kernel for n % 4 != 0 or unaligned pointers.
inline const char* kRmsNorm4 = R"cuda(
extern "C" __global__ void rmsnorm4(const float* __restrict__ x,
                                    const float* __restrict__ w,
                                    float* __restrict__ y,
                                    int n, float eps) {
  extern __shared__ float s[];
  const int row = blockIdx.x;
  const float* xr = x + (long long)row * n;
  float* yr = y + (long long)row * n;
  const bool vec = ((n & 3) == 0) && ((reinterpret_cast<unsigned long long>(xr) & 15u) == 0) &&
                   ((reinterpret_cast<unsigned long long>(yr) & 15u) == 0) &&
                   ((reinterpret_cast<unsigned long long>(w) & 15u) == 0);
  float sum = 0.f;
  const int n4 = n >> 2;
  if (vec) {
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
      const float4 v = x4[i];
      sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += xr[i] * xr[i];
  }
  // Warp-shuffle reduction + one barrier (the scalar kernel used log2(blockDim)
  // __syncthreads rounds, which dominates a 1-block launch).
  const unsigned full = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(full, sum, off);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarp = (blockDim.x + 31) >> 5;
  if (lane == 0) s[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < nwarp) ? s[lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(full, v, off);
    if (lane == 0) s[0] = v;
  }
  __syncthreads();
  const float scale = rsqrtf(s[0] / (float)n + eps);
  if (vec) {
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* y4 = reinterpret_cast<float4*>(yr);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
      const float4 v = x4[i];
      const float4 wv = w4[i];
      float4 o;
      o.x = v.x * scale * wv.x;
      o.y = v.y * scale * wv.y;
      o.z = v.z * scale * wv.z;
      o.w = v.w * scale * wv.w;
      y4[i] = o;
    }
  } else {
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * scale * w[i];
  }
}
)cuda";

// ---------------------------------------------------------------------------
// Device-side top-k over the logits row (decode head).
// Downloading the whole 1 MB logits row over PCIe costs ~0.27 ms/token on this
// WDDM VM, but greedy decoding only needs the arg-max (and the harness prints
// the top-5).  Two tiny kernels reduce the row to k candidates on the device and
// only k (index,value) pairs travel back.
//   topk_partial: grid = nblk blocks, each scans a contiguous slice and writes
//                 that slice's best (value,index) using the same comparison as
//                 the host reference (strict >, so the earliest index wins ties).
//   topk_final  : one block, single thread selection over the nblk candidates.
inline const char* kTopkPartial = R"cuda(
__device__ __forceinline__ void topk_better(float& bv, int& bi, const float v, const int i) {
  if (i < 0) return;
  if (bi < 0 || v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}
extern "C" __global__ void topk_partial(const float* __restrict__ x, float* __restrict__ pv,
                                        int* __restrict__ pi, int n, int nblk, int k) {
  __shared__ float sv[32];
  __shared__ int si[32];
  // Selection bitmap over the block's local range.  512 words = 16384 indices; the
  // host keeps chunk = ceil(vocab/nblk) inside that bound (runner::topk_partial_blocks),
  // so no block can write past the array.  An O(1) skip test instead of scanning the
  // already-taken list keeps k=64/128 cheap.
  __shared__ unsigned int taken[512];
  const int b = blockIdx.x;
  const int chunk = (n + nblk - 1) / nblk;
  const int lo = b * chunk;
  const int hi = min(n, lo + chunk);
  const int tid = threadIdx.x;
  {
    const int words = ((hi - lo) + 31) >> 5;
    for (int w = tid; w < words; w += blockDim.x) taken[w] = 0u;
    __syncthreads();
  }
  for (int q = 0; q < k; ++q) {
    float best = 0.f;
    int bidx = -1;
    for (int i = lo + tid; i < hi; i += blockDim.x) {
      const int off = i - lo;
      if ((taken[off >> 5] >> (off & 31)) & 1u) continue;
      topk_better(best, bidx, x[i], i);
    }
    const unsigned full = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(full, best, off);
      const int oi = __shfl_down_sync(full, bidx, off);
      topk_better(best, bidx, ov, oi);
    }
    const int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { sv[warp] = best; si[warp] = bidx; }
    __syncthreads();
    if (tid == 0) {
      float bv = 0.f;
      int bi = -1;
      const int nw = (blockDim.x + 31) >> 5;
      for (int j = 0; j < nw; ++j) topk_better(bv, bi, sv[j], si[j]);
      pv[b * k + q] = bv;
      pi[b * k + q] = bi;
      if (bi >= 0) {
        const int off = bi - lo;
        taken[off >> 5] |= 1u << (off & 31);
      }
    }
    __syncthreads();
  }
}
)cuda";

inline const char* kTopkFinal = R"cuda(
extern "C" __global__ void topk_final(const float* __restrict__ pv, const int* __restrict__ pi,
                                     int nc, int k, int* __restrict__ oi, float* __restrict__ ov) {
  __shared__ float sv[32];
  __shared__ int si[32];
  __shared__ int taken[128];
  const int tid = threadIdx.x;
  for (int q = 0; q < k; ++q) {
    float best = 0.f;
    int bidx = -1;
    for (int j = tid; j < nc; j += blockDim.x) {
      const int idx = pi[j];
      if (idx < 0) continue;
      bool skip = false;
      for (int p = 0; p < q; ++p)
        if (taken[p] == idx) { skip = true; break; }
      if (skip) continue;
      const float v = pv[j];
      if (bidx < 0 || v > best || (v == best && idx < bidx)) { best = v; bidx = idx; }
    }
    const unsigned full = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
      const float ov2 = __shfl_down_sync(full, best, off);
      const int oi2 = __shfl_down_sync(full, bidx, off);
      if (oi2 >= 0 && (bidx < 0 || ov2 > best || (ov2 == best && oi2 < bidx))) {
        best = ov2;
        bidx = oi2;
      }
    }
    const int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { sv[warp] = best; si[warp] = bidx; }
    __syncthreads();
    if (tid == 0) {
      float bv = 0.f;
      int bi = -1;
      const int nw = (blockDim.x + 31) >> 5;
      for (int j = 0; j < nw; ++j)
        if (si[j] >= 0 && (bi < 0 || sv[j] > bv || (sv[j] == bv && si[j] < bi))) {
          bv = sv[j];
          bi = si[j];
        }
      ov[q] = bv;
      oi[q] = bi;
      taken[q] = bi;
    }
    __syncthreads();
  }
}
)cuda";

// ===========================================================================
// Quantized KV cache (q8_0 = 34 B / 32 elems, q4_0 = 18 B / 32 elems) plus the
// tiled online-softmax attention kernel that reads it.
//
// KV layout with kv_type != 0: [n_kv][max_seq][nb * block_bytes], row-major with
// nb = ceil(D / 32).  The last block is zero padded when D % 32 != 0; its bytes
// past D are never consumed by the dot products.  q8_0 stores one f16 scale per
// 32 values followed by 32 int8s (same 34 B block the GEMV path already uses);
// q4_0 stores one f16 scale followed by 16 B of nibbles (value = (nibble - 8)*d).
// ===========================================================================
inline const char* kKvQuantCommon = R"cuda(
__device__ __forceinline__ unsigned short f32_to_half_bits(float f) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
__device__ __forceinline__ int kv_nblocks(int D) { return (D + 31) / 32; }
__device__ __forceinline__ int kv_block_bytes(int kv_type) { return kv_type == 1 ? 34 : 18; }
__device__ __forceinline__ int kv_row_bytes(int kv_type, int D) {
  return kv_nblocks(D) * kv_block_bytes(kv_type);
}
// value i of a quantized KV row
__device__ __forceinline__ float kv_get(int kv_type, const unsigned char* row, int i) {
  const int b = i >> 5, e = i & 31;
  const unsigned char* p = row + b * kv_block_bytes(kv_type);
  const float d = half_to_float(*(const unsigned short*)p);
  if (kv_type == 1) return d * (float)((signed char)p[2 + e]);
  const unsigned char v = p[2 + (e & 15)];
  const int q = (e < 16) ? (int)(v & 0xF) : (int)(v >> 4);
  return d * (float)(q - 8);
}
)cuda";

// Quantizing KV write.  Grid = n_tokens * n_kv, blockDim = 128 (4 warps): one
// warp quantizes a whole 32-value block (a 32-lane amax reduction + one packed
// store), block index striped over the warps.
inline const char* kKvWriteQ = R"cuda(
extern "C" __global__ void kv_write_q(const float* __restrict__ k,
                                      const float* __restrict__ v,
                                      unsigned char* __restrict__ Kc,
                                      unsigned char* __restrict__ Vc,
                                      const int* __restrict__ pstart, int n_kv, int D,
                                      int max_seq, int kv_type) {
  const int idx = blockIdx.x;
  const int j = idx / n_kv, h = idx % n_kv;
  const int pos = *pstart + j;
  const int nb = kv_nblocks(D);
  const int cb = kv_block_bytes(kv_type);
  const int rowb = nb * cb;
  const long long dst = ((long long)h * max_seq + pos) * rowb;
  const long long src = ((long long)j * n_kv + h) * D;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarp = blockDim.x >> 5;
  const unsigned mask = 0xffffffffu;
  for (int b = warp; b < nb; b += nwarp) {
    const int base = b * 32;
    const int rem = (D - base < 32) ? (D - base) : 32;
    for (int half = 0; half < 2; ++half) {
      const float* sp = (half ? v : k) + src + base;
      unsigned char* dp = (half ? Vc : Kc) + dst + b * cb;
      const float x = (lane < rem) ? sp[lane] : 0.f;
      float amax = fabsf(x);
      for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(mask, amax, off));
      if (kv_type == 1) {
        const float d = amax > 0.f ? amax / 127.f : 0.f;
        const float id = d > 0.f ? 1.f / d : 0.f;
        int q = __float2int_rn(x * id);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        if (lane == 0) *(unsigned short*)dp = f32_to_half_bits(d);
        dp[2 + lane] = (unsigned char)(q & 0xFF);
      } else {
        const float d = amax > 0.f ? amax / 8.f : 0.f;
        const float id = d > 0.f ? 1.f / d : 0.f;
        int q = __float2int_rn(x * id);
        q = q < -8 ? -8 : (q > 7 ? 7 : q);
        const int nib = q + 8;
        const int hi = __shfl_sync(mask, nib, (lane & 15) + 16);
        if (lane == 0) *(unsigned short*)dp = f32_to_half_bits(d);
        if (lane < 16) dp[2 + lane] = (unsigned char)(nib | (hi << 4));
      }
    }
  }
}
)cuda";

// Tiled causal attention over the KV cache.
// Grid: (n_tokens, n_head, nsplit).  blockDim = 32: one warp owns (query token,
// query head) and walks the keys j == blockIdx.z (mod nsplit) in order, keeping
// a running max / denominator / head_dim accumulator in registers (online
// softmax).  The lane owns D/32 contiguous elements of every K/V row, so the
// warp reads whole rows coalesced; no shared-memory score board is needed, so
// the launch never depends on max_seq (the old kernel's 48 KiB limit).
//
// nsplit == 1: the softmax is finished in-register and written to `out`
// (prefill path: n_tokens * n_head independent warps, no scratch).
// nsplit > 1: (m, s, acc[D]) partials go to `scratch`, `attn_combine` merges
// them (decode path: the key range is split across blocks so a long context
// still fills the SMs).
// Steps are bounded to EL <= 8 (D <= 256) and D % 32 == 0; the dispatch falls
// back to the warp-per-(q,head) kernel otherwise.
inline const char* kAttnTiled = R"cuda(
extern "C" __global__ void attn_tiled(const float* __restrict__ q,
                                      const float* __restrict__ Kf,
                                      const float* __restrict__ Vf,
                                      const unsigned char* __restrict__ Kq,
                                      const unsigned char* __restrict__ Vq,
                                      float* __restrict__ out,
                                      float* __restrict__ scratch,
                                      const int* __restrict__ pstart, int n_head, int n_kv,
                                      int D, int max_seq, int kv_type, int nsplit,
                                      int scratch_stride) {
  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const int sp = blockIdx.z;
  const int qpos = *pstart + i;
  const int lane = threadIdx.x & 31;
  const int rep = n_head / n_kv;
  const int kvh = h / rep;
  const float scale = rsqrtf((float)D);
  const int EL = D / 32;
  const int e0 = lane * EL;
  const int grp = 32 / EL;
  const int qblk = lane / grp;
  const int qoff = (lane % grp) * EL;
  const int cb = kv_block_bytes(kv_type);
  const int rowb = kv_row_bytes(kv_type, D);
  const long long row0 = (long long)kvh * max_seq;
  float qr[8];
  const float* qh = q + ((long long)i * n_head + h) * D + e0;
#pragma unroll
  for (int e = 0; e < 8; ++e) qr[e] = (e < EL) ? __ldg(qh + e) : 0.f;
  float m = -1e30f, s = 0.f;
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
  for (int j = sp; j <= qpos; j += nsplit) {
    float dot = 0.f;
    if (kv_type == 0) {
      const float* kp = Kf + (row0 + j) * D + e0;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) dot += qr[e] * __ldg(kp + e);
    } else if (kv_type == 1) {
      const unsigned char* kp = Kq + (row0 + j) * rowb + qblk * 34;
      const float dq = half_to_float(*(const unsigned short*)kp);
      const signed char* qs = (const signed char*)(kp + 2) + qoff;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) dot += qr[e] * (float)qs[e];
      dot *= dq;
    } else {
      const unsigned char* kp = Kq + (row0 + j) * rowb + qblk * 18;
      const float dq = half_to_float(*(const unsigned short*)kp);
      const unsigned char* nb = kp + 2;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        if (e < EL) {
          const int idx = qoff + e;
          const unsigned char byte = nb[idx & 15];
          const int qv = (idx < 16) ? (int)(byte & 0xF) : (int)(byte >> 4);
          dot += qr[e] * (float)(qv - 8);
        }
      }
      dot *= dq;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
    dot *= scale;
    if (dot > m) {
      const float corr = expf(m - dot);
      s *= corr;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] *= corr;
      m = dot;
    }
    const float p = expf(dot - m);
    s += p;
    if (kv_type == 0) {
      const float* vp = Vf + (row0 + j) * D + e0;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) acc[e] += p * __ldg(vp + e);
    } else if (kv_type == 1) {
      const unsigned char* vp = Vq + (row0 + j) * rowb + qblk * 34;
      const float dq = half_to_float(*(const unsigned short*)vp);
      const signed char* qs = (const signed char*)(vp + 2) + qoff;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) acc[e] += p * ((float)qs[e] * dq);
    } else {
      const unsigned char* vp = Vq + (row0 + j) * rowb + qblk * 18;
      const float dq = half_to_float(*(const unsigned short*)vp);
      const unsigned char* nb = vp + 2;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        if (e < EL) {
          const int idx = qoff + e;
          const unsigned char byte = nb[idx & 15];
          const int qv = (idx < 16) ? (int)(byte & 0xF) : (int)(byte >> 4);
          acc[e] += p * ((float)(qv - 8) * dq);
        }
      }
    }
  }
  if (nsplit == 1) {
    float* op = out + ((long long)i * n_head + h) * D + e0;
    const float inv = s > 0.f ? 1.f / s : 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      if (e < EL) op[e] = acc[e] * inv;
  } else {
    float* base = scratch + (long long)((i * n_head + h) * nsplit + sp) * scratch_stride;
    base[D] = m;
    base[D + 1] = s;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      if (e < EL) base[e0 + e] = acc[e];
  }
}
)cuda";

// Merge the nsplit partial (m, s, acc[D]) records of attn_tiled.
// Grid: (n_tokens, n_head), blockDim = D threads (capped at 256).
// GQA-grouped decode attention: one block per (token, KV head, split) with `rep` warps
// (one warp per query head of that KV group).  All warps walk the same K/V rows in lock
// step, so the row one warp fetches is served from L1 to the other rep-1 warps -- the
// decode analogue of what attn_tiled_pre does for prefill.  The per-lane math, dot
// order, online softmax, scratch layout and combine step are identical to attn_tiled,
// so the numerics do not change; only the number of DRAM reads of K/V drops from
// n_head to n_kv per row.
inline const char* kAttnTiledGqa = R"cuda(
extern "C" __global__ void attn_tiled_gqa(const float* __restrict__ q,
                                         const float* __restrict__ Kf,
                                         const float* __restrict__ Vf,
                                         const unsigned char* __restrict__ Kq,
                                         const unsigned char* __restrict__ Vq,
                                         float* __restrict__ out,
                                         float* __restrict__ scratch,
                                         const int* __restrict__ pstart, int n_head, int n_kv,
                                         int D, int max_seq, int kv_type, int nsplit,
                                         int scratch_stride, int rep, int wide) {
  const int i = blockIdx.x;
  const int kvh = blockIdx.y;
  const int sp = blockIdx.z;
  const int warp = threadIdx.x >> 5;
  const int h = kvh * rep + warp;
  if (h >= n_head) return;
  const int qpos = *pstart + i;
  const int lane = threadIdx.x & 31;
  const float scale = rsqrtf((float)D);
  const int EL = D / 32;
  const int e0 = lane * EL;
  const int grp = 32 / EL;
  const int qblk = lane / grp;
  const int qoff = (lane % grp) * EL;
  const int rowb = kv_row_bytes(kv_type, D);
  const long long row0 = (long long)kvh * max_seq;
  const bool f32_wide = (kv_type == 0 && wide && EL == 8);
  float qr[8];
  const float* qh = q + ((long long)i * n_head + h) * D + e0;
#pragma unroll
  for (int e = 0; e < 8; ++e) qr[e] = (e < EL) ? __ldg(qh + e) : 0.f;
  float m = -1e30f, s = 0.f;
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
  for (int j = sp; j <= qpos; j += nsplit) {
    float dot = 0.f;
    float4 v0, v1;
    if (f32_wide) {
      const float* kp = Kf + (row0 + j) * D + e0;
      const float4 k0 = *reinterpret_cast<const float4*>(kp);
      const float4 k1 = *reinterpret_cast<const float4*>(kp + 4);
      dot = qr[0] * k0.x + qr[1] * k0.y + qr[2] * k0.z + qr[3] * k0.w + qr[4] * k1.x +
            qr[5] * k1.y + qr[6] * k1.z + qr[7] * k1.w;
      const float* vp = Vf + (row0 + j) * D + e0;
      v0 = *reinterpret_cast<const float4*>(vp);
      v1 = *reinterpret_cast<const float4*>(vp + 4);
    } else if (kv_type == 0) {
      const float* kp = Kf + (row0 + j) * D + e0;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) dot += qr[e] * __ldg(kp + e);
    } else if (kv_type == 1) {
      const unsigned char* kp = Kq + (row0 + j) * rowb + qblk * 34;
      const float dq = half_to_float(*(const unsigned short*)kp);
      const signed char* qs = (const signed char*)(kp + 2) + qoff;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) dot += qr[e] * (float)qs[e];
      dot *= dq;
    } else {
      const unsigned char* kp = Kq + (row0 + j) * rowb + qblk * 18;
      const float dq = half_to_float(*(const unsigned short*)kp);
      const unsigned char* nb = kp + 2;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        if (e < EL) {
          const int idx = qoff + e;
          const unsigned char byte = nb[idx & 15];
          const int qv = (idx < 16) ? (int)(byte & 0xF) : (int)(byte >> 4);
          dot += qr[e] * (float)(qv - 8);
        }
      }
      dot *= dq;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
    dot *= scale;
    if (dot > m) {
      const float corr = expf(m - dot);
      s *= corr;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] *= corr;
      m = dot;
    }
    const float p = expf(dot - m);
    s += p;
    if (f32_wide) {
      acc[0] += p * v0.x;
      acc[1] += p * v0.y;
      acc[2] += p * v0.z;
      acc[3] += p * v0.w;
      acc[4] += p * v1.x;
      acc[5] += p * v1.y;
      acc[6] += p * v1.z;
      acc[7] += p * v1.w;
    } else if (kv_type == 0) {
      const float* vp = Vf + (row0 + j) * D + e0;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) acc[e] += p * __ldg(vp + e);
    } else if (kv_type == 1) {
      const unsigned char* vp = Vq + (row0 + j) * rowb + qblk * 34;
      const float dq = half_to_float(*(const unsigned short*)vp);
      const signed char* qs = (const signed char*)(vp + 2) + qoff;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) acc[e] += p * ((float)qs[e] * dq);
    } else {
      const unsigned char* vp = Vq + (row0 + j) * rowb + qblk * 18;
      const float dq = half_to_float(*(const unsigned short*)vp);
      const unsigned char* nb = vp + 2;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        if (e < EL) {
          const int idx = qoff + e;
          const unsigned char byte = nb[idx & 15];
          const int qv = (idx < 16) ? (int)(byte & 0xF) : (int)(byte >> 4);
          acc[e] += p * ((float)(qv - 8) * dq);
        }
      }
    }
  }
  if (nsplit == 1) {
    float* op = out + ((long long)i * n_head + h) * D + e0;
    const float inv = s > 0.f ? 1.f / s : 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      if (e < EL) op[e] = acc[e] * inv;
  } else {
    float* base = scratch + (long long)((i * n_head + h) * nsplit + sp) * scratch_stride;
    base[D] = m;
    base[D + 1] = s;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      if (e < EL) base[e0 + e] = acc[e];
  }
}
)cuda";


inline const char* kAttnCombine = R"cuda(
extern "C" __global__ void attn_combine(const float* __restrict__ scratch,
                                        float* __restrict__ out, int n_head, int D,
                                        int nsplit, int scratch_stride) {
  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const float* base = scratch + (long long)((i * n_head + h) * nsplit) * scratch_stride;
  __shared__ float w[64];   // kv_split is capped at 32; 64 is defensive headroom
  __shared__ float inv_s;
  if (threadIdx.x == 0) {
    float mx = -1e30f;
    for (int sp = 0; sp < nsplit; ++sp) {
      const float sv = base[(long long)sp * scratch_stride + D + 1];
      const float mv = base[(long long)sp * scratch_stride + D];
      if (sv > 0.f && mv > mx) mx = mv;
    }
    float tot = 0.f;
    for (int sp = 0; sp < nsplit; ++sp) {
      const float sv = base[(long long)sp * scratch_stride + D + 1];
      const float mv = base[(long long)sp * scratch_stride + D];
      const float wt = (sv > 0.f) ? sv * expf(mv - mx) : 0.f;
      w[sp] = wt;
      tot += wt;
    }
    inv_s = tot > 0.f ? 1.f / tot : 0.f;
  }
  __syncthreads();
  float* op = out + ((long long)i * n_head + h) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float a = 0.f;
    for (int sp = 0; sp < nsplit; ++sp)
      a += base[(long long)sp * scratch_stride + d] * w[sp];
    op[d] = a * inv_s;
  }
}
)cuda";

// ---- dequantise straight to fp16 (no f32 intermediate) ----------------------
// The prefill GEMM is compute bound on fp16 tensor cores, so the weight slab is
// dequantised directly into an fp16 staging buffer: half the store traffic of the
// f32 slab path, and the fp16 tensor-core GEMM runs ~3x faster than fp32 cuBLAS.
// Each thread converts two neighbouring elements and writes one packed 32-bit
// store (4 bytes), which keeps the stores fully coalesced.
inline const char* kDequantF16 = R"cuda(
__device__ __forceinline__ unsigned int pf_pack_half2(float a, float b) {
  unsigned short lo, hi;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(lo) : "f"(a));
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hi) : "f"(b));
  return (unsigned int)lo | ((unsigned int)hi << 16);   // out[2i]=a, out[2i+1]=b
}

// Q4_K: 144 B / 256 values.  e = 32*sub + l, sub = 2g + half.
extern "C" __global__ void dequant_q4k_f16(const unsigned char* __restrict__ blocks,
                                           unsigned int* __restrict__ out, long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const int t = threadIdx.x;                 // 0..127 -> elements 2t, 2t+1
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

// Q5_K: 176 B / 256 values, q in [0,31] (nibble + high bit), value = d*sc*q - dmin*mn.
extern "C" __global__ void dequant_q5k_f16(const unsigned char* __restrict__ blocks,
                                           unsigned int* __restrict__ out, long long n_blocks) {
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

// Q6_K: 210 B / 256 values.  Element e: n2 = e>>7, r = e&127, q = r>>5, l = r&31.
//   low nibble of ql[n2*64 + l + 32*(q&1)] (high nibble for q>=2) | 2 bits of
//   qh[n2*32 + l] at shift 2q, minus 32; scale = d * sc[n2*8 + (l/16) + 2q].
extern "C" __global__ void dequant_q6k_f16(const unsigned char* __restrict__ blocks,
                                           unsigned int* __restrict__ out, long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const int t = threadIdx.x;                 // 0..127 -> elements 2t, 2t+1
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

// Q8_0: 34 B / 32 values, one 2-byte store per element (tiny part of any model).
extern "C" __global__ void dequant_q8_0_f16(const unsigned char* __restrict__ blocks,
                                            unsigned short* __restrict__ out, long long n_blocks) {
  const long long b = blockIdx.x;
  if (b >= n_blocks) return;
  const unsigned char* p = blocks + b * 34;
  const float d = half_to_float(*(const unsigned short*)(p));
  const signed char* q = (const signed char*)(p + 2);
  const int l = threadIdx.x;
  out[b * 32 + l] = (unsigned short)f32_to_f16_bits(d * (float)q[l]);
}

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
)cuda";


inline const char* kAttnTiledPre = R"cuda(// ---- prefill attention: several query rows per block ------------------------
// attn_tiled launches one warp per (token, head), so every block re-reads the whole
// K/V history on its own: for a 1622-token 9B prefill that is ~42 GB of K/V traffic
// (8 layers) and measured 444 ms GPU time.  This variant puts 8 consecutive query
// rows of the same head in one block (8 warps): all warps walk the keys in lock
// step, so the K/V row a warp fetches is served from L1 to the other seven.  The
// f32 path also reads K/V with float4 (2 fully coalesced 512 B loads per warp per
// row) instead of 8 strided 4 B loads.  The math (dot order, online softmax) is
// identical to attn_tiled, so the numerics do not change.
extern "C" __global__ void attn_tiled_pre(const float* __restrict__ q,
                                          const float* __restrict__ Kf,
                                          const float* __restrict__ Vf,
                                          const unsigned char* __restrict__ Kq,
                                          const unsigned char* __restrict__ Vq,
                                          float* __restrict__ out,
                                          const int* __restrict__ pstart, int n_tokens,
                                          int n_head, int n_kv, int D, int max_seq,
                                          int kv_type) {
  const int h = blockIdx.y;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int i = blockIdx.x * (blockDim.x >> 5) + w;
  if (i >= n_tokens) return;
  const int qpos = *pstart + i;
  const int rep = n_head / n_kv;
  const int kvh = h / rep;
  const float scale = rsqrtf((float)D);
  const int EL = D / 32;
  const int e0 = lane * EL;
  const int grp = 32 / EL;
  const int qblk = lane / grp;
  const int qoff = (lane % grp) * EL;
  const int rowb = kv_row_bytes(kv_type, D);
  const long long row0 = (long long)kvh * max_seq;
  const float* qh = q + ((long long)i * n_head + h) * D + e0;
  float qr[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) qr[e] = (e < EL) ? qh[e] : 0.f;
  const bool f32_wide = (kv_type == 0 && EL == 8);
  float m = -1e30f, s = 0.f;
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
  for (int j = 0; j <= qpos; ++j) {
    float dot = 0.f;
    float4 v0, v1;
    if (f32_wide) {
      const float* kp = Kf + (row0 + j) * D + e0;
      const float4 k0 = *reinterpret_cast<const float4*>(kp);
      const float4 k1 = *reinterpret_cast<const float4*>(kp + 4);
      dot = qr[0] * k0.x + qr[1] * k0.y + qr[2] * k0.z + qr[3] * k0.w + qr[4] * k1.x +
            qr[5] * k1.y + qr[6] * k1.z + qr[7] * k1.w;
      const float* vp = Vf + (row0 + j) * D + e0;
      v0 = *reinterpret_cast<const float4*>(vp);
      v1 = *reinterpret_cast<const float4*>(vp + 4);
    } else if (kv_type == 0) {
      const float* kp = Kf + (row0 + j) * D + e0;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) dot += qr[e] * __ldg(kp + e);
    } else if (kv_type == 1) {
      const unsigned char* kp = Kq + (row0 + j) * rowb + qblk * 34;
      const float dq = half_to_float(*(const unsigned short*)kp);
      const signed char* qs = (const signed char*)(kp + 2) + qoff;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) dot += qr[e] * (float)qs[e];
      dot *= dq;
    } else {
      const unsigned char* kp = Kq + (row0 + j) * rowb + qblk * 18;
      const float dq = half_to_float(*(const unsigned short*)kp);
      const unsigned char* nb = kp + 2;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        if (e < EL) {
          const int idx = qoff + e;
          const unsigned char byte = nb[idx & 15];
          const int qv = (idx < 16) ? (int)(byte & 0xF) : (int)(byte >> 4);
          dot += qr[e] * (float)(qv - 8);
        }
      }
      dot *= dq;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
    dot *= scale;
    if (dot > m) {
      const float corr = expf(m - dot);
      s *= corr;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] *= corr;
      m = dot;
    }
    const float p = expf(dot - m);
    s += p;
    if (f32_wide) {
      acc[0] += p * v0.x; acc[1] += p * v0.y; acc[2] += p * v0.z; acc[3] += p * v0.w;
      acc[4] += p * v1.x; acc[5] += p * v1.y; acc[6] += p * v1.z; acc[7] += p * v1.w;
    } else if (kv_type == 0) {
      const float* vp = Vf + (row0 + j) * D + e0;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) acc[e] += p * __ldg(vp + e);
    } else if (kv_type == 1) {
      const unsigned char* vp = Vq + (row0 + j) * rowb + qblk * 34;
      const float dq = half_to_float(*(const unsigned short*)vp);
      const signed char* qs = (const signed char*)(vp + 2) + qoff;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < EL) acc[e] += p * ((float)qs[e] * dq);
    } else {
      const unsigned char* vp = Vq + (row0 + j) * rowb + qblk * 18;
      const float dq = half_to_float(*(const unsigned short*)vp);
      const unsigned char* nb = vp + 2;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        if (e < EL) {
          const int idx = qoff + e;
          const unsigned char byte = nb[idx & 15];
          const int qv = (idx < 16) ? (int)(byte & 0xF) : (int)(byte >> 4);
          acc[e] += p * ((float)(qv - 8) * dq);
        }
      }
    }
  }
  float* op = out + ((long long)i * n_head + h) * D + e0;
  const float inv = s > 0.f ? 1.f / s : 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e)
    if (e < EL) op[e] = acc[e] * inv;
}
)cuda";

inline const char* kGdDelta128Smem = R"cuda(
// gd_delta_rule_128 with the recurrent state held in shared memory across the whole
// token loop.  The reference kernel keeps S (128x128 f32 = 64 KiB per value head) in
// global memory and reads+writes it on every token: 32 heads x 128 KiB = 4 MiB of L2
// traffic per token, which made a 1622-token prefill spend 248 ms here (measured;
// 40% of all non-GEMM GPU time).  Staging S in shared memory turns that into zero
// global traffic inside the loop (one read at entry, one write at exit) and keeps
// every arithmetic operation in the same order, so results are bit-identical.
// Dynamic shared bytes = (128*128 + 128 + 128 + 4*128 + 128) * 4 = 69120 (opt-in).
extern "C" __global__ void gd_delta_rule_128_smem(const float* __restrict__ q,
                                                  const float* __restrict__ k,
                                                  const float* __restrict__ v,
                                                  const float* __restrict__ g,
                                                  const float* __restrict__ beta,
                                                  float* __restrict__ S, float* __restrict__ o,
                                                  int T, int n_vh, int n_kh, int KD, int VD,
                                                  int kh_mode) {
  constexpr int D = 128;
  constexpr int GRP = 4;
  constexpr int KPER = D / GRP;
  extern __shared__ float smem[];
  float* sS = smem;                    // [D][D] state
  float* sk = sS + D * D;              // [D]
  float* sq = sk + D;                  // [D]
  float* part = sq + D;                // [GRP][D]
  float* sdelta = part + GRP * D;      // [D]
  const int h = blockIdx.x;
  const int grp = threadIdx.x / D;
  const int j = threadIdx.x % D;
  const int rep = n_vh / n_kh;
  // q/k head -> value head broadcast.  GGUF checkpoints are written with the
  // llama.cpp convention (v_head h reads k_head h % n_kh, "block repeat"; see
  // ggml/src/ggml-cuda/gated_delta_net.cu `iq1 = fastmodulo(h_idx, neqk1)` and
  // [TAG_GGML_GDN_BCAST] in ggml.h).  HF safetensors checkpoints use
  // repeat_interleave (v_head h reads k_head h / rep).  kh_mode 0 = HF, 1 = GGUF.
  const int kh = kh_mode ? (h % n_kh) : (h / rep);
  float* Sh = S + (long long)h * D * D;
  for (int i = threadIdx.x; i < D * D; i += blockDim.x) sS[i] = Sh[i];
  __syncthreads();
  float reg[KPER];
  for (int t = 0; t < T; ++t) {
    if (grp == 0) {
      sk[j] = k[((long long)t * n_kh + kh) * D + j];
      sq[j] = q[((long long)t * n_kh + kh) * D + j];
    }
    __syncthreads();
    const float decay = expf(g[t * n_vh + h]);
    const float bh = beta[t * n_vh + h];
    float kv = 0.f;
    for (int i = 0; i < KPER; ++i) {
      const float s = sS[(grp * KPER + i) * D + j] * decay;
      reg[i] = s;
      kv += s * sk[grp * KPER + i];
    }
    part[grp * D + j] = kv;
    __syncthreads();
    if (grp == 0) {
      const float tot = part[j] + part[D + j] + part[2 * D + j] + part[3 * D + j];
      sdelta[j] = (v[((long long)t * n_vh + h) * D + j] - tot) * bh;
    }
    __syncthreads();
    const float delta = sdelta[j];
    float acc = 0.f;
    for (int i = 0; i < KPER; ++i) {
      const int ii = grp * KPER + i;
      const float s = reg[i] + sk[ii] * delta;
      sS[ii * D + j] = s;
      acc += s * sq[ii];
    }
    part[grp * D + j] = acc;
    __syncthreads();
    if (grp == 0) {
      o[((long long)t * n_vh + h) * D + j] =
          part[j] + part[D + j] + part[2 * D + j] + part[3 * D + j];
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < D * D; i += blockDim.x) Sh[i] = sS[i];
}
)cuda";

inline const char* kGdDelta128Reg = R"cuda(
// gd_delta_rule_128 with the recurrent state in *registers*.
// The reference kernel re-reads and re-writes the 64 KiB per-head state in global
// memory every token (4 MiB of L2 traffic per token across 32 heads) and a shared
// memory staging variant measured identical, so the real limit is the serial
// dependency chain, not bandwidth.  Here each thread keeps its own 32 state values
// (rows [grp*32, grp*32+32) of column j) in registers for the whole token loop and
// only the small cross-group partials go through shared memory, which brings the
// per-token critical path down to 4 barriers + ~96 FMA.  Arithmetic order is
// unchanged, so results are bit-identical to gd_delta_rule_128.
extern "C" __global__ void gd_delta_rule_128_reg(const float* __restrict__ q,
                                                 const float* __restrict__ k,
                                                 const float* __restrict__ v,
                                                 const float* __restrict__ g,
                                                 const float* __restrict__ beta,
                                                 float* __restrict__ S, float* __restrict__ o,
                                                 int T, int n_vh, int n_kh, int KD, int VD,
                                                 int kh_mode) {
  constexpr int D = 128;
  constexpr int GRP = 4;
  constexpr int KPER = D / GRP;
  __shared__ float sk[D];
  __shared__ float sq[D];
  __shared__ float part[GRP][D];
  __shared__ float sdelta[D];
  const int h = blockIdx.x;
  const int grp = threadIdx.x / D;
  const int j = threadIdx.x % D;
  const int rep = n_vh / n_kh;
  // q/k head -> value head broadcast.  GGUF checkpoints are written with the
  // llama.cpp convention (v_head h reads k_head h % n_kh, "block repeat"; see
  // ggml/src/ggml-cuda/gated_delta_net.cu `iq1 = fastmodulo(h_idx, neqk1)` and
  // [TAG_GGML_GDN_BCAST] in ggml.h).  HF safetensors checkpoints use
  // repeat_interleave (v_head h reads k_head h / rep).  kh_mode 0 = HF, 1 = GGUF.
  const int kh = kh_mode ? (h % n_kh) : (h / rep);
  float* Sh = S + (long long)h * D * D;
  float st[KPER];
#pragma unroll
  for (int i = 0; i < KPER; ++i) st[i] = Sh[(grp * KPER + i) * D + j];
  const long long gstep = n_vh, vstep = (long long)n_vh * D;
  const int krow = kh * D;
  const int orow = h * D;
  for (int t = 0; t < T; ++t) {
    // Issue all global loads for this token before the first barrier so their
    // latency overlaps with the previous token's tail.
    const float vv = v[(long long)t * vstep + orow + j];
    const float gg = g[(long long)t * gstep + h];
    const float bb = beta[(long long)t * gstep + h];
    if (grp == 0) {
      const float* kp = k + (long long)t * n_kh * D + krow;
      const float* qp = q + (long long)t * n_kh * D + krow;
      sk[j] = kp[j];
      sq[j] = qp[j];
    }
    __syncthreads();
    const float decay = expf(gg);
    const float bh = bb;
    float kv = 0.f;
#pragma unroll
    for (int i = 0; i < KPER; ++i) {
      st[i] *= decay;
      kv += st[i] * sk[grp * KPER + i];
    }
    part[grp][j] = kv;
    __syncthreads();
    if (grp == 0)
      sdelta[j] = (vv - (part[0][j] + part[1][j] + part[2][j] + part[3][j])) * bh;
    __syncthreads();
    const float delta = sdelta[j];
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < KPER; ++i) {
      const int ii = grp * KPER + i;
      st[i] += sk[ii] * delta;
      acc += st[i] * sq[ii];
    }
    part[grp][j] = acc;
    __syncthreads();
    if (grp == 0)
      o[(long long)t * vstep + orow + j] = part[0][j] + part[1][j] + part[2][j] + part[3][j];
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < KPER; ++i) Sh[(grp * KPER + i) * D + j] = st[i];
}
)cuda";

inline const char* kF16ToF32 = R"cuda(
// Widen an fp16 tile (as produced by the fp16-accumulate GEMM) into the f32
// activation buffer whose row stride is the full output width.
extern "C" __global__ void f16_to_f32_ld(const unsigned short* __restrict__ in,
                                         float* __restrict__ out, int rows, long long n,
                                         int out_stride) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long r = i / rows, c = i - r * rows;
  out[r * (long long)out_stride + c] = half_to_float(in[i]);
}
)cuda";

inline const char* kAttnTiledPreGqa = R"cuda(
// Prefill attention with the GQA group sharing the K/V stream inside one warp.
// attn_tiled_pre already reuses a K/V row across 8 query rows of the same head, but
// the 4 query heads of one KV head still run in separate blocks and each re-reads the
// whole key history (4x).  Here one warp owns one query row and *all* rep query heads
// of its KV head: the K/V row is loaded once per (row, key) and used by every head.
// Per-head arithmetic (dot order, online softmax, V accumulation) is unchanged, so the
// results are bit-identical to attn_tiled_pre.  f32 KV with head_dim == 256 only.
extern "C" __global__ void attn_tiled_pre_gqa(const float* __restrict__ q,
                                              const float* __restrict__ Kf,
                                              const float* __restrict__ Vf,
                                              float* __restrict__ out,
                                              const int* __restrict__ pstart, int n_tokens,
                                              int n_head, int n_kv, int D, int max_seq,
                                              int rep) {
  const int hkv = blockIdx.y;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int i = blockIdx.x * (blockDim.x >> 5) + w;
  if (i >= n_tokens) return;
  const int qpos = *pstart + i;
  const float scale = rsqrtf((float)D);
  const int e0 = lane * 8;
  const long long row0 = (long long)hkv * max_seq;
  float qr[4][8];
  float acc[4][8];
  float m[4], s[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    if (r < rep) {
      const float* qh = q + ((long long)i * n_head + hkv * rep + r) * D + e0;
      const float4 q0 = *reinterpret_cast<const float4*>(qh);
      const float4 q1 = *reinterpret_cast<const float4*>(qh + 4);
      qr[r][0] = q0.x; qr[r][1] = q0.y; qr[r][2] = q0.z; qr[r][3] = q0.w;
      qr[r][4] = q1.x; qr[r][5] = q1.y; qr[r][6] = q1.z; qr[r][7] = q1.w;
      m[r] = -1e30f;
      s[r] = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[r][e] = 0.f;
    }
  }
  for (int j = 0; j <= qpos; ++j) {
    const float* kp = Kf + (row0 + j) * D + e0;
    const float4 k0 = *reinterpret_cast<const float4*>(kp);
    const float4 k1 = *reinterpret_cast<const float4*>(kp + 4);
    const float* vp = Vf + (row0 + j) * D + e0;
    const float4 v0 = *reinterpret_cast<const float4*>(vp);
    const float4 v1 = *reinterpret_cast<const float4*>(vp + 4);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (r >= rep) break;
      float dot = qr[r][0] * k0.x + qr[r][1] * k0.y + qr[r][2] * k0.z + qr[r][3] * k0.w +
                  qr[r][4] * k1.x + qr[r][5] * k1.y + qr[r][6] * k1.z + qr[r][7] * k1.w;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
      dot *= scale;
      if (dot > m[r]) {
        const float corr = expf(m[r] - dot);
        s[r] *= corr;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[r][e] *= corr;
        m[r] = dot;
      }
      const float p = expf(dot - m[r]);
      s[r] += p;
      acc[r][0] += p * v0.x; acc[r][1] += p * v0.y; acc[r][2] += p * v0.z;
      acc[r][3] += p * v0.w; acc[r][4] += p * v1.x; acc[r][5] += p * v1.y;
      acc[r][6] += p * v1.z; acc[r][7] += p * v1.w;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    if (r >= rep) break;
    float* op = out + ((long long)i * n_head + hkv * rep + r) * D + e0;
    const float inv = s[r] > 0.f ? 1.f / s[r] : 0.f;
    op[0] = acc[r][0] * inv; op[1] = acc[r][1] * inv; op[2] = acc[r][2] * inv;
    op[3] = acc[r][3] * inv; op[4] = acc[r][4] * inv; op[5] = acc[r][5] * inv;
    op[6] = acc[r][6] * inv; op[7] = acc[r][7] * inv;
  }
}
)cuda";

// ---------------------------------------------------------------------------
// Generic architecture blocks (metadata/tensor driven; used by the extended
// per-layer dense path, harmless for every other model):
//   gelu_mul          GELU-tanh(gate)*up, the gemma-family MLP activation
//   scale_inplace     x *= s                  (layer_output_scale)
//   softcap_inplace   x = s*tanh(x/s)         (final logit softcapping)
//   attn_cached_cfg   causal KV attention with configurable logit scale and an
//                     optional sliding window
// ---------------------------------------------------------------------------
inline const char* kGeluMul = R"cuda(
extern "C" __global__ void gelu_mul(const float* __restrict__ gate,
                                    const float* __restrict__ up,
                                    float* __restrict__ out, long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    const float g = gate[i];
    const float c = 0.7978845608028654f;   // sqrt(2/pi)
    const float y = 0.5f * g * (1.0f + tanhf(c * (g + 0.044715f * g * g * g)));
    out[i] = y * up[i];
  }
}
)cuda";

inline const char* kScaleInplace = R"cuda(
extern "C" __global__ void scale_inplace(float* __restrict__ x, float s, long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) x[i] *= s;
}
)cuda";

inline const char* kSoftcapInplace = R"cuda(
extern "C" __global__ void softcap_inplace(float* __restrict__ x, float s, long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) x[i] = s * tanhf(x[i] / s);
}
)cuda";

inline const char* kAttnCfg = R"cuda(
extern "C" __global__ void attn_cached_cfg(const float* __restrict__ q,
                                           const float* __restrict__ Kc,
                                           const float* __restrict__ Vc,
                                           float* __restrict__ out,
                                           const int* __restrict__ pstart, int n_head, int n_kv,
                                           int D, int max_seq, float scale, int window) {
  extern __shared__ float sh[];
  const unsigned mask = 0xffffffffu;
  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int qpos = *pstart + i;
  const int rep = n_head / n_kv;
  const int kvh = h / rep;
  const float* qh = q + ((long long)i * n_head + h) * D;
  const float* Kh = Kc + (long long)kvh * max_seq * D;
  const float* Vh = Vc + (long long)kvh * max_seq * D;
  const int j0 = window > 0 ? (qpos - window + 1 > 0 ? qpos - window + 1 : 0) : 0;
  for (int j = j0 + lane; j <= qpos; j += 32) {
    const float* kh = Kh + (long long)j * D;
    float dot = 0.f;
    for (int d = 0; d < D; ++d) dot += qh[d] * kh[d];
    sh[j] = dot * scale;
  }
  __syncwarp();
  float lmax = -1e30f;
  for (int j = j0 + lane; j <= qpos; j += 32) lmax = fmaxf(lmax, sh[j]);
  for (int off = 16; off > 0; off >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(mask, lmax, off));
  float lsum = 0.f;
  for (int j = j0 + lane; j <= qpos; j += 32) {
    const float e = expf(sh[j] - lmax);
    sh[j] = e;
    lsum += e;
  }
  for (int off = 16; off > 0; off >>= 1) lsum += __shfl_xor_sync(mask, lsum, off);
  __syncwarp();
  const float inv = lsum > 0.f ? 1.f / lsum : 0.f;   // all -inf row -> zeros, not NaN
  for (int d = lane; d < D; d += 32) {
    float a = 0.f;
    for (int j = j0; j <= qpos; ++j) a += sh[j] * inv * Vh[(long long)j * D + d];
    out[((long long)i * n_head + h) * D + d] = a;
  }
}
)cuda";

}  // namespace kernels
