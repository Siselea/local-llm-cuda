// Candidate gated-delta-rule decode kernels (benchmark scratch; promoted into
// include/engine/kernels.h once proven).
#pragma once

namespace gdk {

// v2: same math as gd_delta_rule but state layout is [n_vh][KD][VD] so the inner
// loop over the key dim reads the value dim contiguously (coalesced).  Two passes
// over the row (second pass usually hits L1).
inline const char* kGdDeltaV2 = R"cuda(
extern "C" __global__ void gd_delta_v2(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       const float* __restrict__ g,
                                       const float* __restrict__ beta,
                                       float* __restrict__ S, float* __restrict__ o, int T,
                                       int n_vh, int n_kh, int KD, int VD) {
  extern __shared__ float sh[];
  float* sk = sh;          // [KD]
  float* sq = sh + KD;     // [KD]
  const int h = blockIdx.x;
  const int j = threadIdx.x;
  if (j >= VD) return;
  const int rep = n_vh / n_kh;
  const int kh = h / rep;
  float* Sh = S + (long long)h * KD * VD;
  for (int t = 0; t < T; ++t) {
    if (j < KD) {
      sk[j] = k[((long long)t * n_kh + kh) * KD + j];
      sq[j] = q[((long long)t * n_kh + kh) * KD + j];
    }
    __syncthreads();
    const float decay = expf(g[t * n_vh + h]);
    const float bh = beta[t * n_vh + h];
    float kv = 0.f;
    for (int i = 0; i < KD; ++i) {
      const float s = Sh[i * VD + j] * decay;
      kv += s * sk[i];
    }
    const float delta = (v[((long long)t * n_vh + h) * VD + j] - kv) * bh;
    float acc = 0.f;
    for (int i = 0; i < KD; ++i) {
      const float s = Sh[i * VD + j] * decay + sk[i] * delta;
      Sh[i * VD + j] = s;
      acc += s * sq[i];
    }
    o[((long long)t * n_vh + h) * VD + j] = acc;
    __syncthreads();
  }
}
)cuda";

// v3: register tiled, specialised for KD=VD=128, 4 key-dim groups per block.
// Layout [n_vh][KD][VD].  Thread (grp,j) keeps its 32 state values in registers,
// so the state is read once and written once (2 MB read + 2 MB write / launch).
inline const char* kGdDeltaV3_128 = R"cuda(
extern "C" __global__ void gd_delta_v3_128(const float* __restrict__ q,
                                           const float* __restrict__ k,
                                           const float* __restrict__ v,
                                           const float* __restrict__ g,
                                           const float* __restrict__ beta,
                                           float* __restrict__ S, float* __restrict__ o, int T,
                                           int n_vh, int n_kh, int KD, int VD) {
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
  const int kh = h / rep;
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

}  // namespace gdk
