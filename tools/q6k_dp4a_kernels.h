// Scratch kernels for the Q6_K dp4a experiment (moved into include/engine/kernels.h
// once the winner is picked).  Compiled through NVRTC by tools/bench_q6k_dp4a.cpp.
#pragma once

namespace q6kdp4a {

// One warp per row-slice.  lane = (n2, p, g):
//   n2 = 128-element half, p = chunk pair (A/C when 0, B/D when 1), g = 4-element
//   group.  Each lane owns 4 consecutive elements of two chunks (x +0 / +64 within the
//   half), i.e. 8 elements per superblock, and builds its two int8 weight words from
//   one shared ql word (low nibbles = first chunk, high nibbles = second) plus one
//   shared qh word (bit pairs 0/2 for p = 0, 4/6 for p = 1).
// The 6-bit weight is folded to signed int8 as q-32 = ((q+96)^0x80) per byte, which is
// one add + one xor on the packed word (all intermediate bytes stay < 256, so the add
// never carries across bytes).
// Returns the full-warp sum over superblocks [sb0, sb1) on lane 0.
inline const char* kQ6Kdp4a = R"cuda(
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

// One row per warp.
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

// Two warps (then four) cooperate on one row: each takes a contiguous slice of the
// superblock range, partials are reduced through shared memory, no atomics/zero-fill.
extern "C" __global__ void gemv_q6k_dp4a_sk2(const unsigned char* __restrict__ W,
                                             const signed char* __restrict__ qx,
                                             const float* __restrict__ dx, float* __restrict__ y,
                                             int n_in, int n_out) {
  __shared__ float part[16];
  const int warps = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int split = 2;
  const int row0 = blockIdx.x * (warps / split) + (warp / split);
  const int ps = warp % split;
  const int sb_count = n_in >> 8;
  const int half = sb_count / split;
  const int row = row0 < n_out ? row0 : n_out - 1;
  const float acc = q6k_dp4a_range(W, qx, dx, n_in, row, ps * half, (ps + 1) * half);
  if (lane == 0) part[warp] = acc;
  __syncthreads();
  const int nr = warps / split;
  if (threadIdx.x < nr) {
    float s = 0.f;
    for (int j = 0; j < split; ++j) s += part[threadIdx.x * split + j];
    const int r = blockIdx.x * nr + threadIdx.x;
    if (r < n_out) y[r] = s;
  }
}

extern "C" __global__ void gemv_q6k_dp4a_sk4(const unsigned char* __restrict__ W,
                                             const signed char* __restrict__ qx,
                                             const float* __restrict__ dx, float* __restrict__ y,
                                             int n_in, int n_out) {
  __shared__ float part[16];
  const int warps = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int split = 4;
  const int row0 = blockIdx.x * (warps / split) + (warp / split);
  const int ps = warp % split;
  const int sb_count = n_in >> 8;
  const int quarter = sb_count / split;
  const int row = row0 < n_out ? row0 : n_out - 1;
  const float acc = q6k_dp4a_range(W, qx, dx, n_in, row, ps * quarter, (ps + 1) * quarter);
  if (lane == 0) part[warp] = acc;
  __syncthreads();
  const int nr = warps / split;
  if (threadIdx.x < nr) {
    float s = 0.f;
    for (int j = 0; j < split; ++j) s += part[threadIdx.x * split + j];
    const int r = blockIdx.x * nr + threadIdx.x;
    if (r < n_out) y[r] = s;
  }
}

// Same as gemv_q6k_dp4a but with an occupancy hint (min 8 blocks/SM => <= 32 regs).
extern "C" __global__ void __launch_bounds__(256, 8)
gemv_q6k_dp4a_lb8(const unsigned char* __restrict__ W, const signed char* __restrict__ qx,
                  const float* __restrict__ dx, float* __restrict__ y, int n_in, int n_out) {
  const int warps = blockDim.x >> 5;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= n_out) return;
  const float acc = q6k_dp4a_range(W, qx, dx, n_in, row, 0, n_in >> 8);
  if ((threadIdx.x & 31) == 0) y[row] = acc;
}

extern "C" __global__ void __launch_bounds__(256, 6)
gemv_q6k_dp4a_lb6(const unsigned char* __restrict__ W, const signed char* __restrict__ qx,
                  const float* __restrict__ dx, float* __restrict__ y, int n_in, int n_out) {
  const int warps = blockDim.x >> 5;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= n_out) return;
  const float acc = q6k_dp4a_range(W, qx, dx, n_in, row, 0, n_in >> 8);
  if ((threadIdx.x & 31) == 0) y[row] = acc;
}

// Generic split-K: `split` warps cooperate on one row, each taking a contiguous slice of
// the superblock range; partials are reduced through shared memory (no atomics, no
// zero-fill).  Caller must ensure sb_count % split == 0.  grid = ceil(n_out / (warps/split)).
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

// Pure streaming read probe: 16 B per lane, used to establish the HBM ceiling.
extern "C" __global__ void read_probe(const int4* __restrict__ p, int* __restrict__ out,
                                      long long n4) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;
  int acc = 0;
  for (; i < n4; i += stride) {
    const int4 v = p[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 0x7fffffff) out[0] = acc;
}
)cuda";

}  // namespace q6kdp4a
