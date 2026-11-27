#!/usr/bin/env python3
"""Add a vectorised dequant variant to the fused-GEMM prototype and bench it.

The prototype showed the fused GEMM is numerically right but the naive version is
50x slower: at M-tile 16 every weight element is dequantised 32x.  It also showed the
dequant kernel itself is only ~120 GB/s effective (0.287 ms for an 11.5 MB slab), i.e.
the dequant *compute* is a large part of the ~130 ms per-forward fixed cost.  A cheap,
low-risk optimisation is therefore a faster dequant kernel, which helps the product path
today.  This variant does 8 halves per thread with 16-byte stores (32 threads/block,
one Q5_K super-block per warp quarter) and is measured against the product kernel.
"""
p = 'tests/fused_gemm_proto.h'
s = open(p, encoding='utf-8').read()

# 1) add the vectorised dequant kernel inside the NVRTC source
anchor = '''// ---- fused kernel: out[m][n] = sum_k x[m][k] * dequant(w[n][k]) ------------------'''
assert s.count(anchor) == 1
v2 = '''// ---- variant: vectorised dequant, 8 halves per thread, 16 B stores --------------
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

''' + anchor
s = s.replace(anchor, v2)

# 2) bench it next to the product-style dequant
old = '''  const double dq_ms = time_ms(dq_fn, (int)nblk_total, 1, 1, 128, 1, 1, 0, dq_args_real, 5);
'''
new = '''  const double dq_ms = time_ms(dq_fn, (int)nblk_total, 1, 1, 128, 1, 1, 0, dq_args_real, 5);
  const double dq_v2_ms =
      dq_v2_fn ? time_ms(dq_v2_fn, (int)nblk_total, 1, 1, 32, 1, 1, 0, dq_args_real, 5)
               : -1.0;
'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''  std::printf(
      "  [%s] baseline(dequant+cuBLAS) %8.3f ms = dq %6.3f + gemm %6.3f | fused %8.3f ms | "
      "speedup %5.2fx | rel_l2 %.2e max|diff| %.2e\\n",
      s.label, dq_ms + gemm_ms, dq_ms, gemm_ms, fused_ms, (dq_ms + gemm_ms) / fused_ms, rel, mx);'''
new = '''  std::printf(
      "  [%s] baseline(dequant+cuBLAS) %8.3f ms = dq %6.3f + gemm %6.3f"
      " | dq_v2 %6.3f ms (%.2fx) | fused %8.3f ms | rel_l2 %.2e max|diff| %.2e\\n",
      s.label, dq_ms + gemm_ms, dq_ms, gemm_ms, dq_v2_ms,
      dq_v2_ms > 0 ? dq_ms / dq_v2_ms : 0.0, fused_ms, rel, mx);'''
assert s.count(old) == 1
s = s.replace(old, new)

# 3) pass the v2 kernel handle through
s = s.replace('inline void bench_one(const Shape& s, void* dq_fn, void* fused_fn) {',
              'inline void bench_one(const Shape& s, void* dq_fn, void* dq_v2_fn, void* fused_fn) {')
s = s.replace('''  void* dq_fn = mod.kernel("proto_dequant_q5k_f16");''',
              '''  void* dq_fn = mod.kernel("proto_dequant_q5k_f16");
  void* dq_v2_fn = mod.kernel("proto_dequant_q5k_f16_v2");''')
s = s.replace('for (const auto& s : shapes) bench_one(s, dq_fn, fused_fn);',
              'for (const auto& s : shapes) bench_one(s, dq_fn, dq_v2_fn, fused_fn);')

open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('vectorised dequant variant added to the prototype')
