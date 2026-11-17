// Probe: cuBLAS f32/f16 GEMM throughput at the 9B prefill shapes + reliable int8 mma rate.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "engine/cublas.h"
#include "engine/cuda.h"

namespace {

const char* kMmaSrc = R"cuda(
#define MMA_I8(c)                                                                    \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "                    \
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"               \
               : "+r"(c##0), "+r"(c##1), "+r"(c##2), "+r"(c##3)                      \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

extern "C" __global__ void mma_i8(int* out, int iters) {
  int a0 = threadIdx.x + 1, a1 = threadIdx.x + 3, a2 = threadIdx.x + 5, a3 = threadIdx.x + 7;
  int b0 = blockIdx.x + 1, b1 = blockIdx.x + 3;
  int c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0, c20=0,c21=0,c22=0,c23=0,
      c30=0,c31=0,c32=0,c33=0, c40=0,c41=0,c42=0,c43=0, c50=0,c51=0,c52=0,c53=0,
      c60=0,c61=0,c62=0,c63=0, c70=0,c71=0,c72=0,c73=0;
  for (int i = 0; i < iters; ++i) {
    MMA_I8(c0); MMA_I8(c1); MMA_I8(c2); MMA_I8(c3);
    MMA_I8(c4); MMA_I8(c5); MMA_I8(c6); MMA_I8(c7);
  }
  int s = c00+c01+c02+c03+c10+c11+c12+c13+c20+c21+c22+c23+c30+c31+c32+c33+
          c40+c41+c42+c43+c50+c51+c52+c53+c60+c61+c62+c63+c70+c71+c72+c73;
  if (s == 123456789) out[0] = s;
}

#define MMA_F16(c)                                                                   \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                  \
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"               \
               : "+f"(c##0), "+f"(c##1), "+f"(c##2), "+f"(c##3)                      \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

extern "C" __global__ void mma_f16(float* out, int iters) {
  unsigned a0 = threadIdx.x + 1, a1 = threadIdx.x + 3, a2 = threadIdx.x + 5, a3 = threadIdx.x + 7;
  unsigned b0 = blockIdx.x + 1, b1 = blockIdx.x + 3;
  float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0, c20=0,c21=0,c22=0,c23=0,
        c30=0,c31=0,c32=0,c33=0;
  for (int i = 0; i < iters; ++i) {
    MMA_F16(c0); MMA_F16(c1); MMA_F16(c2); MMA_F16(c3);
  }
  float s = c00+c01+c02+c03+c10+c11+c12+c13+c20+c21+c22+c23+c30+c31+c32+c33;
  if (s == 123456789.f) out[0] = s;
}
)cuda";

double timeit(std::function<void()> fn, int reps = 5) {
  fn();
  cuda::sync();
  double best = 1e9;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    cuda::sync();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best = s < best ? s : best;
  }
  return best;
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  cuda::init();
  cublas::init();
  const auto& di = cuda::device_info();
  std::printf("gpu %s sm=%d\n", di.name.c_str(), di.sm_count);

  // shapes: (M, N, K) for the 9B prefill at M=512
  struct Sh { int m, n, k; const char* tag; };
  Sh shapes[] = {{512, 12288, 4096, "ffn gate/up"}, {512, 4096, 12288, "ffn down"},
                 {512, 8192, 4096, "attn q / qkv"}, {512, 4096, 4096, "proj"},
                 {128, 12288, 4096, "ffn gate/up m128"}};
  for (auto& s : shapes) {
    cuda::Buffer a(static_cast<size_t>(s.m) * s.k * 4);
    cuda::Buffer b(static_cast<size_t>(s.n) * s.k * 4);
    cuda::Buffer c(static_cast<size_t>(s.m) * s.n * 4);
    std::vector<float> ha(static_cast<size_t>(s.m) * s.k, 0.01f), hb(static_cast<size_t>(s.n) * s.k, 0.02f);
    a.upload(ha.data(), ha.size() * 4);
    b.upload(hb.data(), hb.size() * 4);
    const double macs = static_cast<double>(s.m) * s.n * s.k;
    const double t = timeit([&] {
      cublas::gemm_f32_nt(s.m, s.n, s.k, static_cast<const float*>(a.data()),
                          static_cast<const float*>(b.data()), static_cast<float*>(c.data()));
    });
    std::printf("f32  %-18s M=%4d N=%5d K=%5d : %7.2f ms  %6.2f T-MAC/s\n", s.tag, s.m, s.n, s.k,
                t * 1e3, macs / t / 1e12);
  }
  // f16 NT path (half A, half B, f32 C)
  for (auto& s : shapes) {
    cuda::Buffer a(static_cast<size_t>(s.m) * s.k * 2);
    cuda::Buffer b(static_cast<size_t>(s.n) * s.k * 2);
    cuda::Buffer c(static_cast<size_t>(s.m) * s.n * 4);
    std::vector<unsigned short> ha(static_cast<size_t>(s.m) * s.k, 0x2000), hb(static_cast<size_t>(s.n) * s.k, 0x2400);
    a.upload(ha.data(), ha.size() * 2);
    b.upload(hb.data(), hb.size() * 2);
    const double macs = static_cast<double>(s.m) * s.n * s.k;
    const double t = timeit([&] {
      cublas::gemm_f16_nt_ld(s.m, s.n, s.k, a.data(), b.data(), static_cast<float*>(c.data()), s.n);
    });
    std::printf("f16nt %-17s M=%4d N=%5d K=%5d : %7.2f ms  %6.2f T-MAC/s\n", s.tag, s.m, s.n, s.k,
                t * 1e3, macs / t / 1e12);
  }
  // bf16 NN (B [k,n] row major) as a throughput reference
  for (auto& s : shapes) {
    cuda::Buffer a(static_cast<size_t>(s.m) * s.k * 2);
    cuda::Buffer b(static_cast<size_t>(s.k) * s.n * 2);
    cuda::Buffer c(static_cast<size_t>(s.m) * s.n * 2);
    const double macs = static_cast<double>(s.m) * s.n * s.k;
    const double t = timeit([&] {
      cublas::gemm_bf16(s.m, s.n, s.k, a.data(), b.data(), c.data());
    });
    std::printf("bf16 %-17s M=%4d N=%5d K=%5d : %7.2f ms  %6.2f T-MAC/s\n", s.tag, s.m, s.n, s.k,
                t * 1e3, macs / t / 1e12);
  }

  // ---- int8 mma saturation (large, 16 blocks/SM, 4/8 chains) ----
  {
    cuda::Module mod(cuda::compile_ptx(kMmaSrc, "mma.cu"));
    cuda::Buffer ob(4096);
    void* p = ob.data();
    for (int iters : {2000, 8000}) {
      for (int bpsm : {4, 8, 16}) {
        int it = iters;
        void* args[] = {&p, &it};
        void* fn = mod.kernel("mma_i8");
        const int blocks = di.sm_count * bpsm;
        const double t = timeit([&] { cuda::launch(fn, blocks, 1, 1, 256, 1, 1, 0, args); }, 3);
        const double warps = blocks * 8.0;
        const double mma = warps * iters * 8;
        std::printf("mma_i8  iters=%5d bpsm=%2d : %7.2f ms  %7.2f G-mma/s %6.2f T-MAC/s\n", iters,
                    bpsm, t * 1e3, mma / t / 1e9, mma * 512 / t / 1e12);
      }
    }
    for (int bpsm : {4, 16}) {
      int it = 8000;
      void* args[] = {&p, &it};
      void* fn = mod.kernel("mma_f16");
      const int blocks = di.sm_count * bpsm;
      const double t = timeit([&] { cuda::launch(fn, blocks, 1, 1, 256, 1, 1, 0, args); }, 3);
      const double warps = blocks * 8.0;
      const double mma = warps * it * 4;
      std::printf("mma_f16 iters= 8000 bpsm=%2d : %7.2f ms  %7.2f G-mma/s %6.2f T-MAC/s\n", bpsm,
                  t * 1e3, mma / t / 1e9, mma * 256 / t / 1e12);
    }
  }
  return 0;
}
