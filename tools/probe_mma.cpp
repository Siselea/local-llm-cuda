// Microbenchmark v2: saturation test for int8 mma / f16 mma / dp4a with many chains.
#include <chrono>
#include <cstdio>
#include <vector>

#include "engine/cuda.h"

namespace {

const char* kSrc = R"cuda(
#define MMA_I8(c)                                                                    \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "                    \
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"               \
               : "+r"(c##0), "+r"(c##1), "+r"(c##2), "+r"(c##3)                      \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

extern "C" __global__ void bench_mma_i8(int* out, int iters) {
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

extern "C" __global__ void bench_mma_f16(float* out, int iters) {
  unsigned a0 = threadIdx.x + 1, a1 = threadIdx.x + 3, a2 = threadIdx.x + 5,
           a3 = threadIdx.x + 7;
  unsigned b0 = blockIdx.x + 1, b1 = blockIdx.x + 3;
  float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0, c20=0,c21=0,c22=0,c23=0,
        c30=0,c31=0,c32=0,c33=0, c40=0,c41=0,c42=0,c43=0, c50=0,c51=0,c52=0,c53=0,
        c60=0,c61=0,c62=0,c63=0, c70=0,c71=0,c72=0,c73=0;
  for (int i = 0; i < iters; ++i) {
    MMA_F16(c0); MMA_F16(c1); MMA_F16(c2); MMA_F16(c3);
    MMA_F16(c4); MMA_F16(c5); MMA_F16(c6); MMA_F16(c7);
  }
  float s = c00+c01+c02+c03+c10+c11+c12+c13+c20+c21+c22+c23+c30+c31+c32+c33+
            c40+c41+c42+c43+c50+c51+c52+c53+c60+c61+c62+c63+c70+c71+c72+c73;
  if (s == 123456789.f) out[0] = s;
}

extern "C" __global__ void bench_dp4a(int* out, int iters) {
  int a = threadIdx.x * 7 + 1, b = blockIdx.x * 13 + 3;
  int acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = i;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = __dp4a(a, b, acc[j]);
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += acc[i];
  if (s == 123456789) out[0] = s;
}

extern "C" __global__ void bench_mma_i8_16(int* out, int iters) {
  int a0 = threadIdx.x + 1, a1 = threadIdx.x + 3, a2 = threadIdx.x + 5, a3 = threadIdx.x + 7;
  int b0 = blockIdx.x + 1, b1 = blockIdx.x + 3;
  int c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0, c20=0,c21=0,c22=0,c23=0, c30=0,c31=0,c32=0,c33=0, c40=0,c41=0,c42=0,c43=0, c50=0,c51=0,c52=0,c53=0, c60=0,c61=0,c62=0,c63=0, c70=0,c71=0,c72=0,c73=0, c80=0,c81=0,c82=0,c83=0, c90=0,c91=0,c92=0,c93=0, c100=0,c101=0,c102=0,c103=0, c110=0,c111=0,c112=0,c113=0, c120=0,c121=0,c122=0,c123=0, c130=0,c131=0,c132=0,c133=0, c140=0,c141=0,c142=0,c143=0, c150=0,c151=0,c152=0,c153=0;
  for (int i = 0; i < iters; ++i) {
    MMA_I8(c0);
    MMA_I8(c1);
    MMA_I8(c2);
    MMA_I8(c3);
    MMA_I8(c4);
    MMA_I8(c5);
    MMA_I8(c6);
    MMA_I8(c7);
    MMA_I8(c8);
    MMA_I8(c9);
    MMA_I8(c10);
    MMA_I8(c11);
    MMA_I8(c12);
    MMA_I8(c13);
    MMA_I8(c14);
    MMA_I8(c15);
  }
  int s = c00+c01+c02+c03+c10+c11+c12+c13+c20+c21+c22+c23+c30+c31+c32+c33+c40+c41+c42+c43+c50+c51+c52+c53+c60+c61+c62+c63+c70+c71+c72+c73+c80+c81+c82+c83+c90+c91+c92+c93+c100+c101+c102+c103+c110+c111+c112+c113+c120+c121+c122+c123+c130+c131+c132+c133+c140+c141+c142+c143+c150+c151+c152+c153;
  if (s == 123456789) out[0] = s;
}

extern "C" __global__ void bench_mma_f16_16(float* out, int iters) {
  unsigned a0 = threadIdx.x + 1, a1 = threadIdx.x + 3, a2 = threadIdx.x + 5, a3 = threadIdx.x + 7;
  unsigned b0 = blockIdx.x + 1, b1 = blockIdx.x + 3;
  float f00=0,f01=0,f02=0,f03=0, f10=0,f11=0,f12=0,f13=0, f20=0,f21=0,f22=0,f23=0, f30=0,f31=0,f32=0,f33=0, f40=0,f41=0,f42=0,f43=0, f50=0,f51=0,f52=0,f53=0, f60=0,f61=0,f62=0,f63=0, f70=0,f71=0,f72=0,f73=0, f80=0,f81=0,f82=0,f83=0, f90=0,f91=0,f92=0,f93=0, f100=0,f101=0,f102=0,f103=0, f110=0,f111=0,f112=0,f113=0, f120=0,f121=0,f122=0,f123=0, f130=0,f131=0,f132=0,f133=0, f140=0,f141=0,f142=0,f143=0, f150=0,f151=0,f152=0,f153=0;
  for (int i = 0; i < iters; ++i) {
    MMA_F16(f0);
    MMA_F16(f1);
    MMA_F16(f2);
    MMA_F16(f3);
    MMA_F16(f4);
    MMA_F16(f5);
    MMA_F16(f6);
    MMA_F16(f7);
    MMA_F16(f8);
    MMA_F16(f9);
    MMA_F16(f10);
    MMA_F16(f11);
    MMA_F16(f12);
    MMA_F16(f13);
    MMA_F16(f14);
    MMA_F16(f15);
  }
  float s = f00+f01+f02+f03+f10+f11+f12+f13+f20+f21+f22+f23+f30+f31+f32+f33+f40+f41+f42+f43+f50+f51+f52+f53+f60+f61+f62+f63+f70+f71+f72+f73+f80+f81+f82+f83+f90+f91+f92+f93+f100+f101+f102+f103+f110+f111+f112+f113+f120+f121+f122+f123+f130+f131+f132+f133+f140+f141+f142+f143+f150+f151+f152+f153;
  if (s == 123456789.f) out[0] = s;
}
)cuda";

template <typename Fn>
double bench(Fn launch, int reps = 5) {
  launch();
  cuda::sync();
  double best = 1e9;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    launch();
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
  const auto& di = cuda::device_info();
  const int blocks = di.sm_count * 8;
  const int threads = 256;
  const int iters = 20000;
  std::printf("gpu: %s sm=%d blocks=%d\n", di.name.c_str(), di.sm_count, blocks);
  cuda::Module mod(cuda::compile_ptx(kSrc, "probe.cu"));
  cuda::Buffer obuf(4096);
  void* p = obuf.data();
  int it = iters;

  {
    void* fn = mod.kernel("bench_mma_i8");
    void* args[] = {&p, &it};
    const double s = bench([&] { cuda::launch(fn, blocks, 1, 1, threads, 1, 1, 0, args); });
    const double warps = blocks * (threads / 32.0);
    const double mma = warps * iters * 8;
    std::printf("mma int8 m16n8k32 x8         : %8.3f ms  %8.2f G-mma/s  %8.2f T-MAC/s\n",
                s * 1e3, mma / s / 1e9, mma * 512 / s / 1e12);
  }
  {
    void* fn = mod.kernel("bench_mma_f16");
    void* args[] = {&p, &it};
    const double s = bench([&] { cuda::launch(fn, blocks, 1, 1, threads, 1, 1, 0, args); });
    const double warps = blocks * (threads / 32.0);
    const double mma = warps * iters * 8;
    std::printf("mma f16 m16n8k16 x8          : %8.3f ms  %8.2f G-mma/s  %8.2f T-MAC/s\n",
                s * 1e3, mma / s / 1e9, mma * 256 / s / 1e12);
  }
  {
    void* fn = mod.kernel("bench_dp4a");
    void* args[] = {&p, &it};
    const double s = bench([&] { cuda::launch(fn, blocks, 1, 1, threads, 1, 1, 0, args); });
    const double th = blocks * static_cast<double>(threads);
    const double dp = th * iters * 8;
    std::printf("dp4a x8                      : %8.3f ms  %8.2f G-dp4a/s %8.2f T-MAC/s\n",
                s * 1e3, dp / s / 1e9, dp * 4 / s / 1e12);
  }

  {
    void* fn = mod.kernel("bench_mma_i8_16");
    void* args[] = {&p, &it};
    const double s = bench([&] { cuda::launch(fn, blocks, 1, 1, threads, 1, 1, 0, args); });
    const double warps = blocks * (threads / 32.0);
    const double mma = warps * iters * 16;
    std::printf("mma int8 x16                 : %8.3f ms  %8.2f G-mma/s  %8.2f T-MAC/s\n",
                s * 1e3, mma / s / 1e9, mma * 512 / s / 1e12);
  }
  {
    void* fn = mod.kernel("bench_mma_f16_16");
    void* args[] = {&p, &it};
    const double s = bench([&] { cuda::launch(fn, blocks, 1, 1, threads, 1, 1, 0, args); });
    const double warps = blocks * (threads / 32.0);
    const double mma = warps * iters * 16;
    std::printf("mma f16 x16                  : %8.3f ms  %8.2f G-mma/s  %8.2f T-MAC/s\n",
                s * 1e3, mma / s / 1e9, mma * 256 / s / 1e12);
  }
  return 0;
}
