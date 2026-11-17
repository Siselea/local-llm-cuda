// Microbenchmark: dp4a / int2float / f32 FMA peak on this GPU + int8 GEMM roofline.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "engine/cuda.h"

namespace {

const char* kSrc = R"cuda(
extern "C" __global__ void bench_dp4a(int* out, int iters) {
  int a = threadIdx.x * 7 + 1, b = blockIdx.x * 13 + 3;
  int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
  for (int i = 0; i < iters; ++i) {
    acc0 = __dp4a(a, b, acc0);
    acc1 = __dp4a(a, b, acc1);
    acc2 = __dp4a(a, b, acc2);
    acc3 = __dp4a(a, b, acc3);
    acc4 = __dp4a(a, b, acc4);
    acc5 = __dp4a(a, b, acc5);
    acc6 = __dp4a(a, b, acc6);
    acc7 = __dp4a(a, b, acc7);
  }
  int s = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
  if (s == 123456789) out[0] = s;
}

extern "C" __global__ void bench_cvt(float* out, int iters) {
  int a0 = threadIdx.x + 1, a1 = blockIdx.x + 2;
  float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f, f4 = 0.f, f5 = 0.f, f6 = 0.f, f7 = 0.f;
  for (int i = 0; i < iters; ++i) {
    f0 += __int2float_rn(a0 + 0); f1 += __int2float_rn(a1 + 1);
    f2 += __int2float_rn(a0 + 2); f3 += __int2float_rn(a1 + 3);
    f4 += __int2float_rn(a0 + 4); f5 += __int2float_rn(a1 + 5);
    f6 += __int2float_rn(a0 + 6); f7 += __int2float_rn(a1 + 7);
  }
  float s = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7;
  if (s == 123456789.f) out[0] = s;
}

extern "C" __global__ void bench_fma(float* out, int iters) {
  float a = threadIdx.x * 1e-6f + 0.5f, b = blockIdx.x * 1e-6f + 1.5f;
  float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f, f4 = 0.f, f5 = 0.f, f6 = 0.f, f7 = 0.f;
  for (int i = 0; i < iters; ++i) {
    f0 = fmaf(a, b, f0); f1 = fmaf(a, b, f1); f2 = fmaf(a, b, f2); f3 = fmaf(a, b, f3);
    f4 = fmaf(a, b, f4); f5 = fmaf(a, b, f5); f6 = fmaf(a, b, f6); f7 = fmaf(a, b, f7);
  }
  float s = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7;
  if (s == 123456789.f) out[0] = s;
}

// int add (int pipe) for the magic-number conversion comparison
extern "C" __global__ void bench_iadd(int* out, int iters) {
  int a = threadIdx.x + 1, b = blockIdx.x + 2;
  int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
  for (int i = 0; i < iters; ++i) {
    acc0 += a; acc1 += b; acc2 += a; acc3 += b;
    acc4 += a; acc5 += b; acc6 += a; acc7 += b;
  }
  int s = acc0+acc1+acc2+acc3+acc4+acc5+acc6+acc7;
  if (s == 123456789) out[0] = s;
}

extern "C" __global__ void bench_lds(int* out, int iters) {
  __shared__ int s[256];
  s[threadIdx.x] = threadIdx.x;
  __syncthreads();
  int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
  int idx = threadIdx.x;
  for (int i = 0; i < iters; ++i) {
    acc0 += s[(idx + 0) & 255];
    acc1 += s[(idx + 1) & 255];
    acc2 += s[(idx + 2) & 255];
    acc3 += s[(idx + 3) & 255];
    idx += 4;
  }
  int sum = acc0 + acc1 + acc2 + acc3;
  if (sum == 123456789) out[0] = sum;
}
)cuda";

struct Bench {
  const char* name;
  const char* kern;
  double ops_per_iter_thread;   // "ops" counted per thread per loop iteration
  bool is_float;
};

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  cuda::init();
  const auto& di = cuda::device_info();
  std::printf("gpu: %s cc=%d.%d sm=%d\n", di.name.c_str(), di.cc_major, di.cc_minor,
              di.sm_count);
  cuda::Module mod(cuda::compile_ptx(kSrc, "probe.cu"));

  const int iters = 20000;
  const int threads = 256;
  const int blocks = di.sm_count * 8;
  const double iters_d = iters * 8.0;

  Bench benches[] = {
      {"dp4a (4 MAC/instr)", "bench_dp4a", iters_d, false},
      {"int2float cvt", "bench_cvt", iters_d, true},
      {"f32 fma", "bench_fma", iters_d, true},
      {"int add", "bench_iadd", iters_d, false},
      {"shared ld (32b)", "bench_lds", iters_d, false},
  };
  int* dzero = nullptr;
  cuda::Buffer obuf(4096);
  for (const auto& b : benches) {
    void* fn = mod.kernel(b.kern);
    void* p = obuf.data();
    int it = iters;
    void* args[] = {&p, &it};
    // warmup
    cuda::launch(fn, blocks, 1, 1, threads, 1, 1, 0, args);
    cuda::sync();
    std::vector<double> samples;
    for (int rep = 0; rep < 3; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      cuda::launch(fn, blocks, 1, 1, threads, 1, 1, 0, args);
      cuda::sync();
      const double s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      samples.push_back(s);
    }
    double best = samples[0];
    for (double s : samples) best = s < best ? s : best;
    const double total_ops = static_cast<double>(blocks) * threads * b.ops_per_iter_thread;
    const double per_s = total_ops / best;
    std::printf("%-22s : %8.3f ms  %8.2f G-instr/s  %8.2f T-MAC/s\n", b.name, best * 1e3,
                per_s / 1e9, (b.name[0] == 'd') ? per_s * 4 / 1e12 : 0.0);
  }
  (void)dzero;
  // memory bandwidth: read a 512 MB buffer
  {
    const size_t n = 128ull << 20;   // 512 MB of float
    cuda::Buffer buf(n * 4);
    const char* src = R"cuda(
extern "C" __global__ void bench_read(const float* __restrict__ p, float* out, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n; i += stride) acc += p[i];
  if (acc == 123456789.f) out[0] = acc;
}
)cuda";
    cuda::Module m2(cuda::compile_ptx(src, "bw.cu"));
    void* fn = m2.kernel("bench_read");
    void* pp = buf.data();
    void* po = obuf.data();
    long long nn = static_cast<long long>(n);
    void* a[] = {&pp, &po, &nn};
    cuda::launch(fn, blocks, 1, 1, 256, 1, 1, 0, a);
    cuda::sync();
    double best = 1e9;
    for (int rep = 0; rep < 3; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      cuda::launch(fn, blocks, 1, 1, 256, 1, 1, 0, a);
      cuda::sync();
      const double s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      best = s < best ? s : best;
    }
    std::printf("hbm read               : %8.3f ms  %8.2f GB/s\n", best * 1e3,
                (n * 4.0) / best / 1e9);
  }
  return 0;
}
