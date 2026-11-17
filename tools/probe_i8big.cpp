// Probe: cuBLAS int8 GEMM throughput (ground truth for INT8 tensor cores on this GPU).
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>
#include "engine/cuda.h"
#include "engine/cublas.h"

using cublasHandle_t = void*;
typedef int (*pf_create)(cublasHandle_t*);
typedef int (*pf_gemmex)(cublasHandle_t, int, int, int, int, int, const void*, const void*, int,
                         int, const void*, int, int, const void*, void*, int, int, int, int);
typedef int (*pf_setstream)(cublasHandle_t, void*);

static double timeit(std::function<void()> fn, int reps = 5) {
  fn(); cuda::sync();
  double best = 1e9;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    fn(); cuda::sync();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best = s < best ? s : best;
  }
  return best;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  cuda::init();
  cublas::init();
  HMODULE lib = LoadLibraryA("cublas64_12.dll");
  if (!lib) { printf("no cublas dll\n"); return 1; }
  auto create = (pf_create)GetProcAddress(lib, "cublasCreate_v2");
  auto gemmex = (pf_gemmex)GetProcAddress(lib, "cublasGemmEx");
  auto setstream = (pf_setstream)GetProcAddress(lib, "cublasSetStream");
  if (!create || !gemmex) { printf("missing symbols\n"); return 1; }
  cublasHandle_t h = nullptr;
  create(&h);
  if (setstream) setstream(h, nullptr);
  // enum values
  const int CUDA_R_8I = 3, CUDA_R_32I = 10;
  const int OP_N = 0, OP_T = 1;
  struct Sh { int m, n, k; const char* tag; };
  Sh shapes[] = {{512, 12288, 4096, "ffn gate/up"}, {1622, 12288, 4096, "ffn gate/up M1622"},
                 {1622, 4096, 12288, "ffn down M1622"}};
  for (auto& s : shapes) {
    cuda::Buffer A(static_cast<size_t>(s.m) * s.k);
    cuda::Buffer B(static_cast<size_t>(s.k) * s.n);
    cuda::Buffer C(static_cast<size_t>(s.m) * s.n * 4);
    int one = 1, zero = 0;
    const double macs = (double)s.m * s.n * s.k;
    double t = timeit([&] {
      gemmex(h, OP_T, OP_N, s.n, s.m, s.k, &one, B.data(), CUDA_R_8I, s.k, A.data(), CUDA_R_8I,
             s.k, &zero, C.data(), CUDA_R_32I, s.n, 72, -1);
    });
    printf("i8 %-16s M=%4d N=%5d K=%5d : %7.2f ms  %6.2f T-MAC/s\n", s.tag, s.m, s.n, s.k,
           t * 1e3, macs / t / 1e12);
  }
  return 0;
}
