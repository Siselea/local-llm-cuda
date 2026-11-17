// M3 smoke test: NVRTC-compiled RMSNorm kernel vs CPU reference.
//
// Verifies the whole no-toolkit CUDA path: driver API load, NVRTC compile for
// sm_8x, module load, launch, memcpy back, numeric agreement.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "engine/cuda.h"

namespace {

const char* kRmsNorm = R"cuda(
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

void rmsnorm_cpu(const float* x, const float* w, float* y, int rows, int n, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<size_t>(r) * n;
    float* yr = y + static_cast<size_t>(r) * n;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += static_cast<double>(xr[i]) * xr[i];
    const float scale = static_cast<float>(1.0 / std::sqrt(sum / n + eps));
    for (int i = 0; i < n; ++i) yr[i] = xr[i] * scale * w[i];
  }
}

}  // namespace

int main() {
  try {
    cuda::init();
    const auto& d = cuda::device_info();
    std::printf("device     : %s  sm_%d%d  SMs=%d  vram=%.2f GiB\n", d.name.c_str(),
                d.cc_major, d.cc_minor, d.sm_count,
                static_cast<double>(d.mem_total) / (1 << 30));
    std::printf("driver     : %d  (runtime API not used)\n", d.driver_version);

    auto t0 = std::chrono::steady_clock::now();
    std::string ptx = cuda::compile_ptx(kRmsNorm, "rmsnorm.cu");
    auto t1 = std::chrono::steady_clock::now();
    std::printf("nvrtc      : %zu bytes PTX in %.1f ms\n", ptx.size(),
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    cuda::Module mod(ptx);
    void* fn = mod.kernel("rmsnorm");

    const int rows = 8, n = 4096;
    const float eps = 1e-6f;
    std::vector<float> x(static_cast<size_t>(rows) * n), w(n), y(static_cast<size_t>(rows) * n),
        ref(static_cast<size_t>(rows) * n);
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = static_cast<float>(std::sin(i * 0.001) * 1.7);
    for (int i = 0; i < n; ++i) w[i] = static_cast<float>(0.5 + 0.01 * (i % 37));
    rmsnorm_cpu(x.data(), w.data(), ref.data(), rows, n, eps);

    cuda::Buffer dx(x.size() * sizeof(float)), dw(n * sizeof(float)),
        dy(y.size() * sizeof(float));
    dx.upload(x.data(), x.size() * sizeof(float));
    dw.upload(w.data(), n * sizeof(float));

    int n_arg = n;
    float eps_arg = eps;
    void* kx = dx.data();
    void* kw = dw.data();
    void* ky = dy.data();
    void* args[] = {&kx, &kw, &ky, &n_arg, &eps_arg};
    cuda::launch(fn, rows, 1, 1, 256, 1, 1, 256 * sizeof(float), args);
    cuda::sync();
    dy.download(y.data(), y.size() * sizeof(float));

    double max_abs = 0, max_rel = 0;
    for (size_t i = 0; i < y.size(); ++i) {
      const double e = std::fabs(static_cast<double>(y[i]) - ref[i]);
      max_abs = std::max(max_abs, e);
      const double denom = std::max(std::fabs(static_cast<double>(ref[i])), 1e-6);
      max_rel = std::max(max_rel, e / denom);
    }
    std::printf("rmsnorm    : rows=%d n=%d  max_abs=%.3e  max_rel=%.3e  %s\n", rows, n, max_abs,
                max_rel, (max_abs < 1e-4 && max_rel < 1e-3) ? "PASS" : "FAIL");

    // timing: 200 launches + syncs, effective bandwidth over read x + read w + write y
    const int iters = 200;
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      cuda::launch(fn, rows, 1, 1, 256, 1, 1, 256 * sizeof(float), args);
    cuda::sync();
    auto t3 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t3 - t2).count() / iters;
    const double bytes = static_cast<double>(rows) * n * sizeof(float) * 3.0;
    std::printf("timing     : %.2f us/launch  ~%.1f GB/s effective\n", us, bytes / us / 1e3);

    return (max_abs < 1e-4 && max_rel < 1e-3) ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
