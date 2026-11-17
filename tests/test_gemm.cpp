// M3: cuBLAS GEMM (f32 + bf16) correctness and throughput on this GPU.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "engine/cublas.h"
#include "engine/cuda.h"

namespace {

uint16_t f32_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t lsb = (x >> 16) & 1u;
  x += 0x7fffu + lsb;  // round to nearest even
  return static_cast<uint16_t>(x >> 16);
}

float bf16_to_f32(uint16_t h) {
  uint32_t x = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

void gemm_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
              int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0;
      for (int p = 0; p < k; ++p) acc += static_cast<double>(A[static_cast<size_t>(i) * k + p]) *
                                        B[static_cast<size_t>(p) * n + j];
      C[static_cast<size_t>(i) * n + j] = static_cast<float>(acc);
    }
  }
}

}  // namespace

int main() {
  try {
    cuda::init();
    cublas::init();
    const auto& d = cuda::device_info();
    std::printf("device     : %s sm_%d%d SMs=%d\n", d.name.c_str(), d.cc_major, d.cc_minor,
                d.sm_count);

    const int m = 256, n = 256, k = 2048;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> A(static_cast<size_t>(m) * k), B(static_cast<size_t>(k) * n),
        C(static_cast<size_t>(m) * n), ref(static_cast<size_t>(m) * n);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    cuda::Buffer dA(A.size() * 4), dB(B.size() * 4), dC(C.size() * 4);
    dA.upload(A.data(), A.size() * 4);
    dB.upload(B.data(), B.size() * 4);

    // ---- f32 ----
    cublas::gemm_f32(m, n, k, static_cast<const float*>(dA.data()),
                     static_cast<const float*>(dB.data()), static_cast<float*>(dC.data()));
    cuda::sync();
    dC.download(C.data(), C.size() * 4);
    gemm_cpu(A, B, ref, m, n, k);
    // norm-relative error: ||C-ref||_2 / ||ref||_2  (max-relative is meaningless near zero)
    double num = 0, den = 0, max_abs = 0;
    for (size_t i = 0; i < C.size(); ++i) {
      const double e = static_cast<double>(C[i]) - ref[i];
      num += e * e;
      den += static_cast<double>(ref[i]) * ref[i];
      max_abs = std::max(max_abs, std::fabs(e));
    }
    const double rel_l2 = std::sqrt(num / std::max(den, 1e-30));
    std::printf("gemm f32   : %dx%dx%d  max_abs=%.3e  rel_l2=%.3e %s\n", m, n, k, max_abs,
                rel_l2, (rel_l2 < 1e-4) ? "PASS" : "FAIL");

    const int iters = 50;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      cublas::gemm_f32(m, n, k, static_cast<const float*>(dA.data()),
                       static_cast<const float*>(dB.data()), static_cast<float*>(dC.data()));
    cuda::sync();
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count() / iters;
    std::printf("timing f32 : %.3f ms  %.2f TFLOP/s\n", sec * 1e3,
                2.0 * m * n * k / sec / 1e12);

    // ---- bf16 ----
    std::vector<uint16_t> Ab(A.size()), Bb(B.size());
    for (size_t i = 0; i < A.size(); ++i) Ab[i] = f32_to_bf16(A[i]);
    for (size_t i = 0; i < B.size(); ++i) Bb[i] = f32_to_bf16(B[i]);
    cuda::Buffer dAb(A.size() * 2), dBb(B.size() * 2), dCb(C.size() * 2);
    dAb.upload(Ab.data(), Ab.size() * 2);
    dBb.upload(Bb.data(), Bb.size() * 2);

    cublas::gemm_bf16(m, n, k, dAb.data(), dBb.data(), dCb.data());
    cuda::sync();
    std::vector<uint16_t> Cb(C.size());
    dCb.download(Cb.data(), Cb.size() * 2);

    // reference from bf16-rounded inputs, in double
    std::vector<float> Abf(A.size()), Bbf(B.size());
    for (size_t i = 0; i < A.size(); ++i) Abf[i] = bf16_to_f32(Ab[i]);
    for (size_t i = 0; i < B.size(); ++i) Bbf[i] = bf16_to_f32(Bb[i]);
    gemm_cpu(Abf, Bbf, ref, m, n, k);
    double num2 = 0, den2 = 0;
    for (size_t i = 0; i < Cb.size(); ++i) {
      const double e = static_cast<double>(bf16_to_f32(Cb[i])) - ref[i];
      num2 += e * e;
      den2 += static_cast<double>(ref[i]) * ref[i];
    }
    const double rel2 = std::sqrt(num2 / std::max(den2, 1e-30));
    std::printf("gemm bf16  : rel_l2=%.3e %s\n", rel2, (rel2 < 1e-2) ? "PASS" : "FAIL");

    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      cublas::gemm_bf16(m, n, k, dAb.data(), dBb.data(), dCb.data());
    cuda::sync();
    auto t3 = std::chrono::steady_clock::now();
    double sec2 = std::chrono::duration<double>(t3 - t2).count() / iters;
    std::printf("timing bf16: %.3f ms  %.2f TFLOP/s\n", sec2 * 1e3,
                2.0 * m * n * k / sec2 / 1e12);

    return (rel_l2 < 1e-4 && rel2 < 1e-2) ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
