// cuBLAS (from the pip wheel) loaded dynamically — no import lib needed.
// Row-major wrappers: C[m,n] = A[m,k] * B[k,n].
#pragma once

#include <cstdint>

namespace cublas {

void init();      // loads cublas64_12.dll next to cublasLt64_12.dll
void shutdown();
void set_stream(void* stream);

// Row-major f32 GEMM: C = A * B  (A: m x k, B: k x n, C: m x n)
void gemm_f32(int m, int n, int k, const float* A, const float* B, float* C);

// C = A * B^T with both row-major: A [m,k], B [n,k], C [m,n]
void gemm_f32_nt(int m, int n, int k, const float* A, const float* B, float* C);
// same, but C is a strided column block with leading dimension `ldc`
void gemm_f32_nt_ld(int m, int n, int k, const float* A, const float* B, float* C,
                    int ldc);
// half inputs (fp16), float accumulation and float output, strided columns
void gemm_f16_nt_ld(int m, int n, int k, const void* A, const void* B, float* C,
                    int ldc);
// same as gemm_f16_nt_ld but with fp16 *accumulation* (CUBLAS_COMPUTE_16F) and an
// fp16 output tile: measured 1.85x the fp32-accumulate rate on this GPU (22 vs 12
// T-MAC/s), at the cost of fp16 rounding inside the K reduction.
void gemm_f16_nt_ld_c16(int m, int n, int k, const void* A, const void* B, void* C16, int ldc);
// C[m,n] = A[m,k] @ W[k,n] with W row-major fp16 (TN layout, tensor cores)
void gemm_f16_tn(int m, int n, int k, const void* A, const void* W, float* C);

// Row-major bf16 GEMM with f32 accumulate: C = A * B (all bf16 on device)
void gemm_bf16(int m, int n, int k, const void* A, const void* B, void* C);

}  // namespace cublas
