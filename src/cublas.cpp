#include "engine/cublas.h"
#include "engine/config.h"
#include "engine/dll_search.h"

#include <windows.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace cublas {

namespace {

using cublasStatus_t = int;
using cublasHandle_t = void*;
using cublasOperation_t = int;
using cudaDataType_t = int;
using cublasComputeType_t = int;

constexpr int CUBLAS_STATUS_SUCCESS = 0;
constexpr cublasOperation_t CUBLAS_OP_N = 0;
constexpr cublasOperation_t CUBLAS_OP_T = 1;  // used by gemm_f32_nt
constexpr cudaDataType_t CUDA_R_32F = 0;
constexpr cudaDataType_t CUDA_R_16BF = 14;
constexpr cudaDataType_t CUDA_R_16F = 2;
constexpr cublasComputeType_t CUBLAS_COMPUTE_32F = 68;
constexpr cublasComputeType_t CUBLAS_COMPUTE_16F = 64;
constexpr int CUBLAS_GEMM_DEFAULT = -1;

struct Api {
  HMODULE lib = nullptr;
  cublasStatus_t(__stdcall* Create)(cublasHandle_t*) = nullptr;
  cublasStatus_t(__stdcall* Destroy)(cublasHandle_t) = nullptr;
  cublasStatus_t(__stdcall* Sgemm)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int,
                                   int, int, const float*, const float*, int, const float*,
                                   int, const float*, float*, int) = nullptr;
  cublasStatus_t(__stdcall* SetStream)(cublasHandle_t, void*) = nullptr;
  cublasStatus_t(__stdcall* GemmEx)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int,
                                    int, int, const void*, const void*, cudaDataType_t, int,
                                    const void*, cudaDataType_t, int, const void*, void*,
                                    cudaDataType_t, int, cublasComputeType_t, int) = nullptr;
};

Api g;
cublasHandle_t g_handle = nullptr;

void check(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuBLAS ") + what + " failed (status=" +
                             std::to_string(s) + ")");
}

// Ordered directories that may contain cublas64_12.dll.  Kept as a list so the
// error message can show every path that was tried.
std::vector<std::string> cublas_candidate_dirs() {
  namespace ds = engine::dll_search;
  std::vector<std::string> out;
  auto add_dir = [&](const std::string& dir) {
    if (dir.empty()) return;
    if (std::find(out.begin(), out.end(), dir) == out.end()) out.push_back(dir);
  };
  auto add_value = [&](const std::string& value) {
    if (value.empty()) return;
    if (ds::is_dir(value)) {
      add_dir(value);
    } else if (value.size() >= 4 && value.compare(value.size() - 4, 4, ".dll") == 0) {
      add_dir(ds::parent_dir(value));
    } else {
      add_dir(value);
    }
  };
  // 1) explicit config key / env (config maps DSH_CUBLAS_DIR to cublas_dir).
  add_value(engine::config::get_string("cublas_dir"));
  if (const char* env = std::getenv("DSH_CUBLAS_DIR"))
    if (*env) add_value(env);
  // 2) next to engine.exe / launcher.exe (portable drop-in install).
  add_dir(ds::exe_dir());
  // 3) a CUDA Toolkit install (CUDA_PATH / CUDA_HOME bin).
  for (const std::string& dir : ds::cuda_bin_dirs()) add_dir(dir);
  // 4) pip nvidia-cublas-cu12 wheel under %LOCALAPPDATA%/%USERPROFILE%.
  for (const std::string& dir : ds::pip_nvidia_bin_dirs("cublas", "bin")) add_dir(dir);
  return out;
}

}  // namespace

void init() {
  if (g_handle) return;
  const std::vector<std::string> candidates = cublas_candidate_dirs();
  for (const std::string& dir : candidates) {
    SetDllDirectoryA(dir.c_str());
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    {
      std::wstring wdir(dir.begin(), dir.end());
      AddDllDirectory(wdir.c_str());
    }
    g.lib = LoadLibraryExA((dir + "\\cublas64_12.dll").c_str(), nullptr,
                           LOAD_WITH_ALTERED_SEARCH_PATH);
    if (g.lib) break;
  }
  // 5) Last resort: let Windows search PATH (normal CUDA Toolkit install).
  if (!g.lib) g.lib = LoadLibraryA("cublas64_12.dll");
  if (!g.lib) {
    std::string msg =
        "cublas64_12.dll not found.\n"
        "Install the pip wheel (nvidia-cublas-cu12) or a CUDA Toolkit, or point the "
        "engine at the DLL directory explicitly:\n"
        "  engine.exe ... --cublas-dir <directory containing cublas64_12.dll>\n"
        "  set DSH_CUBLAS_DIR=<directory containing cublas64_12.dll>\n"
        "Tried:";
    msg += engine::dll_search::join_lines(candidates);
    msg += "\n  - cublas64_12.dll from PATH";
    throw std::runtime_error(msg);
  }

  auto bind = [](auto& fn, const char* name) {
    fn = reinterpret_cast<std::decay_t<decltype(fn)>>(GetProcAddress(g.lib, name));
    if (!fn) throw std::runtime_error(std::string("cublas: missing symbol ") + name);
  };
  bind(g.Create, "cublasCreate_v2");
  bind(g.Destroy, "cublasDestroy_v2");
  bind(g.Sgemm, "cublasSgemm_v2");
  bind(g.GemmEx, "cublasGemmEx");
  bind(g.SetStream, "cublasSetStream_v2");

  check(g.Create(&g_handle), "cublasCreate");
}

void shutdown() {
  if (g_handle) {
    g.Destroy(g_handle);
    g_handle = nullptr;
  }
}

void set_stream(void* stream) {
  if (!g_handle) throw std::runtime_error("cublas::init() not called");
  check(g.SetStream(g_handle, stream), "cublasSetStream");
}

void gemm_f32(int m, int n, int k, const float* A, const float* B, float* C) {
  if (!g_handle) throw std::runtime_error("cublas::init() not called");
  const float alpha = 1.0f, beta = 0.0f;
  // row-major C = A*B  ==  col-major C^T = B^T * A^T
  check(g.Sgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n),
        "Sgemm");
}

void gemm_f32_nt(int m, int n, int k, const float* A, const float* B, float* C) {
  if (!g_handle) throw std::runtime_error("cublas::init() not called");
  const float alpha = 1.0f, beta = 0.0f;
  // C_rm[m,n] = A[m,k] * B[n,k]^T  ==  col-major C^T(n,m) = B * A^T
  const cublasStatus_t st =
      g.Sgemm(g_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, B, k, A, k, &beta, C, n);
  if (st != 0) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "Sgemm(NT) failed (status=%d, m=%d n=%d k=%d)", int(st), m, n,
                  k);
    throw std::runtime_error(buf);
  }
}

void gemm_f32_nt_ld(int m, int n, int k, const float* A, const float* B, float* C, int ldc) {
  if (!g_handle) throw std::runtime_error("cublas::init() not called");
  const float alpha = 1.0f, beta = 0.0f;
  check(g.Sgemm(g_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, B, k, A, k, &beta, C, ldc),
        "Sgemm(NT,ld)");
}

void gemm_f16_nt_ld(int m, int n, int k, const void* A, const void* B, float* C, int ldc) {
  if (!g_handle) throw std::runtime_error("cublas::init() not called");
  const float alpha = 1.0f, beta = 0.0f;
  check(g.GemmEx(g_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_16F, k, A,
                 CUDA_R_16F, k, &beta, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT),
        "GemmEx(f16)");
}

void gemm_f16_nt_ld_c16(int m, int n, int k, const void* A, const void* B, void* C16,
                         int ldc) {
  if (!g_handle) throw std::runtime_error("cublas::init() not called");
  // alpha/beta are fp16 for CUBLAS_COMPUTE_16F.
  const unsigned short one = 0x3C00u, zero = 0x0000u;
  check(g.GemmEx(g_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, B, CUDA_R_16F, k, A,
                 CUDA_R_16F, k, &zero, C16, CUDA_R_16F, ldc, CUBLAS_COMPUTE_16F,
                 CUBLAS_GEMM_DEFAULT),
        "GemmEx(f16 acc16)");
}

void gemm_f16_tn(int m, int n, int k, const void* A, const void* W, float* C) {
  if (!g_handle) throw std::runtime_error("cublas::init() not called");
  const float alpha = 1.0f, beta = 0.0f;
  // column-major view: C^T[n,m] = W(n×k) * A^T(k×m) with both operands non-transposed
  check(g.GemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, W, CUDA_R_16F, n, A,
                 CUDA_R_16F, k, &beta, C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT),
        "GemmEx(f16 TN)");
}

void gemm_bf16(int m, int n, int k, const void* A, const void* B, void* C) {
  if (!g_handle) throw std::runtime_error("cublas::init() not called");
  const float alpha = 1.0f, beta = 0.0f;
  check(g.GemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_16BF, n, A,
                 CUDA_R_16BF, k, &beta, C, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT),
        "GemmEx(bf16)");
}

}  // namespace cublas
