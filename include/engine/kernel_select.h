// GEMV kernel selection: every "which kernel for this shape" decision lives here.
//
// The dispatcher in src/runner.cpp only launches kernels; the policy (thresholds,
// eligibility, calibrated overrides) is a pure function so it can be reasoned
// about, unit-tested and inspected with --print-config / RUNNER_DEBUG=1.
#pragma once

#include <cstddef>
#include <string>

namespace runner {

enum class GemvPlan {
  kNone,          // no specialized kernel -> dequantise+cuBLAS/GEMV fallback
  Q4K,            // 8 rows/block fp32 dequant dot
  Q4K_X4,         // 4 rows/warp, 32 rows/block
  Q4K_K2,         // split-K across 2 warps (small n_out)
  Q5K,
  Q5K_X4,
  Q5K_DP4A,       // int8 activation dp4a, global loads
  Q5K_DP4A_S,     // int8 activation dp4a, shared-memory staged activations
  Q5K_SK,         // x4 + split-K atomic partials
  Q6K,
  Q6K_X4,
  Q6K_DP4A,       // int8 activation dp4a
  Q6K_DP4A_SK,    // dp4a + split-K across warps of a block
  Q6K_BM,         // block-major repacked weights
  Q6K_SK,
  Q50,
  Q50_X4,
  Q50_K2,
  Q50_P4,
  Q50_BM,
  Q50_DP4A,
  Q80,
  Q80_BM,
  Q80_TINY_SK,    // split-K for tiny n_out (SSM projections)
  Q80_DP4A,
};

// Human readable name; the calibration cache stores exactly these strings
// (with a split suffix where relevant, e.g. "q6k_dp4a_sk4").
const char* gemv_plan_name(GemvPlan plan);

// Parse a cache/override string.  Fills *split when the name encodes one.
// Returns false for an unknown name (caller should warn and ignore).
bool gemv_plan_parse(const std::string& text, GemvPlan* plan, int* split);

// "<type> <n_out>x<n_in>", the key used by the kernel calibration cache.
std::string gemv_shape_key(const std::string& type, int n_out, int n_in);

// Everything the policy needs.  Values are snapshotted from the config once,
// so the hot path never touches the config map.
struct GemvParams {
  bool dp4a_gemv = true;      // Q5_K dp4a path (config dp4a_gemv)
  bool dp4a_q6k = true;       // Q6_K dp4a path
  bool tiny_sk = true;        // Q8_0 tiny-n_out split-K
  bool splitk = false;        // opt-in x4 split-K (measured a wash)
  bool dp4a_optin = false;    // experimental Q8_0/Q5_0 dp4a
  bool has_bm = false;        // block-major weights available for this tensor
  int row_threshold = 2048;   // config row_threshold
  // tiny-n_out split-K gates (config tiny_sk_max_nout / tiny_sk_min_nin).  Defaults
  // are the values measured on this host: below 128 output rows the GEMV leaves most
  // SMs idle, and the split needs at least 16 K-blocks to be worth a reduction.
  int tiny_sk_max_nout = 128;  // = kTinySkMaxNout
  int tiny_sk_min_nin = 512;   // = kTinySkMinNin
  // Q6_K dp4a: split-K is used only when n_out is small enough that one warp per row
  // leaves the SMs underfilled; the crossover was measured at 4096 rows and is also
  // overridable per shape by the calibration cache (config q6k_dp4a_split_max_nout).
  int q6k_dp4a_split_max_nout = 4096;  // = kQ6kDp4aAutoSplitMaxNout
  // Q4_K/Q5_0 small-n_out split-K crossover (config small_nout_split_max); 256 = 8
  // rows/block x 32 blocks, i.e. the minimum row count that covers a 32-SM device.
  int small_nout_split_max = 256;
  int q6k_x4_min = 32768;     // config q6k_x4_min
  int q6k_split = 0;          // config q6k_split (0 = auto)
  size_t dp4a_scratch_bytes = 65536;  // int8 activation scratch capacity
  // Calibrated override for this exact shape (nullptr if none).
  const std::string* override_plan = nullptr;
};

struct GemvChoice {
  GemvPlan plan = GemvPlan::kNone;
  int split = 1;         // meaningful for the *_SK / TINY_SK / DP4A_SK plans
  const char* reason = "default";  // why this plan was chosen (debug logs)
};

// Size boundaries shared with the dispatcher/kernels.
constexpr int kDp4aQ5kSmemLimit = 4096;  // gemv_q5k_dp4a_s: 3*n_in <= 12 KiB
constexpr int kTinySkMaxNout = 128;  // default of config tiny_sk_max_nout
constexpr int kTinySkMinNin = 512;   // default of config tiny_sk_min_nin
constexpr int kQ6kDp4aAutoSplitMaxNout = 4096;  // default of config q6k_dp4a_split_max_nout

// Pure selection policy.  Mirrors the historical hand-written dispatch exactly
// when no calibrated override is present.
GemvChoice choose_gemv(const std::string& type, int n_out, int n_in,
                       const GemvParams& p);

// Check that a plan is legal for this shape; used to reject stale/corrupt
// calibration entries instead of launching a kernel that assumes another shape.
bool gemv_plan_compatible(GemvPlan plan, int split, const std::string& type, int n_out,
                          int n_in, const GemvParams& p);

}  // namespace runner
