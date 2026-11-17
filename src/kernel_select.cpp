#include "engine/kernel_select.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace runner {

namespace {

struct NameEntry {
  const char* name;
  GemvPlan plan;
  bool has_split;
};

const NameEntry kNames[] = {
    {"none", GemvPlan::kNone, false},
    {"q4k", GemvPlan::Q4K, false},
    {"q4k_x4", GemvPlan::Q4K_X4, false},
    {"q4k_k2", GemvPlan::Q4K_K2, false},
    {"q5k", GemvPlan::Q5K, false},
    {"q5k_x4", GemvPlan::Q5K_X4, false},
    {"q5k_dp4a", GemvPlan::Q5K_DP4A, false},
    {"q5k_dp4a_s", GemvPlan::Q5K_DP4A_S, false},
    {"q5k_sk", GemvPlan::Q5K_SK, true},
    {"q6k", GemvPlan::Q6K, false},
    {"q6k_x4", GemvPlan::Q6K_X4, false},
    {"q6k_dp4a", GemvPlan::Q6K_DP4A, false},
    {"q6k_dp4a_sk", GemvPlan::Q6K_DP4A_SK, true},
    {"q6k_bm", GemvPlan::Q6K_BM, false},
    {"q6k_sk", GemvPlan::Q6K_SK, true},
    {"q5_0", GemvPlan::Q50, false},
    {"q5_0_x4", GemvPlan::Q50_X4, false},
    {"q5_0_k2", GemvPlan::Q50_K2, false},
    {"q5_0_p4", GemvPlan::Q50_P4, false},
    {"q5_0_bm", GemvPlan::Q50_BM, false},
    {"q5_0_dp4a", GemvPlan::Q50_DP4A, false},
    {"q8_0", GemvPlan::Q80, false},
    {"q8_0_bm", GemvPlan::Q80_BM, false},
    {"q8_0_tiny_sk", GemvPlan::Q80_TINY_SK, true},
    {"q8_0_dp4a", GemvPlan::Q80_DP4A, false},
};

bool is_dp4a_split(bool dp4a_q6k, const std::string& type, int n_in,
                   size_t scratch, int sb_count, int split) {
  if (!dp4a_q6k || type != "Q6_K") return false;
  if (n_in % 256 != 0 || static_cast<size_t>(n_in) * 2 > scratch) return false;
  if (split < 1 || split > 8) return false;
  return sb_count > 0 && sb_count % split == 0;
}

bool q5k_dp4a_ok(bool enabled, const std::string& type, int n_in, size_t scratch) {
  return enabled && type == "Q5_K" && n_in % 256 == 0 &&
         static_cast<size_t>(n_in) * 2 <= scratch;
}

// Apply a calibrated override, but only when it is legal for this exact shape.
// A stale cache entry (e.g. different model) must never launch a kernel that
// assumes another layout, so incompatible entries are ignored silently.
GemvChoice apply_override(GemvChoice c, const std::string& type, int n_out, int n_in,
                          const GemvParams& p) {
  if (!p.override_plan || p.override_plan->empty()) return c;
  GemvPlan plan = GemvPlan::kNone;
  int split = 1;
  if (!gemv_plan_parse(*p.override_plan, &plan, &split)) return c;
  if (plan == GemvPlan::kNone) return c;
  if (!gemv_plan_compatible(plan, split, type, n_out, n_in, p)) return c;
  c.plan = plan;
  c.split = split;
  c.reason = "calibrated override";
  return c;
}

}  // namespace

const char* gemv_plan_name(GemvPlan plan) {
  for (const auto& e : kNames)
    if (e.plan == plan) return e.name;
  return "none";
}

bool gemv_plan_parse(const std::string& text, GemvPlan* plan, int* split) {
  *split = 1;
  for (const auto& e : kNames) {
    const size_t n = std::strlen(e.name);
    if (text.compare(0, n, e.name) != 0) continue;
    if (text.size() == n) {
      *plan = e.plan;
      return e.plan != GemvPlan::kNone;
    }
    if (!e.has_split) continue;
    const std::string suffix = text.substr(n);
    if (suffix.empty()) continue;
    bool digits = true;
    for (char c : suffix)
      if (c < '0' || c > '9') digits = false;
    if (!digits) continue;
    const int s = std::atoi(suffix.c_str());
    if (s < 1 || s > 8) continue;
    *plan = e.plan;
    *split = s;
    return true;
  }
  return false;
}

std::string gemv_shape_key(const std::string& type, int n_out, int n_in) {
  return type + " " + std::to_string(n_out) + "x" + std::to_string(n_in);
}

bool gemv_plan_compatible(GemvPlan plan, int split, const std::string& type, int n_out,
                          int n_in, const GemvParams& p) {
  const bool q5k_dp4a = q5k_dp4a_ok(p.dp4a_gemv, type, n_in, p.dp4a_scratch_bytes);
  const int sb = n_in / 256;
  switch (plan) {
    case GemvPlan::kNone:
      return true;
    case GemvPlan::Q4K:
    case GemvPlan::Q4K_X4:
    case GemvPlan::Q4K_K2:
      return type == "Q4_K";
    case GemvPlan::Q5K:
    case GemvPlan::Q5K_X4:
      return type == "Q5_K";
    case GemvPlan::Q5K_DP4A:
      return q5k_dp4a;
    case GemvPlan::Q5K_DP4A_S:
      return q5k_dp4a && 3 * n_in <= 12 * 1024;
    case GemvPlan::Q5K_SK:
      return type == "Q5_K" && n_in % 256 == 0 && split >= 1 && split <= 4;
    case GemvPlan::Q6K:
    case GemvPlan::Q6K_X4:
      return type == "Q6_K";
    case GemvPlan::Q6K_DP4A:
      return type == "Q6_K" && p.dp4a_q6k && n_in % 256 == 0 &&
             static_cast<size_t>(n_in) * 2 <= p.dp4a_scratch_bytes;
    case GemvPlan::Q6K_DP4A_SK:
      return is_dp4a_split(p.dp4a_q6k, type, n_in, p.dp4a_scratch_bytes, sb, split);
    case GemvPlan::Q6K_BM:
      return type == "Q6_K" && p.has_bm && n_in % 256 == 0;
    case GemvPlan::Q6K_SK:
      return type == "Q6_K" && n_in % 256 == 0 && split >= 1 && split <= 4;
    case GemvPlan::Q50:
    case GemvPlan::Q50_X4:
    case GemvPlan::Q50_K2:
    case GemvPlan::Q50_P4:
      return type == "Q5_0";
    case GemvPlan::Q50_BM:
      return type == "Q5_0" && p.has_bm && n_in % 32 == 0;
    case GemvPlan::Q50_DP4A:
      return type == "Q5_0" && n_in % 32 == 0;
    case GemvPlan::Q80:
      return type == "Q8_0";
    case GemvPlan::Q80_BM:
      return type == "Q8_0" && p.has_bm && n_in % 32 == 0;
    case GemvPlan::Q80_TINY_SK:
      return type == "Q8_0" && n_out <= p.tiny_sk_max_nout && n_in >= p.tiny_sk_min_nin &&
             (n_in / 32) % split == 0 && split >= 1 && split <= 8;
    case GemvPlan::Q80_DP4A:
      return type == "Q8_0" && n_in % 32 == 0;
  }
  (void)n_out;
  return false;
}

GemvChoice choose_gemv(const std::string& type, int n_out, int n_in,
                       const GemvParams& p) {
  GemvChoice c;
  // The dispatcher excludes Q5_1 from the specialized path (it falls back to the
  // dequantising GEMV); everything else with a kernel for this type is handled.
  if (type == "Q5_1") {
    c.plan = GemvPlan::kNone;
    c.reason = "q5_1 falls back to dequant gemv";
    return c;
  }
  const bool known = type == "Q4_K" || type == "Q5_K" || type == "Q6_K" ||
                     type == "Q5_0" || type == "Q8_0";
  if (!known) {
    c.plan = GemvPlan::kNone;
    c.reason = "no specialized gemv for this quant type";
    return c;
  }

  const bool long_rows = n_in >= 2048 && n_out >= p.row_threshold;
  const bool short_rows = !long_rows;
  bool x4 = long_rows;

  if (type == "Q4_K") {
    c.plan = x4 ? GemvPlan::Q4K_X4 : GemvPlan::Q4K;
  } else if (type == "Q5_K") {
    c.plan = x4 ? GemvPlan::Q5K_X4 : GemvPlan::Q5K;
  } else if (type == "Q6_K") {
    // Measured: the 1-row kernel wins below ~32K rows; the x4 kernel needs its
    // own 8-rows/block launch shape, so both flags move together.
    if (long_rows && n_out < p.q6k_x4_min) {
      c.plan = GemvPlan::Q6K;
      x4 = false;
    } else {
      c.plan = x4 ? GemvPlan::Q6K_X4 : GemvPlan::Q6K;
    }
  } else if (type == "Q5_0") {
    c.plan = x4 ? GemvPlan::Q50_X4 : GemvPlan::Q50;
    x4 = long_rows;
  } else {  // Q8_0
    c.plan = GemvPlan::Q80;
    x4 = false;
  }

  // ---- int8 dp4a GEMV (decode: x is quantised once per token) ----
  if (q5k_dp4a_ok(p.dp4a_gemv, type, n_in, p.dp4a_scratch_bytes)) {
    if (3 * n_in <= 12 * 1024) {
      c.plan = GemvPlan::Q5K_DP4A_S;
      c.reason = "q5k dp4a (shared-staged activations)";
    } else {
      c.plan = GemvPlan::Q5K_DP4A;
      c.reason = "q5k dp4a (global activations)";
    }
    return apply_override(c, type, n_out, n_in, p);
  }
  if (p.dp4a_q6k && type == "Q6_K" && n_in % 256 == 0 &&
      static_cast<size_t>(n_in) * 2 <= p.dp4a_scratch_bytes) {
    const int sb_count = n_in / 256;
    int split = p.q6k_split;
    if (split == 0) {
      if (n_out <= p.q6k_dp4a_split_max_nout && sb_count >= 32 && sb_count % 4 == 0)
        split = 4;
      else if (n_out <= p.q6k_dp4a_split_max_nout && sb_count >= 16 && sb_count % 2 == 0)
        split = 2;
      else
        split = 1;
    }
    if (split > 1 && (sb_count % split != 0 || split > 8)) split = 1;
    if (split > 1) {
      c.plan = GemvPlan::Q6K_DP4A_SK;
      c.split = split;
      c.reason = "q6k dp4a split-K";
    } else {
      c.plan = GemvPlan::Q6K_DP4A;
      c.reason = "q6k dp4a";
    }
    return apply_override(c, type, n_out, n_in, p);
  }
  // ---- tiny output dims: split-K keeps the SMs fed ----
  if (p.tiny_sk && type == "Q8_0" && n_out <= p.tiny_sk_max_nout &&
      n_in >= p.tiny_sk_min_nin) {
    const int ksplit = n_out <= 32 ? 8 : (n_out <= 64 ? 4 : 2);
    if ((n_in / 32) % ksplit == 0) {
      c.plan = GemvPlan::Q80_TINY_SK;
      c.split = ksplit;
      c.reason = "q8_0 tiny-n_out split-K";
      return apply_override(c, type, n_out, n_in, p);
    }
  }
  if (n_out <= p.small_nout_split_max && type == "Q5_0") {
    c.plan = GemvPlan::Q50_K2;
    c.reason = "q5_0 small-n_out k2";
  } else if (n_out <= p.small_nout_split_max && type == "Q4_K") {
    c.plan = GemvPlan::Q4K_K2;
    c.reason = "q4_k small-n_out k2";
  } else if (short_rows && type == "Q5_0") {
    c.plan = GemvPlan::Q50_P4;
    c.reason = "q5_0 short rows p4";
  } else if (short_rows && type == "Q4_K") {
    c.plan = GemvPlan::Q4K_K2;
    c.reason = "q4_k short rows k2";
  } else if (p.has_bm && type == "Q5_0" && n_in % 32 == 0) {
    c.plan = GemvPlan::Q50_BM;
    c.reason = "q5_0 block-major";
  } else if (p.has_bm && type == "Q6_K" && n_in % 256 == 0) {
    c.plan = GemvPlan::Q6K_BM;
    c.reason = "q6_k block-major";
  } else if (p.has_bm && type == "Q8_0" && n_in % 32 == 0) {
    c.plan = GemvPlan::Q80_BM;
    c.reason = "q8_0 block-major";
  } else if ((type == "Q8_0" || type == "Q5_0") && p.dp4a_optin) {
    c.plan = type == "Q8_0" ? GemvPlan::Q80_DP4A : GemvPlan::Q50_DP4A;
    c.reason = "opt-in dp4a";
  } else if (x4) {
    const bool has_sk = type == "Q5_K" || type == "Q6_K";
    int ksplit = 1;
    if (has_sk && p.splitk) {
      const int blocks_now = (n_out + 31) / 32;
      ksplit = blocks_now >= 256 ? 1 : (blocks_now >= 128 ? 2 : 4);
      if (n_out < 256) ksplit = 1;
    }
    if (ksplit > 1) {
      c.plan = type == "Q5_K" ? GemvPlan::Q5K_SK : GemvPlan::Q6K_SK;
      c.split = ksplit;
      c.reason = "x4 split-K (opt-in)";
    } else {
      c.reason = "long-row x4";
    }
  } else {
    c.reason = "short-row 1-row kernels";
  }
  return apply_override(c, type, n_out, n_in, p);
}

}  // namespace runner
