// Regression tests for the external-review items #11/#13 (Gated DeltaNet geometry):
// qwen35.cpp now validates the head/conv geometry at load time, so a checkpoint whose
// geometry the kernels cannot represent is rejected instead of silently computing a
// wrong recurrent state / head map.
//
// The checks throw std::runtime_error; this test exercises them directly with synthetic
// SSMDesc values (no model weights needed) plus the values of the three shipped
// qwen35 checkpoints (9B / Qwythos / 35B all use K=4, 16 key heads, 32 value heads,
// 128/128 head dims, conv_dim 8192).
#include <cstdio>
#include <stdexcept>
#include <string>

#include "engine/ir.h"

namespace runner {
namespace q35_geometry {
void validate_ssm(const ir::SSMDesc& s, bool gguf_container);
void validate_attention_heads(int n_head, int n_kv);
}  // namespace q35_geometry
}  // namespace runner

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
  std::printf("  %-62s %s%s%s\n", what.c_str(), ok ? "PASS" : "FAIL",
              detail.empty() ? "" : ": ", detail.c_str());
  if (!ok) ++g_fail;
}

ir::SSMDesc valid_qwen35_ssm() {
  ir::SSMDesc s;
  s.variant = "gated_delta";
  s.conv_kernel = 4;
  s.n_key_head = 16;
  s.n_value_head = 32;
  s.key_head_dim = 128;
  s.value_head_dim = 128;
  s.key_dim = 16 * 128;
  s.value_dim = 32 * 128;
  s.conv_dim = 2 * s.key_dim + s.value_dim;
  s.time_step_rank = 32;
  return s;
}

bool throws_with(const ir::SSMDesc& s, bool gguf, const std::string& needle) {
  try {
    runner::q35_geometry::validate_ssm(s, gguf);
    return false;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (msg.find(needle) == std::string::npos) {
      std::printf("    (threw, but message '%s' lacks '%s')\n", msg.c_str(), needle.c_str());
      return false;
    }
    return true;
  }
}

bool attn_throws(int nh, int nkv, const std::string& needle) {
  try {
    runner::q35_geometry::validate_attention_heads(nh, nkv);
    return false;
  } catch (const std::exception& e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
}

}  // namespace

int main() {
  std::printf("== #13 Gated DeltaNet head geometry ==\n");
  {
    const ir::SSMDesc s = valid_qwen35_ssm();
    bool ok = true;
    try {
      runner::q35_geometry::validate_ssm(s, true);
      runner::q35_geometry::validate_ssm(s, false);   // 32 % 16 == 0 for HF too
    } catch (const std::exception& e) {
      ok = false;
      std::printf("    unexpected rejection: %s\n", e.what());
    }
    check(ok, "shipped geometry (16/32 heads, 128/128, K=4) accepted");
  }
  check(throws_with([] { auto s = valid_qwen35_ssm(); s.n_value_head = 24; s.value_dim = 24 * 128;
                         s.conv_dim = 2 * s.key_dim + s.value_dim; return s; }(), false,
                    "divisibility"),
        "HF 24 value heads / 16 key heads rejected (repeat_interleave not divisible)");
  {
    auto s = valid_qwen35_ssm();
    s.n_value_head = 24;
    s.value_dim = 24 * 128;
    s.conv_dim = 2 * s.key_dim + s.value_dim;
    bool ok = true;
    try {
      runner::q35_geometry::validate_ssm(s, true);   // GGUF block-repeat needs no divisibility
    } catch (const std::exception& e) {
      ok = false;
      std::printf("    unexpected GGUF rejection: %s\n", e.what());
    }
    check(ok, "GGUF block-repeat map accepts non-divisible key/value head counts");
  }
  check(throws_with([] { auto s = valid_qwen35_ssm(); s.n_key_head = 0; return s; }(), true,
                    "both must be > 0"),
        "zero key heads rejected");
  check(throws_with([] { auto s = valid_qwen35_ssm(); s.value_head_dim = 0; return s; }(), true,
                    "both must be > 0"),
        "zero value head dim rejected");
  check(throws_with([] { auto s = valid_qwen35_ssm(); s.value_head_dim = 2048;
                         s.n_value_head = 2; s.value_dim = 2 * 2048;
                         s.conv_dim = 2 * s.key_dim + s.value_dim; return s; }(), true,
                    "1024"),
        "value_head_dim > 1024 rejected (delta-rule launch block size)");
  check(throws_with([] { auto s = valid_qwen35_ssm(); s.key_head_dim = 256;
                         s.key_dim = 16 * 256; s.conv_dim = 2 * s.key_dim + s.value_dim;
                         return s; }(), true, "staging"),
        "key_head_dim > value_head_dim rejected (generic kernel staging)");
  check(throws_with([] { auto s = valid_qwen35_ssm(); s.value_dim -= 128; return s; }(), true,
                    "value_dim"),
        "value_dim inconsistent with head count*head dim rejected");
  check(throws_with([] { auto s = valid_qwen35_ssm(); s.key_dim -= 128; return s; }(), true,
                    "key_dim"),
        "key_dim inconsistent with head count*head dim rejected");
  check(throws_with([] { auto s = valid_qwen35_ssm(); s.conv_dim += 1; return s; }(), true,
                    "conv_dim"),
        "conv_dim != 2*key_dim + value_dim rejected (conv/head mismatch)");

  std::printf("== #11 Gated DeltaNet conv kernel size ==\n");
  for (int k : {2, 3, 5, 6}) {
    check(throws_with([k] { auto s = valid_qwen35_ssm(); s.conv_kernel = k; return s; }(), true,
                      "K=" + std::to_string(k)),
          "conv_kernel K=" + std::to_string(k) + " rejected at load time");
  }
  {
    auto s = valid_qwen35_ssm();
    s.conv_kernel = 4;
    bool ok = true;
    try {
      runner::q35_geometry::validate_ssm(s, true);
    } catch (const std::exception& e) {
      ok = false;
      std::printf("    unexpected rejection: %s\n", e.what());
    }
    check(ok, "conv_kernel K=4 accepted (the implemented kernel)");
  }
  {
    auto s = valid_qwen35_ssm();
    s.variant = "none";
    s.conv_kernel = 7;
    bool ok = true;
    try {
      runner::q35_geometry::validate_ssm(s, true);   // non-SSM layers are not checked
    } catch (const std::exception& e) {
      ok = false;
      std::printf("    unexpected rejection: %s\n", e.what());
    }
    check(ok, "non-gated-delta layers skip the SSM validation");
  }

  std::printf("== #13 full-attention GQA divisibility ==\n");
  check(!attn_throws(16, 4, ""), "attention 16/4 accepted");
  check(!attn_throws(16, 2, ""), "attention 16/2 accepted");
  check(attn_throws(16, 3, "head_count_kv"), "attention 16/3 rejected");
  check(attn_throws(16, 0, "both must be > 0"), "attention n_kv=0 rejected");
  check(attn_throws(0, 4, "both must be > 0"), "attention n_head=0 rejected");

  std::printf("\n%s (%d failure%s)\n",
              g_fail == 0 ? "Q35 GEOMETRY: GREEN" : "Q35 GEOMETRY: FAILED", g_fail,
              g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
}
