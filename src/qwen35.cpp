// qwen35 (Qwen3.5) hybrid stack: gated delta net linear attention layers and
// qwen3-next style full attention layers (q+gate fused, per-head QK RMSNorm,
// partial rotary on the first 64 dims, sigmoid output gate).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

#include "engine/config.h"
#include "engine/kernels.h"
#include "engine/runner.h"
#include "engine/runner_internal.h"

namespace runner {

namespace {
inline bool gd_smem_on() {
  static const bool on = engine::config::get_bool("gd_smem");
  return on;
}
inline bool gd_reg_on() {
  static const bool on = engine::config::get_bool("gd_reg");
  return on;
}
}  // namespace

namespace {
constexpr float kQ35Eps = 1e-6f;

bool tensor_exists(const std::shared_ptr<void>&, const std::string&);
}  // namespace

// ------------------------------------------ Gated DeltaNet geometry checks ----
// These run at load time (from load_q35_linear / load_q35_attn) so a checkpoint whose
// geometry the kernels cannot represent is rejected with a precise message instead of
// silently computing a wrong head map or recurrent state.
namespace q35_geometry {

void validate_ssm(const ir::SSMDesc& s, bool gguf_container) {
  if (s.variant != "gated_delta") return;
  // gd_conv_state implements the K=4 shift explicitly (and gd_ssm_prep keeps hist[8]).
  // Every shipped qwen35 checkpoint uses conv_kernel=4; reject anything else instead of
  // keeping a wrong conv history.
  if (s.conv_kernel != 4) {
    throw std::runtime_error(
        "Gated DeltaNet conv kernel K=" + std::to_string(s.conv_kernel) +
        " unsupported (this build implements K=4 only); refusing the model instead of "
        "computing a wrong recurrent state");
  }
  if (s.n_key_head <= 0 || s.n_value_head <= 0) {
    throw std::runtime_error("Gated DeltaNet head geometry: n_key_head=" +
                             std::to_string(s.n_key_head) + " n_value_head=" +
                             std::to_string(s.n_value_head) + " (both must be > 0)");
  }
  if (s.key_head_dim <= 0 || s.value_head_dim <= 0) {
    throw std::runtime_error("Gated DeltaNet head geometry: key_head_dim=" +
                             std::to_string(s.key_head_dim) + " value_head_dim=" +
                             std::to_string(s.value_head_dim) + " (both must be > 0)");
  }
  if (s.value_head_dim > 1024) {
    throw std::runtime_error("Gated DeltaNet value_head_dim=" +
                             std::to_string(s.value_head_dim) +
                             " exceeds the 1024-thread delta-rule launch");
  }
  if (s.key_head_dim > s.value_head_dim) {
    throw std::runtime_error(
        "Gated DeltaNet key_head_dim=" + std::to_string(s.key_head_dim) +
        " > value_head_dim=" + std::to_string(s.value_head_dim) +
        ": the generic delta-rule kernel cannot fill its key/query staging");
  }
  if (!gguf_container && s.n_value_head % s.n_key_head != 0) {
    throw std::runtime_error(
        "Gated DeltaNet HF head map needs divisibility: value_head_count=" +
        std::to_string(s.n_value_head) + " % key_head_count=" + std::to_string(s.n_key_head) +
        " != 0 (repeat_interleave mapping)");
  }
  if (s.key_dim != s.n_key_head * s.key_head_dim) {
    throw std::runtime_error("Gated DeltaNet key_dim=" + std::to_string(s.key_dim) +
                             " != key_head_count*key_head_dim=" +
                             std::to_string(s.n_key_head * s.key_head_dim));
  }
  if (s.value_dim != s.n_value_head * s.value_head_dim) {
    throw std::runtime_error("Gated DeltaNet value_dim=" + std::to_string(s.value_dim) +
                             " != value_head_count*value_head_dim=" +
                             std::to_string(s.n_value_head * s.value_head_dim) +
                             " (metadata truncating division?)");
  }
  if (s.conv_dim != 2 * s.key_dim + s.value_dim) {
    throw std::runtime_error(
        "Gated DeltaNet conv_dim=" + std::to_string(s.conv_dim) +
        " != 2*key_dim + value_dim=" + std::to_string(2 * s.key_dim + s.value_dim) +
        " (conv kernel width does not match the q|k|v head geometry)");
  }
}

void validate_attention_heads(int n_head, int n_kv) {
  if (n_head <= 0 || n_kv <= 0) {
    throw std::runtime_error("attention head geometry: head_count=" + std::to_string(n_head) +
                             " head_count_kv=" + std::to_string(n_kv) + " (both must be > 0)");
  }
  if (n_head % n_kv != 0) {
    throw std::runtime_error(
        "attention head_count=" + std::to_string(n_head) + " % head_count_kv=" +
        std::to_string(n_kv) + " != 0: the GQA group map would silently drop heads");
  }
}

}  // namespace q35_geometry


// ---------------------------------------------------------------- loading ----
void DenseRunner::load_q35_linear(int i, LayerW& w) {
  const auto& S = ir_.ssm;
  q35_geometry::validate_ssm(S, container_ == Container::Gguf);
  const int C = S.conv_dim;
  const std::string p = "blk." + std::to_string(i) + ".";
  w.is_linear = true;
  w.ln1 = upload(tensor_f32(p + "attn_norm.weight"));
  w.ln2 = upload(tensor_f32(p + "post_attention_norm.weight"));
  w.q35.conv_w = upload(tensor_f32(p + "ssm_conv1d.weight"));
  if (!load_qtensor(p + "attn_qkv.weight", w.q35.qkv))
    throw std::runtime_error("qwen35: attn_qkv must be quantized (supported: Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1)");
  if (!load_qtensor(p + "attn_gate.weight", w.q35.z))
    throw std::runtime_error("qwen35: attn_gate must be quantized (supported: Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1)");
  if (!load_qtensor(p + "ssm_alpha.weight", w.q35.alpha))
    throw std::runtime_error("qwen35: ssm_alpha must be quantized (supported: Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1)");
  if (!load_qtensor(p + "ssm_beta.weight", w.q35.beta))
    throw std::runtime_error("qwen35: ssm_beta must be quantized (supported: Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1)");
  if (!load_qtensor(p + "ssm_out.weight", w.q35.out))
    throw std::runtime_error("qwen35: ssm_out must be quantized (supported: Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1)");
  w.q35.a_log = upload(tensor_f32(p + "ssm_a"));
  w.q35.dt_bias = upload(tensor_f32(p + "ssm_dt.bias"));
  w.q35.norm_w = upload(tensor_f32(p + "ssm_norm.weight"));
  w.q35.conv_state = cuda::Buffer(static_cast<size_t>(S.conv_kernel - 1) * C * 4);
  std::vector<float> zeros(static_cast<size_t>(S.conv_kernel - 1) * C, 0.f);
  w.q35.conv_state.upload(zeros.data(), zeros.size() * 4);
  const size_t rec = static_cast<size_t>(S.n_value_head) * S.value_head_dim * S.key_head_dim;
  w.q35.rec_state = cuda::Buffer(rec * 4);
  std::vector<float> zrec(rec, 0.f);
  w.q35.rec_state.upload(zrec.data(), zrec.size() * 4);
  load_q35_ffn(i, w);
}

void DenseRunner::load_q35_ffn(int i, LayerW& w) {
  const std::string p = "blk." + std::to_string(i) + ".";
  // The MTP block lives past ir_.n_layer, so its descriptor is not in ir_.layers.
  const ir::LayerDesc& ldesc =
      (ir_.mtp_block >= 0 && i == ir_.mtp_block) ? ir_.mtp_desc : ir_.layers[i];
  if (ir_.moe.n_expert && ldesc.n_expert_tensors > 0) {
    w.is_moe = true;
    auto has = [&](const std::string& n) {
      auto g = std::static_pointer_cast<gguf::Model>(gguf_);
      for (const auto& t : g->tensors)
        if (t.name == n) return true;
      return false;
    };
    static const bool no_shared = engine::config::get_bool("no_shared");
    if (has(p + "ffn_gate_shexp.weight") && !no_shared) {
      const auto g = tensor_f32(p + "ffn_gate_shexp.weight");
      w.has_shared = true;
      w.sh_inter = static_cast<int>(g.size() / static_cast<size_t>(hidden_));
      w.sh_gate = upload(g);
      w.sh_up = upload(tensor_f32(p + "ffn_up_shexp.weight"));
      if (std::getenv("ENGINE_SH_SWAP_GU")) std::swap(w.sh_gate, w.sh_up);
      w.sh_down = upload(tensor_f32(p + "ffn_down_shexp.weight"));
      if (has(p + "ffn_gate_inp_shexp.weight"))
        w.sh_gate_inp = upload(tensor_f32(p + "ffn_gate_inp_shexp.weight"));
    }
    w.router = upload(tensor_f32(p + "ffn_gate_inp.weight"));
    const std::string gate_name = p + "ffn_gate_exps.weight";
    const std::string up_name = p + "ffn_up_exps.weight";
    const std::string down_name = p + "ffn_down_exps.weight";
    auto find_qk = [&](const std::string& nm) -> const QuantKernel* {
      auto g = std::static_pointer_cast<gguf::Model>(gguf_);
      for (const auto& t : g->tensors)
        if (t.name == nm) return quant_kernel(t.type_name);
      return nullptr;
    };
    auto find_type = [&](const std::string& nm) -> std::string {
      auto g = std::static_pointer_cast<gguf::Model>(gguf_);
      for (const auto& t : g->tensors)
        if (t.name == nm) return t.type_name;
      return {};
    };
    const QuantKernel* qk_g = find_qk(gate_name);
    const QuantKernel* qk_u = find_qk(up_name);
    const QuantKernel* qk_d = find_qk(down_name);
    const size_t g_elems = static_cast<size_t>(expert_inter_) * hidden_;
    const size_t d_elems = static_cast<size_t>(hidden_) * expert_inter_;
    const int ne_all = ir_.moe.n_expert;
    int gpu_n = (gpu_experts_ < 0) ? ne_all : std::min(gpu_experts_, ne_all);
    int moe_idx = -1;
    // Layer-granular placement: a whole MoE layer either keeps every expert in VRAM or
    // none (run_moe has the CPU fallback). Keeping all 40 layers resident overflows a
    // 12 GB card and the driver starts paging.
    if (ir_.mtp_block >= 0 && i == ir_.mtp_block) {
      // The MTP block is placed explicitly (mtp_gpu): a GPU-resident draft layer saves
      // a host round-trip + DDR read on every speculative step.
      gpu_n = mtp_gpu_experts_ < 0 ? 0 : std::min(mtp_gpu_experts_, ne_all);
      moe_idx = -2;
    } else if (gpu_moe_layers_ >= 0 && gpu_experts_ < 0) {
      int idx = 0;
      for (int j = 0; j < i; ++j)
        if (ir_.layers[j].n_expert_tensors > 0) ++idx;
      moe_idx = idx;
      gpu_n = (idx < gpu_moe_layers_) ? ne_all : 0;
    }
    w.cpu_begin = gpu_n;
    if (!(qk_g && qk_u && qk_d && g_elems % qk_g->block_elems == 0 &&
          g_elems % qk_u->block_elems == 0 && d_elems % qk_d->block_elems == 0))
      throw std::runtime_error("qwen35 moe: unsupported expert quant layout (expert shapes must be divisible by the quant block and use Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1)");
    const size_t gb_g = g_elems / qk_g->block_elems * qk_g->block_bytes;
    const size_t gb_u = g_elems / qk_u->block_elems * qk_u->block_bytes;
    const size_t db = d_elems / qk_d->block_elems * qk_d->block_bytes;
    // Adaptive placement: measured free VRAM decides, the estimate only seeds it.
    // The arithmetic estimate (planning block above) is deliberately conservative, so
    // the measured check is authoritative in both directions: it can evict a planned
    // layer and it may extend GPU residency by one layer when the measured free VRAM
    // actually allows it.  Only the immediately next layer is tried, which keeps the
    // GPU-resident set a contiguous prefix; the post-load check verifies the margin.
    if (moe_idx >= 0) {
      const size_t layer_dev = static_cast<size_t>(ne_all) * (gb_g + gb_u + db);
      const size_t kv_extra = static_cast<size_t>(n_kv_) * max_seq_ * head_dim_ * 2 * 4;
      if (gpu_n > 0) {
        if (!expert_layer_fits_vram(i, moe_idx, layer_dev, kv_extra)) gpu_n = 0;
      } else if (vram_extend_ && moe_idx == gpu_moe_layers_ &&
                 expert_layer_fits_vram(i, moe_idx, layer_dev, kv_extra)) {
        gpu_n = ne_all;
        gpu_moe_layers_ = moe_idx + 1;
        if (debug_)
          std::fprintf(stderr,
                       "[placement] measured extend: MoE layer #%d (blk.%d) fits -> "
                       "gpu_moe_layers %d -> %d\n",
                       moe_idx, i, moe_idx, gpu_moe_layers_);
      }
      w.cpu_begin = gpu_n;
    }
    // experimental: some converters store gate/up swapped for the MoE FFN
    const bool swap_gu = std::getenv("ENGINE_MOE_SWAP_GU") != nullptr;
    const std::string& g_name = swap_gu ? up_name : gate_name;
    const std::string& u_name = swap_gu ? gate_name : up_name;
    if (std::getenv("ENGINE_DBG_Q35LOAD"))
      std::fprintf(stderr, "[q35load] layer %d gpu_experts_=%d gpu_n=%d gpu_moe_layers=%d ne=%d\n",
                   i, gpu_experts_, gpu_n, gpu_moe_layers_, ne_all);
    w.q_experts.resize(static_cast<size_t>(gpu_n) * 3);
    for (int e = 0; e < gpu_n; ++e) {
      load_qtensor_span(g_name, e * gb_g, gb_g, expert_inter_, hidden_,
                        w.q_experts[e * 3 + 0]);
      load_qtensor_span(u_name, e * gb_u, gb_u, expert_inter_, hidden_,
                        w.q_experts[e * 3 + 1]);
      load_qtensor_span(down_name, e * db, db, hidden_, expert_inter_,
                        w.q_experts[e * 3 + 2]);
    }
    {
      const size_t ne_bytes = static_cast<size_t>(ne_all) * 8;
      w.all_ptr_idx = static_cast<int>(all_gp_.size());
      all_gp_.emplace_back(ne_bytes);
      all_up_.emplace_back(ne_bytes);
      all_dp_.emplace_back(ne_bytes);
      std::vector<const unsigned char*> gp(ne_all, nullptr), upv(ne_all, nullptr),
          dpv(ne_all, nullptr);
      for (int e = 0; e < ne_all; ++e) {
        if (e >= gpu_n) continue;
        gp[e] = static_cast<const unsigned char*>(w.q_experts[e * 3 + 0].data.data());
        upv[e] = static_cast<const unsigned char*>(w.q_experts[e * 3 + 1].data.data());
        dpv[e] = static_cast<const unsigned char*>(w.q_experts[e * 3 + 2].data.data());
      }
      all_gp_.back().upload(gp.data(), ne_bytes);
      all_up_.back().upload(upv.data(), ne_bytes);
      all_dp_.back().upload(dpv.data(), ne_bytes);
    }

    // device pointer arrays over ALL experts of this layer: the fused MoE
    // kernels look weights up on the device, so routing never round-trips.
    {
      const size_t ne_bytes = static_cast<size_t>(ne_all) * 8;
      w.all_ptr_idx = static_cast<int>(all_gp_.size());
      all_gp_.emplace_back(ne_bytes);
      all_up_.emplace_back(ne_bytes);
      all_dp_.emplace_back(ne_bytes);
      std::vector<const unsigned char*> gp(ne_all, nullptr), upv(ne_all, nullptr),
          dpv(ne_all, nullptr);
      for (int e = 0; e < ne_all; ++e) {
        if (e >= gpu_n) continue;
        gp[e] = static_cast<const unsigned char*>(w.q_experts[e * 3 + 0].data.data());
        upv[e] = static_cast<const unsigned char*>(w.q_experts[e * 3 + 1].data.data());
        dpv[e] = static_cast<const unsigned char*>(w.q_experts[e * 3 + 2].data.data());
      }
      all_gp_.back().upload(gp.data(), ne_bytes);
      all_up_.back().upload(upv.data(), ne_bytes);
      all_dp_.back().upload(dpv.data(), ne_bytes);
    }
    if (gpu_n < ne_all) {
      w.h_type_g = find_type(gate_name);
      w.h_type_u = find_type(up_name);
      w.h_type_d = find_type(down_name);
      w.h_bb_g = qk_g->block_bytes;
      w.h_be_g = qk_g->block_elems;
      w.h_bb_u = qk_u->block_bytes;
      w.h_be_u = qk_u->block_elems;
      w.h_bb_d = qk_d->block_bytes;
      w.h_be_d = qk_d->block_elems;
      w.h_gate_bytes = read_gguf_raw(gate_name);
      w.h_up_bytes = read_gguf_raw(up_name);
      w.h_down_bytes = read_gguf_raw(down_name);
    }
  } else {
    const bool qg = load_qtensor(p + "ffn_gate.weight", w.qg);
    const bool qu = load_qtensor(p + "ffn_up.weight", w.qu);
    const bool qd = load_qtensor(p + "ffn_down.weight", w.qd);
    if (!(qg && qu && qd)) throw std::runtime_error("qwen35: ffn must be quantized (supported: Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1)");
    w.is_quant = true;
  }
}

void DenseRunner::load_q35_attn(int i, LayerW& w) {
  q35_geometry::validate_attention_heads(n_head_, n_kv_);
  const std::string p = "blk." + std::to_string(i) + ".";
  w.ln1 = upload(tensor_f32(p + "attn_norm.weight"));
  w.ln2 = upload(tensor_f32(p + "post_attention_norm.weight"));
  w.q35.attn_kind = true;
  if (!load_qtensor(p + "attn_q.weight", w.q35.qp))
    throw std::runtime_error("qwen35: attn_q must be quantized (supported: Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1)");
  if (!load_qtensor(p + "attn_k.weight", w.q35.kp))
    throw std::runtime_error("qwen35: attn_k must be quantized (supported: Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1)");
  if (!load_qtensor(p + "attn_v.weight", w.q35.vp))
    throw std::runtime_error("qwen35: attn_v must be quantized (supported: Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1)");
  if (!load_qtensor(p + "attn_output.weight", w.q35.op))
    throw std::runtime_error("qwen35: attn_output must be quantized (supported: Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1)");
  w.q35.q_norm = upload(tensor_f32(p + "attn_q_norm.weight"));
  w.q35.k_norm = upload(tensor_f32(p + "attn_k_norm.weight"));
  alloc_kv_cache(w);
  load_q35_ffn(i, w);
}

void DenseRunner::rms_buf(cuda::Buffer& xb, cuda::Buffer& wb, cuda::Buffer& yb, int rows,
                          int n) {
  rms_raw(xb.data(), wb.data(), yb.data(), rows, n);
}

void DenseRunner::reset_kv_and_states() {
  const size_t conv_n = static_cast<size_t>(ir_.ssm.conv_kernel - 1) * ir_.ssm.conv_dim;
  const size_t rec_n = static_cast<size_t>(ir_.ssm.n_value_head) *
                       ir_.ssm.value_head_dim * ir_.ssm.key_head_dim;
  std::vector<float> z0(conv_n, 0.f), z1(rec_n, 0.f);
  for (auto& L : layers_) {
    if (!L.is_linear) continue;
    L.q35.conv_state.upload(z0.data(), z0.size() * 4);
    L.q35.rec_state.upload(z1.data(), z1.size() * 4);
  }
}

// ------------------------------------------------------------------ FFN ------
void DenseRunner::run_ffn(int layer, int n_tokens) {
  auto& L = layers_[layer];
  rms_buf(x_, L.ln2, hn_, n_tokens, hidden_);
  if (layer == 0 && std::getenv("ENGINE_DUMP") && std::getenv("ENGINE_DUMP_Q35")) {
    const char* dp = ENGINE_ENV_STR("ENGINE_DUMP");
    cuda::stream_sync();
    std::vector<float> h(static_cast<size_t>(n_tokens) * hidden_);
    hn_.download(h.data(), h.size() * 4);
    std::ofstream fo(std::string(dp) + "_q35_ffn_in.f32", std::ios::binary);
    fo.write(reinterpret_cast<const char*>(h.data()), static_cast<std::streamsize>(h.size() * 4));
  }
  long long hsz = static_cast<long long>(n_tokens) * hidden_;
  if (L.is_moe) {
    run_moe(layer, n_tokens);
    return;
  }
  matmul(L.qg, n_tokens, hn_, gate_);
  matmul(L.qu, n_tokens, hn_, up_);
  {
    long long total = static_cast<long long>(n_tokens) * inter_;
    void* pg = gate_.data();
    void* pu = up_.data();
    void* pa = act_.data();
    void* args[] = {&pg, &pu, &pa, &total};
    cuda::launch(k_silu_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
  matmul(L.qd, n_tokens, act_, down_);
  {
    void* pa = x_.data();
    void* pb = down_.data();
    void* args[] = {&pa, &pb, &hsz};
    cuda::launch(k_add_, static_cast<int>((hsz + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
}

// -------------------------------------------------- gated delta net layer -----
// Attention/mixer part only (no post-attention FFN).  Split out so it can be
// captured as one CUDA graph for CPU-placed MoE layers (run_q35_linear_attn_graph).
void DenseRunner::run_q35_linear_attn(int layer, int n_tokens, int start_pos) {
  (void)start_pos;
  auto& L = layers_[layer];
  const bool kDbg = debug_;
  auto dump = [&](cuda::Buffer& buf, const char* tag, int rows) {
    if (layer != 0 || graph_capture_) return;
    const char* dp = ENGINE_ENV_STR("ENGINE_DUMP");
    if (!dp) return;
    cuda::stream_sync();
    std::vector<float> h(static_cast<size_t>(rows) * hidden_);
    buf.download(h.data(), h.size() * 4);
    std::ofstream f(std::string(dp) + "_" + tag + ".f32", std::ios::binary);
    f.write(reinterpret_cast<const char*>(h.data()),
            static_cast<std::streamsize>(h.size() * 4));
  };
  (void)kDbg;
  const auto& S = ir_.ssm;
  const int C = S.conv_dim;
  const int K = S.conv_kernel;
  const int nh = S.n_value_head;
  const int nkh = S.n_key_head;
  const int KD = S.key_head_dim;
  const int VD = S.value_head_dim;
  const int rep = nh / nkh;
  // ---- GGUF vs HF tensor semantics -----------------------------------------
  // The GGUF converters write the *llama.cpp* convention for the gated delta net:
  //   * qwen35.ssm_a stores -A, not log(A), so the log decay is
  //       g = ssm_a * softplus(alpha + dt_bias)
  //     (llama.cpp src/models/qwen35.cpp: `gate = alpha_softplus * ssm_a`,
  //      comment "// -A_log.exp() * softplus").
  //   * the SSM q/k heads are broadcast to the value heads by *block repeat*:
  //       value head h reads key head h % n_kh
  //     (llama.cpp ggml/src/ggml-cuda/gated_delta_net.cu: `iq1 = fastmodulo(h_idx,
  //      neqk1)`; ggml.h marks the alternative as [TAG_GGML_GDN_BCAST]).
  // HF safetensors checkpoints use log(A) and repeat_interleave (h / rep).  Using
  // the HF convention on GGUF data leaves position 0 almost intact but destroys
  // every position after it (decay ~exp(-softplus) and the wrong k head), which is
  // why the single-token smoke tests stayed green while real text collapsed.
  // ENGINE_Q35_RAW_A / ENGINE_Q35_LOG_A / ENGINE_Q35_HF_BCAST override for A/B.
  const bool gguf_sem = container_ == Container::Gguf;
  static const bool force_log_a = ENGINE_ENV_ON("ENGINE_Q35_LOG_A");
  static const bool force_raw_a = ENGINE_ENV_ON("ENGINE_Q35_RAW_A");
  static const bool force_hf_bcast = ENGINE_ENV_ON("ENGINE_Q35_HF_BCAST");
  const bool raw_a = force_raw_a || (gguf_sem && !force_log_a);
  const int kh_mode = (!force_hf_bcast && gguf_sem) ? 1 : 0;
  long long hsz = static_cast<long long>(n_tokens) * hidden_;

  rms_buf(x_, L.ln1, hn_, n_tokens, hidden_);
  dump(hn_, "q35_hn0", n_tokens);
  matmul(L.q35.qkv, n_tokens, hn_, q35_mixed_);   // [T, C]
  matmul(L.q35.z, n_tokens, hn_, q35_z_);         // [T, value_dim]
  matmul(L.q35.alpha, n_tokens, hn_, q35_a_);     // [T, nh]
  matmul(L.q35.beta, n_tokens, hn_, q35_beta_b_); // [T, nh] raw b

  // Fused decode path (T == 1): conv1d+silu writes q/k/v directly, the conv history
  // is shifted in the same launch, and the beta/g block runs alongside.
  const bool fused_ssm = ssm_fuse_ && gd_delta_128_ && n_tokens == 1 && k_gdprep_ &&
                         k_gddelta128f_ && KD == 128 && VD == 128 && K <= 8;
  // bisection aids for A/B on the two halves of the fused chain
  static const bool prep_only = ENGINE_ENV_ON("ENGINE_SSM_PREP_ONLY");
  static const bool delta_only = ENGINE_ENV_ON("ENGINE_SSM_DELTA_ONLY");
  const bool fused_prep = fused_ssm && !delta_only;
  const bool fused_delta = fused_ssm && !prep_only;
  if (fused_prep) {
    int T = n_tokens, Cc = C, Kk = K, kd = S.key_dim, vd = S.value_dim, nvh = nh;
    int raw = raw_a ? 1 : 0;
    void* px = q35_mixed_.data();
    void* pw = L.q35.conv_w.data();
    void* ps = L.q35.conv_state.data();
    void* pq = q35_q_.data();
    void* pk = q35_k_.data();
    void* pv = q35_v_.data();
    void* pb = q35_beta_b_.data();
    void* pa = q35_a_.data();
    void* palog = L.q35.a_log.data();
    void* pdt = L.q35.dt_bias.data();
    void* pbeta = q35_beta_.data();
    void* pg = q35_g_.data();
    void* args[] = {&px,   &pw,  &ps,   &pq,  &pk,   &pv,  &T,   &Cc,   &Kk,
                    &kd,   &vd,  &pb,   &pa,  &palog, &pdt, &pbeta, &pg, &nvh, &raw};
    const int nb = (C + 255) / 256;
    cuda::launch(k_gdprep_, nb + 1, 1, 1, 256, 1, 1, 0, args);
  } else if (spec_snapshot_ && n_tokens > 1) {
    // Speculative verification: advance the causal conv one token at a time and
    // snapshot the conv history after every position so the rejected draft tokens can
    // be rolled back.  One launch per token is numerically identical to the batched
    // kernel (gd_conv_state only reads the last K-1 rows / the previous state).
    int Cc = C, Kk = K;
    void* pw = L.q35.conv_w.data();
    void* ps = L.q35.conv_state.data();
    for (int t = 0; t < n_tokens; ++t) {
      int T = 1;
      void* px = static_cast<float*>(q35_mixed_.data()) + static_cast<size_t>(t) * C;
      void* py = static_cast<float*>(q35_conv_.data()) + static_cast<size_t>(t) * C;
      void* args[] = {&px, &pw, &ps, &py, &T, &Cc, &Kk};
      cuda::launch(k_gdconv_, 1, 1, 1, 256, 1, 1, 0, args);
      void* args2[] = {&px, &ps, &T, &Cc, &Kk};
      cuda::launch(k_gdstate_, 64, 1, 1, 256, 1, 1, 0, args2);
      if (t + 1 < n_tokens) ssm_save_conv(layer, t);
    }
  } else {
  // conv1d + silu (writes to q35_conv_), then refresh the conv state
  {
    int T = n_tokens, Cc = C, Kk = K;
    void* px = q35_mixed_.data();
    void* pw = L.q35.conv_w.data();
    void* ps = L.q35.conv_state.data();
    void* py = q35_conv_.data();
    void* args[] = {&px, &pw, &ps, &py, &T, &Cc, &Kk};
    cuda::launch(k_gdconv_, n_tokens, 1, 1, 256, 1, 1, 0, args);
    void* args2[] = {&px, &ps, &T, &Cc, &Kk};
    cuda::launch(k_gdstate_, 64, 1, 1, 256, 1, 1, 0, args2);
  }
  }  // !fused_prep
  if (layer == 0 && std::getenv("ENGINE_DUMP_Q35") && std::getenv("ENGINE_DUMP")) {
    const char* dp = ENGINE_ENV_STR("ENGINE_DUMP");
    cuda::stream_sync();
    auto dump_t = [&](cuda::Buffer& buf, const char* tag, int rows, int cols) {
      std::vector<float> h(static_cast<size_t>(rows) * cols);
      buf.download(h.data(), h.size() * 4);
      std::ofstream fo(std::string(dp) + "_" + tag + ".f32", std::ios::binary);
      fo.write(reinterpret_cast<const char*>(h.data()),
               static_cast<std::streamsize>(h.size() * 4));
    };
    dump_t(hn_, "q35_hn", n_tokens, hidden_);
    dump_t(q35_mixed_, "q35_mixed", n_tokens, C);
    dump_t(q35_conv_, "q35_conv", n_tokens, C);
    dump_t(q35_q_, "q35_q", n_tokens, S.key_dim);
    dump_t(q35_k_, "q35_k", n_tokens, S.key_dim);
    dump_t(q35_v_, "q35_v", n_tokens, S.value_dim);
  }
  // split q / k / v out of the conv output
  if (!fused_prep) {
    int T = n_tokens, key_dim = S.key_dim, value_dim = S.value_dim, Cd = C;
    void* ps = q35_conv_.data();
    void* pq = q35_q_.data();
    void* pk = q35_k_.data();
    void* pv = q35_v_.data();
    void* args[] = {&ps, &pq, &pk, &pv, &T, &key_dim, &value_dim, &Cd};
    cuda::launch(k_q35split3_, n_tokens, 1, 1, 256, 1, 1, 0, args);
  }
  // beta = sigmoid(b); g = ssm_a * softplus(a + dt_bias)   (GGUF convention;
  // HF safetensors stores log(A) instead, see ENGINE_Q35_LOG_A above)
  if (!fused_prep) {
    int total = n_tokens * nh;
    void* pb = q35_beta_b_.data();
    void* pa = q35_a_.data();
    void* palog = L.q35.a_log.data();
    void* pdt = L.q35.dt_bias.data();
    void* pbeta = q35_beta_.data();
    void* pg = q35_g_.data();
    int nvh = nh;
    void* args[] = {&pb, &pa, &palog, &pdt, &pbeta, &pg, &total, &nvh};
    cuda::launch(raw_a ? k_q35beta_raw_ : k_q35beta_, static_cast<int>((total + 255) / 256),
                 1, 1, 256, 1, 1, 0, args);
  }
  // l2 norm q,k (per head) and scale q by 1/sqrt(KD)
  if (!fused_delta) {
    int rows = n_tokens * nkh, D = KD;
    float scale = 1.0f / std::sqrt(static_cast<float>(KD));
    void* pq = q35_q_.data();
    void* pk = q35_k_.data();
    void* args[] = {&pq, &pk, &rows, &D, &scale};
    cuda::launch(k_gdqw_, rows, 1, 1, 128, 1, 1, 8, args);
  }
  // gated delta rule.  The register-tiled kernel (head dim 128) keeps the state in
  // layout [n_vh][KD][VD] and reads/writes it once; the generic kernel uses
  // [n_vh][VD][KD].  Only zero-initialised state is ever uploaded, so the layout
  // choice is local to this launch.  `kh_mode` carries the GGUF (block repeat) vs
  // HF (repeat_interleave) q/k->v head broadcast, see the top of this function.
  if (!(spec_snapshot_ && n_tokens > 1)) {
    int T = n_tokens, nvh = nh, nkh_ = nkh, kd = KD, vd = VD;
    int khm = kh_mode;
    void* pq = q35_q_.data();
    void* pk = q35_k_.data();
    void* pv = q35_v_.data();
    void* pg = q35_g_.data();
    void* pbeta = q35_beta_.data();
    void* pS = L.q35.rec_state.data();
    void* po = q35_o_.data();
    void* args[] = {&pq, &pk, &pv, &pg, &pbeta, &pS, &po, &T, &nvh, &nkh_, &kd, &vd, &khm};
    if (fused_delta) {
      // norm_qk prologue + delta rule + gated RMSNorm epilogue in one launch
      float qscale = 1.0f / std::sqrt(static_cast<float>(KD));
      float eps = kQ35Eps;
      void* pz = q35_z_.data();
      void* pnw = L.q35.norm_w.data();
      void* fargs[] = {&pq, &pk, &pv, &pg, &pbeta, &pS, &po, &pz, &pnw, &qscale, &eps,
                       &T,  &nvh, &nkh_, &kd, &vd, &khm};
      cuda::launch(k_gddelta128f_, nh, 1, 1, 512, 1, 1, 0, fargs);
    } else if (gd_delta_128_ && KD == 128 && VD == 128 && n_tokens > 1 &&
               k_gddelta128smem_ && gd_smem_on()) {
      // prefill: the 64 KiB per-head recurrent state lives in shared memory for the
      // whole token loop instead of being re-read from global every token.
      if (k_gddelta128reg_ && gd_reg_on()) {
        cuda::launch(k_gddelta128reg_, nh, 1, 1, 512, 1, 1, 0, args);
      } else {
        static const size_t shbytes = cuda::max_dynamic_shared_mem(k_gddelta128smem_);
        if (shbytes >= 69120) {
          cuda::launch(k_gddelta128smem_, nh, 1, 1, 512, 1, 1, 69120, args);
        } else {
          cuda::launch(k_gddelta128_, nh, 1, 1, 512, 1, 1, 0, args);
        }
      }
    } else if (gd_delta_128_ && k_gddelta128_ && KD == 128 && VD == 128) {
      cuda::launch(k_gddelta128_, nh, 1, 1, 512, 1, 1, 0, args);
    } else {
      cuda::launch(k_gddelta_, nh, 1, 1, VD, 1, 1, static_cast<size_t>(2 * KD) * 4, args);
    }
  }
  if (spec_snapshot_ && n_tokens > 1) {
    // Same split as the conv above: the delta rule is a per-token recurrence, so a
    // loop of T=1 launches reproduces the batched result exactly while letting us save
    // the recurrent state after every position.
    for (int t = 0; t < n_tokens; ++t) {
      int T = 1, nvh = nh, nkh_ = nkh, kd = KD, vd = VD, khm = kh_mode;
      void* pq = static_cast<float*>(q35_q_.data()) + static_cast<size_t>(t) * S.key_dim;
      void* pk = static_cast<float*>(q35_k_.data()) + static_cast<size_t>(t) * S.key_dim;
      void* pv = static_cast<float*>(q35_v_.data()) + static_cast<size_t>(t) * S.value_dim;
      void* pg = static_cast<float*>(q35_g_.data()) + static_cast<size_t>(t) * nh;
      void* pbeta = static_cast<float*>(q35_beta_.data()) + static_cast<size_t>(t) * nh;
      void* pS = L.q35.rec_state.data();
      void* po = static_cast<float*>(q35_o_.data()) + static_cast<size_t>(t) * S.value_dim;
      void* dargs[] = {&pq, &pk, &pv, &pg, &pbeta, &pS, &po, &T, &nvh, &nkh_, &kd, &vd, &khm};
      if (gd_delta_128_ && k_gddelta128_ && KD == 128 && VD == 128) {
        cuda::launch(k_gddelta128_, nh, 1, 1, 512, 1, 1, 0, dargs);
      } else {
        cuda::launch(k_gddelta_, nh, 1, 1, VD, 1, 1, static_cast<size_t>(2 * KD) * 4, dargs);
      }
      if (t + 1 < n_tokens) ssm_save_rec(layer, t);
    }
  }
  // gated RMSNorm over the value head dim
  if (!fused_delta) {
    int rows = n_tokens * nh, D = VD;
    float eps = kQ35Eps;
    void* po = q35_o_.data();
    void* pz = q35_z_.data();
    void* pw = L.q35.norm_w.data();
    void* args[] = {&po, &pz, &pw, &rows, &D, &eps};
    cuda::launch(k_gdnorm_, rows, 1, 1, 128, 1, 1, 128, args);
  }
  // output projection + residual
  matmul(L.q35.out, n_tokens, q35_o_, proj_);
  {
    void* pa = x_.data();
    void* pb = proj_.data();
    void* args[] = {&pa, &pb, &hsz};
    cuda::launch(k_add_, static_cast<int>((hsz + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
  dump(x_, "q35_mid0", n_tokens);
}

void DenseRunner::run_q35_linear(int layer, int n_tokens, int start_pos) {
  auto& L = layers_[layer];
  run_q35_linear_attn(layer, n_tokens, start_pos);
  run_ffn(layer, n_tokens);
  if (layer == 0 && !graph_capture_) {
    const char* dp = ENGINE_ENV_STR("ENGINE_DUMP");
    if (dp) {
      cuda::stream_sync();
      std::vector<float> h(static_cast<size_t>(n_tokens) * hidden_);
      x_.download(h.data(), h.size() * 4);
      std::ofstream f(std::string(dp) + "_q35_x0.f32", std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()),
              static_cast<std::streamsize>(h.size() * 4));
    }
  }
  (void)L;
}

// ------------------------------------------------ qwen3-next style attention --
void DenseRunner::run_q35_attn_attn(int layer, int n_tokens, int start_pos) {
  (void)start_pos;
  auto& L = layers_[layer];
  static const int kDumpLayer = []() {
    const char* e = ENGINE_ENV_STR("ENGINE_DUMP_LAYER");
    return e ? std::atoi(e) : -1;
  }();
  auto dump = [&](cuda::Buffer& buf, const char* tag, int rows) {
    const char* dp = ENGINE_ENV_STR("ENGINE_DUMP");
    if (!dp || layer != kDumpLayer || graph_capture_) return;
    cuda::stream_sync();
    std::vector<float> h(static_cast<size_t>(rows) * hidden_);
    buf.download(h.data(), h.size() * 4);
    std::ofstream f(std::string(dp) + "_" + tag + ".f32", std::ios::binary);
    f.write(reinterpret_cast<const char*>(h.data()),
            static_cast<std::streamsize>(h.size() * 4));
  };
  const int H = n_head_, KV = n_kv_, D = head_dim_;
  const int rope_dim = ir_.attn.rope_dim > 0 ? ir_.attn.rope_dim : 64;
  long long hsz = static_cast<long long>(n_tokens) * hidden_;

  dump(x_, "q35_attn_in", n_tokens);
  rms_buf(x_, L.ln1, hn_, n_tokens, hidden_);
  dump(hn_, "q35_attn_hn", n_tokens);
  matmul(L.q35.qp, n_tokens, hn_, q35_qfull_);   // [T, H*2D]
  // kp/vp read the same normalised input as qp: keep the three GEMVs adjacent so
  // the int8 activation staging is reused (see DenseRunner::quant_reuse_).
  matmul(L.q35.kp, n_tokens, hn_, q35_attn_k_);
  matmul(L.q35.vp, n_tokens, hn_, q35_attn_v_);
  {
    int T = n_tokens, nh = H, d = D;
    void* ps = q35_qfull_.data();
    void* pq = q35_attn_q_.data();
    void* pg = q35_gate_.data();
    void* args[] = {&ps, &pq, &pg, &T, &nh, &d};
    // One block per (token, head): the per-block serial head loop made this a ~27 us
    // latency-bound kernel on the 8 full-attention layers.
    cuda::launch(k_q35split_, n_tokens, H, 1, 128, 1, 1, 0, args);
  }
  // per-head RMSNorm on q and k
  {
    int rows = n_tokens * H, d = D;
    float eps = kQ35Eps;
    void* px = q35_attn_q_.data();
    void* pw = L.q35.q_norm.data();
    void* py = q35_attn_q_.data();
    void* args[] = {&px, &pw, &py, &d, &eps};
    cuda::launch(k_rms_, rows, 1, 1, 256, 1, 1, 256 * sizeof(float), args);
    int rowsk = n_tokens * KV;
    void* pxk = q35_attn_k_.data();
    void* pwk = L.q35.k_norm.data();
    void* args2[] = {&pxk, &pwk, &pxk, &d, &eps};
    cuda::launch(k_rms_, rowsk, 1, 1, 256, 1, 1, 256 * sizeof(float), args2);
  }
  // partial rotary on the first `rope_dim` dims of each head
  auto rope_part = [&](cuda::Buffer& buf, int nheads) {
    int nh = nheads, stride = D, rd = rope_dim, off = 0;
    void* pb = buf.data();
    void* pp = pos_dev_.data();
    void* pf = rope_freqs_dev_.data();
    float ms = rope_mscale_;
    void* args[] = {&pb, &pp, &nh, &stride, &rd, &off, &pf, &ms};
    cuda::launch(k_ropestrided_, n_tokens, nheads, 1, static_cast<int>(rope_dim / 2), 1, 1, 0,
                 args);
  };
  rope_part(q35_attn_q_, H);
  rope_part(q35_attn_k_, KV);
  // k/v cache + attention (layout/precision/dispatch in DenseRunner::kv_write_layer
  // and DenseRunner::run_attn: f32 short path keeps the historical kernel, the
  // quantized / long-context path uses the tiled online-softmax kernel)
  kv_write_layer(L, n_tokens, q35_attn_k_, q35_attn_v_);
  run_attn(L, n_tokens, start_pos, q35_attn_q_, attn_);
  // sigmoid output gate then o_proj
  {
    long long n = static_cast<long long>(n_tokens) * H * D;
    void* pa = attn_.data();
    void* pg = q35_gate_.data();
    void* args[] = {&pa, &pg, &n};
    cuda::launch(k_q35gate_, static_cast<int>((n + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
  matmul(L.q35.op, n_tokens, attn_, proj_);
  {
    void* pa = x_.data();
    void* pb = proj_.data();
    void* args[] = {&pa, &pb, &hsz};
    cuda::launch(k_add_, static_cast<int>((hsz + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
  dump(x_, "q35_attn_mid", n_tokens);
}

void DenseRunner::run_q35_attn(int layer, int n_tokens, int start_pos) {
  run_q35_attn_attn(layer, n_tokens, start_pos);
  run_ffn(layer, n_tokens);
  static const int kDumpLayer = []() {
    const char* e = ENGINE_ENV_STR("ENGINE_DUMP_LAYER");
    return e ? std::atoi(e) : -1;
  }();
  if (layer == kDumpLayer && !graph_capture_) {
    const char* dp = ENGINE_ENV_STR("ENGINE_DUMP");
    if (dp) {
      cuda::stream_sync();
      std::vector<float> h(static_cast<size_t>(n_tokens) * hidden_);
      x_.download(h.data(), h.size() * 4);
      std::ofstream f(std::string(dp) + "_q35_attn_x.f32", std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()),
              static_cast<std::streamsize>(h.size() * 4));
    }
  }
}

// ------------------------------------------- CPU-layer attention CUDA graph --
// For a layer whose MoE experts all live on the host (cpu_begin == 0), the GPU only
// runs the mixer/attention part, followed by the router D2H + a host sync.  That
// part is ~15 small kernel launches (~0.2-0.3 ms of WDDM submission time on this VM
// for ~0.1 ms of real GPU work), so it is captured once per layer as a single graph.
// The graph is position-independent (the position lives in pos_dev_): full-attention
// layers only differ in the attention kernel's warp count, so they get one graph per
// warp-count bucket.
//
// Capture state machine per layer/bucket: 0 = execute normally (this also performs
// every lazy allocation, so the capture below cannot hit a cudaMalloc), 1 = capture
// (record, then replay), 2 = replay.
int DenseRunner::layer_graph_bucket(int layer, int start_pos) const {
  const auto& L = layers_[layer];
  if (!L.q35.attn_kind) return 0;
  const int ctx = start_pos + 1;
  return attn_warp_bucket_for_ctx(ctx);
}

bool DenseRunner::q35_layer_graph_ok(int layer) const {
  auto& L = layers_[layer];
  if (!L.q35.attn_kind && !L.is_linear) return false;
  // Q5_1 has no GEMV plan (it dequantises through cublas), which capture forbids.
  auto bad = [](const QTensor& t) { return t.valid && t.type == "Q5_1"; };
  if (bad(L.q35.qkv) || bad(L.q35.z) || bad(L.q35.alpha) || bad(L.q35.beta) ||
      bad(L.q35.out) || bad(L.q35.qp) || bad(L.q35.kp) || bad(L.q35.vp) || bad(L.q35.op))
    return false;
  return true;
}

void DenseRunner::run_q35_attn_graph(int layer, int start_pos) {
  auto& L = layers_[layer];
  const int bucket = layer_graph_bucket(layer, start_pos);
  auto& st = L.attn_graph_state[static_cast<size_t>(bucket)];
  const int nt = 1;
  if (st == 2) {
    L.attn_graph[static_cast<size_t>(bucket)]->launch();
  } else if (st == 0) {
    // First decode-shaped call for this bucket: execute normally (this also finishes
    // every lazy allocation) and record the graph *in the same call*.  Recording here
    // instead of on the next token keeps the one-time capture out of the steady-state
    // decode window: the caller is running the prefill, and capture records kernels
    // without executing them, so the result is unchanged.
    if (L.q35.attn_kind)
      run_q35_attn_attn(layer, nt, start_pos);
    else
      run_q35_linear_attn(layer, nt, start_pos);
    capture_attn_graph(layer, bucket);
  } else {
    capture_attn_graph(layer, bucket);
    L.attn_graph[static_cast<size_t>(bucket)]->launch();
  }
  if (ENGINE_ENV_ON("ENGINE_DBG_GRAPH_SYNC")) {
    cuda::stream_sync();
    std::fprintf(stderr, "[graph] replay layer %d bucket %d st=%d ok\n", layer, bucket,
                 L.attn_graph_state[static_cast<size_t>(bucket)]);
  }
  run_ffn(layer, nt);
}

// Record the attention/mixer part of one CPU layer as a graph without executing it.
void DenseRunner::capture_attn_graph(int layer, int bucket) {
  auto& L = layers_[layer];
  // The captured grid only depends on the attention warp-count bucket, so the
  // recorded start_pos just has to land in that bucket (kernel positions come from
  // pos_dev_, which is updated before every replay).
  static const int bucket_pos[3] = {0, 100, 400};
  const int cap_pos = bucket_pos[bucket < 0 ? 0 : (bucket > 2 ? 2 : bucket)];
  cuda::stream_sync();
  auto g = std::make_unique<cuda::Graph>();
  g->capture_begin();
  graph_capture_ = true;
  if (L.q35.attn_kind)
    run_q35_attn_attn(layer, 1, cap_pos);
  else
    run_q35_linear_attn(layer, 1, cap_pos);
  graph_capture_ = false;
  g->capture_end();
  L.attn_graph[static_cast<size_t>(bucket)] = std::move(g);
  L.attn_graph_state[static_cast<size_t>(bucket)] = 2;
  if (ENGINE_ENV_ON("ENGINE_DBG_GRAPH"))
    std::fprintf(stderr, "[graph] layer %d attn bucket %d captured\n", layer, bucket);
}

// Record a whole GPU-resident qwen35 MoE layer (device routing keeps it sync free).
void DenseRunner::capture_gpu_layer_graph(int layer, int bucket) {
  auto& L = layers_[layer];
  static const int bucket_pos[3] = {0, 100, 400};
  const int cap_pos = bucket_pos[bucket < 0 ? 0 : (bucket > 2 ? 2 : bucket)];
  cuda::stream_sync();
  auto g = std::make_unique<cuda::Graph>();
  const bool saved = layer_graph_mode_;
  layer_graph_mode_ = true;   // device routing + router kernel + no fused host path
  g->capture_begin();
  graph_capture_ = true;
  run_layer_body(layer, 1, cap_pos);
  graph_capture_ = false;
  g->capture_end();
  layer_graph_mode_ = saved;
  L.layer_graph = std::move(g);
  L.layer_graph_ready = true;
  L.layer_graph_state = 2;
  L.layer_graph_bucket = bucket;
  if (ENGINE_ENV_ON("ENGINE_DBG_GRAPH"))
    std::fprintf(stderr, "[graph] layer %d full bucket %d captured\n", layer, bucket);
}

void DenseRunner::prepare_graph_scratch() {
  // run_moe() allocates the shared-expert scratch lazily; do it here so the capture
  // (which forbids cuMemAlloc) sees a fully allocated buffer set.
  size_t sh_inter = 0;
  for (auto& L : layers_)
    if (L.has_shared) sh_inter = std::max(sh_inter, static_cast<size_t>(L.sh_inter));
  if (sh_inter == 0) return;
  const size_t rows = static_cast<size_t>(std::max(1, max_seq_));
  const size_t need = rows * sh_inter * 4;
  const size_t dneed = rows * static_cast<size_t>(hidden_) * 4;
  if (sh_scratch_.size() < need) sh_scratch_ = cuda::Buffer(need);
  if (shu_scratch_.size() < need) shu_scratch_ = cuda::Buffer(need);
  if (sha_scratch_.size() < need) sha_scratch_ = cuda::Buffer(need);
  if (shd_scratch_.size() < dneed) shd_scratch_ = cuda::Buffer(dneed);
}

void DenseRunner::capture_decode_graphs() {
  if (ir_.ssm.variant == "none") return;   // only the qwen35 hybrid stack is wired up
  const int ne = ir_.moe.n_expert;
  const bool force_all = ENGINE_ENV_ON("ENGINE_GRAPH_FORCE");
  if (ne <= 0 && !force_all) return;
  prepare_graph_scratch();
  // Default: lazy capture (the first decode token runs the plain path, the second
  // records the graph, the third on replays).  Capturing inside the constructor is
  // broken on this driver -- a graph recorded there faults with an illegal memory
  // access on replay even for a single-node graph (isolated on the 9B model), while the
  // identical capture executed from inside a forward pass works.  Keep the pre-capture
  // path behind ENGINE_GRAPH_PRECAPTURE for diagnostics only.
  if (!ENGINE_ENV_ON("ENGINE_GRAPH_PRECAPTURE")) return;
  const bool dev_ready = dev_route_default_ || ENGINE_ENV_ON("ENGINE_DEV_ROUTE");
  const bool skip_cpu = ENGINE_ENV_ON("ENGINE_NO_CPU_LAYER_GRAPH");
  const bool skip_gpu = ENGINE_ENV_ON("ENGINE_NO_GPU_LAYER_GRAPH");
  const bool full_gpu = ENGINE_ENV_ON("ENGINE_GPU_FULL_LAYER_GRAPH");
  const bool selftest = ENGINE_ENV_ON("ENGINE_GRAPH_SELFTEST");
  const int only = [&]() {
    const char* e = ENGINE_ENV_STR("ENGINE_GRAPH_ONLY_LAYER");
    return e ? std::atoi(e) : -1;
  }();
  auto selftest_one = [&](int l) {
    auto& L = layers_[l];
    if (L.cpu_begin == 0 || !full_gpu) {
      if (L.attn_graph[0]) L.attn_graph[0]->launch();
    } else if (L.layer_graph) {
      L.layer_graph->launch();
    }
    cuda::stream_sync();
    std::fprintf(stderr, "[graph] selftest layer %d replay ok (%d kernels recorded)\n", l,
                 cuda::g_capture_kernel_count);
  };
  for (int l = 0; l < ir_.n_layer; ++l) {
    if (only >= 0 && l != only) continue;
    auto& L = layers_[l];
    if (!q35_layer_graph_ok(l)) continue;
    if (force_all) {
      capture_attn_graph(l, 0);
    } else if (L.cpu_begin == 0) {
      if (skip_cpu) continue;
      capture_attn_graph(l, 0);
    } else if (L.cpu_begin == ne) {
      if (skip_gpu) continue;
      if (full_gpu && dev_ready)
        capture_gpu_layer_graph(l, 0);
      else
        capture_attn_graph(l, 0);
    } else {
      continue;
    }
    if (selftest) selftest_one(l);
  }
}

// GPU-resident qwen35 MoE layer: the whole layer (attention + device-routed MoE) is
// host-sync free, so it becomes one graph launch instead of ~25 kernel launches.
void DenseRunner::run_gpu_layer_graph(int layer, int start_pos) {
  auto& L = layers_[layer];
  // Device routing / router kernel / no fused host path for every token of this layer,
  // so the plain run and the captured graph execute the exact same kernel sequence.
  layer_graph_mode_ = true;
  const int bucket = layer_graph_bucket(layer, start_pos);
  if (L.layer_graph_ready && L.layer_graph && L.layer_graph_bucket == bucket) {
    L.layer_graph->launch();
    return;
  }
  if (L.layer_graph_state == 0) {
    // Execute once (finishes lazy allocation), then record in the same call so the
    // one-time capture stays inside the prefill instead of the decode window.
    run_layer_body(layer, 1, start_pos);
    L.layer_graph_bucket = bucket;
    capture_gpu_layer_graph(layer, bucket);
    return;
  }
  capture_gpu_layer_graph(layer, bucket);
  L.layer_graph->launch();
}

}  // namespace runner
