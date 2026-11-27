#include "engine/runner.h"
#include "engine/runner_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

#include "engine/config.h"
#include "engine/cublas.h"
#include "engine/cpu_matvec.h"
#include "engine/cuda.h"
#include "engine/fp16.h"
#include "engine/kernels.h"
#include "engine/st.h"


namespace runner {

namespace {

// #17: ifstream::read leaves the buffer untouched when the file ends early, and a short
// read used to be accepted silently (a truncated or concurrently-modified model file then
// produced zeros or stale bytes instead of an error).  Every loader read goes through
// this helper so the byte count is verified.
void read_exact(std::ifstream& f, void* dst, size_t bytes, const std::string& what) {
  if (bytes == 0) return;
  f.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (!f || static_cast<size_t>(f.gcount()) != bytes)
    throw std::runtime_error("short read (" + std::to_string(f.gcount()) + " of " +
                             std::to_string(bytes) + " bytes) while loading " + what);
}

}  // namespace


int DenseRunner::attn_warps_for_ctx(int ctx_len) {
  static const int mid = static_cast<int>(engine::config::get_int("attn_warp_ctx_mid"));
  static const int hi = static_cast<int>(engine::config::get_int("attn_warp_ctx_hi"));
  return ctx_len > hi ? 32 : (ctx_len > mid ? 16 : 8);
}

int DenseRunner::attn_warp_bucket_for_ctx(int ctx_len) {
  return attn_warps_for_ctx(ctx_len) >= 32 ? 2 : (attn_warps_for_ctx(ctx_len) >= 16 ? 1 : 0);
}


namespace {

// Upper bound for the int8-quantised activation scratch of the dp4a GEMV path.
constexpr size_t kDp4aMaxElems = 65536;

}  // namespace

Container detect_container(const std::string& path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) return Container::Safetensors;
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f) {
    char magic[4] = {0, 0, 0, 0};
    const size_t got = std::fread(magic, 1, 4, f);
    std::fclose(f);
    if (got == 4 && std::memcmp(magic, "GGUF", 4) == 0) return Container::Gguf;
    return Container::Safetensors;
  }
  return path.find(".gguf") != std::string::npos ? Container::Gguf
                                                 : Container::Safetensors;
}

DenseRunner::DenseRunner(const std::string& path, int max_seq, Container container)
    : container_(container), path_(path), max_seq_(max_seq) {
  // Unified configuration (CLI > config file > env > default), snapshotted so the
  // hot paths below never look at the config map again.
  load_config_snapshot();
  cuda::init();
  cublas::init();
  stream_ = cuda::create_stream();
  cuda::set_stream(stream_);
  cublas::set_stream(stream_);
  auto& cfg = engine::config::Config::get();
  if (cfg.get_bool("no_graph")) use_graph_ = false;
  if (cfg.was_set("gpu_experts")) gpu_experts_ = static_cast<int>(cfg.get_int("gpu_experts"));
  {
    const std::string v = cfg.get_string("mla_q_layout");
    if (!v.empty()) {
      mla_q_global_ = v == "global" || v == "global_pe";
      mla_q_pe_first_ = v == "global_pe";
    }
  }
  {
    const std::string v = cfg.get_string("mla_kv_layout");
    if (!v.empty()) {
      mla_kv_perhead_ = v == "perhead" || v == "perhead_vk";
      mla_kv_vfirst_ = v == "global_vk" || v == "perhead_vk";
    }
  }
  prof_ev0_ = std::make_unique<cuda::Event>();
  prof_ev1_ = std::make_unique<cuda::Event>();

  if (container_ == Container::Safetensors) {
    auto shards = std::make_shared<st::Shards>(st::load_dir(path));
    shards_ = shards;
    ir_ = ir::build_ir_from_hf(path);
  } else {
    auto g = std::make_shared<gguf::Model>(gguf::read(path));
    gguf_ = g;
    ir_ = ir::build_from_gguf(*g);
    const double eps = g->get_float(ir_.arch + ".attention.layer_norm_rms_epsilon", 1e-5);
    eps_ = static_cast<float>(eps);
  }
  if (ir_.attn.variant == "mla" && cfg.get_bool("mla_no_graph")) use_graph_ = false;
  // host-side MoE routing needs a sync, which CUDA graph capture forbids
  if (ir_.moe.n_expert > 0) use_graph_ = false;
  // diagnostic: per-layer graphs on a dense model need the whole-pipeline graph off
  // (a cuGraphLaunch is illegal while the stream is capturing).
  if (std::getenv("ENGINE_GRAPH_FORCE")) use_graph_ = false;
  // Device-side routing removes the per-layer router D2H + stream sync + routing-table
  // upload for every GPU-resident MoE layer.  It produces the same expert weights and
  // the same accumulation order as the host path (SPEC §10: 35B MoE output bit-identical),
  // but the Q5_1 device GEMV branch is still unverified (§11), so models whose expert
  // tensors use Q5_1 keep the host round-trip.
  if (cfg.get_bool("dev_route")) {
    bool q51_expert = false;
    for (const auto& t : ir_.tensors) {
      if (t.name.find("_exps") == std::string::npos) continue;
      if (t.dtype == "Q5_1") {
        q51_expert = true;
        break;
      }
    }
    dev_route_default_ = !q51_expert;
  }
  if (!cfg.get_string("router").empty()) ir_.moe.router_kind = cfg.get_string("router");
  if (cfg.get_int("norm_topk") >= 0) ir_.moe.norm_topk = cfg.get_int("norm_topk") != 0;
  auto errs = ir::validate(ir_);
  if (!errs.empty()) {
    std::string msg = "IR validation failed:";
    for (auto& e : errs) msg += "\n  - " + e;
    throw std::runtime_error(msg);
  }

  hidden_ = ir_.hidden;
  n_head_ = ir_.attn.n_head;
  n_kv_ = ir_.attn.n_head_kv ? ir_.attn.n_head_kv : n_head_;
  head_dim_ = ir_.attn.key_length;
  inter_ = ir_.ffn;
  expert_inter_ = ir_.moe.expert_ffn_length ? ir_.moe.expert_ffn_length : inter_;
  vocab_ = static_cast<int>(ir_.vocab);
  rope_base_ = static_cast<float>(ir_.attn.rope_freq_base);
  if (inter_ <= 0) inter_ = hidden_ * 4;

  // ---- generic architecture blocks (metadata/tensor driven) ----------------
  // The IR derives per-layer geometry + optional ops from metadata structure;
  // these switches only let the user veto/override them.  The architecture name
  // is never consulted.
  {
    const std::string ext_cfg = cfg.get_string("arch_ext");
    // Hard safety rails: the generic path replaces the whole dense loader/forward,
    // so it must never take over architectures with dedicated, verified paths
    // (hybrid SSM, MLA, MoE).  Those keep the legacy code unconditionally.
    const bool ext_arch_ok = ir_.ssm.variant == "none" && ir_.attn.variant != "mla" &&
                             ir_.moe.n_expert == 0;
    ext_enabled_ = ir_.caps.extended && ext_arch_ok && ext_cfg != "off";
    if (ir_.caps.extended && !ext_arch_ok)
      std::fprintf(stderr,
                   "[arch] metadata has extended features but this model uses a dedicated "
                   "path (ssm=%s mla=%d moe=%d); keeping the legacy path\n",
                   ir_.ssm.variant.c_str(), (int)(ir_.attn.variant == "mla"),
                   ir_.moe.n_expert);
    if (ext_cfg == "on" && !ir_.caps.extended)
      std::fprintf(stderr,
                   "[arch] arch_ext=on but the metadata needs no extended path; "
                   "keeping the legacy dense path\n");
    ext_gelu_ = ir_.caps.gelu_mlp;
    const std::string act = cfg.get_string("arch_act");
    if (act == "gelu") ext_gelu_ = true;
    else if (act == "silu") ext_gelu_ = false;
    ext_embed_scale_ = ir_.caps.embed_scale;
    const std::string es = cfg.get_string("arch_embed_scale");
    if (es == "none") ext_embed_scale_ = 1.0;
    else if (es == "sqrt") ext_embed_scale_ = std::sqrt(static_cast<double>(hidden_));
    ext_attn_scale_ = ir_.caps.attn_scale > 0.0 ? ir_.caps.attn_scale : -1.0;
    const std::string as = cfg.get_string("arch_attn_scale");
    if (as == "1.0") ext_attn_scale_ = 1.0;
    else if (as == "rsqrt") ext_attn_scale_ = -1.0;
    ext_logit_softcap_ = ir_.caps.logit_softcap;
    const std::string sc = cfg.get_string("arch_softcap");
    if (sc == "off") ext_logit_softcap_ = 0.0;
    const std::string nm = cfg.get_string("arch_norm_mode");
    ext_norm_plus1_ = nm == "plus1" ? 1 : (nm == "direct" ? 0 : -1);
    if (ext_enabled_) {
      use_graph_ = false;          // per-layer geometry: keep the plain launch path
      prefill_graph_ = false;
      no_graph_cfg_ = true;
      if (kv_type_ != 0) {
        std::fprintf(stderr, "[kv] extended attention path keeps f32 KV (kv_type %d -> 0)\n",
                     kv_type_);
        kv_type_ = 0;
      }
      std::fprintf(stderr,
                   "[arch] extended path: per_layer=%d qk_norm=%d v_norm=%d sandwich=%d "
                   "layer_scalar=%d gelu=%d embed_scale=%.3f attn_scale=%.3f softcap=%.1f "
                   "k_eq_v_layers=%d\n",
                   (int)ir_.caps.per_layer_attn, (int)ir_.caps.qk_norm, (int)ir_.caps.v_norm,
                   (int)ir_.caps.sandwich_norm, (int)ir_.caps.layer_scalar,
                   (int)ext_gelu_, ext_embed_scale_,
                   ext_attn_scale_ > 0 ? ext_attn_scale_ : -1.0, ext_logit_softcap_,
                   [&] {
                     int n = 0;
                     for (int i = 0; i < ir_.n_layer; ++i)
                       if (!ir_.layers[i].attn.has_v) ++n;
                     return n;
                   }());
    }
  }
  // ---- KV quantization + adaptive context length ---------------------------
  // Must run before any layer allocates its cache: it may clamp max_seq_ to what
  // the VRAM budget can actually hold and picks the activation row cap used by
  // the prefill chunking below.
  if (ir_.attn.variant == "mla" && kv_type_ != 0) {
    std::fprintf(stderr,
                 "[kv] kv_type requested but MLA models keep the f32 MLA cache "
                 "(no quantized MLA kernels) -> f32\n");
    kv_type_ = 0;
  }
  derive_ctx_limit(max_seq_);
  prefill_rows_ = static_cast<int>(cfg.get_int("prefill_rows"));
  if (prefill_rows_ < 1) prefill_rows_ = 1;
  rows_ = static_cast<size_t>(max_seq_ < prefill_rows_ ? max_seq_ : prefill_rows_);
  // Never let the prefill chunk reach the whole context.  A chunk == ctx means the
  // activation buffers are sized for every position at once (on the 9B at ctx 4096 that
  // pushed startup usage to ~9.7 GB of 12 GB), and it makes serve's startup prewarm
  // skip itself (its "chunk + 16 > ctx" condition) -- and ctx 4096 is the launcher
  // default.  Cap at 3/4 of the context (and never below 512 so tiny contexts still
  // prefill more than one token at a time).
  const size_t ctx_chunk_cap = static_cast<size_t>(std::max(512, max_seq_ * 3 / 4));
  if (rows_ > ctx_chunk_cap) rows_ = ctx_chunk_cap;
  if (rows_ < 1) rows_ = 1;
  // Fit the prefill chunk to the real budget.  Shrinking is the hard requirement;
  // growing matters just as much for speed: every chunk re-dequantises the whole
  // model (95 ms on a 9B) and re-pays the per-chunk launch overhead, so the default
  // conservative cap of 512 rows turns one pass into four.  Grow (up to
  // prefill_rows_max / max_seq) while activations + KV + weights + reserve still fit.
  {
    const size_t weights = static_cast<size_t>(ir_.fixed_bytes());
    const size_t kv_total = kv_layer_bytes(max_seq_) * static_cast<size_t>(n_kv_layers_);
    const size_t reserve = (static_cast<size_t>(cfg.get_int("vram_margin_mb")) << 20) +
                           (static_cast<size_t>(256) << 20);
    const size_t free_now = cuda::mem_info().free_bytes;
    const int want_rows = static_cast<int>(rows_);
    // MoE expert bytes are NOT part of ir_.fixed_bytes() (they are placed later), yet on a
    // 35B they dominate VRAM (15 layers x 448 MiB).  Growing the prefill chunk against a
    // budget that ignores them is what pushed the 35B to "free=0 MiB" and 3-4 tok/s
    // (driver paging).  For MoE models keep the conservative default chunk.
    const bool has_experts = ir_.moe.n_expert > 0;
    const auto fits = [&](size_t rows) {
      if (has_experts && rows > static_cast<size_t>(want_rows)) return false;
      return (weights + kv_total + activation_bytes(static_cast<int>(rows)) + reserve) <=
             free_now;
    };
    while (rows_ > 32 && !fits(rows_)) rows_ >>= 1;
    const size_t grow_cap =
        std::min<size_t>(static_cast<size_t>(cfg.get_int("prefill_rows_max")), ctx_chunk_cap);
    while (rows_ * 2 <= grow_cap && fits(rows_ * 2)) rows_ *= 2;
    if (static_cast<int>(rows_) < want_rows)
      std::fprintf(stderr,
                   "[ctx] prefill chunk reduced %d -> %zu rows so activations+KV fit "
                   "(weights=%.0f MiB kv=%.0f MiB)\n",
                   want_rows, rows_, weights / 1048576.0, kv_total / 1048576.0);
    else if (static_cast<int>(rows_) > want_rows)
      std::fprintf(stderr,
                   "[ctx] prefill chunk grown %d -> %zu rows (fits: weights=%.0f MiB "
                   "kv=%.0f MiB act=%.0f MiB)\n",
                   want_rows, rows_, weights / 1048576.0, kv_total / 1048576.0,
                   activation_bytes(static_cast<int>(rows_)) / 1048576.0);
  }

  // ---- rope frequency table (plain vs YaRN) ----
  {
    const int d = (ir_.attn.rope_dim > 0 && ir_.attn.variant == "mla") ? ir_.attn.rope_dim
                                                                     : ir_.attn.key_length;
    const int half = std::max(1, (ir_.attn.rope_dim > 0 && ir_.attn.variant != "mla")
                                     ? ir_.attn.rope_dim / 2
                                     : d / 2);
    const int use_d = (ir_.attn.rope_dim > 0 && ir_.attn.variant != "mla") ? ir_.attn.rope_dim : d;
    rope_freqs_.resize(static_cast<size_t>(half));
    const double base = rope_base_;
    for (int i = 0; i < half; ++i)
      rope_freqs_[i] = static_cast<float>(std::pow(base, -2.0 * i / use_d));
    if (container_ == Container::Gguf) {
      auto g = std::static_pointer_cast<gguf::Model>(gguf_);
      const std::string stype = cfg.get_bool("no_yarn")
                                    ? std::string()
                                    : g->get_string(ir_.arch + ".rope.scaling.type", "");
      if (stype == "yarn") {
        // Port of transformers' _compute_yarn_parameters (the reference used by the
        // official DeepSeek-V2 implementation).
        const double factor = g->get_float(ir_.arch + ".rope.scaling.factor", 1.0);
        const double orig = g->get_float(ir_.arch + ".rope.scaling.original_context_length", 4096);
        const double mscale_all = g->get_float(ir_.arch + ".rope.scaling.yarn_log_multiplier", 0.0);
        const double beta_fast = g->get_float(ir_.arch + ".rope.scaling.beta_fast", 32.0);
        const double beta_slow = g->get_float(ir_.arch + ".rope.scaling.beta_slow", 1.0);
        const double dim = (ir_.attn.rope_dim > 0) ? ir_.attn.rope_dim : d;
        auto corr_dim = [&](double beta) {
          return dim * std::log(orig / (beta * 6.283185307179586)) / (2.0 * std::log(base));
        };
        const double low = std::max(0.0, std::floor(corr_dim(beta_fast)));
        const double high = std::min(dim - 1.0, std::ceil(corr_dim(beta_slow)));
        const int h2 = static_cast<int>(dim / 2);
        for (int i = 0; i < h2 && i < half; ++i) {
          const double pos_freq = std::pow(base, 2.0 * i / dim);
          const double extrap = 1.0 / pos_freq;
          const double interp = 1.0 / (factor * pos_freq);
          double ramp = (i - low) / std::max(1e-9, high - low);
          ramp = std::min(1.0, std::max(0.0, ramp));
          const double ef = 1.0 - ramp;   // extrapolation weight
          rope_freqs_[i] = static_cast<float>(interp * (1.0 - ef) + extrap * ef);
        }
        auto get_mscale = [](double scale, double m) {
          return scale <= 1.0 ? 1.0 : 0.1 * m * std::log(scale) + 1.0;
        };
        rope_mscale_ = static_cast<float>(get_mscale(factor, 1.0) /
                                          std::max(1e-9, get_mscale(factor, mscale_all)));
        rope_is_yarn_ = true;
        rope_yarn_factor_ = factor;
        rope_yarn_orig_ = orig;
        rope_yarn_logmul_ = mscale_all;
        rope_yarn_c0_ = low;
        rope_yarn_c1_ = high;
        rope_use_d_ = dim;
        std::fprintf(stderr, "[rope] yarn factor=%.1f orig=%.0f dim=%.0f corr=[%.2f,%.2f] mscale=%.5f\n",
                     factor, orig, dim, low, high, static_cast<double>(rope_mscale_));
        if (std::getenv("ENGINE_DUMP_ROPE")) {
          std::fprintf(stderr, "[rope] freqs:");
          for (int i = 0; i < 8 && i < half; ++i) std::fprintf(stderr, " %.6e", rope_freqs_[i]);
          std::fprintf(stderr, "\n");
        }
      }    }
    rope_freqs_dev_ = cuda::Buffer(rope_freqs_.size() * 4);
    rope_freqs_dev_.upload(rope_freqs_.data(), rope_freqs_.size() * 4);
  }

  std::string src = std::string(kernels::kRmsNorm) + kernels::kQuantCommon + kernels::kRope +
                    kernels::kKvWrite + kernels::kAttnCached + kernels::kSiluMul +
                    kernels::kAdd + kernels::kAddThree + kernels::kAddBias + kernels::kDequantQ4K +
                    kernels::kDequantQ5K + kernels::kDequantQ6K + kernels::kDequantQ8_0 +
                    kernels::kDequantQ5_0 + kernels::kGatherRows + kernels::kScatterAddW + kernels::kDequantQ5_1 +
                    kernels::kRopeStrided + kernels::kSplit2 + kernels::kMlaBuildK +
                    kernels::kMlaAttn + kernels::kMlaKvWrite + kernels::kMlaAttnCached +
                    kernels::kGdConv1d + kernels::kGdConvState + kernels::kGdNormQk +
                    kernels::kGdDelta + kernels::kGdDelta128 + kernels::kGdDelta128F +
                    kernels::kGdSsmPrep + kernels::kGdNormGate +
                    kernels::kQ35SplitGate +
                    kernels::kQ35GateMul + kernels::kQ35Split3 + kernels::kQ35Beta + kernels::kQ35BetaRaw +
                    kernels::kMoeSharedGate + kernels::kQuantXQ8 + kernels::kGemvQ80Dp4a + kernels::kGemvQ50Dp4a + kernels::kAttnCachedW + kernels::kRepackBM + kernels::kGemvQ80BM + kernels::kGemvQ50BM + kernels::kGemvQ6KBM + kernels::kF32ToHalf + kernels::kRopeInterleaved + kernels::kTransposeF16 + kernels::kGemvF32 + std::string(std::getenv("PROBE_DEV_KERNEL") ? "#define PROBE_DEV 1\n" : "") + kernels::kMoeFused + kernels::kRouterGemm + kernels::kClockWarmer + kernels::kQuantGemm + kernels::kGemvSK + kernels::kGemvQ8SK + kernels::kGemvDp4a + kernels::kGemvDp4aQ6K +
                    kernels::kMlaTmpLayout +
                    kernels::kGemvQ4K + kernels::kGemvQ5K +
                    kernels::kGemvQ8_0 + kernels::kGemvQ6K + kernels::kGemvQ5_0 +
                    kernels::kGemvQ4K_x4 + kernels::kGemvQ5K_x4 + kernels::kGemvQ6K_x4 +
                    kernels::kGemvQ50_x4 + kernels::kGemvQ50K2 + kernels::kGemvQ4KK2 +
                    kernels::kGemvQ50P4 + kernels::kRmsNorm4 + kernels::kTopkPartial +
                    kernels::kTopkFinal + kernels::kKvQuantCommon + kernels::kKvWriteQ +
                    kernels::kAttnTiled + kernels::kAttnCombine + kernels::kDequantF16 +
                    kernels::kAttnTiledPre + kernels::kAttnTiledPreGqa + kernels::kAttnTiledGqa +
                    kernels::kGdDelta128Smem +
                    kernels::kGdDelta128Reg + kernels::kF16ToF32 + kernels::kGeluMul +
                    kernels::kScaleInplace + kernels::kSoftcapInplace + kernels::kAttnCfg;
  mod_ = std::make_unique<cuda::Module>(cuda::compile_ptx(src, "runner.cu"));
  k_rms_ = mod_->kernel("rmsnorm");
  k_rms4_ = mod_->kernel("rmsnorm4");
  k_topkpart_ = mod_->kernel("topk_partial");
  k_topkfinal_ = mod_->kernel("topk_final");
  k_rope_ = mod_->kernel("rope");
  k_kvwrite_ = mod_->kernel("kv_write");
  k_attn_ = mod_->kernel("attn_cached");
  k_silu_ = mod_->kernel("silu_mul");
  k_add_ = mod_->kernel("add_inplace");
  k_add3_ = mod_->kernel("add_three_inplace");
  k_bias_ = mod_->kernel("add_bias");
  k_gelumul_ = mod_->kernel("gelu_mul");
  k_scale_ = mod_->kernel("scale_inplace");
  k_softcap_ = mod_->kernel("softcap_inplace");
  k_attncfg_ = mod_->kernel("attn_cached_cfg");
  k_gather_ = mod_->kernel("gather_rows");
  k_scatter_ = mod_->kernel("scatter_add_weighted");
  k_ropestrided_ = mod_->kernel("rope_strided");
  k_split2_ = mod_->kernel("split2");
  k_buildk_ = mod_->kernel("mla_build_k");
  k_mlaattn_ = mod_->kernel("mla_attn");
  k_mlakv_ = mod_->kernel("mla_kv_write");
  k_mlaattnc_ = mod_->kernel("mla_attn_cached");
  k_gdconv_ = mod_->kernel("gd_conv1d_silu");
  k_gdstate_ = mod_->kernel("gd_conv_state");
  k_gdqw_ = mod_->kernel("gd_norm_qk");
  k_gddelta_ = mod_->kernel("gd_delta_rule");
  k_gddelta128_ = mod_->kernel("gd_delta_rule_128");
  k_gddelta128smem_ = mod_->kernel("gd_delta_rule_128_smem");
  k_gddelta128reg_ = mod_->kernel("gd_delta_rule_128_reg");
  k_gddelta128f_ = mod_->kernel("gd_delta_rule_128f");
  k_gdprep_ = mod_->kernel("gd_ssm_prep");
  k_gdnorm_ = mod_->kernel("gd_norm_gate");
  k_q35split_ = mod_->kernel("q35_split_qgate");
  k_q35gate_ = mod_->kernel("q35_gate_mul");
  k_q35split3_ = mod_->kernel("q35_split3");
  k_q35beta_ = mod_->kernel("q35_beta_g");
  k_q35beta_raw_ = mod_->kernel("q35_beta_g_raw");
  k_quantx_ = mod_->kernel("quantize_x_q8");
  k_gemvq80dp4a_ = mod_->kernel("gemv_q8_0_dp4a");
  k_gemvq50dp4a_ = mod_->kernel("gemv_q5_0_dp4a");
  k_attnw_ = mod_->kernel("attn_cached_w");
  k_kvwriteq_ = mod_->kernel("kv_write_q");
  k_attntiled_ = mod_->kernel("attn_tiled");
  k_attntiledpre_ = mod_->kernel("attn_tiled_pre");
  k_attntiledgqa_ = mod_->kernel("attn_tiled_gqa");
  k_attntiledpregqa_ = mod_->kernel("attn_tiled_pre_gqa");
  k_attncombine_ = mod_->kernel("attn_combine");
  k_repackbm_ = mod_->kernel("repack_block_major");
  k_gemvq80bm_ = mod_->kernel("gemv_q8_0_bm");
  k_gemvq50bm_ = mod_->kernel("gemv_q5_0_bm");
  k_gemvq6kbm_ = mod_->kernel("gemv_q6_k_bm");
  k_f32tohalf_ = mod_->kernel("f32_to_half");
  k_f16tof32_ = mod_->kernel("f16_to_f32_ld");
  k_transpf16_ = mod_->kernel("transpose_f16");
  k_gemvf32_ = mod_->kernel("gemv_f32");
  k_moegateup_ = mod_->kernel("moe_gateup_silu");
  k_modeown_ = mod_->kernel("moe_down_scatter");
  k_moeroute_ = mod_->kernel("moe_route");
  k_zerofill_ = mod_->kernel("zero_fill");
  k_routergemm_ = mod_->kernel("router_gemm");
  k_quantgemm_ = mod_->kernel("quant_gemm");
  warmer_kernel_ = mod_->kernel("clock_warmer");
  gemv_q5k_sk_ = mod_->kernel("gemv_q5k_x4_sk");
  gemv_q6k_sk_ = mod_->kernel("gemv_q6k_x4_sk");
  gemv_q8_sk_ = mod_->kernel("gemv_q8_0_sk");
  gemv_q5k_dp4a_ = mod_->kernel("gemv_q5k_x4_dp4a");
  gemv_q5k_dp4a_s_ = mod_->kernel("gemv_q5k_x4_dp4a_s");
  gemv_q6k_dp4a_ = mod_->kernel("gemv_q6k_dp4a");
  gemv_q6k_dp4a_sk_ = mod_->kernel("gemv_q6k_dp4a_sk");
  k_quantx_i8_ = mod_->kernel("quant_x_i8");
  k_moegateupdev_ = mod_->kernel("moe_gateup_silu_dev");
  k_modeowndev_ = mod_->kernel("moe_down_scatter_dev");
  k_ropeinter_ = mod_->kernel("rope_interleaved");
  k_moeshgate_ = mod_->kernel("moe_shared_gate");
  k_mlaqreorder_ = mod_->kernel("mla_q_reorder");
  k_mlakvsplit_ = mod_->kernel("mla_kv_split");
  gemv_q4k_ = mod_->kernel("gemv_q4k");
  gemv_q4k_x4_ = mod_->kernel("gemv_q4k_x4");
  gemv_q5k_x4_ = mod_->kernel("gemv_q5k_x4");
  gemv_q6k_x4_ = mod_->kernel("gemv_q6k_x4");
  gemv_q5_0_x4_ = mod_->kernel("gemv_q5_0_x4");
  gemv_q5_0_k2_ = mod_->kernel("gemv_q5_0_k2");
  gemv_q4k_k2_ = mod_->kernel("gemv_q4k_k2");
  gemv_q5_0_p4_ = mod_->kernel("gemv_q5_0_p4");
  gemv_q5k_ = mod_->kernel("gemv_q5k");
  gemv_q6k_ = mod_->kernel("gemv_q6k");
  gemv_q8_0_ = mod_->kernel("gemv_q8_0");
  gemv_q5_0_ = mod_->kernel("gemv_q5_0");
  dq_q4k_ = mod_->kernel("dequant_q4k");
  dq_q5k_ = mod_->kernel("dequant_q5k");
  dq_q6k_ = mod_->kernel("dequant_q6k");
  dq_q8_0_ = mod_->kernel("dequant_q8_0");
  dq_q5_0_ = mod_->kernel("dequant_q5_0");
  dq_q5_1_ = mod_->kernel("dequant_q5_1");
  dq_q4k_f16_ = mod_->kernel("dequant_q4k_f16");
  dq_q5k_f16_ = mod_->kernel("dequant_q5k_f16");
  dq_q6k_f16_ = mod_->kernel("dequant_q6k_f16");
  dq_q4k_f16_x8_ = mod_->kernel("dequant_q4k_f16_x8");
  dq_q5k_f16_x8_ = mod_->kernel("dequant_q5k_f16_x8");
  dq_q6k_f16_x8_ = mod_->kernel("dequant_q6k_f16_x8");
  dq_q8_0_f16_ = mod_->kernel("dequant_q8_0_f16");
  struct VramReport {
    ~VramReport() {
      if (!std::getenv("ENGINE_DBG_VRAM")) return;
      const cuda::MemInfo m = cuda::mem_info();
      std::fprintf(stderr, "[vram] free %.0f MiB / total %.0f MiB, engine buffers %lld (%.0f MiB)\n",
                   m.free_bytes / 1048576.0, m.total_bytes / 1048576.0, cuda::g_buf_count,
                   cuda::g_buf_bytes / 1048576.0);
      auto log = cuda::g_buf_log;
      std::sort(log.begin(), log.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      std::fprintf(stderr, "[vram] largest buffers:");
      for (size_t i = 0; i < log.size() && i < 8; ++i)
        std::fprintf(stderr, " %.0fMB", log[i].first / 1048576.0);
      std::fprintf(stderr, "\n");
    }
  } vram_report;

  if (debug_) std::fprintf(stderr, "[dbg] loading weights\n");
  if (cfg.was_set("prefill_f32")) prefill_f32_ = cfg.get_bool("prefill_f32");
  if (prefill_f32_ && container_ == Container::Gguf) {
    auto g = std::static_pointer_cast<gguf::Model>(gguf_);
    size_t dense_bytes = 0;
    for (const auto& t : g->tensors)
      if (t.name.find("_exps") == std::string::npos) dense_bytes += static_cast<size_t>(t.nbytes);
    // The mirror costs 4x the dense weights. On MoE models the expert placement needs
    // that VRAM far more (the Coder hit 0 MiB free and the driver paged everything),
    // so the mirror is opt-in there.
    if (ir_.moe.n_expert > 0 && !cfg.was_set("prefill_f32")) prefill_f32_ = false;
    if (dense_bytes * 4 + (1ull << 30) > cuda::mem_info().free_bytes) prefill_f32_ = false;
  }

  // ---- expert placement: derive how many experts per layer fit in VRAM ----
  if (ir_.moe.n_expert > 0 && gpu_experts_ < 0 && container_ == Container::Gguf) {
    auto g = std::static_pointer_cast<gguf::Model>(gguf_);
    size_t expert_bytes = 0, other_bytes = 0;
    for (const auto& t : g->tensors) {
      if (t.name.find("_exps") != std::string::npos)
        expert_bytes += static_cast<size_t>(t.nbytes);
      else
        other_bytes += static_cast<size_t>(t.nbytes);
    }
    const int ne = ir_.moe.n_expert;
    size_t budget = 0;
    if (cfg.get_int("vram_budget_mb") > 0) {
      budget = static_cast<size_t>(cfg.get_int("vram_budget_mb")) << 20;
    } else {
      budget = cuda::mem_info().free_bytes;
    }
    // KV cache + activations + workspace reserve.  vram_extra_reserve_ covers the
    // allocations the ledger below cannot see (driver context, cuBLAS workspaces,
    // allocator padding): without it the planned layer count can leave 0 MiB free
    // after load, which makes the driver page (measured: 25 tok/s -> 3 tok/s).
    const size_t kv = kv_layer_bytes(max_seq_) * static_cast<size_t>(n_kv_layers_);
    const size_t reserve = kv + (static_cast<size_t>(cfg.get_int("workspace_mb")) << 20) +
                           vram_extra_reserve_;
    // The f32 lm_head mirror is only built when the GGUF has no quantized output
    // tensor; reserving vocab*hidden*4 unconditionally wastes ~2 GiB on 248k-vocab
    // models and costs several expert layers.
    bool lm_f32_needed = true;
    {
      auto gg = std::static_pointer_cast<gguf::Model>(gguf_);
      for (const auto& t : gg->tensors)
        if (t.name == "output.weight") lm_f32_needed = false;
    }
    const size_t fixed = other_bytes + reserve +
                         (lm_f32_needed ? static_cast<size_t>(vocab_) * hidden_ * 4 : 0);
    // Safety margin: the estimate below under-counts the real allocations (norms,
    // MLA/KV buffers, pool scratch).  Adaptive placement (expert_layer_fits_vram)
    // enforces the same number against *measured* free VRAM while loading, so a
    // violated estimate is corrected layer by layer instead of paging the driver.
    const size_t margin = static_cast<size_t>(cfg.get_int("vram_margin_mb")) << 20;
    const size_t avail = budget > fixed + margin ? budget - fixed - margin : 0;
    // Layer-granular placement (llama.cpp --n-cpu-moe semantics): whole MoE layers
    // stay on the GPU or move to the CPU as a unit. Splitting every layer's experts
    // forces a host round-trip in every MoE layer and is ~3x slower.
    int n_moe_layers = 0;
    for (const auto& ly : ir_.layers)
      if (ly.n_expert_tensors > 0) ++n_moe_layers;
    if (n_moe_layers <= 0) n_moe_layers = 1;
    const size_t per_layer_experts = expert_bytes / static_cast<size_t>(n_moe_layers);
    int layers_fit = per_layer_experts
                         ? static_cast<int>(avail / per_layer_experts)
                         : n_moe_layers;
    layers_fit = std::max(0, std::min(n_moe_layers, layers_fit));
    gpu_moe_layers_ = layers_fit;
    planned_gpu_moe_layers_ = layers_fit;
    if (gpu_layers_explicit_) {
      const int want = static_cast<int>(cfg.get_int("gpu_layers"));
      // An explicit request is a ceiling for the *planned* residency; the measured
      // per-layer check below may still reduce it when VRAM is short.
      if (want >= 0) gpu_moe_layers_ = std::min(want, n_moe_layers);
    }
    // Do NOT set gpu_experts_ here: it means "experts per layer on the GPU" for
    // EVERY layer, which defeats the layer-granular placement and made the Coder
    // load all 26 layers of experts (11.7 GiB) -> 0 MiB free -> driver paging.
    gpu_experts_ = -1;
    // ---- adaptive placement bookkeeping ----
    // Everything that still has to be allocated after each layer: non-expert
    // weights of the remaining layers, their KV caches, and post-load scratch.
    // Byte sums come from the IR, so this is model-agnostic.
    vram_other_suffix_.assign(static_cast<size_t>(ir_.n_layer) + 1, 0);
    vram_post_load_reserve_ = 0;
    size_t dequant_max_elems = 0;
    size_t output_bytes = 0;
    for (const auto& td : ir_.tensors) {
      if (td.role == ir::Role::EXPERT || td.role == ir::Role::EMBED) continue;
      if (td.role == ir::Role::ATTN || td.role == ir::Role::FFN_DENSE ||
          td.role == ir::Role::SHARED_EXPERT)
        dequant_max_elems = std::max(dequant_max_elems, static_cast<size_t>(td.n_elements));
      if (td.layer >= 0 && td.layer < ir_.n_layer)
        vram_other_suffix_[static_cast<size_t>(td.layer)] += static_cast<size_t>(td.nbytes);
      else {
        vram_post_load_reserve_ += static_cast<size_t>(td.nbytes);
        if (td.role == ir::Role::OUTPUT) output_bytes += static_cast<size_t>(td.nbytes);
      }
    }
    for (int i = ir_.n_layer - 1; i >= 0; --i)
      vram_other_suffix_[static_cast<size_t>(i)] +=
          vram_other_suffix_[static_cast<size_t>(i) + 1];
    // dequant_scratch_ is sized from the largest dense quantized tensor.
    vram_post_load_reserve_ += dequant_max_elems * 4;
    // The f32 lm_head mirror is built when the file has no quantized output tensor.
    {
      bool has_output = false;
      if (container_ == Container::Gguf) {
        auto gg = std::static_pointer_cast<gguf::Model>(gguf_);
        for (const auto& t : gg->tensors)
          if (t.name == "output.weight") has_output = true;
      }
      if (!has_output) {
        vram_post_load_reserve_ += static_cast<size_t>(vocab_) * hidden_ * 4;
      } else if (container_ == Container::Gguf) {
        // A GGUF quantized lm_head is uploaded in the constructor, *before* the layer
        // loop, so it is not a future allocation: reserving it here double counted
        // ~0.5 GiB on the 35B (2048x248320 Q8_0) and cost one GPU-resident MoE layer.
        vram_post_load_reserve_ = vram_post_load_reserve_ > output_bytes
                                      ? vram_post_load_reserve_ - output_bytes
                                      : 0;
      }
    }
    // KV cache per layer, in the layout the configured kv_type actually allocates
    // (f32 = 8 B/value-pair, q8_0 = 34/32 B/value, q4_0 = 18/32 B/value).  Hybrid SSM
    // stacks only carry it on their full-attention layers.
    vram_kv_per_layer_ = kv_layer_bytes(max_seq_);
    vram_layer_has_kv_.assign(static_cast<size_t>(ir_.n_layer), 1);
    if (ir_.ssm.variant != "none")
      for (int i = 0; i < ir_.n_layer; ++i)
        vram_layer_has_kv_[static_cast<size_t>(i)] = ir_.layers[i].kind == "attn" ? 1 : 0;
    std::fprintf(stderr,
                 "[placement] gpu_moe_layers=%d/%d (%s), margin=%lld MiB, budget=%.0f MiB "
                 "fixed=%.0f MiB experts=%.0f MiB\n",
                 gpu_moe_layers_, n_moe_layers, gpu_layers_explicit_ ? "explicit" : "auto",
                 static_cast<long long>(margin >> 20), budget / 1048576.0,
                 fixed / 1048576.0, expert_bytes / 1048576.0);
    if (debug_)
      std::fprintf(stderr,
                   "[placement] per-layer experts %.1f MiB, available after reserve %.0f MiB\n",
                   per_layer_experts / 1048576.0, avail / 1048576.0);
  }
  for (const auto& w : ir_.warnings) std::fprintf(stderr, "[warn] %s\n", w.c_str());
  try {
    load_weights();
  } catch (const std::exception& e) {
    const std::string variant = ir_.ssm.variant != "none"
                                    ? "hybrid-ssm/" + ir_.ssm.variant
                                    : ir_.attn.variant;
    throw std::runtime_error(
        "failed to load model '" + path + "' (arch '" + ir_.arch + "', attention '" +
        variant + "'): " + e.what() +
        "\n  supported model families: llama-like dense (incl. Qwen2/Qwen2.5/Qwen3), "
        "MLA (deepseek2), hybrid SSM (qwen35/qwen35moe)."
        "\n  missing/unsupported tensors usually mean the architecture is not implemented "
        "yet or the file is incomplete; run 'engine info <model>' to inspect the parsed "
        "metadata.");
  }
  // int8 dp4a GEMV scratch.  Allocated here, not lazily in matmul: the first
  // decode token is captured into a CUDA graph and cuMemAlloc is illegal then.
  // 64K elements covers every Q5_K n_in in the supported models (max 12288).
  // qx holds two int8 levels (first + residual), dx two scales per 32-value
  // block, xs8 one reconstructed block sum.
  qx_i8_ = cuda::Buffer(2 * kDp4aMaxElems);
  dx_i8_ = cuda::Buffer((kDp4aMaxElems / 16) * 4);
  xs8_i8_ = cuda::Buffer((kDp4aMaxElems / 32) * 4);
  if (debug_) std::fprintf(stderr, "[dbg] weights loaded, scratch=%zu\n", scratch_floats_);
  if (scratch_floats_) dequant_scratch_ = cuda::Buffer(scratch_floats_ * 4);
  if (container_ == Container::Gguf) {
    // Tied-embedding GGUFs have no output.weight: llama.cpp's converter omits it
    // when the HF config has tie_word_embeddings=true (Gemma family, several small
    // Qwen/Llama checkpoints), and the embedding matrix doubles as the LM head.
    // Keep it quantized instead of materialising a multi-GB f32 copy when the
    // vocabulary is large; F32/F16 embeddings fall through to lm_head_f32() below.
    bool has_output_weight = false;
    {
      auto g = std::static_pointer_cast<gguf::Model>(gguf_);
      for (const auto& t : g->tensors)
        if (t.name == "output.weight") {
          has_output_weight = true;
          break;
        }
    }
    if (has_output_weight) {
      load_qtensor("output.weight", lm_head_qt_);
    } else {
      std::fprintf(stderr, "[load] tied embeddings: using token_embd.weight as lm_head\n");
      load_qtensor("token_embd.weight", lm_head_qt_);
    }
  }
  if (!lm_head_qt_.valid) w_lm_ = upload(lm_head_f32());
  (void)0;

  // Re-derive the prefill chunk from the ACTUAL free VRAM: the weights and the KV
  // cache are resident by now, so this check cannot be fooled by the optimistic
  // pre-load estimate (which ignored cuBLAS workspaces / context / fragmentation) nor
  // by another process sharing the GPU.  The old order allocated the activation
  // buffers first and only then printed "Prefill chunk halved" -- by which time the
  // device was already over-committed and the driver started paging (measured: the
  // first prefill of such a process ran 12x slow).
  {
    const size_t free_now = cuda::mem_info().free_bytes;
    const size_t need_margin = vram_margin_bytes_ + (64ull << 20);
    // activation_bytes() is the design formula; the constructor allocates measurably
    // more (measured on the 9B hybrid at rows=4096: 2300 MiB real vs 1925 MiB estimated,
    // i.e. ~18% more -- attention scratch and staging buffers the formula omits).  The
    // growth therefore budgets with a calibrated cushion so the post-load free-VRAM
    // margin stays real; without it the serve post-load guard refused to start.
    auto need_for = [&](size_t r) {
      return static_cast<size_t>(static_cast<double>(activation_bytes(static_cast<int>(r))) *
                                 1.18) + need_margin;
    };
    const int want_rows = static_cast<int>(rows_);
    while (rows_ > 32 && free_now < need_for(rows_)) rows_ >>= 1;
    const size_t grow_cap = std::min<size_t>(
        static_cast<size_t>(engine::config::get_int("prefill_rows_max")), ctx_chunk_cap);
    while (rows_ * 2 <= grow_cap && free_now >= need_for(rows_ * 2)) rows_ *= 2;
    if (static_cast<int>(rows_) != want_rows)
      std::fprintf(stderr,
                   "[ctx] prefill chunk %d -> %zu rows by actual free VRAM (free=%.0f MiB "
                   "after weights+KV, act(%zu)=%.0f MiB, margin=%.0f MiB)\n",
                   want_rows, rows_, free_now / 1048576.0, rows_,
                   activation_bytes(static_cast<int>(rows_)) / 1048576.0,
                   need_margin / 1048576.0);
  }
  // Activation rows are capped at the prefill chunk size (rows_) instead of the KV
  // length: at --ctx 131072 the historical `rows = max_seq_` sizing asked for
  // logits_ = max_seq*vocab*4 = 130 TB.  forward() prefills the prompt in chunks of
  // rows_ tokens and the KV cache alone is sized from max_seq_.
  const size_t rows = rows_;
  if (max_seq_ > static_cast<int>(rows_))
    std::fprintf(stderr,
                 "[ctx] activation buffers sized for %zu rows (prefill chunks), "
                 "KV/positions for %d tokens\n",
                 rows_, max_seq_);
  x_ = cuda::Buffer(rows * hidden_ * 4);
  hn_ = cuda::Buffer(rows * hidden_ * 4);
  q_ = cuda::Buffer(rows * n_head_ * head_dim_ * 4);
  k_ = cuda::Buffer(rows * n_kv_ * head_dim_ * 4);
  v_ = cuda::Buffer(rows * n_kv_ * head_dim_ * 4);
  attn_ = cuda::Buffer(rows * n_head_ * head_dim_ * 4);
  proj_ = cuda::Buffer(rows * hidden_ * 4);
  gate_ = cuda::Buffer(rows * inter_ * 4);
  up_ = cuda::Buffer(rows * inter_ * 4);
  act_ = cuda::Buffer(rows * inter_ * 4);
  down_ = cuda::Buffer(rows * hidden_ * 4);
  // Reserve only the configured number of logits rows: a last_row_only prefill head
  // computes exactly one row, so rows*vocab*4 (1 MB/row at a 248k vocab, 78% of the
  // per-row activation estimate) needlessly capped the prefill chunk at ~1024 rows on
  // a 9B at ctx 32768, and every extra chunk re-dequantises the whole model.  A caller
  // that wants full logits grows the buffer on demand (ensure_logits_rows).
  {
    int logits_rows = static_cast<int>(rows);
    if (prefill_logits_rows_ > 0 && prefill_logits_rows_ < logits_rows)
      logits_rows = prefill_logits_rows_;
    logits_ = cuda::Buffer(static_cast<size_t>(std::max(1, logits_rows)) *
                           static_cast<size_t>(vocab_) * 4);
  }
  // pinned staging for the per-token logits download (see forward())
  logits_pin_.resize(static_cast<size_t>(vocab_) * 4);
  // Split-K scratch of the tiled attention (decode path).  Allocated here, not at
  // launch time: cudaMalloc is illegal while a CUDA graph is being captured.
  if (kv_quant_ok() && (kv_type_ != 0 || max_seq_ >= attn_flash_min_)) {
    attn_scratch_rows_ = attn_split_rows_;
    const size_t stride = static_cast<size_t>(head_dim_ + 2);
    attn_scratch_ = cuda::Buffer(static_cast<size_t>(attn_scratch_rows_) *
                                 static_cast<size_t>(n_head_) *
                                 static_cast<size_t>(kv_split_) * stride * 4);
    if (debug_)
      std::fprintf(stderr, "[kv] attn split-K scratch %.1f MiB (nsplit=%d)\n",
                   attn_scratch_.size() / 1048576.0, kv_split_);
  }
  // decode head top-k staging: allocated here so CUDA-graph capture never mallocs
  head_topk_ = engine::config::get_bool("head_topk");
  {
    // head_topk_max is the number of candidates the device head produces (and hence the
    // largest k forward_topk()/forward_topk_candidates() can return).  kMaxTopK caps it.
    head_topk_max_ = engine::config::get_int("head_topk_max");
    if (head_topk_max_ < 1) head_topk_max_ = 1;
    if (head_topk_max_ > kMaxTopK) head_topk_max_ = kMaxTopK;
    int nblk = static_cast<int>((static_cast<size_t>(vocab_) + 1939) / 1940);
    nblk = nblk < 1 ? 1 : (nblk > 256 ? 256 : nblk);
    topk_nblk_ = nblk;
    const size_t cap = static_cast<size_t>(kMaxTopK);
    topk_pv_ = cuda::Buffer(static_cast<size_t>(nblk) * cap * 4);
    topk_pi_ = cuda::Buffer(static_cast<size_t>(nblk) * cap * 4);
    topk_ov_ = cuda::Buffer(cap * 4);
    topk_oi_ = cuda::Buffer(cap * 4);
    topk_pin_.resize(cap * 4 * 2);
  }
  if (ir_.ssm.variant != "none") {
    const auto& S = ir_.ssm;
    const size_t C = static_cast<size_t>(S.conv_dim);
    q35_mixed_ = cuda::Buffer(rows * C * 4);
    q35_conv_ = cuda::Buffer(rows * C * 4);
    q35_z_ = cuda::Buffer(rows * static_cast<size_t>(S.value_dim) * 4);
    q35_o_ = cuda::Buffer(rows * static_cast<size_t>(S.value_dim) * 4);
    q35_q_ = cuda::Buffer(rows * static_cast<size_t>(S.key_dim) * 4);
    q35_k_ = cuda::Buffer(rows * static_cast<size_t>(S.key_dim) * 4);
    q35_v_ = cuda::Buffer(rows * static_cast<size_t>(S.value_dim) * 4);
    q35_beta_ = cuda::Buffer(rows * static_cast<size_t>(S.n_value_head) * 4);
    q35_beta_b_ = cuda::Buffer(rows * static_cast<size_t>(S.n_value_head) * 4);
    q35_a_ = cuda::Buffer(rows * static_cast<size_t>(S.n_value_head) * 4);
    q35_g_ = cuda::Buffer(rows * static_cast<size_t>(S.n_value_head) * 4);
    q35_qfull_ = cuda::Buffer(rows * static_cast<size_t>(n_head_ * head_dim_ * 2) * 4);
    q35_gate_ = cuda::Buffer(rows * static_cast<size_t>(n_head_ * head_dim_) * 4);
    q35_attn_q_ = cuda::Buffer(rows * static_cast<size_t>(n_head_ * head_dim_) * 4);
    q35_attn_k_ = cuda::Buffer(rows * static_cast<size_t>(n_kv_ * head_dim_) * 4);
    q35_attn_v_ = cuda::Buffer(rows * static_cast<size_t>(n_kv_ * head_dim_) * 4);
  }
  pos_dev_ = cuda::Buffer(4);
  graph_ = std::make_unique<cuda::Graph>();
  if (ir_.attn.variant == "mla") {
    const int qk = ir_.attn.key_length;
    const int v_dim = ir_.attn.value_length;
    const int kv_lora = ir_.attn.kv_lora_rank;
    const int qk_rope = ir_.attn.rope_dim;
    mq_ = cuda::Buffer(rows * static_cast<size_t>(n_head_) * qk * 4);
    mkva_ = cuda::Buffer(rows * static_cast<size_t>(kv_lora + qk_rope) * 4);
    mckv_ = cuda::Buffer(rows * static_cast<size_t>(kv_lora) * 4);
    mckvn_ = cuda::Buffer(rows * static_cast<size_t>(kv_lora) * 4);
    mkpe_ = cuda::Buffer(rows * static_cast<size_t>(qk_rope) * 4);
    mkn_ = cuda::Buffer(rows * static_cast<size_t>(n_head_) * (qk - qk_rope) * 4);
    mv_ = cuda::Buffer(rows * static_cast<size_t>(n_head_) * v_dim * 4);
    mk_ = cuda::Buffer(rows * static_cast<size_t>(n_head_) * qk * 4);
    mout_ = cuda::Buffer(rows * static_cast<size_t>(n_head_) * v_dim * 4);
    mla_qraw_ = cuda::Buffer(rows * static_cast<size_t>(n_head_) * qk * 4);
    mla_kvraw_ = cuda::Buffer(rows * static_cast<size_t>(n_head_) * (qk - qk_rope + v_dim) * 4);
  }
  if (ir_.moe.n_expert) {
    router_logits_ = cuda::Buffer(rows * static_cast<size_t>(ir_.moe.n_expert) * 4);
    xout_ = cuda::Buffer(rows * hidden_ * 4);
    gx_ = cuda::Buffer(rows * hidden_ * 4);
    ggate_ = cuda::Buffer(rows * static_cast<size_t>(expert_inter_) * 4);
    gup_ = cuda::Buffer(rows * static_cast<size_t>(expert_inter_) * 4);
    gact_ = cuda::Buffer(rows * static_cast<size_t>(expert_inter_) * 4);
    gdown_ = cuda::Buffer(rows * hidden_ * 4);
    const size_t route_slots = rows * static_cast<size_t>(std::max(1, ir_.moe.n_expert_used));
    d_idx_ = cuda::Buffer(route_slots * 4);
    d_w_ = cuda::Buffer(route_slots * 4);
    // Pre-allocate the device-routing buffers: cudaMalloc is forbidden while a CUDA
    // graph is being captured and the first captured layer would hit it.
    slot_exp_ = cuda::Buffer(route_slots * 4);
    slot_tok_ = cuda::Buffer(route_slots * 4);
    slot_w_ = cuda::Buffer(route_slots * 4);
    route_slots_zeroed_ = false;
    moe_gact_ = cuda::Buffer(route_slots * static_cast<size_t>(expert_inter_) * 4);
  }
  // MTP / speculative-decoding block (only in models that ship a `nextn` layer).
  // Loaded last: the shared output head and every activation buffer already exist, so
  // the measured free VRAM decides GPU vs host placement of the MTP experts.
  if (ir_.mtp_block >= 0 && engine::config::Config::get().get_bool("mtp")) load_mtp();
}

// ---------------------------------------------------------------------------
// KV cache quantization + long-context attention
// ---------------------------------------------------------------------------
// Mirror of the device-side block layout (src kernels.h, kKvQuantCommon).
static inline int kv_bb_of(int t) { return t == 1 ? 34 : (t == 2 ? 18 : 4); }
static inline int kv_nb_of(int D) { return (D + 31) / 32; }

bool DenseRunner::kv_quant_ok() const {
  // attn_tiled: D % 32 == 0, D / 32 <= 8, GQA group divides the head count.
  if (ir_.attn.variant == "mla") return false;
  if (head_dim_ <= 0 || head_dim_ > 256 || (head_dim_ % 32) != 0) return false;
  if (n_kv_ <= 0 || n_head_ <= 0 || (n_head_ % n_kv_) != 0) return false;
  return true;
}

size_t DenseRunner::kv_layer_bytes(int seq) const {
  if (ir_.attn.variant == "mla")
    return static_cast<size_t>(n_head_) *
           static_cast<size_t>(ir_.attn.key_length + ir_.attn.value_length) * 4 *
           static_cast<size_t>(seq);
  const size_t vals = static_cast<size_t>(n_kv_) * static_cast<size_t>(head_dim_) *
                      static_cast<size_t>(seq);
  if (kv_type_ == 0) return vals * 2 * 4;
  const size_t rowb = static_cast<size_t>(kv_nb_of(head_dim_) * kv_bb_of(kv_type_));
  return static_cast<size_t>(n_kv_) * static_cast<size_t>(seq) * rowb * 2;
}

size_t DenseRunner::activation_bytes(int rows) const {
  const size_t r = static_cast<size_t>(rows);
  size_t per_row = static_cast<size_t>(hidden_) * 4 * 6;   // x/hn/proj/down + global scratch
  per_row += static_cast<size_t>(n_head_) * static_cast<size_t>(head_dim_) * 4 * 2;
  per_row += static_cast<size_t>(n_kv_) * static_cast<size_t>(head_dim_) * 4 * 2;
  per_row += static_cast<size_t>(inter_) * 4 * 3;
  // Hybrid SSM (qwen3.5): the mixer keeps its own per-token working set -- two conv
  // buffers, z/o/v, q/k, four per-value-head vectors, the qfull/gate/attn staging.
  // Missing these made the chunk-growth estimate ~221 KB/row optimistic on the 9B.
  if (ir_.ssm.variant != "none") {
    const auto& S = ir_.ssm;
    per_row += static_cast<size_t>(S.conv_dim) * 4 * 2;              // mixed, conv
    per_row += static_cast<size_t>(S.value_dim) * 4 * 3;             // z, o, v
    per_row += static_cast<size_t>(S.key_dim) * 4 * 2;               // q, k
    per_row += static_cast<size_t>(S.n_value_head) * 4 * 4;          // beta, beta_b, a, g
    per_row += static_cast<size_t>(n_head_ * head_dim_ * 2) * 4;     // qfull
    per_row += static_cast<size_t>(n_head_ * head_dim_) * 4 * 2;     // gate, attn_q
    per_row += static_cast<size_t>(n_kv_ * head_dim_) * 4 * 2;       // attn_k, attn_v
  }
  // MLA projections (DeepSeek-V2 style): the absorbed q/kv staging buffers.
  if (ir_.attn.variant == "mla") {
    const size_t qk = static_cast<size_t>(ir_.attn.key_length);
    const size_t v_dim = static_cast<size_t>(ir_.attn.value_length);
    const size_t kv_lora = static_cast<size_t>(ir_.attn.kv_lora_rank);
    const size_t qk_rope = static_cast<size_t>(ir_.attn.rope_dim);
    const size_t nh = static_cast<size_t>(n_head_);
    per_row += nh * qk * 4 * 3;                                      // mq, mk, mla_qraw
    per_row += (kv_lora + qk_rope) * 4;                              // mkva
    per_row += kv_lora * 4 * 2;                                      // mckv, mckvn
    per_row += qk_rope * 4;                                          // mkpe
    per_row += nh * (qk - qk_rope) * 4;                              // mkn
    per_row += nh * v_dim * 4 * 2;                                   // mv, mout
    per_row += nh * (qk - qk_rope + v_dim) * 4;                      // mla_kvraw
  }
  // MoE router logits and the gather staging buffers.
  if (ir_.moe.n_expert > 0)
    per_row += static_cast<size_t>(ir_.moe.n_expert) * 4 + static_cast<size_t>(hidden_) * 4 * 2;
  // Logits: a last_row_only prefill head computes ONE row, so only the reserved rows
  // (default 1) are part of the footprint.  Keeping this in sync with the allocation
  // above is what lets the VRAM-derived chunk growth pick a much larger chunk.
  const int logits_rows =
      prefill_logits_rows_ > 0
          ? std::min<int>(prefill_logits_rows_, static_cast<int>(r))
          : static_cast<int>(r);
  return r * per_row +
         static_cast<size_t>(std::max(1, logits_rows)) * static_cast<size_t>(vocab_) * 4 +
         (2ull << 20);
}

void DenseRunner::ensure_logits_rows(int rows) {
  const size_t need = static_cast<size_t>(std::max(1, rows)) *
                      static_cast<size_t>(vocab_) * 4;
  if (logits_.size() >= need) return;
  logits_ = cuda::Buffer(need);
  std::fprintf(stderr,
               "[logits] grew the logits buffer to %d rows (%.0f MiB) for a full-logits "
               "call (prefill_logits_rows=%d)\n",
               std::max(1, rows), static_cast<double>(need) / 1048576.0, prefill_logits_rows_);
}

void DenseRunner::derive_ctx_limit(int requested) {
  auto& cfg = engine::config::Config::get();
  const char* kv_name = kv_type_ == 1 ? "q8_0" : (kv_type_ == 2 ? "q4_0" : "f32");
  n_kv_layers_ = 0;
  for (int i = 0; i < ir_.n_layer; ++i) {
    const bool has = !(ir_.ssm.variant != "none" && ir_.layers[i].kind != "attn");
    if (has) ++n_kv_layers_;
  }
  const bool ctx_auto = cfg.get_bool("ctx_auto");
  const bool want_auto = cfg.get_int("ctx") <= 0;   // --ctx 0 / ENGINE_CTX=0: derive
  const int hard = 1 << 20;                          // 1M absolute policy cap
  const int model_ctx = ir_.ctx_train > 0 ? static_cast<int>(ir_.ctx_train) : hard;
  int want = want_auto ? std::min(model_ctx, hard) : requested;
  if (want < 1) want = 1;
  if (!want_auto && requested > hard) want = hard;
  const size_t per_layer = kv_layer_bytes(1);
  const size_t kv_per_token = per_layer * static_cast<size_t>(n_kv_layers_);
  const int pre_rows = static_cast<int>(cfg.get_int("prefill_rows"));
  const int row_cap = pre_rows > 0 ? pre_rows : 1;
  // weights that stay resident (file bytes are the resident bytes for quantized GGUFs)
  const size_t weights = static_cast<size_t>(ir_.fixed_bytes());
  const size_t margin = static_cast<size_t>(cfg.get_int("vram_margin_mb")) << 20;
  const size_t workspace = static_cast<size_t>(cfg.get_int("workspace_mb")) << 20;
  const size_t reserve = margin + workspace;
  const size_t free_at_load = cuda::mem_info().free_bytes;
  // activation bytes as a function of the number of rows actually allocated
  auto act_bytes = [&](int rows) -> size_t { return activation_bytes(rows); };

  // ---- MoE CPU-offload models: the KV cache competes with GPU-resident experts -----
  // The old budget only subtracted ir_.fixed_bytes(), so on a 35B (20.33 GiB of experts,
  // 11-15 layers kept on the GPU) it claimed ctx 32768 fit while post-load free VRAM was
  // 1160 MiB < margin -> serve measured-clamped and reloaded the 60-90 s model three
  // times (258 s startup).  For a *large* KV request on a split MoE model, estimate the
  // bytes of experts that will stay on the GPU plus the calibrated load overhead
  // (workspace + vram_extra_reserve + allocator slack) before loading, so the clamp
  // happens on the first load.  Small-KV cases (MLA models, default 4096) stay on the
  // old path: their KV is tiny and the measured-clamp fallback is cheap.
  const bool split_moe =
      ir_.moe.n_expert > 0 && ir_.expert_bytes() > 0 && ir_.total_bytes() > free_at_load;
  const uint64_t want_kv = kv_per_token * static_cast<uint64_t>(want);
  // Only long explicit/auto requests are worth a pre-load clamp: small contexts (Coder's
  // 4096 default, 35B at 4096) keep the old verbatim path and the cheap measured-clamp
  // fallback, so this optimisation cannot change their behaviour.
  const bool budget_split = split_moe && want >= 16384 && want_kv > (256ull << 20);
  const size_t ledger_gap =
      static_cast<size_t>(std::max<int64_t>(0, cfg.get_int("vram_extra_reserve_mb"))) << 20;
  // Predict the loader's own residency: it keeps fixed + KV + workspace resident and
  // fills the remaining free VRAM with experts, leaving vram_margin(+96 MiB) free.
  // Measured on the 35B: the driver/allocator also eats ~vram_extra_reserve_mb beyond the
  // ledger (calibrated by that knob), so count it here as well.
  auto split_experts_at = [&](int seq, size_t* kv_out) -> size_t {
    const size_t overhead = workspace + ledger_gap + (96ull << 20);
    const int rows = std::min(seq, row_cap);
    const size_t kv = kv_per_token * static_cast<size_t>(seq);
    if (kv_out) *kv_out = kv;
    const size_t base_req = weights + kv + act_bytes(rows) + overhead + margin;
    return free_at_load > base_req
               ? std::min<size_t>(static_cast<size_t>(ir_.expert_bytes()),
                                   free_at_load - base_req)
               : 0;
  };
  int seq = want;
  int fit = want;
  if (budget_split) {
    // Keep the expert residency of the requested length fixed: the loader's adaptive
    // placement only promotes another layer once a full expert layer of free VRAM
    // appears, so this small clamp does not trigger a promotion.
    size_t kv_req = 0;
    const size_t experts = split_experts_at(want, &kv_req);
    const size_t overhead = workspace + ledger_gap + (96ull << 20);
    const size_t safety = 128ull << 20;
    const int rows = std::min(want, row_cap);
    const size_t base_fit = weights + experts + act_bytes(rows) + overhead + margin + safety;
    const size_t avail = free_at_load > base_fit ? free_at_load - base_fit : 0;
    const int next = kv_per_token ? static_cast<int>(avail / kv_per_token) : want;
    fit = std::min(want, std::max(1, next));
    (void)kv_req;
  } else {
    for (int it = 0; it < 4; ++it) {
      const int rows = std::min(seq, row_cap);
      const size_t act = act_bytes(rows);
      const size_t avail = free_at_load > weights + act + reserve
                               ? free_at_load - weights - act - reserve
                               : 0;
      fit = kv_per_token ? static_cast<int>(avail / kv_per_token) : want;
      fit = std::min(fit, want);
      if (fit >= seq) break;
      seq = fit;
      if (seq < 1) seq = 1;
    }
  }
  const bool split_clamp = budget_split && fit < want;
  const int final_seq = (ctx_auto || split_clamp) ? std::min(want, fit) : want;
  const size_t kv_bytes = kv_per_token * static_cast<size_t>(final_seq);
  std::fprintf(stderr,
               "[ctx] requested=%d->%d kv_type=%s kv/token=%zu B (%.2f KiB) "
               "kv_layers=%d/%d kv_total=%.0f MiB\n",
               requested, final_seq, kv_name, kv_per_token,
               kv_per_token / 1024.0, n_kv_layers_, ir_.n_layer,
               kv_bytes / 1048576.0);
  const int rows_used = std::min(final_seq, row_cap);
  size_t experts_used = 0;
  if (budget_split) {
    size_t kv_tmp = 0;
    experts_used = split_experts_at(want, &kv_tmp);
  }
  if (budget_split) {
    std::fprintf(stderr,
                 "[ctx] split budget: free=%.0f MiB - fixed=%.0f MiB - gpu_experts=%.0f MiB "
                 "- act(rows=%d)=%.0f MiB - reserve=%.0f MiB - ledger_gap=%.0f MiB -> "
                 "max_ctx=%d (split MoE: KV competes with GPU-resident experts)\n",
                 free_at_load / 1048576.0, weights / 1048576.0,
                 experts_used / 1048576.0, rows_used,
                 act_bytes(rows_used) / 1048576.0, reserve / 1048576.0,
                 ledger_gap / 1048576.0, fit);
  } else {
    std::fprintf(stderr,
                 "[ctx] budget: free=%.0f MiB - weights=%.0f MiB - act(rows=%d)=%.0f MiB "
                 "- reserve=%.0f MiB -> max_ctx=%d (%s)\n",
                 free_at_load / 1048576.0, weights / 1048576.0, rows_used,
                 act_bytes(rows_used) / 1048576.0, reserve / 1048576.0, fit,
                 ctx_auto ? "ctx_auto on" : "ctx_auto off");
  }
  if (final_seq < requested) {
    std::fprintf(stderr,
                 "[ctx] CLAMPED: %d > max_ctx %d for kv_type=%s (would need %.0f MiB, "
                 "budget %.0f MiB)\n",
                 requested, final_seq, kv_name,
                 kv_per_token * static_cast<size_t>(requested) / 1048576.0,
                 (free_at_load > weights + reserve ? (free_at_load - weights - reserve) : 0) /
                     1048576.0);
    for (int t = 0; t < 3; ++t) {
      const int save = kv_type_;
      kv_type_ = t;
      const size_t pt = kv_layer_bytes(1) * static_cast<size_t>(n_kv_layers_);
      const size_t act = act_bytes(row_cap);
      const size_t avail = free_at_load > weights + act + reserve
                               ? free_at_load - weights - act - reserve
                               : 0;
      std::fprintf(stderr, "[ctx]   with kv_type=%s: max_ctx=%d\n",
                   t == 0 ? "f32" : (t == 1 ? "q8_0" : "q4_0"),
                   pt ? static_cast<int>(avail / pt) : 0);
      kv_type_ = save;
    }
    if (budget_split) {
      size_t kv_tmp = 0;
      const size_t ex = split_experts_at(want, &kv_tmp);
      std::fprintf(stderr,
                   "[ctx] clamp reason: VRAM budget for a split MoE model - KV competes with "
                   "~%.0f MiB of GPU-resident experts + %.0f MiB ledger gap; one load is "
                   "enough (no reload chain)\n",
                   ex / 1048576.0, ledger_gap / 1048576.0);
    }
    if (cfg.get_bool("kv_offload"))
      std::fprintf(stderr,
                   "[ctx]   kv_offload=on: host-mapped KV is not implemented in this build; "
                   "the cache must fit in VRAM (see docs/CONFIG.md)\n");
  }
  max_seq_ = final_seq;
  // Kernel shared-memory limits: attn_cached_cfg stages max_seq floats of scores and
  // mla_attn_cached does the same.  Their dynamic shared memory scale is what actually
  // bounds the context on cards with a 48 KiB default (99 KiB opt-in on this GPU), so
  // clamp here with a clear message instead of failing inside cuLaunchKernel later.
  {
    auto smem_ctx_limit = [](void* fn, int extra_floats) -> int {
      size_t cap = cuda::max_dynamic_shared_mem(fn);
      if (cap == 0) cap = 48u * 1024u;   // default window when the opt-in is unsupported
      const int lim = static_cast<int>(cap / sizeof(float)) - extra_floats;
      return lim < 1 ? 1 : lim;
    };
    if (ext_enabled_ && k_attncfg_) {
      const int lim = smem_ctx_limit(k_attncfg_, 32);
      if (lim < max_seq_) {
        std::fprintf(stderr,
                     "[ctx] kernel-smem clamp: %d -> %d (attn_cached_cfg needs "
                     "(ctx+32)*4 bytes of dynamic shared memory, limit %d)\n",
                     max_seq_, lim, lim);
        max_seq_ = lim;
      }
    }
    if (ir_.attn.variant == "mla" && k_mlaattnc_) {
      const int lim = smem_ctx_limit(k_mlaattnc_, 0);
      if (lim < max_seq_) {
        std::fprintf(stderr,
                     "[ctx] kernel-smem clamp: %d -> %d (mla_attn_cached stages one "
                     "float per position in dynamic shared memory)\n",
                     max_seq_, lim);
        max_seq_ = lim;
      }
    }
  }
  kv_override_ctx_ = true;
}

int DenseRunner::n_moe_layers() const {
  int n = 0;
  for (const auto& ly : layers_)
    if (ly.is_moe) ++n;
  return n;
}

int DenseRunner::cpu_moe_layers() const {
  int n = 0;
  for (const auto& ly : layers_)
    if (ly.is_moe && ly.cpu_begin >= 0) ++n;
  return n;
}

void DenseRunner::alloc_kv_cache(LayerW& w) {
  if (ir_.attn.variant == "mla") {
    const size_t kc = static_cast<size_t>(n_head_) * max_seq_ * ir_.attn.key_length;
    const size_t vc = static_cast<size_t>(n_head_) * max_seq_ * ir_.attn.value_length;
    w.mla_kc = cuda::Buffer(kc * 4);
    w.mla_vc = cuda::Buffer(vc * 4);
    return;
  }
  if (kv_type_ == 0) {
    const size_t cache = static_cast<size_t>(n_kv_) * static_cast<size_t>(max_seq_) *
                         static_cast<size_t>(head_dim_);
    w.kc = cuda::Buffer(cache * 4);
    w.vc = cuda::Buffer(cache * 4);
  } else {
    const size_t rowb = static_cast<size_t>(kv_nb_of(head_dim_) * kv_bb_of(kv_type_));
    const size_t bytes = static_cast<size_t>(n_kv_) * static_cast<size_t>(max_seq_) * rowb;
    w.kc = cuda::Buffer(bytes);
    w.vc = cuda::Buffer(bytes);
  }
}

void DenseRunner::kv_write_layer(LayerW& L, int n_tokens, cuda::Buffer& k,
                                 cuda::Buffer& v) {
  int nkv = n_kv_, d = head_dim_, ms = max_seq_;
  void* pk = k.data();
  void* pv = v.data();
  void* pK = L.kc.data();
  void* pV = L.vc.data();
  void* pp = pos_dev_.data();
  if (kv_type_ == 0) {
    void* args[] = {&pk, &pv, &pK, &pV, &pp, &nkv, &d, &ms};
    cuda::launch(k_kvwrite_, n_tokens * n_kv_, 1, 1, 32, 1, 1, 0, args);
  } else {
    int kt = kv_type_;
    void* args[] = {&pk, &pv, &pK, &pV, &pp, &nkv, &d, &ms, &kt};
    cuda::launch(k_kvwriteq_, n_tokens * n_kv_, 1, 1, 128, 1, 1, 0, args);
  }
}

// Historical warp-per-(query,head) kernel: stages the whole score row in shared
// memory, so it only works while max_seq is small.  Kept for the short-context
// f32 path so the current decode numbers do not move.
void DenseRunner::run_attn_legacy(LayerW& L, int n_tokens, int start_pos, cuda::Buffer& q,
                                  cuda::Buffer& out) {
  int nh = n_head_, nkv = n_kv_, d = head_dim_, ms = max_seq_;
  void* pq = q.data();
  void* pK = L.kc.data();
  void* pV = L.vc.data();
  void* po = out.data();
  void* pp = pos_dev_.data();
  void* args[] = {&pq, &pK, &pV, &po, &pp, &nh, &nkv, &d, &ms};
  const int ctx_len = start_pos + n_tokens;
  int nwarp = attn_warps_for_ctx(ctx_len);
  auto smem_for = [&](int nw) -> size_t {
    return (static_cast<size_t>(max_seq_) + 32 +
            static_cast<size_t>(nw) * static_cast<size_t>(head_dim_)) *
           sizeof(float);
  };
  size_t smem_cap = cuda::max_dynamic_shared_mem(k_attnw_);
  if (smem_cap < 48u * 1024u) smem_cap = 48u * 1024u;
  while (nwarp > 1 && smem_for(nwarp) > smem_cap) nwarp >>= 1;
  cuda::launch(k_attnw_, n_tokens, n_head_, 1, 32 * nwarp, 1, 1, smem_for(nwarp), args);
}

void DenseRunner::run_attn(LayerW& L, int n_tokens, int start_pos, cuda::Buffer& q,
                           cuda::Buffer& out) {
  const int ctx_len = start_pos + n_tokens;
  const bool quant = kv_type_ != 0;
  // The legacy kernel is f32-only and its shared-memory footprint grows with max_seq,
  // so it is used only for the short f32 path this engine was tuned on.
  if (!quant && ctx_len < attn_flash_min_ && max_seq_ <= 12000) {
    run_attn_legacy(L, n_tokens, start_pos, q, out);
    return;
  }
  if (!kv_quant_ok()) {
    // head_dim > 256 or not a multiple of 32: no tiled path.  A quantized cache
    // cannot be decoded by the legacy kernel, so refuse loudly instead of reading
    // garbage.
    if (quant)
      throw std::runtime_error(
          "kv_type != f32 needs head_dim % 32 == 0 and head_dim <= 256 (model has " +
          std::to_string(head_dim_) + ")");
    run_attn_legacy(L, n_tokens, start_pos, q, out);
    return;
  }
  // Split-K only for small batches: a prefill batch already has n_tokens*n_head
  // independent warps, so it runs nsplit=1 and finishes the softmax in-register
  // (no scratch, no combine pass).
  const bool split = n_tokens <= attn_split_rows_;
  const int nsplit = split ? kv_split_ : 1;
  int nh = n_head_, nkv = n_kv_, d = head_dim_, ms = max_seq_, kt = kv_type_;
  int ns = nsplit, stride = head_dim_ + 2, rows_s = attn_scratch_rows_;
  void* pq = q.data();
  void* pKf = L.kc.data();
  void* pVf = L.vc.data();
  void* pKq = L.kc.data();
  void* pVq = L.vc.data();
  void* po = out.data();
  void* psc = attn_scratch_.data();
  void* pp = pos_dev_.data();
  void* args[] = {&pq,  &pKf, &pVf, &pKq,    &pVq, &po,  &psc,
                  &pp,  &nh,  &nkv, &d,      &ms,  &kt,  &ns, &stride};
  (void)rows_s;
  const int gqa_rep = n_kv_ > 0 ? n_head_ / n_kv_ : 1;
  if (!split && attn_pre_rows_ > 1 && k_attntiledpregqa_ && attn_gqa_ && kv_type_ == 0 &&
      head_dim_ == 256 && gqa_rep >= 1 && gqa_rep <= 4 && n_head_ % n_kv_ == 0) {
    // prefill: one warp owns a query row and the whole GQA group of its KV head, so
    // the K/V row is fetched once for all rep query heads (4x less K/V traffic).
    int rpb = attn_pre_rows_, rep = gqa_rep, nt = n_tokens;
    void* pargs[] = {&pq, &pKf, &pVf, &po, &pp, &nt, &nh, &nkv, &d, &ms, &rep};
    cuda::launch(k_attntiledpregqa_, (n_tokens + rpb - 1) / rpb, n_kv_, 1, 32 * rpb, 1, 1, 0,
                 pargs);
    return;
  }
  if (!split && attn_pre_rows_ > 1 && k_attntiledpre_) {
    // Prefill: 8 query rows per block so the K/V rows are fetched once and reused
    // through L1 by the other warps of the block (measured 444 ms -> see report).
    int rpb = attn_pre_rows_, nt = n_tokens;
    void* pargs[] = {&pq, &pKf, &pVf, &pKq, &pVq, &po, &pp, &nt, &nh, &nkv, &d, &ms, &kt};
    cuda::launch(k_attntiledpre_, (n_tokens + rpb - 1) / rpb, n_head_, 1, 32 * rpb, 1, 1, 0,
                 pargs);
    return;
  }
  // Decode (small batch, nsplit>1): one block per KV head with the rep query heads of
  // the GQA group; the K/V row fetched for one head is served from L1 to the others.
  // Same per-lane math / online softmax / scratch layout as attn_tiled, so results are
  // identical.  ctx-gated (attn_gqa_min_ctx) so short contexts can stay on the tuned
  // kernel if the wider block turns out slower there.
  const bool gqa_ok = split && nsplit > 1 && gqa_rep >= 2 && gqa_rep <= 8 && k_attntiledgqa_ &&
                      n_head_ % n_kv_ == 0 && attn_gqa_decode_ &&
                      ctx_len >= attn_gqa_min_ctx_;
  if (gqa_ok) {
    int wide = (kv_type_ == 0 && head_dim_ == 256 && attn_gqa_wide_) ? 1 : 0;
    int rep = gqa_rep;
    void* gargs[] = {&pq,  &pKf, &pVf, &pKq, &pVq, &po,  &psc, &pp,
                     &nh,  &nkv, &d,   &ms,  &kt,  &ns,  &stride, &rep, &wide};
    cuda::launch(k_attntiledgqa_, n_tokens, n_kv_, nsplit, 32 * gqa_rep, 1, 1, 0, gargs);
  } else {
    cuda::launch(k_attntiled_, n_tokens, n_head_, nsplit, 32, 1, 1, 0, args);
  }
  if (split && nsplit > 1) {
    int cb = head_dim_ < 256 ? head_dim_ : 256;
    void* a2[] = {&psc, &po, &nh, &d, &ns, &stride};
    cuda::launch(k_attncombine_, n_tokens, n_head_, 1, cb, 1, 1, 0, a2);
  }
}

namespace prof_acc {
void dump();
}

DenseRunner::~DenseRunner() {
  if (cuda::capturing_active())
    std::fprintf(stderr, "[cuda] WARN capture still open at shutdown\n");
  cuda::kprof_dump();
  prof_acc::dump();
}

DenseRunner::ExpertPlacement DenseRunner::debug_expert_placement(int layer) const {
  ExpertPlacement p;
  if (layer < 0 || layer >= static_cast<int>(layers_.size())) return p;
  const LayerW& L = layers_[static_cast<size_t>(layer)];
  p.gpu_experts = static_cast<int>(L.experts.size() + L.q_experts.size() / 3);
  const int ne = ir_.moe.n_expert;
  const bool quant_host = !L.h_gate_bytes.empty() && !L.h_up_bytes.empty() &&
                          !L.h_down_bytes.empty();
  p.host_experts = static_cast<int>(L.host_gate.size());
  if (quant_host) p.host_experts = ne;   // quantized bytes kept on the host (GGUF path)
  p.cpu_begin = L.cpu_begin;
  return p;
}

std::vector<float> DenseRunner::debug_mla_cache(int layer) {
  if (layer < 0 || layer >= static_cast<int>(layers_.size())) return {};
  auto& L = layers_[layer];
  if (!L.is_mla) return {};
  const size_t n = static_cast<size_t>(n_head_) * max_seq_ * ir_.attn.key_length;
  std::vector<float> out(n);
  cuda::stream_sync();
  L.mla_kc.download(out.data(), n * 4);
  return out;
}

// One rmsnorm launch.  The 1-block scalar kernel is latency bound on the widths this
// engine sees (4096/2048/512 all divisible by 4), so the float4 variant is used when
// enabled and the width allows it; both produce the same y = x*rsqrt(ms+eps)*w.
void DenseRunner::rms_raw(void* px, void* pw, void* py, int rows, int n) {
  float eps = eps_;
  const bool fast = rms4_ && k_rms4_ && (n % 4 == 0) && n >= 512;
  // 512 threads for the vectorised kernel: n/4 elements with 2 iterations per thread
  // (the warp-shuffle reduction handles any block size, shared only holds warp sums).
  // Thread count derived from the width instead of a hardcoded 2048 boundary:
  // aim for ~8 elements per thread, clamped to [256, 512] (the range measured to be
  // flat), so the same rule covers hidden=4096 and vocab=248320.
  int th = 256;
  if (fast) {
    int want = (n + 7) / 8;
    int p = 256;
    while (p < want && p < 512) p <<= 1;
    th = p;
  }
  void* args[] = {&px, &pw, &py, &n, &eps};
  cuda::launch(fast ? k_rms4_ : k_rms_, rows, 1, 1, th, 1, 1, 512 * sizeof(float), args);
}

cuda::Buffer DenseRunner::upload_norm(const std::vector<float>& v) const {
  // Some converters store zero-centred RMSNorm weights (w-1) even for architectures
  // whose HF weights are ones-centred (DeepSeek-V2). Detect it from the data: real
  // ones-centred weights average close to 1, zero-centred ones average well below.
  static const int force = static_cast<int>(engine::config::get_int("norm_plus1"));
  bool plus1 = false;
  if (force >= 0) {
    plus1 = force != 0;
  } else if (ir_.arch == "deepseek2" && v.size() >= 64) {
    // only DeepSeek-V2 files come from converters that may zero-centre the norms
    double sum = 0;
    for (float x : v) sum += x;
    plus1 = (sum / v.size()) < 0.8;   // ones-centred would sit near 1.0
  }
  if (!plus1) return upload(v);
  std::vector<float> w(v.size());
  for (size_t i = 0; i < v.size(); ++i) w[i] = v[i] + 1.0f;
  return upload(w);
}

std::vector<float> DenseRunner::tensor_f32(const std::string& name) const {
  if (container_ == Container::Safetensors) {
    auto shards = std::static_pointer_cast<st::Shards>(shards_);
    auto it = shards->tensors.find(name);
    if (it == shards->tensors.end())
      throw std::runtime_error("missing tensor " + name +
                             " (safetensors naming does not match model.layers.N.*)");
    const st::Tensor& t = it->second;
    std::ifstream f(t.file, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + t.file);
    f.seekg(static_cast<std::streamoff>(t.data_start + t.begin));
    const size_t nbytes = static_cast<size_t>(t.nbytes());
    if (t.dtype == "F32") {
      std::vector<float> out(nbytes / 4);
      read_exact(f, out.data(), nbytes, name);
      return out;
    }
    if (t.dtype == "BF16" || t.dtype == "F16") {
      std::vector<uint16_t> raw(nbytes / 2);
      read_exact(f, raw.data(), nbytes, name);
      std::vector<float> out(raw.size());
      for (size_t i = 0; i < raw.size(); ++i) {
        if (t.dtype == "BF16") {
          const uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
          std::memcpy(&out[i], &bits, 4);
        } else {  // F16
          out[i] = engine::fp16::half_to_float(raw[i]);
        }
      }
      return out;
    }
    throw std::runtime_error("unsupported dtype " + t.dtype + " for " + name);
  }

  // ---- GGUF ----
  auto g = std::static_pointer_cast<gguf::Model>(gguf_);
  const gguf::TensorInfo* ti = nullptr;
  for (const auto& t : g->tensors)
    if (t.name == name) ti = &t;
  if (!ti)
    throw std::runtime_error("missing tensor " + name +
                           " (GGUF naming does not match blk.N.*)");

  std::ifstream f(g->path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + g->path);
  f.seekg(static_cast<std::streamoff>(g->data_offset + ti->offset));
  std::vector<unsigned char> raw(static_cast<size_t>(ti->nbytes));
  read_exact(f, raw.data(), raw.size(), name);

  if (ti->type_name == "F32") {
    std::vector<float> out(raw.size() / 4);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
  }
  if (ti->type_name == "F16" || ti->type_name == "BF16") {
    std::vector<float> out(raw.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
      const uint16_t h = static_cast<uint16_t>(raw[2 * i] | (raw[2 * i + 1] << 8));
      if (ti->type_name == "BF16") {
        const uint32_t bits = static_cast<uint32_t>(h) << 16;
        std::memcpy(&out[i], &bits, 4);
      } else {
        out[i] = engine::fp16::half_to_float(h);
      }
    }
    return out;
  }

  const QuantKernel* qk = quant_kernel(ti->type_name);
  if (!qk)
    throw std::runtime_error(
        "quant type " + ti->type_name + " is not implemented (tensor '" + name +
        "'). This engine implements: Q4_K, Q5_K, Q6_K, Q8_0, Q5_0, Q5_1 (and F32/F16/BF16). "
        "Convert the model with llama.cpp's quantize tool (e.g. Q4_K_M / Q5_K_M).");
  const long long n_blocks = static_cast<long long>(raw.size() / qk->block_bytes);
  cuda::Buffer db(raw.size()), dout(static_cast<size_t>(ti->n_elements) * 4);
  db.upload(raw.data(), raw.size());
  void* fn = mod_->kernel(qk->name);
  void* pb = db.data();
  void* po = dout.data();
  long long nb = n_blocks;
  void* args[] = {&pb, &po, &nb};
  cuda::launch(fn, static_cast<int>(n_blocks), 1, 1, qk->threads, 1, 1, 0, args);
  cuda::sync();
  std::vector<float> out(static_cast<size_t>(ti->n_elements));
  dout.download(out.data(), out.size() * 4);
  return out;
}

// ===========================================================================
// Generic extended dense path (metadata-driven architectures)
// ===========================================================================
// One "extended" layer is built from optional tensor blocks, each enabled by the
// presence of its tensors in the GGUF:
//   attn_norm / ffn_norm                  pre-norms
//   attn_q / attn_k / attn_v / attn_output projections (v optional -> k_eq_v)
//   attn_q_norm / attn_k_norm             per-head QK RMSNorm
//   post_attention_norm / post_ffw_norm   sandwich (output) norms
//   layer_output_scale                    per-layer output scalar
//   ffn_gate / ffn_up / ffn_down          MLP (GELU-tanh or SiLU)
// and per-layer geometry comes from ir::LayerAttn (head counts / head dims /
// rope dim+base / sliding window), which the IR derives from metadata arrays.

void DenseRunner::load_ext_layer(int i, LayerW& w) {
  auto gg = std::static_pointer_cast<gguf::Model>(gguf_);
  auto find_t = [&](const std::string& name) -> const gguf::TensorInfo* {
    for (const auto& t : gg->tensors)
      if (t.name == name) return &t;
    return nullptr;
  };
  const std::string p = "blk." + std::to_string(i) + ".";
  w.ext = true;
  w.ext_attn = ir_.layers[i].attn;
  const auto& la = w.ext_attn;
  w.q_dim = n_head_ * la.key_dim;
  w.kv_dim = la.n_kv * la.key_dim;

  auto norm_w = [&](const std::string& name) {
    std::vector<float> v = tensor_f32(name);
    if (ext_norm_plus1_ > 0)
      for (float& x : v) x += 1.0f;
    return upload(v);
  };
  w.ln1 = norm_w(p + "attn_norm.weight");
  w.ln2 = norm_w(p + "ffn_norm.weight");
  if (la.sandwich) {
    w.post_attn_norm = norm_w(p + "post_attention_norm.weight");
    w.post_ffw_norm = norm_w(p + "post_ffw_norm.weight");
  }
  if (la.layer_scalar) {
    std::vector<float> sv = tensor_f32(p + "layer_output_scale.weight");
    if (!sv.empty()) w.layer_scalar = static_cast<double>(sv[0]);
  }
  if (la.qk_norm) {
    w.q_norm_w = upload(tensor_f32(p + "attn_q_norm.weight"));
    w.k_norm_w = upload(tensor_f32(p + "attn_k_norm.weight"));
  }
  if (la.v_norm || !la.has_v) {
    std::vector<float> ones(static_cast<size_t>(la.value_dim), 1.0f);
    w.ones_head = upload(ones);
  }

  // attention projections: prefer the quantized-resident path (decode bandwidth)
  const bool qq = load_qtensor(p + "attn_q.weight", w.qq);
  const bool qk = load_qtensor(p + "attn_k.weight", w.qk);
  const bool qo = load_qtensor(p + "attn_output.weight", w.qo);
  const bool qv = la.has_v ? load_qtensor(p + "attn_v.weight", w.qv) : true;
  w.is_quant = qq && qk && qo && qv;
  if (!w.is_quant) {
    w.qq = QTensor(); w.qk = QTensor(); w.qo = QTensor(); w.qv = QTensor();
    w.q = upload(tensor_f32(p + "attn_q.weight"));
    w.k = upload(tensor_f32(p + "attn_k.weight"));
    if (la.has_v) w.v = upload(tensor_f32(p + "attn_v.weight"));
    w.o = upload(tensor_f32(p + "attn_output.weight"));
  }
  const bool fg = load_qtensor(p + "ffn_gate.weight", w.qg);
  const bool fu = load_qtensor(p + "ffn_up.weight", w.qu);
  const bool fd = load_qtensor(p + "ffn_down.weight", w.qd);
  if (!(fg && fu && fd)) {
    w.qg = QTensor(); w.qu = QTensor(); w.qd = QTensor();
    w.gate = upload(tensor_f32(p + "ffn_gate.weight"));
    w.up = upload(tensor_f32(p + "ffn_up.weight"));
    w.down = upload(tensor_f32(p + "ffn_down.weight"));
  }

  // per-layer rotary frequencies (same convention as the rope kernels:
  // angle = position * freqs[i], freqs[i] = base^(-2i/rope_dim))
  const int nr = la.rope_dim > 0 ? la.rope_dim : la.key_dim;
  std::vector<float> fr(static_cast<size_t>(nr / 2));
  for (int k = 0; k < nr / 2; ++k)
    fr[static_cast<size_t>(k)] =
        static_cast<float>(std::pow(la.rope_base, -2.0 * static_cast<double>(k) / nr));
  w.rope_freqs = cuda::Buffer(fr.size() * 4);
  w.rope_freqs.upload(fr.data(), fr.size() * 4);

  // per-layer KV cache, [n_kv][max_seq][head_dim]
  const size_t kc_n = static_cast<size_t>(la.n_kv) * max_seq_ *
                      static_cast<size_t>(la.key_dim > la.value_dim ? la.key_dim : la.value_dim);
  w.kc = cuda::Buffer(kc_n * 4);
  w.vc = cuda::Buffer(kc_n * 4);
  if (debug_)
    std::fprintf(stderr,
                 "[arch] layer %2d n_kv=%d d=%d vd=%d rope_dim=%d base=%.0f window=%d "
                 "has_v=%d qk_norm=%d v_norm=%d sandwich=%d scalar=%.3f quant=%d\n",
                 i, la.n_kv, la.key_dim, la.value_dim, la.rope_dim, la.rope_base, la.window,
                 (int)la.has_v, (int)la.qk_norm, (int)la.v_norm, (int)la.sandwich,
                 w.layer_scalar, (int)w.is_quant);
}

void DenseRunner::ext_attention(LayerW& L, int n_tokens, int start_pos) {
  const auto& la = L.ext_attn;
  const int d = la.key_dim;
  const int vd = la.value_dim;
  const int nh = n_head_;
  const int nkv = la.n_kv;
  auto proj = [&](QTensor& qt, cuda::Buffer& f32w, int n_out, int n_in, cuda::Buffer& in,
                  cuda::Buffer& out) {
    if (qt.valid) {
      matmul(qt, n_tokens, in, out);
    } else if (n_tokens == 1) {
      matvec_f32(f32w, n_out, n_in, in, out);
    } else {
      cublas::gemm_f32_nt(n_tokens, n_out, n_in, static_cast<const float*>(in.data()),
                          static_cast<const float*>(f32w.data()),
                          static_cast<float*>(out.data()));
    }
  };
  auto rms_rows = [&](cuda::Buffer& x, cuda::Buffer& wt, cuda::Buffer& y, int rows, int n) {
    int nn = n;
    float e = eps_;
    void* px = x.data();
    void* pw = wt.data();
    void* py = y.data();
    void* args[] = {&px, &pw, &py, &nn, &e};
    const size_t sh = 256 * sizeof(float);
    cuda::launch(k_rms_, rows, 1, 1, 256, 1, 1, sh, args);
  };
  auto rope_dev = [&](cuda::Buffer& buf, int heads, int stride, int rope_dim,
                      cuda::Buffer& freqs) {
    int nhh = heads, hs = stride, rd = rope_dim, off = 0;
    float ms = 1.0f;
    void* px = buf.data();
    void* pp = pos_dev_.data();
    void* pf = freqs.data();
    void* args[] = {&px, &pp, &nhh, &hs, &rd, &off, &pf, &ms};
    const int threads = 128;
    cuda::launch(k_ropestrided_, n_tokens, heads, 1, threads, 1, 1, 0, args);
  };

  // q / k / v projections (k raw is also the source of V for k_eq_v layers)
  proj(L.qq, L.q, L.q_dim, hidden_, hn_, q_);
  proj(L.qk, L.k, nkv * d, hidden_, hn_, k_);
  if (la.has_v) {
    proj(L.qv, L.v, nkv * vd, hidden_, hn_, v_);
  } else {
    // attention_k_eq_v: V = weightless RMSNorm(K_proj) (before k-norm / rope)
    rms_rows(k_, L.ones_head, v_, n_tokens * nkv, vd);
  }
  if (la.has_v && la.v_norm) rms_rows(v_, L.ones_head, v_, n_tokens * nkv, vd);
  if (la.qk_norm) {
    rms_rows(k_, L.k_norm_w, k_, n_tokens * nkv, d);
    rms_rows(q_, L.q_norm_w, q_, n_tokens * nh, d);
  }
  rope_dev(q_, nh, d, la.rope_dim > 0 ? la.rope_dim : d, L.rope_freqs);
  rope_dev(k_, nkv, d, la.rope_dim > 0 ? la.rope_dim : d, L.rope_freqs);

  // KV cache write (f32 path; extended models keep f32 KV)
  {
    int nk = nkv, dd = d, ms = max_seq_;
    void* pk = k_.data();
    void* pv = v_.data();
    void* pK = L.kc.data();
    void* pV = L.vc.data();
    void* pp = pos_dev_.data();
    void* args[] = {&pk, &pv, &pK, &pV, &pp, &nk, &dd, &ms};
    cuda::launch(k_kvwrite_, n_tokens * nkv, 1, 1, 32, 1, 1, 0, args);
  }
  // attention (configurable scale + sliding window)
  {
    float scale = ext_attn_scale_ > 0 ? static_cast<float>(ext_attn_scale_)
                                      : 1.0f / std::sqrt(static_cast<float>(d));
    int window = la.window;
    int nhh = nh, nk = nkv, dd = d, ms = max_seq_;
    void* pq = q_.data();
    void* pK = L.kc.data();
    void* pV = L.vc.data();
    void* po = attn_.data();
    void* pp = pos_dev_.data();
    void* args[] = {&pq, &pK, &pV, &po, &pp, &nhh, &nk, &dd, &ms, &scale, &window};
    const size_t sh = (static_cast<size_t>(max_seq_) + 32) * sizeof(float);
    // Without this the launch is rejected once (max_seq+32)*4 > 48 KiB (ctx > 12256);
    // the ctx is also clamped to the returned limit at load time (see the ctor).
    cuda::max_dynamic_shared_mem(k_attncfg_);
    cuda::launch(k_attncfg_, n_tokens, nh, 1, 32, 1, 1, sh, args);
  }
  // output projection + sandwich norm + residual
  proj(L.qo, L.o, hidden_, L.q_dim, attn_, proj_);
  if (la.sandwich) rms_rows(proj_, L.post_attn_norm, proj_, n_tokens, hidden_);
  {
    void* px = x_.data();
    void* pb = proj_.data();
    long long total = static_cast<long long>(n_tokens) * hidden_;
    void* args[] = {&px, &pb, &total};
    cuda::launch(k_add_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
}

void DenseRunner::ext_ffn(LayerW& L, int n_tokens) {
  auto proj = [&](QTensor& qt, cuda::Buffer& f32w, int n_out, int n_in, cuda::Buffer& in,
                  cuda::Buffer& out) {
    if (qt.valid) {
      matmul(qt, n_tokens, in, out);
    } else if (n_tokens == 1) {
      matvec_f32(f32w, n_out, n_in, in, out);
    } else {
      cublas::gemm_f32_nt(n_tokens, n_out, n_in, static_cast<const float*>(in.data()),
                          static_cast<const float*>(f32w.data()),
                          static_cast<float*>(out.data()));
    }
  };
  auto rms_rows = [&](cuda::Buffer& x, cuda::Buffer& wt, cuda::Buffer& y, int rows, int n) {
    int nn = n;
    float e = eps_;
    void* px = x.data();
    void* pw = wt.data();
    void* py = y.data();
    void* args[] = {&px, &pw, &py, &nn, &e};
    const size_t sh = 256 * sizeof(float);
    cuda::launch(k_rms_, rows, 1, 1, 256, 1, 1, sh, args);
  };
  rms_rows(x_, L.ln2, hn_, n_tokens, hidden_);
  proj(L.qg, L.gate, inter_, hidden_, hn_, gate_);
  proj(L.qu, L.up, inter_, hidden_, hn_, up_);
  {
    long long total = static_cast<long long>(n_tokens) * inter_;
    void* pg = gate_.data();
    void* pu = up_.data();
    void* pa = act_.data();
    void* args[] = {&pg, &pu, &pa, &total};
    void* fn = ext_gelu_ ? k_gelumul_ : k_silu_;
    cuda::launch(fn, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
  proj(L.qd, L.down, hidden_, inter_, act_, proj_);
  if (L.ext_attn.sandwich) rms_rows(proj_, L.post_ffw_norm, proj_, n_tokens, hidden_);
  {
    void* px = x_.data();
    void* pb = proj_.data();
    long long total = static_cast<long long>(n_tokens) * hidden_;
    void* args[] = {&px, &pb, &total};
    cuda::launch(k_add_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
  if (L.layer_scalar != 1.0) {
    void* px = x_.data();
    float sc = static_cast<float>(L.layer_scalar);
    long long total = static_cast<long long>(n_tokens) * hidden_;
    void* args[] = {&px, &sc, &total};
    cuda::launch(k_scale_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
}

void DenseRunner::run_layer_ext(int layer, int n_tokens, int start_pos) {
  auto& L = layers_[layer];
  const auto& la = L.ext_attn;
  // pre-norm -> attention block -> sandwich norm -> residual
  rms_buf(x_, L.ln1, hn_, n_tokens, hidden_);
  ext_attention(L, n_tokens, start_pos);
  // pre-norm -> MLP block -> sandwich norm -> residual -> layer scalar
  ext_ffn(L, n_tokens);
  if (debug_ && layer == 0 && n_tokens > 0)
    std::fprintf(stderr, "[arch] layer 0 ran (n_tokens=%d pos=%d n_kv=%d d=%d)\\n", n_tokens,
                 start_pos, la.n_kv, la.key_dim);
}

void DenseRunner::load_weights() {
  const bool gguf = container_ == Container::Gguf;
  auto L = [&](int i, const char* suffix) {
    return gguf ? ("blk." + std::to_string(i) + "." + suffix)
                : ("model.layers." + std::to_string(i) + "." + suffix);
  };
  auto has = [&](const std::string& n) {
    if (gguf) {
      auto g = std::static_pointer_cast<gguf::Model>(gguf_);
      for (const auto& t : g->tensors)
        if (t.name == n) return true;
      return false;
    }
    auto shards = std::static_pointer_cast<st::Shards>(shards_);
    return shards->tensors.count(n) > 0;
  };

  {
    const std::string emb_name = gguf ? "token_embd.weight" : "model.embed_tokens.weight";
    bool quant_embed = false;
    if (gguf) {
      auto g = std::static_pointer_cast<gguf::Model>(gguf_);
      for (const auto& t : g->tensors)
        if (t.name == emb_name) {
          const QuantKernel* qk = quant_kernel(t.type_name);
          if (qk && hidden_ % qk->block_elems == 0) {
            embed_raw_ = read_gguf_raw(emb_name);
            embed_type_ = t.type_name;
            embed_bb_ = qk->block_bytes;
            embed_be_ = qk->block_elems;
            quant_embed = true;
          }
        }
    }
    if (!quant_embed) embed_host_ = tensor_f32(emb_name);
  }
  const std::string head_name = gguf ? "output.weight" : "lm_head.weight";
  tied_head_ = !has(head_name);
  embed_lazy_f32_ = container_ == Container::Gguf;
  if (tied_head_) lm_head_host_ = embed_host_;
  if (ext_enabled_ && ext_norm_plus1_ > 0) {
    // explicit override only for the generic path (arch_norm_mode=plus1)
    std::vector<float> fv = tensor_f32(gguf ? "output_norm.weight" : "model.norm.weight");
    for (float& x : fv) x += 1.0f;
    w_final_norm_ = upload(fv);
  } else {
    // upload_norm() keeps the architecture-specific zero-centering convention
    // (e.g. deepseek2 MLA); never bypass it for the legacy paths.
    w_final_norm_ = upload_norm(tensor_f32(gguf ? "output_norm.weight" : "model.norm.weight"));
  }

  layers_.resize(static_cast<size_t>(ir_.n_layer));
  for (int i = 0; i < ir_.n_layer; ++i) {
    auto& w = layers_[i];
    if (ext_enabled_ && gguf) {
      load_ext_layer(i, w);
      continue;
    }
    if (ir_.ssm.variant != "none" && gguf) {
      if (ir_.layers[i].kind == "linear")
        load_q35_linear(i, w);
      else
        load_q35_attn(i, w);
      continue;
    }
    if (gguf) {
      w.ln1 = upload_norm(tensor_f32(L(i, "attn_norm.weight")));
      if (ir_.attn.variant == "mla") {
        w.is_mla = true;
        w.wkvan = upload_norm(tensor_f32(L(i, "attn_kv_a_norm.weight")));
        // Keep the projections in their GGUF quant format: dequantising them to f32
        // costs ~142 MB per layer and makes decode read 3.7x more bytes.
        const bool qa = load_qtensor(L(i, "attn_q.weight"), w.qwq_mla);
        const bool qb = load_qtensor(L(i, "attn_kv_a_mqa.weight"), w.qwkva);
        const bool qc = load_qtensor(L(i, "attn_kv_b.weight"), w.qwkb);
        const bool qd = load_qtensor(L(i, "attn_output.weight"), w.qwo_mla);
        w.mla_quant = qa && qb && qc && qd;
        if (std::getenv("ENGINE_DBG_LOAD") && i < 2)
          std::fprintf(stderr, "[load] layer %d mla quant=%d types=%s,%s,%s,%s\n", i,
                       static_cast<int>(w.mla_quant), w.qwq_mla.type.c_str(),
                       w.qwkva.type.c_str(), w.qwkb.type.c_str(), w.qwo_mla.type.c_str());
        if (!w.mla_quant) {
          w.wq_mla = upload(tensor_f32(L(i, "attn_q.weight")));
          w.wkva = upload(tensor_f32(L(i, "attn_kv_a_mqa.weight")));
          w.wkb = upload(tensor_f32(L(i, "attn_kv_b.weight")));
          w.wo_mla = upload(tensor_f32(L(i, "attn_output.weight")));
        }
        const size_t kc_elems = static_cast<size_t>(n_head_) * max_seq_ * ir_.attn.key_length;
        const size_t vc_elems = static_cast<size_t>(n_head_) * max_seq_ * ir_.attn.value_length;
        w.mla_kc = cuda::Buffer(kc_elems * 4);
        w.mla_vc = cuda::Buffer(vc_elems * 4);
      } else {
        const bool qq = load_qtensor(L(i, "attn_q.weight"), w.qq);
        const bool qk = load_qtensor(L(i, "attn_k.weight"), w.qk);
        const bool qv = load_qtensor(L(i, "attn_v.weight"), w.qv);
        const bool qo = load_qtensor(L(i, "attn_output.weight"), w.qo);
        if (qq && qk && qv && qo) {
          w.is_quant = true;
          if (prefill_f32_) {
            w.q = upload(tensor_f32(L(i, "attn_q.weight")));
            w.k = upload(tensor_f32(L(i, "attn_k.weight")));
            w.v = upload(tensor_f32(L(i, "attn_v.weight")));
            w.o = upload(tensor_f32(L(i, "attn_output.weight")));
            w.has_f32 = true;
          }
        } else {
          w.q = upload(tensor_f32(L(i, "attn_q.weight")));
          w.k = upload(tensor_f32(L(i, "attn_k.weight")));
          w.v = upload(tensor_f32(L(i, "attn_v.weight")));
          w.o = upload(tensor_f32(L(i, "attn_output.weight")));
        }
      }
      w.ln2 = upload_norm(tensor_f32(L(i, "ffn_norm.weight")));
      if (ir_.moe.n_expert && ir_.layers[i].n_expert_tensors > 0) {
        w.is_moe = true;
        if (has(L(i, "ffn_gate_shexp.weight"))) {
          const auto g = tensor_f32(L(i, "ffn_gate_shexp.weight"));
          w.has_shared = true;
          w.sh_inter = static_cast<int>(g.size() / static_cast<size_t>(hidden_));
          // quantized copy when possible: the f32 form costs ~69 MB per layer
          // (3 x 2816 x 2048 x 4) vs ~18 MB quantized
          const bool qg = load_qtensor(L(i, "ffn_gate_shexp.weight"), w.qsh_gate);
          const bool qu = load_qtensor(L(i, "ffn_up_shexp.weight"), w.qsh_up);
          const bool qd = load_qtensor(L(i, "ffn_down_shexp.weight"), w.qsh_down);
          w.sh_quant = qg && qu && qd;
          if (!w.sh_quant) {
            w.sh_gate = upload(g);
            w.sh_up = upload(tensor_f32(L(i, "ffn_up_shexp.weight")));
            w.sh_down = upload(tensor_f32(L(i, "ffn_down_shexp.weight")));
          }
          if (has(L(i, "ffn_gate_inp_shexp.weight")))
            w.sh_gate_inp = upload(tensor_f32(L(i, "ffn_gate_inp_shexp.weight")));
          if (std::getenv("ENGINE_DBG_LOAD") && i < 2)
            std::fprintf(stderr, "[load] layer %d shared quant=%d type=%s sh_inter=%d\n", i,
                         static_cast<int>(w.sh_quant), w.qsh_gate.type.c_str(), w.sh_inter);
        }
        w.router = upload(tensor_f32(L(i, "ffn_gate_inp.weight")));
        const std::string gate_name = L(i, "ffn_gate_exps.weight");
        const std::string up_name = L(i, "ffn_up_exps.weight");
        const std::string down_name = L(i, "ffn_down_exps.weight");
        // quantized-resident experts (no f32 expansion) unless CPU experts are requested.
        // gate/up/down may use different quant types, so each is handled on its own.
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
        if (gpu_moe_layers_ >= 0 && gpu_experts_ < 0) {
          // count MoE layers before this one to decide GPU/CPU placement
          int idx = 0;
          for (int j = 0; j < i; ++j)
            if (ir_.layers[j].n_expert_tensors > 0) ++idx;
          moe_idx = idx;
          gpu_n = (idx < gpu_moe_layers_) ? ne_all : 0;
        }
        w.cpu_begin = gpu_n;
        if (std::getenv("ENGINE_DBG_LOAD2") && (i == 1 || i == 20))
          std::fprintf(stderr,
                       "[load2] layer %d ne=%d gpu_n=%d (ml=%d ge=%d) expert_inter=%d hidden=%d\n",
                       i, ne_all, gpu_n, gpu_moe_layers_, gpu_experts_, expert_inter_, hidden_);
        if (qk_g && qk_u && qk_d && g_elems % qk_g->block_elems == 0 &&
            g_elems % qk_u->block_elems == 0 && d_elems % qk_d->block_elems == 0) {
          const size_t gb_g = g_elems / qk_g->block_elems * qk_g->block_bytes;
          const size_t gb_u = g_elems / qk_u->block_elems * qk_u->block_bytes;
          const size_t db = d_elems / qk_d->block_elems * qk_d->block_bytes;
          // Adaptive placement: validate the planned GPU residency against *measured*
          // free VRAM (not just the arithmetic estimate) before committing the layer.
          // The arithmetic estimate in the planning block above is deliberately
          // conservative (it reserves scratch/KV that the real allocator may not use),
          // so the *measured* check is authoritative in both directions: it can still
          // evict a planned layer, and it may also extend GPU residency by one layer
          // when the measured free VRAM actually allows it (contiguity is preserved by
          // only trying the immediately next layer; the post-load check then verifies
          // that the configured margin survives).
          if (moe_idx >= 0) {
            const size_t layer_dev = static_cast<size_t>(ne_all) * (gb_g + gb_u + db);
            const size_t kv_extra =
                static_cast<size_t>(n_kv_) * max_seq_ * head_dim_ * 2 * 4;
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
          w.q_experts.resize(static_cast<size_t>(gpu_n) * 3);
          for (int e = 0; e < gpu_n; ++e) {
            load_qtensor_span(gate_name, e * gb_g, gb_g, expert_inter_, hidden_,
                              w.q_experts[e * 3 + 0]);
            load_qtensor_span(up_name, e * gb_u, gb_u, expert_inter_, hidden_,
                              w.q_experts[e * 3 + 1]);
            load_qtensor_span(down_name, e * db, db, hidden_, expert_inter_,
                              w.q_experts[e * 3 + 2]);
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
          if (std::getenv("ENGINE_DBG_LOAD2") && (i == 1 || i == 20))
            std::fprintf(stderr, "[load2] layer %d quantized-spans path, gpu_n=%d\n", i, gpu_n);
          if (gpu_n < ne_all) {
            // CPU experts keep their quantized bytes on the host: no f32 expansion
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
          // #10 (HIGH): F16/F32 (or mixed-type) experts land here.  This branch used to
          // upload EVERY expert to the GPU and leave the host-side buffers empty, so any
          // layer the placement decided to keep on the CPU had no data at all:
          // run_moe's host branch indexes w.host_gate/w.host_up/w.host_down by expert
          // (the vector is empty -> out-of-range / garbage output).  Mirror the
          // quantized path: keep f32 host copies for every expert and upload only the
          // GPU-resident ones (gpu_n from the layer-granular placement above).
          if (std::getenv("ENGINE_DBG_LOAD2"))
            std::fprintf(stderr, "[load2] layer %d -> F32 fallback (gpu_n=%d/%d)\n", i, gpu_n,
                         ne_all);
          const auto gate_all = tensor_f32(gate_name);
          const auto up_all = tensor_f32(up_name);
          const auto down_all = tensor_f32(down_name);
          const bool need_host = gpu_n < ne_all;
          w.cpu_begin = gpu_n;
          w.experts.resize(static_cast<size_t>(gpu_n));
          if (need_host) {
            w.host_gate.clear();
            w.host_up.clear();
            w.host_down.clear();
            w.host_gate.reserve(static_cast<size_t>(ne_all));
            w.host_up.reserve(static_cast<size_t>(ne_all));
            w.host_down.reserve(static_cast<size_t>(ne_all));
          }
          for (int e = 0; e < ne_all; ++e) {
            std::vector<float> g(gate_all.begin() + static_cast<long>(e) * g_elems,
                                 gate_all.begin() + static_cast<long>(e + 1) * g_elems);
            std::vector<float> u(up_all.begin() + static_cast<long>(e) * g_elems,
                                 up_all.begin() + static_cast<long>(e + 1) * g_elems);
            std::vector<float> dn(down_all.begin() + static_cast<long>(e) * d_elems,
                                  down_all.begin() + static_cast<long>(e + 1) * d_elems);
            if (e < gpu_n) {
              w.experts[e].gate = upload(g);
              w.experts[e].up = upload(u);
              w.experts[e].down = upload(dn);
            }
            if (need_host) {
              w.host_gate.push_back(std::move(g));
              w.host_up.push_back(std::move(u));
              w.host_down.push_back(std::move(dn));
            }
          }
          // The host matvec path needs the tensor types even for f32 experts (the CPU
          // dispatcher reads h_type_* to pick the row kernel).
          w.h_type_g = "F32";
          w.h_type_u = "F32";
          w.h_type_d = "F32";
        }
      } else {
        const bool qg = load_qtensor(L(i, "ffn_gate.weight"), w.qg);
        const bool qu = load_qtensor(L(i, "ffn_up.weight"), w.qu);
        const bool qd = load_qtensor(L(i, "ffn_down.weight"), w.qd);
        if (qg && qu && qd) {
          w.is_quant = true;
          if (prefill_f32_) {
            w.gate = upload(tensor_f32(L(i, "ffn_gate.weight")));
            w.up = upload(tensor_f32(L(i, "ffn_up.weight")));
            w.down = upload(tensor_f32(L(i, "ffn_down.weight")));
            w.has_f32 = true;
          }
        } else {
          w.gate = upload(tensor_f32(L(i, "ffn_gate.weight")));
          w.up = upload(tensor_f32(L(i, "ffn_up.weight")));
          w.down = upload(tensor_f32(L(i, "ffn_down.weight")));
          w.is_quant = false;
        }
      }
      w.has_bias = has(L(i, "attn_q.bias"));
      if (w.has_bias) {
        w.bq = upload(tensor_f32(L(i, "attn_q.bias")));
        w.bk = upload(tensor_f32(L(i, "attn_k.bias")));
        w.bv = upload(tensor_f32(L(i, "attn_v.bias")));
      }
    } else {
      // dequantized safetensors weights already come as f32/bf16
      auto conv = [&](const std::string& n) { return upload(tensor_f32(n)); };
      w.ln1 = conv(L(i, "input_layernorm.weight"));
      w.q = conv(L(i, "self_attn.q_proj.weight"));
      w.k = conv(L(i, "self_attn.k_proj.weight"));
      w.v = conv(L(i, "self_attn.v_proj.weight"));
      w.o = conv(L(i, "self_attn.o_proj.weight"));
      w.ln2 = conv(L(i, "post_attention_layernorm.weight"));
      w.has_bias = has(L(i, "self_attn.q_proj.bias"));
      if (w.has_bias) {
        w.bq = conv(L(i, "self_attn.q_proj.bias"));
        w.bk = conv(L(i, "self_attn.k_proj.bias"));
        w.bv = conv(L(i, "self_attn.v_proj.bias"));
      }
      if (ir_.moe.n_expert && ir_.layers[i].n_expert_tensors > 0) {
        w.is_moe = true;
        if (has(L(i, "mlp.shared_experts.gate_proj.weight"))) {
          w.has_shared = true;
          const auto g = tensor_f32(L(i, "mlp.shared_experts.gate_proj.weight"));
          w.sh_inter = static_cast<int>(g.size() / static_cast<size_t>(hidden_));
          w.sh_gate = upload(g);
          w.sh_up = upload(tensor_f32(L(i, "mlp.shared_experts.up_proj.weight")));
          w.sh_down = upload(tensor_f32(L(i, "mlp.shared_experts.down_proj.weight")));
        }
        w.router = conv(L(i, "mlp.gate.weight"));
        // #26: with an explicit expert budget (gpu_experts_ >= 0) the CPU path keeps host
        // copies, but this loop still uploaded *every* expert to the GPU as well, so the
        // experts kept for the CPU were resident twice (host + device).  Upload only the
        // GPU-resident prefix and keep host copies for the rest, like the GGUF path.
        const int st_ne = ir_.moe.n_expert;
        const int st_gpu_n = (gpu_experts_ < 0) ? st_ne : std::min(gpu_experts_, st_ne);
        w.cpu_begin = st_gpu_n;
        w.experts.resize(static_cast<size_t>(st_gpu_n));
        const bool st_need_host = st_gpu_n < st_ne;
        if (st_need_host) {
          w.host_gate.clear();
          w.host_up.clear();
          w.host_down.clear();
        }
        for (int e = 0; e < st_ne; ++e) {
          const std::string ep = "model.layers." + std::to_string(i) + ".mlp.experts." +
                                 std::to_string(e) + ".";
          auto g = tensor_f32(ep + "gate_proj.weight");
          auto u = tensor_f32(ep + "up_proj.weight");
          auto dn = tensor_f32(ep + "down_proj.weight");
          if (e < st_gpu_n) {
            w.experts[e].gate = upload(g);
            w.experts[e].up = upload(u);
            w.experts[e].down = upload(dn);
          }
          if (st_need_host) {
            w.host_gate.push_back(std::move(g));
            w.host_up.push_back(std::move(u));
            w.host_down.push_back(std::move(dn));
          }
        }
        if (st_need_host) {
          w.h_type_g = "F32";
          w.h_type_u = "F32";
          w.h_type_d = "F32";
        }
      } else {
        w.gate = conv(L(i, "mlp.gate_proj.weight"));
        w.up = conv(L(i, "mlp.up_proj.weight"));
        w.down = conv(L(i, "mlp.down_proj.weight"));
      }
    }
    alloc_kv_cache(w);
  }
  {
    const cuda::MemInfo mi = cuda::mem_info();
    std::fprintf(stderr,
                 "[vram] post-load used=%.0f MiB free=%.0f MiB (gpu_moe_layers=%d) "
                 "alloc_sum=%.0f MiB in %lld buffers\n",
                 (mi.total_bytes - mi.free_bytes) / 1048576.0, mi.free_bytes / 1048576.0,
                 gpu_moe_layers_, cuda::g_buf_bytes / 1048576.0, cuda::g_buf_count);
  }
  start_clock_warmer();
  // Capture the decode-time layer graphs BEFORE the post-load VRAM check and the
  // auto-vram expert promotion.  The CUDA graph exec pools are ledger-invisible and cost
  // ~1.1 GiB on the 35B (measured: free 2910 -> ~1800 before the serve margin check);
  // capturing them after `apply_auto_vram_layers()` used to consume the very margin the
  // promotions had just spent, so the serve guard rejected a default 35B launcher start
  // (and a retry could then hit cuMemAlloc).  Capture only records the kernel sequence
  // (nothing executes), so the recurrent state / KV cache stay untouched.  On any
  // failure fall back to the plain per-launch path for the whole run.
  if (layer_graph_auto_) {
    try {
      capture_decode_graphs();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[graph] layer graphs disabled: %s\n", e.what());
      layer_graph_auto_ = false;
      for (auto& w : layers_) {
        w.layer_graph.reset();
        w.layer_graph_ready = false;
        w.layer_graph_state = 0;
        for (int b = 0; b < 3; ++b) {
          w.attn_graph[b].reset();
          w.attn_graph_state[b] = 0;
        }
      }
      cuda::stream_sync();
    }
  }
  check_post_load_vram();
  apply_auto_vram_layers();
}

double g_prof_mla_ms = 0.0, g_prof_moe_ms = 0.0;
double g_prof_router_ms = 0.0, g_prof_experts_ms = 0.0, g_prof_shared_ms = 0.0;
double g_prof_skip_ms = 0.0, g_prof_active_ms = 0.0;

// ---- per-token accounting (ENGINE_PROF_ACC=1) --------------------------------
// WHY: the historical 35B ledger was built from ENGINE_DBG_LAYERMS, which writes 40
// lines/token to an unbuffered stderr.  On this host that costs ~10 ms/token, so the
// sum of the per-layer lines can never be reconciled with the decoded-token wall clock
// (measured: 32.1 ms of layer lines inside a 43.1 ms token).  This accumulator keeps
// the very same segments in memory and dumps them ONCE at exit: a handful of
// steady_clock reads per layer, no I/O in the hot path.
namespace prof_acc {
int warm_left = 4;   // decode tokens skipped before counting (graph capture / page-in)
static const bool on = std::getenv("ENGINE_PROF_ACC") != nullptr;
bool warm_init = false;
inline bool counting(int n_tokens) {
  if (!warm_init) {
    warm_init = true;
    const char* e = std::getenv("ENGINE_PROF_ACC_SKIP");
    if (e) warm_left = std::atoi(e);
  }
  return on && n_tokens == 1 && warm_left <= 0;
}
inline void tick_warm(int n_tokens) {
  if (on && n_tokens == 1 && warm_left > 0) --warm_left;
}
double token_ms = 0, pre_ms = 0, pipe_ms = 0, head_gpu_ms = 0, head_copy_ms = 0;
double layers_ms = 0, cpu_ms = 0, gpu_ms = 0, gemv_ms = 0;
double moe_ms = 0, moe_pre_ms = 0, moe_route_ms = 0, moe_tail_ms = 0;
double cpu_sync_ms = 0, cpu_launch_ms = 0, gpu_sync_ms = 0, gpu_launch_ms = 0;
long long tokens = 0, layers = 0, cpu_layers = 0, gpu_layers = 0;
long long cpu_syncs = 0, cpu_launches = 0, gpu_syncs = 0, gpu_launches = 0;

inline double since(std::chrono::steady_clock::time_point a) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a)
      .count();
}
void dump() {
  if (!on || tokens <= 0) return;
  const double per = static_cast<double>(tokens);
  const double wall = token_ms / per, pre = pre_ms / per, pipe = pipe_ms / per;
  const double gpuh = head_gpu_ms / per, hcopy = head_copy_ms / per;
  const double lsum = layers_ms / per, cpu = cpu_ms / per, gpu = gpu_ms / per;
  const double gemv = gemv_ms / per;
  const double csync = cpu_sync_ms / per, claunch = cpu_launch_ms / per;
  const double gsync = gpu_sync_ms / per, glaunch = gpu_launch_ms / per;
  const double ncpu = static_cast<double>(cpu_layers) / per;
  const double ngpu = static_cast<double>(gpu_layers) / per;
  std::fprintf(stderr, "\n[acc] ===== per decode token (n=%lld, %lld layers) =====\n", tokens,
               layers / tokens);
  std::fprintf(stderr, "[acc] wall                %7.3f ms\n", wall);
  std::fprintf(stderr, "[acc]   pre (embed/upload/setpos) %7.3f ms\n", pre);
  std::fprintf(stderr, "[acc]   pipeline          %7.3f ms\n", pipe);
  std::fprintf(stderr, "[acc]     sum(run_layer) %7.3f   -> loop/other %7.3f\n", lsum,
               pipe - lsum - gpuh);
  std::fprintf(stderr, "[acc]     head in pipeline (final norm + lm_head + topk) %7.3f\n",
               gpuh);
  std::fprintf(stderr, "[acc]   head copy (topk/logits D2H + sync) %7.3f ms\n", hcopy);
  std::fprintf(stderr, "[acc]   accounted %7.3f ms   UNACCOUNTED %7.3f ms\n",
               pre + pipe + hcopy, wall - pre - pipe - hcopy);
  std::fprintf(stderr, "[acc] layers: %lld total | CPU %.2f layers x %.3f ms = %.3f | GPU %.2f x %.3f = %.3f\n",
               layers / tokens, ncpu, ncpu > 0 ? cpu / ncpu : 0.0, cpu, ngpu,
               ngpu > 0 ? gpu / ngpu : 0.0, gpu);
  std::fprintf(stderr,
               "[acc]   CPU sub: pre(router+D2H+shared enqueue) %.3f  route(host) %.3f  "
               "tail(upload+add3+host part) %.3f  gemv %.3f  sync %.3f  launch %.3f  rest %.3f ms/layer\n",
               ncpu > 0 ? moe_pre_ms / per / ncpu : 0.0,
               ncpu > 0 ? moe_route_ms / per / ncpu : 0.0,
               ncpu > 0 ? moe_tail_ms / per / ncpu : 0.0, ncpu > 0 ? gemv / ncpu : 0.0,
               ncpu > 0 ? csync / ncpu : 0.0, ncpu > 0 ? claunch / ncpu : 0.0,
               ncpu > 0 ? (cpu - gemv - csync - claunch) / ncpu : 0.0);
  std::fprintf(stderr, "[acc]   run_moe total %.3f ms/token (%.3f ms/layer over %.2f layers)\n",
               moe_ms / per, ncpu + ngpu > 0 ? moe_ms / per / (ncpu + ngpu) : 0.0, ncpu + ngpu);
  std::fprintf(stderr, "[acc]   CPU layer parts: host GEMV %.3f ms/layer (%6.2f%%)  sync %.3f ms/layer (%5.1f%%)  launch %.3f ms/layer (%4.1f%%)  other %.3f ms/layer\n",
               ncpu > 0 ? gemv / ncpu : 0.0, cpu > 0 ? 100.0 * gemv / cpu : 0.0,
               ncpu > 0 ? csync / ncpu : 0.0, cpu > 0 ? 100.0 * csync / cpu : 0.0,
               ncpu > 0 ? claunch / ncpu : 0.0, cpu > 0 ? 100.0 * claunch / cpu : 0.0,
               ncpu > 0 ? (cpu - gemv - csync - claunch) / ncpu : 0.0);
  std::fprintf(stderr, "[acc]   GPU layer parts: sync %.3f ms/layer  launch %.3f ms/layer  other %.3f ms/layer\n",
               ngpu > 0 ? gsync / ngpu : 0.0, ngpu > 0 ? glaunch / ngpu : 0.0,
               ngpu > 0 ? (gpu - gsync - glaunch) / ngpu : 0.0);
  std::fprintf(stderr, "[acc] per token: launches %lld (CPU %lld / GPU %lld)  syncs %lld (CPU %lld / GPU %lld)\n",
               (cpu_launches + gpu_launches) / tokens, cpu_launches / tokens,
               gpu_launches / tokens, (cpu_syncs + gpu_syncs) / tokens, cpu_syncs / tokens,
               gpu_syncs / tokens);
  std::fprintf(stderr,
               "[acc] CPU-layer overhead total = %7.3f ms/token (sync %.3f + launch %.3f + other %.3f)\n",
               cpu - gemv, csync, claunch, cpu - gemv - csync - claunch);
}
}  // namespace prof_acc

void DenseRunner::run_layer(int layer, int n_tokens, int start_pos) {
  auto& L = layers_[layer];
  // Per-layer graph: only fully GPU-resident layers can be captured (no host sync).
  static const bool layer_graph_env = layer_graph_;
  layer_graph_mode_ = layer_graph_env;
  bool gpu_only =
      !L.is_moe || L.q_experts.size() == static_cast<size_t>(ir_.moe.n_expert) * 3;
  if (ENGINE_ENV_ON("ENGINE_LG_DENSE_ONLY")) gpu_only = !L.is_moe;
  if (layer_graph_env && n_tokens == 1 && gpu_only && !L.is_linear) {
    if (ENGINE_ENV_ON("ENGINE_DBG_LG"))
      std::fprintf(stderr, "[lg] layer %d %s\n", layer,
                   L.layer_graph_ready ? "replay" : "capture");
    static const bool recapture = ENGINE_ENV_ON("ENGINE_LG_RECAPTURE");
    if (recapture) L.layer_graph.reset(new cuda::Graph());
    if (!L.layer_graph_ready || recapture) {
      if (!L.layer_graph) L.layer_graph = std::make_unique<cuda::Graph>();
      cuda::stream_sync();
      L.layer_graph->capture_begin();
      run_layer_body(layer, n_tokens, start_pos);
      L.layer_graph->capture_end();
      L.layer_graph_ready = true;
      L.layer_graph->launch();   // captured work is recorded, not executed
      return;
    }
    L.layer_graph->launch();
    return;
  }
  // ---- automatic per-layer CUDA graphs (decode only) -------------------------
  // On this VM a single kernel launch costs ~10-20 us of WDDM submission time, and a
  // decoder layer issues 15-30 launches, so submission -- not GPU work -- dominates.
  // Capturing a layer's GPU-side kernel sequence removes that per-launch cost.
  // Two variants: CPU-placed MoE layers capture only the mixer/attention part (the
  // expert FFN stays on the host), GPU-resident qwen35 MoE layers capture the whole
  // layer (device routing keeps it host-sync free).
  if (layer_graph_auto_ && n_tokens == 1 && !graph_capture_ &&
      !ENGINE_ENV_ON("ENGINE_DUMP") && !ENGINE_ENV_ON("ENGINE_DUMP_EVERY") &&
      !ENGINE_ENV_ON("ENGINE_DUMP_LAYER") && q35_layer_graph_ok(layer)) {
    // Diagnostic: force the attention-graph path for models without MoE layers so the
    // qwen35 capture path can be exercised on the fast-loading 9B model.
    const bool force_all = ENGINE_ENV_ON("ENGINE_GRAPH_FORCE");
    if (force_all) {
      run_q35_attn_graph(layer, start_pos);
      return;
    }
    const int ne = ir_.moe.n_expert;
    // CPU-placed MoE layers get their mixer/attention part captured too: it removes
    // ~7 WDDM submissions per layer, which is ~1.6 ms/token on the 35B (24 CPU layers
    // at ~10 us per launch).  This used to measure net-neutral because the one-time
    // capture landed in the timed decode window; the capture now happens right after
    // the first execution (inside the prefill), so only the launch saving remains.
    const bool cpu_layer = ne > 0 && L.cpu_begin == 0 && q35_cpu_layer_graph_;
    const bool gpu_layer = ne > 0 && L.cpu_begin == ne &&
                           !ENGINE_ENV_ON("ENGINE_NO_GPU_LAYER_GRAPH") &&
                           (dev_route_default_ || ENGINE_ENV_ON("ENGINE_DEV_ROUTE"));
    const bool gpu_layer_attn =
        ne > 0 && L.cpu_begin == ne && !ENGINE_ENV_ON("ENGINE_NO_GPU_LAYER_GRAPH");
    if (cpu_layer || gpu_layer_attn) {
      if (gpu_layer && ENGINE_ENV_ON("ENGINE_GPU_FULL_LAYER_GRAPH")) {
        run_gpu_layer_graph(layer, start_pos);   // whole layer (needs device routing)
      } else {
        run_q35_attn_graph(layer, start_pos);    // mixer/attention part only
      }
      return;
    }
  }
  run_layer_body(layer, n_tokens, start_pos);
}

void DenseRunner::run_layer_body(int layer, int n_tokens, int start_pos) {
  auto& L = layers_[layer];
  const bool kDebug = debug_;
  if (L.ext) {
    run_layer_ext(layer, n_tokens, start_pos);
    return;
  }
  if (L.is_linear) {
    run_q35_linear(layer, n_tokens, start_pos);
    return;
  }
  if (L.q35.attn_kind) {
    run_q35_attn(layer, n_tokens, start_pos);
    return;
  }
  if (kDebug)
    std::fprintf(stderr, "[dbg] run_layer %d n=%d pos=%d\n", layer, n_tokens, start_pos);
  const int hd = n_head_ * head_dim_, kd = n_kv_ * head_dim_;

  auto rms = [&](cuda::Buffer& in, cuda::Buffer& w, cuda::Buffer& out) {
    rms_raw(in.data(), w.data(), out.data(), n_tokens, hidden_);
  };
  auto rope = [&](cuda::Buffer& buf, int nheads) {
    int nh = nheads, d = head_dim_;
    float base = rope_base_;
    void* pb = buf.data();
    void* pp = pos_dev_.data();
    void* pf = rope_freqs_dev_.data();
    float ms = rope_mscale_;
    void* args[] = {&pb, &pp, &nh, &d, &pf, &ms};
    cuda::launch(k_rope_, n_tokens, nheads, 1, static_cast<int>(head_dim_ / 2), 1, 1, 0, args);
  };
  auto add = [&](cuda::Buffer& a, cuda::Buffer& b, long long total) {
    if (!b.data()) return;
    void* pa = a.data();
    void* pb = b.data();
    void* args[] = {&pa, &pb, &total};
    cuda::launch(k_add_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  };
  auto bias = [&](cuda::Buffer& buf, cuda::Buffer& b, int n_cols) {
    if (!b.data()) return;
    long long total = static_cast<long long>(n_tokens) * n_cols;
    void* px = buf.data();
    void* pb = b.data();
    void* args[] = {&px, &pb, &n_cols, &total};
    cuda::launch(k_bias_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  };

  static const bool kProf = ENGINE_ENV_ON("ENGINE_PROFILE");
  static double t_qkv = 0, t_attn = 0, t_ffn = 0;
  static int n_prof = 0;
  std::chrono::steady_clock::time_point tp0;
  auto tick = [&]() {
    if (kProf) { cuda::sync(); tp0 = std::chrono::steady_clock::now(); }
  };
  auto tock = [&](double& acc) {
    if (kProf) {
      cuda::sync();
      acc += std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - tp0).count();
    }
  };
  tick();
  if (L.is_mla) {
    // MLA attention segment = ln1 norm + attention + ln2 norm.  Decode records it as a
    // single per-layer CUDA graph (see run_mla_attn_graph): the whole segment is
    // GEMV/small-kernel only (matvec_f32 falls back to the gemv_f32 kernel for
    // n_tokens <= 4), so ~13 WDDM submissions become one cuGraphLaunch.  Coder is a MoE
    // model, so the whole-pipeline graph is disabled and this is its only graph
    // coverage.  Prefill and the diagnostic switches keep the plain sequence.
    if (mla_attn_graph_ && !no_graph_cfg_ && n_tokens == 1 && !graph_capture_ &&
        mla_attn_graph_ok(layer)) {
      run_mla_attn_graph(layer);
    } else {
      mla_attn_segment(layer, n_tokens, start_pos);
    }
    if (L.is_moe) {
      const auto ep0 = std::chrono::steady_clock::now();
      run_moe(layer, n_tokens);
      g_prof_moe_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ep0).count();
      return;
    }
    if (L.is_quant && !(L.has_f32 && n_tokens > 1)) {
      matmul(L.qg, n_tokens, hn_, gate_);
      matmul(L.qu, n_tokens, hn_, up_);
    } else {
      cublas::gemm_f32_nt(n_tokens, inter_, hidden_, static_cast<const float*>(hn_.data()),
                          static_cast<const float*>(L.gate.data()),
                          static_cast<float*>(gate_.data()));
      cublas::gemm_f32_nt(n_tokens, inter_, hidden_, static_cast<const float*>(hn_.data()),
                          static_cast<const float*>(L.up.data()),
                          static_cast<float*>(up_.data()));
    }
    {
      long long total = static_cast<long long>(n_tokens) * inter_;
      void* pg = gate_.data();
      void* pu = up_.data();
      void* pa = act_.data();
      void* args[] = {&pg, &pu, &pa, &total};
      cuda::launch(k_silu_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
    }
    if (L.is_quant && !(L.has_f32 && n_tokens > 1)) {
      matmul(L.qd, n_tokens, act_, down_);
    } else {
      cublas::gemm_f32_nt(n_tokens, hidden_, inter_, static_cast<const float*>(act_.data()),
                          static_cast<const float*>(L.down.data()),
                          static_cast<float*>(down_.data()));
    }
    add(x_, down_, static_cast<long long>(n_tokens) * hidden_);
    return;
  }
  rms(x_, L.ln1, hn_);
  if (layer == 0) {
    if (const char* dp = ENGINE_ENV_STR("ENGINE_DUMP")) {
      cuda::stream_sync();
      std::vector<float> h(static_cast<size_t>(n_tokens) * hidden_);
      hn_.download(h.data(), h.size() * 4);
      std::ofstream f(std::string(dp) + "_hn0.f32", std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()),
              static_cast<std::streamsize>(h.size() * 4));
    }
  }
  const bool batch_f32 = L.has_f32 && n_tokens > 1;
  if (L.is_quant && !batch_f32) {
    matmul(L.qq, n_tokens, hn_, q_);
    matmul(L.qk, n_tokens, hn_, k_);
    matmul(L.qv, n_tokens, hn_, v_);
  } else {
    cublas::gemm_f32_nt(n_tokens, hd, hidden_, static_cast<const float*>(hn_.data()),
                        static_cast<const float*>(L.q.data()), static_cast<float*>(q_.data()));
    cublas::gemm_f32_nt(n_tokens, kd, hidden_, static_cast<const float*>(hn_.data()),
                        static_cast<const float*>(L.k.data()), static_cast<float*>(k_.data()));
    cublas::gemm_f32_nt(n_tokens, kd, hidden_, static_cast<const float*>(hn_.data()),
                        static_cast<const float*>(L.v.data()), static_cast<float*>(v_.data()));
  }
  if (L.has_bias) {
    bias(q_, L.bq, hd);
    bias(k_, L.bk, kd);
    bias(v_, L.bv, kd);
  }
  rope(q_, n_head_);
  rope(k_, n_kv_);
  kv_write_layer(L, n_tokens, k_, v_);
  run_attn(L, n_tokens, start_pos, q_, attn_);
  tock(t_qkv);
  tick();
  if (L.is_quant && !batch_f32) matmul(L.qo, n_tokens, attn_, proj_);
  else cublas::gemm_f32_nt(n_tokens, hidden_, hd, static_cast<const float*>(attn_.data()),
                           static_cast<const float*>(L.o.data()), static_cast<float*>(proj_.data()));
  add(x_, proj_, static_cast<long long>(n_tokens) * hidden_);

  tock(t_attn);
  tick();
  rms(x_, L.ln2, hn_);
  if (L.is_moe) {
    run_moe(layer, n_tokens);
    return;
  }
  if (L.is_quant && !batch_f32) {
    matmul(L.qg, n_tokens, hn_, gate_);
    matmul(L.qu, n_tokens, hn_, up_);
  } else {
    if (L.is_quant && !(L.has_f32 && n_tokens > 1)) {
      matmul(L.qg, n_tokens, hn_, gate_);
      matmul(L.qu, n_tokens, hn_, up_);
    } else {
      cublas::gemm_f32_nt(n_tokens, inter_, hidden_, static_cast<const float*>(hn_.data()),
                          static_cast<const float*>(L.gate.data()),
                          static_cast<float*>(gate_.data()));
      cublas::gemm_f32_nt(n_tokens, inter_, hidden_, static_cast<const float*>(hn_.data()),
                          static_cast<const float*>(L.up.data()),
                          static_cast<float*>(up_.data()));
    }
  }
  {
    long long total = static_cast<long long>(n_tokens) * inter_;
    void* pg = gate_.data();
    void* pu = up_.data();
    void* pa = act_.data();
    void* args[] = {&pg, &pu, &pa, &total};
    cuda::launch(k_silu_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
  if (L.is_quant && !batch_f32) matmul(L.qd, n_tokens, act_, down_);
  else cublas::gemm_f32_nt(n_tokens, hidden_, inter_, static_cast<const float*>(act_.data()),
                           static_cast<const float*>(L.down.data()),
                           static_cast<float*>(down_.data()));
  add(x_, down_, static_cast<long long>(n_tokens) * hidden_);
  tock(t_ffn);
  if (layer == 0) {
    if (const char* dp = ENGINE_ENV_STR("ENGINE_DUMP")) {
      cuda::stream_sync();
      std::vector<float> h(static_cast<size_t>(n_tokens) * hidden_);
      x_.download(h.data(), h.size() * 4);
      std::ofstream f(std::string(dp) + "_x0.f32", std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()), static_cast<std::streamsize>(h.size() * 4));
    }
  }
  if (kProf) {
    static double last_report = 0;
    ++n_prof;
    if (n_prof % ir_.n_layer == 0) {
      const double total = t_qkv + t_attn + t_ffn;
      if (total > 0) {
        std::fprintf(stderr,
                     "[prof] layers=%d  qkv=%.2f ms  attn=%.2f ms  ffn=%.2f ms  total=%.2f ms\n",
                     n_prof, t_qkv, t_attn, t_ffn, total);
      }
      t_qkv = t_attn = t_ffn = 0;
    }
  }
}

void DenseRunner::set_rope_mode(int mode) {
  if (rope_freqs_.empty()) return;
  const int half = static_cast<int>(rope_freqs_.size());
  const int use_d = rope_use_d_ > 0 ? rope_use_d_ : half * 2;
  const double base = rope_base_;
  for (int i = 0; i < half; ++i) {
    const double theta = std::pow(base, -2.0 * i / use_d);
    if (mode == 0 && rope_is_yarn_) {
      const double i0 = 2.0 * i;
      const double ramp =
          std::min(1.0, std::max(0.0, (i0 - rope_yarn_c0_) / std::max(1e-9, rope_yarn_c1_ - rope_yarn_c0_)));
      rope_freqs_[i] = static_cast<float>(theta * (1.0 - ramp) / rope_yarn_factor_ +
                                          theta * ramp);
    } else {
      rope_freqs_[i] = static_cast<float>(theta);
    }
  }
  if (mode == 0 && rope_is_yarn_) {
    rope_mscale_ = static_cast<float>(0.1 * rope_yarn_logmul_ * std::log(rope_yarn_factor_) + 1.0);
  } else if (mode == 2) {
    rope_mscale_ = 1.0f;
  } else {
    rope_mscale_ = 1.0f;
  }
  rope_freqs_dev_.upload(rope_freqs_.data(), rope_freqs_.size() * 4);
}

void DenseRunner::set_position(int start_pos) {
  if (!pos_dev_.data()) return;
  // Async: a synchronous 4-byte H2D still flushes the WDDM queue.  The copy is
  // stream-ordered before every kernel that reads it and the position scalar outlives
  // the forward call, so it is safe to enqueue.
  pos_host_ = start_pos;
  pos_dev_.upload_async(&pos_host_, 4);
}

// ---- prefix-cache recurrent-state snapshot (single slot; hybrid models) -----
// The slot buffers + per-layer save/restore helpers live in src/mtp.cpp (they were
// introduced for speculative-decoding rollback); the server reuses them for
// multi-turn prefix caching.  Only the gated-delta-net layers have recurrent state,
// the attention KV cache is position-addressed and needs no copy (prefix KV is
// reused in place, stale entries beyond the cached length are overwritten before
// they are ever read).
size_t DenseRunner::prefix_state_bytes() const {
  if (ir_.ssm.variant == "none") return 0;
  int n_linear = 0;
  for (const auto& L : layers_)
    if (L.is_linear) ++n_linear;
  if (n_linear == 0) return 0;
  const size_t rec = static_cast<size_t>(ir_.ssm.n_value_head) * ir_.ssm.value_head_dim *
                     ir_.ssm.key_head_dim * 4;
  const size_t conv =
      static_cast<size_t>(ir_.ssm.conv_kernel - 1) * ir_.ssm.conv_dim * 4;
  return static_cast<size_t>(n_linear) * (rec + conv);
}

bool DenseRunner::prefix_state_save(int slot) {
  if (slot < 0) return false;
  if (ir_.ssm.variant == "none") return true;  // pure attention: KV is enough
  try {
    ensure_ssm_slots(slot + 1);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[prefix] ssm slot allocation failed: %s\n", e.what());
    return false;
  }
  if (slot >= ssm_slots_) return false;
  for (int i = 0; i < ir_.n_layer; ++i) {
    if (!layers_[i].is_linear) continue;
    ssm_save_conv(i, slot);
    ssm_save_rec(i, slot);
  }
  cuda::stream_sync();
  return true;
}

bool DenseRunner::prefix_state_restore(int slot) {
  if (slot < 0) return false;
  if (ir_.ssm.variant == "none") return true;
  if (slot >= ssm_slots_) return false;
  ssm_restore(slot);
  cuda::stream_sync();
  return true;
}

// layers + final norm + lm_head, no host synchronization (graph-capturable)
void DenseRunner::run_pipeline(int n, int start_pos, bool last_row_only) {
  n_tokens_cur_ = n;
  int n_layer = ir_.n_layer;
  if (layers_override_ > 0) n_layer = layers_override_;
  if (n_layer > ir_.n_layer) n_layer = ir_.n_layer;
  static const bool pipe_prof = ENGINE_ENV_ON("ENGINE_PIPE_PROF");
  cuda::Event pev0, pev1;
  double cpu_launch_ms = 0.0;
  const auto pipe_t0 = std::chrono::steady_clock::now();
  const bool acc_on = prof_acc::counting(n);
  double acc_layers = 0.0;
  for (int l = 0; l < n_layer; ++l) {
    if (pipe_prof) pev0.record();
    const auto lt0 = std::chrono::steady_clock::now();
    const long long layer_sc0 = cuda::g_sync_calls;
    const double layer_sm0 = cuda::g_sync_ms;
    const long long layer_lc0 = cuda::g_launch_calls;
    const double layer_lm0 = cuda::g_launch_ms;
    const double layer_act0 = g_prof_active_ms;
    run_layer(l, n, start_pos);
    {
      static const bool kLayerMs = ENGINE_ENV_ON("ENGINE_DBG_LAYERMS");
      if (kLayerMs && n <= 2) {
        const bool layer_cpu = layers_[l].is_moe && layers_[l].cpu_begin == 0 &&
                               ir_.moe.n_expert > 0;
        std::fprintf(stderr,
                     "[layerms] layer %2d %6.2f ms  %s  sync %lld/%.2f  launch %lld/%.2f  "
                     "host %5.2f ms\n",
                     l,
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - lt0).count(),
                     layer_cpu ? "CPU" : "GPU", cuda::g_sync_calls - layer_sc0,
                     cuda::g_sync_ms - layer_sm0, cuda::g_launch_calls - layer_lc0,
                     cuda::g_launch_ms - layer_lm0, g_prof_active_ms - layer_act0);
      }
    }
    const double layer_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lt0)
            .count();
    cpu_launch_ms += layer_ms;
    if (acc_on) {
      const auto& LW = layers_[l];
      const bool is_cpu = ir_.moe.n_expert > 0 && LW.is_moe && LW.cpu_begin == 0;
      acc_layers += layer_ms;
      if (is_cpu) {
        prof_acc::cpu_ms += layer_ms;
        prof_acc::cpu_sync_ms += cuda::g_sync_ms - layer_sm0;
        prof_acc::cpu_launch_ms += cuda::g_launch_ms - layer_lm0;
        prof_acc::cpu_syncs += cuda::g_sync_calls - layer_sc0;
        prof_acc::cpu_launches += cuda::g_launch_calls - layer_lc0;
        ++prof_acc::cpu_layers;
      } else {
        prof_acc::gpu_ms += layer_ms;
        prof_acc::gpu_sync_ms += cuda::g_sync_ms - layer_sm0;
        prof_acc::gpu_launch_ms += cuda::g_launch_ms - layer_lm0;
        prof_acc::gpu_syncs += cuda::g_sync_calls - layer_sc0;
        prof_acc::gpu_launches += cuda::g_launch_calls - layer_lc0;
        ++prof_acc::gpu_layers;
      }
      ++prof_acc::layers;
    }
    if (pipe_prof) pev1.record();
    if (const char* dp = ENGINE_ENV_STR("ENGINE_DUMP_EVERY")) {
      cuda::stream_sync();
      std::vector<float> h(static_cast<size_t>(n) * hidden_);
      x_.download(h.data(), h.size() * 4);
      char name[512];
      std::snprintf(name, sizeof(name), "%s_L%d.f32", dp, l);
      std::ofstream f(name, std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()),
              static_cast<std::streamsize>(h.size() * 4));
    }
    if (l == 0) {
      if (const char* dp = ENGINE_ENV_STR("ENGINE_DUMP")) {
        cuda::stream_sync();
        std::vector<float> h(static_cast<size_t>(n) * hidden_);
        x_.download(h.data(), h.size() * 4);
        std::ofstream f(std::string(dp) + "_x0.f32", std::ios::binary);
        f.write(reinterpret_cast<const char*>(h.data()),
                static_cast<std::streamsize>(h.size() * 4));
      }
    }
  }
  const auto acc_head0 = std::chrono::steady_clock::now();
  if (pipe_prof) {
    cuda::sync();
    const double wall =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pipe_t0)
            .count();
    std::fprintf(stderr,
                 "[pipe] n=%d layers=%d wall %.2f ms, cpu-launch %.2f ms, syncs %lld = %.2f ms\n", n,
                 n_layer, wall, cpu_launch_ms, cuda::g_sync_calls, cuda::g_sync_ms);
    std::fprintf(stderr,
                 "[pipe] launches %lld = %.2f ms, mla %.2f ms, moe %.2f ms "
                 "(router %.2f, experts %.2f, shared %.2f, skip %.2f, active %.2f)\n",
                 cuda::g_launch_calls, cuda::g_launch_ms, g_prof_mla_ms, g_prof_moe_ms,
                 g_prof_router_ms, g_prof_experts_ms, g_prof_shared_ms, g_prof_skip_ms,
                 g_prof_active_ms);
  }
  const int rows = last_row_only ? 1 : n;
  ensure_logits_rows(rows);
  {
    void* px = static_cast<float*>(x_.data()) +
               static_cast<size_t>(last_row_only ? (n - 1) * hidden_ : 0);
    rms_raw(px, w_final_norm_.data(), hn_.data(), rows, hidden_);
  }
  if (const char* dp = ENGINE_ENV_STR("ENGINE_DUMP")) {
    if (rows == 1) {
      cuda::stream_sync();
      std::vector<float> h(static_cast<size_t>(hidden_));
      hn_.download(h.data(), h.size() * 4);
      std::ofstream f(std::string(dp) + "_final_norm.f32", std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()),
              static_cast<std::streamsize>(h.size() * 4));
    }
  }
  static const bool skip_lmhead = ENGINE_ENV_ON("ENGINE_SKIP_LMHEAD");
  if (!skip_lmhead && rows == 1 && lm_head_qt_.valid) {
    matmul(lm_head_qt_, 1, hn_, logits_);
  } else if (rows == 1) {
    if (w_lm_.data() == nullptr) w_lm_ = upload(lm_head_f32());
    cublas::gemm_f32_nt(1, vocab_, hidden_, static_cast<const float*>(hn_.data()),
                        static_cast<const float*>(w_lm_.data()),
                        static_cast<float*>(logits_.data()));
  } else if (verify_multi_head_ && lm_head_qt_.valid) {
    // Speculative verification needs one arg-max per row.  Keep the quantized head and
    // run the multi-row GEMM instead of materialising the f32 head (2 GB on the 35B).
    matmul(lm_head_qt_, n, hn_, logits_);
  } else {
    if (w_lm_.data() == nullptr) w_lm_ = upload(lm_head_f32());
    cublas::gemm_f32_nt(n, vocab_, hidden_, static_cast<const float*>(hn_.data()),
                        static_cast<const float*>(w_lm_.data()),
                        static_cast<float*>(logits_.data()));
  }
  // Generic final-logit softcapping (metadata final_logit_softcapping): applied
  // before the device top-k head so every consumer sees the same logits.
  if (ext_logit_softcap_ > 0.0 && logits_.data()) {
    long long total = static_cast<long long>(rows) * vocab_;
    void* px = logits_.data();
    float sc = static_cast<float>(ext_logit_softcap_);
    void* args[] = {&px, &sc, &total};
    cuda::launch(k_softcap_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
  // Decode head: reduce the logits row to its top-k on the device (two ~5 us kernels)
  // so forward()/forward_topk() only copy back k pairs instead of the whole row.
  // Recorded into the decode CUDA graph like the rest of the head.
  if (head_topk_ && rows == 1 && k_topkpart_ && k_topkfinal_ && topk_nblk_ > 0) {
    int kk = head_topk_max_;   // recorded into the decode graph; forward_topk reads <= kk
    void* px = logits_.data();
    void* ppv = topk_pv_.data();
    void* ppi = topk_pi_.data();
    int nv = vocab_, nb = topk_nblk_;
    void* a1[] = {&px, &ppv, &ppi, &nv, &nb, &kk};
    cuda::launch(k_topkpart_, nb, 1, 1, 256, 1, 1, 0, a1);
    void* pov = topk_ov_.data();
    void* poi = topk_oi_.data();
    int nc = nb * kk;
    void* a2[] = {&ppv, &ppi, &nc, &kk, &poi, &pov};
    cuda::launch(k_topkfinal_, 1, 1, 1, 256, 1, 1, 0, a2);
  }
  if (acc_on) {
    prof_acc::layers_ms += acc_layers;
    prof_acc::head_gpu_ms += prof_acc::since(acc_head0);
  }
}

std::vector<float> DenseRunner::forward(const std::vector<int>& tokens, int start_pos,
                                       bool last_row_only) {
  return forward_impl(tokens, start_pos, last_row_only, 0, nullptr);
}

bool DenseRunner::forward_topk_candidates(const std::vector<int>& tokens, int start_pos, int k,
                                          std::vector<int>* ids, std::vector<float>* logits) {
  if (k <= 0 || k > head_topk_max_) return false;
  TopKResult tk;
  const auto rest = forward_topk(tokens, start_pos, k, &tk);
  if (!rest.empty()) return false;   // device top-k head unavailable (e.g. head_topk off)
  if (ids) *ids = tk.idx;
  if (logits) *logits = tk.val;
  return true;
}

std::vector<float> DenseRunner::forward_topk(const std::vector<int>& tokens, int start_pos,
                                            int k, TopKResult* tk) {
  return forward_impl(tokens, start_pos, true, k, tk);
}

std::vector<float> DenseRunner::forward_impl(const std::vector<int>& tokens, int start_pos,
                                            bool last_row_only, int topk_k, TopKResult* tk) {
  const int n = static_cast<int>(tokens.size());
  if (n <= 0 || n > max_seq_) throw std::runtime_error("forward: bad token count");
  if (start_pos < 0 || start_pos + n > max_seq_)
    throw std::runtime_error("forward: position out of range");
  // Long prompts are prefilled in chunks of rows_ tokens: the activation buffers (and
  // above all logits_ = rows*vocab*4) are sized for the chunk, not for max_seq.
  if (n > static_cast<int>(rows_)) {
    if (!last_row_only)
      throw std::runtime_error(
          "forward: " + std::to_string(n) + " tokens exceed the activation row cap (" +
          std::to_string(rows_) + ", prefill_rows) and full logits were requested");
    std::vector<float> out;
    const int chunk = static_cast<int>(rows_);
    for (int off = 0; off < n; off += chunk) {
      const int len = std::min(chunk, n - off);
      std::vector<int> sub(tokens.begin() + off, tokens.begin() + off + len);
      out = forward_impl(sub, start_pos + off, true, topk_k, tk);
    }
    return out;
  }

  const auto fwd_t_entry = std::chrono::steady_clock::now();
  const auto acc_t_entry = std::chrono::steady_clock::now();
  prof_acc::tick_warm(n);
  std::vector<float> x(static_cast<size_t>(n) * hidden_);
  const bool lazy_embed = !embed_raw_.empty();
  const int row_blocks = lazy_embed ? hidden_ / embed_be_ : 0;
  for (int i = 0; i < n; ++i) {
    const int tok = tokens[i];
    if (tok < 0 || tok >= vocab_) throw std::runtime_error("token id out of range");
    if (lazy_embed) {
      const unsigned char* row = embed_raw_.data() +
                                 static_cast<size_t>(tok) * row_blocks * embed_bb_;
      float* dst = x.data() + static_cast<size_t>(i) * hidden_;
      for (int b = 0; b < row_blocks; ++b)
        cpu::dequant_block(embed_type_, row + static_cast<size_t>(b) * embed_bb_,
                           dst + static_cast<size_t>(b) * embed_be_);
      if (ext_enabled_ && ext_embed_scale_ != 1.0)
        for (int d = 0; d < hidden_; ++d) dst[d] *= static_cast<float>(ext_embed_scale_);
    } else {
      std::memcpy(x.data() + static_cast<size_t>(i) * hidden_,
                  embed_host_.data() + static_cast<size_t>(tok) * hidden_,
                  static_cast<size_t>(hidden_) * 4);
      if (ext_enabled_ && ext_embed_scale_ != 1.0) {
        float* dst = x.data() + static_cast<size_t>(i) * hidden_;
        for (int d = 0; d < hidden_; ++d) dst[d] *= static_cast<float>(ext_embed_scale_);
      }
    }
  }
  // Async so the token loop does not pay a WDDM queue flush before the first kernel
  // (x outlives every stream operation: forward() synchronises on the logits copy).
  x_.upload_async(x.data(), x.size() * 4);
  if (const char* dp = ENGINE_ENV_STR("ENGINE_DUMP")) {
    std::ofstream f(std::string(dp) + "_emb.f32", std::ios::binary);
    f.write(reinterpret_cast<const char*>(x.data()),
            static_cast<std::streamsize>(x.size() * 4));
  }
  set_position(start_pos);
  // a fresh sequence must start from an empty recurrent state
  if (start_pos == 0 && ir_.ssm.variant != "none") reset_kv_and_states();
  if (verify_multi_head_) {
    // Speculative verification: one batch forward over the candidate tokens, then the
    // arg-max of every row.  The per-row arg-maxes are all the accept rule needs and
    // keeping them on the host avoids a full logits download per row.
    run_pipeline(n, start_pos, /*last_row_only=*/false);
    cuda::stream_sync();
    const size_t need = static_cast<size_t>(n) * static_cast<size_t>(vocab_) * 4;
    if (mtp_pin_.size() < need) mtp_pin_.resize(need);
    logits_.download_async(mtp_pin_.data(), need);
    cuda::stream_sync();
    verify_argmax_.assign(static_cast<size_t>(n), -1);
    const float* base = static_cast<const float*>(mtp_pin_.data());
    for (int r = 0; r < n; ++r) {
      const float* row = base + static_cast<size_t>(r) * static_cast<size_t>(vocab_);
      int best = 0;
      for (int i = 1; i < vocab_; ++i)
        if (row[i] > row[best]) best = i;
      verify_argmax_[static_cast<size_t>(r)] = best;
    }
    return {};
  }
  const auto fwd_t_pre = std::chrono::steady_clock::now();
  const bool acc_tok = prof_acc::counting(n);
  if (acc_tok) prof_acc::pre_ms += prof_acc::since(acc_t_entry);

  static const bool kProfFwd = ENGINE_ENV_ON("ENGINE_PROFILE");
  static const bool kProfK = ENGINE_ENV_ON("ENGINE_KPROF");
  static int kProfToken = 0;
  const auto kProfT0 = std::chrono::steady_clock::now();
  if (kProfK && n == 1 && !use_graph_) cuda::kprof_reset();
  // Prefill chunks: replay a captured graph keyed by the chunk size.  The pipeline is
  // position-agnostic (the position lives in pos_dev_, uploaded before the launch), so
  // one graph per token count is valid for every chunk boundary.
  static const bool pf_graph = prefill_graph_;
  if (n > 1 && use_graph_ && pf_graph) {
    auto& slot = prefill_graphs_[n];
    if (!slot) {
      // The fp16 staging buffer must be big enough before capture starts: allocating
      // inside a capture is illegal.
      const size_t cap = static_cast<size_t>(rows_) * static_cast<size_t>(x_half_max_nin_) * 2;
      if (cap > x_half_.size()) x_half_ = cuda::Buffer(cap);
      try {
        cuda::stream_sync();
        slot = std::make_unique<cuda::Graph>();
        slot->capture_begin();
        run_pipeline(n, start_pos, last_row_only);
        slot->capture_end();
        if (ENGINE_ENV_ON("ENGINE_DBG_PFGRAPH"))
          std::fprintf(stderr, "[pfgraph] captured n=%d\n", n);
      } catch (const std::exception& e) {
        slot.reset();
        prefill_graph_ = false;
        cuda::stream_sync();
        std::fprintf(stderr, "[pfgraph] disabled: %s\n", e.what());
        run_pipeline(n, start_pos, last_row_only);
      }
    }
    // capture and replay both run the pipeline through the graph: a capture does not
    // execute anything, so the graph is launched immediately after it is recorded.
    slot->launch();
    cuda::stream_sync();
    if (ENGINE_ENV_ON("ENGINE_DBG_PFGRAPH"))
      std::fprintf(stderr, "[pfgraph] ran n=%d (ready=%d)\n", n, (int)slot->ready());
    if (kProfFwd) {
      cuda::stream_sync();
      prof_ev0_->record();
      prof_ev1_->record();
    }
  } else if (n == 1 && use_graph_) {
    if (!graph_ready_) {
      if (kProfK) cuda::kprof_reset();   // events recorded here become graph nodes
      cuda::stream_sync();
      graph_->capture_begin();
      run_pipeline(1, start_pos, last_row_only);
      graph_->capture_end();
      graph_ready_ = true;
    }
    graph_->launch();
    cuda::stream_sync();
    if (kProfK)
      cuda::kprof_add_wall(std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - kProfT0).count());
  } else {
    if (kProfFwd) {
      cuda::stream_sync();
      prof_ev0_->record();
    }
    run_pipeline(n, start_pos, last_row_only);
    if (kProfK && n == 1 && ++kProfToken >= 4)
      cuda::kprof_add_wall(std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - kProfT0).count());
    if (kProfFwd) {
      prof_ev1_->record();
      cuda::stream_sync();
      std::fprintf(stderr, "[gpu] forward gpu=%.2f ms (n=%d)\n", prof_ev0_->elapsed_ms(*prof_ev1_),
                   n);
    } else if (kProfK) {
      cuda::stream_sync();
    }
    // No unconditional stream sync here: the logits download below is a blocking copy
    // that already waits for the lm_head GEMV, so a separate cuStreamSynchronize only
    // adds a WDDM round trip per token.
  }

  const auto acc_pipe_end = std::chrono::steady_clock::now();
  if (acc_tok) prof_acc::pipe_ms += prof_acc::since(fwd_t_pre);
  // Device top-k head: only `k` (index,value) pairs cross PCIe instead of the logits row.
  const bool want_topk = topk_k > 0 && last_row_only && n == 1 && head_topk_ &&
                         k_topkpart_ && k_topkfinal_ && tk != nullptr;
  if (want_topk) {
    const int kk = std::min(topk_k, head_topk_max_);
    float* pin = static_cast<float*>(topk_pin_.data());
    topk_ov_.download_async(pin, static_cast<size_t>(kk) * 4);
    topk_oi_.download_async(reinterpret_cast<int*>(pin + kMaxTopK),
                           static_cast<size_t>(kk) * 4);
    cuda::stream_sync();
    tk->idx.assign(static_cast<size_t>(kk), -1);
    tk->val.assign(static_cast<size_t>(kk), 0.f);
    const int* pi = reinterpret_cast<const int*>(pin + kMaxTopK);
    for (int q = 0; q < kk; ++q) {
      tk->idx[static_cast<size_t>(q)] = pi[q];
      tk->val[static_cast<size_t>(q)] = pin[q];
    }
    if (ENGINE_ENV_ON("ENGINE_DBG_FWD")) {
      std::fprintf(stderr, "[fwd] topk-dl %.2f ms (top1=%d)\n",
                   std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - fwd_t_pre)
                       .count(),
                   tk->idx[0]);
    }
    if (acc_tok) {
      prof_acc::head_copy_ms += prof_acc::since(acc_pipe_end);
      prof_acc::token_ms += prof_acc::since(acc_t_entry);
      ++prof_acc::tokens;
    }
    return {};
  }
  const int out_rows = last_row_only ? 1 : n;
  std::vector<float> out(static_cast<size_t>(out_rows) * vocab_);
  const auto fwd_t_pipe = std::chrono::steady_clock::now();
  const auto fwdT0 = std::chrono::steady_clock::now();
  // Decode: stage the logits row through pinned memory.  A pageable cuMemcpyDtoH
  // of 1 MB costs ~0.3 ms on this WDDM VM; the async copy into a pinned buffer
  // plus a host memcpy is ~0.1 ms (measured with ENGINE_DBG_FWD).
  const size_t logits_bytes = out.size() * 4;
  if (last_row_only && logits_pin_.data() && logits_pin_.size() >= logits_bytes) {
    logits_.download_async(logits_pin_.data(), logits_bytes);
    cuda::stream_sync();
    std::memcpy(out.data(), logits_pin_.data(), logits_bytes);
  } else {
    logits_.download(out.data(), logits_bytes);
  }
  if (ENGINE_ENV_ON("ENGINE_DBG_FWD")) {
    auto ms = [](std::chrono::steady_clock::time_point a) {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a)
          .count();
    };
    std::fprintf(stderr, "[fwd] pre %.2f  pipe %.2f  logits-dl %.2f  total %.2f ms\n",
                 ms(fwd_t_pre), ms(fwd_t_pipe), ms(fwdT0), ms(fwd_t_entry));
  }
  if (const char* lp = ENGINE_ENV_STR("ENGINE_LOGITS")) {
    std::ofstream f(lp, std::ios::binary);
    f.write(reinterpret_cast<const char*>(out.data()),
            static_cast<std::streamsize>(out.size() * 4));
  }
  if (acc_tok) {
    prof_acc::head_copy_ms += prof_acc::since(acc_pipe_end);
    prof_acc::token_ms += prof_acc::since(acc_t_entry);
    ++prof_acc::tokens;
  }
  return out;
}


void DenseRunner::ensure_expert_scratch() {
  if (xexp_ready_) return;
  const int inter = expert_inter_ > 0 ? expert_inter_ : inter_;
  // one weight slab (max of gate/up/down) plus the activation buffers
  const size_t w_bytes = static_cast<size_t>(std::max(inter, hidden_)) *
                         static_cast<size_t>(std::max(inter, hidden_)) / 8 + 4096;
  xexp_w_ = cuda::Buffer(w_bytes);
  xexp_g_ = cuda::Buffer(static_cast<size_t>(inter) * 4);
  xexp_u_ = cuda::Buffer(static_cast<size_t>(inter) * 4);
  xexp_a_ = cuda::Buffer(static_cast<size_t>(inter) * 4);
  xexp_d_ = cuda::Buffer(static_cast<size_t>(hidden_) * 4);
  xexp_ready_ = true;
}

bool DenseRunner::gemv_typed(const std::string& type, const void* W, int n_out, int n_in,
                             cuda::Buffer& in, cuda::Buffer& out) {
  void* fn = nullptr;
  if (type == "Q4_K") fn = gemv_q4k_;
  else if (type == "Q5_K") fn = gemv_q5k_;
  else if (type == "Q6_K") fn = gemv_q6k_;
  else if (type == "Q8_0") fn = gemv_q8_0_;
  else if (type == "Q5_0") fn = gemv_q5_0_;
  if (!fn) return false;
  void* pw = const_cast<void*>(W);
  void* px = in.data();
  void* py = out.data();
  int nin = n_in, nout = n_out;
  void* args[] = {&pw, &px, &py, &nin, &nout};
  cuda::launch(fn, (n_out + 7) / 8, 1, 1, 256, 1, 1, 0, args);
  return true;
}

void DenseRunner::matvec_f32_ptr(const void* W, int n_out, int n_in, cuda::Buffer& in,
                                 cuda::Buffer& out) {
  if (ENGINE_ENV_ON("ENGINE_DBG_MATVEC") && dbg_matvec_++ < 4)
    std::fprintf(stderr, "[matvec] ptr n_tokens_cur=%d n_out=%d n_in=%d\n", n_tokens_cur_, n_out,
                 n_in);
  if (n_tokens_cur_ <= 4) {
    void* pw = const_cast<void*>(W);
    void* px = in.data();
    void* py = out.data();
    int nin = n_in, nout = n_out, ntk = n_tokens_cur_;
    void* args[] = {&pw, &px, &py, &nin, &nout, &ntk};
    cuda::launch(k_gemvf32_, (n_out + 7) / 8, n_tokens_cur_, 1, 256, 1, 1, 0, args);
    return;
  }
  cublas::gemm_f32_nt(n_tokens_cur_, n_out, n_in, static_cast<const float*>(in.data()),
                      static_cast<const float*>(W), static_cast<float*>(out.data()));
}

void DenseRunner::matvec_f32(const cuda::Buffer& W, int n_out, int n_in, cuda::Buffer& in,
                            cuda::Buffer& out) {
  if (ENGINE_ENV_ON("ENGINE_DBG_MATVEC") && dbg_matvec_++ < 6)
    std::fprintf(stderr, "[matvec] buf n_tokens_cur=%d n_out=%d n_in=%d -> %s\n", n_tokens_cur_,
                 n_out, n_in, n_tokens_cur_ <= 4 ? "GEMV" : "cublas");
  if (n_tokens_cur_ <= 4) {
    void* pw = const_cast<void*>(W.data());
    void* px = in.data();
    void* py = out.data();
    int nin = n_in, nout = n_out, ntk = n_tokens_cur_;
    void* args[] = {&pw, &px, &py, &nin, &nout, &ntk};
    cuda::launch(k_gemvf32_, (n_out + 7) / 8, n_tokens_cur_, 1, 256, 1, 1, 0, args);
    return;
  }
  cublas::gemm_f32_nt(n_tokens_cur_, n_out, n_in, static_cast<const float*>(in.data()),
                      static_cast<const float*>(W.data()), static_cast<float*>(out.data()));
}

void DenseRunner::run_moe(int layer, int n_tokens) {
  auto& L = layers_[layer];
  const bool acc_moe = prof_acc::counting(n_tokens);
  const auto acc_moe0 = std::chrono::steady_clock::now();
  static const bool kProf = ENGINE_ENV_ON("ENGINE_PROFILE");
  const auto t0 = std::chrono::steady_clock::now();
  auto ms_since = [&](const std::chrono::steady_clock::time_point& a) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
  };
  const bool kDebug = debug_;
  if (kDebug)
    std::fprintf(stderr, "[dbg] run_moe layer=%d n=%d qexp=%zu exp=%zu\n", layer, n_tokens,
                 L.q_experts.size(), L.experts.size());
  const int ne = ir_.moe.n_expert;
  static const int topk_ov = experts_topk_;
  const int topk = topk_ov > 0 ? std::min(topk_ov, ir_.moe.n_expert_used)
                              : ir_.moe.n_expert_used;
  if (ENGINE_ENV_ON("ENGINE_DBG_LG2") && layer == 1)
    std::fprintf(stderr, "[lg2] router gemm\n");
  const bool router_kernel_mode =
      ENGINE_ENV_ON("ENGINE_ROUTER_KERNEL") ||
      layer_graph_ || layer_graph_mode_;
  if (k_routergemm_ && router_kernel_mode) {
    void* ph = hn_.data();
    void* pw = L.router.data();
    void* po = router_logits_.data();
    int nn = n_tokens, hh = hidden_, nee = ne;
    void* args[] = {&ph, &pw, &po, &nn, &hh, &nee};
    cuda::launch(k_routergemm_, (n_tokens * ne + 127) / 128, 1, 1, 128, 1, 1, 0, args);
  } else {
    cublas::gemm_f32_nt(n_tokens, ne, hidden_, static_cast<const float*>(hn_.data()),
                        static_cast<const float*>(L.router.data()),
                        static_cast<float*>(router_logits_.data()));
  }
  if (kDebug) std::fprintf(stderr, "[dbg]  moe: router gemm done\n");
  const double t_router = kProf ? ms_since(t0) : 0.0;
  // Decide up-front whether the host round-trip can be skipped entirely.
  const int gpu_n_pre = (L.cpu_begin >= 0 && L.cpu_begin <= ne)
                            ? L.cpu_begin
                            : ((gpu_experts_ < 0) ? ne : std::min(gpu_experts_, ne));
  auto type_id_of = [](const std::string& t) {
    if (t == "Q4_K") return 0;
    if (t == "Q5_K") return 1;
    if (t == "Q6_K") return 2;
    if (t == "Q8_0") return 3;
    if (t == "Q5_1") return 4;
    return -1;
  };
  const bool dev_ok = ENGINE_ENV_ON("ENGINE_DEV_ROUTE") ||
                      layer_graph_ || layer_graph_mode_ || dev_route_default_;
  bool use_dev = false;
  if (dev_ok && gpu_n_pre == ne && device_route_supported(ne) && L.all_ptr_idx >= 0 &&
      k_moeroute_ && n_tokens <= 8 && !L.q_experts.empty()) {
    use_dev = type_id_of(L.q_experts[0].type) >= 0 && type_id_of(L.q_experts[1].type) >= 0 &&
              type_id_of(L.q_experts[2].type) >= 0 &&
              !ENGINE_ENV_ON("ENGINE_DEV_NO_Q51");
  }
  if (ENGINE_ENV_ON("ENGINE_DBG_DEVWHY") && layer <= 3)
    std::fprintf(stderr,
                 "[devwhy] layer %d use_dev=%d gpu_n=%d ne=%d all_ptr=%d ntok=%d qexp=%zu "
                 "t=%s,%s,%s\n",
                 layer, static_cast<int>(use_dev), gpu_n_pre, ne, L.all_ptr_idx, n_tokens,
                 L.q_experts.size(),
                 L.q_experts.empty() ? "-" : L.q_experts[0].type.c_str(),
                 L.q_experts.size() > 1 ? L.q_experts[1].type.c_str() : "-",
                 L.q_experts.size() > 2 ? L.q_experts[2].type.c_str() : "-");
  // Diagnostic: skip the router round-trip (wrong results, measures its cost).
  static const bool fixed_route = ENGINE_ENV_ON("ENGINE_MOE_FIXED_ROUTING");
  // A CPU-resident MoE layer needs exactly one host round trip per token.  Its GPU-side
  // shared-expert block does NOT depend on the routing, so it is issued *before* that
  // round trip: the stream flush then covers it, the queue is empty while the host pool
  // runs the expert GEMV, and no WDDM submission work competes with those threads.  The
  // numbers are unchanged (same kernels, same order of the adds into xout_).
  const bool any_cpu_pre =
      gpu_n_pre < ne && (static_cast<int>(L.host_gate.size()) == ne || !L.h_gate_bytes.empty());
  std::chrono::steady_clock::time_point t_shared0;
  double t_experts = 0.0;
  auto run_shared_block = [&]() {
    t_shared0 = std::chrono::steady_clock::now();
    // device-side zeroing: a host upload is a synchronous memcpy, which CUDA graph
    // capture forbids (CUresult=900)
    {
      void* pz = xout_.data();
      long long nz = static_cast<long long>(n_tokens) * hidden_;
      void* zargs[] = {&pz, &nz};
      cuda::launch(k_zerofill_, 64, 1, 1, 256, 1, 1, 0, zargs);
    }
    if (layer == 1) {
      if (const char* dp = ENGINE_ENV_STR("ENGINE_DUMP")) {
        cuda::stream_sync();
        std::vector<float> h(static_cast<size_t>(n_tokens) * hidden_);
        hn_.download(h.data(), h.size() * 4);
        std::ofstream f1(std::string(dp) + "_hn1.f32", std::ios::binary);
        f1.write(reinterpret_cast<const char*>(h.data()),
                 static_cast<std::streamsize>(h.size() * 4));
      }
    }
    if (kDebug) std::fprintf(stderr, "[dbg]  moe: xout zeroed\\n");

    t_experts = ms_since(t0);
    g_prof_experts_ms += t_experts;
    const auto tg_shared = std::chrono::steady_clock::now();
    // shared experts: dense FFN, always active
    if (L.has_shared && !no_shared_) {
      // persistent scratch: allocating cuda::Buffer per layer costs ~1 ms per malloc
      // (cudaMalloc synchronises), which dominated the Coder's decode time.
      const size_t need = static_cast<size_t>(n_tokens) * static_cast<size_t>(L.sh_inter);
      if (ENGINE_ENV_ON("ENGINE_DBG_SCRATCH") && layer < 3)
        std::fprintf(stderr, "[scratch] layer %d need=%zu size=%zu\\n", layer, need * 4,
                     sh_scratch_.size());
      if (sh_scratch_.size() < need * 4) sh_scratch_ = cuda::Buffer(need * 4);
      if (shu_scratch_.size() < need * 4) shu_scratch_ = cuda::Buffer(need * 4);
      if (sha_scratch_.size() < need * 4) sha_scratch_ = cuda::Buffer(need * 4);
      if (shd_scratch_.size() < static_cast<size_t>(n_tokens) * hidden_ * 4)
        shd_scratch_ = cuda::Buffer(static_cast<size_t>(n_tokens) * hidden_ * 4);
      cuda::Buffer& shg = sh_scratch_;
      cuda::Buffer& shu = shu_scratch_;
      cuda::Buffer& sha = sha_scratch_;
      cuda::Buffer& shd = shd_scratch_;
      // f32 GEMV when decoding (cuBLAS Sgemm with M=1 is ~100x slower)
      const auto tg0 = std::chrono::steady_clock::now();
      if (L.sh_quant) {
        matmul(L.qsh_gate, n_tokens, hn_, shg);
        matmul(L.qsh_up, n_tokens, hn_, shu);
      } else {
        matvec_f32(L.sh_gate, L.sh_inter, hidden_, hn_, shg);
        matvec_f32(L.sh_up, L.sh_inter, hidden_, hn_, shu);
      }
      if (kProf) {
        cuda::sync();
        std::fprintf(stderr, "[moe-prof]  layer %d (n=%d): experts %.2f ms, sh gate+up %.2f ms\\n",
                     layer, n_tokens, t_experts, ms_since(tg0));
      }
      {
        long long total = static_cast<long long>(n_tokens) * L.sh_inter;
        void* pg = shg.data();
        void* pu = shu.data();
        void* pa = sha.data();
        void* args[] = {&pg, &pu, &pa, &total};
        cuda::launch(k_silu_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
      }
      const auto td0 = std::chrono::steady_clock::now();
      if (L.sh_quant) matmul(L.qsh_down, n_tokens, sha, shd);
      else matvec_f32(L.sh_down, hidden_, L.sh_inter, sha, shd);
      if (kProf) {
        cuda::sync();
        std::fprintf(stderr, "[moe-prof]  sh down %.2f ms\\n", ms_since(td0));
      }
      double sh_scale = 1.0;
      if (const char* e = ENGINE_ENV_STR("ENGINE_SH_SCALE")) sh_scale = std::atof(e);
      if (sh_scale != 1.0) {
        const long long tot = static_cast<long long>(n_tokens) * hidden_;
        std::vector<float> host(static_cast<size_t>(tot));
        cuda::stream_sync();
        shd.download(host.data(), host.size() * 4);
        for (auto& v : host) v = static_cast<float>(v * sh_scale);
        shd.upload(host.data(), host.size() * 4);
      }
      const auto tgi = std::chrono::steady_clock::now();
      // qwen3.5-MoE scales the shared expert by sigmoid(gate . h)
      if (L.sh_gate_inp.data()) {
        void* po = shd.data();
        void* ph = hn_.data();
        void* pw = L.sh_gate_inp.data();
        int nn = n_tokens, hh = hidden_;
        void* gargs[] = {&po, &ph, &pw, &nn, &hh};
        cuda::launch(k_moeshgate_, n_tokens, 1, 1, 256, 1, 1, 0, gargs);
      }
      {
        long long total = static_cast<long long>(n_tokens) * hidden_;
        void* pa = xout_.data();
        void* pb = shd.data();
        void* args[] = {&pa, &pb, &total};
        cuda::launch(k_add_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
      }
      if (kProf) {
        cuda::sync();
        std::fprintf(stderr, "[moe-prof]  gate_inp+add %.2f ms\\n", ms_since(tgi));
      }
    }
    g_prof_shared_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg_shared)
            .count();
  };
  std::vector<float> rl_owned;
  const float* rl = nullptr;
  const bool dl_routing = !fixed_route && !use_dev;
  if (dl_routing) {
    const size_t rl_bytes = static_cast<size_t>(n_tokens) * ne * 4;
    rl_pin_.resize(rl_bytes);
    // async: the single sync happens after the activation copy below (one flush per
    // CPU layer instead of three)
    router_logits_.download_async(rl_pin_.data(), rl_bytes);
    // The activation copy is independent of routing: enqueue it now so a single
    // stream sync covers both (one WDDM flush per CPU layer).
    const size_t n_elems_pre = static_cast<size_t>(n_tokens) * hidden_;
    if (gpu_n_pre < ne) {
      hn_pin_.resize(n_elems_pre * 4);
      hn_.download_async(static_cast<float*>(hn_pin_.data()), n_elems_pre * 4);
    }
  } else {
    rl_owned.assign(static_cast<size_t>(n_tokens) * ne, 0.f);
    rl = rl_owned.data();
  }
  if (acc_moe) prof_acc::moe_pre_ms += prof_acc::since(acc_moe0);
  // Host-resident experts: issue the shared-expert GPU block now so the flush below is
  // the only one and the queue drains before the host GEMV starts.
  // The block depends only on hn_ and the layer's fixed weights, i.e. it is
  // token-invariant, so on decode it is recorded once as a per-layer graph.  That saves
  // the host submission time (~0.12 ms/layer of cuLaunchKernel calls on the 35B, all of
  // it on the critical path because the host must finish submitting before it can sync),
  // while keeping the exact same kernel sequence and addition order.
  auto submit_shared_block = [&]() {
    auto& SL = layers_[layer];
    const bool sh_graph = cpu_layer_ffn_graph_ && !no_graph_cfg_ && n_tokens == 1 &&
                          !graph_capture_ && !use_dev && !ENGINE_ENV_ON("ENGINE_DUMP") &&
                          !ENGINE_ENV_ON("ENGINE_PROFILE") && !ENGINE_ENV_ON("ENGINE_KPROF") &&
                          !ENGINE_ENV_STR("ENGINE_SH_SCALE");
    if (sh_graph && SL.cpu_ffn_graph_state == 2 && SL.cpu_ffn_graph) {
      SL.cpu_ffn_graph->launch();
    } else {
      run_shared_block();
      if (sh_graph) {
        // First decode-shaped call: the block already ran above (this token's result),
        // now record the same sequence.  Recording executes nothing, so the state it
        // captures stays as the normal path left it.
        cuda::stream_sync();
        auto g = std::make_unique<cuda::Graph>();
        try {
          g->capture_begin();
          graph_capture_ = true;
          run_shared_block();
          graph_capture_ = false;
          g->capture_end();
          SL.cpu_ffn_graph = std::move(g);
          SL.cpu_ffn_graph_state = 2;
          if (ENGINE_ENV_ON("ENGINE_DBG_GRAPH"))
            std::fprintf(stderr, "[graph] layer %d cpu ffn block captured\n", layer);
        } catch (const std::exception& e) {
          graph_capture_ = false;
          g->capture_abort();
          // Never retry in this process: a half-recorded stream would poison later work.
          cpu_layer_ffn_graph_ = false;
          std::fprintf(stderr, "[cpu-ffn-graph] disabled: %s\n", e.what());
        }
      }
    }
  };
  // Decode: submit the block on a second stream *after* the routing sync, so its GPU
  // work overlaps the host GEMV instead of extending the sync.  Prefill keeps the
  // historical order (submit before the sync, which also drains the queue).
  const bool sh_dual = any_cpu_pre && cpu_dual_stream_ && n_tokens == 1 && !graph_capture_ &&
                       !ENGINE_ENV_ON("ENGINE_DUMP") && !ENGINE_ENV_ON("ENGINE_PROFILE") &&
                       !ENGINE_ENV_ON("ENGINE_KPROF");
  bool sh_dual_submitted = false;
  if (any_cpu_pre) {
    if (sh_dual) {
      // The block only reads hn_ and the layer's fixed weights, so it only has to wait
      // for the main stream up to the norms/router (recorded here as an event); after
      // that it runs concurrently with the routing round-trip and the host GEMV.
      if (!sh_stream_) sh_stream_ = cuda::create_stream();
      if (!sh_ev_) sh_ev_ = std::make_unique<cuda::Event>();
      sh_ev_->record_on(cuda::current_stream());
      sh_ev_->stream_wait(sh_stream_);
      void* saved_stream = cuda::current_stream();
      cuda::set_stream(sh_stream_);
      submit_shared_block();
      cuda::set_stream(saved_stream);
      sh_dual_submitted = true;
    } else {
      submit_shared_block();
    }
  }
  if (dl_routing) {
    cuda::stream_sync();
    rl = static_cast<const float*>(rl_pin_.data());
  }
  const auto acc_t_route0 = std::chrono::steady_clock::now();

  // host-side routing: softmax -> top-k -> renormalize (matches the reference)
  std::vector<std::vector<std::pair<int, float>>> route(static_cast<size_t>(n_tokens));
  std::vector<int> idx(static_cast<size_t>(ne));
  std::vector<float> pr(static_cast<size_t>(ne));
  if (!use_dev) {
    for (int t = 0; t < n_tokens; ++t) {
      const float* row = rl + static_cast<size_t>(t) * ne;
      if (ir_.moe.router_kind == "sigmoid_topk") {
        for (int e = 0; e < ne; ++e) pr[e] = 1.f / (1.f + std::exp(-row[e]));
      } else {
        float mx = -1e30f;
        for (int e = 0; e < ne; ++e) mx = std::max(mx, row[e]);
        double sum = 0;
        for (int e = 0; e < ne; ++e) {
          pr[e] = std::exp(row[e] - mx);
          sum += pr[e];
        }
        for (int e = 0; e < ne; ++e) pr[e] = static_cast<float>(pr[e] / sum);
      }
      for (int e = 0; e < ne; ++e) idx[e] = e;
      std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                        [&](int a, int b) { return pr[a] > pr[b]; });
      float wsum = 0;
      for (int k = 0; k < topk; ++k) wsum += pr[idx[k]];
      for (int k = 0; k < topk; ++k)
        route[t].push_back({idx[k], ir_.moe.norm_topk ? pr[idx[k]] / wsum : pr[idx[k]]});
    }
  }

  g_prof_router_ms += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
  if (acc_moe && !use_dev) prof_acc::moe_route_ms += prof_acc::since(acc_t_route0);
  if (fixed_route) {
    for (int t = 0; t < n_tokens; ++t) {
      route[t].clear();
      for (int k = 0; k < topk; ++k) route[t].push_back({k, 1.f / static_cast<float>(topk)});
    }
  }
  if (ENGINE_ENV_ON("ENGINE_DBG_DEVROUTE") && layer == 1) {
    std::fprintf(stderr, "[route-host] layer 1:");
    for (int t = 0; t < n_tokens && t < 1; ++t)
      for (const auto& r : route[t]) std::fprintf(stderr, " (e%d,w%.4f)", r.first, r.second);
    std::fprintf(stderr, "\n");
  }
  if (kDebug) std::fprintf(stderr, "[dbg]  moe: routing done\n");
  // GPU-resident layer: no host round trip to hide, so the shared-expert block is
  // issued here (CPU-resident layers already issued it before their single sync).
  if (!any_cpu_pre) run_shared_block();
  // Per-layer placement wins (layer-granular expert placement); gpu_experts_ is only
  // the fallback for models loaded without it.
  const int gpu_n = (L.cpu_begin >= 0 && L.cpu_begin <= ne) ? L.cpu_begin
                                                            : ((gpu_experts_ < 0) ? ne : std::min(gpu_experts_, ne));
  const bool any_cpu = gpu_n < ne &&
                       (static_cast<int>(L.host_gate.size()) == ne || !L.h_gate_bytes.empty());
  // Pinned staging: pageable copies on WDDM cost ~2 ms each regardless of size,
  // which dominated the per-CPU-layer cost (4-5 copies x 9 layers).
  float* xout_host = nullptr;
  float* hn_host = nullptr;
  if (any_cpu) {
    const size_t n_elems = static_cast<size_t>(n_tokens) * hidden_;
    hn_pin_.resize(n_elems * 4);
    xout_pin_.resize(n_elems * 4);
    hn_host = static_cast<float*>(hn_pin_.data());
    xout_host = static_cast<float*>(xout_pin_.data());
    for (size_t i = 0; i < n_elems; ++i) xout_host[i] = 0.f;
    const double sync_ms = 0.0, dl_ms = 0.0;
    const auto cp_t1 = std::chrono::steady_clock::now();
    if (ENGINE_ENV_ON("ENGINE_DBG_CPULAYER") && n_tokens == 1) {
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - cp_t1).count();
      static double sacc = 0, dacc = 0;
      static int cnt = 0;
      sacc += sync_ms; dacc += ms; ++cnt;
      std::fprintf(stderr, "[cpulayer] sync %.2f ms, D2H %.2f ms (avg %.2f/%.2f over %d)\n",
                   sync_ms, ms, sacc / cnt, dacc / cnt, cnt);
    }
  }

  const auto t_loop0 = std::chrono::steady_clock::now();
  const bool kCpuDbg = any_cpu && ENGINE_ENV_ON("ENGINE_DBG_CPULAYER") &&
                       n_tokens == 2;
  // Pack the whole routing table once: uploading per expert cost two synchronous
  // H2D copies x 6 experts x 26 layers per token (~0.5 s/token on the Coder).
  std::vector<int> all_idx;
  std::vector<float> all_w;
  std::vector<int> exp_off(static_cast<size_t>(ne) + 1, 0);
  for (int e = 0; e < ne; ++e) {
    for (int t = 0; t < n_tokens; ++t)
      for (const auto& pr2 : route[t])
        if (pr2.first == e) {
          all_idx.push_back(t);
          all_w.push_back(pr2.second);
        }
    exp_off[static_cast<size_t>(e) + 1] = static_cast<int>(all_idx.size());
  }
  // The routing table is only consumed from the device by the GPU-resident expert
  // kernels (fused / device-route) and by the mixed-placement gather path; a layer
  // whose experts all live on the host never reads it from the device.  The two
  // cuMemcpyHtoD calls are synchronous and on WDDM they flush the queue, so issuing
  // them for every CPU layer cost a hidden wait per layer.
  if (!all_idx.empty() && gpu_n > 0) {
    // Pinned staging + async copies: a synchronous cuMemcpyHtoD flushes the WDDM queue,
    // which cost two queue flushes per host-routed GPU layer (Coder).  The staging
    // buffers are members and are only rewritten by the next run_moe call, which is
    // always preceded by a stream sync, so the async source stays valid.
    const size_t route_bytes = all_idx.size() * 4;
    d_idx_pin_.resize(route_bytes);
    d_w_pin_.resize(route_bytes);
    std::memcpy(d_idx_pin_.data(), all_idx.data(), route_bytes);
    std::memcpy(d_w_pin_.data(), all_w.data(), route_bytes);
    d_idx_.upload_async(d_idx_pin_.data(), route_bytes);
    d_w_.upload_async(d_w_pin_.data(), route_bytes);
  }
  const int* d_idx_base = static_cast<const int*>(d_idx_.data());
  const float* d_w_base = static_cast<const float*>(d_w_.data());

  // ---- batched host expert pass ------------------------------------------------
  // Every routed CPU expert of this layer is evaluated in two parallel regions
  // (gate+up+silu, then down).  The per-expert loop below woke the thread pool and
  // waited on a barrier three times per expert (24 wake-ups for topk=8); the wake-up
  // alone was ~25 us, i.e. ~0.6 ms per CPU layer.  The row kernel and the accumulation
  // order into xout_host are unchanged, so the result is bit-identical.
  static const bool cpu_moe_batch = engine::config::get_bool("cpu_moe_batch");
  const bool cpu_quant_host = any_cpu && !L.h_gate_bytes.empty() && hn_host != nullptr;
  bool cpu_batched = false;
  if (cpu_quant_host && cpu_moe_batch && !ENGINE_ENV_ON("ENGINE_EXPERT_ARENA")) {
    const int inter = expert_inter_;
    const std::size_t g_slab_g = static_cast<std::size_t>(inter) * hidden_ / L.h_be_g * L.h_bb_g;
    const std::size_t g_slab_u = static_cast<std::size_t>(inter) * hidden_ / L.h_be_u * L.h_bb_u;
    const std::size_t d_slab = static_cast<std::size_t>(hidden_) * inter / L.h_be_d * L.h_bb_d;
    std::vector<const unsigned char*> ag, au, ad;
    std::vector<int> atok;
    std::vector<float> aw;
    for (int e = gpu_n; e < ne; ++e) {
      for (int k = exp_off[static_cast<std::size_t>(e)];
           k < exp_off[static_cast<std::size_t>(e) + 1]; ++k) {
        ag.push_back(L.h_gate_bytes.data() + static_cast<std::size_t>(e) * g_slab_g);
        au.push_back(L.h_up_bytes.data() + static_cast<std::size_t>(e) * g_slab_u);
        ad.push_back(L.h_down_bytes.data() + static_cast<std::size_t>(e) * d_slab);
        atok.push_back(all_idx[static_cast<std::size_t>(k)]);
        aw.push_back(all_w[static_cast<std::size_t>(k)]);
      }
    }
    if (!atok.empty()) {
      const auto bt0 = std::chrono::steady_clock::now();
      cpu::moe_forward_mt(L.h_type_g, ag.data(), L.h_type_u, au.data(), L.h_type_d, ad.data(),
                          atok.data(), aw.data(), static_cast<int>(atok.size()), n_tokens,
                          hidden_, inter, hn_host, xout_host);
      if (prof_acc::counting(n_tokens)) prof_acc::gemv_ms += prof_acc::since(bt0);
      cpu_batched = true;
      if (ENGINE_ENV_ON("ENGINE_DBG_CPUEXEC")) {
        static double ssum = 0;
        static int scnt = 0;
        static double last_sum = 0;
        static int last_cnt = 0;
        const double one_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - bt0)
                .count();
        ssum += one_ms;
        ++scnt;
        if (scnt % 25 == 0) {
          // Cumulative average (historical output) + the exact average of the last 25-layer
          // window.  The window figure is what the priority/margin A/B compares: the first
          // window is dominated by lazy CUDA-graph capture and page-in, so the cumulative
          // value keeps drifting for hundreds of layers.
          const int win_cnt = scnt - last_cnt;
          const double win_avg = win_cnt > 0 ? (ssum - last_sum) / win_cnt : 0.0;
          std::fprintf(stderr,
                       "[cpuexec] batched host matvec avg %.2f ms over %d layers "
                       "(last %d: %.3f ms)\n",
                       ssum / scnt, scnt, win_cnt, win_avg);
          last_sum = ssum;
          last_cnt = scnt;
        }
      }
    }
  }

  // Join the shared-expert stream before the fused/per-expert GPU kernels and the tail
  // add3 read xout_ on the main stream (the host GEMV above ran concurrently with it).
  if (sh_dual_submitted) cuda::stream_sync(sh_stream_);
  const auto acc_t_tail0 = std::chrono::steady_clock::now();
  // ---- fused path: one kernel for gate+up+silu and one for down+scatter ----
  bool fused_done = false;
  static const bool fused_env = !no_moe_fused_;
  const bool can_fuse = fused_env && !layer_graph_mode_ && n_tokens == 1 && gpu_n == ne &&
                        !L.q_experts.empty() && !all_idx.empty();
  if (can_fuse) {
    auto type_id = [](const std::string& t) {
      if (t == "Q4_K") return 0;
      if (t == "Q5_K") return 1;
      if (t == "Q6_K") return 2;
      if (t == "Q8_0") return 3;
      if (t == "Q5_1") return 4;
      return -1;
    };
    const int type_g = type_id(L.q_experts[0].type);
    const int type_u = type_id(L.q_experts[1].type);
    const int type_d = type_id(L.q_experts[2].type);
    const int nact = static_cast<int>(all_idx.size());
    if (type_g >= 0 && type_u >= 0 && type_d >= 0) {
      std::vector<const unsigned char*> gp(nact), up(nact), dp(nact);
      int slot = 0;
      for (int e = 0; e < ne; ++e) {
        const int cnt = exp_off[static_cast<size_t>(e) + 1] - exp_off[static_cast<size_t>(e)];
        if (!cnt) continue;
        const unsigned char* pg =
            static_cast<const unsigned char*>(L.q_experts[e * 3 + 0].data.data());
        const unsigned char* pu =
            static_cast<const unsigned char*>(L.q_experts[e * 3 + 1].data.data());
        const unsigned char* pdv =
            static_cast<const unsigned char*>(L.q_experts[e * 3 + 2].data.data());
        for (int k = 0; k < cnt; ++k) {
          gp[slot] = pg;
          up[slot] = pu;
          dp[slot] = pdv;
          ++slot;
        }
      }
      if (slot == nact) {
        const size_t need = static_cast<size_t>(nact) * 8;
        if (moe_gp_.size() < need) {
          moe_gp_ = cuda::Buffer(need);
          moe_up_ = cuda::Buffer(need);
          moe_dp_ = cuda::Buffer(need);
        }
        if (moe_gact_.size() < static_cast<size_t>(nact) * expert_inter_ * 4)
          moe_gact_ = cuda::Buffer(static_cast<size_t>(nact) * expert_inter_ * 4);
        moe_gp_.upload(gp.data(), gp.size() * 8);
        moe_up_.upload(up.data(), up.size() * 8);
        moe_dp_.upload(dp.data(), dp.size() * 8);
        {
          void* pgp = moe_gp_.data();
          void* pup = moe_up_.data();
          void* pi = const_cast<int*>(d_idx_base);
          void* px = hn_.data();
          void* pa = moe_gact_.data();
          int hid = hidden_, inter = expert_inter_, tg = type_g, tu = type_u;
          void* args[] = {&pgp, &pup, &pi, &px, &pa, &hid, &inter, &tg, &tu};
          cuda::launch(k_moegateup_, (expert_inter_ + 7) / 8, nact, 1, 256, 1, 1, 0, args);
        }
        {
          void* pdp = moe_dp_.data();
          void* pi = const_cast<int*>(d_idx_base);
          void* pw = const_cast<float*>(d_w_base);
          void* pa = moe_gact_.data();
          void* po = xout_.data();
          int hid = hidden_, inter = expert_inter_, td = type_d;
          void* args[] = {&pdp, &pi, &pw, &pa, &po, &hid, &inter, &td};
          cuda::launch(k_modeown_, (hidden_ + 7) / 8, nact, 1, 256, 1, 1, 0, args);
        }
        fused_done = true;
      }
    }
  }

  // ---- host-sync-free path: device routing + fused expert kernels ----
  // NOTE: opt-in while the device router is being debugged (illegal access on the
  // tiny_moe fixture). The plain fused path stays the default.
  // Opt-in: the device router still differs numerically from the host path on real
  // Q4_K/Q5_K experts (fixtures with Q8_0 pass), so it stays off by default.
  // Default on when most MoE layers are GPU-resident: then the per-layer host
  // round-trip dominates and the device router removes it. Models that offload most
  // layers keep the host path (the device router would be a wash there).
  // Default on when the model's expert layout is numerically verified for the device
  // GEMV branches (see the dev_route config key / dev_route_default_ above).
  // Also on while a layer is being captured into a graph: the host round-trip would
  // otherwise put a forbidden stream sync inside the capture.
  static const bool dev_route_auto = dev_route_default_;
  const bool dev_route_env =
      ENGINE_ENV_ON("ENGINE_DEV_ROUTE") || layer_graph_ || layer_graph_mode_ ||
      dev_route_auto;
  const int gpu_n_dr = (L.cpu_begin >= 0 && L.cpu_begin <= ne) ? L.cpu_begin
                                                              : ((gpu_experts_ < 0) ? ne : ne);
  if (dev_route_env && gpu_n_dr == ne && device_route_supported(ne) && L.all_ptr_idx >= 0 &&
      k_moeroute_ && n_tokens <= 8) {
    auto type_id = [](const std::string& t) {
      if (t == "Q4_K") return 0;
      if (t == "Q5_K") return 1;
      if (t == "Q6_K") return 2;
      if (t == "Q8_0") return 3;
      if (t == "Q5_1") return 4;
      return -1;
    };
    const int tg = type_id(L.q_experts[0].type), tu = type_id(L.q_experts[1].type),
              td = type_id(L.q_experts[2].type);
    const int nact = n_tokens * topk;
    if (tg >= 0 && tu >= 0 && td >= 0) {
      if (ENGINE_ENV_ON("ENGINE_DBG_DEVROUTE"))
        std::fprintf(stderr,
                     "[devroute] layer %d ne=%d topk=%d nact=%d idx=%d types=%d/%d/%d "
                     "gp0=%p inter=%d hidden=%d\n",
                     layer, ne, topk, nact, L.all_ptr_idx, tg, tu, td,
                     all_gp_[static_cast<size_t>(L.all_ptr_idx)].data(), expert_inter_, hidden_);
      if (slot_exp_.size() < static_cast<size_t>(nact) * 4) {
        slot_exp_ = cuda::Buffer(static_cast<size_t>(nact) * 4);
        slot_tok_ = cuda::Buffer(static_cast<size_t>(nact) * 4);
        slot_w_ = cuda::Buffer(static_cast<size_t>(nact) * 4);
        route_slots_zeroed_ = false;   // fresh allocation: clear before the next route
      }
      if (moe_gact_.size() < static_cast<size_t>(nact) * expert_inter_ * 4)
        moe_gact_ = cuda::Buffer(static_cast<size_t>(nact) * expert_inter_ * 4);
      // xout_ was zeroed right after routing and the shared experts have already
      // added their contribution -- zeroing again here would wipe it out.
      if (ENGINE_ENV_ON("ENGINE_DBG_LG2") && layer == 1)
        std::fprintf(stderr, "[lg2] route kernel\n");
      ensure_route_slots_zeroed(nact);
      {
        void* pl = router_logits_.data();
        int nen = ne, tk = topk, sg = (ir_.moe.router_kind == "sigmoid_topk") ? 1 : 0;
        int nt = ir_.moe.norm_topk ? 1 : 0;
        void* pe = slot_exp_.data();
        void* pt = slot_tok_.data();
        void* pw = slot_w_.data();
        void* args[] = {&pl, &nen, &tk, &sg, &nt, &pe, &pt, &pw};
        cuda::launch(k_moeroute_, n_tokens, 1, 1, 32, 1, 1, 0, args);
        if (ENGINE_ENV_ON("ENGINE_DBG_DEVROUTE")) {
          cuda::sync();
          std::vector<int> he(nact), ht(nact);
          std::vector<float> hw(nact);
          slot_exp_.download(he.data(), he.size() * 4);
          slot_tok_.download(ht.data(), ht.size() * 4);
          slot_w_.download(hw.data(), hw.size() * 4);
          std::fprintf(stderr, "[devroute]   route ok: act=%zu e0=%d t0=%d w0=%.4f\n",
                       moe_gact_.size(), he[0], ht[0], hw[0]);
        }
      }
      {
        void* pg = all_gp_[static_cast<size_t>(L.all_ptr_idx)].data();
        void* pu = all_up_[static_cast<size_t>(L.all_ptr_idx)].data();
        void* pe = slot_exp_.data();
        void* pt = slot_tok_.data();
        void* px = hn_.data();
        void* pa = moe_gact_.data();
        int hid = hidden_, inter = expert_inter_, t1 = tg, t2 = tu;
        void* args[] = {&pg, &pu, &pe, &pt, &px, &pa, &hid, &inter, &t1, &t2};
        cuda::launch(k_moegateupdev_, (expert_inter_ + 31) / 32, nact, 1, 256, 1, 1, 0, args);
        if (ENGINE_ENV_ON("ENGINE_DBG_DEVROUTE")) {
          cuda::sync();
          std::fprintf(stderr, "[devroute]   gateup ok\n");
        }
      }
      {
        void* pd = all_dp_[static_cast<size_t>(L.all_ptr_idx)].data();
        void* pe = slot_exp_.data();
        void* pt = slot_tok_.data();
        void* pw = slot_w_.data();
        void* pa = moe_gact_.data();
        void* po = xout_.data();
        int hid = hidden_, inter = expert_inter_, t3 = td;
        void* args[] = {&pd, &pe, &pt, &pw, &pa, &po, &hid, &inter, &t3};
        cuda::launch(k_modeowndev_, (hidden_ + 31) / 32, nact, 1, 256, 1, 1, 0, args);
        if (ENGINE_ENV_ON("ENGINE_DBG_DEVROUTE")) {
          cuda::sync();
          std::fprintf(stderr, "[devroute]   down ok\n");
        }
      }
      {
        long long total = static_cast<long long>(n_tokens) * hidden_;
        void* pa = x_.data();
        void* pb = xout_.data();
        void* args[] = {&pa, &pb, &total};
        cuda::launch(k_add_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
      }
      static const int hash_layer = []() { const char* e = ENGINE_ENV_STR("ENGINE_DBG_MOEHASH_LAYER"); return e ? std::atoi(e) : 1; }();
  if (ENGINE_ENV_ON("ENGINE_DBG_MOEHASH") && layer == hash_layer) {
    cuda::sync();
    std::vector<float> hh(static_cast<size_t>(n_tokens) * hidden_);
    xout_.download(hh.data(), hh.size() * 4);
    double acc = 0.0;
    for (float vv : hh) acc += static_cast<double>(vv) * vv;
    std::fprintf(stderr, "[moehash] dev n=%d xout norm %.6f\n", n_tokens, std::sqrt(acc));
  }
  return;   // no host sync, no host routing
    }
  }

  std::vector<int> ti;
  std::vector<float> tw;
  // Host-batched CPU-expert layers (gpu_n == 0) were fully evaluated above, so this
  // per-expert loop would only spin over ne empty ranges: 3 steady_clock reads + 2 vector
  // assigns per iteration, measured 22 us/layer (~0.5 ms/token on the 35B, 24 CPU layers).
  // Skipping it is behaviour-neutral (every iteration would hit a pure-`continue` path).
  const int expert_loop_end = (cpu_batched && gpu_n == 0) ? 0 : ne;
  for (int e = 0; e < expert_loop_end && !fused_done; ++e) {
    const int off_e = exp_off[static_cast<size_t>(e)];
    const int off_e1 = exp_off[static_cast<size_t>(e) + 1];
    ti.assign(all_idx.begin() + off_e, all_idx.begin() + off_e1);
    tw.assign(all_w.begin() + off_e, all_w.begin() + off_e1);
    const auto t_iter = std::chrono::steady_clock::now();
    const int m = static_cast<int>(ti.size());
    if (kDebug)
      std::fprintf(stderr, "[dbg]  expert %d m=%d\n", e, m);
    struct ActiveTimer {
      std::chrono::steady_clock::time_point t0;
      ~ActiveTimer() {
        g_prof_active_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      }
    } active_timer{t_iter};
    if (m == 0) {
      g_prof_skip_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_iter)
              .count();
      continue;
    }
    if (e >= gpu_n) {
      if (cpu_batched) continue;   // already computed in the batched host pass above
      if (ENGINE_ENV_ON("ENGINE_DBG_CPULAYER") && n_tokens == 2 && layer >= 17)
        std::fprintf(stderr, "[cpulayer] layer %d e=%d quant_branch=%d host_f32=%zu\n", layer, e,
                     static_cast<int>(!L.h_gate_bytes.empty()), L.host_gate.size());
      if (!L.h_gate_bytes.empty()) {
        // Batched expert streaming: the per-expert synchronous copies in the
        // single-expert prototype cost more than the CPU matvec they replaced.
        // Copy all routed experts of this layer once, then run GPU GEMVs.
        // Streaming the expert through VRAM only paid off while the CPU matvec was a
        // scalar ~3 GMAC/s loop. With the AVX2 kernels (28 GMAC/s) the host path wins
        // big: Coder 189 -> 147 ms/token, 35B 186 -> 81 ms/token.
        static const bool use_arena = ENGINE_ENV_ON("ENGINE_EXPERT_ARENA");
        // CPU expert: quantized bytes stay quantized on the host
        const int inter = expert_inter_;
        const std::size_t g_slab_g =
            static_cast<std::size_t>(inter) * hidden_ / L.h_be_g * L.h_bb_g;
        const std::size_t g_slab_u =
            static_cast<std::size_t>(inter) * hidden_ / L.h_be_u * L.h_bb_u;
        const std::size_t d_slab =
            static_cast<std::size_t>(hidden_) * inter / L.h_be_d * L.h_bb_d;
        const int eo = e;   // host bytes span the whole fused tensor

        const unsigned char* wg = L.h_gate_bytes.data() + eo * g_slab_g;
        const unsigned char* wu = L.h_up_bytes.data() + eo * g_slab_u;
        const unsigned char* wd = L.h_down_bytes.data() + eo * d_slab;
        // Prefer streaming the expert through VRAM and using the GPU GEMV: the scalar
        // CPU matvec runs at ~3 GMAC/s and dominated the 35B's decode time.
        bool streamed = false;
        static int dbg_arena_hits = 0, dbg_cpu_hits = 0;
        if (ENGINE_ENV_ON("ENGINE_DBG_ARENA") && layer == 20)
          std::fprintf(stderr, "[arena] layer %d e=%d m=%d types=%s/%s arena=%d\n", layer, e, m,
                       L.h_type_g.c_str(), L.h_type_u.c_str(), static_cast<int>(use_arena));
        if (use_arena && L.h_type_g == L.h_type_u && m == 1) {
          // one expert per token in decode: copy the three slabs once, compute on GPU
          ensure_expert_scratch();
          cuda::Buffer& arena = xexp_w_;
          const size_t bytes = g_slab_g + g_slab_u + d_slab;
          if (arena.size() < bytes) arena = cuda::Buffer(bytes);
          unsigned char* base = static_cast<unsigned char*>(arena.data());
          arena.upload(wg, g_slab_g);
          arena.upload_at(wu, g_slab_u, base + g_slab_g);
          arena.upload_at(wd, d_slab, base + g_slab_g + g_slab_u);
          const int t = ti[0];
          const float wt = tw[0];
          cuda::Buffer xin(static_cast<std::size_t>(hidden_) * 4);
          xin.upload(hn_host + static_cast<std::size_t>(t) * hidden_, xin.size());
          streamed = gemv_typed(L.h_type_g, base, inter, hidden_, xin, xexp_g_) &&
                     gemv_typed(L.h_type_u, base + g_slab_g, inter, hidden_, xin, xexp_u_);
          if (streamed) {
            long long total = inter;
            void* pg = xexp_g_.data();
            void* pu = xexp_u_.data();
            void* pa = xexp_a_.data();
            void* args[] = {&pg, &pu, &pa, &total};
            cuda::launch(k_silu_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
            streamed = gemv_typed(L.h_type_d, base + g_slab_g + g_slab_u, hidden_, inter, xexp_a_,
                                  xexp_d_);
          }
          if (streamed) {
            std::vector<float> dv(static_cast<std::size_t>(hidden_));
            xexp_d_.download(dv.data(), dv.size() * 4);
            for (int j = 0; j < hidden_; ++j)
              xout_host[static_cast<std::size_t>(t) * hidden_ + j] += wt * dv[j];
          }
        }
        if (streamed) ++dbg_arena_hits; else ++dbg_cpu_hits;
        if (ENGINE_ENV_ON("ENGINE_DBG_ARENA") && layer == 20)
          std::fprintf(stderr, "[arena]   -> %s (arena=%d cpu=%d)\n",
                       streamed ? "GPU" : "CPU", dbg_arena_hits, dbg_cpu_hits);
        if (!streamed) {
          std::vector<float> g(static_cast<std::size_t>(inter)),
              u(static_cast<std::size_t>(inter)), a(static_cast<std::size_t>(inter)),
              dv(static_cast<std::size_t>(hidden_));
          for (int k = 0; k < m; ++k) {
            const int t = ti[k];
            const float wt = tw[k];
            const float* hv = hn_host + static_cast<std::size_t>(t) * hidden_;
            cpu::matvec_mt(L.h_type_g, wg, inter, hidden_, hv, g.data());
            cpu::matvec_mt(L.h_type_u, wu, inter, hidden_, hv, u.data());
            for (int i = 0; i < inter; ++i) a[i] = (g[i] / (1.f + std::exp(-g[i]))) * u[i];
            cpu::matvec_mt(L.h_type_d, wd, hidden_, inter, a.data(), dv.data());
            for (int j = 0; j < hidden_; ++j)
              xout_host[static_cast<std::size_t>(t) * hidden_ + j] += wt * dv[j];
          }
        }
        if (kDebug && layer == 1) {
          std::fprintf(stderr, "[dbg]   cpu e=%d eo=%d wt=%.6f hn0=%.6f\n", e, eo, tw[0],
                       hn_host[0]);
        }
        continue;
      }
      // CPU expert: dequantized-in-host f32 weights, naive matvec
      const auto& Wg = L.host_gate[e];
      const auto& Wu = L.host_up[e];
      const auto& Wd = L.host_down[e];
      const int inter = expert_inter_;
      std::vector<float> g(static_cast<size_t>(inter)), u(static_cast<size_t>(inter)),
          a(static_cast<size_t>(inter)), dv(static_cast<size_t>(hidden_));
      for (int k = 0; k < m; ++k) {
        const int t = ti[k];
        const float wt = tw[k];
        const float* hv = hn_host + static_cast<size_t>(t) * hidden_;
        for (int i = 0; i < inter; ++i) {
          const float* rg = Wg.data() + static_cast<size_t>(i) * hidden_;
          const float* ru = Wu.data() + static_cast<size_t>(i) * hidden_;
          float sg = 0.f, su = 0.f;
          for (int j = 0; j < hidden_; ++j) {
            sg += rg[j] * hv[j];
            su += ru[j] * hv[j];
          }
          g[i] = sg;
          u[i] = su;
        }
        for (int i = 0; i < inter; ++i) a[i] = (g[i] / (1.f + std::exp(-g[i]))) * u[i];
        for (int j = 0; j < hidden_; ++j) {
          const float* rd = Wd.data() + static_cast<size_t>(j) * inter;
          float s = 0.f;
          for (int i = 0; i < inter; ++i) s += rd[i] * a[i];
          xout_host[static_cast<size_t>(t) * hidden_ + j] += wt * s;
        }
      }
      continue;
    }
    if (kDebug) std::fprintf(stderr, "[dbg]   gather\n");
    { 
      const auto pt0 = std::chrono::steady_clock::now();
      int width = hidden_;
      void* ps = hn_.data();
      void* pd = gx_.data();
      void* pi = const_cast<int*>(d_idx_base + exp_off[static_cast<size_t>(e)]);
      void* args[] = {&ps, &pd, &pi, &width};
      cuda::launch(k_gather_, m, 1, 1, 128, 1, 1, 0, args);
      if (ENGINE_ENV_ON("ENGINE_DBG_GEMVTIME")) {
        cuda::sync();
        static int gn = 0;
        if (gn < 6) {
          std::fprintf(stderr, "[k] gather %.3f ms\n",
                       std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - pt0).count());
          ++gn;
        }
      }
    }
    if (!L.q_experts.empty()) {
      matmul(L.q_experts[e * 3 + 0], m, gx_, ggate_);
      matmul(L.q_experts[e * 3 + 1], m, gx_, gup_);
    } else {
      cublas::gemm_f32_nt(m, expert_inter_, hidden_, static_cast<const float*>(gx_.data()),
                          static_cast<const float*>(L.experts[e].gate.data()),
                          static_cast<float*>(ggate_.data()));
      cublas::gemm_f32_nt(m, expert_inter_, hidden_, static_cast<const float*>(gx_.data()),
                          static_cast<const float*>(L.experts[e].up.data()),
                          static_cast<float*>(gup_.data()));
    }
    {
      const auto pt0 = std::chrono::steady_clock::now();
      long long total = static_cast<long long>(m) * expert_inter_;
      void* pg = ggate_.data();
      void* pu = gup_.data();
      void* pa = gact_.data();
      void* args[] = {&pg, &pu, &pa, &total};
      cuda::launch(k_silu_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
      if (ENGINE_ENV_ON("ENGINE_DBG_GEMVTIME")) {
        cuda::sync();
        static int sn = 0;
        if (sn < 6) {
          std::fprintf(stderr, "[k] silu %.3f ms\n",
                       std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - pt0).count());
          ++sn;
        }
      }
    }
    if (!L.q_experts.empty()) {
      matmul(L.q_experts[e * 3 + 2], m, gact_, gdown_);
    } else {
      cublas::gemm_f32_nt(m, hidden_, expert_inter_, static_cast<const float*>(gact_.data()),
                          static_cast<const float*>(L.experts[e].down.data()),
                          static_cast<float*>(gdown_.data()));
    }
    {
      int width = hidden_;
      void* pd = xout_.data();
      void* ps = gdown_.data();
      void* pi = const_cast<int*>(d_idx_base + exp_off[static_cast<size_t>(e)]);
      void* pw = const_cast<float*>(d_w_base + exp_off[static_cast<size_t>(e)]);
      void* args[] = {&pd, &ps, &pi, &pw, &width};
      cuda::launch(k_scatter_, m, 1, 1, 128, 1, 1, 0, args);
    }
    if (kDebug) {
      std::vector<float> gd(static_cast<size_t>(m) * hidden_);
      cuda::stream_sync();
      gdown_.download(gd.data(), gd.size() * 4);
      double s2 = 0;
      for (size_t i = 0; i < gd.size(); ++i) s2 += gd[i];
      std::fprintf(stderr, "[dbg]   gpu e=%d sum=%.6f\n", e, s2);
    }
  }
  if (kDebug) {
    std::vector<float> chk(static_cast<size_t>(n_tokens) * hidden_);
    cuda::stream_sync();
    xout_.download(chk.data(), chk.size() * 4);
    double sg = 0;
    for (size_t i = 0; i < chk.size(); ++i) sg += chk[i];
    double sc = 0;
    for (size_t i = 0; i < static_cast<size_t>(n_tokens) * hidden_; ++i) sc += xout_host[i];
    std::fprintf(stderr, "[dbg]  gpu_part=%.6f cpu_part=%.6f gpu_n=%d\n", sg, sc, gpu_n);
  }
  if (any_cpu && ENGINE_ENV_ON("ENGINE_DUMP_MOE_SPLIT")) {
    std::vector<float> gpu_out(static_cast<size_t>(n_tokens) * hidden_);
    cuda::stream_sync();
    if (const char* dp = ENGINE_ENV_STR("ENGINE_DUMP_MOE_SPLIT")) {
      xout_.download(gpu_out.data(), gpu_out.size() * 4);
      char n1[512], n2[512];
      std::snprintf(n1, sizeof(n1), "%s_L%d_gpu.f32", dp, layer);
      std::snprintf(n2, sizeof(n2), "%s_L%d_cpu.f32", dp, layer);
      std::ofstream f1(n1, std::ios::binary);
      f1.write(reinterpret_cast<const char*>(gpu_out.data()),
               static_cast<std::streamsize>(gpu_out.size() * 4));
      std::ofstream f2(n2, std::ios::binary);
      f2.write(reinterpret_cast<const char*>(xout_host),
               static_cast<std::streamsize>(static_cast<size_t>(n_tokens) * hidden_ * 4));
    }
    xout_.download(gpu_out.data(), gpu_out.size() * 4);
    for (size_t i = 0; i < gpu_out.size(); ++i) gpu_out[i] += xout_host[i];
    xout_.upload(gpu_out.data(), gpu_out.size() * 4);
  }
  // Debug dumps of the intermediate MoE output read xout_ after the host part has been
  // folded in, so that (env-gated) path keeps the historical two-step merge.
  const bool moe_dbg_path = ENGINE_ENV_ON("ENGINE_DUMP") ||
                            ENGINE_ENV_ON("ENGINE_DUMP_MOE_SPLIT");
  bool host_part_folded = false;
  // Zero-copy tail: xout_host is mapped, so add_three_inplace below reads it over PCIe.
  // A pinned H2D copy would add the copy plus a dependent-kernel round trip (~25-35 us
  // per CPU layer measured on this WDDM host), on the critical path before the next
  // layer can start.
  const bool tail_zc = tail_zero_copy_ && !moe_dbg_path;
  if (any_cpu && !tail_zc) {
    // Debug / fallback merge path: one pinned upload plus an add kernel.
    const size_t moe_elems = static_cast<size_t>(n_tokens) * hidden_;
    if (moe_host_part_.size() < moe_elems * 4) moe_host_part_ = cuda::Buffer(moe_elems * 4);
    moe_host_part_.upload_async(xout_host, moe_elems * 4);
    if (moe_dbg_path) {
      void* pa = xout_.data();
      void* pb = moe_host_part_.data();
      long long total = static_cast<long long>(moe_elems);
      void* args[] = {&pa, &pb, &total};
      cuda::launch(k_add_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
      host_part_folded = true;
    }
  }
  if (layer == 1) {
    if (const char* dp = ENGINE_ENV_STR("ENGINE_DUMP")) {
      cuda::stream_sync();
      std::vector<float> h(static_cast<size_t>(n_tokens) * hidden_);
      xout_.download(h.data(), h.size() * 4);
      std::ofstream f1(std::string(dp) + "_moe1.f32", std::ios::binary);
      f1.write(reinterpret_cast<const char*>(h.data()),
               static_cast<std::streamsize>(h.size() * 4));
    }
  }
  {
    long long total = static_cast<long long>(n_tokens) * hidden_;
    if (any_cpu && !host_part_folded) {
      // x += (xout + host_part): one submission instead of two, bit-identical rounding.
      void* pa = x_.data();
      void* pb = xout_.data();
      void* pc = tail_zc ? xout_pin_.device() : moe_host_part_.data();
      void* args[] = {&pa, &pb, &pc, &total};
      cuda::launch(k_add3_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
    } else {
      void* pa = x_.data();
      void* pb = xout_.data();
      void* args[] = {&pa, &pb, &total};
      cuda::launch(k_add_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
    }
  }
  if (ENGINE_ENV_ON("ENGINE_PIPE_PROF")) cuda::sync();
  if (acc_moe) {
    prof_acc::moe_ms += prof_acc::since(acc_moe0);
    prof_acc::moe_tail_ms += prof_acc::since(acc_t_tail0);
  }
  if (kCpuDbg)
    std::fprintf(stderr,
                 "[cpuexperts] layer %d: moe-total %.2f ms (sync %lld calls / %.2f ms)\n", layer,
                 ms_since(t_loop0), cuda::g_sync_calls, cuda::g_sync_ms);
  g_prof_experts_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_loop0)
          .count();
  if (kProf && n_tokens == 2 && (layer < 1 || layer >= 17)) {
    const double shared_launch = ms_since(t_shared0);
    cuda::sync();
    const double shared_sync = ms_since(t_shared0);
    std::fprintf(stderr,
                 "[moe-prof] layer %d: router %.2f, experts %.2f, gap %.2f, total %.2f\n",
                 layer, t_router, t_experts, ms_since(t_shared0), ms_since(t0));
    std::fprintf(stderr, "[moe-prof]   launch %.2f ms, wait-after-sync %.2f ms\n", shared_launch,
                 shared_sync);
  }
  if (ENGINE_ENV_ON("ENGINE_DBG_MOEHASH") && layer == 1) {
    cuda::sync();
    std::vector<float> hh(static_cast<size_t>(n_tokens) * hidden_);
    xout_.download(hh.data(), hh.size() * 4);
    double acc = 0.0;
    for (float vv : hh) acc += static_cast<double>(vv) * vv;
    std::fprintf(stderr, "[moehash] host n=%d xout norm %.6f\n", n_tokens, std::sqrt(acc));
  }

}



// (run_moe ends above)

void DenseRunner::start_clock_warmer() {
  if (!keep_gpu_busy_ || !warmer_kernel_ || warmer_stream_)
    return;
  warmer_stop_ = std::make_unique<cuda::Buffer>(4);
  const int zero = 0;
  warmer_stop_->upload(&zero, 4);
  warmer_stream_ = cuda::create_stream();
  void* pstop = warmer_stop_->data();
  void* args[] = {&pstop};
  cuda::launch(warmer_kernel_, 1, 1, 1, 1, 1, 1, 0, args, warmer_stream_);
}

void DenseRunner::stop_clock_warmer() {
  if (!warmer_stream_) return;
  const int one = 1;
  if (warmer_stop_) warmer_stop_->upload(&one, 4);
  cuda::stream_sync(warmer_stream_);
  cuda::destroy_stream(warmer_stream_);
  warmer_stream_ = nullptr;
}

void DenseRunner::build_half_transposed(QTensor& q) {
  // Measured: the fp16 TN GEMM is correct but ~20x slower here (cuBLAS picks a poor
  // kernel for this layout/leftover load-time cost), so it is opt-in only.
  static const bool on = half_t_optin_;
  if (!on || !q.valid) return;
  size_t free_b = 0, total_b = 0;
  const cuda::MemInfo mi = cuda::mem_info();
  free_b = mi.free_bytes;
  total_b = mi.total_bytes;
  const size_t need = static_cast<size_t>(q.n_out) * q.n_in * 2;
  const size_t f32_need = static_cast<size_t>(q.n_out) * q.n_in * 4;
  if (need + f32_need + (512ull << 20) > free_b) return;   // leave headroom
  // dequantise to f32 first (reuse the slab path with the whole matrix staged)
  cuda::Buffer tmp(f32_need);
  const int nb_row = q.n_in / q.block_elems;
  const long long n_blocks = static_cast<long long>(q.n_out) * nb_row;
  void* fn = nullptr;
  int threads = 256;
  if (q.type == "Q4_K") fn = dq_q4k_;
  else if (q.type == "Q5_K") fn = dq_q5k_;
  else if (q.type == "Q6_K") { fn = dq_q6k_; threads = 32; }
  else if (q.type == "Q8_0") { fn = dq_q8_0_; threads = 32; }
  else if (q.type == "Q5_0") { fn = dq_q5_0_; threads = 32; }
  else if (q.type == "Q5_1") { fn = dq_q5_1_; threads = 32; }
  if (!fn) return;
  void* pw = q.data.data();
  void* po = tmp.data();
  long long nb = n_blocks;
  void* args[] = {&pw, &po, &nb};
  cuda::launch(fn, static_cast<int>(n_blocks), 1, 1, threads, 1, 1, 0, args);
  q.half_t = cuda::Buffer(need);
  void* ps = tmp.data();
  void* pd = q.half_t.data();
  int rows = q.n_out, cols = q.n_in;
  void* targs[] = {&ps, &pd, &rows, &cols};
  cuda::launch(k_transpf16_, (q.n_in + 31) / 32, (q.n_out + 31) / 32, 1, 32, 8, 1, 0, targs);
  q.has_half_t = true;
}

void DenseRunner::build_block_major(QTensor& q) {
  if (!q.valid) return;
  // measurements: Q8_0 (lm_head) pays off, Q5_0/Q6_K do not; Q6_K currently fails
  // its parity test, so it stays out of the default set.
  std::string allowed = bm_types_.empty() ? std::string("Q8_0") : bm_types_;
  if (allowed.find(q.type) == std::string::npos) return;
  if (q.type != "Q8_0" && q.type != "Q5_0" && q.type != "Q6_K") return;
  static const bool off = no_bm_;
  if (off) return;
  const int nb = q.n_in / q.block_elems;
  const size_t bytes = static_cast<size_t>(q.n_out) * nb * q.block_bytes;
  q.bm = cuda::Buffer(bytes);
  void* ps = q.data.data();
  void* pd = q.bm.data();
  int n_out = q.n_out, nbb = nb, bb = q.block_bytes;
  long long total = static_cast<long long>(bytes);
  void* args[] = {&ps, &pd, &n_out, &nbb, &bb};
  (void)total;
  const int blocks = static_cast<int>((bytes / 4 + 255) / 256);
  cuda::launch(k_repackbm_, blocks, 1, 1, 256, 1, 1, 0, args);
  q.has_bm = true;
}

// zero a device buffer with the strided zero_fill kernel (a cuMemsetD32 would work
// too, but every capture-safe primitive here goes through cuda::launch).
void DenseRunner::zero_device_buffer(cuda::Buffer& b, long long n) {
  if (!k_zerofill_ || !b.data() || n <= 0) return;
  void* p = b.data();
  long long count = n;
  void* args[] = {&p, &count};
  cuda::launch(k_zerofill_, 64, 1, 1, 256, 1, 1, 0, args);
}

// The device router writes every slot for n_expert <= 256; for wider expert tables it
// returns early (see runner::device_route_supported) and the host router is used, so a
// stale/uninitialised slot array can no longer feed a garbage expert index into
// gptrs[e] or an out-of-bounds atomicAdd.  Clearing the slots once (and after any
// reallocation) is defence in depth for paths that still read them.
void DenseRunner::ensure_route_slots_zeroed(int nact) {
  if (route_slots_zeroed_ || nact <= 0) return;
  zero_device_buffer(slot_exp_, nact);
  zero_device_buffer(slot_tok_, nact);
  zero_device_buffer(slot_w_, nact);
  route_slots_zeroed_ = true;
}

bool DenseRunner::load_qtensor(const std::string& name, QTensor& out) {
  if (container_ != Container::Gguf) return false;
  auto g = std::static_pointer_cast<gguf::Model>(gguf_);
  const gguf::TensorInfo* ti = nullptr;
  for (const auto& t : g->tensors)
    if (t.name == name) ti = &t;
  if (!ti)
    throw std::runtime_error("missing tensor " + name +
                           " (GGUF naming does not match blk.N.*)");
  const QuantKernel* qk = quant_kernel(ti->type_name);
  if (!qk) return false;
  std::ifstream f(g->path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + g->path);
  f.seekg(static_cast<std::streamoff>(g->data_offset + ti->offset));
  std::vector<unsigned char> raw(static_cast<size_t>(ti->nbytes));
  read_exact(f, raw.data(), raw.size(), name);
  out.data = cuda::Buffer(raw.size());
  out.data.upload(raw.data(), raw.size());
  out.type = ti->type_name;
  out.block_bytes = qk->block_bytes;
  out.block_elems = qk->block_elems;
  out.n_out = static_cast<int>(ti->shape[0]);
  out.n_in = static_cast<int>(ti->shape[1]);
  out.valid = true;
  // the f32 staging buffer only needs a slab; cap it at 32 MiB worth of floats
  const size_t floats = static_cast<size_t>(out.n_out) * out.n_in;
  const size_t cap = 8u << 20;
  if (std::min(floats, cap) > scratch_floats_) scratch_floats_ = std::min(floats, cap);
  build_block_major(out);
  build_half_transposed(out);
  return true;
}

const std::vector<float>& DenseRunner::lm_head_f32() {
  if (!lm_head_host_.empty()) return lm_head_host_;
  if (container_ == Container::Gguf) {
    auto g = std::static_pointer_cast<gguf::Model>(gguf_);
    std::string name = "token_embd.weight";
    for (const auto& t : g->tensors)
      if (t.name == "output.weight") name = t.name;
    lm_head_host_ = tensor_f32(name);
  } else {
    lm_head_host_ = tensor_f32("lm_head.weight");
  }
  return lm_head_host_;
}

std::vector<unsigned char> DenseRunner::read_gguf_raw(const std::string& name) {
  auto g = std::static_pointer_cast<gguf::Model>(gguf_);
  const gguf::TensorInfo* ti = nullptr;
  for (const auto& t : g->tensors)
    if (t.name == name) ti = &t;
  if (!ti)
    throw std::runtime_error("missing tensor " + name +
                           " (GGUF naming does not match blk.N.*)");
  std::ifstream f(g->path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + g->path);
  f.seekg(static_cast<std::streamoff>(g->data_offset + ti->offset));
  std::vector<unsigned char> raw(static_cast<size_t>(ti->nbytes));
  read_exact(f, raw.data(), raw.size(), name);
  return raw;
}

bool DenseRunner::load_qtensor_span(const std::string& name, size_t byte_offset,
                                    size_t nbytes, int n_out, int n_in, QTensor& out) {
  if (container_ != Container::Gguf) return false;
  auto g = std::static_pointer_cast<gguf::Model>(gguf_);
  const gguf::TensorInfo* ti = nullptr;
  for (const auto& t : g->tensors)
    if (t.name == name) ti = &t;
  if (!ti)
    throw std::runtime_error("missing tensor " + name +
                           " (GGUF naming does not match blk.N.*)");
  const QuantKernel* qk = quant_kernel(ti->type_name);
  if (!qk) return false;
  std::ifstream f(g->path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + g->path);
  f.seekg(static_cast<std::streamoff>(g->data_offset + ti->offset + byte_offset));
  std::vector<unsigned char> raw(nbytes);
  read_exact(f, raw.data(), nbytes, name);
  out.data = cuda::Buffer(nbytes);
  out.data.upload(raw.data(), nbytes);
  out.type = ti->type_name;
  out.block_bytes = qk->block_bytes;
  out.block_elems = qk->block_elems;
  out.n_out = n_out;
  out.n_in = n_in;
  out.valid = true;
  const size_t floats = static_cast<size_t>(n_out) * n_in;
  if (floats > scratch_floats_) scratch_floats_ = floats;
  return true;
}

void DenseRunner::matmul(const QTensor& w, int n_tokens, cuda::Buffer& in, cuda::Buffer& out) {
  if (!w.valid) throw std::runtime_error("matmul: invalid quantized tensor");
  static const bool probe_env = ENGINE_ENV_ON("ENGINE_DBG_GEMVTIME");
  const bool probe_on = probe_env && n_tokens == 1;
  static int probe_n = 0;
  const auto probe_t0 = std::chrono::steady_clock::now();
  struct ProbeEnd {
    const QTensor* w;
    std::chrono::steady_clock::time_point t0;
    bool on;
    ~ProbeEnd() {
      if (!on || probe_n >= 400) return;
      cuda::sync();
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr, "[gemv] %-5s %5dx%-5d %6.2f MB %7.3f ms %6.1f GB/s\n", w->type.c_str(),
                   w->n_out, w->n_in, static_cast<double>(w->data.size()) / 1e6, ms,
                   ms > 0 ? (static_cast<double>(w->data.size()) / 1e9) / (ms / 1000.0) : 0.0);
      ++probe_n;
    }
  } probe_end{&w, probe_t0, probe_on};
  if (n_tokens == 1 && w.type != "Q5_1" && !no_gemv_) {
    // Everything about "which kernel for this shape" is decided by choose_gemv()
    // (src/kernel_select.cpp): the threshold constants and optional calibrated
    // overrides live there, this function only launches the chosen kernel.
    const GemvParams gp = gemv_params(w);
    const GemvChoice c = choose_gemv(w.type, w.n_out, w.n_in, gp);
    if (c.plan == GemvPlan::kNone) {
      throw std::runtime_error(
          "no specialized GEMV kernel for quant type '" + w.type + "' (shape " +
          std::to_string(w.n_out) + "x" + std::to_string(w.n_in) +
          "); supported GEMV types: Q4_K, Q5_K, Q6_K, Q5_0, Q8_0, Q5_1");
    }
    if (debug_) {
      static std::set<std::string> logged;
      const std::string key = gemv_shape_key(w.type, w.n_out, w.n_in);
      if (logged.insert(key).second)
        std::fprintf(stderr, "[gemv] %-5s %6dx%-6d -> %-16s split=%d (%s)\n", w.type.c_str(),
                     w.n_out, w.n_in, gemv_plan_name(c.plan), c.split, c.reason);
    }
    run_gemv_plan(c, w, in, out);
    return;
  }
  // ---- prefill GEMM through fp16 tensor cores -------------------------------
  // The f32 cuBLAS path below runs at ~4 T-MAC/s on this GPU (63% of the f32 FMA
  // peak) while the very same GEMM on fp16 tensor cores runs at 11-13 T-MAC/s.
  // The weight slab is therefore dequantised *directly* into fp16 (no f32 staging
  // buffer: half the store traffic) and fed to cuBLAS with f32 accumulation, which
  // keeps the output within ~3e-4 relative of the exact f32 path.
  // Diagnostic: skip the GEMM body entirely (garbage output, used to attribute the
  // prefill wall clock between the tensor-core GEMMs and everything else).
  static const bool skip_gemm_dbg = ENGINE_ENV_ON("ENGINE_SKIP_GEMM");
  if (skip_gemm_dbg && n_tokens > 1) return;
  static const bool want_f16 = prefill_f16_;
  const bool want_acc16 = want_f16 && prefill_f16_acc_;
  if (want_f16 && n_tokens > 1) {
    // x8 variants: one warp per super-block, 8 values/lane and a 16 B store.  Measured
    // ~305 GB/s vs 181-207 GB/s for the per-block kernels, with a bit-identical fp16
    // slab (so the generated tokens cannot change).  ENGINE_DQ_OLD=1 forces the old
    // kernels for A/B.
    static const bool dq_old = ENGINE_ENV_ON("ENGINE_DQ_OLD");
    void* f16fn = nullptr;
    void* f16x8 = nullptr;
    int f16_threads = 128;
    if (w.type == "Q4_K") { f16fn = dq_q4k_f16_; f16x8 = dq_q4k_f16_x8_; }
    else if (w.type == "Q5_K") { f16fn = dq_q5k_f16_; f16x8 = dq_q5k_f16_x8_; }
    else if (w.type == "Q6_K") { f16fn = dq_q6k_f16_; f16x8 = dq_q6k_f16_x8_; }
    else if (w.type == "Q8_0") { f16fn = dq_q8_0_f16_; f16_threads = 32; }
    if (dq_old) f16x8 = nullptr;
    const long long macs = static_cast<long long>(n_tokens) * w.n_out * w.n_in;
    if (f16fn && macs >= prefill_f16_min_macs_ && scratch_floats_ > 0 &&
        dequant_scratch_.data() != nullptr) {
      // The scratch was sized in *f32* elements for the largest quantized tensor,
      // i.e. twice the fp16 element capacity.
      if (w.n_in > x_half_max_nin_) x_half_max_nin_ = w.n_in;
      const int nb_row = w.n_in / w.block_elems;
      int slab_rows = static_cast<int>((scratch_floats_ * 2) / static_cast<size_t>(w.n_in));
      if (slab_rows < 1) slab_rows = 1;
      if (slab_rows > w.n_out) slab_rows = w.n_out;
      const long long n_x = static_cast<long long>(n_tokens) * w.n_in;
      if (x_half_.size() < static_cast<size_t>(n_x) * 2)
        x_half_ = cuda::Buffer(static_cast<size_t>(n_x) * 2);
      {
        void* ps = in.data();
        void* pd = x_half_.data();
        long long n = n_x;
        void* a[] = {&ps, &pd, &n};
        cuda::launch(k_f32tohalf_, static_cast<int>((n + 255) / 256), 1, 1, 256, 1, 1, 0, a);
      }
      for (int r0 = 0; r0 < w.n_out; r0 += slab_rows) {
        const int rows = std::min(slab_rows, w.n_out - r0);
        const long long blk0 = static_cast<long long>(r0) * nb_row;
        const long long nblk = static_cast<long long>(rows) * nb_row;
        void* ps = static_cast<unsigned char*>(w.data.data()) + blk0 * w.block_bytes;
        void* po = dequant_scratch_.data();
        long long nb = nblk;
        void* a[] = {&ps, &po, &nb};
        if (f16x8)
          cuda::launch(f16x8, static_cast<int>((nblk + 7) / 8), 1, 1, 256, 1, 1, 0, a);
        else
          cuda::launch(f16fn, static_cast<int>(nblk), 1, 1, f16_threads, 1, 1, 0, a);
        if (want_acc16 && k_f16tof32_) {
          // f16 accumulate: write an fp16 tile then widen it into the f32 activation
          // buffer (the tile is contiguous, the destination is strided by n_out).
          const size_t need = static_cast<size_t>(n_tokens) * rows * 2;
          if (y16_scratch_.size() < need) y16_scratch_ = cuda::Buffer(need);
          cublas::gemm_f16_nt_ld_c16(n_tokens, rows, w.n_in, x_half_.data(), po,
                                     y16_scratch_.data(), rows);
          long long nout = static_cast<long long>(n_tokens) * rows;
          void* p16 = y16_scratch_.data();
          void* pf = static_cast<float*>(out.data()) + r0;
          int rr = rows, os = w.n_out;
          void* a16[] = {&p16, &pf, &rr, &nout, &os};
          cuda::launch(k_f16tof32_, static_cast<int>((nout + 255) / 256), 1, 1, 256, 1, 1, 0,
                       a16);
        } else {
          cublas::gemm_f16_nt_ld(n_tokens, rows, w.n_in, x_half_.data(), po,
                                 static_cast<float*>(out.data()) + r0, w.n_out);
        }
      }
      return;
    }
  }
  // Self-developed quantized GEMM (G1): opt-in, used for the prefill matmuls.
  static const bool use_quant_gemm = use_quant_gemm_;
  if (use_quant_gemm && n_tokens > 1 && k_quantgemm_ &&
      (w.type == "Q8_0" || w.type == "Q4_K" || w.type == "Q5_K")) {
    const long long row_bytes = w.data.size() / w.n_out;
    void* pa = in.data();
    void* pw = w.data.data();
    void* pc = out.data();
    int m = n_tokens, nn = w.n_out, kk = w.n_in, ty = w.type == "Q8_0" ? 3 : (w.type == "Q4_K" ? 0 : 1);
    long long rb = row_bytes;
    void* args[] = {&pa, &pw, &pc, &m, &nn, &kk, &ty, &rb};
    cuda::launch(k_quantgemm_, (w.n_out + 31) / 32, (n_tokens + 31) / 32, 1, 256, 1, 1, 0, args);
    return;
  }
  if (n_tokens > 1 && w.has_half_t) {
    // fp16 TN layout: C[m,n] = X[m,k] @ W[k,n], tensor cores, f32 accumulate
    const long long n_x = static_cast<long long>(n_tokens) * w.n_in;
    if (x_half_.size() < static_cast<size_t>(n_x) * 2)
      x_half_ = cuda::Buffer(static_cast<size_t>(n_x) * 2);
    void* ps = in.data();
    void* pd = x_half_.data();
    long long n = n_x;
    void* a[] = {&ps, &pd, &n};
    cuda::launch(k_f32tohalf_, static_cast<int>((n + 255) / 256), 1, 1, 256, 1, 1, 0, a);
    cublas::gemm_f16_tn(n_tokens, w.n_out, w.n_in, x_half_.data(), w.half_t.data(),
                        static_cast<float*>(out.data()));
    return;
  }
  if (scratch_floats_ == 0 || dequant_scratch_.data() == nullptr)
    throw std::runtime_error("matmul: no dequant scratch");
  (void)0;
  void* fn = nullptr;
  int threads = 256;
  if (w.type == "Q4_K") fn = dq_q4k_;
  else if (w.type == "Q5_K") fn = dq_q5k_;
  else if (w.type == "Q6_K") { fn = dq_q6k_; threads = 32; }
  else if (w.type == "Q8_0") { fn = dq_q8_0_; threads = 32; }
  else if (w.type == "Q5_0") { fn = dq_q5_0_; threads = 32; }
  else if (w.type == "Q5_1") { fn = dq_q5_1_; threads = 32; }
  if (!fn) throw std::runtime_error("no dequant for " + w.type);
  // Dequantise in row slabs so the f32 staging buffer stays small: a full-matrix
  // staging copy writes+reads 4 bytes per weight (9x the quantised traffic) and does
  // not fit in VRAM for 6 GB models.
  // fp16 staging + tensor cores are ~3x faster than f32 cuBLAS for big models that
  // cannot hold an f32 weight mirror (prefill is then compute bound).
  // Measured: an fp16 staging copy + tensor-core GEMM gives only +9% at pp192 and
  // is slower at pp64 (cuBLAS's fp16 path does not reach peak for our NT layout and
  // the dequant pass dominates), so it stays opt-in.
  static const bool want_half = half_optin_;
  const bool use_half = want_half && n_tokens > 1;
  const int nb_row = w.n_in / w.block_elems;
  int slab_rows = static_cast<int>(scratch_floats_ / static_cast<size_t>(w.n_in));
  if (slab_rows < 1) slab_rows = 1;
  if (slab_rows > w.n_out) slab_rows = w.n_out;
  for (int r0 = 0; r0 < w.n_out; r0 += slab_rows) {
    const int rows = std::min(slab_rows, w.n_out - r0);
    const long long blk0 = static_cast<long long>(r0) * nb_row;
    const long long nblk = static_cast<long long>(rows) * nb_row;
    void* ps = static_cast<unsigned char*>(w.data.data()) + blk0 * w.block_bytes;
    void* po = dequant_scratch_.data();
    long long nb = nblk;
    void* args[] = {&ps, &po, &nb};
    cuda::launch(fn, static_cast<int>(nblk), 1, 1, threads, 1, 1, 0, args);
    if (use_half) {
      const long long n_half = static_cast<long long>(rows) * w.n_in;
      const long long n_in_half = static_cast<long long>(n_tokens) * w.n_in;
      if (slab_half_.size() < static_cast<size_t>(n_half) * 2)
        slab_half_ = cuda::Buffer(static_cast<size_t>(n_half) * 2);
      if (x_half_.size() < static_cast<size_t>(n_in_half) * 2)
        x_half_ = cuda::Buffer(static_cast<size_t>(n_in_half) * 2);
      {
        void* ps = dequant_scratch_.data();
        void* pd = slab_half_.data();
        long long n = n_half;
        void* a[] = {&ps, &pd, &n};
        cuda::launch(k_f32tohalf_, static_cast<int>((n + 255) / 256), 1, 1, 256, 1, 1, 0, a);
      }
      {
        void* ps = in.data();
        void* pd = x_half_.data();
        long long n = n_in_half;
        void* a[] = {&ps, &pd, &n};
        cuda::launch(k_f32tohalf_, static_cast<int>((n + 255) / 256), 1, 1, 256, 1, 1, 0, a);
      }
      cublas::gemm_f16_nt_ld(n_tokens, rows, w.n_in, x_half_.data(), slab_half_.data(),
                             static_cast<float*>(out.data()) + r0, w.n_out);
    } else {
      cublas::gemm_f32_nt_ld(n_tokens, rows, w.n_in, static_cast<const float*>(in.data()),
                             static_cast<const float*>(dequant_scratch_.data()),
                             static_cast<float*>(out.data()) + r0, w.n_out);
    }
  }
}

// ---------------------------------------------------------------------------
// GEMV dispatch: policy results (GemvChoice) -> kernel launches
// ---------------------------------------------------------------------------
GemvParams DenseRunner::gemv_params(const QTensor& w) const {
  GemvParams p;
  p.dp4a_gemv = dp4a_gemv_;
  p.dp4a_q6k = dp4a_q6k_;
  p.tiny_sk = tiny_sk_;
  p.splitk = splitk_;
  p.dp4a_optin = dp4a_optin_;
  p.has_bm = w.has_bm;
  p.row_threshold = row_threshold_;
  p.q6k_x4_min = q6k_x4_min_;
  p.tiny_sk_max_nout = tiny_sk_max_nout_;
  p.tiny_sk_min_nin = tiny_sk_min_nin_;
  p.q6k_dp4a_split_max_nout = q6k_dp4a_split_max_nout_;
  p.small_nout_split_max = small_nout_split_max_;
  p.q6k_split = q6k_split_cfg_;
  p.dp4a_scratch_bytes = qx_i8_.size();   // qx holds two int8 levels
  if (has_kernel_overrides_)
    p.override_plan = engine::config::kernel_override(gemv_shape_key(w.type, w.n_out, w.n_in));
  return p;
}

void DenseRunner::run_gemv_plan(const GemvChoice& c, const QTensor& w, cuda::Buffer& in,
                                cuda::Buffer& out) {
  void* pw = w.data.data();
  void* px = in.data();
  void* py = out.data();
  int nin = w.n_in, nout = w.n_out;
  void* args[] = {&pw, &px, &py, &nin, &nout};
  switch (c.plan) {
    case GemvPlan::kNone:
      break;
    case GemvPlan::Q4K:
      cuda::launch(gemv_q4k_, (w.n_out + 7) / 8, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q4K_X4:
      cuda::launch(gemv_q4k_x4_, (w.n_out + 31) / 32, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q4K_K2:
      cuda::launch(gemv_q4k_k2_, (w.n_out + 3) / 4, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q5K:
      cuda::launch(gemv_q5k_, (w.n_out + 7) / 8, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q5K_X4:
      cuda::launch(gemv_q5k_x4_, (w.n_out + 31) / 32, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q5K_SK:
    case GemvPlan::Q6K_SK: {
      void* pz = out.data();
      long long nz = w.n_out;
      void* zargs[] = {&pz, &nz};
      cuda::launch(k_zerofill_, 64, 1, 1, 256, 1, 1, 0, zargs);
      int sp = c.split;
      void* sargs[] = {&pw, &px, &py, &nin, &nout, &sp};
      void* fn = c.plan == GemvPlan::Q5K_SK ? gemv_q5k_sk_ : gemv_q6k_sk_;
      cuda::launch(fn, (w.n_out + 31) / 32, 1, 1, 256, 1, 1, 0, sargs);
      return;
    }
    case GemvPlan::Q5K_DP4A:
    case GemvPlan::Q5K_DP4A_S:
    case GemvPlan::Q6K_DP4A:
    case GemvPlan::Q6K_DP4A_SK: {
      void* pxf = in.data();
      void* pq = qx_i8_.data();
      void* pdx = dx_i8_.data();
      void* pxs8 = xs8_i8_.data();
      int nni = w.n_in;
      // gate/up, qkv/z and q/k/v feed the *same* activation to consecutive dp4a
      // GEMVs; reuse the staged int8 copy when no kernel has been launched since
      // (which also proves the buffer cannot have been overwritten).
      const bool reuse = quant_reuse_ && qc_ptr_ == pxf && qc_nin_ == nni &&
                         qc_stamp_ == cuda::g_launch_calls &&
                         qc_graph_ == cuda::g_graph_launches;
      if (!reuse) {
        void* qargs[] = {&pxf, &pq, &pdx, &pxs8, &nni};
        cuda::launch(k_quantx_i8_, (w.n_in + 31) / 32, 1, 1, 32, 1, 1, 0, qargs);
      }
      if (c.plan == GemvPlan::Q5K_DP4A_S) {
        void* sargs[] = {&pw, &pq, &pdx, &pxs8, &py, &nin, &nout};
        const size_t shx = 3 * static_cast<size_t>(w.n_in);
        cuda::launch(gemv_q5k_dp4a_s_, (w.n_out + 31) / 32, 1, 1, 256, 1, 1, shx, sargs);
      } else if (c.plan == GemvPlan::Q5K_DP4A) {
        void* sargs[] = {&pw, &pq, &pdx, &pxs8, &py, &nin, &nout};
        cuda::launch(gemv_q5k_dp4a_, (w.n_out + 31) / 32, 1, 1, 256, 1, 1, 0, sargs);
      } else if (c.plan == GemvPlan::Q6K_DP4A) {
        void* sargs[] = {&pw, &pq, &pdx, &py, &nin, &nout};
        cuda::launch(gemv_q6k_dp4a_, (w.n_out + 7) / 8, 1, 1, 256, 1, 1, 0, sargs);
      } else {
        int sp = c.split;
        const int nr = 8 / c.split;
        void* kargs[] = {&pw, &pq, &pdx, &py, &nin, &nout, &sp};
        cuda::launch(gemv_q6k_dp4a_sk_, (w.n_out + nr - 1) / nr, 1, 1, 256, 1, 1, 0, kargs);
      }
      if (quant_reuse_) {
        qc_ptr_ = pxf;
        qc_nin_ = nni;
        qc_stamp_ = cuda::g_launch_calls;
        qc_graph_ = cuda::g_graph_launches;
      }
      return;
    }
    case GemvPlan::Q6K:
      cuda::launch(gemv_q6k_, (w.n_out + 7) / 8, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q6K_X4:
      cuda::launch(gemv_q6k_x4_, (w.n_out + 31) / 32, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q6K_BM: {
      void* pwb = w.bm.data();
      void* dargs[] = {&pwb, &px, &py, &nin, &nout};
      cuda::launch(k_gemvq6kbm_, (w.n_out + 255) / 256, 1, 1, 256, 1, 1, 0, dargs);
      return;
    }
    case GemvPlan::Q50:
      cuda::launch(gemv_q5_0_, (w.n_out + 7) / 8, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q50_X4:
      cuda::launch(gemv_q5_0_x4_, (w.n_out + 31) / 32, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q50_K2:
      cuda::launch(gemv_q5_0_k2_, (w.n_out + 3) / 4, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q50_P4:
      cuda::launch(gemv_q5_0_p4_, (w.n_out + 7) / 8, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q50_BM: {
      void* pwb = w.bm.data();
      void* dargs[] = {&pwb, &px, &py, &nin, &nout};
      cuda::launch(k_gemvq50bm_, (w.n_out + 255) / 256, 1, 1, 256, 1, 1, 0, dargs);
      return;
    }
    case GemvPlan::Q80:
      cuda::launch(gemv_q8_0_, (w.n_out + 7) / 8, 1, 1, 256, 1, 1, 0, args);
      return;
    case GemvPlan::Q80_BM: {
      void* pwb = w.bm.data();
      void* dargs[] = {&pwb, &px, &py, &nin, &nout};
      cuda::launch(k_gemvq80bm_, (w.n_out + 255) / 256, 1, 1, 256, 1, 1, 0, dargs);
      return;
    }
    case GemvPlan::Q80_TINY_SK: {
      void* pz = out.data();
      long long nz = w.n_out;
      void* zargs[] = {&pz, &nz};
      cuda::launch(k_zerofill_, 8, 1, 1, 128, 1, 1, 0, zargs);
      int ksplit = c.split;
      void* sargs[] = {&pw, &px, &py, &nin, &nout, &ksplit};
      cuda::launch(gemv_q8_sk_, (w.n_out + 7) / 8, ksplit, 1, 256, 1, 1, 0, sargs);
      return;
    }
    case GemvPlan::Q50_DP4A:
    case GemvPlan::Q80_DP4A: {
      const int nb = w.n_in / 32;
      if (!qx_.data() || qx_bytes_ < static_cast<size_t>(w.n_in) + 1) {
        qx_ = cuda::Buffer(static_cast<size_t>(w.n_in) * 4 + 4096);
        dx_ = cuda::Buffer(static_cast<size_t>(nb) * 4 + 256);
        qx_bytes_ = static_cast<size_t>(w.n_in) + 1;
      }
      void* pxf = in.data();
      void* pq = qx_.data();
      void* pd = dx_.data();
      int nblocks = nb;
      void* qargs[] = {&pxf, &pq, &pd, &nblocks};
      cuda::launch(k_quantx_, (nb + 255) / 256, 1, 1, 256, 1, 1, 0, qargs);
      void* dargs[] = {&pw, &pq, &pd, &py, &nin, &nout};
      cuda::launch(c.plan == GemvPlan::Q80_DP4A ? k_gemvq80dp4a_ : k_gemvq50dp4a_,
                   (w.n_out + 7) / 8, 1, 1, 256, 1, 1, static_cast<size_t>(w.n_in), dargs);
      return;
    }
  }
  throw std::runtime_error(std::string("unhandled GEMV plan: ") + gemv_plan_name(c.plan));
}

// ---------------------------------------------------------------------------
// --kernel-calibrate: measure candidate kernels on this model's real shapes
// ---------------------------------------------------------------------------
int DenseRunner::calibrate_kernels(bool verbose) {
  struct ShapeRef {
    std::string type;
    int n_out = 0, n_in = 0;
    const QTensor* tensor = nullptr;
  };
  std::vector<ShapeRef> shapes;
  std::set<std::string> seen;
  auto add = [&](const QTensor& t) {
    if (!t.valid || t.n_out <= 0 || t.n_in <= 0) return;
    const std::string key = gemv_shape_key(t.type, t.n_out, t.n_in);
    if (!seen.insert(key).second) return;
    ShapeRef s;
    s.type = t.type;
    s.n_out = t.n_out;
    s.n_in = t.n_in;
    s.tensor = &t;
    shapes.push_back(s);
  };
  add(lm_head_qt_);
  for (auto& L : layers_) {
    add(L.qq); add(L.qk); add(L.qv); add(L.qo); add(L.qg); add(L.qu); add(L.qd);
    add(L.qwq_mla); add(L.qwkva); add(L.qwkb); add(L.qwo_mla);
    add(L.qsh_gate); add(L.qsh_up); add(L.qsh_down);
    add(L.q35.qkv); add(L.q35.z); add(L.q35.alpha); add(L.q35.beta); add(L.q35.out);
    add(L.q35.qp); add(L.q35.kp); add(L.q35.vp); add(L.q35.op);
    // a couple of experts are enough: all experts of a layer share the shape
    for (size_t i = 0; i < L.q_experts.size() && i < 3; ++i) add(L.q_experts[i]);
  }
  if (shapes.empty()) {
    std::fprintf(stderr, "[calib] no quantized GEMV shapes found in this model\n");
    return 0;
  }
  const int iters = static_cast<int>(engine::config::get_int("calib_iters"));
  auto candidates = [&](const ShapeRef& s, const GemvParams& base,
                        std::vector<GemvChoice>* out) {
    auto push = [&](GemvPlan plan, int split) {
      GemvChoice c;
      c.plan = plan;
      c.split = split;
      if (gemv_plan_compatible(plan, split, s.type, s.n_out, s.n_in, base)) out->push_back(c);
    };
    if (s.type == "Q4_K") { push(GemvPlan::Q4K, 1); push(GemvPlan::Q4K_X4, 1); push(GemvPlan::Q4K_K2, 1); }
    if (s.type == "Q5_K") {
      push(GemvPlan::Q5K, 1); push(GemvPlan::Q5K_X4, 1);
      push(GemvPlan::Q5K_DP4A, 1); push(GemvPlan::Q5K_DP4A_S, 1);
    }
    if (s.type == "Q6_K") {
      push(GemvPlan::Q6K, 1); push(GemvPlan::Q6K_X4, 1); push(GemvPlan::Q6K_DP4A, 1);
      push(GemvPlan::Q6K_DP4A_SK, 2); push(GemvPlan::Q6K_DP4A_SK, 4);
      push(GemvPlan::Q6K_DP4A_SK, 8); push(GemvPlan::Q6K_BM, 1);
    }
    if (s.type == "Q5_0") { push(GemvPlan::Q50, 1); push(GemvPlan::Q50_X4, 1); push(GemvPlan::Q50_K2, 1); push(GemvPlan::Q50_P4, 1); }
    if (s.type == "Q8_0") { push(GemvPlan::Q80, 1); push(GemvPlan::Q80_BM, 1); push(GemvPlan::Q80_TINY_SK, 2); push(GemvPlan::Q80_TINY_SK, 4); push(GemvPlan::Q80_TINY_SK, 8); }
  };

  int written = 0;
  for (const auto& s : shapes) {
    const QTensor& w = *s.tensor;
    GemvParams base = gemv_params(w);
    base.override_plan = nullptr;
    std::vector<GemvChoice> cands;
    candidates(s, base, &cands);
    if (cands.empty()) continue;
    std::vector<float> hx(static_cast<size_t>(s.n_in), 0.01f);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = static_cast<float>((i % 17) - 8) * 0.013f;
    cuda::Buffer x(static_cast<size_t>(s.n_in) * 4);
    x.upload(hx.data(), hx.size() * 4);
    cuda::Buffer y(static_cast<size_t>(s.n_out) * 4);
    GemvChoice best;
    double best_ms = 1e30;
    if (verbose)
      std::fprintf(stderr, "[calib] %s (%zu candidates)\n",
                   gemv_shape_key(s.type, s.n_out, s.n_in).c_str(), cands.size());
    for (auto& c : cands) {
      // warm-up then timed launches (sync per iteration = per-launch latency)
      for (int i = 0; i < 3; ++i) run_gemv_plan(c, w, x, y);
      cuda::sync();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) run_gemv_plan(c, w, x, y);
      cuda::sync();
      const double total_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      const double ms = total_ms / iters;
      const double gbs = ms > 0 ? (static_cast<double>(w.data.size()) / 1e9) / (ms / 1000.0) : 0.0;
      if (verbose)
        std::fprintf(stderr, "  %-18s split=%d %8.4f ms %7.1f GB/s%s\n",
                     gemv_plan_name(c.plan), c.split, ms, gbs, ms < best_ms ? "  *" : "");
      if (ms < best_ms) {
        best_ms = ms;
        best = c;
      }
    }
    const GemvChoice heuristic = choose_gemv(s.type, s.n_out, s.n_in, base);
    // Prefer the heuristic unless the calibrated winner is clearly faster; this
    // keeps the cache stable against timing noise on small shapes.
    GemvChoice chosen = best;
    if (heuristic.plan == best.plan && heuristic.split == best.split) chosen = best;
    else {
      const double noise = 0.04;  // 4%
      // re-run heuristic once for a fair comparison
      for (int i = 0; i < 3; ++i) run_gemv_plan(heuristic, w, x, y);
      cuda::sync();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) run_gemv_plan(heuristic, w, x, y);
      cuda::sync();
      const double h_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / iters;
      if (h_ms <= best_ms * (1.0 + noise)) chosen = heuristic;
      if (verbose)
        std::fprintf(stderr, "  heuristic %-18s split=%d %8.4f ms -> chosen %s%s\n",
                     gemv_plan_name(heuristic.plan), heuristic.split, h_ms,
                     gemv_plan_name(chosen.plan),
                     (chosen.plan == heuristic.plan && chosen.split == heuristic.split)
                         ? " (heuristic)"
                         : " (calibrated)");
    }
    std::string plan_name = gemv_plan_name(chosen.plan);
    if (chosen.split > 1 &&
        (chosen.plan == GemvPlan::Q6K_DP4A_SK || chosen.plan == GemvPlan::Q5K_SK ||
         chosen.plan == GemvPlan::Q6K_SK || chosen.plan == GemvPlan::Q80_TINY_SK))
      plan_name += std::to_string(chosen.split);
    engine::config::Config::get().set_kernel_override(
        gemv_shape_key(s.type, s.n_out, s.n_in), plan_name);
    ++written;
  }
  const std::string cache = engine::config::get_string("kernel_cache");
  std::string err;
  if (!engine::config::Config::get().save_kernel_cache(cache, &err)) {
    std::fprintf(stderr, "[calib] cannot save kernel cache: %s\n", err.c_str());
    return 0;
  }
  std::fprintf(stderr, "[calib] wrote %d shape decisions to %s (iters=%d)\n", written,
               cache.c_str(), iters);
  return written;
}

// ---------------------------------------------------------------------------
// Adaptive VRAM placement
// ---------------------------------------------------------------------------
void DenseRunner::load_config_snapshot() {
  auto& cfg = engine::config::Config::get();
  cfg.ensure_initialized();
  debug_ = cfg.get_bool("debug");
  no_gemv_ = cfg.get_bool("no_gemv");
  dp4a_gemv_ = cfg.get_bool("dp4a_gemv");
  dp4a_q6k_ = cfg.get_bool("dp4a_q6k");
  tiny_sk_ = cfg.get_bool("tiny_sk");
  gd_delta_128_ = cfg.get_bool("gd_delta_128");
  ssm_fuse_ = cfg.get_bool("ssm_fuse");
  quant_reuse_ = cfg.get_bool("quant_reuse");
  splitk_ = cfg.get_bool("splitk");
  dp4a_optin_ = cfg.get_bool("dp4a");
  use_quant_gemm_ = cfg.get_bool("gemm");
  half_optin_ = cfg.get_bool("half");
  half_t_optin_ = cfg.get_bool("half_t");
  prefill_f16_ = cfg.get_bool("prefill_f16");
  prefill_f16_min_macs_ = cfg.get_int("prefill_f16_min_macs");
  attn_pre_rows_ = static_cast<int>(cfg.get_int("attn_pre_rows"));
  attn_gqa_ = cfg.get_bool("attn_gqa");
  attn_gqa_decode_ = cfg.get_bool("attn_gqa_decode");
  attn_gqa_wide_ = cfg.get_bool("attn_gqa_wide");
  attn_gqa_min_ctx_ = static_cast<int>(cfg.get_int("attn_gqa_min_ctx"));
  if (attn_gqa_min_ctx_ < 0) attn_gqa_min_ctx_ = 0;
  prefill_f16_acc_ = cfg.get_bool("prefill_f16_acc");
  prefill_graph_ = cfg.get_bool("prefill_graph");
  prefill_logits_rows_ = static_cast<int>(cfg.get_int("prefill_logits_rows"));
  layer_graph_ = cfg.get_bool("layer_graph");
  mla_attn_graph_ = cfg.get_bool("mla_attn_graph");
  cpu_layer_ffn_graph_ = cfg.get_bool("cpu_layer_ffn_graph");
  cpu_dual_stream_ = cfg.get_bool("cpu_dual_stream");
  tail_zero_copy_ = cfg.get_bool("tail_zero_copy");
  // head_topk_max is read directly in the constructor (it sizes the graph buffers).
  no_graph_cfg_ = cfg.get_bool("no_graph");
  layer_graph_auto_ = cfg.get_bool("layer_graph_auto");
  no_bm_ = cfg.get_bool("no_bm");
  no_shared_ = cfg.get_bool("no_shared");
  no_moe_fused_ = cfg.get_bool("no_moe_fused");
  keep_gpu_busy_ = cfg.get_bool("keep_gpu_busy");
  layers_override_ = static_cast<int>(cfg.get_int("layers"));
  experts_topk_ = static_cast<int>(cfg.get_int("experts_topk"));
  bm_types_ = cfg.get_string("bm_types");
  row_threshold_ = static_cast<int>(cfg.get_int("row_threshold"));
  q6k_x4_min_ = static_cast<int>(cfg.get_int("q6k_x4_min"));
  tiny_sk_max_nout_ = static_cast<int>(cfg.get_int("tiny_sk_max_nout"));
  tiny_sk_min_nin_ = static_cast<int>(cfg.get_int("tiny_sk_min_nin"));
  q6k_dp4a_split_max_nout_ = static_cast<int>(cfg.get_int("q6k_dp4a_split_max_nout"));
  small_nout_split_max_ = static_cast<int>(cfg.get_int("small_nout_split_max"));
  q6k_split_cfg_ = static_cast<int>(cfg.get_int("q6k_split"));
  vram_extend_ = cfg.get_bool("vram_extend");
  vram_auto_layers_ = cfg.get_bool("vram_auto_layers");
  vram_auto_layers_max_ = static_cast<int>(cfg.get_int("vram_auto_layers_max"));
  vram_margin_floor_bytes_ = static_cast<size_t>(cfg.get_int("vram_margin_floor_mb")) << 20;
  q35_cpu_layer_graph_ = cfg.get_bool("cpu_layer_graph");
  rms4_ = cfg.get_bool("rms4");
  head_topk_ = cfg.get_bool("head_topk");
  // KV cache precision: f32 (historical) | q8_0 (34 B / 32 values) | q4_0 (18 B / 32)
  {
    const std::string kvt = cfg.get_string("kv_type");
    kv_type_ = 0;
    if (kvt == "q8_0" || kvt == "q8" || kvt == "8") kv_type_ = 1;
    else if (kvt == "q4_0" || kvt == "q4" || kvt == "4") kv_type_ = 2;
    else if (!kvt.empty() && kvt != "f32" && kvt != "fp32")
      std::fprintf(stderr, "[kv] unknown kv_type '%s' (f32|q8_0|q4_0) -> f32\n",
                   kvt.c_str());
    kv_split_ = static_cast<int>(cfg.get_int("kv_split"));
    if (kv_split_ < 1) kv_split_ = 1;
    if (kv_split_ > 32) kv_split_ = 32;   // config caps at 32; keep the clamp in sync
  }
  vram_margin_bytes_ = static_cast<size_t>(cfg.get_int("vram_margin_mb")) << 20;
  vram_adaptive_ = cfg.get_bool("vram_check");
  vram_extra_reserve_ = static_cast<size_t>(cfg.get_int("vram_extra_reserve_mb")) << 20;
  gpu_layers_explicit_ = cfg.was_set("gpu_layers");
  has_kernel_overrides_ = !cfg.kernel_overrides().empty();
}

bool DenseRunner::expert_layer_fits_vram(int layer, int moe_index, size_t device_bytes,
                                         size_t extra_bytes) {
  // Only applies to the automatic layer-granular placement.  An explicit
  // per-layer expert split (gpu_experts >= 0) keeps the historical behaviour.
  if (!vram_adaptive_ || gpu_moe_layers_ < 0 || gpu_experts_ >= 0) return true;
  const cuda::MemInfo mi = cuda::mem_info();
  // Everything that still has to be allocated after this layer: non-expert
  // weights of the remaining layers, their KV caches, and the post-load scratch.
  size_t residual = vram_post_load_reserve_;
  if (layer + 1 >= 0 && layer + 1 < static_cast<int>(vram_other_suffix_.size()))
    residual += vram_other_suffix_[layer + 1];
  for (size_t j = static_cast<size_t>(layer) + 1; j < vram_layer_has_kv_.size(); ++j)
    if (vram_layer_has_kv_[j]) residual += vram_kv_per_layer_;
  // 96 MiB slack for the device pointer arrays, allocator granularity and the measured
  // post-load estimation gap (Coder-V2-Lite at ctx 22: the last adaptive layer passed the
  // estimate with only ~38 MiB left, then the final free-VRAM check came out 38 MiB below
  // the configured margin).  The slack keeps the "free >= margin after load" contract
  // instead of letting a borderline layer in and then only warning about it.
  // The 96 MiB slack covers pointer arrays / allocator granularity; vram_extra_reserve_
  // covers the ledger-invisible driver/cuBLAS overhead, otherwise the last admitted
  // layer can consume the post-load margin (that was the auto-placement flake: the same
  // command landed on 14/15/17 GPU layers depending on ambient free VRAM and one of
  // those picked free=0 after load).
  const size_t need =
      device_bytes + extra_bytes + residual + (96ull << 20) + vram_extra_reserve_;
  if (mi.free_bytes >= need + vram_margin_bytes_) {
    if (debug_)
      std::fprintf(stderr,
                   "[vram] MoE layer #%d (blk.%d) fits: free %lld MiB, need %lld MiB "
                   "(residual %lld MiB), margin %lld MiB\n",
                   moe_index, layer, static_cast<long long>(mi.free_bytes >> 20),
                   static_cast<long long>(need >> 20),
                   static_cast<long long>(residual >> 20),
                   static_cast<long long>(vram_margin_bytes_ >> 20));
    return true;
  }
  const int before = gpu_moe_layers_;
  gpu_moe_layers_ = moe_index;  // this layer and every later one stay on the CPU
  std::fprintf(stderr,
               "[placement] adaptive: MoE layer #%d (blk.%d) does not fit (free %lld MiB, "
               "layer needs %lld MiB, residual %lld MiB, margin %lld MiB) -> "
               "gpu_moe_layers %d -> %d\n",
               moe_index, layer, static_cast<long long>(mi.free_bytes >> 20),
               static_cast<long long>((device_bytes + extra_bytes) >> 20),
               static_cast<long long>(residual >> 20),
               static_cast<long long>(vram_margin_bytes_ >> 20), before, gpu_moe_layers_);
  return false;
}

void DenseRunner::check_post_load_vram() {
  if (!vram_adaptive_) return;
  const cuda::MemInfo mi = cuda::mem_info();
  const long long free_mb = static_cast<long long>(mi.free_bytes >> 20);
  const long long margin_mb = static_cast<long long>(vram_margin_bytes_ >> 20);
  // The ledger gap (device used - engine allocations) is the value vram_extra_reserve_
  // is calibrated against; printing it makes future calibration data-driven.
  const long long used_mb = static_cast<long long>((mi.total_bytes - mi.free_bytes) >> 20);
  const long long ledger_mb = static_cast<long long>(cuda::g_buf_bytes >> 20);
  if (mi.free_bytes >= vram_margin_bytes_) {
    std::fprintf(stderr,
                 "[vram] check OK: %lld MiB free >= %lld MiB margin (gpu_moe_layers=%d)"
                 "%s [ledger gap %lld MiB, reserve %lld MiB]\n",
                 free_mb, margin_mb, gpu_moe_layers_,
                 gpu_layers_explicit_ ? " [explicit --gpu-layers]" : " [auto]",
                 used_mb - ledger_mb,
                 static_cast<long long>(vram_extra_reserve_ >> 20));
  } else {
    // The activation buffers are sized before the weights are uploaded (see the actual
    // free-VRAM re-derivation there) and this check runs after them, so shrinking rows_
    // here would not free anything -- it used to just print "Prefill chunk halved".  The
    // lever that actually works is the placement estimate above (vram_extra_reserve_);
    // this path is now a loud, actionable backstop.
    std::fprintf(stderr,
                 "[vram] ERROR: only %lld MiB free after load, margin is %lld MiB "
                 "(ledger gap %lld MiB, extra reserve %lld MiB). Driver paging will make "
                 "this run ~10x slower. Fix: lower --gpu-layers (currently auto-placed at "
                 "%d), raise --vram-margin, or raise --vram-extra-reserve.%s\n",
                 free_mb, margin_mb, used_mb - ledger_mb,
                 static_cast<long long>(vram_extra_reserve_ >> 20), gpu_moe_layers_,
                 "");
  }
}

void DenseRunner::apply_auto_vram_layers() {
  if (!vram_auto_layers_ || vram_auto_layers_max_ <= 0) return;
  if (!vram_adaptive_ || gpu_moe_layers_ < 0 || gpu_experts_ >= 0) return;
  if (container_ != Container::Gguf) return;
  if (engine::config::was_set("vram_margin_mb") || engine::config::was_set("gpu_layers") ||
      engine::config::was_set("vram_budget_mb"))
    return;  // explicit placement knobs are the user's contract: never second-guess them
  const int ne = ir_.moe.n_expert;
  if (ne <= 0) return;
  // Only meaningful for a real CPU offload (a single promoted layer on a 2-layer toy
  // model is not "one more layer", it is a different placement).
  int n_moe = 0;
  for (const auto& ly : ir_.layers)
    if (ly.n_expert_tensors > 0) ++n_moe;
  if (n_moe - gpu_moe_layers_ < 2) return;
  int promoted = 0;
  for (int i = 0; i < ir_.n_layer && promoted < vram_auto_layers_max_; ++i) {
    LayerW& w = layers_[i];
    if (!w.is_moe || w.cpu_begin < 0 || w.cpu_begin >= ne) continue;
    if (w.h_gate_bytes.empty() || w.h_up_bytes.empty() || w.h_down_bytes.empty()) break;
    const size_t g_slab = w.h_gate_bytes.size() / static_cast<size_t>(ne);
    const size_t u_slab = w.h_up_bytes.size() / static_cast<size_t>(ne);
    const size_t d_slab = w.h_down_bytes.size() / static_cast<size_t>(ne);
    const size_t need = w.h_gate_bytes.size() + w.h_up_bytes.size() + w.h_down_bytes.size();
    if (!g_slab || !u_slab || !d_slab) break;
    const cuda::MemInfo mi = cuda::mem_info();
    // Leave the configured post-load margin AND the ledger reserve free, plus the
    // measured post-promotion activation/graph pool (35B: ~1.1 GiB allocated after the
    // promotion inside load_weights).  The serve guard rejects a start below --vram-margin,
    // and a promotion must not create that state; when the reserve does not fit, keep the
    // layer on the CPU (explicit --gpu-layers is still available).
    const size_t promo_reserve = std::max(vram_margin_floor_bytes_, vram_margin_bytes_) +
                                 vram_extra_reserve_ + (1152ull << 20);
    if (mi.free_bytes < need + promo_reserve) {
      if (debug_)
        std::fprintf(stderr,
                     "[placement] auto-vram: stop at blk.%d (need %.0f MiB + reserve %.0f MiB "
                     "> free %.0f MiB)\n",
                     i, need / 1048576.0, promo_reserve / 1048576.0,
                     mi.free_bytes / 1048576.0);
      break;
    }
    std::vector<QTensor> qe(static_cast<size_t>(ne) * 3);
    try {
      auto mk = [](QTensor& t, const unsigned char* src, size_t nbytes,
                   const std::string& type, int bb, int be, int n_out, int n_in) {
        t.data = cuda::Buffer(nbytes);
        t.data.upload(src, nbytes);
        t.type = type;
        t.block_bytes = bb;
        t.block_elems = be;
        t.n_out = n_out;
        t.n_in = n_in;
        t.valid = true;
      };
      for (int e = 0; e < ne; ++e) {
        mk(qe[static_cast<size_t>(e) * 3 + 0], w.h_gate_bytes.data() + static_cast<size_t>(e) * g_slab,
           g_slab, w.h_type_g, w.h_bb_g, w.h_be_g, expert_inter_, hidden_);
        mk(qe[static_cast<size_t>(e) * 3 + 1], w.h_up_bytes.data() + static_cast<size_t>(e) * u_slab,
           u_slab, w.h_type_u, w.h_bb_u, w.h_be_u, expert_inter_, hidden_);
        mk(qe[static_cast<size_t>(e) * 3 + 2], w.h_down_bytes.data() + static_cast<size_t>(e) * d_slab,
           d_slab, w.h_type_d, w.h_bb_d, w.h_be_d, hidden_, expert_inter_);
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr,
                   "[placement] auto-vram: promote blk.%d failed (%s) -> keeping it on the CPU\n",
                   i, e.what());
      break;
    }
    w.q_experts = std::move(qe);
    w.cpu_begin = ne;
    // The layer just switched from the CPU-expert path to the GPU-expert path; graphs
    // captured for the old path (e.g. attention-only CPU-MoE graphs) are stale.
    w.layer_graph.reset();
    w.layer_graph_ready = false;
    w.layer_graph_state = 0;
    for (int b = 0; b < 3; ++b) {
      w.attn_graph[b].reset();
      w.attn_graph_state[b] = 0;
    }
    // The per-layer device pointer arrays were uploaded all-nullptr while the layer was
    // host-resident; refresh them with the new device addresses (same layout as load).
    if (w.all_ptr_idx >= 0 && w.all_ptr_idx < static_cast<int>(all_gp_.size())) {
      const size_t ne_bytes = static_cast<size_t>(ne) * 8;
      std::vector<const unsigned char*> gp(static_cast<size_t>(ne), nullptr);
      std::vector<const unsigned char*> upv(static_cast<size_t>(ne), nullptr);
      std::vector<const unsigned char*> dpv(static_cast<size_t>(ne), nullptr);
      for (int e = 0; e < ne; ++e) {
        gp[static_cast<size_t>(e)] = static_cast<const unsigned char*>(
            w.q_experts[static_cast<size_t>(e) * 3 + 0].data.data());
        upv[static_cast<size_t>(e)] = static_cast<const unsigned char*>(
            w.q_experts[static_cast<size_t>(e) * 3 + 1].data.data());
        dpv[static_cast<size_t>(e)] = static_cast<const unsigned char*>(
            w.q_experts[static_cast<size_t>(e) * 3 + 2].data.data());
      }
      all_gp_[static_cast<size_t>(w.all_ptr_idx)].upload(gp.data(), ne_bytes);
      all_up_[static_cast<size_t>(w.all_ptr_idx)].upload(upv.data(), ne_bytes);
      all_dp_[static_cast<size_t>(w.all_ptr_idx)].upload(dpv.data(), ne_bytes);
    }
    // The device copy is authoritative now; the host mirror only wastes ~16 MiB of RAM.
    w.h_gate_bytes.clear();
    w.h_up_bytes.clear();
    w.h_down_bytes.clear();
    ++promoted;
    ++gpu_moe_layers_;
    const cuda::MemInfo mi2 = cuda::mem_info();
    std::fprintf(stderr,
                 "[placement] auto-vram: promoted MoE layer blk.%d to the GPU (%.0f MiB, "
                 "free %.0f -> %.0f MiB, reserve %.0f MiB) gpu_moe_layers -> %d\n",
                 i, need / 1048576.0, mi.free_bytes / 1048576.0, mi2.free_bytes / 1048576.0,
                 promo_reserve / 1048576.0, gpu_moe_layers_);
  }
  if (promoted > 0) {
    const cuda::MemInfo mi = cuda::mem_info();
    std::fprintf(stderr,
                 "[vram] check OK after auto-vram: %lld MiB free >= %lld MiB floor "
                 "(promoted %d layer%s)\n",
                 static_cast<long long>(mi.free_bytes >> 20),
                 static_cast<long long>(vram_margin_floor_bytes_ >> 20), promoted,
                 promoted == 1 ? "" : "s");
  }
}

// ---- deepseek2 MLA: per-layer CUDA graph for the decode attention segment --------
// The segment is kernel-only on decode: the four projections use the quantized GEMV
// path (L.mla_quant) and the fp32 k/v expansion calls matvec_f32_ptr, which selects the
// gemv_f32 *kernel* for n_tokens <= 4 (cuBLAS is only used for batched prefill).  It can
// therefore be recorded exactly like the qwen35 mixer graphs, replacing ~13 WDDM
// submissions per layer with one cuGraphLaunch.  This is Coder's only graph coverage:
// it is a MoE model, so the whole-pipeline graph stays disabled.
bool DenseRunner::mla_attn_graph_ok(int layer) const {
  const auto& L = layers_[layer];
  if (!L.is_mla) return false;
  // f32 mirrors go through cuBLAS even on decode -> not capturable.
  if (!L.mla_quant) return false;
  // The "global" q layout keeps a cuBLAS q projection.  The global/perhead k-v split
  // layouts are all GEMV + launch kernels, so both are fine.
  if (mla_q_global_) return false;
  // Diagnostics take host synchronisations inside the segment (and the dump code cannot
  // run during capture), so they fall back to the plain path.
  // ENGINE_PIPE_PROF is allowed: it only prints the launch counter between pipeline
  // calls (outside the captured segment), so the graph can stay on and the counter
  // difference shows the launch-count reduction.  The dump/profiling switches do
  // synchronise inside the segment and KPROF's events would become graph nodes, so
  // those keep the plain path.
  if (ENGINE_ENV_ON("ENGINE_DUMP") || ENGINE_ENV_ON("ENGINE_DUMP_MLA") ||
      ENGINE_ENV_ON("ENGINE_PROFILE") || ENGINE_ENV_ON("ENGINE_KPROF"))
    return false;
  return true;
}

// Plain sequence: ln1 norm -> MLA attention (which also adds the output projection into
// x_) -> ln2 norm.  Used by prefill and whenever the graph path is disabled.
void DenseRunner::mla_attn_segment(int layer, int n_tokens, int start_pos) {
  auto& L = layers_[layer];
  rms_raw(x_.data(), L.ln1.data(), hn_.data(), n_tokens, hidden_);
  if (layer == 0) {
    if (const char* dp = ENGINE_ENV_STR("ENGINE_DUMP")) {
      cuda::stream_sync();
      std::vector<float> h(static_cast<size_t>(n_tokens) * hidden_);
      hn_.download(h.data(), h.size() * 4);
      std::ofstream f(std::string(dp) + "_hn0.f32", std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()),
              static_cast<std::streamsize>(h.size() * 4));
    }
  }
  if (ENGINE_ENV_ON("ENGINE_PROFILE") && layer < 3 && n_tokens == 1) {
    cuda::sync();
    const auto t0 = std::chrono::steady_clock::now();
    run_mla_attention(layer, n_tokens, start_pos);
    cuda::sync();
    std::fprintf(stderr, "[attn-prof] layer %d mla %.2f ms\n", layer,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                     .count());
  } else {
    const auto mp0 = std::chrono::steady_clock::now();
    run_mla_attention(layer, n_tokens, start_pos);
    g_prof_mla_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mp0).count();
  }
  rms_raw(x_.data(), L.ln2.data(), hn_.data(), n_tokens, hidden_);
}

// Record the segment without executing it.  The caller runs the plain sequence for the
// current token first, so recording cannot change the state it captures.
void DenseRunner::capture_mla_attn_graph(int layer) {
  auto& L = layers_[layer];
  cuda::stream_sync();
  auto g = std::make_unique<cuda::Graph>();
  const long long cap_c0 = cuda::g_launch_calls;
  try {
    g->capture_begin();
    graph_capture_ = true;
    mla_attn_segment(layer, 1, 0);
    graph_capture_ = false;
    g->capture_end();
    if (ENGINE_ENV_ON("ENGINE_DBG_MLA_CAP"))
      std::fprintf(stderr, "[mla-cap] layer %d recorded %lld launches\n", layer,
                   cuda::g_launch_calls - cap_c0);
  } catch (const std::exception& e) {
    graph_capture_ = false;
    g->capture_abort();
    // Never retry in this process: a half-recorded stream would poison later work.
    mla_attn_graph_ = false;
    std::fprintf(stderr, "[mla-graph] disabled: %s\n", e.what());
    return;
  }
  // Recording never executed the kernels, and replay does not advance
  // cuda::g_launch_calls; docs/GRAPH-INVARIANTS.md invariant 1 keeps the int8 staging
  // cache honest via cuda::g_graph_launches, so no manual cache reset is needed here.
  L.mla_attn_graph = std::move(g);
  L.mla_attn_graph_state = 2;
  if (ENGINE_ENV_ON("ENGINE_DBG_GRAPH"))
    std::fprintf(stderr, "[graph] layer %d mla attention captured\n", layer);
}

// Replay the layer's attention segment, or execute it once and record it.
void DenseRunner::run_mla_attn_graph(int layer) {
  auto& L = layers_[layer];
  if (L.mla_attn_graph_state == 2 && L.mla_attn_graph) {
    static int replay_dbg = 0;
    if (ENGINE_ENV_ON("ENGINE_DBG_MLA_CAP") && replay_dbg++ < 6)
      std::fprintf(stderr, "[mla-cap] replay layer %d\n", layer);
    // Replay executes the recorded kernels without advancing cuda::g_launch_calls; the
    // quant_reuse_ cache is kept honest by cuda::g_graph_launches (invariant 1).
    L.mla_attn_graph->launch();
    return;
  }
  mla_attn_segment(layer, 1, 0);
  capture_mla_attn_graph(layer);
}

void DenseRunner::run_mla_attention(int layer, int n_tokens, int start_pos) {
  auto& L = layers_[layer];
  const int qk = ir_.attn.key_length;
  const int qk_rope = ir_.attn.rope_dim;
  const int qk_nope = qk - qk_rope;
  const int v_dim = ir_.attn.value_length;
  const int kv_lora = ir_.attn.kv_lora_rank;

  if (mla_q_global_) {
    cublas::gemm_f32_nt(n_tokens, n_head_ * qk, hidden_, static_cast<const float*>(hn_.data()),
                        static_cast<const float*>(L.wq_mla.data()),
                        static_cast<float*>(mla_qraw_.data()));
    int nh = n_head_, n1 = qk_nope, n2 = qk_rope;
    int pef = mla_q_pe_first_ ? 1 : 0;
    void* ps = mla_qraw_.data();
    void* pd = mq_.data();
    void* args[] = {&ps, &pd, &nh, &n1, &n2, &pef};
    cuda::launch(k_mlaqreorder_, n_tokens, 1, 1, 128, 1, 1, 0, args);
  } else {
    if (L.mla_quant) matmul(L.qwq_mla, n_tokens, hn_, mq_);
    else matvec_f32(L.wq_mla, n_head_ * qk, hidden_, hn_, mq_);
  }
  if (layer == 0 && std::getenv("ENGINE_DUMP_MLA")) {
    // q before rope, so the projection and the rotation can be checked separately
    const char* dp = ENGINE_ENV_STR("ENGINE_DUMP_MLA");
    cuda::stream_sync();
    std::vector<float> hq(static_cast<size_t>(n_tokens) * static_cast<size_t>(n_head_) *
                          static_cast<size_t>(qk));
    mq_.download(hq.data(), hq.size() * 4);
    std::ofstream fq(std::string(dp) + "_q_pre_rope.f32", std::ios::binary);
    fq.write(reinterpret_cast<const char*>(hq.data()),
             static_cast<std::streamsize>(hq.size() * 4));
  }
  static const char* mla_dump = ENGINE_ENV_STR("ENGINE_DUMP_MLA");
  auto mla_dump_save = [&](cuda::Buffer& buf, const char* tag, size_t n) {
    if (!mla_dump || layer != 0) return;
    cuda::stream_sync();
    std::vector<float> h(n);
    buf.download(h.data(), h.size() * 4);
    std::ofstream fo(std::string(mla_dump) + "_" + tag + ".f32", std::ios::binary);
    fo.write(reinterpret_cast<const char*>(h.data()),
             static_cast<std::streamsize>(h.size() * 4));
  };
  mla_dump_save(hn_, "hn", static_cast<size_t>(n_tokens) * static_cast<size_t>(hidden_));
  if (L.mla_quant) matmul(L.qwkva, n_tokens, hn_, mkva_);
  else matvec_f32(L.wkva, kv_lora + qk_rope, hidden_, hn_, mkva_);
  mla_dump_save(mkva_, "ckv",
                static_cast<size_t>(n_tokens) * static_cast<size_t>(kv_lora + qk_rope));
  {
    int na = kv_lora, nb = qk_rope;
    void* ps = mkva_.data();
    void* pa = mckv_.data();
    void* pb = mkpe_.data();
    void* args[] = {&ps, &pa, &pb, &n_tokens, &na, &nb};
    cuda::launch(k_split2_, n_tokens, 1, 1, 128, 1, 1, 0, args);
  }
  {
    rms_raw(mckv_.data(), L.wkvan.data(), mckvn_.data(), n_tokens, kv_lora);
  }
  mla_dump_save(mckvn_, "kvn", static_cast<size_t>(n_tokens) * static_cast<size_t>(kv_lora));
  if (mla_kv_perhead_) {
    if (L.mla_quant) matmul(L.qwkb, n_tokens, mckvn_, mla_kvraw_);
    else matvec_f32(L.wkb, n_head_ * (qk_nope + v_dim), kv_lora, mckvn_, mla_kvraw_);
    int nh = n_head_, n1 = qk_nope, n2 = v_dim, vf = mla_kv_vfirst_ ? 1 : 0;
    void* ps = mla_kvraw_.data();
    void* pk = mkn_.data();
    void* pv = mv_.data();
    void* args[] = {&ps, &pk, &pv, &nh, &n1, &n2, &vf};
    cuda::launch(k_mlakvsplit_, n_tokens, 1, 1, 128, 1, 1, 0, args);
  } else {
    // k_nope / v row ranges inside Wkb (global-split layouts)
    const float* wkb_base = static_cast<const float*>(L.wkb.data());
    const float* wkb_k =
        !mla_kv_vfirst_ ? wkb_base
                        : wkb_base + static_cast<size_t>(n_head_) * v_dim * kv_lora;
    matvec_f32_ptr(wkb_k, n_head_ * qk_nope, kv_lora, mckvn_, mkn_);
    const float* wkb_v = mla_kv_vfirst_
                             ? wkb_base
                             : wkb_base + static_cast<size_t>(n_head_) * qk_nope * kv_lora;
    matvec_f32_ptr(wkb_v, n_head_ * v_dim, kv_lora, mckvn_, mv_);
  }
  {
    int nh = n_head_, n1 = qk_nope, n2 = qk_rope;
    void* pk = mkn_.data();
    void* pp = mkpe_.data();
    void* pd = mk_.data();
    void* args[] = {&pk, &pp, &pd, &nh, &n1, &n2};
    cuda::launch(k_buildk_, n_tokens, 1, 1, 64, 1, 1, 0, args);
  }
  auto rope_part = [&](cuda::Buffer& buf) {
    int nh = n_head_, stride = qk, rd = qk_rope, off = qk_nope;
    void* pb = buf.data();
    void* pp = pos_dev_.data();
    void* pf = rope_freqs_dev_.data();
    float ms = rope_mscale_;
    void* args[] = {&pb, &pp, &nh, &stride, &rd, &off, &pf, &ms};
    // DeepSeek-V2's reference views the rope dims as complex pairs -> interleaved
    static const bool rope_inter = !ENGINE_ENV_ON("ENGINE_MLA_ROPE_NEOX");
    cuda::launch(rope_inter ? k_ropeinter_ : k_ropestrided_, n_tokens, n_head_, 1,
                 static_cast<int>(qk_rope / 2), 1, 1, 0, args);
  };
  rope_part(mq_);
  rope_part(mk_);
  mla_dump_save(mq_, "q", static_cast<size_t>(n_tokens) * static_cast<size_t>(n_head_) *
                            static_cast<size_t>(ir_.attn.key_length));
  mla_dump_save(mk_, "k", static_cast<size_t>(n_tokens) * static_cast<size_t>(n_head_) *
                            static_cast<size_t>(ir_.attn.key_length));
  mla_dump_save(mv_, "v2", static_cast<size_t>(n_tokens) * static_cast<size_t>(n_head_) *
                             static_cast<size_t>(ir_.attn.value_length));
  // write expanded k/v into the per-layer cache, then attend over it
  {
    int nh = n_head_, qk_ = qk, vd = v_dim, ms = max_seq_;
    void* pk = mk_.data();
    void* pv = mv_.data();
    void* pK = L.mla_kc.data();
    void* pV = L.mla_vc.data();
    void* pp = pos_dev_.data();
    void* args[] = {&pk, &pv, &pK, &pV, &pp, &nh, &qk_, &vd, &ms};
    cuda::launch(k_mlakv_, n_tokens * n_head_, 1, 1, 128, 1, 1, 0, args);
  }
  {
    int nh = n_head_, qk_ = qk, vd = v_dim, ms = max_seq_;
    void* pq = mq_.data();
    void* pK = L.mla_kc.data();
    void* pV = L.mla_vc.data();
    void* po = mout_.data();
    void* pp = pos_dev_.data();
    void* args[] = {&pq, &pK, &pV, &po, &pp, &nh, &qk_, &vd, &ms};
    const size_t sh = static_cast<size_t>(max_seq_) * sizeof(float);
    if (!ENGINE_ENV_ON("ENGINE_SKIP_MLA_ATTN")) {
      cuda::max_dynamic_shared_mem(k_mlaattnc_);  // allow > 48 KiB for ctx > 12288
      cuda::launch(k_mlaattnc_, n_tokens, n_head_, 1, 32, 1, 1, sh, args);
    }
  }
  if (L.mla_quant) matmul(L.qwo_mla, n_tokens, mout_, proj_);
  else matvec_f32(L.wo_mla, hidden_, n_head_ * v_dim, mout_, proj_);
  if (layer == 0 && std::getenv("ENGINE_DUMP_MLA")) {
    const char* dp = ENGINE_ENV_STR("ENGINE_DUMP_MLA");
    cuda::stream_sync();
    std::vector<float> hp(static_cast<size_t>(n_tokens) * static_cast<size_t>(hidden_));
    proj_.download(hp.data(), hp.size() * 4);
    std::ofstream fo(std::string(dp) + "_attn0.f32", std::ios::binary);
    fo.write(reinterpret_cast<const char*>(hp.data()),
             static_cast<std::streamsize>(hp.size() * 4));
    std::vector<float> hv(static_cast<size_t>(n_tokens) * static_cast<size_t>(n_head_) *
                          static_cast<size_t>(ir_.attn.value_length));
    mv_.download(hv.data(), hv.size() * 4);
    std::ofstream fv(std::string(dp) + "_v0.f32", std::ios::binary);
    fv.write(reinterpret_cast<const char*>(hv.data()),
             static_cast<std::streamsize>(hv.size() * 4));
  }
  {
    long long total = static_cast<long long>(n_tokens) * hidden_;
    void* pa = x_.data();
    void* pb = proj_.data();
    void* args[] = {&pa, &pb, &total};
    cuda::launch(k_add_, static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1, 0, args);
  }
}

}  // namespace runner
