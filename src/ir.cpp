#include "engine/ir.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>

#include "engine/gguf.h"
#include "engine/json_parse.h"
#include "engine/st.h"

namespace ir {

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool contains(const std::string& s, const char* part) {
  return s.find(part) != std::string::npos;
}

int64_t meta_int(const gguf::Model& g, const std::string& arch, const std::string& suffix,
                 int64_t fallback = 0) {
  return g.get_int(arch + "." + suffix, fallback);
}

double meta_float(const gguf::Model& g, const std::string& arch, const std::string& suffix,
                  double fallback) {
  return g.get_float(arch + "." + suffix, fallback);
}

}  // namespace

const char* role_name(Role r) {
  switch (r) {
    case Role::EMBED: return "embed";
    case Role::OUTPUT: return "output";
    case Role::ATTN: return "attn";
    case Role::FFN_DENSE: return "ffn_dense";
    case Role::EXPERT: return "expert";
    case Role::SHARED_EXPERT: return "shared_expert";
    case Role::ROUTER: return "router";
    case Role::NORM: return "norm";
    case Role::OTHER: return "other";
  }
  return "other";
}

Role classify(const std::string& name) {
  if (name == "token_embd.weight" || ends_with(name, ".token_embd.weight")) return Role::EMBED;
  if (name == "output.weight" || ends_with(name, ".output.weight")) return Role::OUTPUT;
  if (contains(name, "_exps")) return Role::EXPERT;
  if (contains(name, "_shexp")) return Role::SHARED_EXPERT;
  if (contains(name, "ffn_gate_inp")) return Role::ROUTER;
  if (contains(name, "attn_") || contains(name, "ssm_") || contains(name, "time_mix") ||
      contains(name, "rope"))
    return Role::ATTN;
  if (contains(name, "norm")) return Role::NORM;
  if (contains(name, "ffn_")) return Role::FFN_DENSE;
  return Role::OTHER;
}

int layer_of(const std::string& name) {
  if (name.size() < 5 || name.compare(0, 4, "blk.") != 0) return -1;
  size_t i = 4;
  int v = 0;
  bool any = false;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
    v = v * 10 + (name[i] - '0');
    ++i;
    any = true;
  }
  if (!any || i >= name.size() || name[i] != '.') return -1;
  return v;
}

uint64_t Model::total_bytes() const {
  uint64_t t = 0;
  for (const auto& x : tensors) t += x.nbytes;
  return t;
}

uint64_t Model::fixed_bytes() const {
  uint64_t t = 0;
  for (const auto& x : tensors) {
    switch (x.role) {
      case Role::EMBED:
      case Role::OUTPUT:
      case Role::ATTN:
      case Role::FFN_DENSE:
      case Role::NORM:
      case Role::ROUTER:
      case Role::SHARED_EXPERT: t += x.nbytes; break;
      default: break;
    }
  }
  return t;
}

uint64_t Model::expert_bytes() const {
  uint64_t t = 0;
  for (const auto& x : tensors)
    if (x.role == Role::EXPERT) t += x.nbytes;
  return t;
}

std::map<int, uint64_t> Model::expert_bytes_per_layer() const {
  std::map<int, uint64_t> out;
  for (const auto& x : tensors)
    if (x.role == Role::EXPERT && x.layer >= 0) out[x.layer] += x.nbytes;
  return out;
}

Model build_from_gguf(const gguf::Model& g) {
  Model m;
  m.container = "gguf";
  m.path = g.path;
  m.file_size = g.file_size;
  m.arch = g.get_string("general.architecture");
  if (m.arch.empty()) throw std::runtime_error("GGUF missing general.architecture");

  m.n_layer = static_cast<int>(meta_int(g, m.arch, "block_count"));
  m.hidden = static_cast<int>(meta_int(g, m.arch, "embedding_length"));
  m.ffn = static_cast<int>(meta_int(g, m.arch, "feed_forward_length"));
  m.ctx_train = meta_int(g, m.arch, "context_length");
  if (m.n_layer <= 0 || m.hidden <= 0)
    throw std::runtime_error(
        "GGUF metadata for arch '" + m.arch + "' is missing " +
        (m.n_layer <= 0 ? "block_count" : "embedding_length") +
        ". This engine understands llama-style metadata keys (<arch>.block_count, "
        "<arch>.embedding_length, <arch>.attention.*) for these families: "
        "llama-like dense (llama / qwen2 / qwen2moe / qwen3), MLA (deepseek2 / deepseek3), "
        "hybrid SSM (qwen35 / qwen35moe). The model file may use a different architecture "
        "or an unsupported GGUF layout.");

  auto& a = m.attn;
  a.n_head = static_cast<int>(meta_int(g, m.arch, "attention.head_count"));
  a.n_head_kv = static_cast<int>(meta_int(g, m.arch, "attention.head_count_kv"));
  a.key_length = static_cast<int>(meta_int(g, m.arch, "attention.key_length"));
  a.value_length = static_cast<int>(meta_int(g, m.arch, "attention.value_length"));
  a.rope_freq_base = meta_float(g, m.arch, "rope.freq_base", 10000.0);
  a.rope_dim = static_cast<int>(meta_int(g, m.arch, "rope.dimension_count"));
  a.sliding_window = static_cast<int>(meta_int(g, m.arch, "attention.sliding_window"));
  a.full_attention_interval =
      static_cast<int>(meta_int(g, m.arch, "full_attention_interval"));
  a.kv_lora_rank = static_cast<int>(meta_int(g, m.arch, "attention.kv_lora_rank"));
  a.leading_dense_blocks =
      static_cast<int>(meta_int(g, m.arch, "leading_dense_block_count"));

  if (a.key_length == 0 && a.n_head > 0) {
    if (m.hidden % a.n_head != 0)
      throw std::runtime_error("IR: hidden_size " + std::to_string(m.hidden) +
                               " is not divisible by attention.head_count " +
                               std::to_string(a.n_head));
    a.key_length = a.value_length = m.hidden / a.n_head;
  }
  // #13: GQA/MQA group geometry must be integral.  The attention kernels map query head
  // h to kv head h / (n_head / n_head_kv); with n_head % n_head_kv != 0 that integer
  // division silently drops heads and indexes the wrong K/V rows.
  if (a.n_head > 0 && a.n_head_kv > 0) {
    if (a.n_head_kv > a.n_head)
      throw std::runtime_error("IR: attention.head_count_kv (" +
                               std::to_string(a.n_head_kv) +
                               ") exceeds attention.head_count (" + std::to_string(a.n_head) +
                               ")");
    if (a.n_head % a.n_head_kv != 0)
      throw std::runtime_error("IR: attention.head_count (" + std::to_string(a.n_head) +
                               ") is not divisible by attention.head_count_kv (" +
                               std::to_string(a.n_head_kv) + ")");
  }
  if (a.key_length > 0 && m.hidden > 0 && a.key_length > m.hidden)
    throw std::runtime_error("IR: attention.key_length " + std::to_string(a.key_length) +
                             " exceeds hidden_size " + std::to_string(m.hidden));
  if (a.kv_lora_rank > 0) a.variant = "mla";
  else if (a.n_head_kv > 0 && a.n_head > 0) {
    if (a.n_head_kv == 1) a.variant = "mqa";
    else if (a.n_head_kv < a.n_head) a.variant = "gqa";
    else a.variant = "mha";
  }

  auto& mo = m.moe;
  mo.n_expert = static_cast<int>(meta_int(g, m.arch, "expert_count"));
  mo.n_expert_used = static_cast<int>(meta_int(g, m.arch, "expert_used_count"));
  mo.n_shared = static_cast<int>(meta_int(g, m.arch, "expert_shared_count"));
  mo.norm_topk = meta_int(g, m.arch, "expert_weights_norm") != 0;
  mo.expert_ffn_length = static_cast<int>(meta_int(g, m.arch, "expert_feed_forward_length"));
  mo.layout = mo.n_expert ? "fused_3d" : "none";
  // DeepSeek-V2/V3 score experts with sigmoid; qwen-style MoE uses softmax.
  // DeepSeek-V3 routes with sigmoid gates; V2 (Coder-V2-Lite) uses a plain softmax.
  if (m.arch == "deepseek3") mo.router_kind = "sigmoid_topk";

  m.tok.model = g.get_string("tokenizer.ggml.model", "unknown");
  m.tok.pre = g.get_string("tokenizer.ggml.pre");
  m.tok.n_vocab = g.array_len("tokenizer.ggml.tokens");
  m.tok.eos_id = g.get_int("tokenizer.ggml.eos_token_id", -1);
  m.tok.chat_template = g.has("tokenizer.chat_template");

  for (const auto& t : g.tensors) {
    TensorDesc d;
    d.name = t.name;
    d.shape = t.shape;
    d.dtype = t.type_name;
    const gguf::TypeInfo* ti = gguf::type_info(t.type_id);
    if (!ti) {
      std::string supported;
      for (uint32_t id = 0; id <= 40; ++id) {
        const gguf::TypeInfo* info = gguf::type_info(id);
        if (info) supported += std::string(supported.empty() ? "" : ", ") + info->name;
      }
      throw std::runtime_error("tensor '" + t.name + "' uses GGUF quant type id " +
                               std::to_string(t.type_id) +
                               ", which this build does not know how to parse. "
                               "Known GGUF types: " + supported);
    }
    d.quant.scheme = ti->name;
    d.quant.block = ti->block;
    d.quant.bytes_per_block = ti->bytes;
    d.role = classify(t.name);
    d.layer = layer_of(t.name);
    d.file_offset = g.data_offset + t.offset;
    d.nbytes = t.nbytes;
    d.n_elements = t.n_elements;
    m.quant_mix[d.dtype] += d.nbytes;
    m.tensors.push_back(std::move(d));
  }

  // MTP / nextn blocks are not transformer layers (spec §1.2).
  std::vector<int> mtp;
  for (const auto& t : m.tensors)
    if (t.layer >= 0 && t.name.find(".nextn.") != std::string::npos) mtp.push_back(t.layer);
  // Cross-check against the metadata key (both qwen35 and qwen35moe export it).
  const int64_t nextn_meta = meta_int(g, m.arch, "nextn_predict_layers", 0);
  if (!mtp.empty()) {
    std::sort(mtp.begin(), mtp.end());
    mtp.erase(std::unique(mtp.begin(), mtp.end()), mtp.end());
    m.mtp_layers = static_cast<int>(mtp.size());
    m.n_layer = std::min(m.n_layer, mtp.front());
    // descriptor of the MTP block (its tensors get Role::OTHER below, so count the
    // expert slabs by name now).  The MTP layer is always a full-attention layer
    // regardless of the main stack's full_attention_interval: it carries
    // attn_q/attn_k/attn_v/attn_output + a dense FFN or MoE.
    m.mtp_block = mtp.front();
    m.mtp_desc.index = m.mtp_block;
    m.mtp_desc.kind = "attn";
    for (const auto& t : m.tensors) {
      if (t.layer != m.mtp_block) continue;
      ++m.mtp_desc.n_tensors;
      if (t.name.find("_exps.weight") != std::string::npos) ++m.mtp_desc.n_expert_tensors;
    }
    for (auto& t : m.tensors)
      if (t.layer >= m.n_layer) t.role = Role::OTHER;
    m.warnings.push_back("MTP head detected on blocks; excluded from " +
                         std::to_string(m.n_layer) + " layers");
    if (nextn_meta > 0 && nextn_meta != static_cast<int64_t>(m.mtp_layers))
      m.warnings.push_back("nextn_predict_layers=" + std::to_string(nextn_meta) +
                           " but " + std::to_string(m.mtp_layers) +
                           " block(s) carry .nextn. tensors");
  } else if (nextn_meta > 0) {
    m.warnings.push_back("nextn_predict_layers=" + std::to_string(nextn_meta) +
                         " but no .nextn. tensors were found");
  }

  m.layers.resize(static_cast<size_t>(m.n_layer));
  for (int i = 0; i < m.n_layer; ++i) m.layers[i].index = i;
  for (const auto& t : m.tensors) {
    if (t.layer < 0 || t.layer >= m.n_layer) continue;
    auto& L = m.layers[t.layer];
    ++L.n_tensors;
    if (t.role == Role::EXPERT) ++L.n_expert_tensors;
  }

  // qwen35 hybrid stacks: gated delta net parameters
  {
    const int conv_k = static_cast<int>(meta_int(g, m.arch, "ssm.conv_kernel"));
    const int inner = static_cast<int>(meta_int(g, m.arch, "ssm.inner_size"));
    const int state = static_cast<int>(meta_int(g, m.arch, "ssm.state_size"));
    const int groups = static_cast<int>(meta_int(g, m.arch, "ssm.group_count"));
    const int tsr = static_cast<int>(meta_int(g, m.arch, "ssm.time_step_rank"));
    if (conv_k > 0 && inner > 0 && state > 0 && groups > 0) {
      // #13: gated-delta-net geometry.  The mixer indexes q/k/v and the conv state by
      // these strides; `inner / tsr` used to truncate silently for a corrupt (or
      // unsupported) time_step_rank, and a 0/negative tsr produced a zero head dim.
      if (tsr <= 0)
        throw std::runtime_error("IR: ssm.time_step_rank must be positive (got " +
                                 std::to_string(tsr) + ")");
      if (inner % tsr != 0)
        throw std::runtime_error("IR: ssm.inner_size (" + std::to_string(inner) +
                                 ") is not divisible by ssm.time_step_rank (" +
                                 std::to_string(tsr) + ")");
      if (conv_k > 4096 || groups > 1 << 20 || state > 1 << 20)
        throw std::runtime_error("IR: implausible gated-delta-net geometry (conv_kernel=" +
                                 std::to_string(conv_k) + ", group_count=" +
                                 std::to_string(groups) + ", state_size=" +
                                 std::to_string(state) + ")");
      m.ssm.variant = "gated_delta";
      m.ssm.conv_kernel = conv_k;
      m.ssm.n_key_head = groups;
      m.ssm.n_value_head = tsr;
      m.ssm.key_head_dim = state;
      m.ssm.value_head_dim = tsr > 0 ? inner / tsr : 0;
      m.ssm.key_dim = groups * state;
      m.ssm.value_dim = inner;
      m.ssm.conv_dim = m.ssm.key_dim * 2 + inner;
      m.ssm.time_step_rank = tsr;
      // qwen3.5-MoE routes with softmax + normalised top-k and always has a shared expert
    if (m.moe.n_expert > 0) {
      m.moe.norm_topk = true;
      if (m.moe.n_shared == 0) m.moe.n_shared = 1;
    }
    const int interval = static_cast<int>(meta_int(g, m.arch, "full_attention_interval"));
      for (int i = 0; i < m.n_layer; ++i)
        m.layers[i].kind = (interval > 0 && ((i + 1) % interval == 0)) ? "attn" : "linear";
    }
  }

  // -------------------------------------------------------------------------
  // Generic per-layer attention geometry / architecture capabilities.
  // Nothing here matches on the architecture name: scalar metadata provides the
  // global default, array metadata (head_count_kv / sliding_window_pattern) and
  // *_swa variants select per-layer values, and optional tensors enable ops.
  // -------------------------------------------------------------------------
  {
    auto has_t = [&](const std::string& n) {
      for (const auto& t : m.tensors)
        if (t.name == n) return true;
      return false;
    };
    const std::string p = m.arch + ".";
    auto arr = [&](const char* key) { return g.get_int_array(p + key); };
    const std::vector<int64_t> kv_arr = arr("attention.head_count_kv");
    const std::vector<int64_t> swa_arr = arr("attention.sliding_window_pattern");
    const bool kv_per_layer = kv_arr.size() == static_cast<size_t>(m.n_layer);
    const bool swa_per_layer = swa_arr.size() == static_cast<size_t>(m.n_layer);
    if (!kv_arr.empty() && !kv_per_layer)
      m.warnings.push_back("attention.head_count_kv is an array of " +
                           std::to_string(kv_arr.size()) + " values but the model has " +
                           std::to_string(m.n_layer) + " layers; using the scalar value");
    if (!swa_arr.empty() && !swa_per_layer)
      m.warnings.push_back("attention.sliding_window_pattern is an array of " +
                           std::to_string(swa_arr.size()) + " values but the model has " +
                           std::to_string(m.n_layer) + " layers; ignoring it");
    const int kl_swa = static_cast<int>(meta_int(g, m.arch, "attention.key_length_swa"));
    const int vl_swa = static_cast<int>(meta_int(g, m.arch, "attention.value_length_swa"));
    const int rd_swa = static_cast<int>(meta_int(g, m.arch, "rope.dimension_count_swa"));
    const double rb_swa = meta_float(g, m.arch, "rope.freq_base_swa", 0.0);
    auto& caps = m.caps;
    int max_kv = std::max(1, a.n_head_kv);
    int max_kd = std::max(1, a.key_length);
    int max_vd = std::max(1, a.value_length);
    for (int i = 0; i < m.n_layer; ++i) {
      auto& la = m.layers[i].attn;
      const std::string lp = "blk." + std::to_string(i) + ".";
      const bool swa = swa_per_layer && swa_arr[static_cast<size_t>(i)] != 0;
      la.n_kv = kv_per_layer ? static_cast<int>(kv_arr[static_cast<size_t>(i)]) : a.n_head_kv;
      la.key_dim = (swa && kl_swa > 0) ? kl_swa : a.key_length;
      la.value_dim = (swa && vl_swa > 0) ? vl_swa : a.value_length;
      if (la.key_dim <= 0) la.key_dim = m.hidden / std::max(1, a.n_head);
      if (la.value_dim <= 0) la.value_dim = la.key_dim;
      la.rope_dim = (swa && rd_swa > 0) ? rd_swa
                                        : (a.rope_dim > 0 ? a.rope_dim : la.key_dim);
      la.rope_base = (swa && rb_swa > 0) ? rb_swa : a.rope_freq_base;
      la.window = swa ? a.sliding_window : 0;
      const bool has_k = has_t(lp + "attn_k.weight");
      const bool has_v = has_t(lp + "attn_v.weight");
      la.has_v = has_v || !has_k;   // no K projection either -> legacy path
      if (has_k && !has_v) la.has_v = false;   // attention_k_eq_v
      la.qk_norm = has_t(lp + "attn_q_norm.weight") && has_t(lp + "attn_k_norm.weight");
      // V norm is weightless and therefore not detectable from tensors; the
      // family trait is "per-head qk-norm tensors exist".
      la.v_norm = la.qk_norm && la.has_v;
      la.sandwich = has_t(lp + "post_attention_norm.weight") &&
                    has_t(lp + "post_ffw_norm.weight");
      la.layer_scalar = has_t(lp + "layer_output_scale.weight");
      if (la.n_kv > 0) max_kv = std::max(max_kv, la.n_kv);
      max_kd = std::max(max_kd, la.key_dim);
      max_vd = std::max(max_vd, la.value_dim);
      caps.per_layer_attn |= swa || kv_per_layer;
      caps.qk_norm |= la.qk_norm;
      caps.v_norm |= la.v_norm;
      caps.sandwich_norm |= la.sandwich;
      caps.layer_scalar |= la.layer_scalar;
    }
    // Only rewrite the stack-wide geometry when the model actually uses
    // per-layer metadata to vary it.  For every other architecture (MLA,
    // hybrid-SSM, plain dense) these scalar fields are consumed by dedicated
    // code paths and must stay exactly as parsed -- e.g. MLA checkpoints carry
    // attention.value_length = 0 and the MLA cache sizing relies on that.
    const bool per_layer_geom = kv_per_layer || swa_per_layer;
    if (kv_per_layer) a.n_head_kv = max_kv;
    if (per_layer_geom) {
      a.key_length = max_kd;
      a.value_length = max_vd;
    }
    caps.logit_softcap = meta_float(g, m.arch, "final_logit_softcapping", 0.0);
    caps.attn_logit_softcap = meta_float(g, m.arch, "attention.logit_softcapping", 0.0);
    // Family traits (metadata / tensor structure, not the architecture name):
    // models with per-head qk-norm use a GELU-tanh MLP, sqrt(hidden) embedding
    // scaling and an unscaled (1.0) attention logit scale.
    const bool gemma_trait = caps.qk_norm || caps.logit_softcap != 0.0;
    caps.gelu_mlp = gemma_trait;
    caps.embed_scale = gemma_trait ? std::sqrt(static_cast<double>(m.hidden)) : 1.0;
    caps.attn_scale = caps.qk_norm ? 1.0 : 0.0;
    // GGUF gemma4 proportional rope: one global rope_freqs.weight table holds the
    // per-pair frequency divisors for the full-attention layers (llama.cpp binds
    // it to every !is_swa layer, TENSOR_DUPLICATED).  Expected length is
    // (max full-layer rotary dim)/2 for an F32 tensor.  Validate here rather than
    // at load time so `engine info` explains a malformed file.
    {
      const std::string rf_name = "rope_freqs.weight";
      const gguf::TensorInfo* rf = nullptr;
      for (const auto& t : g.tensors)
        if (t.name == rf_name) { rf = &t; break; }
      if (rf) {
        int full_rot = 0;
        for (int i = 0; i < m.n_layer; ++i) {
          const auto& la = m.layers[i].attn;
          if (la.window != 0) continue;
          full_rot = std::max(full_rot, la.rope_dim > 0 ? la.rope_dim : la.key_dim);
        }
        const size_t want = full_rot > 0 ? static_cast<size_t>(full_rot) / 2 : 0;
        if (rf->type_name == "F32" && want > 0 && rf->n_elements == want) {
          caps.rope_freq_factors = true;
        } else {
          m.warnings.push_back(
              "rope_freqs.weight present but unusable (type " + rf->type_name + ", " +
              std::to_string(rf->n_elements) + " elements; expected F32 " +
              std::to_string(want) + " = full-layer rope_dim/2); ignoring it");
        }
      }
    }
    // Only metadata that the legacy dense path genuinely cannot express turns the
    // generic path on.  A bare per-head qk-norm tensor set is NOT enough: several
    // already-supported architectures (qwen3 family) carry attn_q_norm/attn_k_norm
    // and have their own loaders.  In particular never take over hybrid-SSM / MLA /
    // MoE models here; they keep their existing, verified code paths.
    caps.extended = caps.per_layer_attn || caps.sandwich_norm || caps.layer_scalar ||
                    caps.logit_softcap != 0.0 || caps.rope_freq_factors;
    if (caps.extended) {
      int n_eq_v = 0;
      for (int i = 0; i < m.n_layer; ++i)
        if (!m.layers[i].attn.has_v) ++n_eq_v;
      m.warnings.push_back(
          "extended attention path enabled: per_layer=" +
          std::to_string(static_cast<int>(caps.per_layer_attn)) + " qk_norm=" +
          std::to_string(static_cast<int>(caps.qk_norm)) + " v_norm=" +
          std::to_string(static_cast<int>(caps.v_norm)) + " sandwich=" +
          std::to_string(static_cast<int>(caps.sandwich_norm)) + " layer_scalar=" +
          std::to_string(static_cast<int>(caps.layer_scalar)) + " k_eq_v_layers=" +
          std::to_string(n_eq_v) + " logit_softcap=" +
          std::to_string(caps.logit_softcap) + " rope_freq_factors=" +
          std::to_string(static_cast<int>(caps.rope_freq_factors)));
    }
  }

  m.vocab = m.tok.n_vocab;
  if (m.vocab == 0) m.vocab = static_cast<uint64_t>(meta_int(g, m.arch, "vocab_size"));
  if (mo.n_expert && mo.n_expert_used > mo.n_expert)
    m.warnings.push_back("expert_used_count > expert_count");
  return m;
}

std::vector<std::string> validate(Model& m) {
  std::vector<std::string> errs;
  if (m.n_layer <= 0) errs.push_back("n_layer <= 0");
  if (m.tensors.empty()) errs.push_back("no tensors");

  // layers with no tensors
  std::vector<int> seen(static_cast<size_t>(m.n_layer), 0);
  for (const auto& t : m.tensors)
    if (t.layer >= 0 && t.layer < m.n_layer) seen[t.layer] = 1;
  std::string missing;
  for (int i = 0; i < m.n_layer; ++i)
    if (!seen[i]) missing += std::to_string(i) + " ";
  if (!missing.empty()) errs.push_back("layers with no tensors: " + missing);

  // expert table completeness — both layouts must validate:
  //   fused_3d (GGUF): 3 tensors/layer, leading dim == n_expert
  //   split_per_expert (HF): n_expert*3 tensors/layer
  if (m.moe.n_expert) {
    std::map<int, std::vector<const TensorDesc*>> per_layer;
    std::map<int, int> routers;
    for (const auto& t : m.tensors) {
      if (t.role == Role::EXPERT && t.layer >= 0) per_layer[t.layer].push_back(&t);
      if (t.role == Role::ROUTER && t.layer >= 0) routers[t.layer] = 1;
    }
    for (const auto& [layer, _] : routers) {
      auto it = per_layer.find(layer);
      const std::vector<const TensorDesc*> empty;
      const auto& ts = it == per_layer.end() ? empty : it->second;
      bool fused = false;
      for (const auto* t : ts)
        if (t->shape.size() == 3) fused = true;
      if (fused) {
        if (ts.size() != 3) {
          errs.push_back("layer " + std::to_string(layer) +
                         ": fused MoE layer must have 3 tensors, got " +
                         std::to_string(ts.size()));
          continue;
        }
        for (const auto* t : ts)
          if (static_cast<int>(t->shape[0]) != m.moe.n_expert)
            errs.push_back(t->name + ": expert dim " + std::to_string(t->shape[0]) +
                           " != n_expert " + std::to_string(m.moe.n_expert));
      } else {
        const size_t expected = static_cast<size_t>(m.moe.n_expert) * 3;
        if (ts.size() != expected)
          errs.push_back("layer " + std::to_string(layer) + ": split MoE layer must have " +
                         std::to_string(expected) + " tensors (n_expert*3), got " +
                         std::to_string(ts.size()));
        for (const auto* t : ts)
          if (t->shape.size() != 2)
            errs.push_back(t->name + ": expected 2-D expert matrix");
      }
    }
    for (const auto& [layer, _] : per_layer)
      if (!routers.count(layer))
        errs.push_back("layer " + std::to_string(layer) +
                       ": expert tensors present but no ffn_gate_inp router");
    if (routers.empty()) errs.push_back("moe.n_expert > 0 but no router tensors found");
  }

  // dense FFN triplet consistency
  if (m.ffn) {
    std::map<std::string, const TensorDesc*> by_name;
    for (const auto& t : m.tensors) by_name[t.name] = &t;
    for (int i = 0; i < m.n_layer; ++i) {
      auto pg = by_name.find("blk." + std::to_string(i) + ".ffn_gate.weight");
      auto pu = by_name.find("blk." + std::to_string(i) + ".ffn_up.weight");
      auto pd = by_name.find("blk." + std::to_string(i) + ".ffn_down.weight");
      if (pg == by_name.end() || pu == by_name.end()) continue;
      if (pg->second->shape != pu->second->shape)
        errs.push_back("layer " + std::to_string(i) + ": gate/up shape mismatch");
      if (pd != by_name.end() && pg->second->shape.size() == 2 &&
          pd->second->shape.size() == 2 &&
          pd->second->shape[1] != pg->second->shape[0])
        errs.push_back("layer " + std::to_string(i) + ": down/gate shape mismatch");
    }
  }

  // quant block divisibility
  for (const auto& t : m.tensors)
    if (t.quant.block > 1 && t.n_elements % static_cast<uint64_t>(t.quant.block) != 0)
      errs.push_back(t.name + ": element count not divisible by block");

  // attention invariants
  if (m.attn.n_head && m.attn.n_head_kv && m.attn.n_head % m.attn.n_head_kv != 0)
    errs.push_back("n_head not divisible by n_head_kv");

  // ---- geometry / metadata consistency (fail early, explain what is wrong) ----
  if (m.caps.extended) {
    for (int i = 0; i < m.n_layer; ++i) {
      const auto& la = m.layers[i].attn;
      const std::string lp = "layer " + std::to_string(i) + ": ";
      if (la.n_kv <= 0)
        errs.push_back(lp + "per-layer head_count_kv <= 0 (cannot size the KV cache)");
      else if (m.attn.n_head > 0 && m.attn.n_head % la.n_kv != 0)
        errs.push_back(lp + "head_count " + std::to_string(m.attn.n_head) +
                       " not divisible by per-layer head_count_kv " +
                       std::to_string(la.n_kv));
      if (la.key_dim <= 0 || la.key_dim % 2 != 0)
        errs.push_back(lp + "per-layer key/value head dim " + std::to_string(la.key_dim) +
                       " must be positive and even (rope pairs)");
      if (la.value_dim <= 0)
        errs.push_back(lp + "per-layer value head dim <= 0");
      if (la.rope_dim <= 0 || la.rope_dim > la.key_dim || la.rope_dim % 2 != 0)
        errs.push_back(lp + "per-layer rotary dim " + std::to_string(la.rope_dim) +
                       " must be positive, even and <= head dim " +
                       std::to_string(la.key_dim));
      if (la.rope_base <= 0.0)
        errs.push_back(lp + "per-layer rope base must be > 0");
      // The extended path writes K and V with one shared row stride kernel; this
      // is exact for every current gguf layout (gemma4 included) and is reported
      // instead of silently mis-striding when a future model differs.
      if (la.key_dim != la.value_dim)
        errs.push_back(lp + "extended path requires key_dim == value_dim (" +
                       std::to_string(la.key_dim) + " vs " +
                       std::to_string(la.value_dim) + ")");
    }
  }
  if (m.attn.n_head > 0) {
    if (m.attn.n_head_kv <= 0)
      errs.push_back("attention.head_count_kv missing or 0 (head_count=" +
                     std::to_string(m.attn.n_head) + ")");
    if (m.attn.key_length <= 0)
      errs.push_back("attention.key_length missing or 0 (cannot size the attention buffers)");
    if (m.attn.key_length > 0 && m.attn.key_length % 2 != 0 && m.attn.variant != "mla")
      errs.push_back("attention.key_length " + std::to_string(m.attn.key_length) +
                     " is odd; the rope kernels rotate values in pairs");
    if (m.attn.variant != "mla" && m.ssm.variant == "none" && !m.caps.per_layer_attn &&
        m.attn.key_length > 0 && m.hidden > 0 &&
        m.attn.n_head * m.attn.key_length != m.hidden)
      m.warnings.push_back("n_head*key_length = " +
                           std::to_string(m.attn.n_head * m.attn.key_length) + " != hidden " +
                           std::to_string(m.hidden) +
                           " (check attention.key_length metadata; the q/k/v projections must "
                           "match it)");
    if (m.attn.rope_dim > 0) {
      if (m.attn.rope_dim % 2 != 0)
        errs.push_back("rope.dimension_count " + std::to_string(m.attn.rope_dim) +
                       " is odd; rotary embeddings need pairs");
      if (m.attn.key_length > 0 && m.attn.rope_dim > m.attn.key_length)
        errs.push_back("rope.dimension_count " + std::to_string(m.attn.rope_dim) +
                       " > attention.key_length " + std::to_string(m.attn.key_length));
    }
  } else if (m.ssm.variant == "none") {
    errs.push_back("attention.head_count missing/0 and no SSM metadata: no attention "
                   "architecture detected");
  }
  if (m.attn.variant == "mla") {
    if (m.attn.kv_lora_rank <= 0)
      errs.push_back("MLA attention without attention.kv_lora_rank");
    if (m.attn.value_length <= 0)
      errs.push_back("MLA attention without attention.value_length");
    if (m.attn.rope_dim <= 0)
      errs.push_back("MLA attention without rope.dimension_count (the RoPE half of the "
                     "q/k vectors cannot be built)");
  }
  if (m.attn.sliding_window > 0)
    m.warnings.push_back("attention.sliding_window=" +
                         std::to_string(m.attn.sliding_window) +
                         " is parsed but not implemented by the kernels: full attention is "
                         "used, so the KV cache is larger than the model requires");
  if (m.moe.n_expert > 0) {
    // The device-side router (moe_route) formats its routing probabilities in 256 slots
    // of shared memory; wider expert tables are served by the host router instead (one
    // host round-trip per MoE layer/token, correct but slower).  Refuse only absurdly
    // large tables that the host router/logits buffer cannot sensibly hold.
    if (m.moe.n_expert > 4096) {
      errs.push_back("expert_count " + std::to_string(m.moe.n_expert) +
                     " > 4096: beyond the supported expert-table size");
    } else if (m.moe.n_expert > 256) {
      m.warnings.push_back(
          "expert_count " + std::to_string(m.moe.n_expert) +
          " > 256: the device-side router is disabled for this model; host-side routing "
          "is used (correct, slower by one host round-trip per MoE layer)");
    }
    if (m.moe.n_expert_used <= 0)
      errs.push_back("expert_used_count missing/0 (MoE top-k unknown)");
    else if (m.moe.n_expert_used > m.moe.n_expert)
      errs.push_back("expert_used_count " + std::to_string(m.moe.n_expert_used) +
                     " > expert_count " + std::to_string(m.moe.n_expert));
    if (m.moe.expert_ffn_length <= 0)
      m.warnings.push_back("expert_feed_forward_length missing: falls back to the dense "
                           "FFN length for expert shapes");
  }
  if (m.ssm.variant == "gated_delta") {
    if (m.ssm.conv_kernel <= 1)
      errs.push_back("ssm.conv_kernel must be > 1 for the gated delta net");
    if (m.ssm.n_value_head <= 0 || m.ssm.value_head_dim <= 0 || m.ssm.key_head_dim <= 0)
      errs.push_back("SSM head geometry missing (group_count/state_size/inner_size)");
    else if (m.ssm.value_dim != m.ssm.n_value_head * m.ssm.value_head_dim)
      errs.push_back("SSM value_dim " + std::to_string(m.ssm.value_dim) +
                     " != n_value_head*value_head_dim");
    else if (m.ssm.key_dim != m.ssm.n_key_head * m.ssm.key_head_dim)
      errs.push_back("SSM key_dim " + std::to_string(m.ssm.key_dim) +
                     " != n_key_head*key_head_dim");
  }
  if (m.vocab == 0)
    errs.push_back("vocabulary size missing (tokenizer.ggml.tokens / vocab_size)");

  // tensor roles: an unknown naming/layout convention must not fall through silently.
  {
    bool have_embed = false, have_attn = false, have_ffn = false;
    for (const auto& t : m.tensors) {
      if (t.role == Role::EMBED) have_embed = true;
      if (t.role == Role::ATTN) have_attn = true;
      if (t.role == Role::FFN_DENSE || t.role == Role::EXPERT) have_ffn = true;
    }
    if (!have_embed)
      errs.push_back("no embedding tensor recognized (expected token_embd.weight for GGUF "
                     "or model.embed_tokens.weight for safetensors): unsupported model "
                     "layout or wrong file");
    if (!have_attn && m.ssm.variant == "none")
      errs.push_back("no attention tensors recognized (blk.N.attn_*): unsupported model "
                     "layout or architecture");
    if (!have_ffn)
      errs.push_back("no FFN/expert tensors recognized (blk.N.ffn_*): unsupported model "
                     "layout or architecture");
  }

  // quant types wired to the GEMV/dequant kernels.  Unknown types must fail here,
  // not deep inside the loader with a partial model loaded.
  {
    const char* kSupported[] = {"F32", "F16", "BF16", "Q4_K", "Q5_K", "Q6_K",
                                "Q8_0", "Q5_0", "Q5_1"};
    std::set<std::string> unsupported;
    for (const auto& t : m.tensors) {
      if (t.role == Role::OTHER || t.role == Role::NORM) continue;
      bool ok = false;
      for (const char* s : kSupported)
        if (t.dtype == s) {
          ok = true;
          break;
        }
      if (!ok) unsupported.insert(t.dtype);
    }
    if (!unsupported.empty()) {
      std::string names;
      for (const auto& t : unsupported)
        names += std::string(names.empty() ? "" : ", ") + t;
      std::string supported;
      for (const char* s : kSupported)
        supported += std::string(supported.empty() ? "" : ", ") + s;
      errs.push_back("unsupported quant type(s): " + names +
                     ". This engine implements: " + supported +
                     ". Convert the model with llama.cpp's quantize tool (e.g. Q4_K_M or "
                     "Q5_K_M) and try again.");
    }
  }

  // tensor span coverage: overlaps and file bounds
  std::vector<std::pair<uint64_t, std::pair<uint64_t, std::string>>> spans;
  for (const auto& t : m.tensors)
    spans.push_back({t.file_offset, {t.file_offset + t.nbytes, t.name}});
  std::sort(spans.begin(), spans.end());
  for (size_t i = 1; i < spans.size(); ++i) {
    const auto& prev = spans[i - 1];
    const auto& cur = spans[i];
    if (cur.first < prev.second.first)
      errs.push_back("tensor overlap: " + prev.second.second + " > " + cur.second.second);
  }
  if (!spans.empty() && m.file_size && spans.back().second.first > m.file_size)
    errs.push_back("tensor data exceeds file size");

  // vocab vs embedding rows
  for (const auto& t : m.tensors) {
    if (t.role == Role::EMBED && t.shape.size() == 2 && m.vocab &&
        t.shape[0] != m.vocab) {
      errs.push_back("embedding rows " + std::to_string(t.shape[0]) + " != vocab " +
                     std::to_string(m.vocab));
      break;
    }
  }
  return errs;
}

// --------------------------------------------------------------------------
// safetensors / HF config -> IR  (split-expert layout)
// --------------------------------------------------------------------------

namespace {

Role classify_hf(const std::string& n) {
  if (ends_with(n, "embed_tokens.weight")) return Role::EMBED;
  if (ends_with(n, "lm_head.weight")) return Role::OUTPUT;
  if (contains(n, ".mlp.shared_experts.") || contains(n, ".shared_experts."))
    return Role::SHARED_EXPERT;
  if (contains(n, ".mlp.experts.") || contains(n, ".experts.")) return Role::EXPERT;
  if (contains(n, ".mlp.gate.weight") || contains(n, ".router.")) return Role::ROUTER;
  if (contains(n, ".self_attn.") || contains(n, ".attention.")) return Role::ATTN;
  {
    std::string lower = n;
    for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (contains(lower, "norm")) return Role::NORM;
  }
  if (contains(n, ".mlp.")) return Role::FFN_DENSE;
  return Role::OTHER;
}

int layer_of_hf(const std::string& name) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (start <= name.size()) {
    size_t dot = name.find('.', start);
    if (dot == std::string::npos) {
      parts.push_back(name.substr(start));
      break;
    }
    parts.push_back(name.substr(start, dot - start));
    start = dot + 1;
  }
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    if (parts[i] != "layers") continue;
    const std::string& s = parts[i + 1];
    if (s.empty()) continue;
    bool digits = true;
    for (char c : s)
      if (c < '0' || c > '9') digits = false;
    if (digits) return std::atoi(s.c_str());
  }
  return -1;
}

double cfg_num(const jsonp::Value& c, std::initializer_list<const char*> keys, double fb) {
  for (auto k : keys) {
    const jsonp::Value* v = c.find(k);
    if (!v) continue;
    if (v->is_number()) return v->num;
    if (v->type == jsonp::Value::Type::Bool) return v->b ? 1.0 : 0.0;
  }
  return fb;
}

std::string cfg_str(const jsonp::Value& c, std::initializer_list<const char*> keys,
                    const std::string& fb) {
  for (auto k : keys) {
    const jsonp::Value* v = c.find(k);
    if (v && v->is_string()) return v->str;
    if (v && v->is_array() && !v->arr.empty() && v->arr[0].is_string()) return v->arr[0].str;
  }
  return fb;
}

}  // namespace

Model build_ir_from_hf(const std::string& dir) {
  std::filesystem::path root(dir);
  std::filesystem::path cfg_path = root / "config.json";
  if (!std::filesystem::exists(cfg_path))
    throw std::runtime_error(dir + ": config.json not found");

  std::ifstream cf(cfg_path, std::ios::binary);
  if (!cf) throw std::runtime_error("cannot open " + cfg_path.string());
  std::string cfg_text((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
  jsonp::Value cfg = jsonp::parse(cfg_text);

  st::Shards shards = st::load_dir(dir);
  for (const auto& [name, _] : shards.tensors) {
    static const char* bad[] = {"qweight", "qzeros", "scales", "g_idx", "weight_scale",
                                "weight_zero_point"};
    for (const char* s : bad) {
      if (ends_with(name, s))
        throw std::runtime_error(name + ": quantized safetensors container (AWQ/GPTQ) is not "
                                        "supported in v0; use BF16/F16/FP8 safetensors");
    }
  }

  Model m;
  m.container = "safetensors";
  m.path = dir;
  m.file_size = shards.total_bytes;
  m.arch = cfg_str(cfg, {"model_type", "architectures"}, "unknown");

  m.n_layer = static_cast<int>(cfg_num(cfg, {"num_hidden_layers", "n_layer"}, 0));
  m.hidden = static_cast<int>(cfg_num(cfg, {"hidden_size", "n_embd"}, 0));
  m.ffn = static_cast<int>(cfg_num(cfg, {"intermediate_size", "n_inner"}, 0));
  m.vocab = static_cast<uint64_t>(cfg_num(cfg, {"vocab_size"}, 0));
  m.ctx_train = static_cast<int64_t>(
      cfg_num(cfg, {"max_position_embeddings", "n_positions"}, 0));
  if (m.n_layer <= 0 || m.hidden <= 0)
    throw std::runtime_error(dir + ": config.json missing num_hidden_layers/hidden_size");

  auto& a = m.attn;
  a.n_head = static_cast<int>(cfg_num(cfg, {"num_attention_heads", "n_head"}, 0));
  a.n_head_kv = static_cast<int>(
      cfg_num(cfg, {"num_key_value_heads", "n_head_kv"}, a.n_head));
  const int head_dim = static_cast<int>(cfg_num(
      cfg, {"head_dim"}, a.n_head ? m.hidden / a.n_head : 0));
  a.key_length = head_dim;
  a.value_length = head_dim;
  a.rope_freq_base = cfg_num(cfg, {"rope_theta", "rope_embedding_base"}, 10000.0);
  a.rope_dim = static_cast<int>(cfg_num(cfg, {"partial_rotary_factor"}, 1.0) * head_dim);
  a.sliding_window = static_cast<int>(cfg_num(cfg, {"sliding_window"}, 0));
  a.kv_lora_rank = static_cast<int>(cfg_num(cfg, {"kv_lora_rank"}, 0));
  if (a.kv_lora_rank) a.variant = "mla";
  else if (a.n_head_kv > 0 && a.n_head > 0)
    a.variant = a.n_head_kv == 1 ? "mqa" : (a.n_head_kv < a.n_head ? "gqa" : "mha");

  auto& mo = m.moe;
  mo.n_expert = static_cast<int>(
      cfg_num(cfg, {"num_experts", "num_local_experts", "n_routed_experts"}, 0));
  mo.n_expert_used = static_cast<int>(
      cfg_num(cfg, {"num_experts_per_tok", "n_group_used", "num_experts_per_token"}, 0));
  mo.n_shared = static_cast<int>(
      cfg_num(cfg, {"n_shared_experts", "num_shared_experts"}, 0));
  mo.norm_topk = cfg_num(cfg, {"norm_topk_prob", "expert_weights_norm"}, 0) != 0;
  mo.expert_ffn_length = static_cast<int>(
      cfg_num(cfg, {"moe_intermediate_size", "expert_intermediate_size"}, 0));
  mo.layout = mo.n_expert ? "split_per_expert" : "none";

  m.tok.model = "hf-bpe";
  m.tok.pre = cfg_str(cfg, {"tokenizer_class"}, "");
  m.tok.n_vocab = m.vocab;
  m.tok.eos_id = static_cast<int64_t>(cfg_num(cfg, {"eos_token_id"}, -1));
  m.tok.chat_template = std::filesystem::exists(root / "chat_template.jinja") ||
                        cfg.find("chat_template") != nullptr;

  for (const auto& [name, t] : shards.tensors) {
    TensorDesc d;
    d.name = name;
    d.shape = t.shape;
    d.dtype = t.dtype;
    d.quant.scheme = t.dtype;
    d.quant.block = 1;
    d.quant.bytes_per_block = static_cast<int>(st::dtype_bytes(t.dtype));
    d.role = classify_hf(name);
    d.layer = layer_of_hf(name);
    d.file_offset = t.data_start + t.begin;
    d.nbytes = t.nbytes();
    d.n_elements = t.n_elements();
    m.quant_mix[d.dtype] += d.nbytes;
    m.tensors.push_back(std::move(d));
  }

  m.layers.resize(static_cast<size_t>(m.n_layer));
  for (int i = 0; i < m.n_layer; ++i) m.layers[i].index = i;
  for (const auto& t : m.tensors) {
    if (t.layer < 0 || t.layer >= m.n_layer) continue;
    auto& L = m.layers[t.layer];
    ++L.n_tensors;
    if (t.role == Role::EXPERT) ++L.n_expert_tensors;
  }
  if (m.tensors.empty()) m.warnings.push_back("no tensors found");
  return m;
}

}  // namespace ir
