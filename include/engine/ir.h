// Model IR: the single representation every container maps into.
// Mirrors pyref/ir.py — the two must agree field by field (spec §7).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "engine/gguf.h"

namespace ir {

enum class Role { EMBED, OUTPUT, ATTN, FFN_DENSE, EXPERT, SHARED_EXPERT, ROUTER, NORM, OTHER };

const char* role_name(Role r);
Role classify(const std::string& name);
int layer_of(const std::string& name);

struct QuantDesc {
  std::string scheme;
  int block = 1;
  int bytes_per_block = 0;
  double bpp() const { return block > 0 ? bytes_per_block * 8.0 / block : 0.0; }
};

struct TensorDesc {
  std::string name;
  std::vector<uint64_t> shape;
  std::string dtype;
  Role role = Role::OTHER;
  int layer = -1;
  uint64_t file_offset = 0;
  uint64_t nbytes = 0;
  uint64_t n_elements = 0;
  QuantDesc quant;
};

struct AttnDesc {
  std::string variant = "gqa";
  int n_head = 0, n_head_kv = 0, key_length = 0, value_length = 0;
  double rope_freq_base = 10000.0;
  int rope_dim = 0, sliding_window = 0, full_attention_interval = 0;
  int kv_lora_rank = 0, leading_dense_blocks = 0;
};

struct MoEDesc {
  int n_expert = 0, n_expert_used = 0, n_shared = 0;
  std::string router_kind = "topk_softmax";
  bool norm_topk = false;
  int expert_ffn_length = 0;
  std::string layout = "none";   // none | fused_3d (GGUF) | split_per_expert (HF)
};

// Per-layer attention geometry.  Populated generically from metadata structure:
// scalar metadata -> the same value on every layer, array-valued metadata
// (`attention.head_count_kv`) and `*_swa` variants selected by
// `attention.sliding_window_pattern` -> true per-layer values.
struct LayerAttn {
  int n_kv = 0;
  int key_dim = 0;
  int value_dim = 0;
  int rope_dim = 0;
  double rope_base = 0.0;
  int window = 0;          // 0 = full causal attention
  bool has_v = true;       // false => attention_k_eq_v (V reuses the K projection)
  bool qk_norm = false;    // per-head RMSNorm weights on q and k
  bool v_norm = false;     // weightless RMSNorm on V
  bool sandwich = false;   // post_attention_norm / post_ffw_norm present
  bool layer_scalar = false;  // layer_output_scale present
};

struct LayerDesc {
  int index = 0;
  int n_tensors = 0, n_expert_tensors = 0;
  // qwen35 hybrid stacks: "attn" (full attention) or "linear" (gated delta net)
  std::string kind = "attn";
  LayerAttn attn;
};

// Model-level capabilities derived from metadata/tensor structure (never from
// the architecture name).  The engine enables the generic extended layer path
// when `extended` is set and applies the individual ops only when present.
struct ArchCaps {
  bool extended = false;      // use the generic per-layer-capable dense path
  bool per_layer_attn = false;
  bool qk_norm = false;
  bool v_norm = false;
  bool sandwich_norm = false;
  bool layer_scalar = false;
  bool gelu_mlp = false;
  double attn_scale = 0.0;    // >0 explicit scale, 0 = legacy 1/sqrt(head_dim)
  double embed_scale = 1.0;   // multiplier applied to the token embeddings
  double logit_softcap = 0.0; // final logit softcapping (0 = off)
  double attn_logit_softcap = 0.0;
};

// Gated DeltaNet parameters (qwen35 linear attention layers).
struct SSMDesc {
  std::string variant = "none";   // none | gated_delta
  int conv_kernel = 0;
  int n_key_head = 0, n_value_head = 0;
  int key_head_dim = 0, value_head_dim = 0;
  int key_dim = 0, value_dim = 0, conv_dim = 0;
  int time_step_rank = 0;
};

struct TokenizerDesc {
  std::string model = "unknown", pre;
  uint64_t n_vocab = 0;
  int64_t eos_id = -1;
  bool chat_template = false;
};

class Model {
 public:
  std::string arch, container, path;
  uint64_t file_size = 0;
  int n_layer = 0, hidden = 0, ffn = 0;
  uint64_t vocab = 0;
  int64_t ctx_train = 0;
  int mtp_layers = 0;
  // Multi-token-prediction ("nextn") block: a full attention+FFN layer past the
  // transformer stack that drafts token p+2 from (h_p, emb(t_{p+1})).  Only the
  // block index and a layer descriptor are needed; the tensor descriptors stay in
  // `tensors` with their original roles.
  int mtp_block = -1;
  LayerDesc mtp_desc;
  std::map<std::string, uint64_t> quant_mix;
  AttnDesc attn;
  MoEDesc moe;
  SSMDesc ssm;
  TokenizerDesc tok;
  ArchCaps caps;
  std::vector<LayerDesc> layers;
  std::vector<TensorDesc> tensors;
  std::vector<std::string> warnings;

  uint64_t total_bytes() const;
  uint64_t fixed_bytes() const;
  uint64_t expert_bytes() const;
  std::map<int, uint64_t> expert_bytes_per_layer() const;
  uint64_t kv_bytes_per_token(int bpe = 2) const;
};

Model build_from_gguf(const gguf::Model& g);

// safetensors directory (config.json + *.safetensors[/index.json])
Model build_ir_from_hf(const std::string& dir);

// Returns error strings; empty vector means the IR is valid (spec §3.3).
std::vector<std::string> validate(Model& m);

}  // namespace ir
