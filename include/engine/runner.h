// Dense (non-MoE) llama-like runner: embedding -> N layers -> norm -> lm_head.
// Supports both containers: HF safetensors (model.layers.N...) and GGUF (blk.N...).
// v0 keeps weights in f32 on the GPU (dequantized at load via the CUDA kernels).
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "engine/cuda.h"
#include "engine/ir.h"
#include "engine/kernel_select.h"

namespace runner {

enum class Container { Safetensors, Gguf };

// Auto-detect the container from the path: a directory is a HF safetensors
// checkpoint, a file is GGUF when it starts with the GGUF magic (the extension is
// only used as a fallback, so model files can be named anything).
Container detect_container(const std::string& path);

class DenseRunner {
 public:
  DenseRunner(const std::string& path, int max_seq = 256,
              Container container = Container::Safetensors);
  ~DenseRunner();
  DenseRunner(const DenseRunner&) = delete;
  DenseRunner& operator=(const DenseRunner&) = delete;

  // Returns logits [n_tokens, vocab] in host memory.
  // start_pos > 0 continues from an existing KV cache (decode step).
  // `last_row_only` skips the final norm + lm_head for all but the last token,
  // which is all generation needs (and avoids a n*vocab logits download).
  std::vector<float> forward(const std::vector<int>& tokens, int start_pos = 0,
                             bool last_row_only = false);

  // Top-k of the last row, computed on the device.  A single decode step only needs
  // the arg-max (greedy) or k best candidates (sampling), but the logits row is
  // 0.4-1 MB and the pinned D2H of it costs ~0.27 ms/token on this WDDM VM.  The
  // decode head therefore reduces the row with two tiny kernels and only k
  // (index,value) pairs are copied back.  Returns an empty vector when the fast
  // path is unavailable (e.g. prefill); the caller must then fall back to forward().
  struct TopKResult {
    std::vector<int> idx;
    std::vector<float> val;
  };
  std::vector<float> forward_topk(const std::vector<int>& tokens, int start_pos, int k,
                                  TopKResult* out);
  // Device-side top-k candidates for a sampling server: returns k (index, logit) pairs
  // straight from the device head (no full-vocab logits row crosses PCIe).  False when
  // the device head is unavailable or k exceeds head_topk_max.
  bool forward_topk_candidates(const std::vector<int>& tokens, int start_pos, int k,
                               std::vector<int>* ids, std::vector<float>* logits);

  const ir::Model& ir() const { return ir_; }
  // test hook: contents of the MLA k-cache for one layer
  std::vector<float> debug_mla_cache(int layer);

  // experimental switches (used by the layout sweep tool): q=0 perhead / 1 global /
  // 2 global with the rope half first; kv=0 global / 1 perhead / 2 global v-first /
  // 3 perhead v-first.
  void set_mla_layout(int q_mode, int kv_mode) {
    mla_q_global_ = q_mode == 1 || q_mode == 2;
    mla_q_pe_first_ = q_mode == 2;
    mla_kv_perhead_ = kv_mode == 1 || kv_mode == 3;
    mla_kv_vfirst_ = kv_mode == 2 || kv_mode == 3;
  }
  void set_rope_mode(int mode);   // 0 = yarn, 1 = plain rope, 2 = yarn without mscale
  void set_router(const std::string& kind, bool norm_topk) {
    ir_.moe.router_kind = kind;
    ir_.moe.norm_topk = norm_topk;
  }

  int vocab() const { return vocab_; }
  // Number of tokens per prefill chunk (rows_).  forward() splits prompts longer
  // than this into chunks of exactly this size; the serve prefix cache aligns its
  // snapshot/split positions to multiples of it so a cached prefill performs the
  // *same* chunk sequence as a full prefill (bit-exact logits).
  int prefill_chunk() const { return static_cast<int>(rows_); }
  // Context length actually allocated (max_seq_).  With --ctx 0 / serve auto the
  // loader derives and clamps this from measured free VRAM, so the server reports
  // the real value instead of the requested 0.
  int ctx() const { return max_seq_; }
  // MoE expert placement (valid after load): number of MoE layers, and how many of them
  // keep some experts on the host CPU path.  serve uses this to skip the token-warmup on
  // CPU-expert models, where a 1024-token prefill costs >100 s of startup (35B: 133 s).
  int n_moe_layers() const;
  int cpu_moe_layers() const;

  // ---- MTP / speculative decoding -------------------------------------------
  // Available when the GGUF carries a `nextn` (multi-token-prediction) block and the
  // loaded MTP weights are usable (`mtp` config).  Nothing here is model-name based:
  // the block is found from the presence of `blk.N.nextn.*` tensors (ir.mtp_block).
  bool mtp_available() const { return mtp_ok_; }
  struct SpecStats {
    int k = 0;              // drafts per round
    int rounds = 0;
    int drafted = 0;        // total draft tokens proposed
    int accepted = 0;       // total draft tokens accepted
    int verify_fwd = 0;     // batch forwards used for verification
    int fell_back = 0;      // rounds that used the serial fallback
    double draft_ms = 0;    // time spent drafting
    double verify_ms = 0;   // time spent verifying
    double total_ms = 0;    // prefill + all rounds
    double prefill_ms = 0;  // target-model prefill + MTP context pass
    double decode_ms = 0;   // rounds after the first (steady state)
    int decode_tokens = 0;  // tokens emitted in that window
  };
  // Greedy speculative generation.  Prefills `prompt`, then emits exactly `n_gen`
  // tokens; the output is token-for-token identical to the plain greedy loop because
  // a draft is only kept when the target model's own arg-max agrees with it.
  std::vector<int> spec_generate(const std::vector<int>& prompt, int n_gen, SpecStats* st);

  // ---- incremental greedy speculative decoding (used by `engine serve --mtp`) ----
  // spec_begin() must be called right after a SINGLE-CALL prompt prefill whose
  // residual rows are still in x_ (row `last_row` produced `first_token`).  It
  // seeds the MTP attention history and the hidden state used by the draft chain.
  // spec_round() then drafts up to k tokens, verifies them with one batched
  // forward, appends [pending, accepted drafts...] to `emit`, and updates the
  // waiting token to the bonus token that follows the accepted prefix.
  // The emitted sequence is token-for-token identical to plain greedy decoding
  // (a draft is kept only when the target model's own arg-max agrees).
  // Returns false when MTP is unavailable or the context is nearly exhausted;
  // the caller then falls back to the plain single-token loop.
  bool spec_begin(const std::vector<int>& prompt_ids, int first_token, int last_row);
  bool spec_round(int k, std::vector<int>* emit);
  void spec_end() { spec_active_ = false; }
  int spec_pending() const { return spec_pending_; }

  // P0 self-test hook: draft the token that should follow `tok` at position `pos`,
  // given the residual stream that produced it (x_ row `row`).  Returns the draft id
  // and, when `h_out` is given, the MTP block output hidden state (chain drafting).
  int mtp_draft(int tok, int pos, int row);
  // Draft accuracy instrumentation: run a plain greedy loop and compare each MTP
  // draft against the token the target model actually picks.  Prints a summary.
  void mtp_accuracy(const std::vector<int>& prompt, int n_gen, int verbose);

  // ---- prefix-cache state (single slot) ------------------------------------
  // Hybrid qwen35 models carry recurrent state (conv + gated-delta-net state) that
  // is NOT part of the attention KV cache.  A prefix-cache hit therefore has to
  // restore that state in addition to reusing the KV cache.  Pure-attention models
  // have no such state and only need the KV cache (pos = cached length), so the
  // save/restore calls below are cheap no-ops returning true for them.
  //
  // The server uses ONE slot: it snapshots the recurrent state right after the
  // conversation-history part of a prompt has been prefilled, and restores it when
  // the next request shares that history as a prefix (multi-turn chat).
  bool prefix_state_supported() const { return ir_.ssm.variant != "none"; }
  // Bytes required by one slot (0 = pure-attention model, no snapshot needed).
  size_t prefix_state_bytes() const;
  // Snapshot / restore the recurrent state of the sequence processed so far.
  // Returns false when the model is hybrid and the slot cannot be allocated
  // (out of VRAM) or an invalid slot is requested; callers then fall back to a
  // full prefill.
  bool prefix_state_save(int slot);
  bool prefix_state_restore(int slot);

  // --kernel-calibrate: micro-benchmark the GEMV kernels on this model's real
  // shapes and cache the winner per shape.  Returns the number of entries
  // written; verbose prints the measurement table.
  int calibrate_kernels(bool verbose);
 private:
  struct ExpertW {
    cuda::Buffer gate, up, down;
  };

  // Quantized tensor kept in VRAM as raw blocks (no f32 expansion).
  struct QTensor {
    cuda::Buffer data;
    cuda::Buffer bm;            // optional block-major copy for the decode GEMV
    bool has_bm = false;
    cuda::Buffer half_t;        // optional fp16 transpose for the TN prefill GEMM
    bool has_half_t = false;
    std::string type;
    int block_bytes = 0, block_elems = 0;
    int n_out = 0, n_in = 0;
    bool valid = false;
  };

  struct LayerW {
    cuda::Buffer ln1, q, k, v, o, ln2, gate, up, down;
    cuda::Buffer bq, bk, bv;   // optional Qwen2-style biases
    bool has_bias = false;
    cuda::Buffer kc, vc;       // [n_kv][max_seq][D]
    cuda::Buffer router;
    std::vector<float> router_host;   // host mirror for CPU-layer routing
    std::vector<ExpertW> experts;
    std::vector<QTensor> q_experts;   // [gate, up, down] per expert, quantized-resident
    bool is_moe = false;
    // MLA (deepseek2)
    cuda::Buffer wq_mla, wkva, wkvan, wkb, wo_mla;
    // quantized copies (keeping the MLA weights as f32 costs ~142 MB per layer)
    QTensor qwq_mla, qwkva, qwkb, qwo_mla;
    bool mla_quant = false;
    cuda::Buffer mla_kc, mla_vc;   // [n_head][max_seq][qk|v_dim]
    bool is_mla = false;
    // quantized-resident weights (GGUF)
    QTensor qq, qk, qv, qo, qg, qu, qd;
    bool is_quant = false;
    // optional f32 mirror of the quantized weights, used for batch (prefill) GEMMs
    bool has_f32 = false;
    // qwen35 hybrid stacks (gated delta net / qwen3-next attention)
    struct Q35W {
      cuda::Buffer conv_w;                 // [conv_dim * K] f32
      QTensor qkv, z, alpha, beta, out;    // linear-attention projections
      cuda::Buffer a_log, dt_bias, norm_w; // [n_value_head] / [value_head_dim]
      cuda::Buffer conv_state;             // [(K-1) * conv_dim]
      cuda::Buffer rec_state;              // [n_value_head][value_head_dim][key_head_dim]
      QTensor qp, kp, vp, op;              // full-attention projections
      cuda::Buffer q_norm, k_norm;         // per-head RMSNorm weights
      bool attn_kind = false;
    };
    Q35W q35;
    bool is_linear = false;
    // shared experts (DeepSeek style): always-active dense FFN inside a MoE layer
    cuda::Buffer sh_gate, sh_up, sh_down;
    QTensor qsh_gate, qsh_up, qsh_down;
    bool sh_quant = false;
    cuda::Buffer sh_gate_inp;   // [hidden] sigmoid gate for the shared expert
    bool has_shared = false;
    int sh_inter = 0;
    // host copies for CPU-placed experts (filled only when CPU experts are enabled)
    std::vector<std::vector<float>> host_gate, host_up, host_down;
    // quantized host bytes for CPU-placed experts (avoids f32 expansion)
    std::vector<unsigned char> h_gate_bytes, h_up_bytes, h_down_bytes;
    std::string h_type_g, h_type_u, h_type_d;
    int h_bb_g = 0, h_be_g = 0, h_bb_u = 0, h_be_u = 0, h_bb_d = 0, h_be_d = 0;
    int cpu_begin = -1;  // first expert index placed on the CPU (-1 = not decided)
    int all_ptr_idx = -1;  // index into all_gp_ (device pointer arrays)
    std::unique_ptr<cuda::Graph> layer_graph;   // per-layer CUDA graph
    bool layer_graph_ready = false;
    int layer_graph_state = 0;    // 0 = run normally, 1 = capture on next use, 2 = replay
    int layer_graph_bucket = -1;  // attention warp-count bucket baked into the graph
    // attention-only CUDA graph for CPU-placed MoE layers (one per position bucket)
    std::unique_ptr<cuda::Graph> attn_graph[3];
    int attn_graph_state[3] = {0, 0, 0};   // 0 = run normally, 1 = capture pending, 2 = replay
    int attn_graph_bucket = 0;
    // ---- generic extended attention path (metadata/tensor driven) ----------
    // Active on layers whose geometry/capabilities were derived from per-layer
    // metadata arrays or optional tensors (see ir::LayerAttn).  The rest of the
    // LayerW fields above are not used by this path.
    bool ext = false;
    ir::LayerAttn ext_attn;
    cuda::Buffer q_norm_w, k_norm_w;   // per-head RMSNorm weights
    cuda::Buffer ones_head;            // [value_dim] ones -> weightless V norm
    cuda::Buffer rope_freqs;           // [rope_dim/2] inv frequencies (device)
    cuda::Buffer post_attn_norm, post_ffw_norm;  // sandwich norms
    double layer_scalar = 1.0;
    int q_dim = 0, kv_dim = 0;         // q projection out / per-kv-head dim
    // deepseek2 MLA: the decode attention segment (ln1 norm + attention + ln2 norm) as
    // one per-layer CUDA graph.  Valid only when the MLA projections are all quantized
    // (matmul GEMV path) and the q layout is not the cuBLAS "global" variant.
    std::unique_ptr<cuda::Graph> mla_attn_graph;
    int mla_attn_graph_state = 0;   // 0 = execute + record on the first decode call, 2 = replay
    // CPU-placed MoE layer: the shared-expert GPU block (zero-fill + shared gate/up/silu/
    // down + gate_inp + add, ~8 launches) depends only on hn_ and fixed weights, so it
    // is recorded once and replayed; saves the host submission time before the sync.
    std::unique_ptr<cuda::Graph> cpu_ffn_graph;
    int cpu_ffn_graph_state = 0;   // 0 = execute + record on the first decode call, 2 = replay
  };

  // ---- MTP / speculative decoding internals --------------------------------
  bool spec_active_ = false;      // incremental spec decode in progress
  int spec_pending_ = -1;         // token waiting to be emitted next
  int spec_pos_ = 0;              // model position for the next draft round
  bool mtp_ok_ = false;
  int mtp_block_ = -1;                  // block index of the MTP layer
  int mtp_gpu_experts_ = 0;             // MTP-block experts kept in VRAM (0 = host)
  QTensor mtp_eh_;                      // nextn.eh_proj [2*hidden, hidden]
  cuda::Buffer mtp_enorm_, mtp_hnorm_;  // embedding / hidden RMSNorm weights
  cuda::Buffer mtp_shn_;                // nextn.shared_head_norm (head norm)
  cuda::Buffer mtp_cat_;                // [2H] concat(enorm(emb), hnorm(h))
  cuda::Buffer mtp_emb_, mtp_h_;        // [H] draft token embedding / chained hidden
  cuda::Buffer mtp_hrow_;               // [H] residual row captured from the target
  cuda::PinnedBuffer mtp_pin_;          // top-k / logits staging
  std::vector<float> mtp_emb_host_;     // dequantised embedding scratch
  // Verification-mode plumbing (used by run_pipeline / forward_impl).
  bool verify_multi_head_ = false;      // per-row arg-max of the quantized head
  std::vector<int> verify_argmax_;      // one arg-max per verified row
  // Per-token recurrent-state snapshots for rollback (gated delta net layers).
  bool spec_snapshot_ = false;          // linear layers write per-token slots
  int ssm_slots_ = 0;                   // slots currently allocated
  cuda::Buffer ssm_rec_slots_, ssm_conv_slots_;
  std::vector<size_t> ssm_rec_off_, ssm_conv_off_;  // byte offset per layer
  size_t ssm_rec_stride_ = 0, ssm_conv_stride_ = 0;  // bytes of one slot
  void load_mtp();
  void ensure_ssm_slots(int nslots);
  void ssm_save_conv(int layer, int slot);
  void ssm_save_rec(int layer, int slot);
  void ssm_restore(int slot);
  void mtp_build_input(int tok);
  int head_argmax_mtp();
  void mtp_prefill_ctx(const std::vector<int>& toks, int n);
  void verify_rows_forward(const std::vector<int>& tokens, int pos);

  // ---- generic extended dense path (metadata-driven architectures) --------
  void load_ext_layer(int i, LayerW& w);
  void run_layer_ext(int layer, int n_tokens, int start_pos);
  void ext_attention(LayerW& L, int n_tokens, int start_pos);
  void ext_ffn(LayerW& L, int n_tokens);
  bool ext_enabled_ = false;       // model uses the extended path (config may veto)
  double ext_embed_scale_ = 1.0;   // embedding multiplier actually used
  bool ext_gelu_ = false;
  double ext_attn_scale_ = -1.0;   // <0 = legacy rsqrt(d)
  double ext_logit_softcap_ = 0.0;
  int ext_norm_plus1_ = -1;        // -1 auto/direct, 0 direct, 1 (1+w)

  void load_weights();
  void run_layer(int layer, int n_tokens, int start_pos);
  void run_layer_body(int layer, int n_tokens, int start_pos);
  cuda::PinnedBuffer hn_pin_, rl_pin_, logits_pin_;
  // Host-side expert partial sums.  Mapped (device-addressable) so the tail kernel can
  // read them over PCIe instead of a pinned H2D copy + dependent kernel (measured ~25-35
  // us of copy->kernel latency per CPU layer on this WDDM host).
  cuda::MappedBuffer xout_pin_;
  // pinned staging for the per-layer MoE routing tables (d_idx_/d_w_ are uploaded
  // asynchronously from here; a synchronous cuMemcpyHtoD flushed the WDDM queue once
  // per host-routed GPU layer)
  cuda::PinnedBuffer d_idx_pin_, d_w_pin_;
  cuda::Buffer moe_host_part_;
  void start_clock_warmer();
  void stop_clock_warmer();
  void run_moe(int layer, int n_tokens);
  int gpu_experts_ = -1;
  // Layer-granular placement: llama.cpp keeps whole MoE layers on CPU
  // (--n-cpu-moe). Splitting each layer's experts is much slower because every
  // MoE layer then needs a host round-trip.
  int gpu_moe_layers_ = -1;   // number of MoE layers that keep all experts on GPU   // -1 = all on GPU
  bool prefill_f32_ = true;   // keep an f32 mirror for batch>1 GEMMs when VRAM allows
  // ---- config snapshot (read once; hot paths never touch the config map) ----
  bool debug_ = false;
  bool no_gemv_ = false;
  bool dp4a_gemv_ = true, dp4a_q6k_ = true, tiny_sk_ = true;
  bool gd_delta_128_ = true, ssm_fuse_ = true, quant_reuse_ = true;
  // int8 activation staging reuse: consecutive dp4a GEMVs that read the same
  // activation buffer (gate/up, qkv/z, q/k/v) skip the second quant_x_i8 launch.
  void* qc_ptr_ = nullptr;
  int qc_nin_ = 0;
  long long qc_stamp_ = -1;
  // Graph-launch generation the staged int8 copy belongs to.  A graph replay executes
  // kernels without advancing g_launch_calls, so qc_stamp_ alone cannot prove that a
  // staged buffer is still valid; see docs/GRAPH-INVARIANTS.md (invariant 1).
  long long qc_graph_ = -1;
  bool splitk_ = false, dp4a_optin_ = false;
  bool use_quant_gemm_ = false, half_optin_ = false, half_t_optin_ = false;
  // Prefill GEMM via fp16 tensor cores: dequantise the weight slab straight to fp16
  // and call cuBLAS fp16 (f32 accumulate). ~3.2x the f32 cuBLAS rate on this GPU.
  bool prefill_f16_ = true;
  long long prefill_f16_min_macs_ = 0;   // below this per-call MAC count keep the f32 path
  int attn_pre_rows_ = 8;                // query rows per attention block at prefill (1 = off)
  bool attn_gqa_ = true;                 // prefill: GQA group shares one K/V stream
  // fp16-accumulate prefill GEMM: 1.85x the fp32-accumulate tensor rate, at the price
  // of fp16 rounding inside the K reduction (validated numerically in PRE-FILL-REPORT).
  bool prefill_f16_acc_ = false;
  bool layer_graph_ = false, no_bm_ = false;
  bool no_shared_ = false, no_moe_fused_ = false, keep_gpu_busy_ = false;
  int layers_override_ = 0;    // 0 = all layers
  int experts_topk_ = 0;       // 0 = from IR
  std::string bm_types_;       // block-major quant type allow list
  int row_threshold_ = 2048, q6k_x4_min_ = 32768, q6k_split_cfg_ = 0;
  int tiny_sk_max_nout_ = 128, tiny_sk_min_nin_ = 512;
  int q6k_dp4a_split_max_nout_ = 4096, small_nout_split_max_ = 256;
  bool has_kernel_overrides_ = false;
  // ---- adaptive VRAM placement ----
  size_t vram_margin_bytes_ = 1200ull << 20;   // required free VRAM after load
  size_t vram_margin_floor_bytes_ = 900ull << 20;  // hard floor for the auto-margin decision
  bool vram_auto_layers_ = false;              // config: opt-in post-load auto-margin promotion
  int vram_auto_layers_max_ = 2;               // config: max layers promoted after load
  // Ledger-invisible VRAM (driver context, cuBLAS workspaces, allocator padding): the
  // auto layer placement reserves it, otherwise the post-load margin is optimistic and
  // the driver pages (measured gap between device-used and cuda::g_buf_bytes).
  size_t vram_extra_reserve_ = 0;
  bool vram_adaptive_ = true;
  bool gpu_layers_explicit_ = false;           // user pinned gpu_layers
  int planned_gpu_moe_layers_ = -1;            // pre-load estimate (0 = none)
  int actual_gpu_moe_layers_ = 0;              // layers whose experts went to GPU
  void load_config_snapshot();
  // Decide whether one more MoE layer's experts fit with the configured margin.
  // Returns false (and clamps gpu_moe_layers_) when this and later layers must
  // stay on the CPU; the decision and its numbers are printed once.
  // `layer` is the transformer layer index, `moe_index` its rank among MoE layers.
  bool expert_layer_fits_vram(int layer, int moe_index, size_t device_bytes,
                              size_t extra_bytes);
  void check_post_load_vram();
  // Auto-margin decision (config `vram_auto_layers`): after the weights are resident the
  // free-VRAM number is *measured*, so promote CPU-placed expert layers to the GPU while
  // leaving `vram_margin_floor_bytes_` free.  Safe by construction: the pre-load margin
  // has to absorb the estimate error, the post-load promotion only spends what is really
  // left.  Explicit --vram-margin / --gpu-layers / --vram-budget disable it.
  void apply_auto_vram_layers();
  // Non-expert VRAM that is still allocated after `layer` (suffix sums) plus the
  // post-load scratch reserve; used by the adaptive placement check.
  std::vector<size_t> vram_other_suffix_;
  size_t vram_kv_per_layer_ = 0;
  size_t vram_post_load_reserve_ = 0;
  std::vector<char> vram_layer_has_kv_;
  // ---- unified GEMV dispatch (policy lives in src/kernel_select.cpp) ----
  GemvParams gemv_params(const QTensor& w) const;
  void run_gemv_plan(const GemvChoice& c, const QTensor& w, cuda::Buffer& in,
                     cuda::Buffer& out);
  void run_mla_attention(int layer, int n_tokens, int start_pos);
  // ln1 norm + MLA attention + ln2 norm for one layer; split out so the decode path can
  // record it as a per-layer CUDA graph (see the three helpers below).
  void mla_attn_segment(int layer, int n_tokens, int start_pos);
  bool mla_attn_graph_ok(int layer) const;
  void run_mla_attn_graph(int layer);
  void capture_mla_attn_graph(int layer);
  void run_pipeline(int n_tokens, int start_pos, bool last_row_only = false);
  // qwen35 hybrid stack
  void load_q35_linear(int i, LayerW& w);
  void load_q35_attn(int i, LayerW& w);
  void load_q35_ffn(int i, LayerW& w);
  void run_q35_linear(int layer, int n_tokens, int start_pos);
  void run_q35_attn(int layer, int n_tokens, int start_pos);
  // attention/mixer part only (split out so CPU-layer graphs can capture it)
  void run_q35_linear_attn(int layer, int n_tokens, int start_pos);
  void run_q35_attn_attn(int layer, int n_tokens, int start_pos);
  bool q35_layer_graph_ok(int layer) const;
  // Whole CPU layer: replay the captured attention graph, then the host MoE path.
  void run_q35_attn_graph(int layer, int start_pos);
  // Whole GPU-resident qwen35 MoE layer as one captured graph (device routing, no sync).
  void run_gpu_layer_graph(int layer, int start_pos);
  int layer_graph_bucket(int layer, int start_pos) const;
  // Warp count of the cached-attention kernel for a given context length, and the
  // matching per-layer CUDA-graph bucket.  One source of truth: the graph is captured
  // with the blockDim that replay will use, so the two must never disagree.  The
  // bucket boundaries are config attn_warp_ctx_mid / attn_warp_ctx_hi (defaults 96 /
  // 384 are the values measured on this host: below them a wider block loses more to
  // occupancy than it gains in KV latency hiding).
  static int attn_warps_for_ctx(int ctx_len);
  static int attn_warp_bucket_for_ctx(int ctx_len);
  // Capture every decode graph right after loading (capture records kernels without
  // executing them, so this leaves the recurrent/KV state untouched and the first
  // generated token already replays instead of paying for capture).
  void capture_decode_graphs();
  void capture_attn_graph(int layer, int bucket);
  void capture_gpu_layer_graph(int layer, int bucket);
  void prepare_graph_scratch();
  bool graph_capture_ = false;   // true while recording -> no dumps/downloads
  bool layer_graph_auto_ = false; // config: automatic per-layer CUDA graphs
  void run_ffn(int layer, int n_tokens);
  void rms_buf(cuda::Buffer& x, cuda::Buffer& w, cuda::Buffer& y, int rows, int n);
  // Launch one rmsnorm over `rows` rows of length `n`; selects the vectorised kernel
  // when the width allows it (see rms4_).
  void rms_raw(void* px, void* pw, void* py, int rows, int n);
  void matvec_f32(const cuda::Buffer& W, int n_out, int n_in, cuda::Buffer& in,
                  cuda::Buffer& out);
  void matvec_f32_ptr(const void* W, int n_out, int n_in, cuda::Buffer& in,
                      cuda::Buffer& out);
  // GEMV with a raw quantized weight pointer in device memory (streamed expert)
  bool gemv_typed(const std::string& type, const void* W, int n_out, int n_in,
                  cuda::Buffer& in, cuda::Buffer& out);
  // Allocates the streaming scratch; safe to call repeatedly.
  void ensure_expert_scratch();
  void reset_kv_and_states();
  void set_position(int start_pos);
  bool load_qtensor(const std::string& name, QTensor& out);
  void build_block_major(QTensor& q);
  void build_half_transposed(QTensor& q);
  cuda::Buffer upload_norm(const std::vector<float>& v) const;
  const std::vector<float>& lm_head_f32();   // materialise the f32 head on demand
  bool load_qtensor_span(const std::string& name, size_t byte_offset, size_t nbytes,
                         int n_out, int n_in, QTensor& out);
  std::vector<unsigned char> read_gguf_raw(const std::string& name);
  void matmul(const QTensor& w, int n_tokens, cuda::Buffer& in, cuda::Buffer& out);
  std::vector<float> tensor_f32(const std::string& name) const;

  ir::Model ir_;
  Container container_ = Container::Safetensors;
  std::string path_;
  int max_seq_ = 256;
  // ---- KV cache quantization / long-context attention (src/runner.cpp) ----
  // kv_type_: 0 = f32 (historical layout), 1 = q8_0 (34 B / 32 values),
  // 2 = q4_0 (18 B / 32 values).  MLA models keep f32 (no quantized MLA kernels).
  int kv_type_ = 0;
  int kv_split_ = 32;          // split-K partials used by the decode attention
  int prefill_rows_ = 512;     // activation row cap: longer prompts are prefilled in chunks
  size_t rows_ = 0;            // activation rows actually allocated = min(max_seq_, cap)
  int attn_flash_min_ = 385;   // use the tiled online-softmax kernel from this ctx on
  int n_kv_layers_ = 0;        // layers that own a KV cache (hybrid SSM: every 4th)
  cuda::Buffer attn_scratch_;  // [split_rows][n_head][kv_split][D+2] split-K partials
  int attn_scratch_rows_ = 0;
  bool kv_override_ctx_ = false;   // max_seq_ was clamped/derived from the VRAM budget
  int attn_split_rows_ = 8;    // rows of attn_scratch_ (decode / small-batch batches)
  size_t kv_layer_bytes(int seq) const;   // bytes of K+V of one attention layer
  size_t activation_bytes(int rows) const;
  // Grow logits_ if a full-logits call needs more rows than were reserved.
  void ensure_logits_rows(int rows);  // activation VRAM for `rows` prefilled rows
  bool kv_quant_ok() const;               // tiled kernel supports this head_dim
  void derive_ctx_limit(int requested);
  void alloc_kv_cache(LayerW& w);
  void kv_write_layer(LayerW& L, int n_tokens, cuda::Buffer& k, cuda::Buffer& v);
  void run_attn(LayerW& L, int n_tokens, int start_pos, cuda::Buffer& q, cuda::Buffer& out);
  void run_attn_legacy(LayerW& L, int n_tokens, int start_pos, cuda::Buffer& q,
                       cuda::Buffer& out);
  int hidden_ = 0, n_head_ = 0, n_kv_ = 0, head_dim_ = 0, inter_ = 0, vocab_ = 0;
  float rope_base_ = 10000.0f;
  float eps_ = 1e-5f;

  // container tensors (safetensors shards / gguf model)
  std::shared_ptr<void> shards_;
  std::shared_ptr<void> gguf_;

  std::vector<float> embed_host_, lm_head_host_;
  // quantized embedding kept as raw host bytes (dequantised per token on demand),
  // which avoids materialising a multi-GB f32 matrix for large vocab models.
  std::vector<unsigned char> embed_raw_;
  std::string embed_type_;
  int embed_bb_ = 0, embed_be_ = 0;
  bool embed_lazy_f32_ = false;   // lm_head_host_ is materialised on demand
  cuda::Buffer w_lm_;   // lm_head f32 (prefill)
  QTensor lm_head_qt_;  // lm_head quantized (decode)
  bool tied_head_ = false;
  std::vector<LayerW> layers_;
  cuda::Buffer w_final_norm_;

  cuda::Buffer x_, hn_, q_, k_, v_, attn_, proj_, gate_, up_, act_, down_, logits_;
  // decode head top-k staging (allocated with the runner: graph capture must not malloc)
  cuda::Buffer topk_pv_, topk_pi_, topk_ov_, topk_oi_;
  cuda::PinnedBuffer topk_pin_;
  int topk_nblk_ = 0;
  static constexpr int kMaxTopK = 128;  // device-head candidate cap (kernel limit)
  int head_topk_max_ = 8;   // config: candidates produced per decode token
  bool vram_extend_ = true;    // config: measured VRAM check may extend GPU residency
  bool q35_cpu_layer_graph_ = true;  // per-layer CUDA graph for CPU-placed qwen35 MoE layers
  bool head_topk_ = true;      // config: reduce logits on the device for decode
  bool mla_attn_graph_ = true; // config: per-layer CUDA graph for the MLA attention segment
  bool no_graph_cfg_ = false;  // config: --no-graph disables every capture path
  bool cpu_layer_ffn_graph_ = true;  // config: graph for the shared-expert block of CPU layers
  bool cpu_dual_stream_ = true;      // config: overlap that block with the host GEMV
  bool tail_zero_copy_ = true;       // config: let add3 read the host part over PCIe
  // Second stream for the CPU-layer shared-expert block: on decode it waits for hn_
  // (event on the main stream) and then runs concurrently with the routing round-trip
  // and the host GEMV, so its GPU time is hidden instead of extending the routing sync.
  // The main stream joins it before xout_ is consumed.
  void* sh_stream_ = nullptr;
  std::unique_ptr<cuda::Event> sh_ev_;
  bool rms4_ = true;           // config: vectorised rmsnorm kernel
  // impl of forward()/forward_topk(); topk_k > 0 selects the device top-k head
  std::vector<float> forward_impl(const std::vector<int>& tokens, int start_pos,
                                  bool last_row_only, int topk_k, TopKResult* tk);
  // MoE scratch
  cuda::Buffer router_logits_, xout_, gx_, ggate_, gup_, gact_, gdown_, d_idx_, d_w_;
  // MLA scratch
  std::vector<float> rope_freqs_;   // per rope pair frequency
  // raw YaRN parameters so the table can be rebuilt at runtime (experiments)
  int rope_use_d_ = 0;
  double rope_yarn_factor_ = 1.0, rope_yarn_orig_ = 4096.0, rope_yarn_logmul_ = 0.0;
  double rope_yarn_c0_ = 0.0, rope_yarn_c1_ = 1.0;
  bool rope_is_yarn_ = false;
  float rope_mscale_ = 1.0f;
  cuda::Buffer rope_freqs_dev_;
  cuda::Buffer pos_dev_;
  int pos_host_ = 0;   // pinned source for the async position upload
  std::unique_ptr<cuda::Event> prof_ev0_, prof_ev1_;
  void* stream_ = nullptr;
  std::unique_ptr<cuda::Graph> graph_;
  // Whole-pipeline CUDA graphs for prefill chunks, keyed by token count.  A prefill
  // chunk issues ~2300 launches; on this WDDM VM each cuLaunchKernel costs ~150 us of
  // host time (measured with ENGINE_PIPE_PROF), i.e. ~390 ms per chunk, which is pure
  // overhead on top of ~1.3 s of GPU work.
  std::map<int, std::unique_ptr<cuda::Graph>> prefill_graphs_;
  bool prefill_graph_ = true;
  // Logits rows reserved for prefill chunk sizing (default 1: a last_row_only prefill
  // head computes one row; 0 restores rows*vocab*4).  ensure_logits_rows() grows the
  // buffer when a caller asks for full logits.
  int prefill_logits_rows_ = 1;
  int x_half_max_nin_ = 0;   // widest matmul input seen (sizes the fp16 staging buffer)
  bool graph_ready_ = false;
  bool use_graph_ = true;
  cuda::Buffer dequant_scratch_;
  size_t scratch_floats_ = 0;
  void* gemv_q4k_ = nullptr;
  void* gemv_q4k_x4_ = nullptr;
  void* gemv_q5k_x4_ = nullptr;
  void* gemv_q6k_x4_ = nullptr;
  void* gemv_q5_0_x4_ = nullptr;
  void* gemv_q5_0_k2_ = nullptr;
  void* gemv_q4k_k2_ = nullptr;
  void* gemv_q5_0_p4_ = nullptr;
  void* gemv_q5k_ = nullptr;
  void* gemv_q6k_ = nullptr;
  void* gemv_q8_0_ = nullptr;
  void* gemv_q5_0_ = nullptr;
  void* dq_q4k_ = nullptr;
  void* dq_q5k_ = nullptr;
  void* dq_q6k_ = nullptr;
  void* dq_q8_0_ = nullptr;
  void* dq_q5_0_ = nullptr;
  void* dq_q5_1_ = nullptr;
  // fp16 dequant targets for the prefill GEMM (fp16 tensor cores, no f32 staging)
  void* dq_q4k_f16_ = nullptr;
  void* dq_q5k_f16_ = nullptr;
  void* dq_q6k_f16_ = nullptr;
  void* dq_q8_0_f16_ = nullptr;
  // x8 variants: one warp per super-block, 8 values per lane, 16 B stores (measured
  // 1.5-1.7x over the per-block kernels with a bit-identical fp16 slab).
  void* dq_q4k_f16_x8_ = nullptr;
  void* dq_q5k_f16_x8_ = nullptr;
  void* dq_q6k_f16_x8_ = nullptr;
  cuda::Buffer mq_, mkva_, mckv_, mckvn_, mkpe_, mkn_, mv_, mk_, mout_;
  void* k_ropestrided_ = nullptr;
  void* k_split2_ = nullptr;
  void* k_buildk_ = nullptr;
  void* k_mlaattn_ = nullptr;
  void* k_mlakv_ = nullptr;
  void* k_mlaattnc_ = nullptr;
  void* k_mlaqreorder_ = nullptr;
  void* k_mlakvsplit_ = nullptr;
  bool mla_q_global_ = false;
  bool mla_q_pe_first_ = false;
  // The reference MLA expands kv_b as [n_head][qk_nope | v] per head, so per-head is
  // the correct default; the global variants remain available for experiments.
  bool mla_kv_perhead_ = true;
  bool mla_kv_vfirst_ = false;
  cuda::Buffer mla_qraw_, mla_kvraw_;
  // qwen35 scratch (sized by max_seq)
  cuda::Buffer q35_mixed_, q35_z_, q35_beta_, q35_beta_b_, q35_a_, q35_g_, q35_q_, q35_k_,
      q35_v_, q35_o_, q35_conv_, q35_qfull_, q35_gate_, q35_attn_q_, q35_attn_k_, q35_attn_v_;
  void* k_q35split3_ = nullptr;
  void* k_q35beta_ = nullptr;
  void* k_q35beta_raw_ = nullptr;
  void* k_quantx_ = nullptr;
  void* k_gemvq80dp4a_ = nullptr;
  void* k_gemvq50dp4a_ = nullptr;
  void* k_attnw_ = nullptr;
  void* k_repackbm_ = nullptr;
  void* k_gemvq80bm_ = nullptr;
  void* k_gemvq50bm_ = nullptr;
  void* k_gemvq6kbm_ = nullptr;
  cuda::Buffer qx_, dx_;
  cuda::Buffer slab_half_, x_half_, deq_tmp_;
  cuda::Buffer y16_scratch_;   // fp16 GEMM output tile for the f16-accumulate path
  // Streaming scratch for CPU-placed experts: copying one expert to VRAM and
  // running the GPU GEMV is ~6x faster than the scalar CPU matvec.
  cuda::Buffer xexp_w_, xexp_g_, xexp_u_, xexp_a_, xexp_d_;
  cuda::Buffer moe_gp_, moe_up_, moe_dp_;   // device pointer arrays (fused MoE)
  cuda::Buffer moe_gact_;                    // fused MoE activation buffer
  cuda::Buffer slot_exp_, slot_tok_, slot_w_;  // device routing output
  std::vector<cuda::Buffer> all_gp_, all_up_, all_dp_;  // per layer: ne pointers
  bool xexp_ready_ = false;
  bool layer_graph_mode_ = false;
  // Device-side MoE routing for layers whose experts are all GPU-resident.  Enabled by
  // default unless the model's expert tensors still contain Q5_1 (the device GEMV
  // branch for Q5_1 is not numerically verified yet, see SPEC §10/§11).
  bool dev_route_default_ = false;
  std::unique_ptr<cuda::Buffer> warmer_stop_;
  void* warmer_stream_ = nullptr;
  void* warmer_kernel_ = nullptr;
  cuda::Buffer sh_scratch_, shu_scratch_, sha_scratch_, shd_scratch_;  // shared-expert scratch
  void* k_f32tohalf_ = nullptr;
  void* k_f16tof32_ = nullptr;
  void* k_transpf16_ = nullptr;
  void* k_moegateup_ = nullptr;
  void* k_moeroute_ = nullptr;
  void* k_zerofill_ = nullptr;
  void* k_routergemm_ = nullptr;
  void* k_quantgemm_ = nullptr;
  void* gemv_q5k_sk_ = nullptr;
  void* gemv_q6k_sk_ = nullptr;
  void* gemv_q8_sk_ = nullptr;
  void* gemv_q5k_dp4a_ = nullptr;
  void* gemv_q5k_dp4a_s_ = nullptr;
  void* gemv_q6k_dp4a_ = nullptr;
  void* gemv_q6k_dp4a_sk_ = nullptr;
  void* k_quantx_i8_ = nullptr;
  cuda::Buffer qx_i8_, dx_i8_, xs8_i8_;
  void* k_moegateupdev_ = nullptr;
  void* k_modeowndev_ = nullptr;
  void* k_modeown_ = nullptr;
  void* k_gemvf32_ = nullptr;
  int n_tokens_cur_ = 1;   // tokens in the current forward (f32 GEMV dispatch)
  int dbg_matvec_ = 0;
  void* k_ropeinter_ = nullptr;
  size_t qx_bytes_ = 0;
  void* k_moeshgate_ = nullptr;
  void* k_gdconv_ = nullptr;
  void* k_gdstate_ = nullptr;
  void* k_gdqw_ = nullptr;
  void* k_gddelta_ = nullptr;
  void* k_gddelta128_ = nullptr;
  void* k_gddelta128smem_ = nullptr;   // prefill: state staged in shared memory
  void* k_gddelta128reg_ = nullptr;    // prefill: state kept in registers
  void* k_gddelta128f_ = nullptr;
  void* k_gdprep_ = nullptr;
  void* k_gdnorm_ = nullptr;
  void* k_q35split_ = nullptr;
  void* k_q35gate_ = nullptr;
  int expert_inter_ = 0;
  void* k_gather_ = nullptr;
  void* k_scatter_ = nullptr;

  void* k_rms_ = nullptr;
  void* k_rms4_ = nullptr;             // vectorised rmsnorm (same output contract)
  void* k_topkpart_ = nullptr;         // device-side top-k stage 1
  void* k_topkfinal_ = nullptr;        // device-side top-k stage 2
  void* k_rope_ = nullptr;
  void* k_kvwrite_ = nullptr;
  void* k_attn_ = nullptr;
  void* k_kvwriteq_ = nullptr;       // quantizing KV write (q8_0 / q4_0)
  void* k_attntiled_ = nullptr;
  void* k_attntiledpre_ = nullptr;   // prefill: several query rows per block
  void* k_attntiledpregqa_ = nullptr;  // prefill: GQA group shares one K/V stream      // tiled online-softmax attention over f32/quant KV
  void* k_attncombine_ = nullptr;    // split-K merge for attntiled_
  void* k_silu_ = nullptr;
  void* k_gelumul_ = nullptr;
  void* k_scale_ = nullptr;
  void* k_softcap_ = nullptr;
  void* k_attncfg_ = nullptr;
  void* k_add_ = nullptr;
  void* k_add3_ = nullptr;  // a += (b + c) fused host-part + residual add
  void* k_bias_ = nullptr;
  std::unique_ptr<cuda::Module> mod_;
};

}  // namespace runner
