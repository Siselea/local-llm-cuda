#include "engine/config.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "engine/json.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace engine {
namespace config {

namespace {

// The single source of truth for the configuration surface.
// key, env (legacy), cli, cli_takes_value, env_invert, type, default, min, max, help
const OptionDef kDefs[] = {
    // ---------------- model / run shape ----------------
    {"model", "ENGINE_MODEL", "model", true, false, Type::String, "",
     0, 0, "model path (.gguf file or safetensors directory)"},
    {"ctx", "ENGINE_CTX", "ctx", true, false, Type::Int, "256",
     0, 1 << 20,
     "context length in tokens = hard total-sequence budget (prompt + generated; "
     "KV cache is sized from it). Any integer 1..1048576 is accepted verbatim and is "
     "only reduced when the KV cache physically does not fit free VRAM (the loader "
     "prints the requested->actual ledger). 0 = auto: derive the largest length that "
     "fits from free VRAM and the configured kv_type"},
    {"threads", "ENGINE_THREADS", "threads", true, false, Type::Int, "0",
     0, 4096, "CPU worker threads for MoE experts, 0 = auto: 2 x physical cores when SMT=2 "
     "topology is available (12 on a 6C/12T host, paired with win_affinity=2), else the "
     "historical ~3/4 of logical CPUs (9)"},
    {"gpu_layers", "ENGINE_GPU_LAYERS", "gpu-layers", true, false, Type::Int, "-1",
     -1, 4096, "number of MoE layers kept on the GPU, -1 = auto (layer granular)"},
    {"gpu_experts", "ENGINE_GPU_EXPERTS", "gpu-experts", true, false, Type::Int, "-1",
     -1, 100000, "experts per MoE layer kept on the GPU, -1 = auto/layer placement"},
    {"layers", "ENGINE_LAYERS", "layers", true, false, Type::Int, "0",
     0, 1 << 20, "run only the first N transformer layers (0 = all; perf breakdown)"},

    // ---------------- memory placement ----------------
    {"vram_margin_mb", "ENGINE_VRAM_MARGIN_MB", "vram-margin", true, false, Type::Int, "1200",
     0, 1 << 20, "free VRAM (MiB) required to remain after loading (post-load check)"},
    {"vram_extra_reserve_mb", "ENGINE_VRAM_EXTRA_RESERVE_MB", "vram-extra-reserve", true,
     false, Type::Int, "1024", 0, 1 << 20,
     "VRAM reserved for allocations that never show up in the engine's own ledger: CUDA "
     "context, cuBLAS workspaces and allocator padding.  The automatic layer placement "
     "subtracts it so the post-load free-VRAM margin is real instead of optimistic "
     "(measured on this machine as [vram] post-load used - alloc_sum: 1.13-1.25 GiB on the "
     "9B/35B, 2.1 GiB on the Coder MoE).  Tuning curve on an exclusive GPU: reserve 1024 "
     "keeps 2.1-2.4 GiB free after load (margin 1200) at a decode cost of only 0.6-2.4%, "
     "while 2048 costs 6.5-9% (every MiB of reserve can drop a whole MoE layer).  Without "
     "it an auto-placed MoE model can load to 0 MiB free and the driver pages: 3 tok/s "
     "instead of 25.  0 restores the old optimistic behaviour"},
    {"vram_budget_mb", "ENGINE_VRAM_BUDGET_MB", "vram-budget", true, false, Type::Int, "0",
     0, 1 << 20, "simulated VRAM budget in MiB for expert placement, 0 = free VRAM"},
    {"workspace_mb", "ENGINE_WORKSPACE_MB", "workspace", true, false, Type::Int, "768",
     0, 1 << 20, "VRAM reserved for KV/activations/scratch during placement"},
    {"vram_check", "ENGINE_NO_VRAM_CHECK", "no-vram-check", false, true, Type::Bool, "true",
     0, 0, "adaptive post-load free-VRAM enforcement (env layer: none)"},
    {"device", "ENGINE_DEVICE", "device", true, false, Type::Int, "0",
     0, 64, "CUDA device index"},

    // ---------------- placement policy (gpu / vram+ram split / ssd) ----------------
    {"host_margin_mb", "ENGINE_HOST_MARGIN_MB", "host-margin", true, false, Type::Int, "0",
     0, 1 << 20,
     "host RAM (MiB) the placement decider must leave free after loading, 0 = auto "
     "(max(1 GiB, total RAM / 16))"},
    {"placement_policy", "ENGINE_PLACEMENT", "placement", true, false, Type::String, "auto",
     0, 0,
     "weight placement policy: auto (measure free VRAM/RAM and decide) | gpu (require "
     "everything resident) | split (require CPU expert offload) | ssd (SSD streaming - "
     "NOT IMPLEMENTED in this build)"},
    {"ssd_stream", "ENGINE_SSD_STREAM", "ssd-stream", false, false, Type::Bool, "false",
     0, 0,
     "enable SSD weight streaming when the model does not fit VRAM+RAM (NOT IMPLEMENTED; "
     "off by default because on PCIe x8 / HDD streaming is usually a net loss)"},
    {"ssd_cache", "ENGINE_SSD_CACHE", "ssd-cache", true, false, Type::String, "",
     0, 0,
     "on-disk expert cache directory for SSD streaming (interface placeholder; the "
     "decision logger reports it, the streamer is not implemented)"},
    {"allow_oversubscribe", "ENGINE_ALLOW_OVERSUBSCRIBE", "allow-oversubscribe", false,
     false, Type::Bool, "false", 0, 0,
     "continue when the placement decider reports the model does not fit VRAM+RAM "
     "(the driver will page: expect single-digit tok/s)"},
    {"strict_vram", "ENGINE_STRICT_VRAM", "strict-vram", false, false, Type::Bool, "false",
     0, 0,
     "strict VRAM policy for `serve`: refuse to start when the measured free VRAM after "
     "load is below --vram-margin (v0.1.6 behaviour). Default (off) starts with a loud "
     "WARN when free VRAM is between a 700 MiB hard floor and the margin, because desktop "
     "WDDM can consume VRAM between the loader check and the serve guard"},

    // ---------------- startup interaction ----------------
    {"interactive", "ENGINE_INTERACTIVE", "no-interactive", false, false, Type::Bool, "true",
     0, 0,
     "interactive startup prompts (ctx / kv_type / CPU expert offload) when stdin is a "
     "terminal; --no-interactive or ENGINE_INTERACTIVE=0 disables, pipes/CI are unaffected"},
    {"assume_yes", "ENGINE_ASSUME_YES", "yes", false, false, Type::Bool, "false",
     0, 0, "skip the interactive prompts and use the computed defaults (scripts/CI)"},
    {"model_dir", "ENGINE_MODEL_DIR", "model-dir", true, false, Type::String, "",
     0, 0,
     "directory listed by the interactive model picker (run without a model argument), "
     "empty = auto (<USERPROFILE>/models when it exists, else the current directory)"},

    // ---------------- execution / graphs ----------------
    {"no_graph", "ENGINE_NO_GRAPH", "no-graph", false, false, Type::Bool, "false",
     0, 0, "disable CUDA Graph capture"},
    {"layer_graph", "ENGINE_LAYER_GRAPH", "layer-graph", false, false, Type::Bool, "false",
     0, 0, "experimental per-layer CUDA Graph"},
    {"layer_graph_auto", "ENGINE_LAYER_GRAPH_AUTO", "layer-graph-auto", false, false,
     Type::Bool, "true", 0, 0,
     "automatic per-layer CUDA graphs for GPU-resident qwen35 MoE layers: one graph "
     "replay per layer instead of ~15 attention/mixer kernel launches (WDDM submission "
     "was ~20-30% of the decode wall time). CPU-placed layers keep the plain path "
     "(ENGINE_CPU_LAYER_GRAPH opts in; measured neutral-to-negative on this VM)"},
    {"mla_no_graph", "ENGINE_MLA_NO_GRAPH", "mla-no-graph", false, false, Type::Bool, "false",
     0, 0, "disable CUDA Graph for MLA models (default keeps the historical rule)"},
    {"tail_zero_copy", "ENGINE_NO_TAIL_ZC", "no-tail-zc", false, true, Type::Bool, "true",
     0, 0,
     "CPU-layer tail: let add_three_inplace read the host-expert partial sums directly "
     "from mapped (device-addressable) pinned memory instead of a pinned H2D copy "
     "followed by a dependent kernel (zero-copy over PCIe)"},
    {"cpu_dual_stream", "ENGINE_NO_CPU_DUAL_STREAM", "no-cpu-dual-stream", false, true,
     Type::Bool, "true", 0, 0,
     "run the CPU-layer shared-expert block on a second stream, concurrently with the "
     "host routed-expert GEMV (structural overlap: the block is independent of routing); "
     "the main stream joins it before the add3 that consumes xout_"},
    {"cpu_layer_ffn_graph", "ENGINE_NO_CPU_FFN_GRAPH", "no-cpu-ffn-graph", false, true,
     Type::Bool, "true", 0, 0,
     "per-layer CUDA graph for the shared-expert GPU block of CPU-placed MoE layers "
     "(zero-fill + shared gate/up/silu/down + gate_inp + add); measured 9 WDDM "
     "submissions collapse into one cuGraphLaunch per layer on the 35B"},
    {"mla_attn_graph", "ENGINE_NO_MLA_ATTN_GRAPH", "no-mla-attn-graph", false, true, Type::Bool,
     "true", 0, 0,
     "per-layer CUDA graph for the deepseek2 MLA attention segment (ln1 norm + attention "
     "+ ln2 norm) on decode; measured 18 WDDM submissions per layer collapse into one "
     "cuGraphLaunch (~10 us on this host vs ~180 us for the 18 bare launches)"},
    {"prefill_f32", "ENGINE_PREFILL_F32", "prefill-f32", false, false, Type::Bool, "true",
     0, 0, "keep f32 weight mirrors for batched prefill (auto-disabled on MoE / low VRAM)"},
    // ---------------- KV cache precision / long context ----------------
    {"kv_type", "ENGINE_KV_TYPE", "kv-type", true, false, Type::String, "f32",
     0, 0,
     "KV cache precision: f32 (default, historical layout) | q8_0 (34 B / 32 values) | "
     "q4_0 (18 B / 32 values).  Quantized caches use the tiled online-softmax attention "
     "kernel; MLA models keep f32 (no quantized MLA kernels)"},
    {"kv_offload", "ENGINE_KV_OFFLOAD", "kv-offload", true, false, Type::Bool, "false",
     0, 0,
     "keep the KV cache in host memory instead of VRAM (host-mapped KV is not implemented "
     "yet; the option is parsed and reported, the cache must fit in VRAM)"},
    {"kv_split", "ENGINE_KV_SPLIT", "kv-split", true, false, Type::Int, "32",
     1, 32,
     "split-K partials for the tiled decode attention (1..32; 32 measured best after the "
     "GQA kernel: the key range of one query row is split across this many blocks, "
     "prefill batches use 1 and finish in-register)"},
    {"ctx_auto", "ENGINE_CTX_AUTO", "ctx-auto", true, false, Type::Bool, "true",
     0, 0,
     "derive the maximum ctx from free VRAM - weights - workspace - activations for the "
     "configured kv_type and clamp --ctx to it (--ctx 0 derives the limit from scratch)"},
    {"prefill_rows", "ENGINE_PREFILL_ROWS", "prefill-rows", true, false, Type::Int, "512",
     1, 65536,
     "activation row cap: prompts longer than this are prefilled in chunks of this many "
     "tokens, so --ctx can exceed the activation buffer size"},
    {"gemm", "ENGINE_GEMM", "quant-gemm", false, false, Type::Bool, "false",
     0, 0, "self-developed quantized GEMM for prefill (experimental)"},
    {"no_moe_fused", "ENGINE_NO_MOE_FUSED", "no-moe-fused", false, false, Type::Bool, "false",
     0, 0, "disable the fused MoE gate/up/down kernels"},
    {"dev_route", "ENGINE_NO_DEV_ROUTE", "dev-route", false, true, Type::Bool, "true",
     0, 0,
     "device-side MoE routing for GPU-resident layers (removes the per-layer router D2H "
     "download + host sync + routing-table upload; auto-disabled for expert layouts that "
     "still contain Q5_1). Env ENGINE_NO_DEV_ROUTE disables, ENGINE_DEV_ROUTE forces on"},
    {"keep_gpu_busy", "ENGINE_KEEP_GPU_BUSY", "keep-gpu-busy", false, false, Type::Bool, "false",
     0, 0, "run a clock warmer on a second stream during CPU expert layers"},

    // ---------------- model semantics overrides (IR is authoritative) ----------------
    {"norm_plus1", "ENGINE_NORM_PLUS1", "norm-plus1", true, false, Type::Int, "-1",
     -1, 1, "RMSNorm zero-centred (+1) interpretation: -1 auto, 0 off, 1 force"},
    {"no_yarn", "ENGINE_NO_YARN", "no-yarn", false, false, Type::Bool, "false",
     0, 0, "ignore GGUF rope.scaling.type=yarn"},
    {"router", "ENGINE_ROUTER", "router", true, false, Type::String, "",
     0, 0, "MoE router kind override: topk_softmax|sigmoid_topk (default from IR)"},
    {"norm_topk", "ENGINE_NORM_TOPK", "norm-topk", true, false, Type::Int, "-1",
     -1, 1, "normalise MoE top-k weights: -1 auto (from IR), 0/1 force"},
    {"experts_topk", "ENGINE_EXPERTS_TOPK", "experts-topk", true, false, Type::Int, "0",
     0, 256, "override MoE top-k at run time (0 = from IR)"},
    {"no_shared", "ENGINE_NO_SHARED", "no-shared", false, false, Type::Bool, "false",
     0, 0, "skip shared experts (diagnostic)"},
    {"mla_q_layout", "ENGINE_MLA_Q_LAYOUT", "mla-q-layout", true, false, Type::String, "",
     0, 0, "MLA q layout experiment: global|global_pe|perhead"},
    {"mla_kv_layout", "ENGINE_MLA_KV_LAYOUT", "mla-kv-layout", true, false, Type::String, "",
     0, 0, "MLA kv layout experiment: global|global_vk|perhead|perhead_vk"},

    // ---------------- kernel selection ----------------
    {"dp4a_gemv", "ENGINE_NO_DP4A_GEMV", "no-dp4a-gemv", false, true, Type::Bool, "true",
     0, 0, "int8 dp4a GEMV for Q5_K/Q6_K decode (env ENGINE_NO_DP4A_GEMV disables)"},
    {"dp4a_q6k", "ENGINE_NO_DP4A_Q6K", "no-dp4a-q6k", false, true, Type::Bool, "true",
     0, 0, "int8 dp4a Q6_K GEMV; disabling falls back to the fp32 x4 path"},
    {"tiny_sk", "ENGINE_NO_TINY_SK", "no-tiny-sk", false, true, Type::Bool, "true",
     0, 0, "split-K GEMV for tiny output dims (Q8_0, n_out<=128)"},
    {"gd_delta_128", "ENGINE_NO_GD_DELTA_128", "no-gd-delta-128", false, true, Type::Bool,
     "true", 0, 0,
     "register-tiled gated-delta-rule kernel for key/value head dim 128 (qwen35 layers)"},
    {"quant_reuse", "ENGINE_NO_QUANT_REUSE", "no-quant-reuse", false, true, Type::Bool, "true",
     0, 0,
     "reuse one int8 activation staging between consecutive dp4a GEMVs on the same input"},
    {"ssm_fuse", "ENGINE_NO_SSM_FUSE", "no-ssm-fuse", false, true, Type::Bool, "true", 0, 0,
     "fused gated-delta-net decode kernels (conv+state+split+beta in one launch, q/k norm "
     "and gated norm folded into the delta-rule kernel)"},
    {"splitk", "ENGINE_SPLITK", "splitk", false, false, Type::Bool, "false",
     0, 0, "opt-in split-K for the x4 GEMV kernels (measured a wash)"},
    {"cpu_layer_graph", "ENGINE_NO_CPU_LAYER_GRAPH", "cpu-layer-graph", false, true,
     Type::Bool, "true", 0, 0,
     "capture the mixer/attention part of CPU-placed qwen35 MoE layers as a per-layer "
     "CUDA graph (removes ~7 WDDM submissions per layer; the one-time capture happens "
     "right after the first execution, i.e. during prefill)"},
    {"vram_extend", "ENGINE_NO_VRAM_EXTEND", "no-vram-extend", false, true, Type::Bool,
     "true", 0, 0,
     "let the measured per-layer VRAM check extend GPU-resident MoE layers beyond the "
     "conservative arithmetic estimate (contiguous prefix only, margin still enforced)"},
    {"vram_auto_layers", "ENGINE_VRAM_AUTO_LAYERS", "vram-auto-layers", false, false,
     Type::Bool, "false", 0, 0,
     "OPT-IN auto-margin decision: after load, promote CPU-placed expert layers to the GPU "
     "while the *measured* post-load free VRAM still leaves --vram-margin-floor below.  "
     "35B measured: +1 layer, free 1447 -> 935 MiB, decode 36.00 -> 35.34 ms, bit-exact; "
     "default OFF because the post-load per-layer CUDA-graph capture then eats ~850 MiB "
     "more and decode-time free drops to ~82 MiB (default margin keeps 592 MiB) - too close "
     "to driver paging.  Explicit --vram-margin / --gpu-layers / --vram-budget disable it"},
    {"vram_margin_floor_mb", "ENGINE_VRAM_MARGIN_FLOOR_MB", "vram-margin-floor", true, false,
     Type::Int, "900", 0, 1 << 20,
     "hard free-VRAM floor the auto-margin promotion must leave after load (MiB). "
     "900 = the measured safe line on this host (free=0 -> driver paging -> 3 tok/s)"},
    {"vram_auto_layers_max", "ENGINE_VRAM_AUTO_LAYERS_MAX", "vram-auto-layers-max", true,
     false, Type::Int, "2", 0, 64,
     "maximum number of MoE layers the auto-margin decision may promote after load"},
    {"rms4", "ENGINE_NO_RMS4", "no-rms4", false, true, Type::Bool, "true", 0, 0,
     "float4 rmsnorm kernel (4x fewer load/store instructions in the 1-block norm; "
     "requires width % 4 == 0, falls back automatically)"},
    {"head_topk_max", "ENGINE_HEAD_TOPK_MAX", "head-topk-max", true, false, Type::Int, "8",
     1, 128,
     "decode head candidates computed on the device (1-128): forward_topk / "
     "forward_topk_candidates can return up to this many (index, logit) pairs without "
     "copying the full logits row; larger values let a sampling server sample from "
     "top-k candidates instead of a 1 MB D2H"},
    {"head_topk", "ENGINE_NO_HEAD_TOPK", "no-head-topk", false, true, Type::Bool, "true", 0, 0,
     "reduce the decode logits row to the top-k on the device and transfer only k pairs "
     "(the full 0.4-1 MB row download costs ~0.27 ms/token on WDDM)"},
    // ---------------- MTP / speculative decoding ----------------
    {"mtp", "ENGINE_MTP", "mtp", false, false, Type::Bool, "false", 0, 0,
     "MTP (multi-token-prediction) greedy speculative decoding.  Needs a model that "
     "ships a `nextn` block (ir.mtp_block); the draft prefix is only kept where the "
     "target model's own arg-max agrees, so the emitted tokens are identical to the "
     "plain greedy loop.  Measured: +2-4x acceptance on llama.cpp, but the speed-up "
     "depends on how much cheaper a k+1-token verify batch is than k+1 decode steps"},
    {"mtp_k", "ENGINE_MTP_K", "mtp-k", true, false, Type::Int, "2", 1, 3,
     "chained MTP draft tokens per speculative round (1-3)"},
    {"mtp_gpu", "ENGINE_NO_MTP_GPU", "no-mtp-gpu", false, true, Type::Bool, "true", 0, 0,
     "keep the MTP block's routed experts in VRAM when they fit (falls back to the host "
     "otherwise, which costs a host sync per draft step)"},
    {"mtp_vram_margin_mb", "ENGINE_MTP_VRAM_MARGIN_MB", "mtp-vram-margin", true, false,
     Type::Int, "256", 0, 1 << 20,
     "free-VRAM headroom required before the MTP layer's experts are placed on the GPU"},
    {"mtp_ctx_prefill", "ENGINE_NO_MTP_CTX_PREFILL", "no-mtp-ctx-prefill", false, true,
     Type::Bool, "true", 0, 0,
     "populate the MTP layer's own attention history for the prompt in one batched pass "
     "(otherwise the draft layer attends over a zeroed KV window and acceptance drops)"},
    {"dp4a", "ENGINE_DP4A", "dp4a", false, false, Type::Bool, "false",
     0, 0, "experimental dp4a GEMV for Q8_0/Q5_0"},
    {"row_threshold", "ENGINE_ROW_THRESHOLD", "row-threshold", true, false, Type::Int, "2048",
     0, 1 << 20, "n_out at which the 4-rows/warp GEMV kernels become worth it"},
    {"q6k_x4_min", "ENGINE_Q6K_X4_MIN", "q6k-x4-min", true, false, Type::Int, "32768",
     0, 1 << 20, "row count above which Q6_K prefers the x4 GEMV kernel"},
    {"q6k_split", "ENGINE_Q6K_SPLIT", "q6k-split", true, false, Type::Int, "0",
     0, 8, "pin the Q6_K dp4a split-K factor, 0 = auto"},
    {"tiny_sk_max_nout", "ENGINE_TINY_SK_MAX_NOUT", "tiny-sk-max-nout", true, false, Type::Int,
     "128", 0, 1 << 20,
     "Q8_0 split-K is used when n_out <= this (default 128 = measured: below it the "
     "non-split GEMV leaves the SMs idle)"},
    {"tiny_sk_min_nin", "ENGINE_TINY_SK_MIN_NIN", "tiny-sk-min-nin", true, false, Type::Int,
     "512", 0, 1 << 20,
     "Q8_0 split-K requires n_in >= this (default 512 = at least 16 K-blocks, otherwise "
     "the reduction costs more than the wave it saves)"},
    {"half", "ENGINE_HALF", "half", false, false, Type::Bool, "false",
     0, 0, "experimental fp16 NT prefill mirror"},
    {"half_t", "ENGINE_HALF_T", "half-t", false, false, Type::Bool, "false",
     0, 0, "experimental fp16 TN prefill mirror"},
    {"prefill_logits_rows", "ENGINE_PREFILL_LOGITS_ROWS", "prefill-logits-rows", true,
     false, Type::Int, "1", 0, 1 << 20,
     "how many logits rows the prefill activation budget reserves (0 = one per chunk row, "
     "the old behaviour).  A last_row_only prefill head computes exactly ONE row, so "
     "reserving 1 instead of rows*vocab*4 frees ~1 MB per row at a 248k vocab and lets the "
     "prefill chunk grow ~4x at the same VRAM (fewer whole-model dequant passes); a "
     "full-logits call grows the buffer on demand"},
    {"prefill_f16", "ENGINE_PREFILL_F16", "prefill-f16", false, false, Type::Bool, "true",
     0, 0,
     "prefill (n_tokens>1) GEMM through fp16 tensor cores: dequantise the weight slab "
     "straight to fp16 and call cuBLAS fp16 with f32 accumulation (~3x the f32 rate)"},
    {"gd_reg", "ENGINE_GD_REG", "gd-reg", false, false, Type::Bool, "true",
     0, 0,
     "prefill gated-delta rule: keep the per-column state in registers for the whole "
     "token loop (only cross-group partials go through shared memory)"},
    {"gd_smem", "ENGINE_GD_SMEM", "gd-smem", false, false, Type::Bool, "true",
     0, 0,
     "prefill gated-delta rule: keep the 64 KiB per-head recurrent state in shared "
     "memory across the token loop (avoids 4 MiB of L2 traffic per token)"},
    {"attn_gqa", "ENGINE_ATTN_GQA", "attn-gqa", false, false, Type::Bool, "true",
     0, 0,
     "prefill attention: one warp owns a query row and the whole GQA group of its KV "
     "head (f32 KV, head_dim 256; the per-head arithmetic is unchanged)"},
    {"attn_gqa_decode", "ENGINE_NO_ATTN_GQA", "no-attn-gqa", false, true, Type::Bool, "true",
     0, 0,
     "decode attention: one block per KV head with the rep query heads of the GQA group "
     "sharing each K/V row through L1 (same math/scratch as attn_tiled, fewer DRAM reads)"},
    {"attn_gqa_min_ctx", "ENGINE_ATTN_GQA_MIN_CTX", "attn-gqa-min-ctx", true, false, Type::Int,
     "0", 0, 262144,
     "only use the GQA-grouped decode attention when ctx_len >= this (keeps short "
     "contexts on the tuned per-head kernel if the wider block is slower there)"},
    {"attn_gqa_wide", "ENGINE_NO_ATTN_GQA_WIDE", "no-attn-gqa-wide", false, true, Type::Bool,
     "true", 0, 0,
     "GQA decode kernel: read f32 K/V with float4 (two 512 B coalesced loads per warp) "
     "instead of eight strided 4 B loads"},
    {"attn_pre_rows", "ENGINE_ATTN_PRE_ROWS", "attn-pre-rows", true, false, Type::Int, "8",
     1, 32,
     "query rows per attention block when prefilling (all warps share the same K/V row "
     "through L1; 1 restores the one-warp-per-(token,head) kernel for A/B)"},
    {"prefill_rows_max", "ENGINE_PREFILL_ROWS_MAX", "prefill-rows-max", true, false, Type::Int,
     "4096", 1, 65536,
     "upper bound for the automatic prefill chunk growth (the chunk starts at "
     "prefill_rows and doubles while activations+KV still fit; larger chunks amortise "
     "the whole-model dequant pass)"},
    {"prefill_graph", "ENGINE_PREFILL_GRAPH", "prefill-graph", false, false, Type::Bool, "false",
     0, 0,
     "capture/replay a whole-pipeline CUDA graph per prefill chunk size (one launch "
     "instead of ~2300: the WDDM launch overhead is ~390 ms per 1.6k-token chunk)"},
    {"prefill_f16_acc", "ENGINE_PREFILL_F16_ACC", "prefill-f16-acc", false, false, Type::Bool,
     "false", 0, 0,
     "prefill GEMM with fp16 accumulation (CUBLAS_COMPUTE_16F): 1.85x the fp32-accumulate "
     "tensor rate on this GPU; adds fp16 rounding inside the K reduction"},
    {"prefill_f16_min_macs", "ENGINE_PREFILL_F16_MIN_MACS", "prefill-f16-min-macs", true,
     false, Type::Int, "0", 0, 1LL << 40,
     "minimum n_tokens*n_out*n_in per matmul before the fp16 prefill path is used; below "
     "it the f32/dequant path is kept (tiny matrices are launch bound, not compute bound)"},
    {"bm_types", "ENGINE_BM_TYPES", "bm-types", true, false, Type::String, "",
     0, 0, "comma list of quant types to block-major repack (diagnostic)"},
    {"no_bm", "ENGINE_NO_BM", "no-bm", false, false, Type::Bool, "false",
     0, 0, "disable block-major weight repacking"},
    {"no_gemv", "ENGINE_NO_GEMV", "no-gemv", false, false, Type::Bool, "false",
     0, 0, "disable the specialized GEMV path (diagnostic A/B)"},

    // ---------------- CPU backend ----------------
    {"no_avx2", "ENGINE_NO_AVX2", "no-avx2", false, false, Type::Bool, "false",
     0, 0, "disable AVX2 in the CPU quant GEMV path"},
    {"cpu_moe_batch", "ENGINE_NO_CPU_MOE_BATCH", "no-cpu-moe-batch", false, true, Type::Bool,
     "true", 0, 0,
     "batched CPU MoE pass (one parallel region per matrix group instead of one pool "
     "wake-up per matvec; env ENGINE_NO_CPU_MOE_BATCH disables)"},
    {"cpu_int8", "ENGINE_CPU_INT8", "cpu-int8", true, false, Type::Int, "1", 0, 2,
     "int8 host GEMV kernels, selected per quant type and width: 0 = off (fp32 kernels), "
     "1 = default = where measured faster (Q6_K rows: -10% paired median on the isolated "
     "35B CPU layer, 0.81 -> 0.73 ms/layer), 2 = Q4_K too (measured slower: +4.6%, "
     "kept for A/B only); see docs/CPU-INT8.md"},
    {"cpu_pool_spin", "ENGINE_POOL_SPIN", "cpu-pool-spin", true, false, Type::Int, "512", 0,
     1000000,
     "CPU worker idle-spin length in PAUSE iterations before falling back to the condvar "
     "(default 512 ~ 20 us, covers the tail of the previous parallel region)"},

    // ---------------- Windows scheduling (real desktop hosts) ----------------
    {"win_priority", "ENGINE_WIN_PRIORITY", "win-priority", true, false, Type::Int, "1", 0, 2,
     "Windows process priority class: 1 = HIGH_PRIORITY_CLASS (default; the host GEMV "
     "threads stop losing cycles to desktop background apps), 2 = ABOVE_NORMAL_PRIORITY_CLASS, "
     "0 = NORMAL (off, for A/B). NEVER REALTIME - it starves the input/IO/driver threads. "
     "Applied once at startup to `run`, `serve` and every test tool; no-op outside Windows"},
    {"win_thread_priority", "ENGINE_WIN_THREAD_PRIORITY", "win-thread-priority", true, false,
     Type::Int, "-1", -1, 2,
     "CPU pool worker threads' relative priority: -1 = auto (THREAD_PRIORITY_HIGHEST when "
     "win_priority>0, inherit when 0), 0 = inherit, 1 = ABOVE_NORMAL, 2 = HIGHEST (base+2). "
     "A/B switch for 'is the main thread or the workers the bottleneck'"},
    {"win_affinity", "ENGINE_WIN_AFFINITY", "win-affinity", true, false, Type::Int, "2", 0, 2,
     "Windows CPU affinity for the CPU pool: 2 = SMT-paired (DEFAULT; logical thread 2k/2k+1 "
     "pinned to the two hardware siblings of physical core k, pairs read from the ACTUAL "
     "GetLogicalProcessorInformationEx topology mask - not an assumed i,i+n mapping; the main "
     "thread is logical thread 0; an odd --threads count fills k pairs + one single). "
     "0 = off (the OS schedules; pre-2026-11 default), 1 = physical-core-first order. "
     "Measured on this host: paired 12/11/10 threads are 4-12% faster than 9 unbound on the "
     "CPU expert GEMV layer, see THREAD-AFFINITY-REPORT.md"},
    {"attn_warp_ctx_mid", "ENGINE_ATTN_WARP_CTX_MID", "attn-warp-mid", true, false, Type::Int,
     "96", 1, 1 << 20,
     "cached-attention warp bucket: ctx_len above this uses 16 warps instead of 8 "
     "(default 96 = measured on this host; the CUDA-graph bucket follows the same value)"},
    {"attn_warp_ctx_hi", "ENGINE_ATTN_WARP_CTX_HI", "attn-warp-hi", true, false, Type::Int,
     "384", 1, 1 << 20,
     "cached-attention warp bucket: ctx_len above this uses 32 warps (default 384 = "
     "measured; capped by the shared-memory budget at runtime)"},
    {"q6k_dp4a_split_max_nout", "ENGINE_Q6K_DP4A_SPLIT_MAX_NOUT", "q6k-dp4a-split-max-nout",
     true, false, Type::Int, "4096", 0, 1 << 20,
     "Q6_K dp4a automaticsplit-K only when n_out <= this (default 4096 = measured "
     "crossover; the per-shape calibration cache can still override any shape)"},
    {"small_nout_split_max", "ENGINE_SMALL_NOUT_SPLIT_MAX", "small-nout-split-max", true,
     false, Type::Int, "256", 0, 1 << 20,
     "Q4_K/Q5_0 use the 2-warp split-K kernel when n_out <= this (default 256 = 8 "
     "rows/block x 32 blocks, the row count that covers a 32-SM device)"},

    // ---------------- HTTP server (engine serve) ----------------
    {"serve_host", "ENGINE_SERVE_HOST", "host", true, false, Type::String, "127.0.0.1",
     0, 0, "HTTP bind address for `engine serve` (127.0.0.1 = local only, 0.0.0.0 = LAN)"},
    {"serve_port", "ENGINE_SERVE_PORT", "port", true, false, Type::Int, "7030",
     1, 65535, "HTTP port for `engine serve` (OpenAI + llama.cpp compatible API)"},
    {"serve_ctx", "ENGINE_SERVE_CTX", "serve-ctx", true, false, Type::Int, "0",
     0, 1 << 20,
     "context size used by `engine serve`; 0 = --ctx when given, else 4096 (KV is "
     "allocated at startup, so this is the per-request prompt+generation limit). "
     "/props reports n_ctx (actual) and n_ctx_requested (0 = auto)"},
    {"default_max_tokens", "ENGINE_DEFAULT_MAX_TOKENS", "default-max-tokens", true, false,
     Type::Int, "1048576", 1, 1 << 20,
     "serve: max_tokens when the request does not set it (default = unlimited within ctx; "
     "reasoning models emit a long thinking block before the answer)"},
    {"default_temperature", "ENGINE_DEFAULT_TEMPERATURE", "default-temperature", true, false,
     Type::Float, "0.8", 0.0, 10000.0,
     "serve: temperature when the request does not set it (0 = greedy)"},
    {"default_top_p", "ENGINE_DEFAULT_TOP_P", "default-top-p", true, false, Type::Float,
     "0.95", 0.0, 1.0, "serve: top_p when the request does not set it"},
    {"default_top_k", "ENGINE_DEFAULT_TOP_K", "default-top-k", true, false, Type::Int, "40",
     0, 1 << 20, "serve: top_k when the request does not set it (0 = off)"},
    {"default_repeat_penalty", "ENGINE_DEFAULT_REPEAT_PENALTY", "default-repeat-penalty",
     true, false, Type::Float, "1.0", 0.0, 10000.0,
     "serve: repetition_penalty when the request does not set it (1.0 = off)"},
    {"serve_prefix_cache", "ENGINE_SERVE_PREFIX_CACHE", "prefix-cache", false, false,
     Type::Bool, "true",
     0, 0,
     "serve: multi-turn prefix KV cache (reuse the shared conversation prefix instead of "
     "re-prefilling the whole chat every request; hybrid models also snapshot/restore the "
     "recurrent SSM state). --prefix-cache=0 disables it"},
    {"serve_cache_align", "ENGINE_SERVE_CACHE_ALIGN", "cache-align", true, false,
     Type::Int, "128", 0, 1 << 20,
     "serve: prefix-cache snapshot alignment in tokens (0 = follow --prefill-rows). "
     "A small alignment (e.g. 128) keeps the large prefill chunk for long prompts while "
     "multi-turn cache hits only re-prefill the new tokens; cached vs full-prefill "
     "outputs are verified token-for-token identical"},
    {"serve_repeat_guard", "ENGINE_SERVE_REPEAT_GUARD", "repeat-guard", true, false,
     Type::Int, "2", 0, 2,
     "serve: degenerate-loop guard level: 0=off, 1=sentence level (3 identical "
     "sentences in a row), 2=+character/phrase level (a 4..96-byte unit repeated 5x, "
     "or a 16..48-byte phrase repeated >=4x in the last 512 bytes of the answer); "
     "stops generation with finish_reason=repetition"},
    {"serve_think_budget", "ENGINE_SERVE_THINK_BUDGET", "think-budget", true, false,
     Type::Int, "512", 0, 1 << 20,
     "serve: max tokens generated inside an open <think> block before the engine force-closes "
     "it (forces </think> and continues with the answer; 0 = off). Stops Qwen3.x thinking "
     "models from burning the whole reply budget on self-narration"},
    {"serve_no_limit_tokens", "ENGINE_SERVE_NO_LIMIT_TOKENS", "no-limit-tokens", true, false,
     Type::Int, "1024", 1, 1 << 20,
     "serve: reply cap when the client asks for 'no limit' (max_tokens <= 0, the llama.cpp "
     "webui sends -1); avoids multi-minute replies when thinking is enabled"},

    // ---------------- backends / paths ----------------
    {"nvrtc_path", "DSH_NVRTC_PATH", "nvrtc-path", true, false, Type::String, "",
     0, 0, "path to nvrtc64_120_0.dll (default: pip nvidia-cuda-nvrtc location)"},
    {"cublas_dir", "DSH_CUBLAS_DIR", "cublas-dir", true, false, Type::String, "",
     0, 0, "directory containing cublas64_12.dll"},

    // ---------------- diagnostics / CLI commands ----------------
    {"debug", "RUNNER_DEBUG", "debug", false, false, Type::Bool, "false",
     0, 0, "verbose decision log (placement / kernel choice / memory)"},
    {"print_config", "ENGINE_PRINT_CONFIG", "print-config", false, false, Type::Bool, "false",
     0, 0, "print the effective configuration and kernel cache, then exit"},
    {"kernel_calibrate", "ENGINE_KERNEL_CALIBRATE", "kernel-calibrate", false, false, Type::Bool, "false",
     0, 0, "micro-benchmark the GEMV kernels on this model, write the cache, then exit"},
    {"kernel_cache", "ENGINE_KERNEL_CACHE", "kernel-cache", true, false, Type::String,
     "engine-kernel-cache.txt",
     0, 0, "path of the kernel calibration cache file"},
    {"calib_iters", "ENGINE_CALIB_ITERS", "calib-iters", true, false, Type::Int, "40",
     1, 10000, "timed iterations per kernel candidate during --kernel-calibrate"},
    {"config", nullptr, "config", true, false, Type::String, "",
     0, 0, "config file path (key = value; [kernel] section for calibration)"},

    // ---------------- generic architecture blocks (metadata driven) ----------------
    // These only affect models whose GGUF metadata/tensor layout asks for the
    // corresponding feature; the defaults are "use it if the metadata says so".
    {"arch_ext", "ENGINE_ARCH_EXT", "arch-ext", true, false, Type::String, "off",
     0, 0,
     "generic per-layer dense path: off (default; experimental paths disabled) | "
     "auto (metadata decides) | on"},
    {"arch_act", "ENGINE_ARCH_ACT", "arch-act", true, false, Type::String, "auto",
     0, 0, "MLP activation: auto | silu | gelu (gelu-tanh for gemma-family metadata)"},
    {"arch_embed_scale", "ENGINE_ARCH_EMBED_SCALE", "arch-embed-scale", true, false,
     Type::String, "auto",
     0, 0, "embedding multiplier: auto | none | sqrt (sqrt(hidden) for gemma-family)"},
    {"arch_attn_scale", "ENGINE_ARCH_ATTN_SCALE", "arch-attn-scale", true, false,
     Type::String, "auto",
     0, 0, "attention logit scale: auto | 1.0 | rsqrt (1/sqrt(head_dim))"},
    {"arch_norm_mode", "ENGINE_ARCH_NORM_MODE", "arch-norm-mode", true, false,
     Type::String, "auto",
     0, 0, "RMSNorm weight interpretation: auto | direct (w) | plus1 (1+w)"},
    {"arch_softcap", "ENGINE_ARCH_SOFTCAP", "arch-softcap", true, false, Type::String,
     "auto",
     0, 0, "final logit softcapping: auto (metadata) | on | off"},
};

constexpr size_t kNumDefs = sizeof(kDefs) / sizeof(kDefs[0]);

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

bool lower_eq(const std::string& a, const char* b) {
  if (a.size() != std::strlen(b)) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(a[i])) != b[i]) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Recursive config includes: a config file may name another config file with
// `config = <path>` (the child is applied on top of the parent).  There was no
// cycle guard, so a.cfg -> b.cfg -> a.cfg recursed until the process ran out of
// file handles (misleading "cannot open config file") or, on platforms with a
// larger handle limit, the native stack.  Track the include chain per thread and
// reject repeated paths / excessive depth with an explicit diagnostic.
// ---------------------------------------------------------------------------
constexpr int kMaxConfigIncludeDepth = 8;

std::string normalized_config_path(const std::string& path) {
  std::error_code ec;
  const std::filesystem::path abs = std::filesystem::absolute(path, ec);
  if (ec) return path;
  const std::filesystem::path canon = std::filesystem::weakly_canonical(abs, ec);
  return (ec ? abs.lexically_normal() : canon).string();
}

struct ConfigLoadScope {
  std::vector<std::string>* chain;
  ConfigLoadScope(std::vector<std::string>* c, std::string p) : chain(c) {
    chain->push_back(std::move(p));
  }
  ~ConfigLoadScope() { chain->pop_back(); }
};

}  // namespace

const char* source_name(Source s) {
  switch (s) {
    case Source::Default: return "default";
    case Source::Env: return "env";
    case Source::File: return "config";
    case Source::Cli: return "cli";
  }
  return "?";
}

Config& Config::get() {
  static Config c;
  return c;
}

Config::Config() {
  defs_.assign(kDefs, kDefs + kNumDefs);
}

const OptionDef* Config::find_def(const std::string& key) const {
  for (const auto& d : defs_)
    if (key == d.key) return &d;
  return nullptr;
}

const OptionDef* Config::def_of(const std::string& key) const { return find_def(key); }

bool Config::parse_bool_text(const std::string& text, bool* out) const {
  std::string t = trim(text);
  for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (t.empty() || t == "1" || t == "true" || t == "yes" || t == "on") {
    *out = true;
    return true;
  }
  if (t == "0" || t == "false" || t == "no" || t == "off") {
    *out = false;
    return true;
  }
  return false;
}

bool Config::apply_text(const OptionDef& def, const std::string& text, Source src,
                        std::string* err, const char* where) {
  const std::string t = trim(text);
  Item it;
  it.key = def.key;
  it.type = def.type;
  it.raw = t;
  it.src = src;
  switch (def.type) {
    case Type::Int: {
      if (t.empty()) {
        *err = std::string(where) + ": option '" + def.key + "' expects an integer";
        return false;
      }
      errno = 0;
      char* end = nullptr;
      const long long v = std::strtoll(t.c_str(), &end, 10);
      if (errno != 0 || end == t.c_str() || *end != '\0') {
        *err = std::string(where) + ": option '" + def.key + "' expects an integer, got '" + t + "'";
        return false;
      }
      if (static_cast<double>(v) < def.minv || static_cast<double>(v) > def.maxv) {
        *err = std::string(where) + ": option '" + def.key + "' value " + std::to_string(v) +
               " out of range [" + std::to_string(def.minv) + ", " + std::to_string(def.maxv) +
               "]";
        return false;
      }
      it.i = v;
      it.f = static_cast<double>(v);
      it.b = v != 0;
      break;
    }
    case Type::Float: {
      errno = 0;
      char* end = nullptr;
      const double v = std::strtod(t.c_str(), &end);
      if (errno != 0 || end == t.c_str() || *end != '\0') {
        *err = std::string(where) + ": option '" + def.key + "' expects a number, got '" + t + "'";
        return false;
      }
      // strtod("nan") succeeds and every comparison with NaN is false, so the old
      // range check let NaN through and the following (int64_t) cast was UB (observed
      // payload: it.i == INT64_MIN).  Reject non-finite values explicitly.
      if (!std::isfinite(v)) {
        *err = std::string(where) + ": option '" + def.key +
               "' expects a finite number, got '" + t + "'";
        return false;
      }
      if (v < def.minv || v > def.maxv) {
        *err = std::string(where) + ": option '" + def.key + "' value out of range";
        return false;
      }
      it.f = v;
      it.i = static_cast<int64_t>(v);
      it.b = v != 0.0;
      break;
    }
    case Type::Bool: {
      bool b = true;
      if (!t.empty() && !parse_bool_text(t, &b)) {
        *err = std::string(where) + ": option '" + def.key +
               "' expects true/false (also 1/0, yes/no), got '" + t + "'";
        return false;
      }
      it.b = b;
      it.i = b ? 1 : 0;
      it.f = b ? 1.0 : 0.0;
      break;
    }
    case Type::String:
      it.b = !t.empty();
      break;
  }
  items_[def.key] = it;
  return true;
}

void Config::ensure_initialized() const {
  if (env_loaded_) return;
  Config* self = const_cast<Config*>(this);
  self->env_loaded_ = true;
  // 1) defaults
  for (const auto& d : self->defs_) {
    Item it;
    it.key = d.key;
    it.type = d.type;
    it.raw = d.def;
    it.src = Source::Default;
    // Seed typed fields so a corrupt default string cannot leave them zeroed.
    bool ok = true;
    switch (d.type) {
      case Type::Int: {
        char* end = nullptr;
        const long long v = std::strtoll(d.def, &end, 10);
        if (!end || *end != '\0') ok = false;
        it.i = v;
        it.f = static_cast<double>(v);
        it.b = v != 0;
        break;
      }
      case Type::Float: {
        char* end = nullptr;
        const double v = std::strtod(d.def, &end);
        if (!end || *end != '\0') ok = false;
        it.f = v;
        it.i = static_cast<int64_t>(v);
        it.b = v != 0.0;
        break;
      }
      case Type::Bool: {
        bool b = true;
        ok = self->parse_bool_text(d.def, &b);
        it.b = b;
        it.i = b ? 1 : 0;
        it.f = b ? 1.0 : 0.0;
        break;
      }
      case Type::String:
        it.b = d.def && *d.def;
        break;
    }
    (void)ok;
    self->items_[d.key] = it;
  }
  // 2) environment (compatibility layer)
  for (const auto& d : self->defs_) {
    if (!d.env) continue;
    const char* e = std::getenv(d.env);
    if (!e) continue;
    std::string err;
    if (d.type == Type::Bool) {
      bool v = true;
      if (d.env_invert) {
        v = false;  // presence of ENGINE_NO_* means "disable"
      } else if (*e) {
        bool parsed = true;
        if (!self->parse_bool_text(e, &parsed)) parsed = true;
        v = parsed;
      }
      Item& it = self->items_[d.key];
      it.b = v;
      it.i = v ? 1 : 0;
      it.f = v ? 1.0 : 0.0;
      it.raw = v ? "true" : "false";
      it.src = Source::Env;
    } else {
      if (!self->apply_text(d, e, Source::Env, &err, "environment")) {
        throw std::runtime_error(err);
      }
    }
  }
}

bool Config::load_file(const std::string& path, std::string* err) {
  ensure_initialized();
  static thread_local std::vector<std::string> load_chain;
  const std::string norm_path = normalized_config_path(path);
  for (const auto& loaded : load_chain) {
    if (loaded == norm_path) {
      std::string chain;
      for (const auto& p : load_chain) chain += p + " -> ";
      *err = "config include cycle detected: " + chain + norm_path;
      return false;
    }
  }
  if (static_cast<int>(load_chain.size()) >= kMaxConfigIncludeDepth) {
    *err = "config include depth exceeds " + std::to_string(kMaxConfigIncludeDepth) +
           " (check the 'config = <file>' chain)";
    return false;
  }
  std::ifstream f(path);
  if (!f) {
    *err = "cannot open config file '" + path + "'";
    return false;
  }
  loaded_file_ = path;
  ConfigLoadScope load_scope(&load_chain, norm_path);
  std::string line;
  int lineno = 0;
  bool kernel_section = false;
  while (std::getline(f, line)) {
    ++lineno;
    std::string s = trim(line);
    if (s.empty() || s[0] == '#' || s[0] == ';') continue;
    if (s.size() >= 2 && s.front() == '[' && s.back() == ']') {
      const std::string sec = trim(s.substr(1, s.size() - 2));
      if (sec == "kernel" || sec == "kernel_cache") {
        kernel_section = true;
      } else if (sec == "engine" || sec == "main") {
        kernel_section = false;
      } else {
        *err = path + ":" + std::to_string(lineno) + ": unknown section [" + sec +
               "] (expected [engine] or [kernel])";
        return false;
      }
      continue;
    }
    const size_t eq = s.find('=');
    if (eq == std::string::npos) {
      *err = path + ":" + std::to_string(lineno) + ": expected 'key = value', got '" + s + "'";
      return false;
    }
    const std::string key = trim(s.substr(0, eq));
    const std::string val = trim(s.substr(eq + 1));
    if (key.empty()) {
      *err = path + ":" + std::to_string(lineno) + ": empty key";
      return false;
    }
    if (kernel_section) {
      kernel_over_[key] = val;
      continue;
    }
    const OptionDef* def = find_def(key);
    if (!def) {
      *err = path + ":" + std::to_string(lineno) + ": unknown config key '" + key +
             "' (see --help for the list of options)";
      return false;
    }
    if (!apply_text(*def, val, Source::File, err,
                    (path + ":" + std::to_string(lineno)).c_str()))
      return false;
    if (def->cli && std::strcmp(def->cli, "config") == 0) {
      // A config file naming another config file: apply it on top.  The recursive
      // call is guarded by the include chain above (cycles / depth are rejected).
      if (!val.empty() && !load_file(val, err)) return false;
    }
  }
  return true;
}

bool Config::parse_command_line(int argc, char** argv, int first,
                                std::vector<std::string>* positional,
                                std::vector<std::string>* unparsed, std::string* err) {
  ensure_initialized();
  positional->clear();
  unparsed->clear();
  // pass 1: --config (config file outranks the environment)
  std::string cfg_path;
  for (int i = first; i < argc; ++i) {
    std::string a = argv[i];
    std::string val;
    bool has_val = false;
    const size_t eq = a.find('=');
    if (eq != std::string::npos && a.compare(0, 2, "--") == 0) {
      val = a.substr(eq + 1);
      a = a.substr(0, eq);
      has_val = true;
    }
    if (a == "--config") {
      if (has_val) {
        cfg_path = val;
      } else if (i + 1 < argc) {
        cfg_path = argv[++i];
      } else {
        *err = "missing value for --config";
        return false;
      }
    }
  }
  if (!cfg_path.empty())
    if (!load_file(cfg_path, err)) return false;

  // pass 2: every other option (outranks the file)
  for (int i = first; i < argc; ++i) {
    std::string a = argv[i];
    if (a.size() >= 2 && a[0] == '-' && a[1] == '-') {
      std::string val;
      bool has_val = false;
      const size_t eq = a.find('=');
      if (eq != std::string::npos) {
        val = a.substr(eq + 1);
        a = a.substr(0, eq);
        has_val = true;
      }
      const std::string name = a.substr(2);
      if (name == "config") {
        if (!has_val && i + 1 < argc) ++i;
        continue;
      }
      if (name == "help") {
        unparsed->push_back("--help");
        continue;
      }
      const OptionDef* def = nullptr;
      for (const auto& d : defs_) {
        if (d.cli && name == d.cli) {
          def = &d;
          break;
        }
      }
      if (!def) {
        unparsed->push_back(a);
        continue;
      }
      if (!def->cli_takes_value) {
        if (has_val) {
          if (!apply_text(*def, val, Source::Cli, err, "command line")) return false;
        } else {
          // A "no-xxx" switch means "turn xxx off".  For keys that are already
          // negative (no_graph, no_yarn, ...) the switch turns them on, which is
          // how the historical environment variables behaved.
          bool v = true;
          const bool neg_flag = name.compare(0, 3, "no-") == 0;
          const bool neg_key = std::strncmp(def->key, "no_", 3) == 0;
          if (neg_flag && !neg_key) v = false;
          std::string e2;
          if (!apply_text(*def, v ? "true" : "false", Source::Cli, &e2, "command line")) {
            *err = e2;
            return false;
          }
        }
        continue;
      }
      if (!has_val) {
        if (i + 1 >= argc) {
          *err = "missing value for --" + name;
          return false;
        }
        val = argv[++i];
      }
      // convenience: --gpu-layers auto
      if (def->type == Type::Int && lower_eq(val, "auto")) val = "-1";
      if (!apply_text(*def, val, Source::Cli, err, "command line")) return false;
    } else {
      positional->push_back(a);
    }
  }
  return true;
}

bool Config::set(const std::string& key, const std::string& value, Source src, std::string* err) {
  ensure_initialized();
  const OptionDef* def = find_def(key);
  if (!def) {
    *err = "unknown config key '" + key + "'";
    return false;
  }
  return apply_text(*def, value, src, err, "set");
}

int64_t Config::get_int(const std::string& key) const {
  ensure_initialized();
  auto it = items_.find(key);
  if (it == items_.end()) throw std::runtime_error("config: unknown key '" + key + "'");
  return it->second.i;
}
double Config::get_double(const std::string& key) const {
  ensure_initialized();
  auto it = items_.find(key);
  if (it == items_.end()) throw std::runtime_error("config: unknown key '" + key + "'");
  return it->second.f;
}
bool Config::get_bool(const std::string& key) const {
  ensure_initialized();
  auto it = items_.find(key);
  if (it == items_.end()) throw std::runtime_error("config: unknown key '" + key + "'");
  return it->second.b;
}
std::string Config::get_string(const std::string& key) const {
  ensure_initialized();
  auto it = items_.find(key);
  if (it == items_.end()) throw std::runtime_error("config: unknown key '" + key + "'");
  return it->second.raw;
}
Source Config::source_of(const std::string& key) const {
  ensure_initialized();
  auto it = items_.find(key);
  if (it == items_.end()) throw std::runtime_error("config: unknown key '" + key + "'");
  return it->second.src;
}
bool Config::was_set(const std::string& key) const {
  ensure_initialized();
  auto it = items_.find(key);
  if (it == items_.end()) throw std::runtime_error("config: unknown key '" + key + "'");
  return it->second.src != Source::Default;
}

void Config::set_kernel_override(const std::string& shape, const std::string& plan) {
  kernel_over_[shape] = plan;
}
const std::string* Config::find_kernel_override(const std::string& shape) const {
  if (kernel_over_.empty()) return nullptr;
  auto it = kernel_over_.find(shape);
  return it == kernel_over_.end() ? nullptr : &it->second;
}

bool Config::load_kernel_cache(const std::string& path, std::string* err) {
  std::ifstream f(path);
  if (!f) {
    *err = "cannot open kernel cache '" + path + "'";
    return false;
  }
  std::string line;
  int lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    std::string s = trim(line);
    if (s.empty() || s[0] == '#' || s[0] == ';') continue;
    if (s.size() >= 2 && s.front() == '[' && s.back() == ']') continue;  // sections ignored
    const size_t eq = s.find('=');
    if (eq == std::string::npos) continue;
    kernel_over_[trim(s.substr(0, eq))] = trim(s.substr(eq + 1));
  }
  return true;
}

bool Config::save_kernel_cache(const std::string& path, std::string* err) const {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    *err = "cannot write kernel cache '" + path + "'";
    return false;
  }
  f << "# engine kernel calibration cache (generated by --kernel-calibrate)\n";
  f << "# shape = GEMV plan; delete this file or rerun calibration to refresh\n";
  f << "[kernel]\n";
  for (const auto& [shape, plan] : kernel_over_) f << shape << " = " << plan << "\n";
  if (!f) {
    *err = "write failed for kernel cache '" + path + "'";
    return false;
  }
  return true;
}

int Config::load_kernel_cache_if_exists(std::string* err) {
  ensure_initialized();
  const std::string path = get_string("kernel_cache");
  if (path.empty()) return 0;
  std::ifstream f(path);
  if (!f) return 0;
  return load_kernel_cache(path, err) ? 1 : -1;
}

void Config::print_effective(FILE* f) const {
  std::fprintf(f, "== effective configuration ==\n");
  std::fprintf(f, "priority: cli > config > env > default");
  if (!loaded_file_.empty()) std::fprintf(f, "   (config file: %s)", loaded_file_.c_str());
  std::fprintf(f, "\n");
  size_t w = 0;
  for (const auto& d : defs_) w = std::max(w, std::strlen(d.key));
  for (const auto& d : defs_) {
    const auto& it = items_.at(d.key);
    std::string val = it.raw;
    if (it.type == Type::Bool) val = it.b ? "true" : "false";
    std::fprintf(f, "  %-*s = %-22s [%s]  %s\n", static_cast<int>(w), d.key, val.c_str(),
                 source_name(it.src), d.help);
  }
  std::fprintf(f, "== kernel calibration cache ==\n");
  if (!kernel_over_.empty()) {
    std::fprintf(f, "  path: %s (%zu entries)\n", items_.at("kernel_cache").raw.c_str(),
                 kernel_over_.size());
    for (const auto& [shape, plan] : kernel_over_)
      std::fprintf(f, "    %-32s -> %s\n", shape.c_str(), plan.c_str());
  } else {
    std::fprintf(f, "  (none; run --kernel-calibrate to measure this GPU)\n");
  }
  std::fprintf(f, "== end ==\n");
}

void Config::print_options(FILE* f) const {
  std::fprintf(f,
               "usage: <tool> --model <path> [options]\n"
               "       <tool> --config <file> ...\n\n"
               "priority: cli > config file > environment > default\n\n");
  for (const auto& d : defs_) {
    std::string line = "  ";
    line += d.cli ? std::string("--") + d.cli : std::string("  ");
    if (d.cli && d.cli_takes_value) line += " <v>";
    line += std::string("  (config: ") + d.key + ")";
    std::fprintf(f, "%-34s %s\n", line.c_str(), d.help);
    if (d.env) std::fprintf(f, "%-34s env: %s%s\n", "", d.env, d.env_invert ? " (presence disables)" : "");
    std::fprintf(f, "%-34s default: %s\n", "", d.def);
  }
}

// Machine-readable config surface for the GUI launcher (output only).  The
// launcher builds its "advanced" panel from this list, so a new option in
// kDefs[] shows up in the GUI without any GUI change.
void Config::print_schema_json(FILE* f) const {
  static const char* kTypeNames[] = {"int", "float", "bool", "string"};
  json::Writer w;
  w.begin_obj();
  w.key("engine"); w.str("engine");
  w.key("priority"); w.str("cli > config file > environment > default");
  w.key("count"); w.integer(static_cast<int64_t>(defs_.size()));
  w.key("options"); w.begin_arr();
  for (const auto& d : defs_) {
    const auto it = items_.find(d.key);
    w.begin_obj();
    w.key("key"); w.str(d.key);
    w.key("cli"); if (d.cli != nullptr) w.str(d.cli); else w.null();
    w.key("env"); if (d.env != nullptr) w.str(d.env); else w.null();
    w.key("takes_value"); w.boolean(d.cli_takes_value);
    w.key("invert_env"); w.boolean(d.env_invert);
    w.key("type"); w.str(kTypeNames[static_cast<int>(d.type)]);
    w.key("def"); w.str(d.def);
    w.key("min"); w.number(d.minv);
    w.key("max"); w.number(d.maxv);
    w.key("help"); w.str(d.help);
    if (it != items_.end()) {
      if (d.type == Type::Bool) {
        w.key("value"); w.boolean(it->second.b);
      } else {
        w.key("value"); w.str(it->second.raw);
      }
      w.key("source"); w.str(source_name(it->second.src));
    } else {
      w.key("value"); w.null();
      w.key("source"); w.str("default");
    }
    w.end_obj();
  }
  w.end_arr();
  w.end_obj();
  std::fwrite(w.out.data(), 1, w.out.size(), f);
  std::fputc('\n', f);
}

void apply_win_priority() {
#if defined(_WIN32)
  static bool applied = false;
  if (applied) return;
  applied = true;
  const int64_t lvl = get_int("win_priority");
  // 0 = normal.  Set it explicitly (a child inherits its parent's class, so a benchmark
  // started from a HIGH shell would otherwise not have a real "normal" arm).
  DWORD cls = NORMAL_PRIORITY_CLASS;
  const char* name = "NORMAL";
  if (lvl >= 1) {
    if (lvl >= 2) {
      cls = ABOVE_NORMAL_PRIORITY_CLASS;
      name = "ABOVE_NORMAL";
    } else {
      cls = HIGH_PRIORITY_CLASS;
      name = "HIGH";
    }
  }
  if (!SetPriorityClass(GetCurrentProcess(), cls)) {
    std::fprintf(stderr, "[win] SetPriorityClass(%s) failed: error %lu\n", name,
                 static_cast<unsigned long>(GetLastError()));
    return;
  }
  if (lvl > 0) {
    // One line of evidence in every log; it is cheap and makes the measurements
    // self-describing (the timing scripts grep for it).
    std::fprintf(stderr, "[win] process priority class = %s (win_priority=%lld)\n", name,
                 static_cast<long long>(lvl));
  }
#else
  // Non-Windows: nothing to do; the config key stays accepted so scripts are portable.
#endif
}

}  // namespace config
}  // namespace engine
