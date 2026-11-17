// engine CLI: info | ir | plan | run
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#define _isatty isatty
#define _fileno fileno
#endif

#include "engine/config.h"
#include "engine/cuda.h"
#include "engine/gguf.h"
#include "engine/ir.h"
#include "engine/json.h"
#include "engine/model_scan.h"
#include "engine/placement.h"
#include "engine/planner.h"
#include "engine/runner.h"
#include "engine/serve.h"

namespace {

std::string human(uint64_t n) {
  const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = static_cast<double>(n);
  int u = 0;
  while (v >= 1024.0 && u < 4) {
    v /= 1024.0;
    ++u;
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f %s", v, units[u]);
  return buf;
}

// Placement helpers are defined further down (after usage()); forward-declare the
// ones `plan` needs so the budget planner and the launcher share one decider.
int kv_type_from_name(const std::string& raw);
int kv_type_from_cfg();
uint64_t placement_reserve_bytes();
uint64_t host_reserve_bytes();
placement::Decision compute_decision(const ir::Model& m, int ctx, int kv_type,
                                     const placement::MemoryInfo& vram,
                                     const placement::MemoryInfo& host);

struct Loaded {
  ir::Model model;
  std::vector<std::string> errors;
  double parse_ms = 0;
};

Loaded load(const std::string& path) {
  Loaded l;
  auto t0 = std::chrono::steady_clock::now();
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) {
    l.model = ir::build_ir_from_hf(path);
  } else {
    gguf::Model g = gguf::read(path);
    l.model = ir::build_from_gguf(g);
  }
  auto t1 = std::chrono::steady_clock::now();
  l.parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  l.errors = ir::validate(l.model);
  return l;
}

int cmd_info(const std::string& path, bool json_mode) {
  Loaded l = load(path);
  const auto& m = l.model;
  if (json_mode) {
    // Output-only JSON view used by build/launcher.exe: model metadata + the
    // byte accounting the GUI's memory estimate needs (per-layer expert bytes,
    // KV bytes per token for every kv_type) + the current free VRAM / host RAM.
    json::Writer w;
    w.begin_obj();
    w.key("model"); w.str(m.path);
    w.key("arch"); w.str(m.arch);
    w.key("container"); w.str(m.container);
    w.key("file_size"); w.uinteger(m.file_size);
    w.key("n_layer"); w.integer(m.n_layer);
    w.key("hidden"); w.integer(m.hidden);
    w.key("ffn"); w.integer(m.ffn);
    w.key("vocab"); w.uinteger(m.vocab);
    w.key("ctx_train"); w.integer(m.ctx_train);
    w.key("attn"); w.begin_obj();
    w.key("variant"); w.str(m.attn.variant);
    w.key("n_head"); w.integer(m.attn.n_head);
    w.key("n_head_kv"); w.integer(m.attn.n_head_kv);
    w.key("key_length"); w.integer(m.attn.key_length);
    w.key("value_length"); w.integer(m.attn.value_length);
    w.key("rope_freq_base"); w.number(m.attn.rope_freq_base);
    w.key("full_attention_interval"); w.integer(m.attn.full_attention_interval);
    w.key("kv_lora_rank"); w.integer(m.attn.kv_lora_rank);
    w.end_obj();
    w.key("moe"); w.begin_obj();
    w.key("n_expert"); w.integer(m.moe.n_expert);
    w.key("n_expert_used"); w.integer(m.moe.n_expert_used);
    w.key("n_shared"); w.integer(m.moe.n_shared);
    w.key("expert_ffn_length"); w.integer(m.moe.expert_ffn_length);
    w.key("layout"); w.str(m.moe.layout);
    w.end_obj();
    w.key("bytes"); w.begin_obj();
    w.key("total"); w.uinteger(m.total_bytes());
    w.key("fixed"); w.uinteger(m.fixed_bytes());
    w.key("experts"); w.uinteger(m.expert_bytes());
    w.end_obj();
    w.key("expert_per_layer"); w.begin_obj();
    for (const auto& [layer, bytes] : m.expert_bytes_per_layer()) {
      w.key(std::to_string(layer)); w.uinteger(bytes);
    }
    w.end_obj();
    w.key("kv_layers"); w.integer(placement::kv_layer_count(m));
    w.key("kv_per_token"); w.begin_obj();
    w.key("f32"); w.uinteger(placement::kv_bytes_per_token(m, 0));
    w.key("q8_0"); w.uinteger(placement::kv_bytes_per_token(m, 1));
    w.key("q4_0"); w.uinteger(placement::kv_bytes_per_token(m, 2));
    w.end_obj();
    w.key("mtp_layers"); w.integer(m.mtp_layers);
    w.key("quant_mix"); w.begin_obj();
    for (const auto& [name, bytes] : m.quant_mix) { w.key(name); w.uinteger(bytes); }
    w.end_obj();
    w.key("validation"); w.str(l.errors.empty() ? "PASS" : "FAIL");
    w.key("errors"); w.begin_arr();
    for (const auto& e : l.errors) w.str(e);
    w.end_arr();
    w.key("warnings"); w.begin_arr();
    for (const auto& x : m.warnings) w.str(x);
    w.end_arr();
    w.key("parse_ms"); w.number(l.parse_ms);
    // Environment probes (they do not depend on the model, but `info` is the one
    // command the launcher already runs per model selection: one process, one
    // read of the machine state).  Output only: no inference path consumes it.
    const placement::MemoryInfo vram = placement::device_memory();
    const placement::MemoryInfo host = placement::host_memory();
    w.key("system"); w.begin_obj();
    w.key("vram_valid"); w.boolean(vram.valid);
    w.key("vram_total"); w.uinteger(vram.total_bytes);
    w.key("vram_free"); w.uinteger(vram.free_bytes);
    w.key("host_valid"); w.boolean(host.valid);
    w.key("host_total"); w.uinteger(host.total_bytes);
    w.key("host_free"); w.uinteger(host.free_bytes);
    w.end_obj();
    w.end_obj();
    std::fwrite(w.out.data(), 1, w.out.size(), stdout);
    std::fputc('\n', stdout);
    return l.errors.empty() ? 0 : 2;
  }
  std::printf("model      : %s\n", m.path.c_str());
  std::printf("arch       : %s   container=%s  file=%s\n", m.arch.c_str(),
              m.container.c_str(), human(m.file_size).c_str());
  std::printf("layers     : %d   hidden=%d  ffn=%d  vocab=%llu\n", m.n_layer, m.hidden,
              m.ffn, static_cast<unsigned long long>(m.vocab));
  std::printf("attn       : variant=%s heads=%d/%d klen=%d vlen=%d rope_base=%.0f\n",
              m.attn.variant.c_str(), m.attn.n_head, m.attn.n_head_kv, m.attn.key_length,
              m.attn.value_length, m.attn.rope_freq_base);
  if (m.moe.n_expert)
    std::printf("moe        : experts=%d used=%d shared=%d inter=%d\n", m.moe.n_expert,
                m.moe.n_expert_used, m.moe.n_shared, m.moe.expert_ffn_length);
  std::printf("tokenizer  : %s pre='%s' vocab=%llu chat_template=%d\n", m.tok.model.c_str(),
              m.tok.pre.c_str(), static_cast<unsigned long long>(m.tok.n_vocab),
              m.tok.chat_template ? 1 : 0);
  std::printf("bytes      : total=%s fixed=%s experts=%s\n", human(m.total_bytes()).c_str(),
              human(m.fixed_bytes()).c_str(), human(m.expert_bytes()).c_str());

  std::vector<std::pair<std::string, uint64_t>> mix(m.quant_mix.begin(), m.quant_mix.end());
  std::sort(mix.begin(), mix.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string mixs;
  for (size_t i = 0; i < mix.size() && i < 6; ++i)
    mixs += (i ? ", " : "") + mix[i].first + "=" + human(mix[i].second);
  std::printf("quant mix  : %s\n", mixs.c_str());

  std::map<std::string, int> roles;
  for (const auto& t : m.tensors) roles[ir::role_name(t.role)]++;
  std::string rs;
  for (const auto& [k, v] : roles) rs += k + ":" + std::to_string(v) + " ";
  std::printf("roles      : %s\n", rs.c_str());
  if (l.errors.empty()) {
    std::printf("VALIDATION : PASS\n");
  } else {
    std::printf("VALIDATION : FAIL (%zu errors)\n", l.errors.size());
    for (size_t i = 0; i < l.errors.size() && i < 10; ++i)
      std::printf("  - %s\n", l.errors[i].c_str());
  }
  for (const auto& w : m.warnings) std::printf("warn       : %s\n", w.c_str());
  std::printf("warn       : parsed in %.1f ms\n", l.parse_ms);
  return l.errors.empty() ? 0 : 2;
}

void write_ir_json(json::Writer& w, const ir::Model& m, bool with_tensors) {
  w.begin_obj();
  w.key("arch"); w.str(m.arch);
  w.key("container"); w.str(m.container);
  w.key("path"); w.str(m.path);
  w.key("file_size"); w.uinteger(m.file_size);
  w.key("n_layer"); w.integer(m.n_layer);
  w.key("hidden"); w.integer(m.hidden);
  w.key("ffn"); w.integer(m.ffn);
  w.key("vocab"); w.uinteger(m.vocab);
  w.key("ctx_train"); w.integer(m.ctx_train);

  w.key("quant_mix"); w.begin_obj();
  for (const auto& [k, v] : m.quant_mix) { w.key(k); w.uinteger(v); }
  w.end_obj();

  w.key("attn"); w.begin_obj();
  w.key("variant"); w.str(m.attn.variant);
  w.key("n_head"); w.integer(m.attn.n_head);
  w.key("n_head_kv"); w.integer(m.attn.n_head_kv);
  w.key("key_length"); w.integer(m.attn.key_length);
  w.key("value_length"); w.integer(m.attn.value_length);
  w.key("rope_freq_base"); w.number(m.attn.rope_freq_base);
  w.key("rope_dim"); w.integer(m.attn.rope_dim);
  w.key("sliding_window"); w.integer(m.attn.sliding_window);
  w.key("full_attention_interval"); w.integer(m.attn.full_attention_interval);
  w.key("kv_lora_rank"); w.integer(m.attn.kv_lora_rank);
  w.key("leading_dense_blocks"); w.integer(m.attn.leading_dense_blocks);
  w.end_obj();

  w.key("moe"); w.begin_obj();
  w.key("n_expert"); w.integer(m.moe.n_expert);
  w.key("n_expert_used"); w.integer(m.moe.n_expert_used);
  w.key("n_shared"); w.integer(m.moe.n_shared);
  w.key("router_kind"); w.str(m.moe.router_kind);
  w.key("norm_topk"); w.boolean(m.moe.norm_topk);
  w.key("expert_ffn_length"); w.integer(m.moe.expert_ffn_length);
  w.key("layout"); w.str(m.moe.layout);
  w.end_obj();

  if (m.ssm.variant != "none") {
    w.key("ssm"); w.begin_obj();
    w.key("variant"); w.str(m.ssm.variant);
    w.key("conv_kernel"); w.integer(m.ssm.conv_kernel);
    w.key("n_key_head"); w.integer(m.ssm.n_key_head);
    w.key("n_value_head"); w.integer(m.ssm.n_value_head);
    w.key("key_head_dim"); w.integer(m.ssm.key_head_dim);
    w.key("value_head_dim"); w.integer(m.ssm.value_head_dim);
    w.key("key_dim"); w.integer(m.ssm.key_dim);
    w.key("value_dim"); w.integer(m.ssm.value_dim);
    w.key("conv_dim"); w.integer(m.ssm.conv_dim);
    w.key("time_step_rank"); w.integer(m.ssm.time_step_rank);
    w.end_obj();
  }

  w.key("tokenizer"); w.begin_obj();
  w.key("model"); w.str(m.tok.model);
  w.key("pre"); w.str(m.tok.pre);
  w.key("n_vocab"); w.uinteger(m.tok.n_vocab);
  w.key("eos_id"); w.integer(m.tok.eos_id);
  w.key("chat_template"); w.boolean(m.tok.chat_template);
  w.end_obj();

  w.key("bytes"); w.begin_obj();
  w.key("total"); w.uinteger(m.total_bytes());
  w.key("fixed"); w.uinteger(m.fixed_bytes());
  w.key("experts"); w.uinteger(m.expert_bytes());
  w.key("kv_per_token_bpe2"); w.uinteger(m.kv_bytes_per_token(2));
  w.end_obj();

  w.key("mtp_layers"); w.integer(m.mtp_layers);
  w.key("warnings"); w.begin_arr();
  for (const auto& x : m.warnings) w.str(x);
  w.end_arr();

  w.key("layers"); w.begin_arr();
  for (const auto& L : m.layers) {
    w.begin_obj();
    w.key("index"); w.integer(L.index);
    w.key("n_tensors"); w.integer(L.n_tensors);
    w.key("n_expert_tensors"); w.integer(L.n_expert_tensors);
    w.key("kind"); w.str(L.kind);
    w.end_obj();
  }
  w.end_arr();

  if (with_tensors) {
    w.key("tensors"); w.begin_arr();
    for (const auto& t : m.tensors) {
      w.begin_obj();
      w.key("name"); w.str(t.name);
      w.key("shape"); w.begin_arr();
      for (auto d : t.shape) w.uinteger(d);
      w.end_arr();
      w.key("dtype"); w.str(t.dtype);
      w.key("role"); w.str(ir::role_name(t.role));
      w.key("layer");
      if (t.layer < 0) w.null(); else w.integer(t.layer);
      w.key("offset"); w.uinteger(t.file_offset);
      w.key("nbytes"); w.uinteger(t.nbytes);
      w.key("bpp"); w.number(t.quant.bpp());
      w.end_obj();
    }
    w.end_arr();
  }
  w.end_obj();
}

int cmd_ir(const std::string& path, const std::string& out, bool with_tensors, bool force) {
  Loaded l = load(path);
  if (!l.errors.empty() && !force) {
    std::fprintf(stderr, "refusing to dump IR: validation failed\n");
    for (const auto& e : l.errors) std::fprintf(stderr, "  - %s\n", e.c_str());
    return 2;
  }
  json::Writer w;
  write_ir_json(w, l.model, with_tensors);
  if (out.empty()) {
    std::fwrite(w.out.data(), 1, w.out.size(), stdout);
    std::fputc('\n', stdout);
  } else {
    std::ofstream f(out, std::ios::binary);
    if (!f.is_open()) {
      std::fprintf(stderr, "error: cannot open %s for writing (errno=%d: %s)\n", out.c_str(),
                   errno, std::strerror(errno));
      return 3;
    }
    f.write(w.out.data(), static_cast<std::streamsize>(w.out.size()));
    f.close();
    if (!f) {
      std::fprintf(stderr, "error: failed while writing %s\n", out.c_str());
      return 3;
    }
    std::printf("wrote %s (%zu bytes)\n", out.c_str(), w.out.size());
  }
  return 0;
}

int cmd_plan(const std::string& path, int ctx, double vram_gib, bool vram_given, int kv_bpe,
             const std::string& json_out) {
  Loaded l = load(path);
  const auto& m = l.model;
  auto p = planner::compute(m, ctx, kv_bpe, static_cast<uint64_t>(vram_gib * (1ull << 30)));
  if (!json_out.empty()) {
    json::Writer w;
    w.begin_obj();
    w.key("model"); w.str(m.path);
    w.key("ctx"); w.integer(p.ctx);
    w.key("kv_bpe"); w.integer(p.kv_bpe);
    w.key("vram_total"); w.uinteger(p.vram_total);
    w.key("vram_reserve"); w.uinteger(p.vram_reserve);
    w.key("fixed_bytes"); w.uinteger(p.fixed_bytes);
    w.key("kv_bytes"); w.uinteger(p.kv_bytes);
    w.key("workspace_bytes"); w.uinteger(p.workspace_bytes);
    w.key("expert_budget"); w.uinteger(p.expert_budget);
    w.key("expert_total"); w.uinteger(p.expert_total);
    w.key("expert_per_layer"); w.begin_obj();
    for (const auto& [k, v] : p.expert_per_layer) { w.key(std::to_string(k)); w.uinteger(v); }
    w.end_obj();
    w.key("slots_per_layer"); w.integer(p.slots_per_layer);
    w.key("layers_on_gpu"); w.integer(p.layers_on_gpu);
    w.key("layers_on_cpu"); w.integer(p.layers_on_cpu);
    w.key("fully_resident"); w.boolean(p.fully_resident);
    w.key("proj_decode_cpu_ms"); w.number(p.proj_decode_cpu_ms);
    w.key("proj_decode_pcie_ms"); w.number(p.proj_decode_pcie_ms);
    w.key("notes"); w.begin_arr();
    for (const auto& n : p.notes) w.str(n);
    w.end_arr();
    w.end_obj();
    std::ofstream f(json_out, std::ios::binary);
    if (!f.is_open()) {
      std::fprintf(stderr, "error: cannot open %s for writing\n", json_out.c_str());
      return 3;
    }
    f << w.out;
    f.close();
    std::printf("wrote %s (%zu bytes)\n", json_out.c_str(), w.out.size());
    return 0;
  }
  std::printf("model        : %s\n", m.path.c_str());
  std::printf("ctx=%d kv_bpe=%d vram=%.1f GiB\n", p.ctx, p.kv_bpe, vram_gib);
  std::printf("fixed        : %s\n", human(p.fixed_bytes).c_str());
  std::printf("kv cache     : %s\n", human(p.kv_bytes).c_str());
  std::printf("workspace    : %s\n", human(p.workspace_bytes).c_str());
  std::printf("expert budget: %s / total experts %s\n", human(p.expert_budget).c_str(),
              human(p.expert_total).c_str());
  if (m.moe.n_expert) {
    std::printf("slots/layer  : %d\n", p.slots_per_layer);
    std::printf("layers       : %d on GPU / %d on CPU  (fully_resident=%s)\n",
                p.layers_on_gpu, p.layers_on_cpu, p.fully_resident ? "true" : "false");
  } else {
    const uint64_t need = p.fixed_bytes + p.kv_bytes + p.workspace_bytes;
    const uint64_t usable = p.vram_total > p.vram_reserve ? p.vram_total - p.vram_reserve : 0;
    std::printf("dense model  : no experts; resident need = %s (vram usable = %s)\n",
                human(need).c_str(), human(usable).c_str());
    std::printf("fits         : %s\n", need <= usable ? "true" : "false");
  }
  std::printf("projection   : cpu-lane %.1f ms/token, pcie %.1f ms/token\n",
              p.proj_decode_cpu_ms, p.proj_decode_pcie_ms);
  for (const auto& n : p.notes) std::printf("note         : %s\n", n.c_str());

  // ---- placement decision (same decider the launcher runs before loading) ----
  // --vram-gib simulates the device budget (offline planning); otherwise the real
  // free VRAM / host RAM are measured.
  placement::MemoryInfo vram;
  if (vram_given) {
    vram.valid = true;
    vram.total_bytes = vram.free_bytes = static_cast<uint64_t>(vram_gib * (1ull << 30));
  } else {
    vram = placement::device_memory();
  }
  const placement::MemoryInfo host = placement::host_memory();
  if (!vram.valid) {
    std::printf("[placement] skipped: CUDA/VRAM probe unavailable\n");
    return 0;
  }
  const int kv_type = kv_type_from_cfg();
  const placement::Decision d = compute_decision(m, p.ctx, kv_type, vram, host);
  placement::print(d, stdout);
  return 0;
}

void usage() {
  std::fprintf(stderr,
               "engine - local LLM inference engine\n"
               "\n"
               "usage: engine <command> [options]\n"
               "\n"
               "commands:\n"
               "  info   <model>                 parsed IR summary + validation\n"
               "  ir     <model> [--out FILE]    export the full IR as JSON\n"
               "  plan   <model> [--ctx N] [--vram-gib F] [--kv-bpe N]\n"
               "                                 budget / expert placement plan + placement decision\n"
               "  run    <model> --ids 1,2,3 [--n 8]\n"
               "                                 greedy generate N tokens (interactive when tty)\n"
               "  serve  <model> [--host 127.0.0.1] [--port 7030] [--ctx 4096]\n"
               "                                 HTTP server: OpenAI /v1/chat/completions,\n"
               "                                 /v1/completions, llama.cpp /completion,\n"
               "                                 /health, /v1/models, /props, browser UI at /\n"
               "  models [--json|--plain|--tsv]   list the GGUF models found by the model scan\n"
               "                                 (largest first; the index works with --pick)\n"
               "  models --refresh [--progress] [--scan-depth N] [--min-size-mb N]\n"
               "                                 full-disk scan of all fixed/removable drives;\n"
               "                                 writes engine-models-cache.json (default: read cache)\n"
               "  models --no-scan --model-dir DIR only the explicitly given directories\n"
               "\n"
               "global options (any command, see also --help):\n"
               "  --config <file>   key = value config file (cli > config > env > default)\n"
               "  --ctx N           context length (KV cache size); 0 = derive from free VRAM\n"
               "  --threads N       CPU worker threads (0 = auto)\n"
               "  --gpu-layers N    MoE layers kept in VRAM (-1 = auto, 'auto' accepted)\n"
               "  --vram-margin MB  free VRAM required after load (default 1200)\n"
               "  --no-graph        disable CUDA graphs\n"
               "  --print-config    print the effective configuration and exit\n"
               "  --kernel-calibrate  micro-benchmark the GEMV kernels and cache the choice\n"
               "\n"
               "placement policy (see FEATURE-MATRIX.md section 2):\n"
               "  --placement auto|gpu|split|ssd   weight placement policy (default auto:\n"
               "                                   measure free VRAM/RAM and decide)\n"
               "  --host-margin MB  host RAM kept free by the decider (0 = auto: max(1GiB, RAM/16))\n"
               "  --ssd-stream      allow SSD expert streaming when VRAM+RAM are exceeded\n"
               "                    (NOT IMPLEMENTED in this build; off by default)\n"
               "  --ssd-cache DIR   on-disk expert cache directory for SSD streaming (placeholder)\n"
               "  --allow-oversubscribe  attempt the load even if the decider says it does not fit\n"
               "\n"
               "interactive startup (only when stdin is a terminal; pipes/CI are never prompted):\n"
               "  --no-interactive  never prompt (same as ENGINE_INTERACTIVE=0)\n"
               "  --yes             accept the computed defaults without asking (scripts/CI)\n"
               "  --model-dir DIR   extra model scan directory (repeatable / ';'-separated);\n"
               "                    scanned before the built-in dirs, so it can reorder discovery\n"
               "\n"
               "run options: --ids <a,b,c> --n <count>\n"
               "serve options: --host <addr> --port <n> --ctx <n> --serve-ctx <n>\n"
               "               --model auto       pick the largest model found by the scan\n"
               "               --pick N           use model #N from `engine models`\n"
               "               --open-browser     open the chat UI once the server is ready\n"
               "               --default-max-tokens / --default-temperature / --default-p\n"
               "               --default-top-k / --default-repeat-penalty\n");
}

// ---------------------------------------------------------------------------
// Interactive startup + placement decision plumbing (used by `run` and `serve`)
//
// Everything here is skipped unless stdin is a terminal (or the debug hook
// ENGINE_FORCE_INTERACTIVE is set, which exists so the flow can be tested from a
// pipe).  Non-interactive behaviour is unchanged: no prompts, same defaults and
// the same generated tokens.
// ---------------------------------------------------------------------------

bool stdin_is_tty() { return _isatty(_fileno(stdin)) != 0; }

bool interactive_available() {
  engine::config::Config& cfg = engine::config::Config::get();
  // Explicit switches always win, including over the debug force hook.
  if (!cfg.get_bool("interactive")) return false;
  if (cfg.get_bool("assume_yes")) return false;
  if (std::getenv("ENGINE_FORCE_INTERACTIVE") != nullptr) return true;  // debug/test hook
  if (!stdin_is_tty()) return false;  // pipe / service manager / CI: never prompt
  return true;
}

bool read_prompt_line(const char* prompt, std::string* out) {
  if (prompt && *prompt) {
    std::fprintf(stderr, "%s", prompt);
    std::fflush(stderr);
  }
  char buf[1024];
  if (std::fgets(buf, sizeof(buf), stdin) == nullptr) {
    *out = "";
    return false;
  }
  std::string s(buf);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  const size_t b = s.find_first_not_of(" \t");
  const size_t e = s.find_last_not_of(" \t");
  *out = (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
  return true;
}

bool parse_ll(const std::string& s, long long* out) {
  if (s.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0') return false;
  *out = v;
  return true;
}

std::string basename_of(const std::string& p) {
  const size_t i = p.find_last_of("/\\");
  return i == std::string::npos ? p : p.substr(i + 1);
}

struct StartupChoices {
  bool active = false;
  int ctx = -1;         // -1 = not chosen, 0 = auto
  int kv_type = -1;     // -1 = not chosen
  int gpu_layers = -2;  // -2 = not chosen, -1 = auto, >= 0 explicit
  int n_gen = -1;
};

// Number of the resident-weight bytes as a 2-decimal GiB string.
std::string gib(uint64_t bytes) {
  char b[64];
  std::snprintf(b, sizeof(b), "%.2f GiB", static_cast<double>(bytes) / (1ull << 30));
  return b;
}

std::string mib(uint64_t bytes) {
  char b[64];
  std::snprintf(b, sizeof(b), "%.0f MiB", static_cast<double>(bytes) / 1048576.0);
  return b;
}

int kv_type_from_name(const std::string& raw) {
  std::string s;
  for (char c : raw) s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  if (s == "q8_0" || s == "q8" || s == "8" || s == "1") return 1;
  if (s == "q4_0" || s == "q4" || s == "4" || s == "2") return 2;
  return 0;  // f32 / auto / anything else
}

std::string kv_name(int t) { return t == 1 ? "q8_0" : (t == 2 ? "q4_0" : "f32"); }

int kv_type_from_cfg() { return kv_type_from_name(engine::config::Config::get().get_string("kv_type")); }

uint64_t placement_reserve_bytes() {
  engine::config::Config& cfg = engine::config::Config::get();
  return (static_cast<uint64_t>(cfg.get_int("vram_margin_mb")) +
          static_cast<uint64_t>(cfg.get_int("workspace_mb")))
         << 20;
}

uint64_t host_reserve_bytes() {
  engine::config::Config& cfg = engine::config::Config::get();
  const uint64_t v = static_cast<uint64_t>(cfg.get_int("host_margin_mb")) << 20;
  if (v) return v;
  return placement::default_host_reserve(placement::host_memory().total_bytes);
}

placement::Decision compute_decision(const ir::Model& m, int ctx, int kv_type,
                                     const placement::MemoryInfo& vram,
                                     const placement::MemoryInfo& host) {
  engine::config::Config& cfg = engine::config::Config::get();
  placement::Input in;
  in.model = &m;
  in.ctx = ctx > 0 ? ctx : 1;
  in.kv_type = kv_type;
  in.vram = vram;
  in.host = host;
  in.vram_reserve = placement_reserve_bytes();
  in.host_reserve = host_reserve_bytes();
  in.workspace = static_cast<uint64_t>(cfg.get_int("workspace_mb")) << 20;
  in.ssd_requested = cfg.get_bool("ssd_stream") || cfg.get_string("placement_policy") == "ssd";
  return placement::decide(in);
}

// Ledger + policy gate.  Returns 0 to continue, or the process exit code when an
// explicitly requested / impossible placement must stop the launch.
int announce_placement(const ir::Model& m, int ctx, int kv_type) {
  engine::config::Config& cfg = engine::config::Config::get();
  const placement::MemoryInfo vram = placement::device_memory();
  const placement::MemoryInfo host = placement::host_memory();
  if (!vram.valid) {
    std::fprintf(stderr, "[placement] skipped: CUDA/VRAM probe unavailable (no GPU driver)\n");
    return 0;
  }
  const placement::Decision d = compute_decision(m, ctx, kv_type, vram, host);
  placement::print(d, stderr);
  const std::string policy = cfg.get_string("placement_policy");

  // SSD streaming is a candidate only when nothing else fits, and even then it is
  // opt-in: this build ships the interface placeholder, not the streamer.
  if (cfg.get_bool("ssd_stream") || policy == "ssd") {
    placement::SsdStreamConfig sc;
    sc.cache_dir = cfg.get_string("ssd_cache");
    placement::SsdExpertStreamer streamer(sc);
    std::string err;
    streamer.open(m.path, &err);
    std::fprintf(stderr, "[placement] ERROR: %s\n", err.c_str());
    std::fprintf(stderr,
                 "[placement] hint : SSD streaming stays off by default because the measured lanes "
                 "(PCIe %.1f GB/s H2D, host RAM %.1f GB/s) make it a net loss unless the model "
                 "really cannot fit; lower --ctx / use --kv-type q8_0 or a smaller quantisation, "
                 "or drop --ssd-stream.\n",
                 planner::BW_PCIE_H2D_GBPS, planner::BW_RAM_COPY_GBPS);
    return 3;
  }
  if (policy == "split" && d.n_moe_layers == 0) {
    std::fprintf(stderr,
                 "[placement] ERROR: --placement split requested but this is a dense model; the "
                 "engine has no host dense-weight path (only MoE experts can be offloaded)\n");
    return 3;
  }
  if (policy == "gpu" && !d.fits_vram) {
    std::fprintf(stderr,
                 "[placement] WARN : --placement gpu requested but the weights need %s while only "
                 "%s of VRAM is usable; the loader will hit the driver limit\n",
                 mib(d.vram_need + d.expert_bytes).c_str(), mib(d.vram_usable).c_str());
    return 0;  // explicit override: the user asked for it, do not turn it into a refusal
  }
  if (policy == "split" && d.kind == placement::Kind::Gpu) {
    std::fprintf(stderr,
                 "[placement] WARN : --placement split requested but everything fits VRAM; keeping "
                 "full GPU residency (pass --gpu-layers to force an explicit split)\n");
  }
  if (d.kind == placement::Kind::SsdStream) {
    if (cfg.get_bool("allow_oversubscribe")) {
      std::fprintf(stderr,
                   "[placement] WARN : continuing because --allow-oversubscribe is set; the driver "
                   "will page, expect single-digit tok/s\n");
      return 0;
    }
    std::fprintf(stderr,
                 "[placement] ERROR: model does not fit VRAM + host RAM (%s needed, %s usable) and "
                 "SSD streaming is not implemented in this build\n",
                 mib(d.dense_offload_unsupported ? d.vram_need : d.vram_need + d.expert_bytes).c_str(),
                 mib(d.vram_usable + d.host_usable).c_str());
    std::fprintf(stderr,
                 "[placement] hint : %s\n",
                 d.dense_offload_unsupported
                     ? "dense weights cannot be split: free VRAM, use a smaller quantisation, or "
                       "pass --allow-oversubscribe to attempt the load anyway"
                     : "lower --ctx, use --kv-type q8_0/q4_0, free host RAM, use a smaller "
                       "quantisation, or pass --allow-oversubscribe to attempt the load anyway");
    return 3;
  }
  return 0;
}

// Extra model scan directories: --model-dir / config model_dir / ENGINE_MODEL_DIR.
// The config value may hold several directories separated by ';'; they are
// searched before the built-in dirs (exe-dir/models, ../models, %USERPROFILE%/models,
// %USERPROFILE%/Downloads, %USERPROFILE%/.cache).
std::vector<std::string> model_extra_dirs() {
  std::vector<std::string> out;
  const std::string md = engine::config::Config::get().get_string("model_dir");
  size_t begin = 0;
  while (begin <= md.size()) {
    size_t end = md.find(';', begin);
    if (end == std::string::npos) end = md.size();
    const std::string part = md.substr(begin, end - begin);
    const size_t b = part.find_first_not_of(" \t");
    const size_t e = part.find_last_not_of(" \t");
    if (b != std::string::npos) out.push_back(part.substr(b, e - b + 1));
    if (end == md.size()) break;
    begin = end + 1;
  }
  return out;
}

// Interactive picker shown by `serve` / `run` when no model was given on a tty.
// The scan itself is C++ (include/engine/model_scan.h), so `engine models`, the
// picker and the double-click launcher all list exactly the same entries.
std::string prompt_model_path() {
  const engine::model_scan::Report report =
      engine::model_scan::scan_cached_or_quick(model_extra_dirs());
  std::fprintf(stderr, "\n[interactive] no model given; model scan found %zu GGUF model(s)\n",
               report.entries.size());
  if (report.entries.empty()) {
    std::fputs(engine::model_scan::no_models_hint(report).c_str(), stderr);
    std::string line;
    read_prompt_line("[interactive] model path > ", &line);
    return line;
  }
  std::fputs(engine::model_scan::format_table(report).c_str(), stderr);
  std::string line;
  read_prompt_line("[interactive] model # or path [1] > ", &line);
  if (line.empty()) return report.entries.front().path;
  long long idx = 0;
  if (parse_ll(line, &idx) && idx >= 1 &&
      idx <= static_cast<long long>(report.entries.size()))
    return report.entries[static_cast<size_t>(idx - 1)].path;
  return line;
}

// `engine models`: list what the scan finds (exit 0 = at least one model,
// exit 1 = none).  The double-click launcher uses the exit code to show a clear
// hint instead of trying to start without a model.
int cmd_models(bool json, bool plain, bool tsv, const engine::model_scan::ScanOptions& opts) {
  engine::model_scan::ScanInfo info;
  const engine::model_scan::Report report = engine::model_scan::scan_with_options(opts, &info);
  if (json) {
    std::printf("%s\n", engine::model_scan::format_json(report).c_str());
  } else if (plain) {
    std::fputs(engine::model_scan::format_plain(report).c_str(), stdout);
  } else if (tsv) {
    std::fputs(engine::model_scan::format_tsv(report).c_str(), stdout);
  } else {
    std::fputs(engine::model_scan::format_table(report).c_str(), stdout);
  }
  std::fflush(stdout);
  if (report.entries.empty()) {
    std::fputs(engine::model_scan::no_models_hint(report).c_str(), stderr);
    return 1;
  }
  return 0;
}

struct StartupProbe {
  bool ok = false;
  std::string error;
  ir::Model model;
};

// Metadata-only probe (GGUF header / HF config); weights are NOT loaded.  The
// interactive prompt and the placement ledger must run before the runner
// allocates the KV cache, which is exactly why this exists.
StartupProbe probe_model(const std::string& path) {
  StartupProbe p;
  try {
    Loaded l = load(path);
    p.model = std::move(l.model);
    p.ok = !p.model.arch.empty();
  } catch (const std::exception& e) {
    p.error = e.what();
  }
  return p;
}

void interactive_startup(const std::string& path, const ir::Model& m, bool for_serve,
                         bool ctx_given, bool n_given, int* n_gen, StartupChoices* ch) {
  auto& cfg = engine::config::Config::get();
  ch->active = true;
  const placement::MemoryInfo vram = placement::device_memory();
  const placement::MemoryInfo host = placement::host_memory();
  const uint64_t reserve = placement_reserve_bytes();
  const int row_cap = static_cast<int>(cfg.get_int("prefill_rows"));
  const int n_moe = static_cast<int>(m.expert_bytes_per_layer().size());
  const bool mla = m.attn.variant == "mla";

  std::fprintf(stderr, "\n[interactive] model : %s\n", basename_of(path).c_str());
  std::fprintf(stderr,
               "[interactive] shape : %s, %d layers%s, ctx_train=%lld, weights %.2f GiB\n",
               m.arch.c_str(), m.n_layer, n_moe ? " (MoE)" : "",
               static_cast<long long>(m.ctx_train), static_cast<double>(m.fixed_bytes()) / (1ull << 30));
  if (vram.valid)
    std::fprintf(stderr,
                 "[interactive] vram  : free %s / total %s, reserve %s (vram_margin+workspace)\n",
                 mib(vram.free_bytes).c_str(), mib(vram.total_bytes).c_str(), mib(reserve).c_str());
  else
    std::fprintf(stderr,
                 "[interactive] vram  : probe unavailable (no GPU driver); recommendations use "
                 "the configured defaults\n");
  if (host.valid)
    std::fprintf(stderr, "[interactive] host  : free %s / total %s, reserve %s\n",
                 mib(host.free_bytes).c_str(), mib(host.total_bytes).c_str(),
                 mib(host_reserve_bytes()).c_str());

  // ---- 1) KV cache precision ------------------------------------------------
  int kv = cfg.was_set("kv_type") ? kv_type_from_cfg() : 0;
  if (mla) {
    std::fprintf(stderr,
                 "[interactive] kv    : MLA model keeps the f32 MLA cache (no quantised MLA "
                 "attention kernel); kv_type choice skipped\n");
    ch->kv_type = 0;
  } else if (cfg.was_set("kv_type")) {
    std::fprintf(stderr, "[interactive] kv    : %s (from command line/config)\n", kv_name(kv).c_str());
  } else {
    placement::CtxFit fits[3];
    for (int t = 0; t < 3; ++t)
      fits[t] = placement::estimate_ctx(m, t, vram.valid ? vram.free_bytes : 0, reserve, row_cap);
    std::fprintf(stderr,
                 "[interactive] kv    : KV/token f32 %.1f KiB | q8_0 %.1f KiB | q4_0 %.1f KiB\n",
                 fits[0].kv_per_token / 1024.0, fits[1].kv_per_token / 1024.0,
                 fits[2].kv_per_token / 1024.0);
    if (vram.valid) {
      std::fprintf(stderr,
                   "[interactive] ctx fit: f32 <= %d | q8_0 <= %d | q4_0 <= %d tokens (free VRAM - "
                   "weights - reserve)\n",
                   fits[0].max_ctx, fits[1].max_ctx, fits[2].max_ctx);
    }
    int def = 0;
    if (vram.valid) {
      if (fits[0].max_ctx < 4096 && fits[1].max_ctx >= 4096) def = 1;
      if (fits[def].max_ctx < 4096 && fits[2].max_ctx >= 4096) def = 2;
    }
    std::string line;
    read_prompt_line(("\n[interactive] KV type [f32/q8_0/q4_0] (Enter = " + kv_name(def) + ") > ").c_str(),
                     &line);
    if (!line.empty()) {
      long long v = 0;
      if (parse_ll(line, &v) && v >= 0 && v <= 2) {
        kv = static_cast<int>(v);
      } else {
        const std::string lower = [&] {
          std::string s;
          for (char c : line) s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
          return s;
        }();
        if (lower == "f32" || lower == "fp32") kv = 0;
        else if (lower == "q8" || lower == "q8_0" || lower == "8") kv = 1;
        else if (lower == "q4" || lower == "q4_0" || lower == "4") kv = 2;
        else {
          std::fprintf(stderr, "[interactive] note: unknown kv type '%s', using %s\n", line.c_str(),
                       kv_name(def).c_str());
          kv = def;
        }
      }
    } else {
      kv = def;
    }
    ch->kv_type = kv;
  }

  // ---- 2) CPU expert offload (MoE only) -------------------------------------
  if (n_moe > 0 && !cfg.was_set("gpu_layers")) {
    int use_ctx = ch->ctx >= 0 ? ch->ctx : (cfg.was_set("ctx") ? static_cast<int>(cfg.get_int("ctx")) : 4096);
    if (use_ctx <= 0)
      use_ctx = std::max(1, placement::estimate_ctx(m, kv, vram.valid ? vram.free_bytes : 0,
                                                    reserve, row_cap)
                                .max_ctx);
    const placement::Decision d = compute_decision(m, use_ctx, kv, vram, host);
    const bool rec = d.kind == placement::Kind::Split;
    std::fprintf(stderr,
                 "[interactive] experts: %d MoE layers, %s expert weights; %s\n", n_moe,
                 gib(m.expert_bytes()).c_str(),
                 rec ? "do not all fit VRAM -> CPU expert offload recommended"
                     : "fit VRAM -> keep all experts on the GPU");
    std::string line;
    read_prompt_line(rec ? "[interactive] CPU expert offload [Y/n] (Enter = yes) > "
                         : "[interactive] CPU expert offload [y/N] (Enter = no) > ",
                     &line);
    bool yes = rec;
    if (!line.empty()) {
      std::string lower;
      for (char c : line) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
      if (lower == "y" || lower == "yes" || lower == "1" || lower == "true" || lower == "on")
        yes = true;
      else if (lower == "n" || lower == "no" || lower == "0" || lower == "false" || lower == "off")
        yes = false;
      else
        std::fprintf(stderr, "[interactive] note: unknown answer '%s', using default (%s)\n",
                     line.c_str(), rec ? "yes" : "no");
    }
    if (yes) {
      ch->gpu_layers = -1;  // automatic layer-granular placement
    } else {
      ch->gpu_layers = n_moe;  // every expert must be GPU-resident
      if (rec)
        std::fprintf(stderr,
                     "[interactive] WARN : all %d MoE layers on the GPU needs %s of experts while "
                     "only %s of VRAM is usable after the non-expert weights; expect an OOM\n",
                     n_moe, gib(m.expert_bytes()).c_str(), mib(d.vram_left).c_str());
    }
  }

  // ---- 3) Context length ----------------------------------------------------
  if (!ctx_given) {
    const int use_kv = ch->kv_type >= 0 ? ch->kv_type : kv;
    const placement::CtxFit fit =
        placement::estimate_ctx(m, use_kv, vram.valid ? vram.free_bytes : 0, reserve, row_cap);
    if (vram.valid) {
      std::fprintf(stderr,
                   "[interactive] context: recommended <= %d tokens (kv=%s, KV %s at that length, "
                   "weights %s, free VRAM %s)\n",
                   fit.max_ctx, kv_name(use_kv).c_str(), mib(fit.kv_at_max).c_str(),
                   gib(m.fixed_bytes()).c_str(), mib(vram.free_bytes).c_str());
    }
    std::string line;
    read_prompt_line(("\n[interactive] ctx tokens [Enter = auto (" + std::to_string(fit.max_ctx) +
                      "), 0 = auto] > ")
                         .c_str(),
                     &line);
    if (line.empty()) {
      ch->ctx = 0;
    } else {
      long long v = 0;
      if (!parse_ll(line, &v)) {
        std::fprintf(stderr, "[interactive] note: '%s' is not a number, using auto\n", line.c_str());
        ch->ctx = 0;
      } else {
        if (v < 0) v = 0;
        if (v > (1 << 20)) v = 1 << 20;
        ch->ctx = static_cast<int>(v);
        if (v == 0) {
          // auto
        } else if (v > fit.max_ctx) {
          std::fprintf(stderr,
                       "[interactive] note: %lld > recommended %d; the loader clamps to what fits "
                       "(ctx_auto) and prints its own [ctx] ledger\n",
                       v, fit.max_ctx);
        }
        if (m.ctx_train > 0 && v > m.ctx_train)
          std::fprintf(stderr, "[interactive] note: %lld > ctx_train %lld (quality will degrade)\n",
                       v, static_cast<long long>(m.ctx_train));
      }
    }
  }

  // ---- 4) generation length (`run` only) ------------------------------------
  if (!for_serve && !n_given) {
    std::string line;
    read_prompt_line(("\n[interactive] tokens to generate [" + std::to_string(*n_gen) + "] > ").c_str(),
                     &line);
    if (!line.empty()) {
      long long v = 0;
      if (parse_ll(line, &v) && v >= 1 && v <= 8192) {
        *n_gen = static_cast<int>(v);
        ch->n_gen = *n_gen;
      } else {
        std::fprintf(stderr, "[interactive] note: '%s' invalid (1..8192), using %d\n", line.c_str(),
                     *n_gen);
      }
    }
  }
  std::fprintf(stderr, "[interactive] choices: ctx=%s kv=%s gpu_layers=%s\n",
               ch->ctx < 0 ? "unchanged" : (ch->ctx == 0 ? "auto" : std::to_string(ch->ctx).c_str()),
               ch->kv_type < 0 ? "unchanged" : kv_name(ch->kv_type).c_str(),
               ch->gpu_layers == -2 ? "unchanged"
                                    : (ch->gpu_layers < 0 ? "auto" : std::to_string(ch->gpu_layers).c_str()));
}

bool apply_startup_choices(const StartupChoices& ch, std::string* err) {
  auto& cfg = engine::config::Config::get();
  if (ch.ctx >= 0 && !cfg.set("ctx", std::to_string(ch.ctx), engine::config::Source::Cli, err))
    return false;
  if (ch.kv_type >= 0 &&
      !cfg.set("kv_type", kv_name(ch.kv_type), engine::config::Source::Cli, err))
    return false;
  if (ch.gpu_layers != -2 &&
      !cfg.set("gpu_layers", std::to_string(ch.gpu_layers), engine::config::Source::Cli, err))
    return false;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 1;
  }
  std::string cmd = argv[1];
  const bool subcommand = !cmd.empty() && cmd[0] != '-';
  const int start = subcommand ? 2 : 1;

  std::string out, json_out, ids_arg;
  bool with_tensors = true, force = false;
  bool n_given = false, vram_given = false;
  bool models_json = false, plain_out = false, tsv_out = false, open_browser = false;
  bool models_refresh = false, models_no_scan = false, models_progress = false;
  int scan_depth = 7;
  long long scan_min_size_mb = 100;
  bool info_json = false, schema_json = false;
  // `--print-config --json` emits the machine-readable config surface (the GUI
  // launcher builds its advanced panel from it).  Pre-scan so `--json` can be
  // parsed as a flag in that mode instead of as the `ir --json FILE` value.
  bool wants_print_config = false;
  for (int i = start; i < argc; ++i)
    if (std::strcmp(argv[i], "--print-config") == 0) wants_print_config = true;
  int pick = -1;
  int n_gen = 4, kv_bpe = 2;
  double vram = 12.0;

  // Subcommand-specific options are consumed here so the common config parser
  // only sees engine-wide options (they all live in docs/CONFIG.md).
  std::vector<std::string> filtered;
  filtered.push_back(argv[0]);
  try {
    for (int i = start; i < argc; ++i) {
      std::string a = argv[i];
      auto need = [&](const char* what) -> std::string {
        if (i + 1 >= argc)
          throw std::runtime_error(std::string("missing value for ") + what);
        return argv[++i];
      };
      if (a == "--out") out = need("--out");
      else if (a == "--json") {
        // `ir --json FILE` writes a file; `models --json` / `info --json` are
        // flags; with --print-config it selects the schema JSON.
        if (cmd == "models") models_json = true;
        else if (cmd == "info") info_json = true;
        else if (wants_print_config) schema_json = true;
        else json_out = need("--json");
      }
      else if (a == "--plain") plain_out = true;
      else if (a == "--tsv") tsv_out = true;
      else if (a == "--pick") pick = std::stoi(need("--pick"));
      else if (a == "--open-browser") open_browser = true;
      else if (a == "--refresh") models_refresh = true;
      else if (a == "--no-scan") models_no_scan = true;
      else if (a == "--progress") models_progress = true;
      else if (a == "--scan-depth") scan_depth = std::stoi(need("--scan-depth"));
      else if (a == "--min-size-mb") scan_min_size_mb = std::stoll(need("--min-size-mb"));
      else if (a == "--no-tensors") with_tensors = false;
      else if (a == "--force") force = true;
      else if (a == "--kv-bpe") kv_bpe = std::stoi(need("--kv-bpe"));
      else if (a == "--vram-gib") {
        vram = std::stod(need("--vram-gib"));
        vram_given = true;
      }
      else if (a == "--ids" || a == "--tokens") ids_arg = need("--ids");
      else if (a == "--n" || a == "--max-new-tokens") {
        n_gen = std::stoi(need("--n"));
        n_given = true;
      }
      else filtered.push_back(a);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }

  engine::config::Config& cfg = engine::config::Config::get();
  std::vector<char*> cargv;
  cargv.reserve(filtered.size());
  for (auto& s : filtered) cargv.push_back(const_cast<char*>(s.c_str()));
  std::vector<std::string> positional, unparsed;
  std::string cfg_err;
  if (!cfg.parse_command_line(static_cast<int>(cargv.size()), cargv.data(), 1, &positional,
                              &unparsed, &cfg_err)) {
    std::fprintf(stderr, "config error: %s\n", cfg_err.c_str());
    return 2;
  }
  for (const auto& u : unparsed) {
    if (u == "--help" || u == "-h") {
      cfg.print_options(stdout);
      usage();
      return 0;
    }
    std::fprintf(stderr, "unknown option %s (try --help)\n", u.c_str());
    return 2;
  }
  std::string cache_err;
  cfg.load_kernel_cache_if_exists(&cache_err);
  if (cfg.get_bool("print_config")) {
    if (schema_json) cfg.print_schema_json(stdout);
    else cfg.print_effective(stdout);
    return 0;
  }
  // Real-desktop scheduling: apply win_priority before any allocation / thread pool, so
  // `run`, `serve` and the placement probes all run at the configured class.
  engine::config::apply_win_priority();

  try {
    // `models` needs no model file: list the scan results and exit.
    if (cmd == "models") {
      engine::model_scan::ScanOptions sopts;
      sopts.extra_dirs = model_extra_dirs();
      sopts.refresh = models_refresh;
      sopts.no_scan = models_no_scan;
      sopts.progress = models_progress;
      if (scan_depth < 1) scan_depth = 1;
      if (scan_depth > 64) scan_depth = 64;
      sopts.max_depth = scan_depth;
      if (scan_min_size_mb < 0) scan_min_size_mb = 0;
      sopts.min_size_bytes = static_cast<uint64_t>(scan_min_size_mb) << 20;
      return cmd_models(models_json, plain_out, tsv_out, sopts);
    }

    std::string path = cfg.get_string("model");
    size_t pi = 0;
    if (path.empty() && pi < positional.size()) path = positional[pi++];

    const bool is_serve = cmd == "serve";
    const bool is_run = cmd == "run";

    // `serve` can resolve the model itself, without a path on the command line:
    //   --model auto  -> largest model found by the scan
    //   --pick N      -> the N-th entry of `engine models` (used by 启动.bat)
    if (is_serve) {
      const engine::model_scan::Report report =
          engine::model_scan::scan_cached_or_quick(model_extra_dirs());
      if (path == "auto") {
        if (report.entries.empty()) {
          std::fputs(engine::model_scan::no_models_hint(report).c_str(), stderr);
          return 1;
        }
        path = report.entries.front().path;
        std::printf("[models] auto -> %s\n", path.c_str());
      } else if (path.empty() && pick >= 0) {
        if (report.entries.empty()) {
          std::fputs(engine::model_scan::no_models_hint(report).c_str(), stderr);
          return 1;
        }
        if (pick < 1 || pick > static_cast<int>(report.entries.size())) {
          std::fprintf(stderr, "error: --pick %d out of range (1..%zu)\n", pick,
                       report.entries.size());
          std::fputs(engine::model_scan::format_table(report).c_str(), stdout);
          return 2;
        }
        path = report.entries[static_cast<size_t>(pick - 1)].path;
        std::printf("[models] pick %d -> %s\n", pick, path.c_str());
      } else if (!path.empty() && pick >= 0) {
        std::fprintf(stderr, "[models] note: explicit model path given; ignoring --pick\n");
      }
    }

    // ---- interactive startup + metadata probe (run/serve, before any allocation) ----
    // The prompt must run before the model loads because ctx / kv_type decide the KV
    // allocation.  The probe reads only the GGUF header / HF config (no weights).
    // Non-tty stdin (pipe / service manager / CI) never reaches this block's prompts,
    // so scripted startup is unchanged.
    StartupProbe probe;
    StartupChoices choices;
    if (is_serve || is_run) {
      if (path.empty() && interactive_available()) path = prompt_model_path();
      if (path.empty()) {
        // No model and no prompt (pipe / CI / --yes): still show what the scan
        // saw and how to fix it, instead of a bare usage dump.
        const engine::model_scan::Report report =
            engine::model_scan::scan_cached_or_quick(model_extra_dirs());
        if (report.entries.empty()) {
          std::fputs(engine::model_scan::no_models_hint(report).c_str(), stderr);
        } else {
          std::fputs(engine::model_scan::format_table(report).c_str(), stderr);
          std::fprintf(stderr,
                       "\n[models] no model selected; pass a path or use "
                       "`engine.exe models` + `serve --pick N`\n");
        }
        usage();
        return 1;
      }
      probe = probe_model(path);
      if (interactive_available()) {
        if (probe.ok) {
          const bool ctx_given =
              is_serve ? (cfg.was_set("ctx") || static_cast<int>(cfg.get_int("serve_ctx")) > 0)
                       : cfg.was_set("ctx");
          interactive_startup(path, probe.model, is_serve, ctx_given, n_given, &n_gen, &choices);
          std::string apply_err;
          if (!apply_startup_choices(choices, &apply_err)) {
            std::fprintf(stderr, "error: %s\n", apply_err.c_str());
            return 2;
          }
        } else {
          std::fprintf(stderr, "[interactive] skipped: cannot parse model metadata (%s)\n",
                       probe.error.c_str());
        }
      }
    }
    if (path.empty()) {
      usage();
      return 1;
    }
    if (cmd == "info") return cmd_info(path, info_json);
    if (cmd == "ir") return cmd_ir(path, out, with_tensors, force);
    if (cmd == "plan") {
      const int ctx = cfg.was_set("ctx") ? static_cast<int>(cfg.get_int("ctx")) : 8192;
      return cmd_plan(path, ctx, vram, vram_given, kv_bpe, json_out);
    }
    if (is_serve) {
      engine::serve::Options o;
      o.model = path;
      o.host = cfg.get_string("serve_host");
      o.port = static_cast<int>(cfg.get_int("serve_port"));
      o.open_browser = open_browser;
      const int serve_ctx = static_cast<int>(cfg.get_int("serve_ctx"));
      if (serve_ctx > 0) o.ctx = serve_ctx;
      else if (cfg.was_set("ctx")) o.ctx = static_cast<int>(cfg.get_int("ctx"));
      else o.ctx = 4096;
      // Placement ledger + policy gate: full accounting before the runner allocates
      // the KV cache.  `auto` (the default) is advisory: it prints where the weights
      // will live and refuses only when the model cannot fit at all or the user
      // explicitly requested the unimplemented SSD path.
      if (probe.ok) {
        const int kv_type = kv_type_from_cfg();
        int ledger_ctx = o.ctx;
        if (ledger_ctx <= 0) {
          const placement::MemoryInfo vram = placement::device_memory();
          ledger_ctx = placement::estimate_ctx(
                           probe.model, kv_type, vram.valid ? vram.free_bytes : 0,
                           placement_reserve_bytes(),
                           static_cast<int>(cfg.get_int("prefill_rows")))
                           .max_ctx;
          if (ledger_ctx <= 0) ledger_ctx = 4096;
          std::fprintf(stderr, "[placement] ctx=auto -> ledger uses %d tokens\n", ledger_ctx);
        }
        const int rc = announce_placement(probe.model, ledger_ctx, kv_type);
        if (rc != 0) return rc;
      }
      o.default_max_tokens = static_cast<int>(cfg.get_int("default_max_tokens"));
      o.default_temperature = cfg.get_double("default_temperature");
      o.default_top_p = cfg.get_double("default_top_p");
      o.default_top_k = static_cast<int>(cfg.get_int("default_top_k"));
      o.default_repetition_penalty = cfg.get_double("default_repeat_penalty");
      return engine::serve::run(o);
    }
    if (is_run) {
      std::string ids_text = ids_arg;
      if (ids_text.empty() && pi < positional.size()) ids_text = positional[pi++];
      if (ids_text.empty()) {
        std::fprintf(stderr, "run: missing --ids <a,b,c>\n");
        return 2;
      }
      std::vector<int> ids;
      {
        size_t p = 0;
        while (p < ids_text.size()) {
          size_t q = ids_text.find(',', p);
          if (q == std::string::npos) q = ids_text.size();
          ids.push_back(std::atoi(ids_text.substr(p, q - p).c_str()));
          p = q + 1;
        }
      }
      if (ids.empty()) throw std::runtime_error("no prompt ids");
      int max_seq = static_cast<int>(ids.size()) + n_gen + 16;
      if (cfg.was_set("ctx")) max_seq = static_cast<int>(cfg.get_int("ctx"));
      if (max_seq < static_cast<int>(ids.size())) max_seq = static_cast<int>(ids.size());
      if (probe.ok) {
        const int kv_type = kv_type_from_cfg();
        int ledger_ctx = max_seq;
        if (cfg.was_set("ctx")) ledger_ctx = static_cast<int>(cfg.get_int("ctx"));
        if (ledger_ctx <= 0) {
          const placement::MemoryInfo vram = placement::device_memory();
          ledger_ctx = placement::estimate_ctx(
                           probe.model, kv_type, vram.valid ? vram.free_bytes : 0,
                           placement_reserve_bytes(),
                           static_cast<int>(cfg.get_int("prefill_rows")))
                           .max_ctx;
          if (ledger_ctx <= 0) ledger_ctx = 4096;
          std::fprintf(stderr, "[placement] ctx=auto -> ledger uses %d tokens\n", ledger_ctx);
        }
        const int rc = announce_placement(probe.model, ledger_ctx, kv_type);
        if (rc != 0) return rc;
      }
      runner::DenseRunner r(path, max_seq, runner::detect_container(path));
      if (cfg.get_bool("kernel_calibrate")) {
        const int n = r.calibrate_kernels(true);
        return n > 0 ? 0 : 1;
      }
      const auto t0 = std::chrono::steady_clock::now();
      auto logits = r.forward(ids, 0, true);
      const double prefill_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
              .count();
      const int vocab = r.vocab();
      std::vector<float> last(logits.end() - vocab, logits.end());
      auto argmax = [&](const std::vector<float>& row) {
        int best = 0;
        for (int i = 1; i < vocab; ++i)
          if (row[i] > row[best]) best = i;
        return best;
      };
      std::printf("prompt :");
      for (int t : ids) std::printf(" %d", t);
      std::printf("\ngen    :");
      std::vector<int> gen;
      int pos = static_cast<int>(ids.size());
      const auto td0 = std::chrono::steady_clock::now();
      for (int k = 0; k < n_gen; ++k) {
        const int next = argmax(last);
        gen.push_back(next);
        std::printf(" %d", next);
        if (k + 1 == n_gen) break;
        auto o = r.forward({next}, pos++, true);
        last.assign(o.end() - vocab, o.end());
      }
      const double dec_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0)
              .count();
      std::printf("\n");
      if (n_gen > 1)
        std::fprintf(stderr, "[time] prefill %.1f ms, decode %.2f ms/token (%.0f tok/s)\n",
                     prefill_ms, dec_ms / (n_gen - 1), (n_gen - 1) * 1000.0 / dec_ms);
      return 0;
    }
    usage();
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 3;
  }
}
