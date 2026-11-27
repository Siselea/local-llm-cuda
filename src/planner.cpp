#include "engine/planner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "engine/cuda.h"
#include "engine/placement.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace planner {

PlacementPlan compute(const ir::Model& m, int ctx, int kv_type, uint64_t vram_total,
                      uint64_t vram_reserve, uint64_t workspace) {
  PlacementPlan p;
  p.ctx = ctx;
  p.kv_type = kv_type;
  p.kv_name = kv_type == 1 ? "q8_0" : (kv_type == 2 ? "q4_0" : "f32");
  p.vram_total = vram_total;
  p.vram_reserve = vram_reserve;
  p.workspace_bytes = workspace;

  p.fixed_bytes = m.fixed_bytes();
  // The budget must use the layout the runner really allocates (kv_type), not the
  // legacy bytes-per-value estimate: the two differ 2x for the default f32 cache, and
  // the plan's own [placement] section already used kv_type (same command printed
  // "kv cache 256 MiB" next to "KV 512 MiB").
  p.kv_bytes = placement::kv_bytes(m, ctx, kv_type);
  p.expert_total = m.expert_bytes();
  p.expert_per_layer = m.expert_bytes_per_layer();
  const int n_moe_layers = static_cast<int>(p.expert_per_layer.size());

  const uint64_t usable = vram_total > vram_reserve ? vram_total - vram_reserve : 0;
  const uint64_t need_fixed = p.fixed_bytes + p.kv_bytes + p.workspace_bytes;
  if (usable <= need_fixed) {
    p.expert_budget = 0;
    p.notes.push_back("fixed weights + KV already exceed VRAM; no expert can be resident");
  } else {
    p.expert_budget = usable - need_fixed;
  }

  if (n_moe_layers == 0) {
    p.slots_per_layer = 0;
    p.fully_resident = true;
  } else {
    // Advisory uniform estimate (what --n-cpu-moe expresses): how many MoE layers fit if
    // every layer had the average expert size.  Kept for display/back-compat.
    // double division to match pyref/planner.py exactly (int vs float diverges at boundaries)
    const double avg_layer = static_cast<double>(p.expert_total) / n_moe_layers;
    const uint64_t divisor = avg_layer >= 1.0 ? static_cast<uint64_t>(avg_layer) : 1;
    p.slots_per_layer = static_cast<int>(p.expert_budget / divisor);
    p.fully_resident = p.expert_budget >= p.expert_total;
    if (m.moe.n_expert > 0)
      p.slots_per_layer = std::min(p.slots_per_layer, m.moe.n_expert);
    p.slots_per_layer = std::max(0, p.slots_per_layer);
    // Exact layer-granular placement: walk the real per-layer expert bytes instead of
    // assuming the average (#27).  Tensor sizes differ per layer, so the average can
    // over/under-estimate where VRAM actually runs out.
    p.layers_on_gpu = 0;
    uint64_t used = 0;
    for (const auto& [layer, bytes] : p.expert_per_layer) {
      if (used + bytes > p.expert_budget) break;
      used += bytes;
      ++p.layers_on_gpu;
    }
    p.layers_on_cpu = std::max(0, n_moe_layers - p.layers_on_gpu);
    int idx = 0;
    for (const auto& [layer, bytes] : p.expert_per_layer) {
      (void)bytes;
      p.layer_placement[layer] = (idx++ < p.layers_on_gpu) ? "gpu" : "cpu";
    }
  }

  // projections
  uint64_t active_expert_bytes = 0;
  if (m.moe.n_expert > 0 && m.moe.n_expert_used > 0 && n_moe_layers > 0) {
    const double avg = static_cast<double>(p.expert_total) / n_moe_layers;
    const double share = static_cast<double>(m.moe.n_expert_used) / m.moe.n_expert;
    active_expert_bytes = static_cast<uint64_t>(
        avg * share * static_cast<double>(n_moe_layers));
  }
  const double hit = n_moe_layers ? static_cast<double>(p.layers_on_gpu) / n_moe_layers : 1.0;
  const uint64_t cpu_bytes = static_cast<uint64_t>(static_cast<double>(active_expert_bytes) * (1.0 - hit));
  // round to 2 decimals like pyref/planner.py so JSON dumps compare byte-equal
  auto round2 = [](double v) { return std::round(v * 100.0) / 100.0; };
  p.proj_decode_cpu_ms = round2(static_cast<double>(cpu_bytes) / (BW_CPU_GEMV_GBPS * 1e9) * 1e3);
  p.proj_decode_pcie_ms = round2(
      static_cast<double>(active_expert_bytes) / (BW_PCIE_H2D_GBPS * 1e9) * 1e3);

  if (!p.fully_resident && n_moe_layers)
    p.notes.push_back("expert placement: " + std::to_string(p.layers_on_gpu) + "/" +
                      std::to_string(n_moe_layers) + " MoE layers fit; llama.cpp equivalent is --n-cpu-moe " +
                      std::to_string(p.layers_on_cpu));
  if (m.ctx_train && ctx > m.ctx_train)
    p.notes.push_back("requested ctx " + std::to_string(ctx) + " > trained ctx " +
                      std::to_string(m.ctx_train));
  if (m.attn.variant == "mla")
    p.notes.push_back("MLA KV: " + std::to_string(m.attn.kv_lora_rank) + " latent + " +
                      std::to_string(m.attn.rope_dim) + " rope per layer/token");
  if (m.attn.full_attention_interval > 1)
    p.notes.push_back("hybrid model: only " +
                      std::to_string(static_cast<int>(std::ceil(
                          static_cast<double>(m.n_layer) / m.attn.full_attention_interval))) +
                      " of " + std::to_string(m.n_layer) + " layers hold KV");
  return p;
}

}  // namespace planner

// ===========================================================================
// Placement decider (declared in engine/placement.h)
// ===========================================================================
namespace placement {

namespace {

std::string fmt_bytes(uint64_t n) {
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

std::string fmt_mib(uint64_t n) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.0f MiB", static_cast<double>(n) / 1048576.0);
  return buf;
}

std::string fmt_u64(uint64_t v) { return std::to_string(v); }

// Number of layers that actually hold a KV cache (qwen35 hybrid stacks keep KV
// only on the full-attention layers).  Mirrors DenseRunner::derive_ctx_limit().
// Exported (placement.h) because `engine info --json` and the GUI estimate need
// the same layer count.

inline int nb32(int d) { return (d + 31) / 32; }

constexpr int kMaxCtxPolicy = 1 << 20;   // same absolute cap as derive_ctx_limit()

}  // namespace

int kv_layer_count(const ir::Model& m) {
  int n = 0;
  const size_t layers = m.layers.size();
  for (int i = 0; i < m.n_layer; ++i) {
    const bool kind_attn = i >= static_cast<int>(layers) || m.layers[static_cast<size_t>(i)].kind == "attn";
    const bool has = !(m.ssm.variant != "none" && !kind_attn);
    if (has) ++n;
  }
  if (n <= 0) n = m.n_layer;
  return n;
}

const char* kind_name(Kind k) {
  switch (k) {
    case Kind::Gpu: return "gpu";
    case Kind::Split: return "vram+ram split";
    case Kind::SsdStream: return "ssd stream";
  }
  return "?";
}

MemoryInfo host_memory() {
  MemoryInfo m;
#ifdef _WIN32
  MEMORYSTATUSEX ms;
  std::memset(&ms, 0, sizeof(ms));
  ms.dwLength = sizeof(ms);
  if (GlobalMemoryStatusEx(&ms)) {
    m.valid = true;
    m.total_bytes = static_cast<uint64_t>(ms.ullTotalPhys);
    m.free_bytes = static_cast<uint64_t>(ms.ullAvailPhys);
  }
#else
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long avail = sysconf(_SC_AVPHYS_PAGES);
  const long page = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && avail >= 0 && page > 0) {
    m.valid = true;
    m.total_bytes = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page);
    m.free_bytes = static_cast<uint64_t>(avail) * static_cast<uint64_t>(page);
  }
#endif
  return m;
}

MemoryInfo device_memory() {
  MemoryInfo m;
  try {
    cuda::init();
    const cuda::MemInfo mi = cuda::mem_info();
    m.total_bytes = static_cast<uint64_t>(mi.total_bytes);
    m.free_bytes = static_cast<uint64_t>(mi.free_bytes);
    m.valid = m.total_bytes > 0;
  } catch (...) {
    m.valid = false;
  }
  return m;
}

uint64_t default_host_reserve(uint64_t host_total_bytes) {
  const uint64_t gib = 1ull << 30;
  const uint64_t by_frac = host_total_bytes / 16;
  return by_frac > gib ? by_frac : gib;
}

uint64_t kv_bytes_per_token(const ir::Model& m, int kv_type) {
  const int n = kv_layer_count(m);
  if (n <= 0) return 0;
  if (m.attn.variant == "mla") {
    // MLA: the runner allocates the expanded per-head K and V cache (f32 only).
    return static_cast<uint64_t>(n) * static_cast<uint64_t>(m.attn.n_head) *
           static_cast<uint64_t>(m.attn.key_length + m.attn.value_length) * 4ull;
  }
  if (m.attn.n_head_kv <= 0 || m.attn.key_length <= 0) return 0;
  const uint64_t nkv = static_cast<uint64_t>(m.attn.n_head_kv);
  const uint64_t klen = static_cast<uint64_t>(m.attn.key_length);
  const uint64_t vlen = static_cast<uint64_t>(m.attn.value_length ? m.attn.value_length : m.attn.key_length);
  if (kv_type == 0) return static_cast<uint64_t>(n) * nkv * (klen + vlen) * 4ull;
  const uint64_t bb = kv_type == 1 ? 34ull : 18ull;
  return static_cast<uint64_t>(n) * nkv *
         (static_cast<uint64_t>(nb32(static_cast<int>(klen))) +
          static_cast<uint64_t>(nb32(static_cast<int>(vlen)))) *
         bb;
}

uint64_t kv_bytes(const ir::Model& m, int ctx, int kv_type) {
  const int c = ctx > 0 ? ctx : 1;
  return kv_bytes_per_token(m, kv_type) * static_cast<uint64_t>(c);
}

uint64_t activation_bytes(const ir::Model& m, int rows) {
  const size_t r = rows > 0 ? static_cast<size_t>(rows) : 1;
  const int hidden = m.hidden;
  const int n_head = m.attn.n_head;
  const int head_dim = m.attn.key_length;
  const int n_kv = m.attn.n_head_kv ? m.attn.n_head_kv : n_head;
  const int inter = m.ffn > 0 ? m.ffn : hidden * 4;
  size_t per_row = static_cast<size_t>(hidden) * 4 * 6;
  per_row += static_cast<size_t>(n_head) * static_cast<size_t>(head_dim) * 4 * 2;
  per_row += static_cast<size_t>(n_kv) * static_cast<size_t>(head_dim) * 4 * 2;
  per_row += static_cast<size_t>(inter) * 4 * 3;
  per_row += static_cast<size_t>(m.vocab) * 4;
  return r * per_row + (2ull << 20);
}

CtxFit estimate_ctx(const ir::Model& m, int kv_type, uint64_t free_vram,
                    uint64_t vram_reserve, int row_cap) {
  CtxFit f;
  f.kv_type = kv_type;
  f.kv_name = kv_type == 1 ? "q8_0" : (kv_type == 2 ? "q4_0" : "f32");
  if (m.attn.variant == "mla" && kv_type != 0) {
    f.kv_type = 0;
    f.kv_name = "f32";
  }
  f.kv_per_token = kv_bytes_per_token(m, f.kv_type);
  const int model_ctx = m.ctx_train > 0 ? static_cast<int>(std::min<int64_t>(m.ctx_train, kMaxCtxPolicy))
                                      : kMaxCtxPolicy;
  const int want = std::max(1, model_ctx);
  if (f.kv_per_token == 0) {
    f.max_ctx = want;
    return f;
  }
  const int cap = row_cap > 0 ? row_cap : 1;
  const uint64_t weights = m.fixed_bytes();
  const uint64_t reserve = vram_reserve;
  int seq = want;
  int fit = want;
  // Same fixed-point iteration as DenseRunner::derive_ctx_limit(): a longer ctx
  // means a bigger activation row buffer, which in turn lowers the KV budget.
  for (int it = 0; it < 4; ++it) {
    const int rows = std::max(1, std::min(seq, cap));
    const uint64_t act = activation_bytes(m, rows);
    const uint64_t avail = free_vram > weights + act + reserve
                              ? free_vram - weights - act - reserve
                              : 0;
    fit = static_cast<int>(avail / f.kv_per_token);
    fit = std::min(fit, want);
    if (fit >= seq) break;
    seq = fit;
    if (seq < 1) seq = 1;
  }
  f.max_ctx = std::max(0, std::min(want, fit));
  f.kv_at_max = f.kv_per_token * static_cast<uint64_t>(f.max_ctx);
  return f;
}

Decision decide(const Input& in) {
  Decision d;
  if (!in.model) {
    d.reason = "no model supplied";
    return d;
  }
  const ir::Model& m = *in.model;
  const uint64_t total = m.total_bytes();
  d.fixed_bytes = m.fixed_bytes();
  d.expert_bytes = m.expert_bytes();
  d.model_bytes = total ? total : d.fixed_bytes + d.expert_bytes;
  const int ctx = std::max(1, in.ctx);
  d.kv_bytes = kv_bytes(m, ctx, in.kv_type);
  d.workspace_bytes = in.workspace;

  d.vram_total = in.vram.total_bytes;
  d.vram_free = in.vram.free_bytes;
  d.vram_reserve = in.vram_reserve;
  d.vram_usable = d.vram_free > d.vram_reserve ? d.vram_free - d.vram_reserve : 0;
  d.host_total = in.host.total_bytes;
  d.host_free = in.host.free_bytes;
  d.host_reserve = in.host_reserve;
  d.host_usable = d.host_free > d.host_reserve ? d.host_free - d.host_reserve : 0;

  d.vram_need = d.fixed_bytes + d.kv_bytes + d.workspace_bytes;  // non-expert resident part
  const auto per_layer = m.expert_bytes_per_layer();
  d.n_moe_layers = static_cast<int>(per_layer.size());
  const uint64_t vram_need_all = d.vram_need + d.expert_bytes;

  if (!in.vram.valid) {
    d.kind = Kind::Gpu;
    d.reason = "VRAM probe unavailable: the loader's measured free VRAM decides";
    d.notes.push_back("device memory probe failed (no GPU/driver); residency was not verified");
    return d;
  }
  if (vram_need_all <= d.vram_usable) {
    d.kind = Kind::Gpu;
    d.fits_vram = true;
    d.fits_all = true;
    d.gpu_moe_layers = d.n_moe_layers;
    d.reason = "fixed + experts + KV + workspace (" + fmt_bytes(vram_need_all) +
               ") fit in usable VRAM (" + fmt_bytes(d.vram_usable) + ")";
    if (d.n_moe_layers)
      d.notes.push_back("all " + std::to_string(d.n_moe_layers) +
                        " MoE layers stay GPU-resident; no CPU/SSD path needed");
    return d;
  }
  if (d.n_moe_layers > 0 && d.vram_need <= d.vram_usable) {
    d.vram_left = d.vram_usable - d.vram_need;
    d.host_need = d.expert_bytes > d.vram_left ? d.expert_bytes - d.vram_left : 0;
    d.fits_all = in.host.valid && d.host_need <= d.host_usable;
    if (d.fits_all) {
      d.kind = Kind::Split;
      // Exact layer-granular estimate (same rule as the budget plan): walk the real
      // per-layer expert bytes instead of dividing by the average layer size, which
      // over/under-estimates where VRAM actually runs out (#27).
      d.gpu_moe_layers = 0;
      uint64_t used = 0;
      for (const auto& [layer, bytes] : per_layer) {
        (void)layer;
        if (used + bytes > d.vram_left) break;
        used += bytes;
        ++d.gpu_moe_layers;
      }
      d.gpu_moe_layers = std::max(0, std::min(d.gpu_moe_layers, d.n_moe_layers));
      d.reason = "weights exceed VRAM but fit VRAM + host RAM: CPU expert offload path";
      d.notes.push_back("VRAM keeps " + fmt_bytes(d.vram_left) + " of experts -> about " +
                        std::to_string(d.gpu_moe_layers) + "/" +
                        std::to_string(d.n_moe_layers) +
                        " MoE layers GPU-resident, the rest run on the host CPU path");
    } else {
      d.kind = Kind::SsdStream;
      d.reason = "weights do not fit VRAM + host RAM: only disk streaming remains";
      if (!in.host.valid)
        d.notes.push_back("host RAM probe unavailable: treated as not fitting");
    }
  } else {
    d.kind = Kind::SsdStream;
    d.dense_offload_unsupported = (d.n_moe_layers == 0);
    if (d.dense_offload_unsupported)
      d.reason = "dense model does not fit VRAM and this build has no dense host/SSD weight path";
    else
      d.reason = "non-expert weights + KV alone exceed usable VRAM";
  }
  if (in.ssd_requested && d.kind != Kind::SsdStream) {
    d.kind = Kind::SsdStream;
    d.reason = "SSD streaming forced by --ssd-stream / --placement ssd though the weights would fit";
  }
  if (d.kind == Kind::SsdStream) {
    if (d.host_need > 0)
      d.notes.push_back("would spill " + fmt_bytes(d.host_need) + " past host RAM (" +
                        fmt_bytes(d.host_usable) + " usable)");
    d.notes.push_back("SSD streaming is not implemented in this build; enable it explicitly with "
                      "--ssd-stream (interface placeholder) after lowering --ctx / using a smaller "
                      "quantisation, or pass --allow-oversubscribe to attempt the load anyway");
  }
  return d;
}

void print(const Decision& d, std::FILE* f) {
  std::fprintf(f, "[placement] ledger : model %s = fixed %s + experts %s",
               fmt_bytes(d.model_bytes).c_str(), fmt_bytes(d.fixed_bytes).c_str(),
               fmt_bytes(d.expert_bytes).c_str());
  if (d.n_moe_layers) std::fprintf(f, " (%d MoE layers)", d.n_moe_layers);
  std::fprintf(f, "\n");
  std::fprintf(f, "[placement] vram   : free %s (%s) / total %s, reserve %s -> usable %s\n",
               fmt_mib(d.vram_free).c_str(), fmt_u64(d.vram_free).c_str(),
               fmt_mib(d.vram_total).c_str(), fmt_mib(d.vram_reserve).c_str(),
               fmt_mib(d.vram_usable).c_str());
  std::fprintf(f, "[placement] host   : free %s / total %s, reserve %s -> usable %s\n",
               fmt_mib(d.host_free).c_str(), fmt_mib(d.host_total).c_str(),
               fmt_mib(d.host_reserve).c_str(), fmt_mib(d.host_usable).c_str());
  std::fprintf(f,
               "[placement] need   : fixed %s + KV %s + workspace %s = %s"
               " (+ experts -> %s)\n",
               fmt_mib(d.fixed_bytes).c_str(), fmt_mib(d.kv_bytes).c_str(),
               fmt_mib(d.workspace_bytes).c_str(), fmt_mib(d.vram_need).c_str(),
               fmt_mib(d.vram_need + d.expert_bytes).c_str());
  if (d.n_moe_layers) {
    std::fprintf(f,
                 "[placement] experts: %s total, %s fit VRAM -> %d/%d layers GPU-resident, "
                 "%s needs host RAM\n",
                 fmt_mib(d.expert_bytes).c_str(), fmt_mib(d.vram_left).c_str(),
                 d.gpu_moe_layers, d.n_moe_layers, fmt_mib(d.host_need).c_str());
  }
  std::fprintf(f, "[placement] decision: %s\n", kind_name(d.kind));
  std::fprintf(f, "[placement] reason : %s\n", d.reason.c_str());
  for (const auto& n : d.notes) std::fprintf(f, "[placement] note   : %s\n", n.c_str());
}

bool SsdExpertStreamer::open(const std::string& model_path, std::string* err) {
  const std::string dir = cfg_.cache_dir.empty() ? std::string("<unset: --ssd-cache>") : cfg_.cache_dir;
  const std::string msg =
      "SSD expert streaming is NOT IMPLEMENTED in this build (interface placeholder). "
      "would cache expert weights under '" + dir + "' for '" + model_path +
      "' (prefetch_depth=" + std::to_string(cfg_.prefetch_depth) + ", chunk=" +
      fmt_bytes(cfg_.io_chunk_bytes) + (cfg_.use_overlapped ? ", FILE_FLAG_OVERLAPPED" : "") +
      "); planned design: pread double-buffer + (layer,expert) LRU, see FEATURE-MATRIX.md §3";
  if (err) *err = msg;
  return false;
}

}  // namespace placement
