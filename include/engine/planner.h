// Budget planner: measured constants -> VRAM placement plan (spec §4).
// Constants are P0 measurements on this machine; do not swap for spec-sheet values.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "engine/ir.h"

namespace planner {

constexpr double BW_PCIE_H2D_GBPS = 12.0;
constexpr double BW_PCIE_D2H_GBPS = 11.0;
constexpr double BW_RAM_COPY_GBPS = 22.5;
constexpr double BW_CPU_GEMV_GBPS = 4.4;   // llama.cpp haswell AVX2, batch=1, Q5_K
constexpr double BW_HBM_GBPS = 360.0;
constexpr double T_COPY_LAT_US = 16.6;
constexpr uint64_t DEFAULT_VRAM_RESERVE = 1024ull << 20;
constexpr uint64_t DEFAULT_WORKSPACE = 256ull << 20;

struct PlacementPlan {
  int ctx = 0;
  // KV layout the runner actually allocates (0 f32 / 1 q8_0 / 2 q4_0).  This used to
  // be a legacy `kv_bpe` bytes-per-value parameter (default 2) that ignored kv_type and
  // under-reported the KV cache by 2x for the default f32 layout.
  int kv_type = 0;
  const char* kv_name = "f32";
  uint64_t vram_total = 0, vram_reserve = 0;
  uint64_t fixed_bytes = 0, kv_bytes = 0, workspace_bytes = 0;
  uint64_t expert_budget = 0, expert_total = 0;
  std::map<int, uint64_t> expert_per_layer;
  // MoE block index -> "gpu" / "cpu" (the layer-granular placement the plan suggests).
  std::map<int, std::string> layer_placement;
  int slots_per_layer = 0;   // advisory: uniform expert slots/layer that fit the budget
  int layers_on_gpu = 0, layers_on_cpu = 0;
  bool fully_resident = false;
  double proj_decode_cpu_ms = 0, proj_decode_pcie_ms = 0;
  std::vector<std::string> notes;
};

PlacementPlan compute(const ir::Model& m, int ctx = 8192, int kv_type = 0,
                      uint64_t vram_total = 12ull << 30,
                      uint64_t vram_reserve = DEFAULT_VRAM_RESERVE,
                      uint64_t workspace = DEFAULT_WORKSPACE);

}  // namespace planner
