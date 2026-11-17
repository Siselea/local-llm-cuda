// Placement decision: where do the weights live?
//
// The engine has three placement strategies, decided from *measured* free VRAM,
// *measured* free host RAM, the model's real resident bytes (quantised GGUF
// tensor sizes, not parameter counts) and configurable reserves:
//
//   gpu    every weight + KV cache + activations fit in VRAM (the fast path;
//          on a MoE model this also means every expert stays resident)
//   split  the weights do not fit VRAM alone but VRAM + host RAM do, so the MoE
//          experts are placed per layer: GPU-resident layers are fast, the rest
//          run on the host CPU path.  Dense models cannot be split (the engine
//          has no host dense-weight path); for them this decision degrades to
//          "does not fit" instead of silently paging.
//   ssd    the weights do not fit VRAM + RAM: the only remaining option is
//          streaming expert/gate weights from disk.  **Not implemented in this
//          build** - see SsdExpertStreamer below for the interface placeholder
//          and FEATURE-MATRIX.md §3 for the design (pread double-buffer /
//          FILE_FLAG_OVERLAPPED + expert LRU cache).  It is a candidate ONLY in
//          this case and is never enabled by default: on PCIe x8 / HDD the
//          measured streaming bandwidth (~4.4 GB/s CPU lane, ~12 GB/s PCIe H2D)
//          is usually a net loss against a smaller/kv-quantised model.
//
// The implementation lives in src/planner.cpp next to the budget planner because
// both share the same byte-accounting model (and the build source list already
// contains planner.cpp; the decider must not fork the accounting rules).
#pragma once

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "engine/ir.h"

namespace placement {

// ---------------------------------------------------------------------------
// Memory probes
// ---------------------------------------------------------------------------
struct MemoryInfo {
  bool valid = false;
  uint64_t total_bytes = 0;
  uint64_t free_bytes = 0;   // currently available (not total - used by others)
};

// Host RAM via GlobalMemoryStatusEx (Windows) / sysconf (POSIX).
MemoryInfo host_memory();

// Free/total VRAM via cuda::mem_info().  Calls cuda::init() on first use and
// returns valid=false (instead of throwing) when the driver/GPU is unavailable.
MemoryInfo device_memory();

// Default host reserve used when host_margin_mb = 0: max(1 GiB, total RAM / 16).
// Expressed as a fraction of the machine instead of a hard-coded MiB so a 16 GiB
// or 64 GiB box gets a sensible reserve without retuning.
uint64_t default_host_reserve(uint64_t host_total_bytes);

// ---------------------------------------------------------------------------
// Byte accounting that mirrors the real runner allocations
// ---------------------------------------------------------------------------

// KV cache bytes per token for the layout the engine actually allocates:
//   kv_type 0 = f32 (historical layout), 1 = q8_0 (34 B / 32 values),
//   2 = q4_0 (18 B / 32 values).  MLA models allocate the expanded per-head
//   K/V cache (f32 only; quantised MLA attention kernels do not exist yet).
uint64_t kv_bytes_per_token(const ir::Model& m, int kv_type);
uint64_t kv_bytes(const ir::Model& m, int ctx, int kv_type);

// Number of layers that actually hold a KV cache (hybrid stacks keep KV only on
// the full-attention layers; dense/MoE-attention models use every layer).
int kv_layer_count(const ir::Model& m);

// Activation/scratch bytes for `rows` batch rows - mirror of
// DenseRunner::activation_bytes() (same fields, same formula).
uint64_t activation_bytes(const ir::Model& m, int rows);

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------
enum class Kind { Gpu, Split, SsdStream };
const char* kind_name(Kind k);   // "gpu" | "vram+ram split" | "ssd stream"

struct Input {
  const ir::Model* model = nullptr;
  int ctx = 0;                 // context length used for the KV accounting
  int kv_type = 0;             // 0 f32 / 1 q8_0 / 2 q4_0
  MemoryInfo vram;
  MemoryInfo host;
  uint64_t vram_reserve = 0;   // free VRAM to keep after loading (vram_margin_mb)
  uint64_t host_reserve = 0;   // free host RAM to keep after loading
  uint64_t workspace = 0;      // KV/activation/scratch reserve used for planning
  bool ssd_requested = false;  // user explicitly asked for SSD streaming
};

struct Decision {
  Kind kind = Kind::Gpu;

  uint64_t model_bytes = 0;      // fixed + experts (resident weight bytes)
  uint64_t fixed_bytes = 0;      // everything except MoE experts
  uint64_t expert_bytes = 0;
  uint64_t kv_bytes = 0;
  uint64_t workspace_bytes = 0;

  uint64_t vram_total = 0, vram_free = 0, vram_reserve = 0, vram_usable = 0;
  uint64_t vram_need = 0, vram_left = 0;
  uint64_t host_total = 0, host_free = 0, host_reserve = 0, host_usable = 0;
  uint64_t host_need = 0;        // expert bytes that must live in host RAM

  int n_moe_layers = 0;
  int gpu_moe_layers = 0;        // layers the decide() would keep on the GPU

  bool fits_vram = false;        // fixed + KV + workspace fit VRAM
  bool fits_all = false;         // ... and the remaining experts fit host RAM
  bool dense_offload_unsupported = false;  // dense model that does not fit VRAM

  std::string reason;            // one-line rationale for the decision
  std::vector<std::string> notes;
};

Decision decide(const Input& in);

// Print the full accounting in the [placement] log style (stderr by default).
void print(const Decision& d, std::FILE* f = stderr);

// ---------------------------------------------------------------------------
// Pre-load context recommendation (used by the interactive picker and `plan`)
//
// This mirrors DenseRunner::derive_ctx_limit(): it iterates the activation row
// cap until the KV size that fits stops shrinking the row buffer.  The load-time
// derivation is still authoritative (it re-measures free VRAM after the weights
// are resident); this estimate exists because the interactive prompt must run
// *before* the model is loaded.
// ---------------------------------------------------------------------------
struct CtxFit {
  int kv_type = 0;
  const char* kv_name = "f32";
  uint64_t kv_per_token = 0;
  int max_ctx = 0;        // clamped to ctx_train / 1M policy cap
  uint64_t kv_at_max = 0;
};

CtxFit estimate_ctx(const ir::Model& m, int kv_type, uint64_t free_vram,
                    uint64_t vram_reserve, int row_cap);

// ---------------------------------------------------------------------------
// SSD streaming interface (PLACEHOLDER - not implemented in this build)
//
// A runnable design (documented in FEATURE-MATRIX.md §3):
//   * experts are the only weights that stream (attention / norms / embeddings
//     stay resident; they are small and touched every token);
//   * a pinned host staging buffer, double buffered: while expert slot n is
//     consumed, slot n+1 is filled with pread()/ReadFile(FILE_FLAG_OVERLAPPED);
//   * an expert LRU keyed by (layer, expert) with a byte budget = ssd_cache_mb;
//     capacity is derived from measured free RAM, not hard coded;
//   * hits are served from the LRU, misses schedule an overlapped read and the
//     layer waits on its event (no global sync, no CUDA managed memory: the
//     earlier zero-copy/UVM experiments on this VM were net-negative).
// open() returns false with an explicit "not implemented" reason today, so a
// user who forces --ssd-stream gets a deterministic error instead of a silent
// fallback that would thrash the disk.
// ---------------------------------------------------------------------------
struct SsdStreamConfig {
  std::string cache_dir;              // on-disk expert cache location
  int prefetch_depth = 2;             // 2 = one read in flight + one in use
  size_t io_chunk_bytes = 4u << 20;   // read granularity
  bool use_overlapped = true;         // FILE_FLAG_OVERLAPPED (Windows)
};

class SsdExpertStreamer {
 public:
  explicit SsdExpertStreamer(const SsdStreamConfig& cfg) : cfg_(cfg) {}
  bool open(const std::string& model_path, std::string* err);
  bool ready() const { return false; }
  const SsdStreamConfig& config() const { return cfg_; }

 private:
  SsdStreamConfig cfg_;
};

}  // namespace placement
