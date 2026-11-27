// Regression test for review item #10: the GGUF F16/F32 MoE expert fallback.
//
// The fallback path (taken when a MoE GGUF stores its expert tensors as F16/F32 instead
// of a quantized type) used to upload every expert to the GPU and never fill the
// host-side expert buffers (w.host_gate / w.host_up / w.host_down).  A MoE layer that the
// layer-granular placement keeps on the CPU then had no weights at all: run_moe's host
// branch indexes those vectors by expert id, so it read out of bounds (crash or garbage
// logits).  This test loads an F16-expert tiny MoE model with different placements and
// compares the logits against the generator's numpy reference:
//   A) default (all experts on the GPU)
//   B) --gpu-layers 0 (every expert on the CPU: the case that used to crash)
//   C) --gpu-experts 1 (mixed: one expert per layer on the GPU, the rest on the CPU)
//
// Usage: test_gguf_moe_offload <model.gguf> <fixture_dir>
// Skips (exit 0) with a message when no CUDA device is available, like the other GPU tests.
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <algorithm>

#include "engine/config.h"
#include "engine/cuda.h"
#include "engine/runner.h"

namespace {

std::vector<int> read_i32(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  f.seekg(0, std::ios::end);
  const auto n = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<int> v(n / 4);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
  return v;
}

std::vector<float> read_f32(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  f.seekg(0, std::ios::end);
  const auto n = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<float> v(n / 4);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
  return v;
}

// Run one placement and compare against the reference logits.  `set` applies config keys
// before the runner is constructed.
struct Placement {
  int gpu_experts = -1;
  int host_experts = -1;
  int cpu_begin = -2;
};
Placement g_placement;

bool run_case(const char* label, const std::string& model, const std::vector<int>& tokens,
              const std::vector<float>& ref, const char* key, const char* value,
              bool freeze_placement = false) {
  engine::config::Config& cfg = engine::config::Config::get();
  // Fresh process state is not available, so reset the key we may have set before.
  std::string err;
  if (key != nullptr) cfg.set(key, value, engine::config::Source::Cli, &err);
  // The measured per-layer VRAM check may promote CPU experts back to the GPU when the
  // model fits trivially (the tiny fixture always does); disable that so the placement
  // under test is exactly what the config asked for.
  if (freeze_placement) cfg.set("vram_extend", "0", engine::config::Source::Cli, &err);
  else cfg.set("vram_extend", "1", engine::config::Source::Cli, &err);
  try {
    runner::DenseRunner r(model, 64, runner::Container::Gguf);
    // Placement contract exposed by the loader: a CPU-placed expert must have a host
    // copy and must NOT be uploaded.  The pre-fix F16/F32 fallback ignored the placement
    // and uploaded every expert, leaving the host buffers empty.
    const auto pl = r.debug_expert_placement(0);
    g_placement = {pl.gpu_experts, pl.host_experts, pl.cpu_begin};
    const auto got = r.forward(tokens);
    if (got.size() != ref.size()) {
      std::fprintf(stderr, "FAIL %s: logits size %zu vs %zu\n", label, got.size(), ref.size());
      return false;
    }
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double e = static_cast<double>(got[i]) - ref[i];
      num += e * e;
      den += static_cast<double>(ref[i]) * ref[i];
      mx = std::fmax(mx, std::fabs(e));
    }
    const double rel = std::sqrt(num / std::fmax(den, 1e-30));
    const int argmax_ok =
        static_cast<int>(std::distance(got.begin(),
                                       std::max_element(got.end(), got.end()))) >= 0;
    (void)argmax_ok;
    const bool finite = std::isfinite(rel) && rel < 2e-2;
    std::printf("  %-28s rel_l2=%.3e max|diff|=%.3e gpu=%d host=%d cpu_begin=%d -> %s\n", label,
                rel, mx, g_placement.gpu_experts, g_placement.host_experts,
                g_placement.cpu_begin, finite ? "PASS" : "FAIL");
    return finite;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL %s: %s\n", label, e.what());
    return false;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: test_gguf_moe_offload <model.gguf> <fixture_dir>\n");
    return 1;
  }
  const std::string model = argv[1], dir = argv[2];
  // Optional expected expert count (default 4).  Used by the >256-expert cases.
  const int ne = (argc > 3) ? std::atoi(argv[3]) : 4;
  if (ne <= 0) {
    std::fprintf(stderr, "invalid expert count %d\n", ne);
    return 1;
  }
  try {
    cuda::init();
  } catch (const std::exception& e) {
    std::printf("test_gguf_moe_offload: SKIPPED (no CUDA device: %s)\n", e.what());
    return 0;
  }
  try {
    const auto tokens = read_i32(dir + "/tokens.i32");
    const auto ref = read_f32(dir + "/ref_logits.f32");
    bool ok = true;
    // A) all experts on the GPU (baseline path, must keep matching the reference)
    ok &= run_case("all-GPU experts", model, tokens, ref, "gpu_layers", "0x7fffffff");
    // all-GPU: every expert uploaded, no host copies needed.
    if (!(g_placement.gpu_experts == ne && g_placement.host_experts == 0)) {
      std::fprintf(stderr, "FAIL: all-GPU placement is gpu=%d host=%d (want %d/0)\n",
                   g_placement.gpu_experts, g_placement.host_experts, ne);
      ok = false;
    }
    // B) every expert on the CPU.  The device footprint must shrink: before the fix the
    // fallback uploaded every expert regardless of the placement (and left the host
    // buffers empty, so the CPU path had no weights).
    ok &= run_case("all-CPU experts", model, tokens, ref, "gpu_layers", "0", true);
    // all-CPU: nothing uploaded, all four experts kept on the host, cpu_begin == 0.
    if (!(g_placement.gpu_experts == 0 && g_placement.host_experts == ne &&
          g_placement.cpu_begin == 0)) {
      std::fprintf(stderr,
                   "FAIL: all-CPU placement is gpu=%d host=%d cpu_begin=%d (want 0/%d/0)\n",
                   g_placement.gpu_experts, g_placement.host_experts, g_placement.cpu_begin,
                   ne);
      ok = false;
    }
    // C) mixed placement: one expert per layer on the GPU, the rest on the CPU.
    engine::config::Config::get().set("gpu_layers", "-1", engine::config::Source::Cli, nullptr);
    ok &= run_case("mixed experts (1 GPU)", model, tokens, ref, "gpu_experts", "1", true);
    // mixed: one expert uploaded, host copies for all (the CPU path indexes by expert
    // id), cpu_begin == 1.
    if (!(g_placement.gpu_experts == 1 && g_placement.host_experts == ne &&
          g_placement.cpu_begin == 1)) {
      std::fprintf(stderr,
                   "FAIL: mixed placement is gpu=%d host=%d cpu_begin=%d (want 1/%d/1)\n",
                   g_placement.gpu_experts, g_placement.host_experts, g_placement.cpu_begin,
                   ne);
      ok = false;
    }
    if (!ok) {
      std::fprintf(stderr, "test_gguf_moe_offload: FAILED\n");
      return 1;
    }
    std::printf("test_gguf_moe_offload: PASS (GPU / all-CPU / mixed expert placement)\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gguf_moe_offload: error: %s\n", e.what());
    return 2;
  }
}
