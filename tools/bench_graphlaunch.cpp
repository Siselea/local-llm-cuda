// Microbenchmark: what does one cuGraphLaunch cost versus one cuLaunchKernel on this
// WDDM host?  This decides whether per-layer CUDA-graph coverage is worth it: a
// DeepSeek-V2-Lite (MLA) decode layer issues ~18 tiny launches, a graph replaces them
// with a single cuGraphLaunch.
//
//   bash tools/bench_graphlaunch.sh        (build + run)
//
// Reported per-op times are host-side submission costs (wall clock for a batch, divided
// by the batch size, GPU-side elapsed also shown for the 18-kernel sequences).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "engine/cuda.h"

namespace {

const char* kSrc = R"(
extern "C" __global__ void nop(float* out, int n) {
  if (blockIdx.x > n) return;
  if (threadIdx.x == 0x3fffffff) out[0] = 1.0f;
}
)";

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// Median us/iteration over `reps` batches of `iters` iterations of `once`.
double per_iter_us(const char* tag, const std::function<void()>& once, int iters, int reps) {
  std::vector<double> us;
  for (int r = 0; r < reps; ++r) {
    cuda::sync();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) once();
    cuda::sync();
    const auto t1 = std::chrono::steady_clock::now();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / iters);
  }
  const double m = median(us);
  std::printf("  %-26s %8.2f us/iter   (min %.2f, n=%d)\n", tag, m,
              *std::min_element(us.begin(), us.end()), reps);
  return m;
}

}  // namespace

int main() {
  cuda::init();
  // The engine runs on a non-default stream; graph capture is not allowed on the
  // legacy default stream, so mirror the engine's setup here.
  void* stream = cuda::create_stream();
  cuda::set_stream(stream);
  const auto& dev = cuda::device_info();
  std::printf("device: %s (sm %d.%d)\n", dev.name.c_str(), dev.cc_major, dev.cc_minor);

  cuda::Module mod(cuda::compile_ptx(kSrc, "graphlaunch_bench"));
  void* fn = mod.kernel("nop");
  cuda::Buffer buf(16);
  float* pout = static_cast<float*>(buf.data());
  int n = 0;
  void* args[] = {&pout, &n};

  const int iters = 200, reps = 7;

  // 1) bare kernel launches, one per iteration
  const double bare1 = per_iter_us("bare launch x1", [&]() {
    cuda::launch(fn, 1, 1, 1, 32, 1, 1, 0, args);
  }, iters, reps);

  // 2) graph with one node, replayed once per iteration
  cuda::Graph g1;
  cuda::stream_sync();
  g1.capture_begin();
  cuda::launch(fn, 1, 1, 1, 32, 1, 1, 0, args);
  g1.capture_end();
  const double graph1 = per_iter_us("graph launch (1 node)", [&]() { g1.launch(); }, iters, reps);

  // 3) the shape that matters: 18 bare launches vs one 18-node graph
  std::vector<void*> kernel_fn(18, fn);
  const double bare18 = per_iter_us("bare launch x18", [&]() {
    for (int i = 0; i < 18; ++i) cuda::launch(fn, 1, 1, 1, 32, 1, 1, 0, args);
  }, iters, reps);

  cuda::Graph g18;
  cuda::stream_sync();
  g18.capture_begin();
  for (int i = 0; i < 18; ++i) cuda::launch(fn, 1, 1, 1, 32, 1, 1, 0, args);
  g18.capture_end();
  const double graph18 = per_iter_us("graph launch (18 nodes)", [&]() { g18.launch(); }, iters, reps);

  // GPU-side elapsed for both 18-kernel sequences (should be identical: the graph does
  // not change the work, only the submission path).
  auto gpu_batch_us = [&](const std::function<void()>& once) {
    cuda::Event e0, e1;
    cuda::sync();
    e0.record();
    for (int i = 0; i < iters; ++i) once();
    e1.record();
    e1.synchronize();
    return e0.elapsed_ms(e1) * 1000.0 / iters;
  };
  const double gpu_bare18 = gpu_batch_us([&]() {
    for (int i = 0; i < 18; ++i) cuda::launch(fn, 1, 1, 1, 32, 1, 1, 0, args);
  });
  const double gpu_graph18 = gpu_batch_us([&]() { g18.launch(); });
  std::printf("  %-26s %8.2f us/batch (GPU elapsed)\n", "18 bare launches", gpu_bare18);
  std::printf("  %-26s %8.2f us/batch (GPU elapsed)\n", "one 18-node graph", gpu_graph18);

  // Current engine launch-call count per decode token for Coder (27 layers x 18).
  const int launches_per_token = 27 * 18;
  std::printf("\nbare kernel launch:        %.2f us\n", bare1);
  std::printf("cuGraphLaunch (1 node):    %.2f us   (%.1fx one kernel launch)\n", graph1,
              bare1 > 0 ? graph1 / bare1 : 0.0);
  std::printf("18 bare launches:          %.2f us\n", bare18);
  std::printf("one 18-node graph:         %.2f us\n", graph18);
  std::printf("saving per layer:          %+.2f us\n", bare18 - graph18);
  std::printf("Coder per-token (%d launches in 27 graphs):\n", launches_per_token);
  std::printf("  graph path  ~ %.2f ms/token of submission (27 x %.2f us)\n", graph18 * 27 / 1000.0,
              graph18);
  std::printf("  bare path   ~ %.2f ms/token of submission (27 x %.2f us)\n", bare18 * 27 / 1000.0,
              bare18);
  std::printf("  verdict:    %s\n",
              (bare18 - graph18) * 27 / 1000.0 > 0.5 ? "WORTH IT (>=0.5 ms/token)"
                                                     : "NOT WORTH IT (<0.5 ms/token)");
  return 0;
}
