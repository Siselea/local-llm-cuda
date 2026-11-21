// Per-token fixed-cost microbench for the decode budget audit.
// Measures host-side cost of the driver calls / copies the engine makes once per token:
//   * cuEventCreate + cuEventDestroy (run_pipeline constructs 2 Events per call)
//   * pageable H2D async (4 B position scalar, 16 KiB activation row) vs pinned H2D async
//   * cuStreamSynchronize with an empty queue
//   * one cuda::launch (submission cost, warm queue)
// Run when the GPU is free; numbers are host-side microseconds per call.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "engine/cuda.h"

namespace {
double med(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}
template <class F>
double per_call_us(F&& fn, int iters, int reps) {
  std::vector<double> us;
  for (int r = 0; r < reps; ++r) {
    cuda::sync();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    cuda::sync();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / iters);
  }
  return med(us);
}
}  // namespace

int main() {
  cuda::init();
  void* stream = cuda::create_stream();
  cuda::set_stream(stream);
  const char* src = R"(
extern "C" __global__ void nop(float* out, int n) { if (blockIdx.x > n) return; if (threadIdx.x == 0x3fffffff) out[0] = 1.0f; }
)";
  cuda::Module mod(cuda::compile_ptx(src, "overhead_bench"));
  void* fn = mod.kernel("nop");
  cuda::Buffer buf(16);
  float* pout = static_cast<float*>(buf.data());
  int n = 0;
  void* args[] = {&pout, &n};

  const int pageable16k = 16384 / 4;
  std::vector<float> pageable(static_cast<size_t>(pageable16k), 0.f);
  cuda::PinnedBuffer pinned(16384);
  cuda::Buffer dev16k(16384), dev4(4);
  int pos = 7;

  const int iters = 300, reps = 7;
  const double ev_create = per_call_us([] {
    cuda::Event e;
    (void)e;
  }, 200, reps);
  const double sync_empty = per_call_us([] { cuda::stream_sync(); }, iters, reps);
  const double launch1 = per_call_us([&] { cuda::launch(fn, 1, 1, 1, 32, 1, 1, 0, args); },
                                     iters, reps);
  const double pageable_4b =
      per_call_us([&] { dev4.upload_async(&pos, 4); }, iters, reps);
  const double pinned_4b = per_call_us([&] {
    static cuda::PinnedBuffer p4(4);
    std::memcpy(p4.data(), &pos, 4);
    dev4.upload_async(p4.data(), 4);
  }, iters, reps);
  const double pageable_16k =
      per_call_us([&] { dev16k.upload_async(pageable.data(), 16384); }, iters, reps);
  const double pinned_16k = per_call_us([&] {
    std::memcpy(pinned.data(), pageable.data(), 16384);
    dev16k.upload_async(pinned.data(), 16384);
  }, iters, reps);
  // round trip that includes the host wait (what a per-token sync costs when the queue
  // has one kernel in it)
  const double kernel_roundtrip = per_call_us([&] {
    cuda::launch(fn, 1, 1, 1, 32, 1, 1, 0, args);
    cuda::stream_sync();
  }, 200, reps);

  std::printf("host-side per-call cost (median us):\n");
  std::printf("  Event ctor+dtor (cuEventCreate+Destroy) %7.2f\n", ev_create);
  std::printf("  stream_sync, empty queue                %7.2f\n", sync_empty);
  std::printf("  cuda::launch (warm queue)               %7.2f\n", launch1);
  std::printf("  upload_async 4B  from pageable          %7.2f\n", pageable_4b);
  std::printf("  upload_async 4B  from pinned+memcpy     %7.2f\n", pinned_4b);
  std::printf("  upload_async 16K from pageable          %7.2f\n", pageable_16k);
  std::printf("  upload_async 16K from pinned+memcpy     %7.2f\n", pinned_16k);
  std::printf("  launch + stream_sync round trip         %7.2f\n", kernel_roundtrip);
  std::printf("\nper-token engine candidates (x = estimate):\n");
  std::printf("  2 Events in run_pipeline                ~2 x %.2f = %.2f us\n", ev_create,
              2 * ev_create);
  std::printf("  pageable pos 4B -> pinned equivalent    %.2f us saved\n",
              pageable_4b - pinned_4b);
  std::printf("  pageable x 16K -> pinned equivalent     %.2f us saved\n",
              pageable_16k - pinned_16k);
  return 0;
}
