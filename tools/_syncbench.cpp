#include "engine/cuda.h"
#include <chrono>
#include <cstdio>
int main() {
  cuda::init();
  void* s = cuda::create_stream();
  cuda::set_stream(s);
  cuda::Buffer a(64), b(64);
  b.upload(a.data() ? &a : nullptr, 0);  // no-op-ish
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto us = [](auto x, auto y) { return std::chrono::duration<double, std::micro>(y - x).count(); };
  // warm-up
  for (int i = 0; i < 20; ++i) { cuda::copy_d2d_async(a.data(), b.data(), 64); cuda::stream_sync(); }
  for (int batch : {1, 2, 4, 8}) {
    double t_launch = 0, t_sync = 0;
    const int iters = 200;
    for (int i = 0; i < iters; ++i) {
      auto t0 = now();
      for (int k = 0; k < batch; ++k) cuda::copy_d2d_async(a.data(), b.data(), 64);
      auto t1 = now();
      cuda::stream_sync();
      auto t2 = now();
      t_launch += us(t0, t1);
      t_sync += us(t1, t2);
    }
    std::printf("batch=%d  issue %.1f us  sync(flush+wait) %.1f us\n", batch, t_launch / iters,
                t_sync / iters);
  }
  cuda::destroy_stream(s);
  return 0;
}
