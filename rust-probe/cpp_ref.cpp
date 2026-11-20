// Exposes the engine's own AVX2 Q4_K row-dot kernel to the A/B harness.
//
// The kernel is a `static` function inside src/cpu_matvec.cpp, so the cleanest way
// to call *the exact same machine code* is to include that translation unit here
// and provide a C wrapper.  This file then replaces src/cpu_matvec.cpp on the
// harness link line (it contains all of its definitions).
#include "../src/cpu_matvec.cpp"

extern "C" void cpp_q4k_dot_row(const unsigned char* row, const float* x, int n_in, float* out) {
  cpu::avx2_dot_q4k(row, x, n_in, out);
}

extern "C" void cpp_q4k_matvec(const unsigned char* w, int n_out, int n_in, const float* x,
                               float* y) {
  const size_t row_bytes = static_cast<size_t>(n_in / 256) * 144;
  for (int r = 0; r < n_out; ++r)
    cpu::avx2_dot_q4k(w + static_cast<size_t>(r) * row_bytes, x, n_in, y + r);
}
