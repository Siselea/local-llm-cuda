// CPU-side matvec over quantized blocks (fallback path for CPU-placed experts).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace cpu {

// Dequantize one block into `out` (block_elems floats).
void dequant_block(const std::string& type, const unsigned char* p, float* out);

// y[n_out] = W[n_out][n_in] * x[n_in], W stored row-major as quantized blocks.
void matvec(const std::string& type, const unsigned char* w, int n_out, int n_in,
            const float* x, float* y);

// Same, but parallelised over output rows (used by the CPU-placed expert path).
void matvec_mt(const std::string& type, const unsigned char* w, int n_out, int n_in,
               const float* x, float* y);

// Batched CPU MoE for one layer: every routed expert of the layer is evaluated in
// two parallel regions (gate+up+silu, then down) instead of one pool wake-up per
// matvec (24 wake-ups x ~25 us dominated the 35B's CPU layers).  Numerically
// identical to calling matvec_mt per expert: each row uses the same dot kernel and
// the partial sums are accumulated into xout in the same expert order.
//
//   wg/wu/wd[act]  quantized gate/up/down slabs of the act-th routed expert
//   tokens[act]    token row in x this act applies to
//   act_w[act]     routing weight
//   x              [n_tokens][hidden] activations (host)
//   xout           [n_tokens][hidden] accumulator (host, pre-zeroed)
void moe_forward_mt(const std::string& tg, const unsigned char* const* wg,
                    const std::string& tu, const unsigned char* const* wu,
                    const std::string& td, const unsigned char* const* wd,
                    const int* tokens, const float* act_w, int n_act, int n_tokens, int hidden,
                    int inter, const float* x, float* xout);

}  // namespace cpu
