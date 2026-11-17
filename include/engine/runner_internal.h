// Internal helpers shared between the runner translation units.
#pragma once

#include <string>
#include <vector>

#include "engine/cuda.h"

namespace runner {

struct QuantKernel {
  const char* name;
  int block_bytes;
  int block_elems;
  int threads;
  const char* src;
};

inline cuda::Buffer upload(const std::vector<float>& v) {
  cuda::Buffer b(v.size() * 4);
  b.upload(v.data(), v.size() * 4);
  return b;
}

inline const QuantKernel* quant_kernel(const std::string& type) {
  static const QuantKernel kQ4K{"dequant_q4k", 144, 256, 256, nullptr};
  static const QuantKernel kQ5K{"dequant_q5k", 176, 256, 256, nullptr};
  static const QuantKernel kQ6K{"dequant_q6k", 210, 256, 32, nullptr};
  static const QuantKernel kQ80{"dequant_q8_0", 34, 32, 32, nullptr};
  static const QuantKernel kQ50{"dequant_q5_0", 22, 32, 32, nullptr};
  static const QuantKernel kQ51{"dequant_q5_1", 24, 32, 32, nullptr};
  if (type == "Q4_K") return &kQ4K;
  if (type == "Q5_K") return &kQ5K;
  if (type == "Q6_K") return &kQ6K;
  if (type == "Q8_0") return &kQ80;
  if (type == "Q5_0") return &kQ50;
  if (type == "Q5_1") return &kQ51;
  return nullptr;
}

}  // namespace runner
