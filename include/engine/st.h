// safetensors + HF config.json reader (split-expert layout).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace st {

struct Tensor {
  std::string name;
  std::string dtype;
  std::vector<uint64_t> shape;
  uint64_t begin = 0;       // offset inside the data section
  uint64_t end = 0;
  std::string file;
  uint64_t data_start = 0;  // absolute offset of the data section in `file`

  uint64_t nbytes() const { return end - begin; }
  uint64_t n_elements() const;
};

uint64_t dtype_bytes(const std::string& dtype);  // throws on unsupported

std::map<std::string, Tensor> read_file(const std::string& path);

struct Shards {
  std::map<std::string, Tensor> tensors;
  uint64_t total_bytes = 0;
};

Shards load_dir(const std::string& dir);

}  // namespace st
