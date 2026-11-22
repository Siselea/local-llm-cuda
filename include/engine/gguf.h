// GGUF v2/v3 reader: header + metadata + tensor table only (no tensor data).
// Mirrors pyref/gguf_parse.py byte for byte.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gguf {

constexpr uint32_t MAGIC = 0x46554747u;  // "GGUF"

enum class ValueType : uint32_t {
  UINT8 = 0, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
  STRING, ARRAY, UINT64, INT64, FLOAT64,
};

struct Value {
  enum class Kind { INT, FLOAT, STRING, BOOL, ARRAY } kind = Kind::INT;
  int64_t i = 0;
  double f = 0.0;
  bool b = false;
  std::string s;
  uint64_t array_len = 0;
  uint32_t array_elem_type = 0;
  // Array payload is materialised for (a) the tokenizer keys requested through
  // read(path, capture_keys), and (b) every small numeric array: model geometry
  // metadata such as per-layer head counts / sliding-window patterns lives in
  // such arrays, and the IR derives per-layer behaviour from them generically.
  std::vector<std::string> strs;   // STRING array elements
  std::vector<unsigned char> raw;  // scalar array elements, little endian

  // Decode element `idx` of a numeric array (returns 0 when out of range).
  int64_t num_at(size_t idx) const;
};

// ggml tensor type -> (name, block_size, bytes_per_block)
struct TypeInfo { const char* name; int block; int bytes; };
const TypeInfo* type_info(uint32_t type_id);  // nullptr when unsupported

struct TensorInfo {
  std::string name;
  std::vector<uint64_t> shape;   // logical order (gguf dims reversed)
  uint32_t type_id = 0;
  std::string type_name;
  uint64_t offset = 0;           // relative to data section
  uint64_t n_elements = 0;
  uint64_t nbytes = 0;
};

class Model {
 public:
  std::string path;
  uint32_t version = 0;
  uint64_t n_tensors = 0;
  uint64_t n_kv = 0;
  std::map<std::string, Value> kv;
  std::vector<TensorInfo> tensors;
  uint64_t data_offset = 0;
  uint64_t file_size = 0;
  uint32_t alignment = 32;

  bool has(const std::string& key) const { return kv.count(key) != 0; }

  int64_t get_int(const std::string& key, int64_t fallback = INT64_MIN) const;
  double get_float(const std::string& key, double fallback) const;
  std::string get_string(const std::string& key, const std::string& fallback = "") const;
  bool get_bool(const std::string& key, bool fallback = false) const;
  uint64_t array_len(const std::string& key) const;
  // Numeric-array metadata as integers (empty when the key is absent / not an
  // array / not numeric).  Per-layer model geometry is carried in these.
  std::vector<int64_t> get_int_array(const std::string& key) const;
  // True when the key exists and is an array (any element type).
  bool is_array(const std::string& key) const;
};

Model read(const std::string& path);

// Same parser, but materialises the payload of the ARRAY values whose keys are
// listed in `capture_keys` (Value::strs / Value::raw).  Used by the tokenizer
// loader; every other array is still skipped.
Model read(const std::string& path, const std::vector<std::string>& capture_keys);

}  // namespace gguf
