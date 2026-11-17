#include "engine/gguf.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace gguf {

namespace {

struct TypeEntry { uint32_t id; TypeInfo info; };

const TypeEntry kTypes[] = {
    {0, {"F32", 1, 4}},        {1, {"F16", 1, 2}},        {2, {"Q4_0", 32, 18}},
    {3, {"Q4_1", 32, 20}},     {6, {"Q5_0", 32, 22}},     {7, {"Q5_1", 32, 24}},
    {8, {"Q8_0", 32, 34}},     {9, {"Q8_1", 32, 36}},     {10, {"Q2_K", 256, 84}},
    {11, {"Q3_K", 256, 110}},  {12, {"Q4_K", 256, 144}},  {13, {"Q5_K", 256, 176}},
    {14, {"Q6_K", 256, 210}},  {15, {"Q8_K", 256, 292}},  {16, {"IQ2_XXS", 256, 66}},
    {17, {"IQ2_XS", 256, 74}}, {18, {"IQ3_XXS", 256, 98}},
    {19, {"IQ1_S", 256, 50}},  {20, {"IQ4_NL", 32, 18}},  {21, {"IQ3_S", 256, 110}},
    {22, {"IQ2_S", 256, 82}},  {23, {"IQ4_XS", 256, 136}},
    {24, {"I8", 1, 1}},        {25, {"I16", 1, 2}},       {26, {"I32", 1, 4}},
    {27, {"I64", 1, 8}},       {28, {"F64", 1, 8}},       {29, {"IQ1_M", 256, 56}},
    {30, {"BF16", 1, 2}},      {34, {"TQ1_0", 256, 54}},  {35, {"TQ2_0", 256, 66}},
};

class Reader {
 public:
  explicit Reader(const std::string& path) : in_(path, std::ios::binary) {
    if (!in_) throw std::runtime_error("cannot open " + path);
    in_.seekg(0, std::ios::end);
    size_ = static_cast<uint64_t>(in_.tellg());
    in_.seekg(0, std::ios::beg);
  }

  uint64_t size() const { return size_; }
  uint64_t tell() { return static_cast<uint64_t>(in_.tellg()); }
  void seek(uint64_t pos) { in_.seekg(static_cast<std::streamoff>(pos)); }

  void read_raw(void* dst, size_t n) {
    in_.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (!in_ || static_cast<size_t>(in_.gcount()) != n)
      throw std::runtime_error("unexpected end of file at offset " + std::to_string(tell()));
  }

  template <typename T>
  T get() {
    T v{};
    read_raw(&v, sizeof(T));
    return v;
  }

  std::string str() {
    uint64_t n = get<uint64_t>();
    if (n > size_) throw std::runtime_error("string length exceeds file size");
    std::string s(n, '\0');
    if (n) read_raw(&s[0], n);
    return s;
  }

  void skip(uint64_t n) { seek(tell() + n); }

 private:
  std::ifstream in_;
  uint64_t size_ = 0;
};

Value read_value(Reader& r, uint32_t vtype, bool capture);

Value read_array(Reader& r, bool capture) {
  Value v;
  v.kind = Value::Kind::ARRAY;
  v.array_elem_type = r.get<uint32_t>();
  v.array_len = r.get<uint64_t>();
  if (v.array_len > 100000000ull) throw std::runtime_error("implausible array length");
  // Without a capture request we only need the length for token arrays; still
  // skip the payload so parsing a 20 GB model stays cheap.
  auto take_bytes = [&](uint64_t bytes) {
    if (capture) {
      v.raw.resize(static_cast<size_t>(bytes));
      if (bytes) r.read_raw(v.raw.data(), static_cast<size_t>(bytes));
    } else {
      r.skip(bytes);
    }
  };
  switch (static_cast<ValueType>(v.array_elem_type)) {
    case ValueType::UINT8:
    case ValueType::INT8:
    case ValueType::BOOL:
      take_bytes(v.array_len);
      break;
    case ValueType::UINT16:
    case ValueType::INT16:
      take_bytes(v.array_len * 2);
      break;
    case ValueType::UINT32:
    case ValueType::INT32:
    case ValueType::FLOAT32:
      take_bytes(v.array_len * 4);
      break;
    case ValueType::UINT64:
    case ValueType::INT64:
    case ValueType::FLOAT64:
      take_bytes(v.array_len * 8);
      break;
    case ValueType::STRING:
      if (capture) {
        v.strs.reserve(static_cast<size_t>(v.array_len));
        for (uint64_t i = 0; i < v.array_len; ++i) v.strs.push_back(r.str());
      } else {
        for (uint64_t i = 0; i < v.array_len; ++i) r.str();
      }
      break;
    case ValueType::ARRAY:
      for (uint64_t i = 0; i < v.array_len; ++i) read_array(r, /*capture=*/false);
      break;
    default:
      throw std::runtime_error("bad array element type");
  }
  return v;
}

Value read_value(Reader& r, uint32_t vtype, bool capture) {
  Value v;
  switch (static_cast<ValueType>(vtype)) {
    case ValueType::UINT8: v.kind = Value::Kind::INT; v.i = r.get<uint8_t>(); break;
    case ValueType::INT8: v.kind = Value::Kind::INT; v.i = r.get<int8_t>(); break;
    case ValueType::UINT16: v.kind = Value::Kind::INT; v.i = r.get<uint16_t>(); break;
    case ValueType::INT16: v.kind = Value::Kind::INT; v.i = r.get<int16_t>(); break;
    case ValueType::UINT32: v.kind = Value::Kind::INT; v.i = r.get<uint32_t>(); break;
    case ValueType::INT32: v.kind = Value::Kind::INT; v.i = r.get<int32_t>(); break;
    case ValueType::FLOAT32: v.kind = Value::Kind::FLOAT; v.f = r.get<float>(); break;
    case ValueType::BOOL: v.kind = Value::Kind::BOOL; v.b = r.get<uint8_t>() != 0; break;
    case ValueType::STRING: v.kind = Value::Kind::STRING; v.s = r.str(); break;
    case ValueType::ARRAY: return read_array(r, capture);
    case ValueType::UINT64: v.kind = Value::Kind::INT; v.i = static_cast<int64_t>(r.get<uint64_t>()); break;
    case ValueType::INT64: v.kind = Value::Kind::INT; v.i = r.get<int64_t>(); break;
    case ValueType::FLOAT64: v.kind = Value::Kind::FLOAT; v.f = r.get<double>(); break;
    default: throw std::runtime_error("unknown gguf value type " + std::to_string(vtype));
  }
  return v;
}

}  // namespace

const TypeInfo* type_info(uint32_t type_id) {
  for (const auto& e : kTypes)
    if (e.id == type_id) return &e.info;
  return nullptr;
}

int64_t Model::get_int(const std::string& key, int64_t fallback) const {
  auto it = kv.find(key);
  if (it == kv.end()) return fallback;
  if (it->second.kind == Value::Kind::INT) return it->second.i;
  if (it->second.kind == Value::Kind::BOOL) return it->second.b ? 1 : 0;
  return fallback;
}

double Model::get_float(const std::string& key, double fallback) const {
  auto it = kv.find(key);
  if (it == kv.end()) return fallback;
  if (it->second.kind == Value::Kind::FLOAT) return it->second.f;
  if (it->second.kind == Value::Kind::INT) return static_cast<double>(it->second.i);
  return fallback;
}

std::string Model::get_string(const std::string& key, const std::string& fallback) const {
  auto it = kv.find(key);
  if (it == kv.end() || it->second.kind != Value::Kind::STRING) return fallback;
  return it->second.s;
}

bool Model::get_bool(const std::string& key, bool fallback) const {
  auto it = kv.find(key);
  if (it == kv.end()) return fallback;
  if (it->second.kind == Value::Kind::BOOL) return it->second.b;
  return fallback;
}

uint64_t Model::array_len(const std::string& key) const {
  auto it = kv.find(key);
  if (it == kv.end() || it->second.kind != Value::Kind::ARRAY) return 0;
  return it->second.array_len;
}

Model read(const std::string& path) {
  static const std::vector<std::string> kNoCapture;
  return read(path, kNoCapture);
}

Model read(const std::string& path, const std::vector<std::string>& capture_keys) {
  Reader r(path);
  Model m;
  m.path = path;
  m.file_size = r.size();

  uint32_t magic = r.get<uint32_t>();
  if (magic != MAGIC) throw std::runtime_error("bad GGUF magic");
  m.version = r.get<uint32_t>();
  if (m.version != 2 && m.version != 3)
    throw std::runtime_error("unsupported GGUF version " + std::to_string(m.version));
  m.n_tensors = r.get<uint64_t>();
  m.n_kv = r.get<uint64_t>();

  // Sorted capture list -> binary search, no allocation per metadata key.
  std::vector<std::string> capture = capture_keys;
  std::sort(capture.begin(), capture.end());
  for (uint64_t i = 0; i < m.n_kv; ++i) {
    std::string key = r.str();
    uint32_t vtype = r.get<uint32_t>();
    const bool want_capture =
        !capture.empty() && std::binary_search(capture.begin(), capture.end(), key);
    m.kv.emplace(std::move(key), read_value(r, vtype, want_capture));
  }

  m.tensors.reserve(static_cast<size_t>(m.n_tensors));
  for (uint64_t i = 0; i < m.n_tensors; ++i) {
    TensorInfo t;
    t.name = r.str();
    uint32_t n_dims = r.get<uint32_t>();
    if (n_dims > 8) throw std::runtime_error("implausible n_dims for " + t.name);
    std::vector<uint64_t> dims(n_dims);
    for (auto& d : dims) d = r.get<uint64_t>();
    t.type_id = r.get<uint32_t>();
    t.offset = r.get<uint64_t>();

    const TypeInfo* ti = type_info(t.type_id);
    if (!ti) throw std::runtime_error("tensor " + t.name + ": unsupported type id " +
                                      std::to_string(t.type_id));
    t.type_name = ti->name;
    t.shape.assign(dims.rbegin(), dims.rend());
    t.n_elements = 1;
    for (auto d : dims) t.n_elements *= d;
    if (ti->block > 1 && t.n_elements % static_cast<uint64_t>(ti->block) != 0)
      throw std::runtime_error("tensor " + t.name + ": element count not divisible by block");
    t.nbytes = t.n_elements / static_cast<uint64_t>(ti->block) * static_cast<uint64_t>(ti->bytes);
    m.tensors.push_back(std::move(t));
  }

  int64_t align = m.get_int("general.alignment", 32);
  m.alignment = static_cast<uint32_t>(align > 0 ? align : 32);
  uint64_t pos = r.tell();
  m.data_offset = (pos + m.alignment - 1) / m.alignment * m.alignment;
  return m;
}

}  // namespace gguf
