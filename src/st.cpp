#include "engine/st.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "engine/json_parse.h"

namespace st {

namespace {

// #14: safetensors is JSON, so every number arrives as a double.  Shape entries and
// data offsets must be non-negative integers: a cast of -1 or 1.5 silently turned into
// 0xFFFFFFFFFFFFFFFF / a truncated value and the header was accepted.
uint64_t as_index(const jsonp::Value& v, const std::string& ctx) {
  if (!v.is_number()) throw std::runtime_error(ctx + ": expected a number");
  const double d = v.num;
  if (!(d >= 0.0)) throw std::runtime_error(ctx + ": negative value");
  if (d != std::floor(d)) throw std::runtime_error(ctx + ": non-integer value");
  if (d > 9007199254740992.0) throw std::runtime_error(ctx + ": value too large");
  return static_cast<uint64_t>(d);
}

// #15: overflow-checked multiplication for element counts / byte sizes.
uint64_t checked_mul(uint64_t a, uint64_t b, const std::string& ctx) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
    throw std::runtime_error(ctx + ": size overflow");
  return a * b;
}

std::string read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  auto n = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::string s(n, '\0');
  f.read(&s[0], static_cast<std::streamsize>(n));
  if (static_cast<size_t>(f.gcount()) != n) throw std::runtime_error("short read: " + path);
  return s;
}

uint64_t file_size(const std::string& path) {
  std::error_code ec;
  auto sz = std::filesystem::file_size(path, ec);
  if (ec) throw std::runtime_error("cannot stat " + path + ": " + ec.message());
  return static_cast<uint64_t>(sz);
}

const jsonp::Value& need(const jsonp::Value& obj, const char* key, const std::string& ctx) {
  const jsonp::Value* v = obj.find(key);
  if (!v) throw std::runtime_error(ctx + ": missing field '" + key + "'");
  return *v;
}

}  // namespace

uint64_t Tensor::n_elements() const {
  uint64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

uint64_t dtype_bytes(const std::string& dtype) {
  static const std::map<std::string, uint64_t> m = {
      {"F64", 8}, {"F32", 4}, {"F16", 2}, {"BF16", 2}, {"I64", 8}, {"I32", 4},
      {"I16", 2}, {"I8", 1}, {"U8", 1}, {"BOOL", 1}, {"F8_E4M3", 1}, {"F8_E5M2", 1},
  };
  auto it = m.find(dtype);
  if (it == m.end()) throw std::runtime_error("unsupported safetensors dtype " + dtype);
  return it->second;
}

std::map<std::string, Tensor> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  uint64_t hlen = 0;
  f.read(reinterpret_cast<char*>(&hlen), 8);
  if (!f || hlen > (100ull << 20)) throw std::runtime_error(path + ": bad header length");
  std::string hdr(hlen, '\0');
  f.read(&hdr[0], static_cast<std::streamsize>(hlen));
  if (!f) throw std::runtime_error(path + ": truncated header");

  jsonp::Value doc = jsonp::parse(hdr);
  if (!doc.is_object()) throw std::runtime_error(path + ": header is not an object");

  const uint64_t data_start = 8 + hlen;
  const uint64_t fsize = file_size(path);
  std::map<std::string, Tensor> out;
  for (const auto& [name, info] : doc.obj) {
    if (name == "__metadata__") continue;
    if (!info.is_object()) throw std::runtime_error(path + ": tensor entry is not an object");
    Tensor t;
    t.name = name;
    t.file = path;
    t.data_start = data_start;
    t.dtype = need(info, "dtype", name).str;
    const auto& shape = need(info, "shape", name);
    for (const auto& d : shape.arr)
      t.shape.push_back(as_index(d, name + ": shape"));
    const auto& offs = need(info, "data_offsets", name);
    if (offs.arr.size() != 2) throw std::runtime_error(name + ": bad data_offsets");
    // #14: offsets are uint64 in the format; require two non-negative integers.
    t.begin = as_index(offs.arr[0], name + ": data_offsets[0]");
    t.end = as_index(offs.arr[1], name + ": data_offsets[1]");

    // #15: every step here can overflow with a hostile header.
    uint64_t n_elems = 1;
    for (uint64_t d : t.shape) n_elems = checked_mul(n_elems, d, name + ": shape");
    const uint64_t expect = checked_mul(n_elems, dtype_bytes(t.dtype), name + ": size");
    if (t.end < t.begin) throw std::runtime_error(name + ": data_offsets end < begin");
    if (t.end - t.begin != expect)
      throw std::runtime_error(name + ": data size mismatch (shape x dtype)");
    if (t.end > fsize || data_start > fsize || data_start + t.end > fsize)
      throw std::runtime_error(name + ": data exceeds file size");
    out.emplace(name, std::move(t));
  }
  return out;
}

Shards load_dir(const std::string& dir) {
  namespace fs = std::filesystem;
  Shards out;
  const fs::path index = fs::path(dir) / "model.safetensors.index.json";
  if (fs::exists(index)) {
    jsonp::Value doc = jsonp::parse(read_text(index.string()));
    const jsonp::Value* wm = doc.find("weight_map");
    if (!wm || !wm->is_object())
      throw std::runtime_error(index.string() + ": missing weight_map");
    std::map<std::string, std::map<std::string, Tensor>> cache;
    for (const auto& [name, shard] : wm->obj) {
      const std::string sfile = (fs::path(dir) / shard.str).string();
      if (!cache.count(sfile)) {
        cache[sfile] = read_file(sfile);
        out.total_bytes += file_size(sfile);
      }
      auto it = cache[sfile].find(name);
      if (it == cache[sfile].end())
        throw std::runtime_error(name + ": not found in shard " + shard.str);
      out.tensors.emplace(name, it->second);
    }
    return out;
  }

  std::vector<std::string> files;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.path().extension() == ".safetensors") files.push_back(e.path().string());
  }
  if (files.empty()) throw std::runtime_error(dir + ": no .safetensors files");
  std::sort(files.begin(), files.end());
  for (const auto& f : files) {
    auto ts = read_file(f);
    out.total_bytes += file_size(f);
    for (auto& [k, v] : ts) out.tensors.emplace(k, std::move(v));
  }
  return out;
}

}  // namespace st
