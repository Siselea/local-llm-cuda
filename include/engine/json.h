// Minimal JSON writer (no deps). Deterministic key order, UTF-8 passthrough.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace json {

class Writer {
 public:
  std::string out;

  void begin_obj() { open('{'); }
  void end_obj() { close('}'); }
  void begin_arr() { open('['); }
  void end_arr() { close(']'); }

  void key(const std::string& k) {
    pre_value();
    out += quote(k);
    out += ": ";
    after_key_ = true;
  }

  void integer(int64_t v) { pre_value(); out += std::to_string(v); }
  void uinteger(uint64_t v) { pre_value(); out += std::to_string(v); }
  void number(double v) { pre_value(); out += fmt_num(v); }
  void boolean(bool v) { pre_value(); out += v ? "true" : "false"; }
  void str(const std::string& v) { pre_value(); out += quote(v); }
  void null() { pre_value(); out += "null"; }

 private:
  int indent_ = 0;
  bool after_key_ = false;
  std::vector<bool> first_;

  void open(char c) {
    pre_value();
    out += c;
    ++indent_;
    first_.push_back(true);
  }

  void close(char c) {
    bool empty = !first_.empty() && first_.back();
    first_.pop_back();
    --indent_;
    if (!empty) { out += '\n'; out.append(static_cast<size_t>(indent_) * 2, ' '); }
    out += c;
  }

  void pre_value() {
    if (after_key_) { after_key_ = false; return; }
    if (first_.empty()) return;                 // document root
    if (!first_.back()) out += ',';
    first_.back() = false;
    out += '\n';
    out.append(static_cast<size_t>(indent_) * 2, ' ');
  }

  static std::string fmt_num(double v) {
    char buf[64];
    if (v == static_cast<double>(static_cast<int64_t>(v)))
      std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
    else
      std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
  }

  static std::string quote(const std::string& s) {
    std::string r = "\"";
    for (unsigned char c : s) {
      switch (c) {
        case '"': r += "\\\""; break;
        case '\\': r += "\\\\"; break;
        case '\n': r += "\\n"; break;
        case '\r': r += "\\r"; break;
        case '\t': r += "\\t"; break;
        default:
          if (c < 0x20) {
            char b[8];
            std::snprintf(b, sizeof(b), "\\u%04x", c);
            r += b;
          } else {
            r += static_cast<char>(c);
          }
      }
    }
    r += '"';
    return r;
  }
};

}  // namespace json
