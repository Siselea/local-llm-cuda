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

  // JSON strings must be valid UTF-8.  Model output can occasionally contain a
  // truncated character (a generation cut in the middle of a multi-byte sequence) or
  // raw byte tokens; pass good sequences through and replace anything invalid with
  // U+FFFD so a response is always parseable by strict clients.
  static std::string quote(const std::string& s) {
    static const char kReplacement[] = "\xEF\xBF\xBD";
    std::string r = "\"";
    for (size_t i = 0; i < s.size();) {
      const unsigned char c = static_cast<unsigned char>(s[i]);
      if (c < 0x80) {
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
        ++i;
        continue;
      }
      size_t len = 0;
      unsigned cp = 0;
      if ((c & 0xE0) == 0xC0) {
        len = 2;
        cp = c & 0x1F;
      } else if ((c & 0xF0) == 0xE0) {
        len = 3;
        cp = c & 0x0F;
      } else if ((c & 0xF8) == 0xF0) {
        len = 4;
        cp = c & 0x07;
      } else {
        r += kReplacement;
        ++i;
        continue;
      }
      const bool fits = i + len <= s.size();
      bool ok = fits;
      size_t consumed = 1;  // lead byte + continuation bytes that were valid
      if (ok) {
        for (size_t k = 1; k < len; ++k) {
          const unsigned char cc = static_cast<unsigned char>(s[i + k]);
          if ((cc & 0xC0) != 0x80) {
            ok = false;
            break;
          }
          ++consumed;
          cp = (cp << 6) | (cc & 0x3F);
        }
      }
      if (ok) {
        const bool overlong = (len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) ||
                              (len == 4 && cp < 0x10000);
        ok = !overlong && cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF);
      }
      if (!ok) {
        // Consume the whole malformed sequence (if it is complete) so one broken
        // character becomes exactly one U+FFFD, not one per continuation byte.
        r += kReplacement;
        i += fits ? consumed : 1;
        continue;
      }
      r.append(s, i, len);
      i += len;
    }
    r += '"';
    return r;
  }
};

}  // namespace json
