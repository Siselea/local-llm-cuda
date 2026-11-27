#include "engine/json_parse.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace jsonp {

namespace {

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}

  Value parse_document() {
    skip_ws();
    Value v = parse_value();
    skip_ws();
    if (pos_ != s_.size()) throw std::runtime_error("trailing characters in JSON");
    return v;
  }

 private:
  // Bounded nesting: a recursive-descent parser must cap recursion depth, otherwise a
  // single hostile body such as "[[[[...]]]]" overflows the native stack and kills the
  // process (reproduced: POST /tokenize with 200k '[' -> server dies, no 4xx).  128 is
  // far beyond any real request shape (OpenAI-style bodies nest < 10 deep) while leaving
  // plenty of stack for the 32 MiB body limit.
  static constexpr int kMaxDepth = 128;

  const std::string& s_;
  size_t pos_ = 0;
  int depth_ = 0;

  [[noreturn]] void fail(const char* what) {
    throw std::runtime_error(std::string("JSON parse error: ") + what + " at offset " +
                             std::to_string(pos_));
  }

  void skip_ws() {
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
      else break;
    }
  }

  char peek() {
    if (pos_ >= s_.size()) fail("unexpected end");
    return s_[pos_];
  }

  char take() { return s_[pos_++]; }

  void expect(char c) {
    if (pos_ >= s_.size() || s_[pos_] != c) fail("unexpected character");
    ++pos_;
  }

  bool match_word(const char* w) {
    size_t n = std::strlen(w);
    if (s_.compare(pos_, n, w) != 0) return false;
    pos_ += n;
    return true;
  }

  Value parse_value() {
    skip_ws();
    char c = peek();
    if (c == '{' || c == '[') {
      if (depth_ >= kMaxDepth) fail("nesting too deep");
      ++depth_;
      Value v = (c == '{') ? parse_object() : parse_array();
      --depth_;
      return v;
    }
    switch (c) {
      case '"': {
        Value v;
        v.type = Value::Type::Str;
        v.str = parse_string();
        return v;
      }
      case 't':
        if (!match_word("true")) fail("bad literal");
        {
          Value v;
          v.type = Value::Type::Bool;
          v.b = true;
          return v;
        }
      case 'f':
        if (!match_word("false")) fail("bad literal");
        {
          Value v;
          v.type = Value::Type::Bool;
          v.b = false;
          return v;
        }
      case 'n':
        if (!match_word("null")) fail("bad literal");
        return Value{};
      default: return parse_number();
    }
  }

  Value parse_object() {
    Value v;
    v.type = Value::Type::Obj;
    expect('{');
    skip_ws();
    if (peek() == '}') {
      ++pos_;
      return v;
    }
    while (true) {
      skip_ws();
      if (peek() != '"') fail("object key must be a string");
      std::string key = parse_string();
      skip_ws();
      expect(':');
      Value val = parse_value();
      v.obj.emplace_back(std::move(key), std::move(val));
      skip_ws();
      char c = take();
      if (c == '}') break;
      if (c != ',') fail("expected , or }");
    }
    return v;
  }

  Value parse_array() {
    Value v;
    v.type = Value::Type::Arr;
    expect('[');
    skip_ws();
    if (peek() == ']') {
      ++pos_;
      return v;
    }
    while (true) {
      v.arr.push_back(parse_value());
      skip_ws();
      char c = take();
      if (c == ']') break;
      if (c != ',') fail("expected , or ]");
    }
    return v;
  }

  static void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  unsigned parse_hex4() {
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = take();
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
      else fail("bad \\u escape");
    }
    return v;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= s_.size()) fail("unterminated string");
      unsigned char c = static_cast<unsigned char>(take());
      if (c == '"') break;
      if (c != '\\') {
        out += static_cast<char>(c);
        continue;
      }
      char e = take();
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned cp = parse_hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (pos_ + 1 < s_.size() && s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
              pos_ += 2;
              unsigned lo = parse_hex4();
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            }
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }

  Value parse_number() {
    size_t start = pos_;
    if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
        ++pos_;
      else
        break;
    }
    if (pos_ == start) fail("bad number");
    Value v;
    v.type = Value::Type::Num;
    v.num = std::strtod(s_.substr(start, pos_ - start).c_str(), nullptr);
    return v;
  }
};

}  // namespace

const Value* Value::find(const std::string& key) const {
  for (const auto& [k, v] : obj)
    if (k == key) return &v;
  return nullptr;
}

double Value::get_num(const std::string& key, double fallback) const {
  const Value* v = find(key);
  if (!v) return fallback;
  if (v->type == Type::Num) return v->num;
  if (v->type == Type::Bool) return v->b ? 1.0 : 0.0;
  return fallback;
}

std::string Value::get_str(const std::string& key, const std::string& fallback) const {
  const Value* v = find(key);
  if (!v || v->type != Type::Str) return fallback;
  return v->str;
}

bool Value::get_bool(const std::string& key, bool fallback) const {
  const Value* v = find(key);
  if (!v || v->type != Type::Bool) return fallback;
  return v->b;
}

Value parse(const std::string& text) { return Parser(text).parse_document(); }

}  // namespace jsonp
