// Minimal JSON DOM + recursive-descent parser (for safetensors headers and config.json).
#pragma once

#include <map>
#include <string>
#include <vector>

namespace jsonp {

struct Value {
  enum class Type { Null, Bool, Num, Str, Arr, Obj } type = Type::Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;  // insertion order preserved

  bool is_object() const { return type == Type::Obj; }
  bool is_array() const { return type == Type::Arr; }
  bool is_number() const { return type == Type::Num; }
  bool is_string() const { return type == Type::Str; }

  const Value* find(const std::string& key) const;
  // typed getters with fallback
  double get_num(const std::string& key, double fallback) const;
  std::string get_str(const std::string& key, const std::string& fallback) const;
  bool get_bool(const std::string& key, bool fallback) const;
};

Value parse(const std::string& text);

}  // namespace jsonp
