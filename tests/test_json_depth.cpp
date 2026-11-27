// Regression: the JSON reader is recursive descent, so unbounded nesting used to
// overflow the native stack and kill the process.  Reproduced on the shipped v0.1.7
// binary: POST /tokenize with 200000 '[' -> connection reset, process gone (no 4xx).
// The parser now caps nesting depth (kMaxDepth) and reports a normal parse error,
// which serve maps to HTTP 400.
#include <cstdio>
#include <string>

#include "engine/json_parse.h"

namespace {

int fails = 0;
int checks = 0;

void expect_true(const char* name, bool ok) {
  ++checks;
  if (ok) {
    std::printf("  PASS %s\n", name);
  } else {
    std::printf("  FAIL %s\n", name);
    ++fails;
  }
}

std::string nested_arrays(int depth) {
  return std::string(depth, '[') + std::string(depth, ']');
}

std::string nested_objects(int depth) {
  std::string s;
  for (int i = 0; i < depth; ++i) s += "{\"a\":";
  s += "1";
  for (int i = 0; i < depth; ++i) s += "}";
  return s;
}

bool parses(const std::string& text) {
  try {
    jsonp::parse(text);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace

int main() {
  std::printf("json_parse depth regression\n");

  // Real request shapes must still parse.
  expect_true("flat object", parses(R"({"a":1,"b":[1,2,3],"c":{"d":true}})"));
  expect_true("chat body", parses(R"({"messages":[{"role":"user","content":"hi"}],)"
                                  R"("max_tokens":16,"temperature":0.6})"));
  expect_true("10-deep arrays", parses(nested_arrays(10)));
  expect_true("50-deep arrays", parses(nested_arrays(50)));
  expect_true("50-deep objects", parses(nested_objects(50)));

  // Just below/at the cap still parses; over the cap raises instead of recursing.
  expect_true("127-deep arrays", parses(nested_arrays(127)));
  expect_true("128-deep arrays (cap) accepted", parses(nested_arrays(128)));
  expect_true("129-deep arrays rejected", !parses(nested_arrays(129)));
  expect_true("300-deep arrays rejected", !parses(nested_arrays(300)));
  expect_true("300-deep objects rejected", !parses(nested_objects(300)));

  // The historical crash: huge depth must raise, never terminate the process.
  expect_true("20000-deep arrays rejected (no stack overflow)",
              !parses(nested_arrays(20000)));
  expect_true("200000-deep arrays rejected (no stack overflow)",
              !parses(nested_arrays(200000)));
  expect_true("1M-deep arrays rejected (no stack overflow)",
              !parses(nested_arrays(1000000)));

  // A rejected deep parse must not poison later parses.
  expect_true("parser usable after depth rejection", parses(R"({"ok":true})"));

  std::printf("JSON DEPTH: %d checks, %d failed\n", checks, fails);
  return fails == 0 ? 0 : 1;
}
