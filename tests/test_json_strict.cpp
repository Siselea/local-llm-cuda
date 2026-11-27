// Strictness regression for the JSON reader (external review v0.1.7 items #1-#4).
//
// The parser is reachable from untrusted input: tools/serve passes every HTTP
// request body through jsonp::parse() and maps a parse error to HTTP 400.  v0.1.7
// accepted several non-JSON constructs:
//   #1 invalid numbers:  01, -01, 1., +1, ".", "-", "1e", "1.e5", "e5", 1e999
//   #2 raw control chars inside strings (U+0000..U+001F, incl. tab/newline)
//   #3 invalid surrogate escapes: \ud800 alone, \udc00 alone, \ud800\ud800,
//      and a high surrogate followed by ordinary text
//   #4 reported as a possible out-of-bounds read on truncated escapes: the
//      reader never actually indexed past s_[size()] (std::string::operator[]
//      at size() is defined), and all truncated forms throw normally; take() is
//      now bounds-checked as defence in depth and the test fuzzes every prefix.
//
// The strict parser must keep accepting real request shapes (chat bodies, tool
// arguments, unicode content) — the VALID cases below are part of the contract.
#include <cstdio>
#include <string>
#include <vector>

#include "engine/json_parse.h"

namespace {

int fails = 0;
int checks = 0;

void check(const char* name, bool ok) {
  ++checks;
  if (!ok) {
    std::printf("  FAIL %s\n", name);
    ++fails;
  }
}

bool parses(const std::string& text) {
  try {
    jsonp::parse(text);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

void expect_ok(const char* name, const std::string& text) {
  const bool ok = parses(text);
  check(name, ok);
  if (!ok) std::printf("    expected VALID: %s\n", text.c_str());
}

void expect_reject(const char* name, const std::string& text) {
  const bool rejected = !parses(text);
  check(name, rejected);
  if (!rejected) std::printf("    expected REJECT: %s\n", text.c_str());
}

// Validates a parsed document round-trip for a few hand-picked payloads.
void expect_number(const char* name, const std::string& text, double want) {
  try {
    const jsonp::Value v = jsonp::parse(text);
    check(name, v.is_number() && v.num == want);
  } catch (const std::exception&) {
    check(name, false);
    std::printf("    expected number %s, got parse error\n", text.c_str());
  }
}

void expect_bytes(const char* name, const std::string& text, const std::string& want) {
  try {
    const jsonp::Value v = jsonp::parse(text);
    check(name, v.is_string() && v.str == want);
  } catch (const std::exception& e) {
    check(name, false);
    std::printf("    expected string %s, got %s\n", text.c_str(), e.what());
  }
}

// #4: no truncated input may crash the process.  Every prefix of strings that
// contain escapes/"u" runs is parsed; complete documents may legitimately be valid
// only at full length, everything shorter must throw, never fault.
void fuzz_truncated() {
  const std::vector<std::string> seeds = {
      R"("abc")",          // trailing backslash
      R"("ab\u12")",       // truncated \u
      R"("ab\u123")",
      R"("ab\uD800")",     // truncated surrogate pair
      R"("ab\uD800\u")",
      R"("a\uD800\udc00")",
      R"([1,2,{"a":"b\u00e9"}])",
      R"({"s":"a\",b\":c\u001f"})",
  };
  int parsed = 0;
  for (const auto& seed : seeds) {
    for (size_t len = 0; len <= seed.size(); ++len) {
      const std::string prefix = seed.substr(0, len);
      try {
        jsonp::parse(prefix);
        ++parsed;
      } catch (const std::exception&) {
      }
    }
  }
  (void)parsed;  // only reaching here without a fault matters
  check("truncated-escape prefixes never fault", true);
}

// Deterministic ASCII+escape fuzz: malformed documents must raise, never crash.
void fuzz_random() {
  const char alphabet[] = "{}[]\":,0123456789.eE+-\\/*u \tnan";
  unsigned state = 0x12345678u;
  for (int iter = 0; iter < 5000; ++iter) {
    state = state * 1664525u + 1013904223u;
    const size_t len = (state >> 16) % 40;
    std::string doc;
    doc.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      state = state * 1664525u + 1013904223u;
      doc += alphabet[(state >> 8) % (sizeof(alphabet) - 1)];
    }
    try {
      jsonp::parse(doc);
    } catch (const std::exception&) {
    }
  }
  check("random malformed documents never fault", true);
}

}  // namespace

int main() {
  std::printf("json strictness (RFC 8259) regression\n");

  // ---- real request shapes stay valid -------------------------------------
  expect_ok("flat object", R"({"a":1,"b":[1,2,3],"c":{"d":true}})");
  expect_ok("chat body", R"({"messages":[{"role":"user","content":"hi"}],)"
                         R"("max_tokens":16,"temperature":0.6})");
  expect_ok("unicode content", R"({"content":"你好，世界 😀"})");
  expect_ok("escaped unicode", R"({"s":"\u4f60\u597d"})");
  expect_ok("escaped controls", R"({"s":"a\u0000b\u001fc"})");
  expect_ok("surrogate pair", R"({"s":"\ud83d\ude00"})");
  expect_ok("empty containers", R"({"a":[],"b":{},"c":"","d":0,"e":-0})");
  expect_ok("exponents", R"([0, -0, 1E+2, 1e-2, 2.5e10, -1.25e-3])");
  expect_ok("max-ish finite", R"([1.7976931348623157e308, 5e-324])");

  // ---- #1 numbers --------------------------------------------------------
  expect_number("int", "123", 123.0);
  expect_number("neg frac", "-1.25e-3", -0.00125);
  expect_number("zero", "0", 0.0);
  expect_reject("leading zero", "01");
  expect_reject("leading zero negative", "-01");
  expect_reject("trailing dot", "1.");
  expect_reject("leading plus", "+1");
  expect_reject("dot only", ".");
  expect_reject("plus dot", "+.");
  expect_reject("minus only", "-");
  expect_reject("plus only", "+");
  expect_reject("nan literal", "nan");
  expect_reject("NaN literal", "NaN");
  expect_reject("inf literal", "inf");
  expect_reject("Infinity literal", "Infinity");
  expect_reject("-inf literal", "-inf");
  expect_reject("overflow to inf", "1e999");
  expect_reject("overflow to -inf", "-1e999");
  expect_reject("exponent without digits", "1e");
  expect_reject("fraction without digits", "1.e5");
  expect_reject("bare exponent", "e5");
  expect_reject("hex number", "0x10");
  expect_reject("number trailing garbage", "1x");
  expect_reject("number in array invalid", "[1x]");
  expect_reject("double minus", "--1");
  expect_reject("plus exp no number", "e+1");
  expect_reject("space inside number", "1 2");

  // ---- #2 raw control characters in strings ------------------------------
  {
    std::string nul("a\0b", 3);
    expect_reject("raw NUL in string", "\"" + nul + "\"");
    expect_reject("raw 0x01 in string", std::string("\"a\x01" "b\""));
    expect_reject("raw 0x1f in string", std::string("\"a\x1f" "b\""));
    expect_reject("raw tab in string", std::string("\"a\tb\""));
    expect_reject("raw newline in string", std::string("\"a\nb\""));
    expect_reject("raw CR in string", std::string("\"a\rb\""));
    expect_ok("escaped NUL in string", R"("a\u0000b")");
    expect_ok("DEL (0x7f) raw is allowed", std::string("\"a\x7f" "b\""));
  }

  // ---- #3 surrogate escapes ----------------------------------------------
  expect_reject("lone high surrogate", R"("\ud800")");
  expect_reject("lone low surrogate", R"("\udc00")");
  expect_reject("high+high surrogate", R"("\ud800\ud800")");
  expect_reject("high surrogate then text", R"("\ud800x")");
  expect_reject("high surrogate then escape", R"("\ud800\u0041")");
  expect_reject("low surrogate after text", R"("\udc00ab")");
  expect_bytes("valid surrogate pair", R"("\ud83d\ude00")",
               std::string("\xF0\x9F\x98\x80"));
  expect_bytes("valid BMP escape", R"("\u4f60")", std::string("\xE4\xBD\xA0"));

  // ---- #4 truncated escapes must raise cleanly ---------------------------
  expect_reject("trailing backslash", "\"abc\\");
  expect_reject("truncated u0", "\"ab\\u");
  expect_reject("truncated u1", "\"ab\\u1");
  expect_reject("truncated u2", "\"ab\\u12");
  expect_reject("truncated u3", "\"ab\\u123");
  expect_reject("truncated pair", "\"ab\\ud800\\u");
  expect_reject("truncated pair hex", "\"ab\\ud800\\udc");
  expect_reject("unterminated", "\"abc");
  expect_reject("escape then EOF", "\"ab\\");

  // ---- structure ---------------------------------------------------------
  expect_reject("empty document", "");
  expect_reject("trailing comma array", "[1,]");
  expect_reject("trailing comma object", R"({"a":1,})");
  expect_reject("bad literal", "tru");
  expect_reject("two documents", "{} {}");
  expect_ok("null document", "null");

  fuzz_truncated();
  fuzz_random();

  std::printf("JSON STRICT: %d checks, %d failed\n", checks, fails);
  return fails == 0 ? 0 : 1;
}
