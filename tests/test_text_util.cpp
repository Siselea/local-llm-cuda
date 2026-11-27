// Strict UTF-8 regression for include/engine/text_util.h (review items 28-30):
// overlong forms, UTF-16 surrogate halves and codepoints above U+10FFFF must be
// rejected (previously eu8_next accepted them as codepoints).
#include <cstdio>
#include <string>

#include "engine/text_util.h"

namespace {
int failures = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++failures;
}
std::string bytes(std::initializer_list<int> v) {
  std::string s;
  for (int b : v) s.push_back(static_cast<char>(b));
  return s;
}
}  // namespace

int main() {
  using engine::text::is_valid_utf8;
  check(is_valid_utf8("hello"), "ASCII valid");
  check(is_valid_utf8("\xE4\xB8\xAD\xE6\x96\x87"), "CJK valid");
  check(is_valid_utf8("\xF0\x9F\x98\x80"), "emoji U+1F600 valid");
  check(!is_valid_utf8(bytes({0xC0, 0xAF})), "overlong C0 AF rejected");
  check(!is_valid_utf8(bytes({0xE0, 0x80, 0xAF})), "overlong E0 80 AF rejected");
  check(!is_valid_utf8(bytes({0xF0, 0x80, 0x80, 0xAF})), "overlong F0 80 80 AF rejected");
  check(!is_valid_utf8(bytes({0xED, 0xA0, 0x80})), "surrogate ED A0 80 rejected");
  check(!is_valid_utf8(bytes({0xF5, 0x80, 0x80, 0x80})), "> U+10FFFF rejected");
  check(!is_valid_utf8(bytes({0xC3})), "truncated sequence rejected");
  check(!is_valid_utf8(bytes({0x80})), "lone continuation rejected");
  {
    size_t i = 0;
    check(engine::text::utf8_next(bytes({0xC0, 0xAF}), i) == 0xFFFD, "utf8_next invalid -> U+FFFD");
    check(i == 1, "utf8_next advances one byte on invalid");
  }
  {
    const std::string decoded = engine::text::decode_utf8_lossy(bytes({0xED, 0xA0, 0x80}));
    check(decoded == "\xEF\xBF\xBD", "lossy decode replaces surrogate with U+FFFD");
  }
  {
    const std::string in = "a\xC0\xAF" "b";
    check(!is_valid_utf8(in), "valid+invalid mix rejected");
    const std::string out = engine::text::decode_utf8_lossy(in);
    check(out.find("\xEF\xBF\xBD") != std::string::npos, "lossy decode inserts U+FFFD");
  }
  std::printf(failures ? "TEXT_UTIL UTF8 FAILURES\n" : "TEXT_UTIL UTF8 GREEN\n");
  return failures ? 1 : 0;
}
