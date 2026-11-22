#include <cstdio>
#include <string>
#include "engine/json.h"
int main() {
  struct Case { const char* name; std::string in; int bad; };
  std::string lone(1, (char)0x80);
  std::string trunc2(1, (char)0xC3);
  std::string trunc3 = std::string(1, (char)0xE6) + std::string(1, (char)0x9D);
  std::string overlong = std::string(1, (char)0xC0) + std::string(1, (char)0xAF);
  std::string good = "hi \xE4\xB8\xAD\xE6\x96\x87 \xF0\x9F\x98\x80";
  Case cases[] = {
      {"lone-continuation", "a" + lone + "b", 1},
      {"truncated-2byte", "a" + trunc2, 1},
      {"truncated-3byte", trunc3 + "b", 1},
      {"overlong", overlong, 1},
      {"truncated-3byte-then-ascii", trunc3 + "b", 1},  // 'b' must survive after U+FFFD
      {"surrogate", std::string(1,(char)0xED)+std::string(1,(char)0xA0)+std::string(1,(char)0x80), 1},
      {"valid-passthrough", good, 0},
  };
  int fails = 0;
  for (const auto& c : cases) {
    json::Writer w;
    w.begin_obj(); w.key("s"); w.str(c.in); w.end_obj();
    int repl = 0;
    for (size_t i = 0; i + 2 < w.out.size(); ++i)
      if ((unsigned char)w.out[i] == 0xEF && (unsigned char)w.out[i+1] == 0xBF &&
          (unsigned char)w.out[i+2] == 0xBD) ++repl;
    bool pass = repl == c.bad;
    if (!pass) ++fails;
    std::printf("%s %-18s bytes=%zu replacement=%d (expect %d)\n",
                pass ? "PASS" : "FAIL", c.name, w.out.size(), repl, c.bad);
    std::printf("     %s\n", w.out.c_str());
  }
  std::printf(fails ? "JSON UTF8 FAIL\n" : "JSON UTF8 GREEN\n");
  return fails ? 1 : 0;
}
