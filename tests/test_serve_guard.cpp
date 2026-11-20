// Unit test for engine::serve::RepeatGuard (no GPU / model needed).
//
//   ziglang c++ -std=c++17 -O2 -Iinclude tests/test_serve_guard.cpp -o build/test_serve_guard.exe
//   build/test_serve_guard.exe
//
// Covers the two real 35B loops observed in the serve benchmark (whole-sentence
// loop and character-level loop) plus normal Chinese/markdown answers that must
// NOT be stopped.
#include <cstdio>
#include <string>

#include "engine/serve_guard.h"

namespace {

int g_fail = 0;

void check(bool ok, const char* name) {
  std::printf("  %-58s %s\n", name, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
}

// Feed text to the guard in small chunks (like streaming deltas).
bool fired_on(const std::string& text, int level, bool reasoning = false,
              size_t chunk = 8) {
  engine::serve::RepeatGuard g;
  g.configure(level);
  for (size_t i = 0; i < text.size(); i += chunk) {
    const size_t n = std::min(chunk, text.size() - i);
    if (g.update(text.substr(i, n), reasoning)) return true;
  }
  return false;
}

}  // namespace

int main() {
  std::printf("== repeat guard unit test ==\n");

  // 1) the real sentence-level loop: "老城的秘密，是老人。" x 28
  {
    std::string loop;
    for (int i = 0; i < 28; ++i) loop += "老城的秘密，是老人。";
    check(fired_on(loop, 1), "sentence loop caught at level 1");
    check(fired_on(loop, 2), "sentence loop caught at level 2");
  }

  // 2) the real character-level loop: "--134--134--..." (no sentence terminator)
  {
    std::string loop;
    for (int i = 0; i < 40; ++i) loop += "—134";
    check(fired_on(loop, 2), "char-level loop caught at level 2");
    check(!fired_on(loop, 1), "char-level loop NOT caught at level 1 (expected)");
  }

  // 3) the other real char-level loop: "—你——你——你—"
  {
    std::string loop;
    for (int i = 0; i < 30; ++i) loop += "—你";
    check(fired_on(loop, 2), "char loop '--你' caught at level 2");
  }

  // 4) variant sentence loop: same phrase with tiny variations, 8 times
  {
    std::string loop;
    for (int i = 0; i < 8; ++i)
      loop += "岁月的秘密，是老人" + std::string(1, static_cast<char>('A' + i)) + "。";
    check(fired_on(loop, 2), "variant sentence loop caught at level 2 (normalised)");
  }

  // 4b) legitimate Python code (repeated `print(is_prime(...))` lines) must NOT fire
  {
    const std::string code =
        "# Python prime check\n\n```python\ndef is_prime(n: int) -> bool:\n"
        "    if n < 2:\n        return False\n    if n == 2:\n        return True\n"
        "    if n % 2 == 0:\n        return False\n"
        "    for i in range(3, int(n ** 0.5) + 1, 2):\n"
        "        if n % i == 0:\n            return False\n    return True\n```\n\n"
        "print(is_prime(17))    # True\n"
        "print(is_prime(18))    # False\n"
        "print(is_prime(19))    # True\n"
        "print(is_prime(20))    # False\n"
        "print(is_prime(23))    # True\n"
        "print(is_prime(25))    # False\n";
    check(!fired_on(code, 2), "legitimate Python code not stopped");
  }

  // 4b2) interleaved 5-sentence cycle repeated only 3x (early-stop requirement:
  //      the user should see at most 3 repeats, not 8+)
  {
    const char* cycle[] = {
        "信的内容不长，只有寥寥数语，却让他读出了千百种滋味。",
        "他读到了父亲的思念，读到了时光的流转，读到了生命深处的秘密。",
        "“原来，”他说，“原来我一直没有真正告别。”",
        "苏婉轻轻握住他的手，“时间会给你答案。”",
        "林默拿起那封信，拆开了它。",
    };
    std::string loop;
    for (int rep = 0; rep < 3; ++rep)
      for (const char* line : cycle) loop += std::string(line) + "\n";
    check(fired_on(loop, 2), "3-cycle refrain caught early (duplicate-3x)");
  }

  // 4b3) a normal 1000-char Chinese story must NOT fire
  {
    std::string story =
        "# 雨夜归途\n\n深秋的雨总是来得猝不及防。\n\n"
        "林婉站在写字楼的落地窗前，看着窗外灰蒙蒙的天空，雨水顺着玻璃蜿蜒而下，"
        "像极了她此刻纷乱的思绪。今天是她辞职的第三天，也是她决定重新开始的第四天。\n\n"
        "手机震动，是母亲发来的消息：“明天回来吃饭吗？”\n\n她回复：“回。”\n\n"
        "林婉今年三十二岁，在上海一家广告公司做创意总监。外人看来她事业有成、生活优渥，"
        "但只有她自己知道，每一个加班的深夜，她都在怀疑这份工作是否值得。\n\n";
    const char* days[] = {
        "周一早晨，她把工牌放在前台，最后一次刷开那扇玻璃门。",
        "午休时同事们拉她去楼下的面馆，谁都没有提离职的事。",
        "下午她把电脑里的文件归档，命名为一个新的开始。",
        "傍晚的电梯里，她遇见了刚入职的实习生，年轻得刺眼。",
        "走出大楼时雨停了，天边露出一线橘红色的晚霞。",
        "地铁上她翻看手机相册，发现三年里几乎全是加班的夜景。",
        "到家后母亲打来电话，问她晚饭吃了没有，声音温和。",
        "她站在阳台上看城市亮起灯火，第一次觉得未来并不慌张。",
    };
    for (const char* day : days) story += std::string(day) + "\n\n";
    check(!fired_on(story, 2), "normal 1000-char Chinese story not stopped");
  }

  // 4c) interleaved 5-sentence cycle repeated 12x (real 35B loop from the benchmark)
  {
    const char* cycle[] = {
        "信的内容不长，只有寥寥数语，却让他读出了千百种滋味。",
        "他读到了父亲的思念，读到了时光的流转，读到了生命深处的秘密。",
        "“原来，”他说，“原来我一直没有真正告别。”",
        "苏婉轻轻握住他的手，“时间会给你答案。”",
        "林默拿起那封信，拆开了它。",
    };
    std::string loop;
    for (int rep = 0; rep < 12; ++rep)
      for (const char* line : cycle) loop += std::string(line) + "\n";
    check(fired_on(loop, 2), "interleaved 5-sentence cycle caught at level 2");
  }

  // 5) normal Chinese story answer must NOT fire
  {
    const std::string story =
        "# 雨夜的最后一通电话\n\n"
        "深夜十一点，城市被一场突如其来的暴雨笼罩。林默坐在窗前，手里握着那部老式"
        "翻盖手机，屏幕上的时间一分一秒地跳动。\n\n"
        "这是她父亲去世后的第七天，也是她最后一次试图拨通那个熟悉的号码。\n\n"
        "“嘟——嘟——”\n\n"
        "电话接通的瞬间，林默的心猛地揪紧。\n\n"
        "“喂？”\n\n"
        "“爸，是我。”她的声音有些颤抖。\n\n"
        "“默儿啊，怎么这么晚还没睡？”父亲的声音依然温和，仿佛从未离开过。\n\n"
        "“爸，我想你了。”\n\n"
        "电话那头沉默了一会儿，然后传来一声轻叹：“傻孩子，爸一直都在。”\n\n"
        "林默的眼泪终于落了下来。她想起小时候，父亲总是这样，在她最需要的时候"
        "出现，用最平常的话，给她最踏实的安慰。\n\n"
        "雨还在下，但窗外的城市似乎没有那么冷了。\n";
    check(!fired_on(story, 2), "normal Chinese story not stopped");
  }

  // 6) normal thinking text with repeated markdown headers must NOT fire
  {
    const std::string think =
        "Thinking Process:\n\n"
        "1.  **Analyze the Request:**\n"
        "    *   Topic: Hangzhou (杭州).\n"
        "    *   Constraint: exactly three sentences.\n\n"
        "2.  **Draft the Answer:**\n"
        "    *   Sentence 1: location + reputation.\n"
        "    *   Sentence 2: landmark + culture.\n"
        "    *   Sentence 3: modern economy.\n\n"
        "3.  **Refine the Draft:**\n"
        "    *   Check the sentence count is exactly three.\n"
        "    *   Check the language is Chinese.\n"
        "    *   Check the tone is elegant and warm.\n\n"
        "4.  **Final Check:**\n"
        "    *   Three sentences.\n"
        "    *   Chinese.\n"
        "    *   Ready to answer.\n";
    check(!fired_on(think, 2, /*reasoning=*/true),
          "thinking text with repeated headers not stopped");
  }

  // 7) long normal answer with markdown list must NOT fire
  {
    const char* points[] = {
        "第1点：西湖的四季景色各不相同，春天苏堤春晓，夏天曲院风荷。",
        "第2点：灵隐寺始建于东晋，是江南著名的佛教古刹，香火鼎盛。",
        "第3点：龙井茶产于西湖群山之间，以明前茶最为珍贵。",
        "第4点：京杭大运河穿城而过，承载了千年的漕运历史。",
        "第5点：南宋御街保留着古都的市井风貌，如今是步行商业街。",
        "第6点：钱塘江大潮天下闻名，每年中秋前后吸引无数观潮者。",
    };
    std::string answer = "杭州是浙江省省会，素有“人间天堂”的美誉。\n\n";
    for (const char* line : points) answer += std::string(line) + "\n";
    check(!fired_on(answer, 2), "normal varied markdown list not stopped");
  }

  // 8) level 0 disables the guard entirely
  {
    std::string loop;
    for (int i = 0; i < 28; ++i) loop += "老城的秘密，是老人。";
    check(!fired_on(loop, 0), "level 0 disables the guard");
  }

  // 10) CJK sentence repeated with only digit changes -> duplicate-3x fires
  {
    std::string loop;
    for (int i = 1; i <= 4; ++i)
      loop += "第" + std::to_string(i) + "次他推开门，屋里还是那股旧木头的味道。\n";
    check(fired_on(loop, 2), "digit-variant sentence x4 caught (duplicate-3x)");
  }

  // 11) period-2 refrain (alternating sentences) caught at level 2
  {
    std::string loop;
    for (int i = 0; i < 5; ++i)
      loop += "雨还在下。\n他把窗关上。\n";
    check(fired_on(loop, 2), "period-2 refrain caught");
  }

  // 12) normal English prose must NOT fire
  {
    const std::string prose =
        "The morning train was late again, so she walked along the river instead. "
        "The water was high after the storm and the air smelled of wet stone. "
        "She thought about the letter she had never answered, and about the years "
        "that had passed since then. A man was fishing near the bridge; he nodded "
        "as she passed, and she nodded back without stopping. By the time she "
        "reached the office the rain had started again, soft and steady.\n";
    check(!fired_on(prose, 2), "normal English prose not stopped");
  }

  // 13) normal Chinese dialogue with short repeated replies must NOT fire
  {
    const char* lines[] = {
        "“今天还顺利吗？”母亲在电话那头问。",
        "“还行。”她轻声回答。",
        "“工作别太拼了，身体要紧。”母亲又说。",
        "“我知道。”她望着窗外的雨。",
        "“周末回家吃饭吧，你爸买了你爱吃的鱼。”",
        "“好，我周六回去。”她终于笑了一下。",
        "“路上小心，到了给我发消息。”",
        "“嗯，妈，你也早点休息。”",
    };
    std::string dialog;
    for (const char* line : lines) dialog += std::string(line) + "\n";
    check(!fired_on(dialog, 2), "normal Chinese dialogue not stopped");
  }

  // 14) thinking text with the same sentence 3x non-consecutively must NOT fire
  {
    std::string think;
    for (int i = 0; i < 4; ++i) {
      think += "I should be helpful.\n";
      think += "Now I will draft sentence number " + std::to_string(i) + " carefully.\n";
    }
    check(!fired_on(think, 2, /*reasoning=*/true),
          "thinking with repeated line not stopped by duplicate rule");
  }

  // 14b) Python code with short Chinese labels/comments must NOT fire
  {
    const std::string code =
        "```python\n"
        "def is_prime(n: int) -> bool:\n"
        "    if n < 2:\n        return False\n"
        "    if n == 2:\n        return True\n"
        "    if n % 2 == 0:\n        return False\n"
        "    return True\n```\n\n"
        "print(f\"1 是素数: {is_prime(1)}\")\n"
        "print(f\"2 是素数: {is_prime(2)}\")\n"
        "print(f\"3 是素数: {is_prime(3)}\")\n"
        "print(f\"4 是素数: {is_prime(4)}\")\n";
    check(!fired_on(code, 2), "Python code with Chinese labels not stopped");
  }

  // 15) normal long markdown table must NOT fire
  {
    std::string table = "| 项目 | 数值 | 说明 |\n|---|---|---|\n";
    const char* rows[] = {
        "| 上下文 | 4096 | 最大 token 数 |", "| KV 精度 | q8_0 | 量化缓存 |",
        "| 专家层 | 15/40 | GPU 驻留 |", "| 线程 | 9 | CPU 专家并行 |",
        "| 采样 | 0.6/20/0.95 | 推荐值 |", "| 首字 | 0.6s | 实测 |",
    };
    for (const char* r : rows) table += std::string(r) + "\n";
    check(!fired_on(table, 2), "normal markdown table not stopped");
  }

  // 15b) >=12-byte unit repeated 4x fires (stop before the user sees 5 lines)
  {
    std::string loop;
    for (int i = 0; i < 4; ++i) loop += "他找到了自己。\n";
    check(fired_on(loop, 2), "12-byte unit x4 caught (early stop)");
    std::string dashes(48, '=');  // decorative line, not a loop
    check(!fired_on(dashes, 2), "long decorative line not stopped");
  }

  // 16) duplicate-3x requires 3 occurrences: interleaved with unique sentences is
  //     caught, but only two occurrences are not
  {
    const char* loopline = "墙上的老钟停在三点十七分，像在等一个不会回来的人。";
    std::string two = std::string(loopline) + "\n" +
                      "窗外的风把落叶吹进屋里，落在地板的缝隙上。\n" +
                      std::string(loopline) + "\n";
    check(!fired_on(two, 2), "sentence twice (not thrice) not stopped");
    std::string three = two +
                        "她伸手拂去钟面上的灰，指腹被划出一道细痕。\n" +
                        std::string(loopline) + "\n";
    check(fired_on(three, 2), "sentence thrice interleaved caught (duplicate-3x)");
  }

  // 9) a repeated single short word is NOT enough (min unit length 4 bytes)
  {
    check(!fired_on("哈哈哈哈哈", 2), "single repeated char not stopped");
  }

  std::printf("%s\n", g_fail == 0 ? "ALL REPEAT-GUARD TESTS PASSED"
                                  : "REPEAT-GUARD TESTS FAILED");
  return g_fail == 0 ? 0 : 1;
}
