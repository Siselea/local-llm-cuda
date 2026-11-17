// Tokenizer test tool + batch driver for tools/check_tokenizer.py.
//
//   test_tokenizer <model.gguf> --self-test
//   test_tokenizer <model.gguf> --info
//   test_tokenizer <model.gguf> --ids-file <text> --out <ids>      # encode a whole file
//   test_tokenizer <model.gguf> --decode-ids <ids> --out <text>    # detokenize
//   test_tokenizer <model.gguf> --chat-template-file <json> --out <text>
//
// --self-test needs no external oracle: it checks decode(encode(s)) == s over a
// corpus of tricky UTF-8 strings, special-token atomicity, and that the rendered
// ChatML chat template tokenizes to the documented Qwen message sequence.
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/json_parse.h"
#include "engine/tokenizer.h"

namespace {

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void write_file(const std::string& path, const std::string& data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

std::vector<int> parse_ids(const std::string& text) {
  std::vector<int> ids;
  size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    if (i >= text.size()) break;
    size_t j = i;
    while (j < text.size() && !std::isspace(static_cast<unsigned char>(text[j]))) ++j;
    ids.push_back(std::atoi(text.substr(i, j - i).c_str()));
    i = j;
  }
  return ids;
}

std::string escape(const std::string& s) {
  std::string r;
  for (unsigned char c : s) {
    switch (c) {
      case '\n': r += "\\n"; break;
      case '\r': r += "\\r"; break;
      case '\t': r += "\\t"; break;
      case '\\': r += "\\\\"; break;
      default:
        if (c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\x%02x", c);
          r += b;
        } else {
          r.push_back(static_cast<char>(c));
        }
    }
  }
  return r;
}

// UTF-8 mixed corpus: every branch of the Qwen3.5 pre-tokenizer plus byte-level
// BPE round-trip hazards (astral plane, combining marks, ZWJ emoji, NBSP).
const std::vector<std::string>& corpus() {
  static const std::vector<std::string> lines = {
      "Hello world! This is a tokenizer test.",
      "It's 2026 and I'd like to say you'll see we're fine; he's gone.",
      "Numbers: 0 1 12 123 1234 12345 999999 3.14159 2026-11-15",
      "CJK: \xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\x96\xE7\x95\x8C\xE3\x80\x82"
      "\xE8\xBF\x99\xE6\x98\xAF\xE4\xB8\x80\xE4\xB8\xAA\xE6\xB5\x8B\xE8\xAF\x95\xE3\x80\x82"
      "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E""abc\xE6\xB7\xB7\xE5\x9C\xA8\xE3\x80\x82",
      "Astral: \xF0\x9F\x98\x80\xF0\x9F\x8E\x89\xF0\x9F\x9A\x80 "
      "\xF0\x9D\x94\x98\xF0\x9D\x94\xAB\xF0\x9D\x94\xA6\xF0\x9D\x94\xAA "
      "\xC2\xBC \xC2\xBD \xD9\xA3\xD9\xA4\xD9\xA5 \xEF\xBC\x91\xEF\xBC\x92\xEF\xBC\x93",
      "Combining: e\xCC\x81 vs \xC3\xA9, zoom\xCC\x88, A\xCC\x8A\xCC\x81",
      "Whitespace: a   b\tc",
      "d",
      "e",
      "",
      "",
      "f",
      "trailing spaces   end",
      "Punctuation: !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ... ?! --- /// ",
      "Quotes: \"double\" 'single' \xE2\x80\x9C""curly\xE2\x80\x9D "
      "\xE2\x80\x98q\xE2\x80\x99 \xE3\x80\x8C\xE3\x81\x8B\xE3\x81\x8E\xE3\x80\x8D",
      "Code: def foo(x): return x*2 # comment",
      "    if x != 0:",
      "        print(f\"{x:03d}\")",
      "Unicode letters: \xC3\x9Cn\xC3\xAF\x63\xC3\xB6\x64\xC3\xA9 "
      "\xC3\x80\xC3\x89\xC3\x8E\xC3\x95\xC3\x9C \xC3\xB1 \xC3\xA7 \xC3\xB8 "
      "\xC3\xA5 \xC5\x93 \xC3\x9F",
      "Zero width: a\xE2\x80\x8B""b\xE2\x80\x8C""c\xE2\x80\x8D""d",
      "Symbols: \xC2\xB1 \xC2\xA7 \xC2\xA9 \xC2\xAE \xE2\x84\xA2 \xC2\xB0 "
      "\xC2\xB5 \xE2\x82\xAC \xC2\xA3 \xC2\xA5 \xE2\x82\xB9",
      "Spaces: a\xC2\xA0""b\xE3\x80\x80""c\xE2\x80\xA8""d",
      "Fullwidth: \xEF\xBC\xA1\xEF\xBC\xA2\xEF\xBC\xA3\xEF\xBC\x91\xEF\xBC\x92\xEF\xBC\x93\xEF\xBC\x81\xEF\xBC\x9F",
      "Emoji ZWJ: \xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D"
      "\xF0\x9F\x91\xA7\xE2\x80\x8D\xF0\x9F\x91\xA6 \xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD",
      "Repeated: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "Mixed \xE4\xB8\xAD\xE6\x96\x87""English123!@# end",
  };
  return lines;
}

std::string join_corpus() {
  std::string text;
  for (const auto& l : corpus()) {
    text += l;
    text += '\n';
  }
  return text;
}

int run_self_test(const engine::tokenizer::Tokenizer& tk) {
  int fail = 0;
  auto check = [&](bool ok, const char* what) {
    std::printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  };

  const std::string text = join_corpus();
  const std::vector<int> ids = tk.encode(text, /*add_special=*/false, /*parse_special=*/true);
  const std::string back = tk.decode(ids, /*skip_special=*/true);
  const bool rt = (back == text);
  check(rt, "decode(encode(corpus)) == corpus");
  if (!rt) {
    size_t i = 0;
    while (i < back.size() && i < text.size() && back[i] == text[i]) ++i;
    std::printf("    first diff at byte %zu\n    got : %s\n    want: %s\n", i,
                escape(back.substr(i > 40 ? i - 40 : 0, 80)).c_str(),
                escape(text.substr(i > 40 ? i - 40 : 0, 80)).c_str());
  }
  std::printf("  corpus: %zu bytes -> %zu tokens (%.3f bytes/token)\n", text.size(),
              ids.size(), ids.empty() ? 0.0 : static_cast<double>(text.size()) / ids.size());

  // encode must be deterministic
  check(tk.encode(text) == ids, "encode is deterministic");

  // special tokens stay atomic
  const int im_start = tk.token_id("<|im_start|>");
  const int im_end = tk.token_id("<|im_end|>");
  if (im_start >= 0 && im_end >= 0) {
    const std::string s = "<|im_start|>user\nhi<|im_end|>\n";
    const std::vector<int> sids = tk.encode(s, false, true);
    bool atomic = !sids.empty() && sids.front() == im_start && sids.back() == 198;
    bool has_im_end = false;
    for (int id : sids) has_im_end = has_im_end || id == im_end;
    check(atomic && has_im_end, "special tokens <|im_start|>/<|im_end|> atomic");
    // skip_special decode drops the control tokens
    const std::string dec = tk.decode(sids, true);
    check(dec == "user\nhi\n", "decode skips control tokens only");
    // parse_special=false must NOT produce the control ids
    const std::vector<int> plain = tk.encode(s, false, false);
    bool plain_has = false;
    for (int id : plain) plain_has = plain_has || id == im_start;
    check(!plain_has, "parse_special=false does not emit control token");
  } else {
    std::printf("  (no <|im_start|> token in this vocab: special-token check skipped)\n");
  }

  // Chat template (Qwen ChatML).  Qwen3.5+ templates are thinking-aware (they
  // open a <think> block in the generation prompt); older Qwen2.5 ones are not.
  const bool thinking_aware =
      tk.chat_template().find("enable_thinking") != std::string::npos &&
      tk.chat_template().find("<think>") != std::string::npos;
  if (tk.has_chat_template() && std::strcmp(tk.template_name(), "chatml") == 0) {
    std::vector<engine::tokenizer::ChatMessage> msgs;
    msgs.push_back({"system", "You are a helpful assistant.", ""});
    msgs.push_back({"user", "\xE4\xBD\xA0\xE5\xA5\xBD", ""});  // 你好
    const std::string prompt = tk.apply_chat_template(msgs, true, /*thinking=*/true);
    const std::string tail = thinking_aware ? "<think>\n" : "";
    const bool shape = prompt.rfind("<|im_start|>system\n", 0) == 0 &&
                       prompt.find("<|im_end|>\n<|im_start|>user\n") != std::string::npos &&
                       prompt.size() >= tail.size() + 23 &&
                       prompt.compare(prompt.size() - tail.size(), tail.size(), tail) == 0 &&
                       prompt.find("<|im_start|>assistant\n") != std::string::npos;
    check(shape, thinking_aware ? "chat template renders ChatML + <think> generation prompt"
                                : "chat template renders ChatML generation prompt");
    std::printf("  rendered prompt (%zu bytes):\n    %s\n", prompt.size(),
                escape(prompt).c_str());

    const std::vector<int> pids = tk.encode(prompt, false, true);
    bool seq_ok = !pids.empty() && pids.front() == im_start;
    if (im_end >= 0) {
      bool has_end = false;
      for (int id : pids) has_end = has_end || id == im_end;
      seq_ok = seq_ok && has_end;
    }
    const int think_id = tk.token_id("<think>");
    // the template ends with "<*think*>\n": the <*think*> token then a newline token
    if (think_id >= 0)
      seq_ok = seq_ok && pids.size() >= 2 && pids[pids.size() - 2] == think_id;
    check(seq_ok, "chat prompt token ids start <|im_start|> and end <think>");
    std::printf("  chat prompt ids:");
    for (size_t i = 0; i < pids.size() && i < 24; ++i) std::printf(" %d", pids[i]);
    if (pids.size() > 24) std::printf(" ... (%zu total)", pids.size());
    std::printf("\n");

    if (thinking_aware) {
      const std::string no_think = tk.apply_chat_template(msgs, true, /*thinking=*/false);
      check(no_think.size() > 19 &&
                no_think.compare(no_think.size() - 19, 19, "<think>\n\n</think>\n\n") == 0,
            "enable_thinking=false renders the empty-think prefill");
    }

    const std::string no_gen = tk.apply_chat_template(msgs, false, true);
    check(no_gen.size() >= 11 && no_gen.compare(no_gen.size() - 11, 11, "<|im_end|>\n") == 0,
          "add_generation_prompt=false ends with <|im_end|>\\n");
  } else {
    std::printf("  (chat template not ChatML: template checks skipped)\n");
  }

  std::printf("\n%s\n", fail == 0 ? "TOKENIZER SELF-TEST PASSED" : "TOKENIZER SELF-TEST FAILED");
  return fail == 0 ? 0 : 1;
}

int run_chat_template_file(const engine::tokenizer::Tokenizer& tk, const std::string& path,
                           const std::string& out_path) {
  const jsonp::Value doc = jsonp::parse(read_file(path));
  const jsonp::Value* msgs = doc.find("messages");
  if (!msgs || !msgs->is_array()) throw std::runtime_error("chat-template JSON needs messages[]");
  std::vector<engine::tokenizer::ChatMessage> messages;
  for (const auto& m : msgs->arr) {
    engine::tokenizer::ChatMessage msg;
    msg.role = m.get_str("role", "user");
    msg.content = m.get_str("content", "");
    msg.reasoning_content = m.get_str("reasoning_content", "");
    messages.push_back(std::move(msg));
  }
  const bool add_gen = doc.get_bool("add_generation_prompt", true);
  const bool thinking = doc.get_bool("enable_thinking", true);
  const std::string rendered = tk.apply_chat_template(messages, add_gen, thinking);
  if (out_path.empty()) {
    std::fwrite(rendered.data(), 1, rendered.size(), stdout);
  } else {
    write_file(out_path, rendered);
  }
  return rendered.empty() ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: test_tokenizer <model.gguf> [--self-test|--info|--ids-file F "
                           "--out F|--decode-ids F --out F|--chat-template-file F --out F]\n");
      return 2;
    }
    const std::string model = argv[1];
    std::string mode = "--self-test";
    std::string ids_file, out_file, json_file;
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--self-test" || a == "--info") {
        mode = a;
      } else if ((a == "--ids-file" || a == "--decode-ids" || a == "--out" ||
                  a == "--chat-template-file") && i + 1 < argc) {
        const std::string v = argv[++i];
        if (a == "--ids-file") { mode = a; ids_file = v; }
        else if (a == "--decode-ids") { mode = a; ids_file = v; }
        else if (a == "--out") out_file = v;
        else { mode = a; json_file = v; }
      } else {
        std::fprintf(stderr, "unknown option %s\n", a.c_str());
        return 2;
      }
    }

    const engine::tokenizer::Tokenizer tk = engine::tokenizer::Tokenizer::load(model);
    if (mode == "--info") {
      std::printf("tokenizer.model : %s\n", tk.kind() == engine::tokenizer::Kind::Bpe ? "gpt2 (BPE)"
                                                                                     : "llama (SPM)");
      std::printf("tokenizer.pre   : %s\n", tk.pre().empty() ? "(none)" : tk.pre().c_str());
      std::printf("vocab           : %d\n", tk.vocab_size());
      std::printf("bos/eos         : %d / %d  (add_bos=%d add_eos=%d)\n", tk.bos_id(),
                  tk.eos_id(), tk.add_bos(), tk.add_eos());
      std::printf("chat template   : %s (%zu bytes)\n", tk.template_name(),
                  tk.chat_template().size());
      return 0;
    }
    if (mode == "--ids-file") {
      const std::string text = read_file(ids_file);
      const std::vector<int> ids = tk.encode(text, false, true);
      std::string out;
      out.reserve(ids.size() * 7);
      for (int id : ids) {
        out += std::to_string(id);
        out += '\n';
      }
      if (out_file.empty()) std::fwrite(out.data(), 1, out.size(), stdout);
      else write_file(out_file, out);
      return 0;
    }
    if (mode == "--decode-ids") {
      const std::vector<int> ids = parse_ids(read_file(ids_file));
      const std::string text = tk.decode(ids, /*skip_special=*/true);
      if (out_file.empty()) std::fwrite(text.data(), 1, text.size(), stdout);
      else write_file(out_file, text);
      return 0;
    }
    if (mode == "--chat-template-file") {
      return run_chat_template_file(tk, json_file, out_file);
    }
    return run_self_test(tk);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
