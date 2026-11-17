// GGUF tokenizer: byte-level BPE + SentencePiece.
//
// The pre-tokenizer splitters are ports of llama.cpp b12128
// src/unicode.cpp:
//   unicode_regex_split_custom_qwen35()   (LLAMA_VOCAB_PRE_TYPE_QWEN35)
//   unicode_regex_split_custom_qwen2()    (LLAMA_VOCAB_PRE_TYPE_QWEN2)
// and the classic GPT-2 regex for every other "gpt2" vocabulary.
// The BPE merge algorithm mirrors src/llama-vocab.cpp
// llm_tokenizer_bpe_session (rank-ordered merges, byte fallback) and the SPM
// path mirrors llm_tokenizer_spm_session (score-ordered merges + resegment).
//
// Everything here was validated against llama-tokenize (see
// tools/check_tokenizer.py) on the four local GGUF models.
#include "engine/tokenizer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "engine/gguf.h"
#include "engine/text_util.h"
#include "engine/unicode_tables.h"

namespace engine {
namespace tokenizer {

namespace {

constexpr uint32_t kOutOfRange = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// GPT-2 byte <-> unicode mapping (bytes_to_unicode from the GPT-2 paper).
// ---------------------------------------------------------------------------
void init_byte_alphabet(std::string (&byte_to_utf8)[256],
                        std::unordered_map<uint32_t, uint8_t>& cp_to_byte) {
  std::vector<int> bs;
  std::vector<int> cs;
  auto push = [&](int b, int c) {
    bs.push_back(b);
    cs.push_back(c);
  };
  for (int c = 0x21; c <= 0x7E; ++c) push(c, c);        // '!' .. '~'
  for (int c = 0xA1; c <= 0xAC; ++c) push(c, c);        // '¡' .. '¬'
  for (int c = 0xAE; c <= 0xFF; ++c) push(c, c);        // '®' .. 'ÿ'
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) push(b, 256 + n++);
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    byte_to_utf8[bs[i]] = text::from_codepoint(static_cast<uint32_t>(cs[i]));
    cp_to_byte[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
  }
}

uint32_t ascii_tolower(uint32_t c) {
  return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

// ---------------------------------------------------------------------------
// Pre-tokenizer pieces, as [begin,end) byte ranges of the source text.
// ---------------------------------------------------------------------------
using Span = std::pair<size_t, size_t>;
using Spans = std::vector<Span>;

struct CptView {
  const std::string* text;
  std::vector<uint32_t> cpts;
  std::vector<size_t> offs;  // offs.size() == cpts.size() + 1

  explicit CptView(const std::string& s) : text(&s) {
    cpts = text::to_codepoints(s, &offs);
  }
  size_t size() const { return cpts.size(); }
  uint32_t cpt(size_t p) const { return p < cpts.size() ? cpts[p] : kOutOfRange; }
  uint8_t flags(size_t p) const { return p < cpts.size() ? unicode::flags_of(cpts[p]) : 0; }
  bool letter(size_t p) const { return (flags(p) & unicode::kLetter) != 0; }
  bool number(size_t p) const { return (flags(p) & unicode::kNumber) != 0; }
  bool mark(size_t p) const { return (flags(p) & unicode::kMark) != 0; }
  bool space(size_t p) const { return p < cpts.size() && unicode::is_whitespace(cpts[p]); }
  Span span(size_t begin, size_t end) const { return {offs[begin], offs[end]}; }
};

// Shared tail of the qwen2/qwen35 splitters: punctuation run + whitespace runs.
// Mirrors llama.cpp unicode_regex_split_custom_qwen{,2,35}() exactly.
// Returns true when it emitted a span.
bool split_punct_or_space(const CptView& v, size_t& pos, Spans& out,
                          bool include_marks) {
  const uint32_t cpt = v.cpt(pos);
  // !(is_whitespace | is_letter | (\p{M}?) | is_number)
  auto plain = [&](size_t p) {
    if (p >= v.size()) return false;
    if (v.space(p)) return false;
    const uint8_t f = v.flags(p);
    if (f & (unicode::kLetter | unicode::kNumber)) return false;
    if (include_marks && (f & unicode::kMark)) return false;
    return true;
  };
  // llama.cpp checks flags.as_uint() != 0.  The generated table gives every
  // codepoint a category bit (unassigned -> kUndefined), exactly like
  // llama.cpp's unicode_ranges_flags, so this is just "inside the text".
  auto nonzero = [&](size_t p) { return p < v.size() && (v.flags(p) != 0 || v.space(p)); };

  // regex: <space>?[^\s\p{L}(\p{M})\p{N}]+[\r\n]*
  const size_t f2pos = (cpt == ' ') ? pos + 1 : pos;
  const bool f2_ok = f2pos < v.size() && plain(f2pos) && nonzero(f2pos);
  if (f2_ok && nonzero(pos)) {
    size_t p = pos + (cpt == ' ' ? 1 : 0);
    while (p < v.size() && plain(p) && nonzero(p)) ++p;
    while (v.cpt(p) == '\r' || v.cpt(p) == '\n') ++p;
    out.push_back(v.span(pos, p));
    pos = p;
    return true;
  }

  size_t num_whitespaces = 0;
  size_t last_end_r_or_n = 0;
  while (v.space(pos + num_whitespaces)) {
    const uint32_t c2 = v.cpt(pos + num_whitespaces);
    if (c2 == '\r' || c2 == '\n') last_end_r_or_n = pos + num_whitespaces + 1;
    ++num_whitespaces;
  }
  if (last_end_r_or_n > 0) {  // regex: \s*[\r\n]+
    out.push_back(v.span(pos, last_end_r_or_n));
    pos = last_end_r_or_n;
    return true;
  }
  if (num_whitespaces > 1 && v.cpt(pos + num_whitespaces) != kOutOfRange) {
    // regex: \s+(?!\S) -- leave the last whitespace for the next piece
    const size_t p = pos + num_whitespaces - 1;
    out.push_back(v.span(pos, p));
    pos = p;
    return true;
  }
  if (num_whitespaces > 0) {  // regex: \s+
    out.push_back(v.span(pos, pos + num_whitespaces));
    pos += num_whitespaces;
    return true;
  }
  return false;
}

// regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)
bool split_contraction(const CptView& v, size_t& pos, Spans& out) {
  if (v.cpt(pos) != '\'' || pos + 1 >= v.size()) return false;
  const uint32_t c1 = ascii_tolower(v.cpt(pos + 1));
  if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
    out.push_back(v.span(pos, pos + 2));
    pos += 2;
    return true;
  }
  if (pos + 2 < v.size()) {
    const uint32_t c2 = ascii_tolower(v.cpt(pos + 2));
    if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
      out.push_back(v.span(pos, pos + 3));
      pos += 3;
      return true;
    }
  }
  return false;
}

// regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}(\p{M})\p{N}]?[\p{L}(\p{M})]+|\p{N}| ?[^\s\p{L}(\p{M})\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
Spans split_qwen(const std::string& text, bool include_marks) {
  CptView v(text);
  Spans out;
  auto is_word = [&](size_t p) { return v.letter(p) || (include_marks && v.mark(p)); };
  size_t pos = 0;
  while (pos < v.size()) {
    if (split_contraction(v, pos, out)) continue;
    const uint32_t cpt = v.cpt(pos);
    // optional single non-CR/LF/number prefix followed by a letter/mark run
    if (!(cpt == '\r' || cpt == '\n' || v.number(pos))) {
      if (is_word(pos) || is_word(pos + 1)) {
        size_t p = pos + 1;
        while (is_word(p)) ++p;
        out.push_back(v.span(pos, p));
        pos = p;
        continue;
      }
    }
    if (v.number(pos)) {  // regex: \p{N}
      out.push_back(v.span(pos, pos + 1));
      ++pos;
      continue;
    }
    if (split_punct_or_space(v, pos, out, include_marks)) continue;
    out.push_back(v.span(pos, pos + 1));  // no match: one codepoint
    ++pos;
  }
  return out;
}

// Canonical GPT-2 regex:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Port of llama.cpp unicode_regex_split_custom_gpt2().
Spans split_gpt2(const std::string& text) {
  CptView v(text);
  Spans out;
  size_t pos = 0;
  while (pos < v.size()) {
    const uint32_t cpt = v.cpt(pos);
    // contractions are case sensitive in the original GPT-2 regex
    if (cpt == '\'' && pos + 1 < v.size()) {
      const uint32_t c1 = v.cpt(pos + 1);
      if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
        out.push_back(v.span(pos, pos + 2));
        pos += 2;
        continue;
      }
      if (pos + 2 < v.size()) {
        const uint32_t c2 = v.cpt(pos + 2);
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
          out.push_back(v.span(pos, pos + 3));
          pos += 3;
          continue;
        }
      }
    }
    const size_t f2pos = (cpt == ' ') ? pos + 1 : pos;
    const bool lead_space = (cpt == ' ');

    // regex: <space>?\p{L}+
    if (f2pos < v.size() && v.letter(f2pos)) {
      size_t p = pos + (lead_space ? 1 : 0);
      while (p < v.size() && v.letter(p)) ++p;
      out.push_back(v.span(pos, p));
      pos = p;
      continue;
    }
    // regex: <space>?\p{N}+
    if (f2pos < v.size() && v.number(f2pos)) {
      size_t p = pos + (lead_space ? 1 : 0);
      while (p < v.size() && v.number(p)) ++p;
      out.push_back(v.span(pos, p));
      pos = p;
      continue;
    }
    // regex: <space>?[^\s\p{L}\p{N}]+
    if (f2pos < v.size() && !v.space(f2pos) && !v.letter(f2pos) && !v.number(f2pos) &&
        v.flags(f2pos) != 0) {
      size_t p = pos + (lead_space ? 1 : 0);
      while (p < v.size() && !v.space(p) && !v.letter(p) && !v.number(p) && v.flags(p) != 0) ++p;
      out.push_back(v.span(pos, p));
      pos = p;
      continue;
    }
    // regex: \s+(?!\S) then \s+
    size_t num_whitespaces = 0;
    while (v.space(pos + num_whitespaces)) ++num_whitespaces;
    if (num_whitespaces > 1 && v.cpt(pos + num_whitespaces) != kOutOfRange) {
      out.push_back(v.span(pos, pos + num_whitespaces - 1));
      pos += num_whitespaces - 1;
      continue;
    }
    if (num_whitespaces > 0) {
      out.push_back(v.span(pos, pos + num_whitespaces));
      pos += num_whitespaces;
      continue;
    }
    out.push_back(v.span(pos, pos + 1));
    ++pos;
  }
  return out;
}


// ---------------------------------------------------------------------------
// GGUF metadata helpers (captured arrays are raw little-endian bytes).
// ---------------------------------------------------------------------------
int64_t read_int_elem(const std::vector<unsigned char>& raw, uint32_t elem_type, size_t i) {
  const unsigned char* p = raw.data();
  switch (static_cast<gguf::ValueType>(elem_type)) {
    case gguf::ValueType::UINT8: return p[i];
    case gguf::ValueType::INT8: return static_cast<int8_t>(p[i]);
    case gguf::ValueType::UINT16: {
      uint16_t v;
      std::memcpy(&v, p + i * 2, 2);
      return v;
    }
    case gguf::ValueType::INT16: {
      int16_t v;
      std::memcpy(&v, p + i * 2, 2);
      return v;
    }
    case gguf::ValueType::UINT32:
    case gguf::ValueType::INT32: {
      int32_t v;
      std::memcpy(&v, p + i * 4, 4);
      return v;
    }
    case gguf::ValueType::UINT64:
    case gguf::ValueType::INT64: {
      int64_t v;
      std::memcpy(&v, p + i * 8, 8);
      return v;
    }
    default: return 0;
  }
}

double read_float_elem(const std::vector<unsigned char>& raw, uint32_t elem_type, size_t i) {
  const unsigned char* p = raw.data();
  if (static_cast<gguf::ValueType>(elem_type) == gguf::ValueType::FLOAT64) {
    double v;
    std::memcpy(&v, p + i * 8, 8);
    return v;
  }
  if (static_cast<gguf::ValueType>(elem_type) == gguf::ValueType::FLOAT32) {
    float v;
    std::memcpy(&v, p + i * 4, 4);
    return static_cast<double>(v);
  }
  return static_cast<double>(read_int_elem(raw, elem_type, i));
}

// Token text of a BFPE/SPM special token for the encode-time matcher.
bool template_has(const std::string& s, const char* needle) {
  return s.find(needle) != std::string::npos;
}

std::string trim_copy(const std::string& s) {
  size_t b = 0, e = s.size();
  auto is_ws = [](unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
  };
  while (b < e && is_ws(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && is_ws(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

}  // namespace

// ---------------------------------------------------------------------------
const char* Tokenizer::template_name() const {
  switch (template_kind_) {
    case Template::Chatml: return "chatml";
    default: return "none";
  }
}

// ---------------------------------------------------------------------------
Tokenizer Tokenizer::load(const std::string& gguf_path) {
  Tokenizer t;
  const std::vector<std::string> capture = {
      "tokenizer.ggml.tokens",
      "tokenizer.ggml.token_type",
      "tokenizer.ggml.scores",
      "tokenizer.ggml.merges",
  };
  const gguf::Model g = gguf::read(gguf_path, capture);

  const std::string model = g.get_string("tokenizer.ggml.model", "");
  if (model == "gpt2") {
    t.kind_ = Kind::Bpe;
  } else if (model == "llama") {
    t.kind_ = Kind::Spm;
  } else if (model.empty()) {
    throw std::runtime_error("GGUF has no tokenizer.ggml.model (no tokenizer in this file)");
  } else {
    throw std::runtime_error("unsupported tokenizer.ggml.model '" + model +
                             "' (supported: gpt2, llama)");
  }
  t.pre_ = g.get_string("tokenizer.ggml.pre", "");
  t.chat_template_ = g.get_string("tokenizer.chat_template");
  if (t.chat_template_.empty()) t.chat_template_ = g.get_string("tokenizer.ggml.chat_template");

  auto it_tokens = g.kv.find("tokenizer.ggml.tokens");
  if (it_tokens == g.kv.end() || it_tokens->second.kind != gguf::Value::Kind::ARRAY) {
    throw std::runtime_error("GGUF has no tokenizer.ggml.tokens");
  }
  t.tokens_ = it_tokens->second.strs;
  if (t.tokens_.empty()) throw std::runtime_error("tokenizer.ggml.tokens is empty");
  t.token_types_.assign(t.tokens_.size(), 1);
  for (int& b : t.byte_token_) b = -1;

  auto it_types = g.kv.find("tokenizer.ggml.token_type");
  if (it_types != g.kv.end() && it_types->second.kind == gguf::Value::Kind::ARRAY) {
    const auto& v = it_types->second;
    for (size_t i = 0; i < t.tokens_.size() && i < v.array_len; ++i)
      t.token_types_[i] = static_cast<int>(read_int_elem(v.raw, v.array_elem_type, i));
  }
  auto it_scores = g.kv.find("tokenizer.ggml.scores");
  if (it_scores != g.kv.end() && it_scores->second.kind == gguf::Value::Kind::ARRAY) {
    const auto& v = it_scores->second;
    t.scores_.resize(t.tokens_.size(), 0.0f);
    for (size_t i = 0; i < t.tokens_.size() && i < v.array_len; ++i)
      t.scores_[i] = static_cast<float>(read_float_elem(v.raw, v.array_elem_type, i));
  }

  // token_to_id: first occurrence wins (llama.cpp unordered_map::emplace)
  t.token_to_id_.reserve(t.tokens_.size() * 2);
  for (size_t i = 0; i < t.tokens_.size(); ++i) t.token_to_id_.emplace(t.tokens_[i], (int)i);

  // special tokens cache: control / user-defined / unknown, longest text first
  for (size_t i = 0; i < t.tokens_.size(); ++i) {
    const int ty = t.token_types_[i];
    if (ty == 2 || ty == 3 || ty == 4) t.special_ids_.push_back(static_cast<int>(i));
  }
  std::sort(t.special_ids_.begin(), t.special_ids_.end(), [&](int a, int b) {
    return t.tokens_[a].size() > t.tokens_[b].size();
  });

  if (t.kind_ == Kind::Bpe) {
    init_byte_alphabet(t.byte_to_utf8_, t.cp_to_byte_);
    auto it_merges = g.kv.find("tokenizer.ggml.merges");
    if (it_merges != g.kv.end() && it_merges->second.kind == gguf::Value::Kind::ARRAY) {
      const auto& merges = it_merges->second.strs;
      t.merge_ranks_.reserve(merges.size() * 2);
      for (size_t i = 0; i < merges.size(); ++i) {
        const std::string& m = merges[i];
        const size_t sep = m.empty() ? std::string::npos : m.find(' ', 1);
        if (sep == std::string::npos) continue;
        std::string key = m.substr(0, sep);
        key.push_back(' ');
        key.append(m, sep + 1, std::string::npos);
        t.merge_ranks_.emplace(std::move(key), static_cast<int>(i));
      }
    }
    // pre-tokenizer family
    if (t.pre_ == "qwen35") t.pre_kind_ = Pre::Qwen35;
    else if (t.pre_ == "qwen2" || t.pre_ == "deepseek-r1-qwen" || t.pre_ == "kormo" ||
             t.pre_ == "f2llmv2") t.pre_kind_ = Pre::Qwen2;
    else t.pre_kind_ = Pre::Gpt2;
  } else {
    // SPM byte fallback: <0xNN> token ids
    for (size_t i = 0; i < t.tokens_.size(); ++i) {
      const std::string& s = t.tokens_[i];
      if (s.size() == 6 && s.compare(0, 3, "<0x") == 0 && s.back() == '>') {
        auto hexv = [](char c) -> int {
          if (c >= '0' && c <= '9') return c - '0';
          if (c >= 'A' && c <= 'F') return c - 'A' + 10;
          if (c >= 'a' && c <= 'f') return c - 'a' + 10;
          return -1;
        };
        const int hi = hexv(s[3]), lo = hexv(s[4]);
        if (hi >= 0 && lo >= 0 && t.byte_token_[hi * 16 + lo] < 0)
          t.byte_token_[hi * 16 + lo] = static_cast<int>(i);
      }
    }
  }

  // special ids with a byte type (SPM/BPE <0xNN>) need the byte table even for BPE
  if (t.kind_ == Kind::Bpe && t.byte_token_[0] == -1) {
    for (size_t i = 0; i < t.tokens_.size(); ++i) {
      const std::string& s = t.tokens_[i];
      if (t.token_types_[i] == 6) {  // GGUF BYTE
        if (s.size() == 6 && s.compare(0, 3, "<0x") == 0 && s.back() == '>') {
          auto hexv = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            return -1;
          };
          const int hi = hexv(s[3]), lo = hexv(s[4]);
          if (hi >= 0 && lo >= 0) t.byte_token_[hi * 16 + lo] = static_cast<int>(i);
        }
      }
    }
  }

  t.bos_id_ = static_cast<int>(g.get_int("tokenizer.ggml.bos_token_id", -1));
  t.eos_id_ = static_cast<int>(g.get_int("tokenizer.ggml.eos_token_id", -1));
  t.unk_id_ = static_cast<int>(g.get_int("tokenizer.ggml.unknown_token_id", -1));
  // llama.cpp defaults: BPE does not add BOS/EOS, SPM adds BOS; GGUF overrides.
  t.add_bos_ = g.has("tokenizer.ggml.add_bos_token")
                   ? g.get_bool("tokenizer.ggml.add_bos_token")
                   : (t.kind_ == Kind::Spm);
  t.add_eos_ = g.has("tokenizer.ggml.add_eos_token")
                   ? g.get_bool("tokenizer.ggml.add_eos_token")
                   : false;
  if (t.bos_id_ < 0) t.add_bos_ = false;
  if (t.eos_id_ < 0) t.add_eos_ = false;
  // llama.cpp: add_space_prefix defaults to true for SPM, only read for SPM.
  t.add_space_prefix_ =
      g.has("tokenizer.ggml.add_space_prefix")
          ? g.get_bool("tokenizer.ggml.add_space_prefix")
          : (t.kind_ == Kind::Spm);

  if (template_has(t.chat_template_, "<|im_start|>") &&
      template_has(t.chat_template_, "<|im_end|>") &&
      template_has(t.chat_template_, "add_generation_prompt")) {
    t.template_kind_ = Template::Chatml;
  }
  return t;
}

// ---------------------------------------------------------------------------
namespace {

// llama.cpp llama_escape_whitespace(): " " -> U+2581 LOWER ONE EIGHTH BLOCK.
std::string escape_spm_spaces(const std::string& text) {
  static const char kSpaceMark[] = "\xE2\x96\x81";
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    if (c == ' ') out += kSpaceMark;
    else out.push_back(c);
  }
  return out;
}

// Split `text` into pre-tokenizer pieces.
std::vector<std::string> pre_split_impl(const std::string& text, const Spans& spans) {
  std::vector<std::string> pieces;
  pieces.reserve(spans.size());
  for (const auto& sp : spans) pieces.push_back(text.substr(sp.first, sp.second - sp.first));
  return pieces;
}

// UTF-8 codepoint split of an already byte-encoded BPE word.
std::vector<std::string> utf8_symbols(const std::string& s) {
  std::vector<std::string> syms;
  size_t i = 0;
  while (i < s.size()) {
    const size_t len = text::utf8_seq_len(static_cast<unsigned char>(s[i]));
    syms.push_back(s.substr(i, len));
    i += len;
  }
  return syms;
}

}  // namespace

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
  if (kind_ == Kind::Spm) return {text};
  Spans spans;
  switch (pre_kind_) {
    case Pre::Qwen35: spans = split_qwen(text, /*include_marks=*/true); break;
    case Pre::Qwen2: spans = split_qwen(text, /*include_marks=*/false); break;
    case Pre::Gpt2: spans = split_gpt2(text); break;
  }
  return pre_split_impl(text, spans);
}

std::vector<std::string> Tokenizer::pre_split(const std::string& text) const {
  return pre_tokenize(text);
}

// ---------------------------------------------------------------------------
std::vector<int> Tokenizer::bpe_word(const std::string& word) const {
  auto cached = word_cache_.find(word);
  if (cached != word_cache_.end()) return cached->second;

  // byte-level unicode form -> codepoint symbols
  std::string encoded;
  encoded.reserve(word.size() * 2);
  for (unsigned char b : word) encoded += byte_to_utf8_[b];
  std::vector<std::string> syms = utf8_symbols(encoded);

  auto rank_of = [&](const std::string& a, const std::string& b) -> int {
    std::string key;
    key.reserve(a.size() + b.size() + 1);
    key += a;
    key.push_back(' ');
    key += b;
    auto it = merge_ranks_.find(key);
    return it == merge_ranks_.end() ? -1 : it->second;
  };

  while (syms.size() > 1) {
    int best = -1;
    int best_rank = std::numeric_limits<int>::max();
    for (size_t i = 0; i + 1 < syms.size(); ++i) {
      const int r = rank_of(syms[i], syms[i + 1]);
      if (r >= 0 && r < best_rank) {
        best_rank = r;
        best = static_cast<int>(i);
      }
    }
    if (best < 0) break;
    syms[best] += syms[best + 1];
    syms.erase(syms.begin() + best + 1);
  }

  std::vector<int> ids;
  ids.reserve(syms.size());
  for (const auto& s : syms) {
    auto it = token_to_id_.find(s);
    if (it != token_to_id_.end()) {
      ids.push_back(it->second);
      continue;
    }
    // byte fallback (should not happen for a consistent vocab)
    for (unsigned char b : s) {
      auto bt = token_to_id_.find(byte_to_utf8_[b]);
      if (bt != token_to_id_.end()) ids.push_back(bt->second);
    }
  }
  if (word_cache_.size() > 1u << 17) word_cache_.clear();
  word_cache_.emplace(word, ids);
  return ids;
}

std::vector<int> Tokenizer::encode_ordinary(const std::string& text) const {
  std::vector<int> ids;
  if (text.empty()) return ids;
  if (kind_ == Kind::Bpe) {
    for (const auto& word : pre_tokenize(text)) {
      std::vector<int> w = bpe_word(word);
      ids.insert(ids.end(), w.begin(), w.end());
    }
    return ids;
  }
  return spm_text(text);
}

// ---------------------------------------------------------------------------
// SentencePiece: score-ordered bigram merges + resegment.
// Port of llama.cpp llm_tokenizer_spm_session (b12128).
// ---------------------------------------------------------------------------
std::vector<int> Tokenizer::spm_text(const std::string& text) const {
  struct Sym {
    std::string text;
    int prev = -1, next = -1;
    bool alive = true;
  };
  std::vector<Sym> syms;
  size_t i = 0;
  while (i < text.size()) {
    const size_t len = text::utf8_seq_len(static_cast<unsigned char>(text[i]));
    const size_t clen = std::min(len, text.size() - i);
    Sym s;
    s.text = text.substr(i, clen);
    syms.push_back(std::move(s));
    i += clen;
  }
  for (size_t k = 1; k < syms.size(); ++k) {
    syms[k - 1].next = static_cast<int>(k);
    syms[k].prev = static_cast<int>(k - 1);
  }

  struct Bigram {
    int left, right;
    float score;
    size_t size;
  };
  struct Cmp {
    bool operator()(const Bigram& l, const Bigram& r) const {
      if (l.score != r.score) return l.score < r.score;  // max-heap on score
      return l.left > r.left;                            // ties: smallest left index
    }
  };
  std::priority_queue<Bigram, std::vector<Bigram>, Cmp> queue;
  std::unordered_map<std::string, std::pair<int, int>> rev_merge;

  auto try_add = [&](int left, int right) {
    if (left < 0 || right < 0) return;
    const std::string text2 = syms[left].text + syms[right].text;
    auto it = token_to_id_.find(text2);
    if (it == token_to_id_.end()) return;
    if (it->second >= static_cast<int>(tokens_.size())) return;
    Bigram b;
    b.left = left;
    b.right = right;
    b.score = it->second < static_cast<int>(scores_.size()) ? scores_[it->second] : 0.0f;
    b.size = text2.size();
    queue.push(b);
    rev_merge[text2] = {left, right};
  };

  for (size_t k = 1; k < syms.size(); ++k) try_add(static_cast<int>(k - 1), static_cast<int>(k));

  while (!queue.empty()) {
    const Bigram b = queue.top();
    queue.pop();
    Sym& left = syms[b.left];
    Sym& right = syms[b.right];
    if (!left.alive || !right.alive) continue;
    if (left.text.size() + right.text.size() != b.size) continue;
    left.text += right.text;
    right.alive = false;
    left.next = right.next;
    if (right.next >= 0) syms[right.next].prev = b.left;
    try_add(left.prev, b.left);
    try_add(b.left, left.next);
  }

  std::vector<int> ids;
  std::vector<int> todo;
  for (int idx = 0; idx >= 0 && idx < static_cast<int>(syms.size()); idx = syms[idx].next) {
    if (!syms[idx].alive) continue;
    todo.push_back(idx);
    while (!todo.empty()) {
      const int cur = todo.back();
      todo.pop_back();
      const std::string& t = syms[cur].text;
      auto it = token_to_id_.find(t);
      if (it != token_to_id_.end()) {
        ids.push_back(it->second);
        continue;
      }
      auto rm = rev_merge.find(t);
      if (rm == rev_merge.end()) {
        // output whatever did not form a token as bytes
        for (unsigned char b : t) {
          if (byte_token_[b] >= 0) ids.push_back(byte_token_[b]);
          else if (unk_id_ >= 0) ids.push_back(unk_id_);
        }
        continue;
      }
      todo.push_back(rm->second.second);
      todo.push_back(rm->second.first);
    }
  }
  return ids;
}

// ---------------------------------------------------------------------------
// Special-token partition: match control/user-defined/unknown token texts
// (longest first) and keep them atomic.
// ---------------------------------------------------------------------------
std::vector<int> Tokenizer::encode(const std::string& text, bool add_special,
                                   bool parse_special) const {
  std::vector<int> out;
  if (add_special && add_bos_) out.push_back(bos_id_);
  if (text.empty()) {
    if (add_special && add_eos_) out.push_back(eos_id_);
    return out;
  }

  // Which specials participate: parse_special=false still isolates
  // user-defined tokens (llama.cpp tokenizer_st_partition behaviour).
  std::vector<int> specials;
  for (int id : special_ids_) {
    const int ty = token_types_[id];
    if (ty == 3 || ty == 2) {
      if (parse_special) specials.push_back(id);
    } else {
      specials.push_back(id);
    }
  }
  if (kind_ == Kind::Spm) {
    // SPM path escapes spaces before tokenizing; do the text partition first.
  }

  if (specials.empty()) {
    if (kind_ == Kind::Spm) {
      // SentencePiece runs on the whole text with metaspace escaping
      std::string escaped;
      if (add_space_prefix_) escaped += " ";
      escaped += text;
      escaped = escape_spm_spaces(escaped);
      std::vector<int> ids = spm_text(escaped);
      out.insert(out.end(), ids.begin(), ids.end());
    } else {
      std::vector<int> ids = encode_ordinary(text);
      out.insert(out.end(), ids.begin(), ids.end());
    }
    if (add_special && add_eos_) out.push_back(eos_id_);
    return out;
  }

  // Split into raw / special fragments.
  struct Piece {
    bool special = false;
    int id = -1;
    std::string text;
  };
  std::vector<Piece> frags;
  size_t pos = 0;
  while (pos < text.size()) {
    int best_id = -1;
    size_t best_pos = std::string::npos;
    size_t best_len = 0;
    for (int id : specials) {
      const std::string& tok = tokens_[id];
      if (tok.empty()) continue;
      const size_t p = text.find(tok, pos);
      if (p == std::string::npos) continue;
      if (p < best_pos || (p == best_pos && tok.size() > best_len)) {
        best_pos = p;
        best_len = tok.size();
        best_id = id;
      }
    }
    if (best_id < 0) {
      Piece p;
      p.text = text.substr(pos);
      frags.push_back(std::move(p));
      break;
    }
    if (best_pos > pos) {
      Piece p;
      p.text = text.substr(pos, best_pos - pos);
      frags.push_back(std::move(p));
    }
    Piece sp;
    sp.special = true;
    sp.id = best_id;
    frags.push_back(std::move(sp));
    pos = best_pos + best_len;
  }

  bool prev_special = true;
  for (const auto& f : frags) {
    if (f.special) {
      out.push_back(f.id);
      prev_special = true;
      continue;
    }
    if (kind_ == Kind::Spm) {
      std::string escaped = f.text;
      if (add_space_prefix_ && prev_special) escaped = " " + escaped;
      escaped = escape_spm_spaces(escaped);
      std::vector<int> ids = spm_text(escaped);
      out.insert(out.end(), ids.begin(), ids.end());
      prev_special = false;
    } else {
      std::vector<int> ids = encode_ordinary(f.text);
      out.insert(out.end(), ids.begin(), ids.end());
    }
  }
  if (add_special && add_eos_) out.push_back(eos_id_);
  return out;
}

// ---------------------------------------------------------------------------
// Decode.
// ---------------------------------------------------------------------------
std::string Tokenizer::token_piece(int id, bool skip_special) const {
  if (id < 0 || id >= static_cast<int>(tokens_.size())) return "";
  const int ty = token_types_[id];
  const bool is_control = (ty == 2 || ty == 3);
  if (is_control && skip_special) return "";
  const std::string& tok = tokens_[id];
  if (ty == 4 || ty == 3 || ty == 2) return tok;  // user-defined / control / unknown: literal
  if (ty == 6 && tok.size() == 6 && tok.compare(0, 3, "<0x") == 0) {
    auto hexv = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      return -1;
    };
    const int hi = hexv(tok[3]), lo = hexv(tok[4]);
    if (hi >= 0 && lo >= 0) return std::string(1, static_cast<char>(hi * 16 + lo));
  }
  if (kind_ == Kind::Bpe) {
    // token text is the unicode form of the raw bytes -> map back
    std::string bytes;
    bytes.reserve(tok.size());
    size_t i = 0;
    while (i < tok.size()) {
      const size_t start = i;
      const uint32_t cp = text::utf8_next(tok, i);
      auto it = cp_to_byte_.find(cp);
      if (it != cp_to_byte_.end()) {
        bytes.push_back(static_cast<char>(it->second));
      } else {
        bytes.append(tok, start, i - start);  // keep the raw UTF-8
      }
    }
    return bytes;
  }
  // SPM: ▁ -> space
  std::string s = tok;
  const std::string kSpaceMark = "\xE2\x96\x81";  // U+2581 LOWER ONE EIGHTH BLOCK
  size_t p = 0;
  while ((p = s.find(kSpaceMark, p)) != std::string::npos) {
    s.replace(p, kSpaceMark.size(), " ");
    p += 1;
  }
  return s;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
  std::string bytes;
  for (int id : ids) bytes += token_piece(id, skip_special);
  return text::decode_utf8_lossy(bytes);
}

int Tokenizer::token_id(const std::string& text) const {
  auto it = token_to_id_.find(text);
  return it == token_to_id_.end() ? -1 : it->second;
}

const std::string& Tokenizer::token_text(int id) const {
  static const std::string kEmpty;
  if (id < 0 || id >= static_cast<int>(tokens_.size())) return kEmpty;
  return tokens_[id];
}

int Tokenizer::token_type(int id) const {
  if (id < 0 || id >= static_cast<int>(token_types_.size())) return 0;
  return token_types_[id];
}

bool Tokenizer::is_special(int id) const {
  const int ty = token_type(id);
  return ty == 2 || ty == 3;
}

bool Tokenizer::is_eog(int id) const {
  if (id == eos_id_) return true;
  if (token_type(id) != 3) return false;
  const std::string& t = token_text(id);
  static const char* kEogNames[] = {
      "<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end|>", "</s>",
      "<|endofprompt|>", "<|end▁of▁sentence|>", "<|eot|>", "<|return|>",
  };
  for (const char* n : kEogNames)
    if (t == n) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Chat template: Qwen ChatML (tokenizer.chat_template family).
// Renders what the jinja template in the GGUF produces for text-only messages;
// tool/vision support is intentionally out of scope (documented limitation).
// ---------------------------------------------------------------------------
std::string Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages,
                                           bool add_generation_prompt, bool thinking) const {
  if (template_kind_ != Template::Chatml) return "";
  const std::string im_start = "<|im_start|>";
  const std::string im_end = "<|im_end|>";
  const std::string kThinkOpen = "<think>";
  const std::string kThinkClose = "</think>";
  const bool thinking_aware = template_has(chat_template_, "enable_thinking");

  std::string out;
  size_t num_sys = 0;
  std::string merged_system;
  if (!messages.empty() &&
      (messages[0].role == "system" || messages[0].role == "developer")) {
    merged_system = trim_copy(messages[0].content);
    if (messages.size() > 1 &&
        (messages[1].role == "system" || messages[1].role == "developer")) {
      merged_system += "\n" + trim_copy(messages[1].content);
      num_sys = 2;
    } else {
      num_sys = 1;
    }
  }
  if (!merged_system.empty()) {
    out += im_start + "system\n" + merged_system + im_end + "\n";
  }

  // index of the last "real" user query (pure tool responses do not reset it);
  // the jinja template initialises this to messages|length - 1.
  long last_query_index = static_cast<long>(messages.size()) - 1;
  for (long i = static_cast<long>(messages.size()) - 1; i >= 0; --i) {
    if (messages[i].role != "user") continue;
    const std::string c = trim_copy(messages[i].content);
    const bool tool_response = c.rfind("<tool_response>", 0) == 0 && c.size() >= 16 &&
                               c.compare(c.size() - 16, 16, "</tool_response>") == 0;
    if (!tool_response) {
      last_query_index = i;
      break;
    }
  }

  for (size_t i = num_sys; i < messages.size(); ++i) {
    const ChatMessage& m = messages[i];
    if (m.role == "system" || m.role == "developer") continue;  // merged above
    const std::string content = trim_copy(m.content);

    if (m.role == "user") {
      out += im_start + "user\n" + content + im_end + "\n";
      continue;
    }
    if (m.role == "assistant") {
      std::string reasoning = m.reasoning_content;
      std::string body = content;
      if (reasoning.empty()) {
        const size_t close = body.find(kThinkClose);
        if (close != std::string::npos) {
          std::string head = body.substr(0, close);
          const size_t open = head.rfind(kThinkOpen);
          if (open != std::string::npos) head = head.substr(open + kThinkOpen.size());
          reasoning = head;
          std::string tail = body.substr(close + kThinkClose.size());
          body = tail;
        }
      }
      reasoning = trim_copy(reasoning);
      body = trim_copy(body);
      out += im_start + "assistant\n";
      const bool current_turn = static_cast<long>(i) > last_query_index;
      if (!reasoning.empty() && current_turn) {
        out += kThinkOpen + "\n" + reasoning + "\n" + kThinkClose + "\n\n" + body;
      } else {
        out += body;
      }
      out += im_end + "\n";
      continue;
    }
    if (m.role == "tool") {
      // Qwen tool turns: a run of consecutive tool messages shares one user
      // turn and only the last one closes it (matches the jinja template).
      const bool prev_tool = i > 0 && messages[i - 1].role == "tool";
      if (!prev_tool) out += im_start + "user";
      out += "\n<tool_response>\n" + content + "\n</tool_response>";
      const bool next_tool = i + 1 < messages.size() && messages[i + 1].role == "tool";
      if (!next_tool) out += im_end + "\n";
      continue;
    }
    // unknown role: render as user (defensive)
    out += im_start + "user\n" + content + im_end + "\n";
  }

  if (add_generation_prompt) {
    out += im_start + "assistant\n";
    if (thinking_aware) {
      if (thinking) {
        out += kThinkOpen + "\n";
      } else {
        out += kThinkOpen + "\n\n" + kThinkClose + "\n\n";
      }
    }
  }
  return out;
}

}  // namespace tokenizer
}  // namespace engine
