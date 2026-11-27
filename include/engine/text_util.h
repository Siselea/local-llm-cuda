// UTF-8 helpers shared by the tokenizer and the HTTP server.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine {
namespace text {

// Length of the UTF-8 sequence starting at s[i] according to the leading byte
// (1..4).  Invalid leading bytes report 1 so callers can resynchronise.
inline size_t utf8_seq_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c & 0xE0) == 0xC0) return 2;
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xF8) == 0xF0) return 4;
  return 1;
}

// Strict UTF-8 codepoint decode: returns false for overlong sequences, UTF-16
// surrogate half-pairs (ED A0 80..), codepoints above U+10FFFF (F5..), truncated
// sequences and bad continuation bytes.  On failure `i` advances by one byte so a
// caller can resynchronise; *cp is set to U+FFFD.
inline bool utf8_decode_strict(const std::string& s, size_t& i, uint32_t* cp) {
  if (cp) *cp = 0xFFFD;
  if (i >= s.size()) return false;
  const unsigned char c0 = static_cast<unsigned char>(s[i]);
  if (c0 < 0x80) {
    if (cp) *cp = c0;
    ++i;
    return true;
  }
  const size_t len = utf8_seq_len(c0);
  if (len == 1 || i + len > s.size()) {
    ++i;
    return false;
  }
  uint32_t v = c0 & (len == 2 ? 0x1F : len == 3 ? 0x0F : 0x07);
  for (size_t k = 1; k < len; ++k) {
    const unsigned char c = static_cast<unsigned char>(s[i + k]);
    if ((c & 0xC0) != 0x80) {
      ++i;
      return false;
    }
    v = (v << 6) | (c & 0x3F);
  }
  const uint32_t min_cp = (len == 2) ? 0x80u : (len == 3) ? 0x800u : 0x10000u;
  if (v < min_cp || v > 0x10FFFFu || (v >= 0xD800u && v <= 0xDFFFu)) {
    ++i;
    return false;
  }
  if (cp) *cp = v;
  i += len;
  return true;
}

// Decode one codepoint at s[i]; invalid input yields U+FFFD and advances by one
// byte (strict validation: overlong forms, surrogate halves and codepoints above
// U+10FFFF are rejected instead of being passed through).
inline uint32_t utf8_next(const std::string& s, size_t& i) {
  uint32_t cp = 0xFFFD;
  utf8_decode_strict(s, i, &cp);
  return cp;
}

// True when the whole string is well-formed UTF-8.
inline bool is_valid_utf8(const std::string& s) {
  size_t i = 0;
  while (i < s.size()) {
    uint32_t cp = 0;
    if (!utf8_decode_strict(s, i, &cp)) return false;
  }
  return true;
}

// Number of UTF-8 codepoints, counting at most `cap` (request budgets use it so a
// 32 MiB body is never scanned twice).
inline size_t count_utf8_codepoints(const std::string& s,
                                    size_t cap = static_cast<size_t>(-1)) {
  size_t n = 0;
  size_t i = 0;
  while (i < s.size() && n < cap) {
    uint32_t cp = 0;
    if (!utf8_decode_strict(s, i, &cp)) {
      ++n;
      continue;
    }
    ++n;
  }
  return n;
}

inline void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

inline std::string from_codepoint(uint32_t cp) {
  std::string s;
  append_utf8(s, cp);
  return s;
}

// Split into codepoints, recording the byte offset of every codepoint in
// `byte_offsets` (size = #codepoints + 1, last = s.size()).
inline std::vector<uint32_t> to_codepoints(const std::string& s,
                                           std::vector<size_t>* byte_offsets = nullptr) {
  std::vector<uint32_t> cpts;
  cpts.reserve(s.size());
  if (byte_offsets) {
    byte_offsets->clear();
    byte_offsets->reserve(s.size() + 1);
  }
  size_t i = 0;
  while (i < s.size()) {
    if (byte_offsets) byte_offsets->push_back(i);
    cpts.push_back(utf8_next(s, i));
  }
  if (byte_offsets) byte_offsets->push_back(s.size());
  return cpts;
}

// Number of trailing bytes that form an incomplete UTF-8 sequence (0 when the
// string ends exactly on a codepoint boundary).  Used to hold back the tail of
// a streamed token until the multi-byte character is complete.
inline size_t incomplete_utf8_tail(const std::string& s) {
  if (s.empty()) return 0;
  // at most 3 bytes back
  const size_t max_back = s.size() < 3 ? s.size() : 3;
  for (size_t back = 1; back <= max_back; ++back) {
    const unsigned char c = static_cast<unsigned char>(s[s.size() - back]);
    if ((c & 0xC0) == 0x80) continue;  // continuation byte, keep looking back
    const size_t need = utf8_seq_len(c);
    if (need == 1) return 0;          // ASCII byte: previous string is complete
    return (back < need) ? back : 0;  // incomplete prefix of a sequence
  }
  return 0;
}

// Decode a byte string that may end in the middle of a multi-byte sequence.
// Invalid encodings (truncated / overlong / surrogate / > U+10FFFF) are replaced
// with U+FFFD, matching json::Writer::quote's sanitisation.
inline std::string decode_utf8_lossy(const std::string& bytes) {
  std::string out;
  out.reserve(bytes.size());
  size_t i = 0;
  while (i < bytes.size()) {
    const size_t start = i;
    uint32_t cp = 0;
    if (utf8_decode_strict(bytes, i, &cp)) {
      out.append(bytes, start, i - start);
      continue;
    }
    out.append("\xEF\xBF\xBD");
    // Consume the maximal structurally-complete prefix of the malformed sequence
    // (lead byte + following continuation bytes) so one bad character becomes one
    // replacement char, matching json::Writer::quote.
    const size_t lead_len = utf8_seq_len(static_cast<unsigned char>(bytes[start]));
    size_t adv = 1;
    while (adv < lead_len && start + adv < bytes.size() &&
           (static_cast<unsigned char>(bytes[start + adv]) & 0xC0) == 0x80)
      ++adv;
    i = start + adv;
  }
  return out;
}

}  // namespace text
}  // namespace engine
