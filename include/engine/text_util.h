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

// Decode one codepoint at s[i]; returns the codepoint and advances i.
// Invalid bytes are returned as their byte value (0..255) and advance by one,
// which keeps invalid input round-trippable instead of throwing.
inline uint32_t utf8_next(const std::string& s, size_t& i) {
  const unsigned char c0 = static_cast<unsigned char>(s[i]);
  if (c0 < 0x80) {
    ++i;
    return c0;
  }
  const size_t len = utf8_seq_len(c0);
  if (len == 1 || i + len > s.size()) {
    ++i;
    return c0;
  }
  uint32_t cp = c0 & (len == 2 ? 0x1F : len == 3 ? 0x0F : 0x07);
  for (size_t k = 1; k < len; ++k) {
    const unsigned char c = static_cast<unsigned char>(s[i + k]);
    if ((c & 0xC0) != 0x80) {
      ++i;
      return c0;
    }
    cp = (cp << 6) | (c & 0x3F);
  }
  i += len;
  return cp;
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
// `replacement` is appended for invalid bytes.
inline std::string decode_utf8_lossy(const std::string& bytes) {
  std::string out;
  out.reserve(bytes.size());
  size_t i = 0;
  while (i < bytes.size()) {
    const size_t start = i;
    const unsigned char c = static_cast<unsigned char>(bytes[i]);
    const size_t len = utf8_seq_len(c);
    if (len == 1 && c >= 0x80) {
      out.push_back(static_cast<char>(0xEF));  // U+FFFD
      out.append("\xBF\xBD");
      ++i;
      continue;
    }
    if (i + len > bytes.size()) {
      out.append("\xEF\xBF\xBD");
      break;
    }
    bool ok = true;
    for (size_t k = 1; k < len; ++k) {
      if ((static_cast<unsigned char>(bytes[i + k]) & 0xC0) != 0x80) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      out.append("\xEF\xBF\xBD");
      i = start + 1;
      continue;
    }
    out.append(bytes, i, len);
    i += len;
  }
  return out;
}

}  // namespace text
}  // namespace engine
