// Degenerate-loop detector for `engine serve`.
//
// The 35B Qwen3.x abliterated GGUF can fall into loops that the user sees as
// "思考/回答复读":
//   * whole-sentence loops        "老城的秘密，是老人。" x28
//   * character-level loops       "--你--你--...--134--134--"
//   * variant loops               same phrase with tiny changes / separators
//
// This class is header-only and side-effect free so it can be unit-tested without
// CUDA/model (tests/test_serve_guard.cpp):
//   level 0 = off
//   level 1 = sentence level only (3 byte-identical sentences in a row)
//   level 2 = + character level (default)
//        (a) a 4..96-byte unit repeated 5x back to back, or a >=12-byte unit
//            repeated 4x (short-sentence loops)
//        (b) answer-sentence repetition:
//            * 3 consecutive sentences identical after normalising digits/ASCII letters;
//            * a repeated cycle of 2..8 such sentences (interleaved refrain);
//            * a CJK sentence (>=36 normalised bytes) appearing >=3 times within the
//              last 30 sentences - this catches interleaved loops early (the user
//              sees 3 repeats instead of 8+) without flagging Python code such as
//              `return False` (pure-ASCII sentences are exempt from this rule).
//            Thinking text is exempt from (b).
#pragma once

#include <algorithm>
#include <cstring>
#include <deque>
#include <string>

namespace engine {
namespace serve {

class RepeatGuard {
 public:
  void configure(int level) { level_ = std::max(0, std::min(2, level)); }

  int level() const { return level_; }

  // Feed newly emitted text; returns true exactly once when the guard fires.
  // `is_reasoning` = the text belongs to the open  thinking block.
  bool update(const std::string& text, bool is_reasoning) {
    if (level_ <= 0 || fired_) return false;
    if (level_ >= 1 && sentence_check(text, level_ >= 2 && !is_reasoning))
      return fire("sentence-3x");
    if (level_ >= 2 && !is_reasoning && cycle_check()) return fire("sentence-cycle");
    if (level_ >= 2 && !is_reasoning && duplicate_sentence_check())
      return fire("sentence-duplicate-3x");
    if (level_ < 2) return false;
    guard_win_ += text;
    if (guard_win_.size() > 1024) guard_win_.erase(0, guard_win_.size() - 1024);
    if (unit_loop_check()) return fire("unit-5x");
    return false;
  }

  bool fired() const { return fired_; }

  // Diagnostics: which rule fired and a snippet of the window that triggered it.
  const std::string& rule() const { return rule_; }
  std::string snippet() const {
    const size_t n = guard_win_.size();
    const size_t k = std::min<size_t>(80, n);
    return n ? guard_win_.substr(n - k) : std::string();
  }

  void reset() {
    fired_ = false;
    rule_.clear();
    sent_buf_.clear();
    prev_sent_.clear();
    guard_win_.clear();
    recent_sents_.clear();
    sent_run_ = 0;
  }

 private:
  bool fire(const char* rule) {
    fired_ = true;
    rule_ = rule;
    return true;
  }

  // True when a sentence is predominantly CJK (>= half of its bytes are non-ASCII).
  // Code lines that merely *contain* a short Chinese label (e.g. print(f"1 是素数: ..."))
  // are NOT normalised, otherwise their ASCII parts would be stripped and three
  // different code lines would look identical.
  static bool cjk_sentence(const std::string& s) {
    if (s.size() < 12) return false;
    size_t cjk = 0;
    for (unsigned char c : s)
      if (c >= 0x80) ++cjk;
    return cjk * 2 >= s.size();
  }

  // Normalise a sentence for the level-2 variant-loop check.
  //   * ASCII digits are always dropped (list/loop indices differ);
  //   * ASCII letters are dropped ONLY when the sentence contains non-ASCII (CJK)
  //     text, so "岁月的秘密，是老人A。" variants collapse while a pure-ASCII code
  //     block such as `print(is_prime(17)) # True` / `... # False` stays distinct.
  static std::string normalize_sentence(const std::string& s) {
    bool has_nonascii = false;
    for (unsigned char c : s)
      if (c >= 0x80) { has_nonascii = true; break; }
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
      const bool digit = c >= '0' && c <= '9';
      const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
      const bool space = c == ' ' || c == '\t' || c == '\r';
      if (digit) continue;
      if (alpha && has_nonascii) continue;
      if (space) continue;
      out.push_back(static_cast<char>(c));
    }
    return out;
  }

  // ---- level 1 (exact) / level 2 (normalised): 3 sentences in a row ----
  bool sentence_check(const std::string& text, bool normalized) {
    sent_buf_ += text;
    while (true) {
      size_t cut = std::string::npos;
      for (size_t i = 0; i < sent_buf_.size(); ++i) {
        const char c = sent_buf_[i];
        if (static_cast<unsigned char>(c) >= 0x80) {
          // CJK sentence terminators are 3-byte UTF-8 sequences
          if (i + 3 <= sent_buf_.size()) {
            const std::string t3 = sent_buf_.substr(i, 3);
            if (t3 == "\xE3\x80\x82" /* 。 */ || t3 == "\xEF\xBC\x81" /* ！ */ ||
                t3 == "\xEF\xBC\x9F" /* ？ */ || t3 == "\xEF\xBC\x9B" /* ； */) {
              cut = i + 3;
              break;
            }
          }
          continue;
        }
        if (c == '\n') {
          cut = i + 1;
          break;
        }
        // ASCII terminators need a following space/newline in the same chunk
        if ((c == '.' || c == '!' || c == '?') && i + 1 < sent_buf_.size() &&
            (sent_buf_[i + 1] == ' ' || sent_buf_[i + 1] == '\n')) {
          cut = i + 1;
          break;
        }
      }
      if (cut == std::string::npos) break;
      std::string sent = sent_buf_.substr(0, cut);
      sent_buf_.erase(0, cut);
      size_t b = 0, e = sent.size();
      while (b < e && (sent[b] == ' ' || sent[b] == '\n' || sent[b] == '\r' ||
                       sent[b] == '\t'))
        ++b;
      while (e > b && (sent[e - 1] == ' ' || sent[e - 1] == '\n' || sent[e - 1] == '\r' ||
                       sent[e - 1] == '\t'))
        --e;
      sent = sent.substr(b, e - b);
      const bool norm_this = normalized && cjk_sentence(sent);
      const std::string key = norm_this ? normalize_sentence(sent) : sent;
      const size_t kMinLen = norm_this ? 6 : 8;  // normalised text is shorter
      if (key.size() < kMinLen) {
        prev_sent_.clear();
        sent_run_ = 0;
        continue;
      }
      if (normalized) {
        recent_sents_.push_back(key);
        if (recent_sents_.size() > 30) recent_sents_.pop_front();
      }
      if (key == prev_sent_) {
        ++sent_run_;
      } else {
        prev_sent_ = key;
        sent_run_ = 1;
      }
      if (sent_run_ >= 3) return true;
    }
    return false;
  }

  static bool has_cjk(const std::string& s) {
    for (unsigned char c : s)
      if (c >= 0x80) return true;
    return false;
  }

  // ---- level 2b: same CJK sentence >= 3 times within the last 30 sentences ----
  bool duplicate_sentence_check() const {
    const size_t n = recent_sents_.size();
    if (n < 3) return false;
    const std::string& cur = recent_sents_.back();
    if (cur.size() < 36 || !has_cjk(cur)) return false;  // >=12 CJK chars: loops are long sentences
    int count = 0;
    for (const auto& k : recent_sents_)
      if (k == cur) ++count;
    return count >= 3;
  }

  // ---- level 2c: repeated sentence cycle (period 2..8, three cycles) ----
  bool cycle_check() const {
    const size_t n = recent_sents_.size();
    for (size_t p = 2; p <= 8; ++p) {
      if (n < 3 * p) continue;
      bool same = true;
      size_t cjk_keys = 0;
      for (size_t i = 0; i < p && same; ++i) {
        const std::string& a = recent_sents_[n - 3 * p + i];
        const std::string& b = recent_sents_[n - 2 * p + i];
        const std::string& c = recent_sents_[n - p + i];
        if (a != b || b != c) same = false;
        // Only Chinese (non-ASCII) keys count: this keeps a legitimate ASCII
        // `print(is_prime(17)) # True` / `... # False` test list (whose digits are
        // normalised away, forming a 2-cycle) from being flagged.
        for (unsigned char ch : a)
          if (ch >= 0x80) { ++cjk_keys; break; }
      }
      if (same && cjk_keys >= p / 2 + 1) return true;
    }
    return false;
  }

  // ---- level 2a: a short unit repeated back to back ----
  //   * any 4..96-byte unit repeated 5x  (char loops like "--134--134...")
  //   * a >=12-byte unit repeated 4x     (word/short-sentence loops like
  //                                       "他找到了自己。" x4: stop earlier so the user
  //                                       does not see 5 identical lines)
  // A unit only counts as a loop if it carries real content: at least 2 bytes are
  // CJK ideographs (UTF-8 lead 0xE4..0xE9) or ASCII alphanumerics.  This keeps
  // decorative lines like "================..." / "————————" from matching.
  static bool unit_informative(const char* base, size_t p) {
    int info = 0;
    for (size_t i = 0; i < p; ++i) {
      const unsigned char b = static_cast<unsigned char>(base[i]);
      if ((b >= 0xE4 && b <= 0xE9) || b == 0xF0) ++info;
      else if ((b >= '0' && b <= '9') || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z'))
        ++info;
      if (info >= 2) return true;
    }
    return false;
  }

  bool unit_loop_check() const {
    const size_t n = guard_win_.size();
    size_t maxp = std::min<size_t>(96, n / 5);
    for (size_t p = 4; p <= maxp; ++p) {
      const char* base = guard_win_.data() + n - 5 * p;
      if (!unit_informative(base, p)) continue;
      if (std::memcmp(base, base + p, p) == 0 &&
          std::memcmp(base, base + 2 * p, p) == 0 &&
          std::memcmp(base, base + 3 * p, p) == 0 &&
          std::memcmp(base, base + 4 * p, p) == 0)
        return true;
    }
    maxp = std::min<size_t>(96, n / 4);
    for (size_t p = 12; p <= maxp; ++p) {
      const char* base = guard_win_.data() + n - 4 * p;
      if (!unit_informative(base, p)) continue;
      if (std::memcmp(base, base + p, p) == 0 &&
          std::memcmp(base, base + 2 * p, p) == 0 &&
          std::memcmp(base, base + 3 * p, p) == 0)
        return true;
    }
    return false;
  }

  // ---- (removed: trigram diversity rule; normalised sentences replace it) ----

  int level_ = 2;
  bool fired_ = false;
  std::string rule_;
  std::string sent_buf_;
  std::string prev_sent_;
  std::string guard_win_;
  std::deque<std::string> recent_sents_;
  int sent_run_ = 0;
};

}  // namespace serve
}  // namespace engine
