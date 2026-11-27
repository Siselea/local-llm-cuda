// GGUF tokenizer: byte-level BPE (tokenizer.ggml.model == "gpt2") and
// SentencePiece (== "llama").  No external dependencies; the Unicode
// general-category tables are generated into include/engine/unicode_tables.h.
//
// Supported pre-tokenizers (tokenizer.ggml.pre):
//   qwen35                      exact port of llama.cpp b12128 QWEN35 splitter
//   qwen2 / deepseek-r1-qwen    exact port of the QWEN2 splitter
//   gpt2 / unknown              canonical GPT-2 splitter (best effort)
//
// Special tokens (control / user-defined / unknown) can be kept atomic during
// encode(), which is what the chat template needs.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine {
namespace tokenizer {

enum class Kind { Bpe, Spm };

// One chat message.  content is plain text; reasoning_content is the optional
// Qwen-style "thinking" part (server-side clients may also pass an assistant
// content that already contains <｜end▁of▁thinking｜> - it is split automatically).
struct ChatMessage {
  std::string role;
  std::string content;
  std::string reasoning_content;
};

// Work budget for encode_bounded().  max_chars == 0 disables the input-size check,
// max_ms == 0 disables the wall-clock check.  Violations throw std::runtime_error so
// network-facing callers can map them to HTTP 400.
struct EncodeBudget {
  size_t max_chars = 0;
  int max_ms = 0;
  std::chrono::steady_clock::time_point deadline{};
};

class Tokenizer {
 public:
  Tokenizer() = default;

  // Reads the tokenizer metadata straight from the GGUF file.  Throws
  // std::runtime_error when the file has no usable tokenizer.
  static Tokenizer load(const std::string& gguf_path);

  bool valid() const { return !tokens_.empty(); }
  Kind kind() const { return kind_; }
  const std::string& pre() const { return pre_; }
  int vocab_size() const { return static_cast<int>(tokens_.size()); }
  int bos_id() const { return bos_id_; }
  int eos_id() const { return eos_id_; }
  int unk_id() const { return unk_id_; }
  bool add_bos() const { return add_bos_; }
  bool add_eos() const { return add_eos_; }
  bool has_chat_template() const { return !chat_template_.empty(); }
  const std::string& chat_template() const { return chat_template_; }
  // "chatml" when the GGUF template is the Qwen ChatML family and
  // apply_chat_template() can render it, "none" otherwise.
  const char* template_name() const;

  // ---- encode / decode -----------------------------------------------------
  // add_special: prepend BOS (SPM default) / append EOS per GGUF metadata
  // parse_special: keep special tokens (control/user-defined/unknown) atomic
  std::vector<int> encode(const std::string& text, bool add_special = false,
                          bool parse_special = true) const;
  // Bounded-work variant for network-facing callers (serve).  Rejects inputs above
  // max_chars UTF-8 codepoints and aborts a single encode that runs longer than
  // max_ms (0 disables either check).  Throws std::runtime_error on violation.
  std::vector<int> encode_bounded(const std::string& text, bool add_special,
                                  bool parse_special, size_t max_chars,
                                  int max_ms) const;

  // Detokenize.  skip_special=true drops CONTROL/UNKNOWN tokens, like
  // llama.cpp's token_to_piece(special=false); user-defined tokens are kept.
  std::string decode(const std::vector<int>& ids, bool skip_special = true) const;
  // One token as bytes; empty when skipped.
  std::string token_piece(int id, bool skip_special = true) const;

  int token_id(const std::string& text) const;
  const std::string& token_text(int id) const;
  int token_type(int id) const;   // GGUF token_type (1 normal, 2 unknown, 3 control,
                                  // 4 user-defined, 5 unused)
  bool is_special(int id) const;  // control | unknown (llama.cpp attr_special)
  bool is_eog(int id) const;      // eos or one of the end-of-generation tokens

  // ---- chat template (Qwen ChatML) ----------------------------------------
  // Renders the conversation to the exact text the model expects, including the
  // <|im_start|>assistant\n generation prompt.  `thinking` mirrors the
  // template's enable_thinking kwarg: true -> opens "<think>\n" (the model then
  // reasons), false -> emits the empty-think prefill "<think>\n\n</think>\n\n".
  // Returns an empty string when the template is unsupported.
  std::string apply_chat_template(const std::vector<ChatMessage>& messages,
                                  bool add_generation_prompt = true,
                                  bool thinking = true) const;

  // Debug helper: pre-tokenizer pieces for a text (UTF-8 substrings).
  std::vector<std::string> pre_tokenize(const std::string& text) const;

 private:
  enum class Pre { Qwen35, Qwen2, Gpt2 };
  enum class Template { None, Chatml };

  std::vector<std::string> pre_split(const std::string& text) const;
  std::vector<int> encode_impl(const std::string& text, bool add_special,
                               bool parse_special, const EncodeBudget* budget) const;
  std::vector<int> encode_ordinary(const std::string& text,
                                   const EncodeBudget* budget) const;
  std::vector<int> bpe_word(const std::string& word, const EncodeBudget* budget) const;
  std::vector<int> spm_text(const std::string& text, const EncodeBudget* budget) const;

  Kind kind_ = Kind::Bpe;
  Pre pre_kind_ = Pre::Qwen2;
  Template template_kind_ = Template::None;
  std::string pre_;
  std::string chat_template_;
  std::vector<std::string> tokens_;
  std::vector<int> token_types_;
  std::vector<float> scores_;
  std::vector<int> special_ids_;  // control/user-defined/unknown, sorted by length desc
  std::unordered_map<std::string, int> token_to_id_;
  // BPE merge ranks, keyed by "left right" (token strings never contain ' ').
  std::unordered_map<std::string, int> merge_ranks_;
  // Byte-level BPE alphabet: raw byte -> unicode representation.
  std::string byte_to_utf8_[256];
  std::unordered_map<uint32_t, uint8_t> cp_to_byte_;
  // SPM byte fallback: byte -> <0xNN> token id (-1 = none).
  int byte_token_[256];
  int bos_id_ = -1, eos_id_ = -1, unk_id_ = -1;
  bool add_bos_ = false, add_eos_ = false;
  bool add_space_prefix_ = false;  // SPM: prepend ▁ after BOS / special tokens
  // Small per-word encode cache (system prompts repeat every request).
  mutable std::unordered_map<std::string, std::vector<int>> word_cache_;
};

}  // namespace tokenizer
}  // namespace engine
