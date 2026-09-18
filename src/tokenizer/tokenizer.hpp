#pragma once
// GLM-5.3 byte-level BPE tokenizer (tokenizer.json) + chat template rendering.
#include "core/chat.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace helios {

class Tokenizer {
public:
  // dir: model directory containing tokenizer.json
  bool load(const std::string& dir);
  int vocab_size() const { return (int)id_to_tok_.size(); }
  int eos_id() const { return eos_; }

  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;
  // keep_special: keep special/added tokens (the tool-call XML tags are special tokens in this
  // checkpoint, so the server needs them to parse <tool_call> blocks out of the output).
  std::string decode(const std::vector<int>& ids, bool keep_special) const;

  // GLM chat template: [gMASK]<sop> + per-message turns, ending with the generation prompt.
  // reasoning_effort: "low"|"high"|"max" (default max, prepended as a system turn).
  std::string apply_chat_template(const std::vector<ChatMsg>& msgs,
                                  bool add_generation_prompt = true,
                                  const std::string& reasoning_effort = "max") const;

  // special token id lookup (-1 if unknown)
  int special_id(const std::string& s) const;

private:
  // byte <-> byte-level-unicode symbol (GPT-2 mapping)
  static const std::string& byte_to_sym();
  // pretokenizer split (GPT-4 style pattern), returns byte ranges
  void pretokenize(const std::string& text, std::vector<std::pair<size_t, size_t>>& out) const;
  std::vector<int> bpe_piece(const std::string& piece) const;

  std::unordered_map<std::string, int> tok_to_id_;
  std::vector<std::string> id_to_tok_;
  std::unordered_map<std::string, int> merge_rank_;   // "a\x01b" -> merge rank
  std::vector<std::pair<std::string, int>> added_; // special tokens (sorted by length desc)
  std::unordered_map<std::string, int> added_map_;
  int eos_ = 0;
};

}  // namespace helios