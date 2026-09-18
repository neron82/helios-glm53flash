#pragma once
// Chat template rendering and output parsing for the OpenAI-compatible server.
//
// The renderer reproduces glm53flash/chat_template.jinja byte-for-byte for the message shapes the
// server accepts (system/user/assistant/tool, optional tools, reasoning_effort). It was validated
// against transformers' apply_chat_template output for: plain chat, tools, a tool-call round trip,
// and multi-turn reasoning clearing.
#include <string>
#include <vector>

#include "json.hpp"

// Ordered so rendered JSON matches the reference template's insertion order (nlohmann's plain json
// sorts object keys).
using helios_json = nlohmann::ordered_json;

namespace helios {

struct ToolCall {
  std::string id;          // call_<n> as emitted to clients
  std::string name;
  std::string arguments;   // JSON object as a string, e.g. {"city": "Paris"}
};

struct ChatMsg {
  std::string role;                        // system | user | assistant | tool
  std::string content;
  std::string reasoning;                   // assistant reasoning_content
  std::vector<ToolCall> tool_calls;        // assistant
  std::string tool_call_id;                // tool
};

// Render the prompt. `tools` are the raw OpenAI tool objects ({"type":"function","function":{...}}).
// `thinking`: when false the prompt ends with a closed empty think block (<|assistant|><think>
// </think>), which is exactly how historical assistant turns are rendered, so the model answers
// directly instead of reasoning first. Agents that do not want a long hidden reasoning phase set
// chat_template_kwargs.enable_thinking = false (or the GLM-style "thinking": false).
std::string render_chat(const std::vector<ChatMsg>& msgs, const helios_json& tools,
                        const std::string& reasoning_effort, bool add_generation_prompt,
                        bool thinking = true);

// Parsed model output.
struct ParsedOutput {
  std::string reasoning;
  std::string content;
  std::vector<ToolCall> tool_calls;
};

// Streaming parser: feed decoded text pieces, receive the deltas that are safe to emit.
class OutputParser {
public:
  // start_in_think: the generation prompt opens <think>, so generated text is reasoning until the
  // model closes it. With thinking disabled the prompt carries a closed empty block and the model's
  // first tokens are already the answer.
  explicit OutputParser(bool start_in_think = true) : in_think_(start_in_think) {}
  struct Delta {
    std::string reasoning;                    // new reasoning text (may be empty)
    std::string content;                      // new visible content (may be empty)
    bool tool_begin = false;                  // start of a tool call (name is set)
    std::string tool_name;
    std::string tool_args_fragment;           // arguments JSON fragment for the active tool call
    bool stop = false;                        // a turn boundary was reached: end generation
  };

  std::vector<Delta> feed(const std::string& piece);
  std::vector<Delta> finish();                // flush anything buffered (call before [DONE])
  const ParsedOutput& parsed() const { return out_; }
  bool in_think() const { return in_think_; }

private:
  void drain(bool flush);

  std::string buf_;
  bool in_think_ = true;
  bool started_in_think_ = true;
  bool in_tool_ = false;
  std::string tool_body_;
  ParsedOutput out_;
  std::vector<Delta> pending_;
};

// Parse a complete output string (non-streaming path).
ParsedOutput parse_output(const std::string& text, bool start_in_think = true);

}  // namespace helios
