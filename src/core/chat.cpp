// Chat template rendering + output parsing. See chat.hpp for the validation notes.
#include "core/chat.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace helios {

using json = nlohmann::ordered_json;

namespace {

// json.dumps(&, separators=(', ', ': ')) - what jinja's tojson produces and what the reference
// renders contain. nlohmann's dump() is compact, so rebuild the string with those separators while
// escaping scalars through nlohmann (identical escaping rules).
void dump_pretty(const json& j, std::string& out) {
  if (j.is_object()) {
    out += '{';
    bool first = true;
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (!first) out += ", ";
      first = false;
      out += json(it.key()).dump();
      out += ": ";
      dump_pretty(it.value(), out);
    }
    out += '}';
  } else if (j.is_array()) {
    out += '[';
    bool first = true;
    for (const auto& v : j) {
      if (!first) out += ", ";
      first = false;
      dump_pretty(v, out);
    }
    out += ']';
  } else {
    out += j.dump();
  }
}

std::string to_json_like_jinja(const json& j) {
  std::string s;
  dump_pretty(j, s);
  return s;
}

// A tool object may be {"type":"function","function":{...}} or the function object directly.
json tool_function(const json& tool) {
  if (tool.is_object() && tool.contains("function")) return tool["function"];
  return tool;
}

std::string cap_first(const std::string& s) {
  std::string c = s;
  if (!c.empty()) c[0] = (char)toupper((unsigned char)c[0]);
  return c;
}

std::string strip_ws(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// Argument values: strings go in raw, everything else as JSON (matches the reference renders).
void append_args(const json& args, std::string& out) {
  if (!args.is_object()) {
    if (!args.is_null()) out += to_json_like_jinja(args);
    return;
  }
  for (auto it = args.begin(); it != args.end(); ++it) {
    out += "<arg_key>" + it.key() + "</arg_key><arg_value>";
    if (it.value().is_string()) out += it.value().get<std::string>();
    else out += to_json_like_jinja(it.value());
    out += "</arg_value>";
  }
}

}  // namespace

std::string render_chat(const std::vector<ChatMsg>& msgs, const json& tools,
                        const std::string& reasoning_effort, bool add_generation_prompt,
                        bool thinking) {
  std::string out = "[gMASK]<sop>";
  std::string eff = (reasoning_effort == "low" || reasoning_effort == "high") ? reasoning_effort
                                                                             : "max";
  out += "<|system|>Reasoning Effort: " + cap_first(eff);

  // Tools block (only when tools were supplied and any survives the defer_loading filter).
  if (tools.is_array() && !tools.empty()) {
    std::vector<json> shown;
    for (const auto& t : tools) {
      json fn = tool_function(t);
      if (fn.contains("defer_loading") && fn["defer_loading"].is_boolean() &&
          fn["defer_loading"].get<bool>())
        continue;
      shown.push_back(fn);
    }
    if (!shown.empty()) {
      out += "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user "
             "query.\n\nYou are provided with function signatures within <tools></tools> XML "
             "tags:\n<tools>\n";
      for (const json& fn : shown) {
        json clean = json::object();
        for (auto it = fn.begin(); it != fn.end(); ++it) {
          if (it.key() == "strict") continue;
          clean[it.key()] = it.value();
        }
        out += to_json_like_jinja(clean);
        out += "\n";
      }
      out += "</tools>\n\nFor each function call, output the function name and arguments within the "
             "following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
             "<arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>"
             "<arg_value>{arg-value-2}</arg_value>...</tool_call>";
    }
  }

  int last_user = -1;
  for (size_t i = 0; i < msgs.size(); i++)
    if (msgs[i].role == "user") last_user = (int)i;

  for (size_t i = 0; i < msgs.size(); i++) {
    const ChatMsg& m = msgs[i];
    if (m.role == "user") {
      out += "<|user|>" + m.content;
    } else if (m.role == "system") {
      out += "<|system|>" + m.content;
    } else if (m.role == "assistant") {
      out += "<|assistant|>";
      // Reasoning is rendered only for an assistant turn after the last user turn; history is
      // cleared (<think></think>), which is what the reference template does.
      bool keep_reasoning = (int)i > last_user && !m.reasoning.empty();
      out += keep_reasoning ? ("<think>" + m.reasoning + "</think>") : "<think></think>";
      std::string c = strip_ws(m.content);
      if (!c.empty()) out += c;
      for (const ToolCall& tc : m.tool_calls) {
        json args = json::object();
        try {
          json parsed = json::parse(tc.arguments.empty() ? "{}" : tc.arguments);
          if (parsed.is_object()) args = parsed;
        } catch (...) {
        }
        out += "<tool_call>" + tc.name;
        append_args(args, out);
        out += "</tool_call>";
      }
    } else if (m.role == "tool") {
      // A run of consecutive tool messages becomes one <|observation|> block.
      if (i == 0 || msgs[i - 1].role != "tool") {
        out += "<|observation|>";
        size_t end = i;
        while (end + 1 < msgs.size() && msgs[end + 1].role == "tool") end++;
        bool have_ids = true;
        for (size_t k = i; k <= end; k++) if (msgs[k].tool_call_id.empty()) have_ids = false;
        const std::vector<ToolCall>* order = nullptr;
        if (have_ids && i > 0 && msgs[i - 1].role == "assistant" &&
            !msgs[i - 1].tool_calls.empty()) {
          bool all_match = true;
          for (size_t k = i; k <= end; k++) {
            bool found = false;
            for (const ToolCall& tc : msgs[i - 1].tool_calls)
              if (tc.id == msgs[k].tool_call_id) found = true;
            if (!found) all_match = false;
          }
          if (all_match) order = &msgs[i - 1].tool_calls;
        }
        if (order) {
          for (const ToolCall& tc : *order)
            for (size_t k = i; k <= end; k++)
              if (msgs[k].tool_call_id == tc.id)
                out += "<tool_response>" + msgs[k].content + "</tool_response>";
        } else {
          for (size_t k = i; k <= end; k++)
            out += "<tool_response>" + msgs[k].content + "</tool_response>";
        }
      }
    }
  }
  if (add_generation_prompt) out += thinking ? "<|assistant|><think>" : "<|assistant|><think></think>";
  return out;
}

// ---------------------------------------------------------------- output parsing
namespace {

// This checkpoint closes its thinking with `</think>` (token 154842 in tokenizer.json); the
// DeepSeek-style marker is accepted too in case a variant emits it.
const char* kThinkEnd = "</think>";
const char* kThinkEndAlt = "<ï½" "endâofâthinkingï½>";
const char* kToolBegin = "<tool_call>";
const char* kToolEnd = "</tool_call>";
const char* kObservation = "<|observation|>";
const char* kEos = "<|endoftext|>";
const char* kEndS = "<|end|>";
const char* kUser = "<|user|>";
const char* kAsst = "<|assistant|>";
const char* kSys = "<|system|>";

// Longest suffix of `s` that is a proper prefix of one of the markers (so we never emit text that
// could still turn into a tag).
size_t holdback_len(const std::string& s) {
  static const std::vector<std::string> marks = {kThinkEnd, kToolBegin, kToolEnd, kObservation,
                                                 kEos, kEndS, kUser, kAsst, kSys};
  size_t best = 0;
  for (const std::string& m : marks)
    for (size_t len = 1; len < m.size() && len <= s.size(); len++)
      if (s.compare(s.size() - len, len, m, 0, len) == 0) best = std::max(best, len);
  return best;
}

std::string args_to_json(const std::string& body) {
  // body: NAME<arg_key>K</arg_key><arg_value>V</arg_value>...
  json args = json::object();
  size_t pos = 0;
  while (true) {
    size_t ak = body.find("<arg_key>", pos);
    if (ak == std::string::npos) break;
    size_t ak_end = body.find("</arg_key>", ak);
    size_t av = body.find("<arg_value>", ak_end);
    size_t av_end = body.find("</arg_value>", av);
    if (ak_end == std::string::npos || av == std::string::npos || av_end == std::string::npos) break;
    std::string key = body.substr(ak + 9, ak_end - ak - 9);
    std::string val = body.substr(av + 11, av_end - av - 11);
    json v;
    bool ok = true;
    try {
      v = json::parse(val);
    } catch (...) {
      ok = false;
    }
    args[key] = ok ? v : json(val);
    pos = av_end + 12;
  }
  return args.dump();
}

std::string tool_name_of(const std::string& body) {
  size_t p = body.find("<arg_key>");
  return strip_ws(body.substr(0, p == std::string::npos ? body.size() : p));
}

}  // namespace

std::vector<OutputParser::Delta> OutputParser::feed(const std::string& piece) {
  buf_ += piece;
  pending_.clear();
  drain(false);
  return pending_;
}

std::vector<OutputParser::Delta> OutputParser::finish() {
  pending_.clear();
  drain(true);
  if (in_tool_ && !tool_body_.empty()) {   // unterminated tool call: emit what we have
    Delta d;
    d.tool_begin = true;
    d.tool_name = tool_name_of(tool_body_);
    d.tool_args_fragment = args_to_json(tool_body_);
    ToolCall tc{"call_0", d.tool_name, d.tool_args_fragment};
    out_.tool_calls.push_back(tc);
    pending_.push_back(d);
    tool_body_.clear();
    in_tool_ = false;
  }
  return pending_;
}

void OutputParser::drain(bool flush) {
  size_t pos = 0;
  while (pos < buf_.size()) {
    size_t keep = flush ? 0 : holdback_len(buf_.substr(pos));
    size_t limit = buf_.size() - keep;
    if (in_tool_) {
      size_t end = buf_.find(kToolEnd, pos);
      if (end == std::string::npos) {
        size_t take = limit > pos ? limit - pos : 0;
        tool_body_ += buf_.substr(pos, take);
        pos += take;
        break;
      }
      tool_body_ += buf_.substr(pos, end - pos);
      pos = end + strlen(kToolEnd);
      std::string name = tool_name_of(tool_body_);
      Delta d;
      d.tool_begin = true;
      d.tool_name = name;
      d.tool_args_fragment = args_to_json(tool_body_);
      ToolCall tc;
      char idbuf[32];
      snprintf(idbuf, sizeof(idbuf), "call_%d", (int)out_.tool_calls.size());
      tc.id = idbuf;
      tc.name = name;
      tc.arguments = d.tool_args_fragment;
      out_.tool_calls.push_back(tc);
      pending_.push_back(d);
      tool_body_.clear();
      in_tool_ = false;
      continue;
    }
    // find the earliest marker in [pos, limit)
    size_t best = std::string::npos;
    const char* which = nullptr;
    size_t which_len = 0;
    struct M {
      const char* tag;
      size_t len;
    } marks[] = {{kThinkEnd, strlen(kThinkEnd)}, {kToolBegin, strlen(kToolBegin)},
                 {kObservation, strlen(kObservation)}, {kEos, strlen(kEos)}, {kEndS, strlen(kEndS)},
                 {kUser, strlen(kUser)}, {kAsst, strlen(kAsst)}, {kSys, strlen(kSys)}};
    for (const M& m : marks) {
      size_t f = buf_.find(m.tag, pos);
      if (f != std::string::npos && f < limit && f < best) { best = f; which = m.tag; which_len = m.len; }
    }
    if (best == std::string::npos) {
      size_t take = limit > pos ? limit - pos : 0;
      std::string text = buf_.substr(pos, take);
      pos += take;
      if (!text.empty()) {
        Delta d;
        if (in_think_) { d.reasoning = text; out_.reasoning += text; }
        else { d.content = text; out_.content += text; }
        pending_.push_back(d);
      }
      break;
    }
    if (best > pos) {
      std::string text = buf_.substr(pos, best - pos);
      Delta d;
      if (in_think_) { d.reasoning = text; out_.reasoning += text; }
      else { d.content = text; out_.content += text; }
      pending_.push_back(d);
    }
    pos = best + which_len;
    if (which == std::string(kThinkEnd)) {
      in_think_ = false;
    } else if (which == std::string(kToolBegin)) {
      in_tool_ = true;
      tool_body_.clear();
    } else if (which == std::string(kUser) || which == std::string(kAsst) ||
               which == std::string(kSys)) {
      // The model started a new turn: end generation here instead of streaming hallucinated
      // conversation turns to the client.
      Delta d;
      d.stop = true;
      pending_.push_back(d);
      buf_.clear();
      return;
    } else if (which == std::string(kObservation) || which == std::string(kEos) ||
               which == std::string(kEndS)) {
      // control tokens: drop them, and stop treating text as visible content
      break;
    }
  }
  buf_.erase(0, pos);
}

ParsedOutput parse_output(const std::string& text, bool start_in_think) {
  OutputParser p(start_in_think);
  p.feed(text);
  p.finish();
  return p.parsed();
}

}  // namespace helios
