// Chat template + output parser tests.
//  1. render_chat output must match transformers' apply_chat_template byte-for-byte (the same three
//     cases are rendered by /tmp/tmpl_ref.py and compared by test/chat_parity.sh).
//  2. parse_output / OutputParser must agree with each other for any chunking of the input.
#include "core/chat.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace helios;
using json = nlohmann::ordered_json;

static void print_case(const char* name, const std::string& text) {
  printf("%s %s\n", name, json(text).dump().c_str());
}

int main() {
  const char* THINK_END = "</think>";          // token 154842 in this checkpoint
  // ---- 1. template renders (compared against the transformers reference by the shell script)
  json tools = json::array({{{"type", "function"},
                             {"function", {{"name", "get_weather"},
                                           {"description", "Get weather for a city"},
                                           {"parameters", {{"type", "object"},
                                                           {"properties", {{"city", {{"type", "string"},
                                                                                     {"description", "City name"}}}}},
                                                           {"required", json::array({"city"})}}}}}}});

  std::vector<ChatMsg> c1 = {{"user", "Weather in Paris?", "", {}, ""}};
  print_case("CASE1", render_chat(c1, tools, "max", true));

  std::vector<ChatMsg> c2 = {
      {"user", "Weather in Paris?", "", {}, ""},
      {"assistant", "", "", {{"call_1", "get_weather", "{\"city\": \"Paris\"}"}}, ""},
      {"tool", "18C and sunny", "", {}, "call_1"}};
  print_case("CASE2", render_chat(c2, tools, "max", true));

  std::vector<ChatMsg> c3 = {{"user", "hi", "", {}, ""},
                             {"assistant", "hello", "step 1", {}, ""},
                             {"user", "bye", "", {}, ""}};
  print_case("CASE3", render_chat(c3, json::array(), "max", true));

  // ---- 2. parser: reasoning, content, tool call
  const std::string full = std::string("Let me think about this.") + THINK_END +
                           "The weather is 18C.<tool_call>get_weather<arg_key>city</arg_key>"
                           "<arg_value>Paris</arg_value></tool_call>";
  ParsedOutput a = parse_output(full);
  printf("PARSE reasoning=%s\n", json(a.reasoning).dump().c_str());
  printf("PARSE content=%s\n", json(a.content).dump().c_str());
  printf("PARSE ncalls=%zu\n", a.tool_calls.size());
  if (!a.tool_calls.empty()) {
    printf("PARSE call name=%s args=%s\n", a.tool_calls[0].name.c_str(),
           a.tool_calls[0].arguments.c_str());
  }

  // streaming must produce exactly the same aggregate for any chunk size
  bool same = true;
  for (int step : {1, 2, 3, 7, 13}) {
    OutputParser p;
    for (size_t i = 0; i < full.size(); i += step) p.feed(full.substr(i, step));
    p.finish();
    ParsedOutput s = p.parsed();
    if (s.reasoning != a.reasoning || s.content != a.content ||
        s.tool_calls.size() != a.tool_calls.size() ||
        (!s.tool_calls.empty() && (s.tool_calls[0].name != a.tool_calls[0].name ||
                                   s.tool_calls[0].arguments != a.tool_calls[0].arguments))) {
      same = false;
      printf("STREAM MISMATCH at step %d: reasoning=%s content=%s calls=%zu\n", step,
             json(s.reasoning).dump().c_str(), json(s.content).dump().c_str(), s.tool_calls.size());
    }
  }
  printf("STREAM identical=%d\n", (int)same);

  // a tool call without any visible content, and an unterminated one
  ParsedOutput b = parse_output(std::string(THINK_END) +
                                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris"
                                "</arg_value></tool_call>");
  printf("PARSE2 content_empty=%d calls=%zu\n", (int)b.content.empty(), b.tool_calls.size());
  ParsedOutput c = parse_output(std::string("reasoning only") + THINK_END);
  printf("PARSE3 reasoning=%s content_empty=%d\n", json(c.reasoning).dump().c_str(),
         (int)c.content.empty());
  printf("CHAT TEST DONE\n");
  return 0;
}
