// Chat template + output parser tests.
//  1. render_chat output must match transformers' apply_chat_template byte-for-byte. The CASE lines
//     printed below are re-rendered from the checkpoint's own chat_template.jinja and diffed by
//     test/chat_parity.py (run it with ~/shared-venv-gpu/bin/python - it needs transformers).
//     Keep the two case lists in sync when adding one.
//  2. parse_output / OutputParser must agree with each other for any chunking of the input, and a
//     turn boundary in the stream (a role token, or the <|observation|> the template writes before
//     tool responses) must stop the parser and suppress everything after it.
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
  {
    std::string text, error;
    if (!chat_content_text(json::array({{{"type", "text"}, {"text", "hello"}}, " world"}),
                           text, error) || text != "hello world") return 1;
    for (const char* kind : {"image_url", "image", "video", "input_audio"}) {
      json parts = json::array({{{"type", "text"}, {"text", "Describe this"}},
                               {{"type", kind}, {"text", "must not masquerade as text"}}});
      if (chat_content_text(parts, text, error) || !text.empty() || error.empty()) return 1;
    }
    printf("CONTENT text parts accepted, media parts rejected: PASS\n");
    std::vector<std::string> urls;
    if (!chat_content_text(json::array({{{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,AAAA"}}}},
                                        {{"type", "text"}, {"text", "describe"}}}),
                           text, error, &urls) || urls.size() != 1 ||
        text != "[HELIOS_IMAGE_0]describe") return 1;
  }
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

  // Two tools, one carrying the two keys the reference template strips (strict, defer_loading), and
  // non-string argument types. Clients' own shapes are the ones that matter, so the render has to
  // match the template for these too - a divergence here stays invisible until it does not.
  auto mk_tool = [](const std::string& name, const std::string& desc, json fn_extra,
                    json params) {
    json f = json::object();
    f["name"] = name;
    f["description"] = desc;
    for (auto it = fn_extra.begin(); it != fn_extra.end(); ++it) f[it.key()] = it.value();
    f["parameters"] = params;
    json t = json::object();
    t["type"] = "function";
    t["function"] = f;
    return t;
  };
  json params_term = json::object();
  params_term["type"] = "object";
  params_term["properties"] = json::object();
  params_term["properties"]["command"] = {{"type", "string"}};
  params_term["properties"]["timeout"] = {{"type", "integer"}, {"default", 30}};
  params_term["properties"]["flags"] = {{"type", "array"}, {"items", {{"type", "string"}}}};
  params_term["required"] = json::array({"command"});
  json params_search = json::object();
  params_search["type"] = "object";
  params_search["properties"] = json::object();
  params_search["properties"]["query"] = {{"type", "string"}};
  params_search["required"] = json::array({"query"});
  json fn_extra = json::object();
  fn_extra["defer_loading"] = false;
  fn_extra["strict"] = true;
  json tools2 = json::array({mk_tool("terminal", "Run a shell command", json::object(), params_term),
                             mk_tool("web_search", "Search", fn_extra, params_search)});
  std::vector<ChatMsg> c4 = {{"user", "how many pythons are running?", "", {}, ""}};
  print_case("CASE4", render_chat(c4, tools2, "high", true));

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

  // ---- 3. the model does not stop after a tool call: it writes the observation the client owes it
  // and keeps going. Anything after <|observation|> is the model's hallucination of the next turn -
  // it must terminate generation, not be streamed as content or parsed as a second tool call.
  const std::string hallucinating =
      std::string("I will list the processes.") + THINK_END +
      "<tool_call>terminal<arg_key>command</arg_key><arg_value>ps -eo pid,ppid,etime,user,args"
      "</arg_value></tool_call>" + "<|observation|><tool_response>no output</tool_response>" +
      " The filter ate it, let me retry." + "<tool_call>terminal<arg_key>command</arg_key>"
      "<arg_value>ps -eo ppend,etime</arg_value></tool_call>";
  ParsedOutput h = parse_output(hallucinating);
  printf("PARSE4 content=%s ncalls=%zu\n", json(h.content).dump().c_str(), h.tool_calls.size());
  if (!h.tool_calls.empty())
    printf("PARSE4 call0 name=%s args=%s\n", h.tool_calls[0].name.c_str(),
           h.tool_calls[0].arguments.c_str());

  // streaming form: generation must stop at the boundary, the tail must never reach the client, and
  // the aggregate must equal parse_output's for any chunking
  {
    bool ok = true;
    for (int step : {1, 2, 3, 7, 13, 64}) {
      OutputParser p;
      std::string seen_content, seen_reason;
      size_t ncalls = 0;
      bool stopped = false;
      auto take = [&](std::vector<OutputParser::Delta> ds) {
        for (auto& d : ds) {
          seen_content += d.content;
          seen_reason += d.reasoning;
          if (d.tool_begin) ncalls++;
          if (d.stop) stopped = true;
        }
      };
      for (size_t i = 0; i < hallucinating.size(); i += step)
        take(p.feed(hallucinating.substr(i, step)));
      take(p.finish());
      if (!stopped || ncalls != h.tool_calls.size() || seen_content != h.content ||
          seen_reason != h.reasoning) {
        ok = false;
        printf("PARSE4S MISMATCH step=%d stopped=%d ncalls=%zu content=%s\n", step, (int)stopped,
               ncalls, json(seen_content).dump().c_str());
      }
    }
    printf("PARSE4S boundary_and_aggregate_hold=%d\n", (int)ok);
  }

  // ---- 4. several tool calls in one turn must carry distinct ids: clients round-trip results by id
  ParsedOutput m = parse_output(std::string(THINK_END) +
                                "<tool_call>terminal<arg_key>command</arg_key><arg_value>ls"
                                "</arg_value></tool_call>" +
                                "<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/x"
                                "</arg_value></tool_call>");
  printf("PARSE5 ncalls=%zu", m.tool_calls.size());
  for (const ToolCall& tc : m.tool_calls) printf(" id=%s", tc.id.c_str());
  printf("\n");
  // ... and an unterminated final call must not reuse the first call's id
  ParsedOutput u = parse_output(std::string(THINK_END) +
                                "<tool_call>terminal<arg_key>command</arg_key><arg_value>ls"
                                "</arg_value></tool_call>" +
                                "<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/x");
  printf("PARSE6 ncalls=%zu", u.tool_calls.size());
  for (const ToolCall& tc : u.tool_calls) printf(" id=%s", tc.id.c_str());
  printf("\n");
  printf("CHAT TEST DONE\n");
  return 0;
}
