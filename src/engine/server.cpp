// OpenAI-compatible HTTP server for the Helios engine.
//
// Production surface: /health, /v1/models, /v1/metrics (and /metrics), /v1/completions,
// /v1/chat/completions. Chat requests support: system/user/assistant/tool messages, OpenAI `tools`
// (rendered into the model's <tools> block and parsed back out of <tool_call> XML), `tool_choice`,
// reasoning_effort, stop sequences, and SSE streaming that separates <think> content into
// `reasoning_content` deltas.
//
// The engine is a single-sequence runner, so one generation runs at a time: the request lock is held
// for the whole generation (including while streaming). httplib still serves other connections
// concurrently, and a client that disconnects cancels its generation.
#include "engine/runner.hpp"
#include "core/chat.hpp"
#include "tokenizer/tokenizer.hpp"
#include "json.hpp"
#include "engine/utf8.hpp"
#include "httplib.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <functional>
#include <vector>

namespace helios {

using json = nlohmann::ordered_json;

namespace {

constexpr const char* kModelId = "glm-5.3-flash-exl3";

struct ServerOpts {
  std::string api_key;          // empty = no auth
  int default_max_tokens = 32768;
  // Default reasoning effort for requests that do not set one. The model's template accepts
  // low/high/max and treats anything else as max, so the server validates too.
  std::string default_reasoning_effort = "max";
  int n_threads = 4;
};
ServerOpts g_opts;
int g_ctx_cap = 0;      // KV capacity in tokens, from the runner; advertised in /v1/models

std::string new_id(const char* prefix) {
  static std::atomic<uint64_t> ctr{0};
  auto now = std::chrono::system_clock::now().time_since_epoch();
  uint64_t ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  char buf[64];
  snprintf(buf, sizeof(buf), "%s-%llx%04llx", prefix, (unsigned long long)ms,
           (unsigned long long)(ctr++ & 0xffff));
  return buf;
}

void error_response(httplib::Response& res, int status, const std::string& msg, const char* type) {
  res.status = status;
  json j{{"error", {{"message", msg}, {"type", type}, {"param", nullptr}, {"code", nullptr}}}};
  res.set_content(j.dump(), "application/json");
}

bool authorized(const httplib::Request& req) {
  if (g_opts.api_key.empty()) return true;
  auto it = req.headers.find("Authorization");
  if (it == req.headers.end()) return false;
  return it->second == "Bearer " + g_opts.api_key;
}

// Incremental UTF-8-safe decoding: byte-level BPE can split a codepoint across tokens, so hold back
// any incomplete trailing sequence until the next token arrives.
// Turns the token stream into text without ever emitting a partial UTF-8 character.
//
// Tokenizer::decode concatenates the byte string of each token, so decoding the whole id list is
// append-only and diffing against the previous length is sound. What is not sound is emitting a
// prefix: a byte-level BPE vocabulary can split a character across tokens, so the tail of the
// decoded bytes may be an incomplete sequence. Those bytes are held back here and emitted once
// their continuations arrive.
class Utf8Streamer {
public:
  std::string push(Tokenizer& tk, const std::vector<int>& all_ids) {
    const std::string bytes = tk.decode(all_ids, /*keep_special=*/true);
    if (bytes.size() <= prev_) return {};
    const std::string_view fresh(bytes.data() + prev_, bytes.size() - prev_);
    const size_t take = utf8_complete_prefix(fresh);
    std::string out(fresh.substr(0, take));
    prev_ += take;
    return out;
  }

private:
  size_t prev_ = 0;
};

struct ChatRequest {
  std::vector<ChatMsg> msgs;
  json tools = json::array();
  std::string reasoning_effort = "max";
  bool thinking = true;
  bool tools_off = false;
  GenParams gen;
  bool stream = false;
};

bool parse_common(const json& body, GenParams& p, std::string& err) {
  p.max_tokens = body.contains("max_tokens") && body["max_tokens"].is_number()
                     ? body["max_tokens"].get<int>()
                     : g_opts.default_max_tokens;
  if (p.max_tokens <= 0) p.max_tokens = g_opts.default_max_tokens;
  p.temperature = body.value("temperature", 0.7f);
  p.top_p = body.value("top_p", 0.95f);
  p.top_k = body.value("top_k", 40);
  p.min_p = body.value("min_p", 0.0f);
  p.rep_penalty = body.value("repetition_penalty", 1.0f);
  p.greedy = p.temperature <= 0.01f;
  if (body.contains("stop")) {
    const json& s = body["stop"];
    if (s.is_string()) p.stop.push_back(s.get<std::string>());
    else if (s.is_array())
      for (const auto& v : s) if (v.is_string()) p.stop.push_back(v.get<std::string>());
  }
  if (body.contains("n") && body["n"].is_number() && body["n"].get<int>() > 1) {
    err = "n > 1 is not supported";
    return false;
  }
  return true;
}

std::string content_to_text(const json& c) {
  if (c.is_string()) return c.get<std::string>();
  if (c.is_array()) {   // OpenAI content parts: keep the text parts
    std::string out;
    for (const auto& part : c) {
      if (part.is_object() && part.contains("text") && part["text"].is_string())
        out += part["text"].get<std::string>();
    }
    return out;
  }
  return "";
}

bool parse_chat(const json& body, ChatRequest& out, std::string& err) {
  if (!body.contains("messages") || !body["messages"].is_array()) {
    err = "messages is required";
    return false;
  }
  for (const auto& m : body["messages"]) {
    ChatMsg msg;
    msg.role = m.value("role", "user");
    msg.content = content_to_text(m.contains("content") ? m["content"] : json(""));
    if (m.contains("reasoning_content") && m["reasoning_content"].is_string())
      msg.reasoning = m["reasoning_content"].get<std::string>();
    if (m.contains("tool_call_id") && m["tool_call_id"].is_string())
      msg.tool_call_id = m["tool_call_id"].get<std::string>();
    if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
      for (const auto& tc : m["tool_calls"]) {
        ToolCall c;
        c.id = tc.value("id", "");
        if (tc.contains("function")) {
          c.name = tc["function"].value("name", "");
          const json& a = tc["function"].contains("arguments") ? tc["function"]["arguments"]
                                                              : json("");
          c.arguments = a.is_string() ? a.get<std::string>() : a.dump();
        }
        msg.tool_calls.push_back(c);
      }
    }
    out.msgs.push_back(std::move(msg));
  }
  if (body.contains("tools") && body["tools"].is_array()) out.tools = body["tools"];
  if (out.tools.empty()) out.tools_off = true;
  if (body.contains("tool_choice")) {
    const json& tc = body["tool_choice"];
    if (tc.is_string() && tc.get<std::string>() == "none") out.tools_off = true;
  }
  out.reasoning_effort = g_opts.default_reasoning_effort;   // server default; request wins below
  if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string())
    out.reasoning_effort = body["reasoning_effort"].get<std::string>();
  if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
    const json& kw = body["chat_template_kwargs"];
    if (kw.contains("reasoning_effort") && kw["reasoning_effort"].is_string())
      out.reasoning_effort = kw["reasoning_effort"].get<std::string>();
    if (kw.contains("enable_thinking") && kw["enable_thinking"].is_boolean())
      out.thinking = kw["enable_thinking"].get<bool>();
    if (kw.contains("thinking") && kw["thinking"].is_boolean())
      out.thinking = kw["thinking"].get<bool>();
  }
  if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean())
    out.thinking = body["enable_thinking"].get<bool>();
  if (body.contains("thinking") && body["thinking"].is_boolean())
    out.thinking = body["thinking"].get<bool>();
  return parse_common(body, out.gen, err);
}

struct GenOutcome {
  std::string reasoning, content;
  std::vector<ToolCall> tool_calls;
  int completion_tokens = 0;
  bool hit_stop = false;
};

}  // namespace

int run_server(Runner& runner, Tokenizer& tk, const std::string& host, int port, int n_threads,
               const std::string& api_key, int default_max_tokens,
               const std::string& default_reasoning_effort) {
  httplib::Server srv;
  g_opts.api_key = api_key;
  g_opts.n_threads = n_threads;
  if (default_max_tokens > 0) g_opts.default_max_tokens = default_max_tokens;
  if (default_reasoning_effort == "low" || default_reasoning_effort == "high" ||
      default_reasoning_effort == "max") {
    g_opts.default_reasoning_effort = default_reasoning_effort;
  } else if (!default_reasoning_effort.empty()) {
    fprintf(stderr, "[server] unknown reasoning effort '%s' (want low|high|max); using max\n",
            default_reasoning_effort.c_str());
  }
  g_ctx_cap = runner.context_cap();
  static std::mutex gen_mu;          // one generation at a time (single-sequence engine)
  static std::atomic<uint64_t> served{0};

  // A stop caused by running out of KV capacity is reported as "length", like a max_tokens stop:
  // the caller must be able to tell truncation from a natural end.
  auto finish_reason = [&](int completion_tokens, int max_tokens) -> const char* {
    if (completion_tokens >= max_tokens) return "length";
    if (runner.context_cap() > 0 && runner.pos() >= runner.context_cap()) return "length";
    return "stop";
  };

  srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
  });

  srv.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
    // Advertise the limits as well: clients that read them can size their own request controls
    // instead of guessing, and `max_tokens` here is what an omitted request field will use.
    json j{{"object", "list"},
           {"data", json::array({{{"id", kModelId},
                                  {"object", "model"},
                                  {"created", 0},
                                  {"owned_by", "helios"},
                                  {"root", kModelId},
                                  {"max_tokens", g_opts.default_max_tokens},
                                  {"context_length", g_ctx_cap},
                                  {"reasoning_effort", g_opts.default_reasoning_effort}}})}};
    res.set_content(j.dump(), "application/json");
  });

  auto metrics = [&](const httplib::Request&, httplib::Response& res) {
    const auto& t = runner.timings();
    json j{{"requests", served.load()},
           {"prefill_ms", t.prefill_ms}, {"prefill_tokens", t.prefill_tokens},
           {"decode_ms", t.decode_ms}, {"decode_tokens", t.decode_tokens},
           {"decode_tps", t.decode_ms > 0 ? t.decode_tokens * 1000.0 / t.decode_ms : 0.0},
           {"prefill_tps", t.prefill_ms > 0 ? t.prefill_tokens * 1000.0 / t.prefill_ms : 0.0}};
    res.set_content(j.dump(), "application/json");
  };
  srv.Get("/metrics", metrics);
  srv.Get("/v1/metrics", metrics);

  // Shared generation driver: runs the model, feeds the parser, and streams or accumulates.
  auto generate = [&](const std::vector<int>& prompt, const GenParams& p, ChatRequest& req,
                      GenOutcome& out, std::function<void(OutputParser::Delta&)> emit) {
    OutputParser parser(/*start_in_think=*/req.thinking);
    Utf8Streamer utf8;
    std::vector<int> ids;
    uint64_t tok_count = 0;
    std::string visible;   // for stop-string matching on client-visible text
    bool stop_hit = false;
    runner.generate(prompt, p, [&](int tok) {
      ids.push_back(tok);
      tok_count++;
      std::string piece = utf8.push(tk, ids);
      if (piece.empty()) return true;
      std::vector<OutputParser::Delta> deltas = parser.feed(piece);
      for (auto& d : deltas) {
        if (d.stop) { stop_hit = true; if (emit) emit(d); return false; }
        if (!d.content.empty()) {
          visible += d.content;
          for (const auto& s : p.stop)
            if (!s.empty() && visible.size() >= s.size() &&
                visible.compare(visible.size() - s.size(), s.size(), s) == 0) {
              stop_hit = true;
              return false;
            }
        }
        if (emit) emit(d);
      }
      return true;
    });
    std::vector<OutputParser::Delta> rest = parser.finish();
    for (auto& d : rest) { if (d.stop) { out.hit_stop = true; } if (emit) emit(d); }
    const ParsedOutput& parsed = parser.parsed();
    // No further bytes will arrive, so a truncated final character must be replaced rather than
    // held back - otherwise serialising it would still throw.
    out.reasoning = utf8_sanitize(parsed.reasoning);
    out.content = utf8_sanitize(parsed.content);
    out.tool_calls = parsed.tool_calls;
    out.completion_tokens = (int)tok_count;
    out.hit_stop = stop_hit;
  };

  auto chat_handler = [&](const httplib::Request& hreq, httplib::Response& res) {
    if (!authorized(hreq)) { error_response(res, 401, "invalid api key", "invalid_request_error"); return; }
    json body;
    try {
      body = json::parse(hreq.body);
    } catch (...) {
      error_response(res, 400, "could not parse JSON body", "invalid_request_error");
      return;
    }
    ChatRequest req;
    std::string err;
    if (!parse_chat(body, req, err)) { error_response(res, 400, err, "invalid_request_error"); return; }
    req.gen.stop.clear();
    parse_common(body, req.gen, err);
    req.stream = body.value("stream", false);
    const bool want_usage = body.contains("stream_options") && body["stream_options"].is_object() &&
                            body["stream_options"].value("include_usage", false);
    json tools = req.tools_off ? json::array() : req.tools;
    std::string prompt_text = render_chat(req.msgs, tools, req.reasoning_effort,
                                          /*add_generation_prompt=*/true, req.thinking);
    std::vector<int> prompt = tk.encode(prompt_text);
    const std::string id = new_id("chatcmpl");
    const int64_t created = (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count();

    if (!req.stream) {
      GenOutcome out;
      std::lock_guard<std::mutex> lock(gen_mu);
      generate(prompt, req.gen, req, out, nullptr);
      json msg{{"role", "assistant"}, {"content", out.content}};
      if (!out.reasoning.empty()) msg["reasoning_content"] = out.reasoning;
      if (!out.tool_calls.empty()) {
        msg["tool_calls"] = json::array();
        for (const auto& tc : out.tool_calls) {
          msg["tool_calls"].push_back({{"id", tc.id},
                                       {"type", "function"},
                                       {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}});
        }
      }
      const char* finish = !out.tool_calls.empty() ? "tool_calls"
                           : out.hit_stop ? "stop"
                           : finish_reason(out.completion_tokens, req.gen.max_tokens);
      json outj{{"id", id},         {"object", "chat.completion"},
                {"created", created}, {"model", kModelId},
                {"choices", json::array({{{"index", 0},
                                          {"message", msg},
                                          {"logprobs", nullptr},
                                          {"finish_reason", finish}}})},
                {"usage", {{"prompt_tokens", (int)prompt.size()},
                           {"completion_tokens", out.completion_tokens},
                           {"total_tokens", (int)prompt.size() + out.completion_tokens}}}};
      served++;
      res.set_content(outj.dump(), "application/json");
      return;
    }

    // ---- SSE streaming
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_chunked_content_provider(
        "text/event-stream",
        [&, prompt, req, id, created, want_usage](size_t, httplib::DataSink& sink) mutable {
          std::lock_guard<std::mutex> lock(gen_mu);
          auto send = [&](const json& j) {
            std::string s = "data: " + j.dump() + "\n\n";
            return sink.write(s.data(), s.size());
          };
          json base{{"id", id}, {"object", "chat.completion.chunk"},
                    {"created", created}, {"model", kModelId}};
          json first = base;
          first["choices"] = json::array({{{"index", 0}, {"delta", {{"role", "assistant"}}},
                                           {"finish_reason", nullptr}}});
          if (!send(first)) { sink.done(); return true; }
          bool alive = true;
          int tool_index = -1;
          GenOutcome out;
          generate(prompt, req.gen, req, out, [&](OutputParser::Delta& d) {
            if (!alive) return;
            if (!d.reasoning.empty()) {
              json j = base;
              j["choices"] = json::array({{{"index", 0},
                                           {"delta", {{"reasoning_content", d.reasoning}}},
                                           {"finish_reason", nullptr}}});
              alive = send(j);
            }
            if (!d.content.empty()) {
              json j = base;
              j["choices"] = json::array({{{"index", 0}, {"delta", {{"content", d.content}}},
                                           {"finish_reason", nullptr}}});
              alive = send(j);
            }
            if (d.tool_begin) {
              tool_index++;
              json j = base;
              j["choices"] = json::array(
                  {{{"index", 0},
                    {"delta", {{"tool_calls", json::array({{{"index", tool_index},
                                                            {"id", out.tool_calls.empty()
                                                                       ? std::string("call_0")
                                                                       : out.tool_calls.back().id},
                                                            {"type", "function"},
                                                            {"function", {{"name", d.tool_name},
                                                                          {"arguments", ""}}}}})}}},
                    {"finish_reason", nullptr}}});
              alive = send(j);
              if (!d.tool_args_fragment.empty() && alive) {
                json a = base;
                a["choices"] = json::array(
                    {{{"index", 0},
                      {"delta", {{"tool_calls", json::array({{{"index", tool_index},
                                                              {"function",
                                                               {{"arguments", d.tool_args_fragment}}}}})}}},
                      {"finish_reason", nullptr}}});
                alive = send(a);
              }
            }
          });
          const char* finish = !out.tool_calls.empty() ? "tool_calls"
                               : out.hit_stop ? "stop"
                               : finish_reason(out.completion_tokens, req.gen.max_tokens);
          if (alive) {
            json last = base;
            last["choices"] = json::array({{{"index", 0}, {"delta", json::object()},
                                            {"finish_reason", finish}}});
            alive = send(last);
          }
          if (alive && want_usage) {
            json u = base;
            u["choices"] = json::array();
            u["usage"] = {{"prompt_tokens", (int)prompt.size()},
                          {"completion_tokens", out.completion_tokens},
                          {"total_tokens", (int)prompt.size() + out.completion_tokens}};
            alive = send(u);
          }
          if (alive) {
            const char* done = "data: [DONE]\n\n";
            sink.write(done, strlen(done));
          }
          sink.done();
          served++;
          return true;
        });
  };

  auto text_handler = [&](const httplib::Request& hreq, httplib::Response& res) {
    if (!authorized(hreq)) { error_response(res, 401, "invalid api key", "invalid_request_error"); return; }
    json body;
    try {
      body = json::parse(hreq.body);
    } catch (...) {
      error_response(res, 400, "could not parse JSON body", "invalid_request_error");
      return;
    }
    GenParams p;
    std::string err;
    if (!parse_common(body, p, err)) { error_response(res, 400, err, "invalid_request_error"); return; }
    std::string text;
    if (body.contains("prompt")) {
      if (body["prompt"].is_string()) text = body["prompt"].get<std::string>();
      else if (body["prompt"].is_array() && !body["prompt"].empty() && body["prompt"][0].is_string())
        text = body["prompt"][0].get<std::string>();
    }
    std::vector<int> prompt = tk.encode(text);
    GenOutcome out;
    std::lock_guard<std::mutex> lock(gen_mu);
    // raw completions carry no chat template, so the model's output is taken as literal text
    OutputParser parser(/*start_in_think=*/false);
    Utf8Streamer utf8;
    std::vector<int> ids;
    runner.generate(prompt, p, [&](int tok) {
      ids.push_back(tok);
      std::string piece = utf8.push(tk, ids);
      if (!piece.empty()) parser.feed(piece);
      return true;
    });
    parser.finish();
    const ParsedOutput& parsed = parser.parsed();
    out.content = parsed.content;
    out.reasoning = parsed.reasoning;
    out.completion_tokens = (int)ids.size();
    const char* finish = finish_reason(out.completion_tokens, p.max_tokens);
    json outj{{"id", new_id("cmpl")},
              {"object", "text_completion"},
              {"created", (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::system_clock::now().time_since_epoch()).count()},
              {"model", kModelId},
              {"choices", json::array({{{"index", 0}, {"text", out.content}, {"finish_reason", finish}}})},
              {"usage", {{"prompt_tokens", (int)prompt.size()},
                         {"completion_tokens", out.completion_tokens},
                         {"total_tokens", (int)prompt.size() + out.completion_tokens}}}};
    served++;
    res.set_content(outj.dump(), "application/json");
  };

  srv.Post("/v1/chat/completions", chat_handler);
  srv.Post("/v1/completions", text_handler);
  srv.set_read_timeout(600, 0);
  srv.set_write_timeout(600, 0);
  srv.set_payload_max_length(32 * 1024 * 1024);
  printf("[server] model=%s listening on %s:%d (%d threads)%s\n", kModelId, host.c_str(), port,
         n_threads, api_key.empty() ? "" : " [api key required]");
  fflush(stdout);
  if (!srv.listen(host.c_str(), port)) {
    fprintf(stderr, "[server] failed to listen on %s:%d\n", host.c_str(), port);
    return 1;
  }
  return 0;
}

}  // namespace helios
