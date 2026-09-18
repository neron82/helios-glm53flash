// Boundary tests for the UTF-8 helpers used by the streaming server.
//
// The bug these exist to prevent: a byte-level BPE vocabulary splits a multi-byte character across
// tokens, so the decoded byte prefix can end mid-character. Emitting that prefix puts invalid UTF-8
// into a JSON string, which nlohmann rejects by throwing (type_error.316) - killing the server
// mid-request. The streamer must hold the incomplete tail back instead, and must NOT hold back a
// character that happens to be complete.
#include "engine/utf8.hpp"

#include <cstdio>
#include <string>

using helios::utf8_complete_prefix;
using helios::utf8_sanitize;

static int failures = 0;

static void check_prefix(const char* what, const std::string& s, size_t want) {
  const size_t got = utf8_complete_prefix(s);
  if (got != want) {
    printf("FAIL %s: utf8_complete_prefix(len %zu) = %zu, want %zu\n", what, s.size(), got, want);
    failures++;
  }
}

static void check_sanitize(const char* what, const std::string& in, const std::string& want) {
  const std::string got = utf8_sanitize(in);
  if (got != want) {
    printf("FAIL %s: utf8_sanitize -> %zu bytes, want %zu\n", what, got.size(), want.size());
    failures++;
  }
}

int main() {
  const std::string emdash = "\xE2\x80\x94";      // U+2014, 3 bytes
  const std::string euro = "\xE2\x82\xAC";        // U+20AC, 3 bytes
  const std::string emoji = "\xF0\x9F\x98\x80";   // U+1F600, 4 bytes
  const std::string eacute = "\xC3\xA9";          // U+00E9, 2 bytes
  const std::string repl = "\xEF\xBF\xBD";        // U+FFFD

  // Complete input is emitted in full - including a complete character at the very end. Holding this
  // case back was the original bug: the length test ran against an already-walked-back index.
  check_prefix("ascii", "abc", 3);
  check_prefix("empty", "", 0);
  check_prefix("2-byte complete", "a" + eacute, 3);
  check_prefix("3-byte complete", "a" + emdash, 4);
  check_prefix("4-byte complete", "a" + emoji, 5);
  check_prefix("char complete at end", emdash, 3);
  check_prefix("two 3-byte chars", emdash + euro, 6);

  // Incomplete tails are held back, at every prefix length of every sequence size.
  check_prefix("2-byte lead only", "a\xC3", 1);
  check_prefix("3-byte lead only", "a\xE2", 1);
  check_prefix("3-byte lead+1", "a\xE2\x80", 1);
  check_prefix("4-byte lead only", "a\xF0", 1);
  check_prefix("4-byte lead+1", "a\xF0\x9F", 1);
  check_prefix("4-byte lead+2", "a\xF0\x9F\x98", 1);
  check_prefix("incomplete after complete", emdash + "a\xE2", 4);

  // A stray continuation byte can never be emitted - it is not valid UTF-8 on its own.
  check_prefix("stray continuation", "a\x80", 1);
  check_prefix("only continuation", "\x80", 0);
  check_prefix("continuations only", "\x80\x94", 0);

  // Sanitize leaves valid text untouched and replaces what cannot be completed.
  check_sanitize("valid passthrough", "hello " + emdash + emoji, "hello " + emdash + emoji);
  check_sanitize("truncated 3-byte", "a\xE2\x80", "a" + repl);
  check_sanitize("truncated 4-byte", "a\xF0\x9F", "a" + repl);
  check_sanitize("stray continuation", "a\x80" "b", "a" + repl + "b");
  check_sanitize("ascii unchanged", "plain", "plain");

  // The streamer's accumulation contract: hold back, then emit once the character is complete.
  // This mirrors what happens across three consecutive tokens.
  {
    const std::string bytes = "hi " + emdash + " there";
    size_t prev = 0;
    std::string assembled;
    for (size_t upto : {size_t(3), size_t(4), size_t(5), bytes.size()}) {   // byte at a time over the dash
      const std::string fresh = bytes.substr(0, upto).substr(prev);
      const size_t take = utf8_complete_prefix(fresh);
      assembled += fresh.substr(0, take);
      prev += take;
    }
    if (assembled != bytes) {
      printf("FAIL incremental: assembled %zu bytes, want %zu\n", assembled.size(), bytes.size());
      failures++;
    }
  }

  if (failures) { printf("UTF8: %d FAILURES\n", failures); return 1; }
  printf("UTF8: all boundary checks pass\n");
  return 0;
}
