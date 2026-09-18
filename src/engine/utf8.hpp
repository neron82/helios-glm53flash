#pragma once
// UTF-8 boundary helpers for incremental output.
//
// A byte-level BPE vocabulary can split a multi-byte character across two tokens (an em-dash may
// arrive as the lead byte 0xE2 and then 0x80 0x94). Concatenating the decoded bytes is therefore
// fine, but emitting a *prefix* of them is not: a partial sequence is not valid UTF-8, and
// serialising it into JSON throws (nlohmann rejects malformed strings). The server must hold back
// an incomplete trailing sequence and emit it once its continuation bytes arrive.
#include <cstddef>
#include <string>
#include <string_view>

namespace helios {

// Expected length of the UTF-8 sequence starting at `b`, or 0 if `b` cannot start one
// (i.e. it is a continuation byte or an invalid lead).
inline int utf8_seq_len(unsigned char b) {
  if (b < 0x80) return 1;
  if ((b & 0xE0) == 0xC0) return 2;
  if ((b & 0xF0) == 0xE0) return 3;
  if ((b & 0xF8) == 0xF0) return 4;
  return 0;
}

// Number of leading bytes of `s` that form complete UTF-8 characters. Stops at the first byte that
// cannot be part of a complete sequence, so the result is always safe to put in a JSON string.
// `s` is the freshly-decoded tail of the stream (a few bytes), so a forward scan is cheap.
inline std::size_t utf8_complete_prefix(std::string_view s) {
  std::size_t i = 0;
  const std::size_t n = s.size();
  while (i < n) {
    const int need = utf8_seq_len(static_cast<unsigned char>(s[i]));
    if (need == 0) break;                                   // stray continuation byte
    if (i + static_cast<std::size_t>(need) > n) break;       // incomplete tail: wait for the rest
    bool ok = true;
    for (int k = 1; k < need; k++) {
      if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) { ok = false; break; }
    }
    if (!ok) break;
    i += static_cast<std::size_t>(need);
  }
  return i;
}

// Copy of `s` with every malformed byte (stray continuation, invalid lead, truncated tail)
// replaced by U+FFFD. Used for text that will not receive further bytes, where holding back is
// not an option.
inline std::string utf8_sanitize(std::string_view s) {
  static constexpr char kReplacement[] = "\xEF\xBF\xBD";   // U+FFFD
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    const int n = utf8_seq_len(static_cast<unsigned char>(s[i]));
    if (n == 0) { out += kReplacement; i++; continue; }
    if (i + static_cast<std::size_t>(n) > s.size()) { out += kReplacement; break; }
    bool ok = true;
    for (int k = 1; k < n; k++) {
      if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) { ok = false; break; }
    }
    if (!ok) { out += kReplacement; i++; continue; }
    out.append(s.data() + i, static_cast<std::size_t>(n));
    i += static_cast<std::size_t>(n);
  }
  return out;
}

}  // namespace helios
