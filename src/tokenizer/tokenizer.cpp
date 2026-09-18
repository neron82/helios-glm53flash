// GLM-5.3 byte-level BPE tokenizer + chat template (matches HF tokenizers output).
#include "tokenizer/tokenizer.hpp"
#include "tokenizer/unicode_class.hpp"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>

using nlohmann::json;

namespace helios {

// ---------------------------------------------------------------- byte mapping
namespace {
const std::array<std::string, 256>& b2s() {
  static std::array<std::string, 256> t;
  static bool init = false;
  if (!init) {
    std::vector<int> bs;
    for (int b = 33; b <= 126; b++) bs.push_back(b);
    for (int b = 161; b <= 172; b++) bs.push_back(b);
    for (int b = 174; b <= 255; b++) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n); n++; }
    }
    for (size_t i = 0; i < bs.size(); i++) {
      uint32_t cp = (uint32_t)cs[i];
      std::string s;
      if (cp < 0x80) s += (char)cp;
      else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
      else { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
      t[bs[i]] = s;
    }
    init = true;
  }
  return t;
}
const std::unordered_map<std::string, uint8_t>& s2b() {
  static std::unordered_map<std::string, uint8_t> m;
  if (m.empty()) for (int b = 0; b < 256; b++) m[b2s()[b]] = (uint8_t)b;
  return m;
}

inline bool in_ranges(uint32_t cp, const URange* r, int n) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    if (cp < r[mid].lo) hi = mid - 1;
    else if (cp > r[mid].hi) lo = mid + 1;
    else return true;
  }
  return false;
}
inline bool is_L(uint32_t cp) { return cp < 128 ? ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) : in_ranges(cp, kLetterRanges, kLetterRanges_n); }
inline bool is_N(uint32_t cp) { return cp < 128 ? (cp >= '0' && cp <= '9') : in_ranges(cp, kNumberRanges, kNumberRanges_n); }
inline bool is_nl(uint32_t cp) { return cp == '\r' || cp == '\n'; }
inline bool is_ws(uint32_t cp) {
  switch (cp) {
    case ' ': case '\t': case '\n': case '\r': case 0x0B: case 0x0C: case 0x85: case 0xA0:
    case 0x1680: case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000: return true;
    default: return cp >= 0x2000 && cp <= 0x200A;
  }
}
// decode one UTF-8 codepoint; returns length (1..4), cp on success
inline int dec_cp(const char* s, size_t len, size_t i, uint32_t& cp) {
  unsigned char c = (unsigned char)s[i];
  if (c < 0x80) { cp = c; return 1; }
  if ((c & 0xE0) == 0xC0 && i + 1 < len) { cp = ((c & 0x1F) << 6) | ((unsigned char)s[i+1] & 0x3F); return 2; }
  if ((c & 0xF0) == 0xE0 && i + 2 < len) { cp = ((c & 0x0F) << 12) | (((unsigned char)s[i+1] & 0x3F) << 6) | ((unsigned char)s[i+2] & 0x3F); return 3; }
  if ((c & 0xF8) == 0xF0 && i + 3 < len) { cp = ((c & 0x07) << 18) | (((unsigned char)s[i+1] & 0x3F) << 12) | (((unsigned char)s[i+2] & 0x3F) << 6) | ((unsigned char)s[i+3] & 0x3F); return 4; }
  cp = c; return 1;
}
}  // namespace

bool Tokenizer::load(const std::string& dir) {
  std::ifstream f(dir + "/tokenizer.json", std::ios::binary);
  if (!f) { fprintf(stderr, "[tok] cannot open tokenizer.json\n"); return false; }
  json j = json::parse(f);
  const auto& vocab = j.at("model").at("vocab");
  size_t maxid = 0;
  for (auto it = vocab.begin(); it != vocab.end(); ++it) maxid = std::max(maxid, (size_t)it.value().get<int>());
  id_to_tok_.assign(maxid + 1, std::string());
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    int id = it.value().get<int>();
    tok_to_id_[it.key()] = id;
    id_to_tok_[id] = it.key();
  }
  const auto& merges = j.at("model").at("merges");
  int rank = 0;
  for (const auto& m : merges) {
    std::string a, b;
    if (m.is_array()) { a = m[0].get<std::string>(); b = m[1].get<std::string>(); }
    else { const std::string s = m.get<std::string>(); size_t sp = s.find(' '); a = s.substr(0, sp); b = s.substr(sp + 1); }
    merge_rank_[a + '\x01' + b] = rank++;
  }
  if (j.contains("added_tokens")) {
    for (const auto& t : j["added_tokens"]) {
      std::string c = t.at("content").get<std::string>();
      int id = t.at("id").get<int>();
      added_.push_back({c, id});
      added_map_[c] = id;
      if ((int)id_to_tok_.size() <= id) id_to_tok_.resize(id + 1);
      id_to_tok_[id] = c;
    }
    std::sort(added_.begin(), added_.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  }
  auto it = added_map_.find("<|endoftext|>");
  eos_ = it != added_map_.end() ? it->second : 0;
  printf("[tok] vocab=%zu merges=%zu added=%zu eos=%d\n", id_to_tok_.size(), merge_rank_.size(), added_.size(), eos_);
  return true;
}

int Tokenizer::special_id(const std::string& s) const {
  auto it = added_map_.find(s);
  return it == added_map_.end() ? -1 : it->second;
}

void Tokenizer::pretokenize(const std::string& t, std::vector<std::pair<size_t, size_t>>& out) const {
  const char* s = t.data();
  const size_t n = t.size();
  size_t i = 0;
  auto cp_at = [&](size_t k, uint32_t& cp) { return k < n ? dec_cp(s, n, k, cp) : (cp = 0, 0); };
  while (i < n) {
    uint32_t c0, c1;
    int l0 = cp_at(i, c0);
    int l1 = cp_at(i + l0, c1);
    size_t j = i + l0;
    // 1) contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c0 == '\'') {
      static const char* ct[] = {"s", "t", "re", "ve", "m", "ll", "d"};
      size_t best = 0;
      for (const char* x : ct) {
        size_t xl = strlen(x);
        if (j + xl <= n) {
          bool ok = true;
          for (size_t k = 0; k < xl; k++) if (tolower((unsigned char)s[j + k]) != x[k]) { ok = false; break; }
          if (ok && xl > best) best = xl;
        }
      }
      if (best) { out.push_back({i, j + best}); i = j + best; continue; }
    }
    // 2) [^\r\n\p{L}\p{N}]?\p{L}+
    if (!is_L(c0) && !is_N(c0) && !is_nl(c0) && l1 && is_L(c1)) {
      size_t k = i + l0 + l1;
      while (k < n) { uint32_t c; int l = cp_at(k, c); if (!is_L(c)) break; k += l; }
      out.push_back({i, k}); i = k; continue;
    }
    if (is_L(c0)) {
      size_t k = i + l0;
      while (k < n) { uint32_t c; int l = cp_at(k, c); if (!is_L(c)) break; k += l; }
      out.push_back({i, k}); i = k; continue;
    }
    // 3) \p{N}{1,3}
    if (is_N(c0)) {
      size_t k = i; int cnt = 0;
      while (k < n && cnt < 3) { uint32_t c; int l = cp_at(k, c); if (!is_N(c)) break; k += l; cnt++; }
      out.push_back({i, k}); i = k; continue;
    }
    // 4) ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    {
      size_t k = i;
      if (c0 == ' ') k = i + 1;
      size_t run = k;
      while (run < n) {
        uint32_t c; int l = cp_at(run, c);
        if (is_ws(c) || is_L(c) || is_N(c)) break;
        run += l;
      }
      if (run > k) {
        while (run < n) { uint32_t c; int l = cp_at(run, c); if (!is_nl(c)) break; run += l; }
        out.push_back({i, run}); i = run; continue;
      }
    }
    // 5) \s*[\r\n]+  6) \s+(?!\S)  7) \s+
    if (is_ws(c0)) {
      size_t k = i, last_nl = i;
      while (k < n) {
        uint32_t c; int l = cp_at(k, c);
        if (!is_ws(c)) break;
        if (is_nl(c)) last_nl = k + l;
        k += l;
      }
      if (last_nl > i) { out.push_back({i, last_nl}); i = last_nl; continue; }  // \s*[\r\n]+
      if (k == n) { out.push_back({i, k}); i = k; continue; }                   // \s+(?!\S) at EOF
      if (k - i >= 2) { out.push_back({i, k - 1}); i = k - 1; continue; }       // \s+(?!\S) backtrack
      out.push_back({i, k}); i = k; continue;                                   // \s+
    }
    out.push_back({i, j});
    i = j;
  }
}

std::vector<int> Tokenizer::bpe_piece(const std::string& piece) const {
  thread_local std::unordered_map<std::string, std::vector<int>> cache;
  auto cit = cache.find(piece);
  if (cit != cache.end()) return cit->second;
  std::vector<std::string> syms;
  syms.reserve(piece.size());
  for (unsigned char c : piece) syms.push_back(b2s()[c]);
  while (syms.size() > 1) {
    int best = -1, best_rank = INT32_MAX;
    for (size_t i = 0; i + 1 < syms.size(); i++) {
      auto it = merge_rank_.find(syms[i] + '\x01' + syms[i + 1]);
      if (it != merge_rank_.end() && it->second < best_rank) { best_rank = it->second; best = (int)i; }
    }
    if (best < 0) break;
    syms[best] += syms[best + 1];
    syms.erase(syms.begin() + best + 1);
  }
  std::vector<int> ids;
  ids.reserve(syms.size());
  for (const auto& sy : syms) {
    auto it = tok_to_id_.find(sy);
    if (it != tok_to_id_.end()) { ids.push_back(it->second); continue; }
    // byte fallback: decompose the symbol into single-byte symbols
    for (size_t k = 0; k < sy.size();) {
      uint32_t cp; int l = dec_cp(sy.data(), sy.size(), k, cp);
      auto bit = s2b().find(sy.substr(k, l));
      if (bit != s2b().end()) {
        auto bt = tok_to_id_.find(b2s()[bit->second]);
        if (bt != tok_to_id_.end()) ids.push_back(bt->second);
      }
      k += l;
    }
  }
  if (cache.size() < 200000) cache[piece] = ids;
  return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
  std::vector<int> out;
  size_t i = 0, seg_start = 0;
  auto flush = [&](size_t end) {
    if (end <= seg_start) return;
    std::string seg = text.substr(seg_start, end - seg_start);
    std::vector<std::pair<size_t, size_t>> pieces;
    pretokenize(seg, pieces);
    for (const auto& [a, b] : pieces) {
      auto ids = bpe_piece(seg.substr(a, b - a));
      out.insert(out.end(), ids.begin(), ids.end());
    }
  };
  while (i < text.size()) {
    int matched = -1;
    size_t mlen = 0;
    for (const auto& [c, id] : added_) {          // sorted by length desc
      if (c.size() <= mlen) break;
      if (i + c.size() <= text.size() && text.compare(i, c.size(), c) == 0) { matched = id; mlen = c.size(); }
    }
    if (matched >= 0) {
      flush(i);
      out.push_back(matched);
      i += mlen;
      seg_start = i;
      continue;
    }
    i++;
  }
  flush(text.size());
  return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const { return decode(ids, false); }

std::string Tokenizer::decode(const std::vector<int>& ids, bool keep_special) const {
  std::string out;
  for (int id : ids) {
    if (id < 0 || id >= (int)id_to_tok_.size()) continue;
    const std::string& tk = id_to_tok_[id];
    if (added_map_.count(tk)) {
      if (keep_special) out += tk;
      continue;
    }
    for (size_t k = 0; k < tk.size();) {
      uint32_t cp; int l = dec_cp(tk.data(), tk.size(), k, cp);
      auto it = s2b().find(tk.substr(k, l));
      if (it == s2b().end()) { out.append(tk, k, l); k += l; continue; }
      out += (char)it->second;
      k += l;
    }
  }
  return out;
}

std::string Tokenizer::apply_chat_template(const std::vector<ChatMsg>& msgs, bool gen,
                                           const std::string& effort) const {
  std::string out = "[gMASK]<sop>";
  std::string eff = (effort == "low" || effort == "high" || effort == "max") ? effort : "max";
  std::string cap = eff;
  cap[0] = (char)toupper(cap[0]);
  out += "<|system|>Reasoning Effort: " + cap;
  int last_user = -1;
  for (size_t k = 0; k < msgs.size(); k++) if (msgs[k].role == "user") last_user = (int)k;
  for (size_t k = 0; k < msgs.size(); k++) {
    const ChatMsg& m = msgs[k];
    if (m.role == "system") out += "<|system|>" + m.content;
    else if (m.role == "user") out += "<|user|>" + m.content;
    else if (m.role == "assistant") {
      out += "<|assistant|>";
      out += "<think></think>";
      std::string c = m.content;
      size_t a = c.find_first_not_of(" \t\r\n");
      size_t b = c.find_last_not_of(" \t\r\n");
      if (a != std::string::npos) out += c.substr(a, b - a + 1);
      (void)last_user;
    } else if (m.role == "tool") {
      out += "<|observation|>" + m.content;
    }
  }
  if (gen) out += "<|assistant|><think>";
  return out;
}

}  // namespace helios