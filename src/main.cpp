// helios — GLM-5.3-Flash EXL3 engine
// CLI: `helios inspect <model_dir>` dumps tensor inventory + bit analysis.
#include "core/model.hpp"
#include "core/device.hpp"
#include "engine/cache.hpp"
#include "engine/slotmgr.hpp"
#include "engine/runner.hpp"
#include "tokenizer/tokenizer.hpp"
#include <cstring>
#include "core/safetensors.hpp"
#include <cstdio>
#include <regex>
#include <map>

using namespace helios;

static const char* dtype_name(Dtype d) {
  switch (d) { case Dtype::F16: return "f16"; case Dtype::BF16: return "bf16"; case Dtype::F32: return "f32";
    case Dtype::I32: return "i32"; case Dtype::I16: return "i16"; default: return "?"; }
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: helios inspect <model_dir>\n"); return 1; }

  std::string cmd = argv[1], dir = argv[2];
  if (cmd == "bench") {
    // Dense EXL3 gemm throughput at the shapes the fused MoE sees, so the fused kernel's rate can be
    // compared against what the same weights achieve through the plain gemm path.
    int max_m = 512;
    for (int i = 3; i < argc; i++) {
      std::string a = argv[i];
      if (a == "--max-m" && i + 1 < argc) max_m = atoi(argv[++i]);
    }
    if (!Engine::instance().init()) return 2;
    Model m;
    if (!m.load(dir, false, false)) return 3;
    Cache cache;
    int cap = 4096;
    if (!cache.init(m, cap, max_m, mtp_enabled())) return 5;
    SlotMgr slots;
    Runner runner;
    if (!runner.init(m, cache, slots, nullptr, max_m)) return 7;
    if (getenv("HELIOS_BENCH_ARENA")) return runner.bench_arena_copy(atoi(getenv("HELIOS_BENCH_ARENA"))) ? 0 : 9;
    return runner.bench_gemm(max_m) ? 0 : 9;
  }
  if (cmd == "dbg") {
    std::string dump, sub, ids_arg;
    int upto = 1, cap = 1024;
    for (int i = 3; i < argc; i++) {
      std::string a = argv[i];
      if (a == "--upto" && i + 1 < argc) upto = atoi(argv[++i]);
      else if (a == "--cap" && i + 1 < argc) cap = atoi(argv[++i]);
      else if (a == "--ids" && i + 1 < argc) ids_arg = argv[++i];
      else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
      else if (a == "--sub" && i + 1 < argc) sub = argv[++i];
    }
    if (!Engine::instance().init()) return 2;
    printf("[dbg] gpu0.phys=%d gpu1.phys=%d\n", Engine::instance().gpu(0).phys_idx(),
           Engine::instance().gpu(1).phys_idx());
    Model m;
    if (!m.load(dir, false, false)) return 3;
    Tokenizer tk;
    tk.load(dir);
    Cache cache;
    if (!cache.init(m, cap, 512, mtp_enabled())) return 5;
    SlotMgr slots;
    std::string census = dir + "/.helios.census";
    if (m.slot_pool && m.n_slots > 0) { if (!slots.init_from_pool(m, m.slot_pool, m.n_slots, census.c_str())) return 6; }
    Runner runner;
    if (!runner.init(m, cache, slots, &tk, 32)) return 7;
    std::vector<int> ids;
    if (!ids_arg.empty()) { std::string cur; for (char c : ids_arg) { if (c == ',') { ids.push_back(atoi(cur.c_str())); cur.clear(); } else cur += c; } if (!cur.empty()) ids.push_back(atoi(cur.c_str())); }
    if (ids.empty()) ids = {154822, 154824, 154826, 154826, 9703};   // fallback: gMASK sop system...
    return runner.dump_layers(ids, upto, dump.empty() ? "/tmp/helios_streams.bin" : dump, sub) ? 0 : 8;
  }
  if (cmd == "load" || cmd == "serve" || cmd == "gen") {
    int cap = 262144, port = 8080, max_chunk = 256;
    // Default output length for requests that omit `max_tokens`. The engine imposes no output cap of
    // its own beyond the KV capacity, so this only bounds what an omission produces.
    int default_max_tokens = getenv("HELIOS_MAX_TOKENS") ? atoi(getenv("HELIOS_MAX_TOKENS")) : 0;
    // Default reasoning effort (low|high|max) for requests that do not send one.
    std::string reasoning_effort = getenv("HELIOS_REASONING_EFFORT") ? getenv("HELIOS_REASONING_EFFORT") : "";
    const char* host = "127.0.0.1";
    std::string api_key;
    bool ram_only = false;
    std::vector<int> gen_ids;
    std::string gen_text;
    int max_tokens = 64;
    float temperature = 0.7f;
    for (int i = 3; i < argc; i++) {
      std::string a = argv[i];
      if (a == "--ram-only") ram_only = true;
      else if (a == "--cap" && i + 1 < argc) cap = atoi(argv[++i]);
      else if (a == "--port" && i + 1 < argc) port = atoi(argv[++i]);
      else if (a == "--host" && i + 1 < argc) host = argv[++i];
      else if (a == "--api-key" && i + 1 < argc) api_key = argv[++i];
      else if (a == "--max-tokens" && i + 1 < argc) default_max_tokens = atoi(argv[++i]);
      else if (a == "--reasoning-effort" && i + 1 < argc) reasoning_effort = argv[++i];
      else if (a == "--chunk" && i + 1 < argc) max_chunk = atoi(argv[++i]);
      else if (a == "--tokens" && i + 1 < argc) max_tokens = atoi(argv[++i]);
      else if (a == "--temp" && i + 1 < argc) temperature = atof(argv[++i]);
      else if (a == "--prompt" && i + 1 < argc) gen_text = argv[++i];
      else if (a == "--prompt-file" && i + 1 < argc) {
        std::string path = argv[++i];
        FILE* pf = fopen(path.c_str(), "rb");
        if (!pf) { fprintf(stderr, "cannot open prompt file %s\n", path.c_str()); return 2; }
        std::string raw; char buf[1 << 16]; size_t got;
        while ((got = fread(buf, 1, sizeof(buf), pf)) > 0) raw.append(buf, got);
        fclose(pf);
        size_t t = raw.find_first_not_of(" \t\r\n");
        if (t != std::string::npos) raw = raw.substr(t);
        gen_text = raw;
      }
    }
    if (!Engine::instance().init()) { fprintf(stderr, "engine init failed\n"); return 2; }
    Model m;
    if (!m.load(dir, ram_only, true)) { fprintf(stderr, "model load failed\n"); return 3; }
    if (ram_only) {
      printf("LOAD OK: layers=%zu moe=%d arena=%.1fGB gpu0=%.2fGB gpu1=%.2fGB slots=%d stride=%zu\n",
             m.layers.size(), (int)m.cfg.moe[3], (double)m.ram_bytes/GiB, (double)m.gpu0_bytes/GiB,
             (double)m.gpu1_bytes/GiB, m.n_slots, m.slot_stride);
      return 0;
    }
    bool do_gen = (cmd == "gen");
    if (!do_gen && cmd == "load") {
      printf("LOAD OK: layers=%zu moe=%d arena=%.1fGB gpu0=%.2fGB gpu1=%.2fGB slots=%d stride=%zu\n",
             m.layers.size(), (int)m.cfg.moe[3], (double)m.ram_bytes/GiB, (double)m.gpu0_bytes/GiB,
             (double)m.gpu1_bytes/GiB, m.n_slots, m.slot_stride);
      return 0;
    }
    Tokenizer tk;
    if (!tk.load(dir)) { fprintf(stderr, "tokenizer load failed\n"); return 4; }
    Cache cache;
    if (!cache.init(m, cap, max_chunk, mtp_enabled())) return 5;
    SlotMgr slots;
    std::string census = dir + "/.helios.census";
    if (m.slot_pool && m.n_slots > 0) { if (!slots.init_from_pool(m, m.slot_pool, m.n_slots, census.c_str())) return 6; }
    else {
      Device& g1 = Engine::instance().gpu(1);
      size_t pool_bytes = g1.stats().free_vram > (4ull << 30) ? g1.stats().free_vram - (4ull << 30) : 0;
      const char* slots_env = getenv("HELIOS_SLOTS");
      if (slots_env) pool_bytes = (size_t)atoll(slots_env) * m.slot_stride;
      if (!slots.init(m, pool_bytes)) return 6;
    }
    Runner runner;
    if (!runner.init(m, cache, slots, &tk, max_chunk)) return 7;
    {
      size_t f0 = 0, t0 = 0, f1 = 0, t1 = 0;
      cudaSetDevice(Engine::instance().gpu(0).phys_idx()); cudaMemGetInfo(&f0, &t0);
      cudaSetDevice(Engine::instance().gpu(1).phys_idx()); cudaMemGetInfo(&f1, &t1);
      cudaSetDevice(Engine::instance().gpu(0).phys_idx());
      printf("[mem] gpu0 used %.2f GB free %.2f | gpu1 used %.2f GB free %.2f\n",
             (t0 - f0) / 1073741824.0, f0 / 1073741824.0,
             (t1 - f1) / 1073741824.0, f1 / 1073741824.0);
    }
    if (do_gen) {
      std::vector<ChatMsg> msgs = {{"user", gen_text.empty() ? "Hello!" : gen_text}};
      GenParams p;
      p.max_tokens = max_tokens; p.temperature = temperature;
      std::string prompt = tk.apply_chat_template(msgs, true);
      auto ids = tk.encode(prompt);
      printf("[gen] prompt tokens=%zu\n", ids.size());
      double t0 = (double)clock() / CLOCKS_PER_SEC;
      auto out = runner.generate(ids, p, [&](int tok) {
        printf("%s", tk.decode({tok}).c_str());
        fflush(stdout);
        return true;
      });
      double dt = (double)clock() / CLOCKS_PER_SEC - t0;
      printf("\n[gen] %zu tokens in %.2fs (%.2f tok/s)  prefill %.1f tok/s decode %.2f tok/s\n",
             out.size(), dt, dt > 0 ? out.size() / dt : 0.0,
             runner.timings().prefill_ms > 0 ? runner.timings().prefill_tokens * 1000.0 / runner.timings().prefill_ms : 0.0,
             runner.timings().decode_ms > 0 ? runner.timings().decode_tokens * 1000.0 / runner.timings().decode_ms : 0.0);
      slots.print_stats();
      slots.print_skew();
      return 0;
    }
    slots.print_stats();
    return run_server(runner, tk, host, port, 8, api_key, default_max_tokens, reasoning_effort);
  }
  if (cmd == "load") {
    bool ram_only = argc > 3 && !strcmp(argv[3], "--ram-only");
    if (!ram_only && !Engine::instance().init()) { fprintf(stderr, "engine init failed\n"); return 2; }
    Model m;
    if (!m.load(dir, ram_only, true)) { fprintf(stderr, "model load failed\n"); return 3; }
    printf("LOAD OK: layers=%zu moe=%d arena=%.1fGB gpu0=%.2fGB gpu1=%.2fGB slots=%d stride=%zu\n",
           m.layers.size(), (int)m.cfg.moe[3], (double)m.ram_bytes/GiB, (double)m.gpu0_bytes/GiB,
           (double)m.gpu1_bytes/GiB, m.n_slots, m.slot_stride);
    return 0;
  }
  bool dump_mode = cmd == "dump";
  if (cmd != "inspect" && !dump_mode) { fprintf(stderr, "usage: helios inspect|dump <model_dir> [substr]\n"); return 1; }
  ShardSet ss;
  const char* filt = argc > 3 ? argv[3] : nullptr;

  ss.load_dir(dir);
  if (!dump_mode) printf("shards=%d tensors=%zu total=%.2f GB\n", ss.shard_count(), ss.tensors().size(), ss.total_bytes()/1e9);
  std::map<std::string, uint64_t> cat_bytes;
  std::map<std::string, std::pair<uint64_t,uint64_t>> trellis_bits; // category -> (bits_total, params)
  std::map<int, int> layer_shard;
  int n_trellis = 0;
  for (auto* t : ss.tensors()) {
    if (dump_mode) {
      if (!filt || t->name.find(filt) != std::string::npos) {
        printf("%-72s %-4s [", t->name.c_str(), dtype_name(t->dtype));
        for (auto d : t->shape) printf("%lld,", (long long)d);
        printf("] %zu B\n", t->bytes);
      }
      continue;
    }
    std::string cat = "other";
    std::smatch m;
    std::regex layer_re("model\\.language_model\\.layers\\.(\\d+)\\.(.+)");
    if (std::regex_match(t->name, m, layer_re)) {
      int layer = std::stoi(m[1]);
      std::string rest = m[2];
      layer_shard[layer] = t->shard;
      if (rest.find("experts.") != std::string::npos && rest.find("shared_experts") == std::string::npos) {
        cat = "routed_experts";
      } else if (rest.find("shared_experts.") == 0) cat = "shared_experts";
      else if (rest.find("self_attn.") == 0) cat = "attention";
      else if (rest.find("mlp.gate.") == 0) cat = "router";
      else if (rest.find("mlp.") == 0) cat = "dense_mlp";
      else if (rest.find("hc_") == 0 || rest.find("input_layernorm") == 0 || rest.find("post_attention") == 0) cat = "mhc_norms";
      else cat = "layer_other";
    } else if (t->name.find("embed_tokens") != std::string::npos) cat = "embed";
    else if (t->name.find("lm_head") != std::string::npos) cat = "lm_head";
    else if (t->name.find("vision") != std::string::npos) cat = "vision";
    else if (t->name.find("mtp") != std::string::npos || t->name.find("nextn") != std::string::npos) cat = "mtp";
    cat_bytes[cat] += t->bytes;

    if (t->name.size() > 8 && t->name.substr(t->name.size()-8) == ".trellis") {
      n_trellis++;
      // infer logical matrix dims from sibling svh/suh lengths
      std::string base = t->name.substr(0, t->name.size()-8);
      uint64_t out = 0, in = 0;
      if (auto* svh = ss.find(base + ".svh")) out = svh->elems;
      if (auto* suh = ss.find(base + ".suh")) in = suh->elems;
      uint64_t syms = t->elems;
      if (out && in) {
        std::string cat2 = "routed_experts";
        if (base.find("shared_experts") != std::string::npos) cat2 = "shared_experts";
        else if (base.find("self_attn") != std::string::npos) cat2 = "attention";
        else if (base.find("down_proj") != std::string::npos && base.find("experts") == std::string::npos) cat2 = "dense_mlp";
        auto& acc = trellis_bits[cat2];
        acc.first += syms * 16; acc.second += out * in;
      }
    }
  }
  printf("categories (GB):\n");
  for (auto& [k, v] : cat_bytes) printf("  %-16s %7.2f GB\n", k.c_str(), v/1e9);
  printf("trellis tensors=%d\nbits-per-param by category:\n", n_trellis);
  for (auto& [k, v] : trellis_bits) printf("  %-16s %.3f bpw (%zu params)\n", k.c_str(), (double)v.first/v.second, v.second);
  printf("layer->shard map:\n");
  int prev = -1;
  for (auto& [l, s] : layer_shard) if (s != prev) { printf("  L%-2d -> S%d\n", l, s); prev = s; }
  return 0;
}