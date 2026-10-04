// CPU regression tests for mixed EXL3 layouts; optionally validate a real checkpoint.
#include "core/expert_layout.hpp"
#include "json.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

using namespace helios;
using nlohmann::json;

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

static void fixture(const std::filesystem::path& dir, std::array<int, 3> bits,
                    const std::string& defect = "") {
  json header = json::object();
  std::vector<char> data;
  const char* projections[] = {"gate_proj", "up_proj", "down_proj"};
  const char* suffix[] = {"trellis", "suh", "svh", "mul1"};
  for (int e = 0; e < 2; ++e) for (int p = 0; p < 3; ++p) {
    int b = bits[p] + (defect == "mixed" && e == 1 && p == 0 ? 1 : 0);
    size_t sizes[] = {(size_t)128 * 128 * b / 8, 256, 256, 4};
    const char* dtype[] = {"I16", "F16", "F16", "I32"};
    json shapes[] = {json::array({8, 8, 16 * b}), json::array({128}),
                     json::array({128}), json::array()};
    for (int q = 0; q < 4; ++q) {
      std::string name = "model.language_model.layers.3.mlp.experts." + std::to_string(e) +
                         "." + projections[p] + "." + suffix[q];
      size_t begin = data.size();
      data.resize(begin + sizes[q], char(0x5a));
      if (q == 3) {
        uint32_t word = defect == "codebook" ? 1u : 0x83DCD12Du;
        std::memcpy(data.data() + begin, &word, 4);
      }
      header[name] = {{"dtype", dtype[q]}, {"shape", shapes[q]},
                      {"data_offsets", json::array({begin, data.size()})}};
      if (e == 0 && p == 0 && q == 0 && defect == "size")
        header[name]["data_offsets"][1] = data.size() - 2;
      if (e == 0 && p == 0 && q == 1 && defect == "dtype") header[name]["dtype"] = "BF16";
      if (e == 0 && p == 0 && q == 1 && defect == "shape") header[name]["shape"] = {64, 2};
      if (e == 0 && p == 0 && q == 2 && defect == "missing") header.erase(name);
    }
  }
  std::string encoded = header.dump();
  uint64_t length = encoded.size();
  std::ofstream out(dir / "model.safetensors", std::ios::binary);
  out.write((const char*)&length, 8);
  out.write(encoded.data(), encoded.size());
  out.write(data.data(), data.size());
}

static void regression() {
  auto dir = std::filesystem::temp_directory_path() /
             ("helios-layout-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(dir);
  try {
    for (int bits : {2, 3, 4}) {
      fixture(dir, {bits, bits, bits});
      ShardSet shards; shards.load_dir(dir);
      auto layout = expert_layout(shards, 3, 2, 128, 128);
      require(layout.bits == std::array<int, 3>{bits, bits, bits}, "wrong bit widths");
      require(layout.stride % 64 == 0, "unaligned slab");
      for (int p = 0; p < 3; ++p) {
        require(layout.piece[p * 4] == (size_t)128 * 128 * bits / 8, "wrong trellis capacity");
        require(layout.off[p * 4] % 64 == 0, "unaligned projection");
        require(layout.off[p * 4 + 3] + 4 <= layout.stride, "slab overflow");
      }
    }
    fixture(dir, {2, 3, 4});
    ShardSet mixed; mixed.load_dir(dir);
    auto layout = expert_layout(mixed, 3, 2, 128, 128);
    require(layout.bits == std::array<int, 3>{2, 3, 4}, "per-projection bits lost");
    for (const char* defect : {"mixed", "size", "dtype", "shape", "missing", "codebook"}) {
      fixture(dir, {2, 2, 2}, defect);
      ShardSet shards; shards.load_dir(dir);
      bool rejected = false;
      try { expert_layout(shards, 3, 2, 128, 128); }
      catch (const std::runtime_error&) { rejected = true; }
      require(rejected, defect);
    }
    std::filesystem::remove_all(dir);
  } catch (...) {
    std::filesystem::remove_all(dir);
    throw;
  }
  printf("EXPERT LAYOUT: uniform 2/3/4-bit, mixed projections, six invalid checkpoints PASS\n");
}

static void checkpoint(const std::string& dir) {
  json config; std::ifstream(dir + "/config.json") >> config;
  const auto& text = config.contains("text_config") ? config["text_config"] : config;
  ShardSet shards; shards.load_dir(dir);
  const int layers = text.at("num_hidden_layers"), experts = text.at("n_routed_experts");
  const int hidden = text.at("hidden_size"), inter = text.at("moe_intermediate_size");
  const bool mtp = text.value("num_nextn_predict_layers", 0) > 0;
  size_t arena = 0, slot = 0;
  int count = 0;
  for (int l = 0; l <= layers; ++l) {
    if (l == layers ? !mtp : text.at("mlp_layer_types")[l] != "sparse") continue;
    auto layout = expert_layout(shards, l, experts, hidden, inter);
    arena += experts * layout.stride;
    if (l < layers) slot = std::max(slot, layout.stride);
    ++count;
    printf("L%d bits=%d/%d/%d stride=%zu\n", l, layout.bits[0], layout.bits[1], layout.bits[2], layout.stride);
  }
  const auto* head = shards.find("lm_head.trellis");
  require(head && head->shape.size() == 3, "missing quantized head");
  printf("CHECKPOINT PASS: %s shards=%d expert_layers=%d arena=%.3fGiB slot=%zu head_bits=%lld\n",
         dir.c_str(), shards.shard_count(), count, arena / 1073741824.0, slot,
         (long long)head->shape[2] / 16);
}

int main(int argc, char** argv) {
  try {
    if (argc == 1) regression();
    else for (int i = 1; i < argc; ++i) checkpoint(argv[i]);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "EXPERT LAYOUT FAIL: %s\n", e.what()); return 1;
  }
}
