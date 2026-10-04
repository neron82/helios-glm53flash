#include "core/vision.hpp"
#include "core/safetensors.hpp"
#include "json.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

using namespace helios;
using nlohmann::json;

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

static void fixture(const std::filesystem::path& directory, bool merger,
                    const std::string& quant_suffix = "", const std::string& dtype = "BF16") {
  json header;
  auto add = [&](const std::string& name, const std::string& type) {
    header[name] = {{"dtype", type}, {"shape", json::array({2})},
                    {"data_offsets", json::array({0, 4})}};
  };
  add("model.visual.patch_embed.proj.weight", dtype);
  if (merger) add("model.visual.merger.down_proj.weight", "BF16");
  if (!quant_suffix.empty()) add("model.visual.blocks.0.attn.proj." + quant_suffix, "I16");
  const auto encoded = header.dump();
  uint64_t size = encoded.size();
  std::ofstream output(directory / "model.safetensors", std::ios::binary);
  output.write(reinterpret_cast<const char*>(&size), sizeof(size));
  output.write(encoded.data(), encoded.size());
  output.write("0000", 4);
}

int main(int argc, char** argv) {
  if (argc == 3) {
    ShardSet shards; shards.load_dir(argv[1]);
    bool supported = supports_vision_input(shards);
    require(supported == (std::string(argv[2]) == "true"), "checkpoint capability mismatch");
    printf("%s supports_vision=%s PASS\n", argv[1], supported ? "true" : "false");
    return 0;
  }
  ShardSet empty;
  ImageEmbedding red{100, 2, 2, {1, 2, 3, 4}};
  ImageEmbedding blue{100, 2, 2, {4, 3, 2, 1}};
  require(image_prefix_limit({}, {}) == std::numeric_limits<int>::max(), "text prefix restricted");
  require(image_prefix_limit({red}, {red}) == std::numeric_limits<int>::max(), "identical image invalidated");
  require(image_prefix_limit({red}, {blue}) == 100, "changed pixels reused");
  require(image_prefix_limit({red}, {}) == 100, "removed image reused");
  require(image_prefix_limit({}, {red}) == 100, "new image reused");
  auto later = blue; later.start = 200;
  require(image_prefix_limit({red}, {red, later}) == 200, "later image discarded earlier prefix");
  require(image_prefix_limit({red}, {later}) == 100, "moved image reused");
  ImageEmbeddingCache cache(20);
  cache.put("red", red); cache.put("blue", blue);
  ImageEmbedding found;
  require(!cache.get("red", found), "oldest entry not evicted");
  require(cache.get("blue", found) && found.values == blue.values, "wrong cached embeddings");
  cache.put("blue", red);
  require(cache.get("blue", found) && found.values == red.values, "replacement not cached");
  auto huge = red; huge.values.resize(100);
  cache.put("huge", huge);
  require(!cache.get("huge", found) && cache.bytes() <= 20, "image cache budget exceeded");
  ImageEmbeddingCache lru(32);
  lru.put("one", red); lru.put("two", blue);
  require(lru.get("one", found), "first entry missing");
  lru.put("third", red);
  require(!lru.get("two", found) && lru.get("one", found), "LRU order lost on cache hit");
  require(!supports_vision_input(empty), "text-only model advertised vision");
  auto directory = std::filesystem::temp_directory_path() /
      ("helios-vision-support-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(directory);
  try {
    auto supported = [&]() {
      ShardSet shards; shards.load_dir(directory); return supports_vision_input(shards);
    };
    fixture(directory, true);
    require(supported(), "dense BF16 tower rejected");
    fixture(directory, false);
    require(!supported(), "missing merger accepted");
    // A dense patch embedding can coexist with EXL3-quantized linears.
    for (const char* suffix : {"trellis", "suh", "svh", "mul1"}) {
      fixture(directory, true, suffix);
      require(!supported(), "EXL3 tower advertised vision");
    }
    fixture(directory, true, "", "I16");
    require(!supported(), "non-floating vision weights accepted");
  } catch (...) {
    std::filesystem::remove_all(directory); throw;
  }
  std::filesystem::remove_all(directory);
  printf("VISION SUPPORT: dense, missing, text-only and EXL3 towers PASS\n");
}
