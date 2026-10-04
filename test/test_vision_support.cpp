#include "core/vision.hpp"
#include "core/safetensors.hpp"
#include "json.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
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
